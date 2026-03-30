import argparse
import json
import sys
import math
import torch
import wandb
import importlib.util
import torch.distributed as dist
import torch.distributed as dist
from pathlib import Path

from spacetorch.utils.torch_utils import resolve_sequential_module_from_str
from spacetorch.utils.generic_utils import convert_to_serializable
from spacetorch.variants.base_arch import BaseArch
from spacetorch.variants.positions import NetworkPositions
from spacetorch.losses.losses_torch import spatial_correlation_loss
from spacetorch.utils.gpu_utils import is_master_process
from spacetorch.variants.assets.mae.main_pretrain import main
from spacetorch.variants.assets.mae.pos_embed import interpolate_pos_embed
from spacetorch.variants.assets.mae.main_linprobe import main as eval_main


class MAE(BaseArch):
    """
    Implements the Masked Auto-Encoder self-supervised objective function.
    """
    def __init__(self):
        super().__init__()
    
    def get_model_layernames(self, args):
        return list(args.spatial_loss.spatial_weight.keys())

    def set_cfg(self, args):
        self.cfg = argparse.Namespace(**{
            "spatial_args": args,
            "output_dir": Path(args.output_dir) / "eval",
            "log_dir": None,
            "device": "cuda",
            "seed": 0,
            "resume": "",
            "start_epoch": 0,
            "num_workers": 10,
            "pin_mem": True,
            "world_size": 1,
            "local_rank": -1,
            "dist_on_itp": False,
            "dist_url": "env://",
            "data_path": args.variant.params.dataset_path,
            "batch_size": 64,
            "epochs": 400,
            "accum_iter": 1,
            "model": "mae_vit_base_patch16",
            "input_size": 224,
            "mask_ratio": 0.75,
            "norm_pix_loss": True,
            "weight_decay": 0.05,
            "lr": None,
            "blr": 1e-3,
            "min_lr": 0.,
            "warmup_epochs": 40,
        })

    def set_eval_cfg(self, args):
        self.eval_cfg = argparse.Namespace(**{
            "batch_size": 512,
            "epochs": 90,
            "accum_iter": 1,
            "model": "vit_base_patch16",
            # "input_size": 224,
            # "mask_ratio": 0.75,
            # "norm_pix_loss": True,
            "weight_decay": 0,
            "lr": None,
            "blr": 0.1,
            "min_lr": 0.,
            "warmup_epochs": 10,
            "finetune": args.variant.params.pretrained_weights,
            "global_pool": False,
            "data_path": args.variant.params.dataset_path,
            "nb_classes": 1000,
            "output_dir": Path(args.output_dir) / "eval" / "linear",
            "log_dir": None,
            "device": "cuda",
            "seed": 0,
            "resume": "",
            "start_epoch": 0,
            "eval": False,
            "dist_eval": True,
            "num_workers": 10,
            "pin_mem": False,
            "world_size": 1,
            "local_rank": -1,
            "dist_on_itp": False,
            "dist_url": "env://",
        })

    def _load_pretrained_weights(self, args, model):
        checkpoint = torch.load(args.variant.params.pretrained_weights, map_location="cpu", weights_only=False)
        checkpoint_model = checkpoint['model']
        for k in list(checkpoint_model.keys()):
            if k.startswith('model.'):
                # remove prefix
                checkpoint_model[k[len("model."):]] = checkpoint_model[k]
                del checkpoint_model[k]
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(("Pretrained weights found at {} and loaded with msg: {}".format(args.variant.params.pretrained_weights, msg)))

    def _load_positions(self, position_dir):
        network_positions = NetworkPositions.load_from_dir(position_dir)
        network_positions.to_torch()
        return network_positions.layer_positions

    def set_model(self, args):
        sys.path.insert(0, str(Path(args.variant.setup.model).parent))
        models_mae = load_function_from_file(args.variant.setup.model, "models_mae")
        model = models_mae.__dict__[self.cfg.model](norm_pix_loss=self.cfg.norm_pix_loss)
        self.layernames = self.get_model_layernames(args)
        self.positions = self._load_positions(args.spatial_loss.position_dir)
        self.model = SpatialMAE(model, args, self.positions, self.layernames)

    def set_eval_model(self, args):
        sys.path.insert(0, str(Path(args.variant.setup.model).parent))
        models_vit = load_function_from_file(args.variant.setup.eval_model, "models_vit")
        self.eval_model = models_vit.vit_base_patch16(
            num_classes=1000,
            global_pool=False,
        )
        # models_mae = load_function_from_file(args.variant.setup.eval_model, "models_mae")
        # self.eval_model = models_mae.__dict__[self.eval_cfg.model](norm_pix_loss=self.eval_cfg.norm_pix_loss)
        if args.variant.params.pretrained_weights:
            self._load_pretrained_weights(args, self.eval_model)
        self.eval_model.cuda()

    def start_training_protocol(self, args):
        if self.cfg.output_dir:
            Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)
        engine_pretrain = load_function_from_file(args.variant.setup.train, "engine_pretrain")
        train_one_epoch = engine_pretrain.train_one_epoch
        main(self.cfg, self.model, train_one_epoch)

    def set_eval_protocol(self, args):
        engine_finetune = load_function_from_file(args.variant.setup.eval, "engine_finetune")
        self.train_one_epoch = engine_finetune.train_one_epoch
        self.evaluate = engine_finetune.evaluate

    def start_eval_protocol(self):
        if self.eval_cfg.output_dir:
            Path(self.eval_cfg.output_dir).mkdir(parents=True, exist_ok=True)
        eval_main(self.eval_cfg, self.eval_model, self.train_one_epoch, self.evaluate)

    def set_kinetics_cfg(self, args):
        pass

    def set_kinetics_protocol(self, args):
        pass
    
    def start_kinetics_protocol(self):
        pass


def load_function_from_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class SpatialMAE(torch.nn.Module):
    def __init__(self, model, args=None, positions=None, layernames=None):
        super().__init__()
        self.model = model
        self.positions = positions
        self.args = args
        self.iteration_counter = 0

        self.intermediate_outputs = {}

        self.scaler = torch.cuda.amp.GradScaler()

        # Register hooks on the intermediate layers of the model
        for layername in layernames:
            layer = resolve_sequential_module_from_str(self.model, layername)
            layer.register_forward_hook(self.get_hook_fn(layername))
            print(f"Registered hook for {layername}")

        run_name = args.name
        if args.wandb and is_master_process():
            wandb.init(
                project=args.name.split("_")[1],
                name=run_name,
            )

        save_dir = Path(args.output_dir) / "eval"
        save_dir.mkdir(parents=True, exist_ok=True)

    def get_hook_fn(self, layername):
        """Creates a hook function to capture the outputs of a specific layer."""
        def hook_fn(module, input, output):
            spatial_features = output[:, 1:].float()
            N, L, H = spatial_features.shape
            sL = int(math.sqrt(L))
            spatial_features = spatial_features.reshape(N, sL, sL, H).permute(0, 3, 1, 2)
            self.intermediate_outputs[layername] = spatial_features
        return hook_fn

    def forward(self, imgs, mask_ratio=0.75):
        self.intermediate_outputs.clear()

        loss_accumulator, _, _ = self.model.forward(imgs, mask_ratio)
        self.model.forward(imgs, mask_ratio=0.0)

        spatial_outputs = {}
        for layername, spatial_features in self.intermediate_outputs.items(): 
            spatial_outputs[layername] = (spatial_features, self.positions[layername])

        joint_loss = self.get_joint_loss((loss_accumulator, spatial_outputs))

        return joint_loss, _, _

    def get_joint_loss(self, loss):
        if isinstance(loss, tuple):
            loss_accumulator, spatial_outputs = loss

            task_loss = loss_accumulator.clone()
            joint_loss = loss_accumulator

            spatial_losses = {}
            for layername, layer_output in spatial_outputs.items():
                features, pos = layer_output

                spatial_losses[layername] = spatial_correlation_loss(
                    features.cuda(),
                    pos.coordinates.cuda(),
                    pos.neighborhood_indices.cuda(),
                )

                if dist.get_world_size() > 1:
                    dist.all_reduce(spatial_losses[layername])
                spatial_losses[layername] = spatial_losses[layername] / dist.get_world_size()

                joint_loss += self.args.spatial_loss.spatial_weight[layername] * spatial_losses[layername]

            if self.iteration_counter % 10 == 0:
                if is_master_process():
                    serializable_spatial_losses = convert_to_serializable(spatial_losses)
                    print(json.dumps(serializable_spatial_losses))
                    save_spatial_losses_path = Path(self.args.output_dir) / "spatial_losses.json"
                    with open(save_spatial_losses_path, 'a') as f:
                        f.write(json.dumps(serializable_spatial_losses) + "\n")

                    if self.args.wandb:
                        wandb.define_metric("iter")
                        log_data = {
                            "task_loss": task_loss,
                            "joint_loss": joint_loss,
                            "spatial_loss": {
                                **spatial_losses
                            },
                            "iter": self.iteration_counter
                        }
                        wandb.log(log_data)

            self.iteration_counter += 1
        else:
            joint_loss = loss

        return joint_loss
