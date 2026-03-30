import os
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
from spacetorch.variants.assets.swin_moby.moby_main import main
from spacetorch.variants.assets.swin_moby.moby_linear import main as eval_main
from spacetorch.variants.assets.swin_moby.moby_kinetics import main as kinetics_main
from spacetorch.variants.assets.swin_moby.models.build import build_model
from spacetorch.variants.assets.swin_moby.config import get_config


class SwinMoBY(BaseArch):
    """
    Implements the MoBY (MoCo v2 + BYOL) objective function for Swin Transformers
    """
    def __init__(self):
        super().__init__()
    
    def get_model_layernames(self, args):
        return list(args.spatial_loss.spatial_weight.keys())

    def set_cfg(self, args):
        variant_args = argparse.Namespace(**{
            "cfg": args.variant.params.config,
            "batch_size": 128,
            "opts": [],
            "data_path": args.variant.params.dataset_path,
            "zip": False,
            "cache_mode": "no",
            "resume": "",
            "accumulation_steps": 0,
            "use_checkpoint": False,
            "amp_opt_level": "O0",
            "output": str(Path(args.output_dir) / "eval"),
            "seed": args.variant.params.seed,
            "tag": "",
            "eval": False,
            "throughput": False,
            "distributed": True,
            "local_rank": -1,
        })
        variant_cfg = get_config(variant_args)
        self.cfg = variant_cfg

    def set_eval_cfg(self, args):
        variant_args = argparse.Namespace(**{
            "cfg": args.variant.params.config,
            "batch_size": 128,
            "opts": [],
            "data_path": args.variant.params.dataset_path,
            "zip": False,
            "cache_mode": "no",
            "resume": "",
            "accumulation_steps": 0,
            "use_checkpoint": False,
            "amp_opt_level": "O0",
            "seed": 0,
            "output": str(Path(args.output_dir) / "eval"),
            "tag": "",
            "eval": False,
            "throughput": False,
            "distributed": False,
            "local_rank": 0,
            "lr": 1.0,
            "drop_path_rate": 0.2,
        })
        variant_cfg = get_config(variant_args)
        variant_cfg.defrost()
        variant_cfg.LINEAR_EVAL.PRETRAINED = args.variant.params.pretrained_weights
        variant_cfg.OUTPUT = os.path.join(variant_args.output, "linear")
        variant_cfg.MODEL.TYPE = 'linear'
        variant_cfg.MODEL.DROP_PATH_RATE = variant_args.drop_path_rate
        variant_cfg.AUG.SSL_AUG = False
        variant_cfg.AUG.SSL_LINEAR_AUG = True
        variant_cfg.AUG.MIXUP = 0.0
        variant_cfg.AUG.CUTMIX = 0.0
        variant_cfg.AUG.CUTMIX_MINMAX = None
        variant_cfg.TRAIN.EPOCHS = 30
        variant_cfg.TRAIN.WARMUP_EPOCHS = 5
        variant_cfg.TRAIN.LR_SCHEDULER.NAME = 'cosine'
        variant_cfg.TRAIN.OPTIMIZER.NAME = 'sgd'
        variant_cfg.TRAIN.OPTIMIZER.MOMENTUM = 0.9
        variant_cfg.TRAIN.BASE_LR = variant_args.lr
        variant_cfg.TRAIN.WEIGHT_DECAY = 0.0
        variant_cfg.freeze()
        self.eval_cfg = variant_cfg

    def _load_pretrained_weights(self, args, model):
        model_dict = model.state_dict()
        state_dict = torch.load(args.variant.params.pretrained_weights, map_location='cpu', weights_only=False)['model']
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if 'model.' in k}
        state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if 'encoder.' in k}
        for k in model_dict.keys():
            if 'head' in k:
                state_dict[k] = model_dict[k]
        msg = model.load_state_dict(state_dict, strict=False)
        print(("Pretrained weights found at {} and loaded with msg: {}".format(args.variant.params.pretrained_weights, msg)))

    def _load_positions(self, position_dir):
        network_positions = NetworkPositions.load_from_dir(position_dir)
        network_positions.to_torch()
        return network_positions.layer_positions

    def set_model(self, args):
        self.layernames = self.get_model_layernames(args)
        self.positions = None
        self.positions = self._load_positions(args.spatial_loss.position_dir)

    def set_eval_model(self, args):
        if self.eval_cfg is None:
            self.set_eval_cfg(args)
        self.eval_model = build_model(self.eval_cfg)
        if args.variant.params.pretrained_weights:
            self._load_pretrained_weights(args, self.eval_model)
        self.eval_model.cuda()

    def start_training_protocol(self, args):
        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        main(self.cfg, SpatialSwinMoBY, args, self.positions, self.layernames)

    def set_eval_protocol(self, args):
        pass

    def start_eval_protocol(self):
        eval_main(self.eval_cfg)

    def set_kinetics_cfg(self, args):
        variant_args = argparse.Namespace(**{
            "cfg": "../Transformer-SSL/configs/moby_swin_tiny.yaml",
            "batch_size": 32,
            "opts": [],
            "data_path": "/data2/ynshah/Kinetics400/k400/",
            "zip": False,
            "cache_mode": "no",
            "resume": "",
            "accumulation_steps": 0,
            "use_checkpoint": False,
            "amp_opt_level": "O0",
            "seed": 0,
            "output": str(Path(args.output_dir) / "kinetics"),
            "tag": "",
            "eval": False,
            "throughput": False,
            "distributed": True,
            "local_rank": 0,
            "lr": 1.0,
            "drop_path_rate": 0.2,
        })
        variant_cfg = get_config(variant_args)
        variant_cfg.defrost()
        variant_cfg.DATA.DATASET = 'kinetics400'
        variant_cfg.LINEAR_EVAL.PRETRAINED = args.variant.params.pretrained_weights
        variant_cfg.OUTPUT = os.path.join(variant_args.output, "kinetics")
        variant_cfg.MODEL.TYPE = 'linear'
        variant_cfg.MODEL.DROP_PATH_RATE = variant_args.drop_path_rate
        variant_cfg.AUG.SSL_AUG = False
        variant_cfg.AUG.SSL_LINEAR_AUG = True
        variant_cfg.AUG.MIXUP = 0.0
        variant_cfg.AUG.CUTMIX = 0.0
        variant_cfg.AUG.CUTMIX_MINMAX = None
        variant_cfg.TRAIN.EPOCHS = 10
        variant_cfg.TRAIN.WARMUP_EPOCHS = 5
        variant_cfg.TRAIN.LR_SCHEDULER.NAME = 'cosine'
        variant_cfg.TRAIN.OPTIMIZER.NAME = 'sgd'
        variant_cfg.TRAIN.OPTIMIZER.MOMENTUM = 0.9
        variant_cfg.TRAIN.BASE_LR = variant_args.lr
        variant_cfg.TRAIN.WEIGHT_DECAY = 0.0
        variant_cfg.freeze()
        self.kinetics_cfg = variant_cfg

    def set_kinetics_protocol(self, args):
        pass
    
    def start_kinetics_protocol(self):
        kinetics_main(self.kinetics_cfg)


def load_function_from_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class SpatialSwinMoBY(torch.nn.Module):
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
            hook_layername = layername
            if "attn_output" in layername:
                hook_layername = layername[:-12]
            layer = resolve_sequential_module_from_str(self.model.encoder, hook_layername)
            layer.register_forward_hook(self.get_hook_fn(layername))
            print(f"Registered hook for {layername}")

        run_name = args.name
        if args.wandb and is_master_process():
            wandb.init(
                project=args.variant.name,
                name=run_name,
            )

        save_dir = Path(args.output_dir) / "eval"
        save_dir.mkdir(parents=True, exist_ok=True)

    def get_hook_fn(self, layername):
        """Creates a hook function to capture the outputs of a specific layer."""
        def hook_fn(module, input, output):
            if "attn_output" in layername:
                output = module.attn_output
            spatial_features = output.float()
            N, L, H = spatial_features.shape
            sL = int(math.sqrt(L))
            spatial_features = spatial_features.reshape(N, sL, sL, H).permute(0, 3, 1, 2)
            self.intermediate_outputs[layername] = spatial_features
        return hook_fn

    def forward(self, samples1, samples2):
        self.intermediate_outputs.clear()

        loss_accumulator = self.model.forward(samples1, samples2)

        spatial_outputs = {}
        for layername, spatial_features in self.intermediate_outputs.items(): 
            spatial_outputs[layername] = (spatial_features, self.positions[layername])

        joint_loss = self.get_joint_loss((loss_accumulator, spatial_outputs))

        return joint_loss

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

            if self.iteration_counter % 100 == 0:
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
