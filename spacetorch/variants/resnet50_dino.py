import os
import argparse
import json
import math
import torch
import wandb
import torch.distributed as dist
from pathlib import Path
from torchvision.models import resnet50

from spacetorch.configs.dotpath import get_fn
from spacetorch.utils.torch_utils import resolve_sequential_module_from_str
from spacetorch.utils.generic_utils import convert_to_serializable
from spacetorch.variants.base_arch import BaseArch
from spacetorch.variants.positions import NetworkPositions
from spacetorch.losses.losses_torch import spatial_correlation_loss
from spacetorch.utils.gpu_utils import is_master_process
from spacetorch.variants.assets.dino.eval_linear import eval_linear
from spacetorch.variants.assets.dino.main_dino import train_dino
from spacetorch.variants.assets.swin_moby.config import get_config
from spacetorch.variants.assets.swin_moby.moby_kinetics import main as kinetics_main


RESNET50_OUTPUT_DIMS = {
    "layer1.0": (256, 56, 56),
    "layer1.1": (256, 56, 56),
    "layer1.2": (256, 56, 56),
    "layer2.0": (512, 28, 28),
    "layer2.1": (512, 28, 28),
    "layer2.2": (512, 28, 28),
    "layer2.3": (512, 28, 28),
    "layer3.0": (1024, 14, 14),
    "layer3.1": (1024, 14, 14),
    "layer3.2": (1024, 14, 14),
    "layer3.3": (1024, 14, 14),
    "layer3.4": (1024, 14, 14),
    "layer3.5": (1024, 14, 14),
    "layer4.0": (2048, 7, 7),
    "layer4.1": (2048, 7, 7),
    "layer4.2": (2048, 7, 7),
}


class ResNet50_DINO(BaseArch):
    """
    Implements the DINO self-supervised objective function.
    """
    def __init__(self):
        super().__init__()
    
    def get_model_layernames(self, args):
        return list(args.spatial_loss.spatial_weight.keys())

    def set_cfg(self, args):
        variant_args = argparse.Namespace(**{
            "arch": "resnet50",
            "patch_size": 16,
            "out_dim": 65536,
            "norm_last_layer": True,
            "momentum_teacher": 0.996,
            "use_bn_in_head": False,
            "warmup_teacher_temp": 0.04,
            "teacher_temp": 0.04,
            "warmup_teacher_temp_epochs": 0,
            "use_fp16": False,
            "weight_decay": 1e-4,
            "weight_decay_end": 1e-4,
            "clip_grad": 3.0,
            "batch_size_per_gpu": 64,
            "epochs": 100,
            "freeze_last_layer": 1,
            "lr": 0.03,
            "warmup_epochs": 10,
            "min_lr": 1e-6,
            "optimizer": "sgd",
            "drop_path_rate": 0.1,
            "global_crops_scale": (0.14, 1.),
            "local_crops_number": 8,
            "local_crops_scale": (0.05, 0.14),
            "data_path": "/data/ynshah/imagenet/train/",
            "output_dir": Path(args.output_dir) / "eval",
            "saveckp_freq": 20,
            "seed": 0,
            "num_workers": 10,
            "dist_url": "env://",
            "local_rank": 0,
        })
        self.cfg = variant_args

    def set_eval_cfg(self, args):
        self.eval_cfg = argparse.Namespace(**{
            "n_last_blocks": 4,
            "avgpool_patchtokens": False,
            "arch": "resnet50",
            "patch_size": 16,
            "pretrained_weights": args.variant.params.pretrained_weights,
            "checkpoint_key": "teacher",
            "epochs": 30,
            "lr": 0.3,
            "batch_size_per_gpu": 128,
            "dist_url": "env://",
            "local_rank": 0,
            "data_path": "/data2/ynshah/imagenet/",
            "num_workers": 10,
            "val_freq": 1,
            "output_dir": Path(args.output_dir) / "eval",
            "num_labels": 1000,
            "evaluate": False,
        })

    def _load_pretrained_weights(self, args, model):
        state_dict = torch.load(args.variant.params.pretrained_weights, map_location="cpu")
        if "teacher" in state_dict:
            state_dict = state_dict["teacher"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(("Pretrained weights found at {} and loaded with msg: {}".format(args.variant.params.pretrained_weights, msg)))

    def _load_positions(self, position_dir):
        network_positions = NetworkPositions.load_from_dir(position_dir)
        network_positions.to_torch()
        return network_positions.layer_positions

    def set_model(self, args):
        self.layernames = self.get_model_layernames(args)
        self.positions = self._load_positions(args.spatial_loss.position_dir)

    def set_eval_model(self, args):
        self.eval_model = resnet50()
        if args.variant.params.pretrained_weights:
            self._load_pretrained_weights(args, self.eval_model)
        self.eval_model.cuda()


    def start_training_protocol(self, args):
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        train_dino(self.cfg, SpatialDINO, args, self.positions, self.layernames)

    def set_eval_protocol(self, args):
        pass

    def start_eval_protocol(self):
        eval_linear(self.eval_cfg)

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
        kinetics_main(self.kinetics_cfg, self.eval_model)


class SpatialDINO(torch.nn.Module):
    def __init__(self, backbone, args=None, positions=None, layernames=None):
        super().__init__()
        self.model = backbone
        print(self.model)

        self.positions = positions
        self.args = args
        self.iteration_counter = 0

        self.intermediate_outputs = {}

        self.scaler = torch.cuda.amp.GradScaler()

        # Register hooks on the intermediate layers of the model
        for layername in layernames:
            layer = resolve_sequential_module_from_str(self.model.backbone, layername)
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
            if output.shape[1:] in RESNET50_OUTPUT_DIMS.values():
                spatial_features = output.float()
                self.intermediate_outputs[layername] = spatial_features
        return hook_fn

    def forward(self, images):
        self.intermediate_outputs.clear()

        output = self.model.forward(images)

        spatial_outputs = {}
        for layername, spatial_features in self.intermediate_outputs.items():
            spatial_outputs[layername] = (spatial_features, self.positions[layername])

        return output, self.get_spatial_loss(spatial_outputs)

    def get_spatial_loss(self, spatial_outputs):
        spatial_losses = {}
        spatial_loss = 0.

        for layername, layer_output in spatial_outputs.items():
            features, pos = layer_output

            spatial_losses[layername] = spatial_correlation_loss(
                features,
                pos.coordinates.cuda(),
                pos.neighborhood_indices.cuda(),
            )
            
            spatial_loss += self.args.spatial_loss.spatial_weight[layername] * spatial_losses[layername]

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
                        "spatial_loss": {
                            **spatial_losses
                        },
                        "iter": self.iteration_counter
                    }
                    wandb.log(log_data)
        
        self.iteration_counter += 1

        return spatial_loss
