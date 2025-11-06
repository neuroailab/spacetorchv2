import argparse
import json
import sys
import math
import torch
import wandb
import importlib.util
import numpy as np
import torch.distributed as dist
import torch.distributed as dist
from pathlib import Path
from scipy import interpolate
from timm.models import create_model

from spacetorch.utils.torch_utils import resolve_sequential_module_from_str
from spacetorch.utils.generic_utils import convert_to_serializable
from spacetorch.variants.base_arch import BaseArch
from spacetorch.variants.positions import NetworkPositions
from spacetorch.losses.losses_torch import spatial_correlation_loss
from spacetorch.utils.gpu_utils import is_master_process
from spacetorch.variants.assets.convit.main import main


class ConViT(BaseArch):
    """
    Implements ConViT: Vision transformers with convolutional inductive biases
    """
    def __init__(self):
        super().__init__()
    
    def get_model_layernames(self, args):
        return list(args.spatial_loss.spatial_weight.keys())

    def set_cfg(self, args):
        variant_args = argparse.Namespace(**{
            "batch_size": 128,
            "epochs": 300,
            "model": "convit_base",
            "pretrained": False,
            "input_size": 224,
            "embed_dim": 48,
            "drop": 0.0,
            "drop_path": 0.1,
            "drop_block": None,
            "model_ema": False,
            "model_ema_decay": 0.99996,
            "model_ema_force_cpu": False,
            "opt": "adamw",
            "opt_eps": 1e-8,
            "opt_betas": None,
            "clip_grad": 5,
            "momentum": 0.9,
            "weight_decay": 0.05,
            "sched": "cosine",
            "lr": 5e-4,
            "lr_noise": None,
            "lr_noise_pct": 0.67,
            "lr_noise_std": 1.0,
            "warmup_lr": 1e-6,
            "min_lr": 1e-5,
            "decay_epochs": 30,
            "warmup_epochs": 5,
            "cooldown_epochs": 10,
            "patience_epochs": 10,
            "decay_rage": 0.1,
            "color_jitter": 0.4,
            "aa": "rand-m9-mstd0.5-inc1",
            "smoothing": 0.1,
            "train_interpolation": "bicubic",
            "repeated_aug": True,
            "reprob": 0.25,
            "remode": "pixel",
            "recount": 1,
            "resplit": False,
            "mixup": 0.8,
            "cutmix": 1.0,
            "cutmix_minmax": None,
            "mixup_prob": 1.0,
            "mixup_switch_prob": 0.5,
            "mixup_mode": "batch",
            "data_path": args.variant.params.dataset_path,
            "data_set": "IMNET",
            "sampling_ratio": 1.,
            "nb_classes": 1000,
            "inat_category": "name",
            "output_dir": Path(args.output_dir) / "eval",
            "device": "cuda",
            "seed": 0,
            "resume": "/ccn2/u/ynshah/spacetorchv2/checkpoints/convit_base_supervised_imagenet/eval/checkpoint.pth",
            "eval": False,
            "save_every": 25,
            "start_epoch": 0,
            "num_workers": 10,
            "pin_mem": True,
            "world_size": 1,
            "dist_url": "env://",
            "local_up_to_layer": 10,
            "locality_strength": 1.,
            "local_rank": 0,
        })
        self.cfg = variant_args

    def set_eval_cfg(self, args):
        variant_args = argparse.Namespace(**{
            "batch_size": 128,
            "epochs": 300,
            "model": "convit_base",
            "pretrained": False,
            "input_size": 224,
            "embed_dim": 48,
            "drop": 0.0,
            "drop_path": 0.1,
            "drop_block": None,
            "model_ema": False,
            "model_ema_decay": 0.99996,
            "model_ema_force_cpu": False,
            "opt": "adamw",
            "opt_eps": 1e-8,
            "opt_betas": None,
            "clip_grad": 5,
            "momentum": 0.9,
            "weight_decay": 0.05,
            "sched": "cosine",
            "lr": 5e-4,
            "lr_noise": None,
            "lr_noise_pct": 0.67,
            "lr_noise_std": 1.0,
            "warmup_lr": 1e-6,
            "min_lr": 1e-5,
            "decay_epochs": 30,
            "warmup_epochs": 5,
            "cooldown_epochs": 10,
            "patience_epochs": 10,
            "decay_rage": 0.1,
            "color_jitter": 0.4,
            "aa": "rand-m9-mstd0.5-inc1",
            "smoothing": 0.1,
            "train_interpolation": "bicubic",
            "repeated_aug": True,
            "reprob": 0.25,
            "remode": "pixel",
            "recount": 1,
            "resplit": False,
            "mixup": 0.8,
            "cutmix": 1.0,
            "cutmix_minmax": None,
            "mixup_prob": 1.0,
            "mixup_switch_prob": 0.5,
            "mixup_mode": "batch",
            "data_path": args.variant.params.dataset_path,
            "data_set": "IMNET",
            "sampling_ratio": 1.,
            "nb_classes": 1000,
            "inat_category": "name",
            "output_dir": Path(args.output_dir) / "eval",
            "device": "cuda",
            "seed": 0,
            "resume": "",
            "eval": False,
            "save_every": 25,
            "start_epoch": 0,
            "num_workers": 10,
            "pin_mem": True,
            "world_size": 1,
            "dist_url": "env://",
            "local_up_to_layer": 10,
            "locality_strength": 1.,
            "local_rank": 0,
        })
        self.eval_cfg = variant_args

    def _load_pretrained_weights(self, args, model):
        checkpoint_dict = torch.load(args.variant.params.pretrained_weights, map_location='cpu')
        msg = model.load_state_dict(checkpoint_dict, strict=False)
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
        self.eval_model = create_model(
            self.eval_cfg.model,
            pretrained=self.eval_cfg.pretrained,
            num_classes=self.eval_cfg.nb_classes,
            drop_rate=self.eval_cfg.drop,
            drop_path_rate=self.eval_cfg.drop_path,
            drop_block_rate=self.eval_cfg.drop_block,
            local_up_to_layer=self.eval_cfg.local_up_to_layer,
            locality_strength=self.eval_cfg.locality_strength,
            embed_dim = self.eval_cfg.embed_dim,
        )
        if args.variant.params.pretrained_weights:
            self._load_pretrained_weights(args, self.eval_model)
        self.eval_model.cuda()

    def start_training_protocol(self, args):
        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        main(self.cfg, SpatialConViT, args, self.positions, self.layernames)

    def set_eval_protocol(self, args):
        pass

    def start_eval_protocol(self):
        pass


def load_function_from_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class SpatialConViT(torch.nn.Module):
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
            layer = resolve_sequential_module_from_str(self.model, hook_layername)
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
            spatial_features = output.float()
            N, L, H = spatial_features.shape
            sL = int(math.sqrt(L))
            if sL * sL != L:
                spatial_features = spatial_features[:, 1:]
                N, L, H = spatial_features.shape
                sL = int(math.sqrt(L))
            spatial_features = spatial_features.reshape(N, sL, sL, H).permute(0, 3, 1, 2)
            self.intermediate_outputs[layername] = spatial_features
        return hook_fn

    def forward(self, samples, criterion, targets):
        self.intermediate_outputs.clear()

        outputs = self.model.forward(samples)
        loss_accumulator = criterion(outputs, targets)

        spatial_outputs = {}
        for layername, spatial_features in self.intermediate_outputs.items(): 
            spatial_outputs[layername] = (spatial_features, self.positions[layername])

        joint_loss = self.get_joint_loss((loss_accumulator, spatial_outputs))

        return joint_loss, outputs

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
