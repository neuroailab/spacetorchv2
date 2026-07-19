import os
import re
import imp
import math
import json
import torch
import wandb
import torch.nn as nn
import argparse
from pathlib import Path
from typing import Union, Sequence
import torch.nn.functional as F

import spacetorch.variants.assets.rotnet.algorithms as alg
from spacetorch.variants.base_arch import BaseArch
from spacetorch.variants.positions import NetworkPositions
from spacetorch.variants.assets.swin_moby.config import get_config
from spacetorch.utils.generic_utils import convert_to_serializable
from spacetorch.losses.losses_torch import spatial_correlation_loss
from spacetorch.utils.torch_utils import resolve_sequential_module_from_str
from spacetorch.variants.assets.swin_moby.moby_kinetics import main as kinetics_main
from spacetorch.variants.assets.deepcluster.main import main
from spacetorch.variants.assets.deepcluster.util import load_model
from spacetorch.variants.assets.deepcluster.eval_linear import main as eval_main


class AlexNetDeepCluster(BaseArch):
    """
    Implements the DeepCluster objective function.
    """
    def __init__(self):
        super().__init__()
    
    def get_model_layernames(self, args):
        return list(args.spatial_loss.spatial_weight.keys())

    def set_cfg(self, args):
        self.cfg = argparse.Namespace(**{
            "data": Path(args.variant.params.dataset_path) / "train",
            "arch": "alexnet",
            "sobel": True,
            "clustering": "Kmeans",
            "nmb_cluster": 10000,
            "lr": 0.05,
            "wd": -5,
            "reassign": 1.,
            "workers": 12,
            "epochs": 500,
            "start_epoch": 0,
            "batch": 256,
            "momentum": 0.9,
            "resume": "",
            "checkpoints": 25000,
            "seed": 31,
            "exp": Path(args.output_dir) / "eval",
            "verbose": True,
        })

    def set_eval_cfg(self, args):
        self.eval_cfg = argparse.Namespace(**{
            "data": Path(args.variant.params.dataset_path),
            "model": args.variant.params.pretrained_weights,
            "conv": 3,
            "tencrops": True,
            "exp": Path(args.output_dir) / "linear",
            "lr": 0.01,
            "weight_decay": -7,
            "workers": 12,
            "epochs": 30,
            "batch_size": 256,
            "momentum": 0.9,
            "seed": 31,
            "verbose": True,
        })

    def _load_pretrained_weights(self, args, model):
        ckpt = torch.load(args.variant.params.pretrained_weights, map_location="cpu")
        model_params = ckpt["network"]

        adjusted_model_params = {k.replace("base_model.", ""): v for k, v in model_params.items()}

        msg = model.load_state_dict(adjusted_model_params, strict=False)
        print(("Pretrained weights found at {} and loaded with msg: {}".format(args.variant.params.pretrained_weights, msg)))

    def _load_positions(self, position_dir):
        network_positions = NetworkPositions.load_from_dir(position_dir)
        network_positions.to_torch()
        return network_positions.layer_positions

    def set_model(self, args):
        self.layernames = self.get_model_layernames(args)
        self.positions = self._load_positions(args.spatial_loss.position_dir)

    def set_eval_model(self, args):
        self.eval_model = load_model(args.variant.params.pretrained_weights)
        self.eval_model.cuda()

    def start_training_protocol(self, args):
        main(self.cfg, args, self.positions, self.layernames, SpatialAlexNetDeepCluster)

    def set_eval_protocol(self, args):
        eval_main(self.eval_cfg)

    def start_eval_protocol(self):
        pass

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


class SpatialAlexNetDeepCluster(torch.nn.Module):
    def __init__(self, base_encoder, args=None, positions=None, layernames=None):
        super().__init__()
        self.model = base_encoder
        self.positions = positions
        self.args = args
        self.iteration_counter = 0
        self.intermediate_outputs = {}
        self.hooks_enabled = True
        self.scaler = torch.cuda.amp.GradScaler()
        print(self.model)

        for layername in layernames:
            layer = resolve_sequential_module_from_str(self.model, layername)
            layer.register_forward_hook(self.get_hook_fn(layername))
            print(f"Registered hook for {layername}")

        run_name = args.name
        if args.wandb:
            wandb.init(
                project=args.name.split("_")[1],
                name=run_name,
            )
        save_dir = Path(args.output_dir) / "eval"
        save_dir.mkdir(parents=True, exist_ok=True)

    def get_hook_fn(self, layername):
        def hook_fn(module, input, output):
            if not self.hooks_enabled:
                return
            self.intermediate_outputs[layername] = output.float()
        return hook_fn

    def forward(self, x, compute_spatial=True):
        self.hooks_enabled = compute_spatial
        self.intermediate_outputs.clear()
        preds = self.model.forward(x)

        if not compute_spatial:
            return preds, 0.

        spatial_outputs = {
            layername: (spatial_features, self.positions[layername])
            for layername, spatial_features in self.intermediate_outputs.items()
        }
        return preds, self.get_spatial_loss(spatial_outputs)

    def get_spatial_loss(self, spatial_outputs):
        loss = 0.
        spatial_losses = {}
        for layername, layer_output in spatial_outputs.items():
            features, pos = layer_output
            spatial_losses[layername] = spatial_correlation_loss(
                features.cuda(),
                pos.coordinates.cuda(),
                pos.neighborhood_indices.cuda(),
            )
            loss += self.args.spatial_loss.spatial_weight[layername] * spatial_losses[layername]

        if self.iteration_counter % 50 == 0:
            serializable_spatial_losses = convert_to_serializable(spatial_losses)
            print(json.dumps(serializable_spatial_losses))
            save_spatial_losses_path = Path(self.args.output_dir) / "spatial_losses.json"
            with open(save_spatial_losses_path, 'a') as f:
                f.write(json.dumps(serializable_spatial_losses) + "\n")
            if self.args.wandb:
                wandb.define_metric("iter")
                wandb.log({
                    "spatial_loss": {**spatial_losses},
                    "iter": self.iteration_counter
                })

        self.iteration_counter += 1
        return loss


class KineticsWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.fc = nn.Identity()

    def _forward_stages(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        layer_outputs = []

        x = self.model.layer1(x)
        layer_outputs.append(x)

        x = self.model.layer2(x)
        layer_outputs.append(x)

        x = self.model.layer3(x)
        layer_outputs.append(x)

        x = self.model.layer4(x)
        layer_outputs.append(x)

        return layer_outputs

    def forward(self, x):
        layer_outputs = self._forward_stages(x)
        x = layer_outputs[-1]
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return x

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
    ):
        layer_outputs = self._forward_stages(x)

        total_layers = len(layer_outputs)

        if isinstance(n, int):
            layers_to_take = range(total_layers - n, total_layers)
        else:
            layers_to_take = n

        outputs = [layer_outputs[i] for i in layers_to_take]

        final_outputs = []
        for out in outputs:

            if reshape:
                pooled = F.adaptive_avg_pool2d(out, 1).flatten(1)

                if return_class_token:
                    B, C, H, W = out.shape
                    patch_tokens = out.flatten(2).transpose(1, 2)  # [B, HW, C]
                    pooled = F.adaptive_avg_pool2d(out, 1).flatten(1)  # [B, C]
                    final_outputs.append((patch_tokens, pooled))
                    continue

                out = pooled

            final_outputs.append(out)

        return tuple(final_outputs)
