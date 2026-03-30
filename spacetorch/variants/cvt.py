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

from spacetorch.utils.torch_utils import resolve_sequential_module_from_str
from spacetorch.utils.generic_utils import convert_to_serializable
from spacetorch.variants.base_arch import BaseArch
from spacetorch.variants.positions import NetworkPositions
from spacetorch.losses.losses_torch import spatial_correlation_loss
from spacetorch.utils.gpu_utils import is_master_process
from spacetorch.variants.assets.cvt.train import main
from spacetorch.variants.assets.cvt.models.build import build_model
from spacetorch.variants.assets.cvt.config import config
from spacetorch.variants.assets.cvt.config import update_config


class CvT(BaseArch):
    """
    Implements Convolutional Vision Transformers
    """
    def __init__(self):
        super().__init__()
    
    def get_model_layernames(self, args):
        return list(args.spatial_loss.spatial_weight.keys())

    def set_cfg(self, args):
        variant_args = argparse.Namespace(**{
            "cfg": args.variant.params.config,
            "opts": ["OUTPUT_DIR", str(Path(args.output_dir) / "eval")],
            "port": 9000,
            "local_rank": 0,
        })
        self.cfg = variant_args

    def set_eval_cfg(self, args):
        variant_args = argparse.Namespace(**{
            "cfg": args.variant.params.config,
            "opts": ["OUTPUT_DIR", args.output_dir],
            "port": 9000,
            "local_rank": 0,
        })
        update_config(config, variant_args)
        variant_cfg = config
        self.eval_cfg = variant_cfg

    def _load_pretrained_weights(self, args, model):
        checkpoint_dict = torch.load(args.variant.params.pretrained_weights, map_location='cpu')
        if "state_dict" in checkpoint_dict:
            state_dict = checkpoint_dict['state_dict']
        else:
            state_dict = checkpoint_dict
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if 'model.' in k}
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
        self.eval_model = build_model(self.eval_cfg)
        if args.variant.params.pretrained_weights:
            self._load_pretrained_weights(args, self.eval_model)
        self.eval_model.cuda()

    def start_training_protocol(self, args):
        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        main(self.cfg, SpatialCvT, args, self.positions, self.layernames)

    def set_eval_protocol(self, args):
        pass

    def start_eval_protocol(self):
        pass

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


class SpatialCvT(torch.nn.Module):
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
            if "attn_output" in layername:
                output = module.attn_output
            spatial_features = output.float()
            N, L, H = spatial_features.shape

            # Check if L is a perfect square
            sL = int(math.sqrt(L))
            is_square = sL * sL == L

            # Remove CLS token only if L is not already a square
            if not is_square:
                spatial_features = spatial_features[:, 1:]
                L = spatial_features.shape[1]
                sL = int(math.sqrt(L))

            spatial_features = spatial_features.reshape(N, sL, sL, H).permute(0, 3, 1, 2)
            self.intermediate_outputs[layername] = spatial_features
        return hook_fn

    def forward(self, x, criterion, y):
        self.intermediate_outputs.clear()

        outputs = self.model.forward(x)
        loss_accumulator = criterion(outputs, y)

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
