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
from spacetorch.variants.assets.swinv2_simmim.main_simmim_pt import main
from spacetorch.variants.assets.swinv2_simmim.models.build import build_model
from spacetorch.variants.assets.swinv2_simmim.config import get_config


class SwinV2SimMIM(BaseArch):
    """
    Implements the SimMIM (Masked Image Modeling) objective function for Swin Transformers
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
            "enable_amp": True,
            "output": str(Path(args.output_dir) / "eval"),
            "distributed": True,
            "local_rank": -1,
        })
        variant_cfg = get_config(variant_args)
        self.cfg = variant_cfg

    def set_eval_cfg(self, args):
        variant_args = argparse.Namespace(**{
            "cfg": args.variant.params.config,
            "batch_size": 256,
            "opts": [],
            "data_path": args.variant.params.dataset_path,
            "enable_amp": True,
            "output": str(Path(args.output_dir) / "eval"),
            "distributed": False,
            "local_rank": 0,
        })
        variant_cfg = get_config(variant_args)
        self.eval_cfg = variant_cfg

    def _load_pretrained_weights(self, args, model):
        checkpoint = torch.load(args.variant.params.pretrained_weights, map_location="cpu", weights_only=False)
        checkpoint_model = checkpoint['model']
        if any([True if 'model.' in k else False for k in checkpoint_model.keys()]):
            checkpoint_model = {k.replace('model.', ''): v for k, v in checkpoint_model.items() if k.startswith('model.')}
        if any([True if 'encoder.' in k else False for k in checkpoint_model.keys()]):
            checkpoint_model = {k.replace('encoder.', ''): v for k, v in checkpoint_model.items() if k.startswith('encoder.')}
        checkpoint = remap_pretrained_keys_swin(model, checkpoint_model)
        msg = model.load_state_dict(checkpoint_model, strict=False)
        del checkpoint
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
        self.eval_model = build_model(self.eval_cfg, is_pretrain=False)
        if args.variant.params.pretrained_weights:
            self._load_pretrained_weights(args, self.eval_model)
        self.eval_model.cuda()

    def start_training_protocol(self, args):
        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        main(self.cfg, SpatialSwinV2SimMIM, args, self.positions, self.layernames)

    def set_eval_protocol(self, args):
        pass

    def start_eval_protocol(self):
        pass


def remap_pretrained_keys_swin(model, checkpoint_model):
    state_dict = model.state_dict()
    
    # Geometric interpolation when pre-trained patch size mismatch with fine-tuned patch size
    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_bias_table" in key:
            relative_position_bias_table_pretrained = checkpoint_model[key]
            relative_position_bias_table_current = state_dict[key]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            if nH1 != nH2:
                print(f"Error in loading {key}, passing......")
            else:
                if L1 != L2:
                    print(f"{key}: Interpolate relative_position_bias_table using geo.")
                    src_size = int(L1 ** 0.5)
                    dst_size = int(L2 ** 0.5)

                    def geometric_progression(a, r, n):
                        return a * (1.0 - r ** n) / (1.0 - r)

                    left, right = 1.01, 1.5
                    while right - left > 1e-6:
                        q = (left + right) / 2.0
                        gp = geometric_progression(1, q, src_size // 2)
                        if gp > dst_size // 2:
                            right = q
                        else:
                            left = q

                    # if q > 1.090307:
                    #     q = 1.090307

                    dis = []
                    cur = 1
                    for i in range(src_size // 2):
                        dis.append(cur)
                        cur += q ** (i + 1)

                    r_ids = [-_ for _ in reversed(dis)]

                    x = r_ids + [0] + dis
                    y = r_ids + [0] + dis

                    t = dst_size // 2.0
                    dx = np.arange(-t, t + 0.1, 1.0)
                    dy = np.arange(-t, t + 0.1, 1.0)

                    print("Original positions = %s" % str(x))
                    print("Target positions = %s" % str(dx))

                    all_rel_pos_bias = []

                    for i in range(nH1):
                        z = relative_position_bias_table_pretrained[:, i].view(src_size, src_size).float().numpy()
                        f_cubic = interpolate.interp2d(x, y, z, kind='cubic')
                        all_rel_pos_bias.append(torch.Tensor(f_cubic(dx, dy)).contiguous().view(-1, 1).to(
                            relative_position_bias_table_pretrained.device))

                    new_rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
                    checkpoint_model[key] = new_rel_pos_bias

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in checkpoint_model.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del checkpoint_model[k]

    # delete relative_coords_table since we always re-init it
    relative_coords_table_keys = [k for k in checkpoint_model.keys() if "relative_coords_table" in k]
    for k in relative_coords_table_keys:
        del checkpoint_model[k]

    # re-map keys due to name change
    rpe_mlp_keys = [k for k in checkpoint_model.keys() if "rpe_mlp" in k]
    for k in rpe_mlp_keys:
        checkpoint_model[k.replace('rpe_mlp', 'cpb_mlp')] = checkpoint_model.pop(k)

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in checkpoint_model.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del checkpoint_model[k]

    return checkpoint_model


def load_function_from_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class SpatialSwinV2SimMIM(torch.nn.Module):
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

    def forward(self, x, mask):
        self.intermediate_outputs.clear()

        loss_accumulator = self.model.forward(x, mask)

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
