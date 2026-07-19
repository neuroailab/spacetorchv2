import os
import torch
import argparse
import torch.nn as nn
from pathlib import Path
from collections import namedtuple
from typing import Union, Sequence
import torch.nn.functional as F

from vissl.hooks import default_hook_generator
from vissl.data.dataset_catalog import VisslDatasetCatalog
from vissl.utils.distributed_launcher import launch_distributed
from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict
from spacetorch.variants.base_arch import BaseArch
from spacetorch.variants.positions import NetworkPositions
from spacetorch.variants.assets.swin_moby.config import get_config
from spacetorch.variants.assets.llcnn.resnet_imagenet_continuoustopo_LLC_car import ResNet18
from spacetorch.variants.assets.swin_moby.moby_kinetics import main as kinetics_main


class LLCNN(BaseArch):
    """
    Implements the LLCNN architecture (Qian et al., 2024).
    """
    def __init__(self):
        super().__init__()

    def get_model_layernames(self, args):
        return list(args.spatial_loss.spatial_weight.keys())

    def set_cfg(self, args):
        pass

    def set_eval_cfg(self, args):
        try:
            cfg = compose_hydra_configuration([f"config=tdann_supervised_config_eval.yaml"])
            _, config = convert_to_attrdict(cfg)
            config["CHECKPOINT"]["DIR"] = (Path(args.output_dir)  / "eval") / "im1k"
            config["MODEL"]["TRUNK"]["TRUNK_PARAMS"]["position_dir"] = args.spatial_loss.position_dir
            print(config)
            self.eval_cfg = config
            VisslDatasetCatalog.register_data(
                name="custom_dataset",
                data_dict={
                    "train": [Path(args.variant.params.dataset_path) / "train", Path(args.variant.params.dataset_path) / "train"],
                    "test": [Path(args.variant.params.dataset_path) / "val", Path(args.variant.params.dataset_path) / "val"],
                }
            )
        except:
            pass

    def _load_pretrained_weights(self, args, model):
        state_dict = torch.load(args.variant.params.pretrained_weights, map_location="cpu")
        state_dict['state_dict'] = {k.replace('module.', ''): state_dict['state_dict'][k] for k in state_dict['state_dict'].keys()}
        msg = model.load_state_dict(state_dict['state_dict'], strict=False)
        print(("Pretrained weights found at {} and loaded with msg: {}".format(args.variant.params.pretrained_weights, msg)))

    def _load_positions(self, position_dir):
        network_positions = NetworkPositions.load_from_dir(position_dir)
        network_positions.to_torch()
        return network_positions.layer_positions

    def set_model(self, args):
        pass

    def set_eval_model(self, args):
        self.eval_model = load_model(pool_type='gaussian', kap_kernelsize=11, continuous=False, local_conv=False)
        if args.variant.params.pretrained_weights:
            self._load_pretrained_weights(args, self.eval_model)
        self.eval_model.cuda()
        self.eval_model.eval()

    def start_training_protocol(self, args):
        pass

    def set_eval_protocol(self, args):
        pass

    def start_eval_protocol(self):
        launch_distributed(
            cfg=self.eval_cfg,
            node_id=0,
            engine_name="train",
            hook_generator=default_hook_generator,
        )

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


def load_model(pool_type, kap_kernelsize, continuous, local_conv):
    Args = namedtuple('nt', ['dataset', 'arch', 'pool_type', 'max_num_pools', 'noise_std', 
                             'kap_kernelsize', 'kap_stride', 'expansion', 'do_prob', 
                             'continuous', 'local_conv'])
    
    args = Args(dataset="imagenet", arch="resnet18contopo_LLC_car", pool_type=pool_type, 
                max_num_pools=1, noise_std=0., kap_kernelsize=kap_kernelsize, kap_stride=1, 
                expansion=1, do_prob=0., continuous=continuous, local_conv=local_conv)
    
    model = ResNet18(1000, args.pool_type, args.max_num_pools, args.noise_std, args.kap_kernelsize, args.continuous, args.local_conv)

    return model
