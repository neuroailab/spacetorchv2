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
from spacetorch.variants.assets.rotnet.dataloader import DataLoader, GenericDataset
from spacetorch.variants.assets.rotnet.architectures.AlexNet import create_model


class AlexNetRotNet(BaseArch):
    """
    Implements the RotNet objective function.
    """
    def __init__(self):
        super().__init__()
    
    def get_model_layernames(self, args):
        return list(args.spatial_loss.spatial_weight.keys())

    def set_cfg(self, args):
        self.cfg = argparse.Namespace(**{
            "exp": "ImageNet_RotNet_AlexNet",
            "evaluate": False,
            "checkpoint": 0,
            "num_workers": 4,
            "cuda": True,
            "disp_step": 50,
        })

    def set_eval_cfg(self, args):
        self.eval_cfg = argparse.Namespace(**{
            "exp": "ImageNet_LinearClassifiers_ImageNet_RotNet_AlexNet_Features",
            "evaluate": False,
            "checkpoint": 0,
            "num_workers": 4,
            "cuda": True,
            "disp_step": 50,
        })

    def _load_pretrained_weights(self, args, model):
        ckpt = torch.load(args.variant.params.pretrained_weights, map_location="cpu")
        model_params = ckpt["network"]

        adjusted_model_params = {k.replace("model.", ""): v for k, v in model_params.items()}

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
        opt = {'num_classes': 4}
        self.eval_model = create_model(opt)
        if args.variant.params.pretrained_weights:
            self._load_pretrained_weights(args, self.eval_model)
        self.eval_model.cuda()

    def start_training_protocol(self, args):
        main(args, self.positions, self.layernames, args_opt=self.cfg)

    def set_eval_protocol(self, args):
        self.args = args

    def start_eval_protocol(self):
        main(self.args, None, None, args_opt=self.eval_cfg)

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


def main(args, positions, layernames, args_opt):
    exp_config_file = os.path.join('spacetorch/variants/assets/rotnet/', args_opt.exp+'.py')

    if positions is not None:
        exp_directory = os.path.join(args.output_dir, "eval")
    else:
        exp_directory = os.path.join(args.output_dir, "eval", "linear")

    # Load the configuration params of the experiment
    print('Launching experiment: %s' % exp_config_file)
    config = imp.load_source("", exp_config_file).config
    config['exp_dir'] = exp_directory # the place where logs, models, and other stuff will be stored
    print("Loading experiment %s from file: %s" % (args_opt.exp, exp_config_file))
    print("Generated logs, snapshots, and model files will be stored on %s" % (config['exp_dir']))

    # Set train and test datasets and the corresponding data loaders
    data_train_opt = config['data_train_opt']
    data_test_opt = config['data_test_opt']
    num_imgs_per_cat = data_train_opt['num_imgs_per_cat'] if ('num_imgs_per_cat' in data_train_opt) else None


    dataset_train = GenericDataset(
        dataset_name=data_train_opt['dataset_name'],
        split=data_train_opt['split'],
        random_sized_crop=data_train_opt['random_sized_crop'],
        num_imgs_per_cat=num_imgs_per_cat)
    dataset_test = GenericDataset(
        dataset_name=data_test_opt['dataset_name'],
        split=data_test_opt['split'],
        random_sized_crop=data_test_opt['random_sized_crop'])

    dloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=data_train_opt['batch_size'],
        unsupervised=data_train_opt['unsupervised'],
        epoch_size=data_train_opt['epoch_size'],
        num_workers=args_opt.num_workers,
        shuffle=True)

    dloader_test = DataLoader(
        dataset=dataset_test,
        batch_size=data_test_opt['batch_size'],
        unsupervised=data_test_opt['unsupervised'],
        epoch_size=data_test_opt['epoch_size'],
        num_workers=args_opt.num_workers,
        shuffle=False)

    config['disp_step'] = args_opt.disp_step

    if positions is not None:
        algorithm = getattr(alg, config['algorithm_type'])(config, model_wrapper=SpatialAlexNetRotNet, args=args, positions=positions, layernames=layernames)
    else:
        config['networks']['feat_extractor']['pretrained'] = args.variant.params.pretrained_weights
        algorithm = getattr(alg, config['algorithm_type'])(config)
        
    if args_opt.cuda: # enable cuda
        algorithm.load_to_gpu()
    if args_opt.checkpoint > 0: # load checkpoint
        algorithm.load_checkpoint(args_opt.checkpoint, train= (not args_opt.evaluate))

    if not args_opt.evaluate: # train the algorithm
        algorithm.solve(dloader_train, dloader_test)
    else:
        algorithm.evaluate(dloader_test) # evaluate the algorithm


class SpatialAlexNetRotNet(torch.nn.Module):
    def __init__(self, base_encoder, args=None, positions=None, layernames=None):
        super().__init__()
        self.model = base_encoder
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
        if args.wandb:
            wandb.init(
                project=args.name.split("_")[1],
                name=run_name,
            )
        
        save_dir = Path(args.output_dir) / "eval"
        save_dir.mkdir(parents=True, exist_ok=True)

    def get_hook_fn(self, layername):
        """Creates a hook function to capture the outputs of a specific layer."""
        def hook_fn(module, input, output):
            spatial_features = output.float()
            self.intermediate_outputs[layername] = spatial_features
        return hook_fn

    def forward(self, x):
        self.intermediate_outputs.clear()

        preds = self.model.forward(x)

        spatial_outputs = {}
        for layername, spatial_features in self.intermediate_outputs.items(): 
            spatial_outputs[layername] = (spatial_features, self.positions[layername])

        return preds, spatial_outputs

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

                spatial_losses[layername] = spatial_losses[layername]
                
                joint_loss += self.args.spatial_loss.spatial_weight[layername] * spatial_losses[layername]

            if self.iteration_counter % 50 == 0:
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
