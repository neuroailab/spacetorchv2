import argparse
import json
import math
import torch
import wandb
import torch.distributed as dist
from pathlib import Path

from spacetorch.configs.dotpath import get_fn
from spacetorch.utils.generic_utils import convert_to_serializable
from spacetorch.variants.base_arch import BaseArch
from spacetorch.variants.positions import NetworkPositions
from spacetorch.losses.losses_torch import spatial_correlation_loss
from spacetorch.utils.gpu_utils import is_master_process


class DINOv2(BaseArch):
    """
    Implements the DINOv2 self-supervised objective function.
    """
    def __init__(self):
        super().__init__()
    
    def get_model_layernames(self):
        if self.base_model:
            return [f"blocks.{i}.mlp.act" for i in range(len(self.base_model.blocks))]
        else:
            return [f"blocks.{i}.mlp.act" for i in range(len(self.eval_model.blocks))]

    def set_cfg(self, args):
        variant_cfg_setup = get_fn(args.variant.setup.config)
        variant_args = argparse.Namespace(**{
            "output_dir": args.output_dir,
            "config_file": args.variant.params.config,
            "opts": [f"train.dataset_path={args.variant.params.dataset_path}"],
        })
        variant_cfg = variant_cfg_setup(variant_args)
        self.cfg = variant_cfg

    def set_eval_cfg(self, args):
        self.eval_cfg = argparse.Namespace(**{
            "output_dir": args.output_dir,
            "config_file": args.variant.params.config,
            "opts": [],
            "pretrained_weights": args.variant.params.pretrained_weights,
            "train_dataset_str": args.variant.params.dataset_path,
            "val_dataset_str": args.variant.params.dataset_path.replace("TRAIN", "VAL"),
        })

    def _load_pretrained_weights(self, args, model):
        state_dict = torch.load(args.variant.params.pretrained_weights, map_location="cpu")
        state_dict = state_dict["teacher"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(("Pretrained weights found at {} and loaded with msg: {}".format(args.variant.params.pretrained_weights, msg)))

    def _load_positions(self, position_dir):
        network_positions = NetworkPositions.load_from_dir(position_dir)
        network_positions.to_torch()
        return network_positions.layer_positions

    def set_model(self, args):
        variant_model_class = get_fn(args.variant.setup.model)

        def spatial_init(self, cfg, positions):
            super(SpatialDINOv2, self).__init__(cfg)
            self.positions = positions
            self.iteration_counter = 0

            self.intermediate_outputs = {}

            # Register hooks on the intermediate layers of the model
            for i, block in enumerate(self.student.backbone.blocks):
                if i in [2, 3, 4, 11]:
                    block.mlp.act.register_forward_hook(self.get_hook_fn(i))
                    print(f"Registered hook for blocks.{i}.mlp.act")

            run_name = args.name
            if args.wandb and is_master_process():
                wandb.init(
                    project="dinov2_imagenet_gelu",
                    name=run_name,
                )

        def get_hook_fn(self, layer_idx):
            """Creates a hook function to capture the outputs of a specific layer."""
            def hook_fn(module, input, output):
                features, _ = module.attn_bias.split(output)
                spatial_features = features[:, 1:].float()
                N, L, H = spatial_features.shape
                sL = int(math.sqrt(L))
                spatial_features = spatial_features.reshape(N, sL, sL, H).permute(0, 3, 1, 2)
                self.intermediate_outputs[layer_idx] = spatial_features
            return hook_fn

        def spatial_forward(self, images, teacher_temp):
            self.intermediate_outputs.clear()

            loss_accumulator, loss_dict = super(SpatialDINOv2, self).forward(images, teacher_temp)

            spatial_outputs = {}
            for i, spatial_features in self.intermediate_outputs.items(): 
                spatial_outputs[f'blocks.{i}.mlp.act'] = (spatial_features, self.positions[f'blocks.{i}.mlp.act'])

            return (loss_accumulator, spatial_outputs), loss_dict
        
        def spatial_backward(self, loss):
            if isinstance(loss, tuple):
                loss_accumulator, spatial_outputs = loss

                task_loss = loss_accumulator.clone()
                joint_loss = loss_accumulator
                
                spatial_losses = {}
                for layer, layer_output in spatial_outputs.items():
                    features, pos = layer_output

                    spatial_losses[layer] = spatial_correlation_loss(
                        features.cuda(),
                        pos.coordinates.cuda(),
                        pos.neighborhood_indices.cuda(),
                    )

                    if dist.get_world_size() > 1:
                        dist.all_reduce(spatial_losses[layer])
                    spatial_losses[layer] = spatial_losses[layer] / dist.get_world_size()
                    
                    joint_loss += args.spatial_loss.spatial_weight[layer] * spatial_losses[layer]

                if self.iteration_counter % 10 == 0:
                    if is_master_process():
                        serializable_spatial_losses = convert_to_serializable(spatial_losses)
                        print(json.dumps(serializable_spatial_losses))
                        save_spatial_losses_path = Path(args.output_dir) / "spatial_losses.json"
                        with open(save_spatial_losses_path, 'a') as f:
                            f.write(json.dumps(serializable_spatial_losses) + "\n")

                        if args.wandb:
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

            return super(SpatialDINOv2, self).backprop_loss(joint_loss)

        SpatialDINOv2 = type(
            "SpatialDINOv2", 
            (variant_model_class,),
            {
                "__init__": spatial_init,
                "forward": spatial_forward,
                "backprop_loss": spatial_backward,
                "get_hook_fn": get_hook_fn,
            }
        )

        # load positions
        positions = self._load_positions(args.spatial_loss.position_dir)

        self.model = SpatialDINOv2(self.cfg, positions)
        print(self.model)

        # set base model
        if args.variant.params.pretrained_weights:
            self._load_pretrained_weights(args, self.model.teacher)

        self.model.to(self.device)

        self.base_model = self.model.teacher.backbone
        
        self.model.prepare_for_distributed_training()

    def set_eval_model(self, args):
        eval_model = get_fn(args.variant.setup.eval_model)
        self.eval_model, _ = eval_model(self.eval_cfg)

    def set_training_protocol(self, args):
        train = get_fn(args.variant.setup.train)
        self.training_procotol = train

    def start_training_protocol(self):
        self.training_procotol(self.cfg, self.model, resume=False)

    def set_eval_protocol(self, args):
        eval = get_fn(args.variant.setup.eval)
        self.eval_protocol = eval

    def start_eval_protocol(self):
        self.eval_protocol(
            model=self.eval_model,
            output_dir=self.eval_cfg.output_dir,
            train_dataset_str=self.eval_cfg.train_dataset_str,
            val_dataset_str=self.eval_cfg.val_dataset_str,
            test_dataset_strs=None,
            batch_size=128,
            epochs=10,
            epoch_length=1250,
            num_workers=8,
            save_checkpoint_frequency=20,
            eval_period_iterations=1250,
            learning_rates=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1],
            autocast_dtype=torch.half,
            resume=False,
            classifier_fpath=None,
            test_metric_types=None,
            val_class_mapping_fpath=None,
            test_class_mapping_fpaths=[None],
        )
