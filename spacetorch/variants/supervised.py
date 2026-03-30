import torch
import argparse
from pathlib import Path

from vissl.hooks import default_hook_generator
from vissl.models.trunks.vision_transformer import VisionTransformer
from vissl.data.dataset_catalog import VisslDatasetCatalog
from vissl.utils.distributed_launcher import launch_distributed
from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict
from spacetorch.configs.dotpath import get_fn
from spacetorch.variants.base_arch import BaseArch
from spacetorch.variants.positions import NetworkPositions
from spacetorch.variants.assets.vissl.hooks import spatial_hook_generator


class Supervised(BaseArch):
    """
    Implements the supervised Cross Entropy objective function.
    """
    def __init__(self):
        super().__init__()
    
    def get_model_layernames(self, args):
        return list(args.spatial_loss.spatial_weight.keys())

    def set_cfg(self, args):
        cfg = compose_hydra_configuration([f"config={args.variant.params.config}"])
        _, config = convert_to_attrdict(cfg)
        config["CHECKPOINT"]["DIR"] = Path(args.output_dir) / "eval"
        config["MODEL"]["TRUNK"]["TRUNK_PARAMS"]["position_dir"] = args.spatial_loss.position_dir
        for layername, layer_weight in args.spatial_loss.spatial_weight.items():
            config["LOSS"][config["LOSS"]["name"]]["layer_weights"][layername] = layer_weight
            config["MODEL"]["TRUNK"]["TRUNK_PARAMS"]["layer_weights"][layername] = layer_weight
        print(config)
        self.cfg = config
        VisslDatasetCatalog.register_data(
            name="custom_dataset",
            data_dict={
                "train": [Path(args.variant.params.dataset_path) / "train", Path(args.variant.params.dataset_path) / "train"],
                "test": [Path(args.variant.params.dataset_path) / "val", Path(args.variant.params.dataset_path) / "val"],
            }
        )

    def set_eval_cfg(self, args):
        try:
            cfg = compose_hydra_configuration([f"config=vit_config_eval.yaml"])
            _, config = convert_to_attrdict(cfg)
            config["CHECKPOINT"]["DIR"] = Path(args.output_dir) / "eval"
            config["MODEL"]["WEIGHTS_INIT"]["PARAMS_FILE"] = args.variant.params.pretrained_weights
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
        state_dict = state_dict["classy_state_dict"]["base_model"]["model"]["trunk"]
        for k in list(state_dict.keys()):
            if k.startswith('base_model.'):
                state_dict[k[len("base_model."):]] = state_dict[k]
                del state_dict[k]
        msg = model.load_state_dict(state_dict, strict=False)
        print(("Pretrained weights found at {} and loaded with msg: {}".format(args.variant.params.pretrained_weights, msg)))

    def _load_positions(self, position_dir):
        network_positions = NetworkPositions.load_from_dir(position_dir)
        network_positions.to_torch()
        return network_positions.layer_positions

    def set_model(self, args):
        pass

    def set_eval_model(self, args):
        try:
            self.eval_cfg.MODEL
        except:
            self.set_eval_cfg(args)
        self.eval_model = VisionTransformer(self.eval_cfg.MODEL, self.eval_cfg.MODEL.TRUNK.NAME)
        if args.variant.params.pretrained_weights:
            self._load_pretrained_weights(args, self.eval_model)
        self.eval_model.cuda().eval()

    def start_training_protocol(self, args):
        launch_distributed(
            cfg=self.cfg,
            node_id=0,
            engine_name="train",
            hook_generator=spatial_hook_generator,
        )

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
        self.kinetics_cfg = argparse.Namespace(**{
            "output_dir": Path(args.output_dir) / "kinetics",
            "config_file": "../dinov2/dinov2/configs/train/vitb14_short.yaml",
            "opts": [],
            "pretrained_weights": args.variant.params.pretrained_weights,
            "train_dataset_str": "Kinetics400:split=TRAIN:root=/data2/ynshah/Kinetics400/k400/:extra=./",
            "val_dataset_str": "Kinetics400:split=VAL:root=/data2/ynshah/Kinetics400/k400/:extra=./",
        })
        config_file = get_fn("dinov2.utils.config.setup")
        config_file(self.kinetics_cfg)

    def set_kinetics_protocol(self, args):
        eval = get_fn("dinov2.eval.linear.run_eval_linear")
        self.eval_protocol = eval
    
    def start_kinetics_protocol(self):
        self.eval_protocol(
            model=self.eval_model,
            output_dir=self.kinetics_cfg.output_dir,
            train_dataset_str=self.kinetics_cfg.train_dataset_str,
            val_dataset_str=self.kinetics_cfg.val_dataset_str,
            test_dataset_strs=None,
            batch_size=32,
            epochs=10,
            epoch_length=1250,
            num_workers=8,
            save_checkpoint_frequency=20,
            eval_period_iterations=1250,
            learning_rates=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1],
            autocast_dtype=torch.half,
            resume=True,
            classifier_fpath="/ccn2/u/ynshah/spacetorchv2/checkpoints/vitb16a_supervised_imagenet/kinetics/last_checkpoint",
            test_metric_types=None,
            val_class_mapping_fpath=None,
            test_class_mapping_fpaths=[None],
        )
