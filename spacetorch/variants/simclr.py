
import torch
from pathlib import Path

from vissl.models.trunks.vision_transformer import VisionTransformer
from vissl.data.dataset_catalog import VisslDatasetCatalog
from vissl.utils.distributed_launcher import launch_distributed
from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict
from spacetorch.utils.generic_utils import load_config_from_yaml
from spacetorch.variants.base_arch import BaseArch
from spacetorch.variants.positions import NetworkPositions
from spacetorch.variants.assets.vissl.hooks import spatial_hook_generator


class SimCLR(BaseArch):
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
        cfg = compose_hydra_configuration([f"config={args.variant.params.config}"])
        _, config = convert_to_attrdict(cfg)
        config["CHECKPOINT"]["DIR"] = args.output_dir
        print(config)
        
        self.eval_cfg = config
        VisslDatasetCatalog.register_data(
            name="custom_dataset",
            data_dict={
                "train": [Path(args.variant.params.dataset_path) / "train", Path(args.variant.params.dataset_path) / "train"],
                "test": [Path(args.variant.params.dataset_path) / "val", Path(args.variant.params.dataset_path) / "val"],
            }
        )

    def _load_pretrained_weights(self, args, model):
        state_dict = torch.load(args.variant.params.pretrained_weights, map_location="cpu")
        state_dict = state_dict["classy_state_dict"]["base_model"]["model"]["trunk"]
        msg = model.load_state_dict(state_dict, strict=False)
        print(("Pretrained weights found at {} and loaded with msg: {}".format(args.variant.params.pretrained_weights, msg)))

    def _load_positions(self, position_dir):
        network_positions = NetworkPositions.load_from_dir(position_dir)
        network_positions.to_torch()
        return network_positions.layer_positions

    def set_model(self, args):
        pass

    def set_eval_model(self, args):
        self.eval_model = VisionTransformer(self.eval_cfg.MODEL, self.eval_cfg.MODEL.TRUNK.NAME)
        if args.variant.params.pretrained_weights:
            self._load_pretrained_weights(args, self.eval_model)

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
        pass
