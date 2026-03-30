import toponets
from pathlib import Path
from torchvision.models import vit_b_32, ViT_B_32_Weights

from spacetorch.variants.base_arch import BaseArch
from spacetorch.variants.positions import NetworkPositions


class TopoNet(BaseArch):
    """
    Implements the TopoNet architecture (Deb et al., 2025).
    """
    def __init__(self):
        super().__init__()
    
    def get_model_layernames(self, args):
        return list(args.spatial_loss.spatial_weight.keys())

    def set_cfg(self, args):
        pass

    def set_eval_cfg(self, args):
        pass

    def _load_pretrained_weights(self, args, model):
        pass

    def _load_positions(self, position_dir):
        network_positions = NetworkPositions.load_from_dir(position_dir)
        network_positions.to_torch()
        return network_positions.layer_positions

    def set_model(self, args):
        pass

    def set_eval_model(self, args):
        if "resnet18_spatial" in args.name:
            eval_model = toponets.resnet18(tau=10.0, checkpoint_path = Path(args.output_dir) / f"resnet18_tau_{10}.pt")
        elif "vitb32_spatial" in args.name:
            eval_model = toponets.vit_b_32(tau=10.0, checkpoint_path = Path(args.output_dir) / f"vitb32_tau_{10}.pt")
        else:
            eval_model = vit_b_32(weights=ViT_B_32_Weights)
        eval_model.eval()
        eval_model.to(self.device)
        self.eval_model = eval_model

    def start_training_protocol(self, args):
        pass

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