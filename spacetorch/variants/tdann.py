import re
import torch
from torchvision.models import resnet18
from spacetorch.variants.base_arch import BaseArch


class TDANN(BaseArch):
    """
    Implements the DINOv2 self-supervised objective function.
    """
    def __init__(self):
        super().__init__()
    
    def get_model_layernames(self):
        return [f"layer{i}.{j}" for i in range(1, 5) for j in range(2)]

    def set_cfg(self, args):
        pass

    def set_eval_cfg(self, args):
        pass

    def _load_pretrained_weights(self, args, model):
        ckpt = torch.load(args.variant.params.pretrained_weights, map_location="cpu")
        model_params = ckpt["classy_state_dict"]["base_model"]["model"]["trunk"]

        adjusted_model_params = {}
        pattern = re.compile(r"(\d+)_(\d+)")

        for key, value in model_params.items():
            new_key = pattern.sub(r'\1.\2', key)
            adjusted_model_params[new_key] = value

        adjusted_model_params = {k.replace("base_model.", ""): v for k, v in adjusted_model_params.items()}
        adjusted_model_params = {k.replace("_feature_blocks.", ""): v for k, v in adjusted_model_params.items()}

        msg = model.load_state_dict(adjusted_model_params, strict=False)
        print(("Pretrained weights found at {} and loaded with msg: {}".format(args.variant.params.pretrained_weights, msg)))

    def _load_positions(self, position_dir):
        pass

    def set_model(self, args):
        pass

    def set_eval_model(self, args):
        eval_model = resnet18()
        if args.variant.params.pretrained_weights:
            self._load_pretrained_weights(args, eval_model)
        eval_model.eval()
        eval_model.to(self.device)
        self.eval_model = eval_model

    def set_training_protocol(self, args):
        pass

    def start_training_protocol(self):
        pass

    def set_eval_protocol(self, args):
        pass

    def start_eval_protocol(self):
        pass
