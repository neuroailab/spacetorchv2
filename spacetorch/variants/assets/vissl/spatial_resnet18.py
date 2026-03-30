import math
import torch.nn as nn
from torchvision.models.resnet import resnet18

from vissl.utils.hydra_config import AttrDict
from vissl.models.trunks import register_model_trunk
from spacetorch.utils.torch_utils import resolve_sequential_module_from_str
from spacetorch.variants.positions import NetworkPositions


@register_model_trunk("spatial_resnet18")
class SpatialResNet18(nn.Module):
    def __init__(self, model_config: AttrDict, model_name: str):
        super(SpatialResNet18, self).__init__()

        # get the params trunk takes from the config
        self.model_config = model_config

        self.positions = self._load_positions(model_config["TRUNK"]["TRUNK_PARAMS"]["position_dir"])

        self.base_model = resnet18()
        self.base_model.fc = nn.Identity()

        self.intermediate_outputs = {}

        # Register hooks on the intermediate layers of the model
        for layername, layer_weight in model_config["TRUNK"]["TRUNK_PARAMS"]["layer_weights"].items():
            layer = resolve_sequential_module_from_str(self.base_model, layername)
            layer.register_forward_hook(self.get_hook_fn(layername))
            print(f"Registered hook for {layername}")

    def _load_positions(self, position_dir):
        network_positions = NetworkPositions.load_from_dir(position_dir)
        network_positions.to_torch()
        return network_positions.layer_positions
    
    def get_hook_fn(self, layername):
            """Creates a hook function to capture the outputs of a specific layer."""
            def hook_fn(module, input, output):
                spatial_features = output.float()
                self.intermediate_outputs[layername] = spatial_features
            return hook_fn

    # VISSL requires this signature for forward passes
    def forward(self, x, out_feat_keys=None):
        self.intermediate_outputs.clear()

        outputs = self.base_model(x)

        spatial_outputs = {}
        for layername, spatial_features in self.intermediate_outputs.items(): 
            spatial_outputs[layername] = (spatial_features, self.positions[layername])

        return outputs, spatial_outputs
