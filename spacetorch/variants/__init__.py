from .base_arch import BaseArch
from .dinov2 import DINOv2
from .tdann import TDANN


OUTPUT_DIMS_FOR_224_INPUTS = {
    "resnet18": {
        "layer1.0": (64, 56, 56),
        "layer1.1": (64, 56, 56),
        "layer2.0": (128, 28, 28),
        "layer2.1": (128, 28, 28),
        "layer3.0": (256, 14, 14),
        "layer3.1": (256, 14, 14),
        "layer4.0": (512, 7, 7),
        "layer4.1": (512, 7, 7),
    },
    "vitb14": {
        "blocks.0": (768, 16, 16),
        "blocks.1": (768, 16, 16),
        "blocks.2": (768, 16, 16),
        "blocks.3": (768, 16, 16),
        "blocks.4": (768, 16, 16),
        "blocks.5": (768, 16, 16),
        "blocks.6": (768, 16, 16),
        "blocks.7": (768, 16, 16),
        "blocks.8": (768, 16, 16),
        "blocks.9": (768, 16, 16),
        "blocks.10": (768, 16, 16),
        "blocks.11": (768, 16, 16),
    },
    "vitb14am": {
        "blocks.0.attn": (768, 16, 16),
        "blocks.0.mlp": (768, 16, 16),
        "blocks.1.attn": (768, 16, 16),
        "blocks.1.mlp": (768, 16, 16),
        "blocks.2.attn": (768, 16, 16),
        "blocks.2.mlp": (768, 16, 16),
        "blocks.3.attn": (768, 16, 16),
        "blocks.3.mlp": (768, 16, 16),
        "blocks.4.attn": (768, 16, 16),
        "blocks.4.mlp": (768, 16, 16),
        "blocks.5.attn": (768, 16, 16),
        "blocks.5.mlp": (768, 16, 16),
        "blocks.6.attn": (768, 16, 16),
        "blocks.6.mlp": (768, 16, 16),
        "blocks.7.attn": (768, 16, 16),
        "blocks.7.mlp": (768, 16, 16),
        "blocks.8.attn": (768, 16, 16),
        "blocks.8.mlp": (768, 16, 16),
        "blocks.9.attn": (768, 16, 16),
        "blocks.9.mlp": (768, 16, 16),
        "blocks.10.attn": (768, 16, 16),
        "blocks.10.mlp": (768, 16, 16),
        "blocks.11.attn": (768, 16, 16),
        "blocks.11.mlp": (768, 16, 16),
    }
}


VARIANT_MAPPING = {
    "dinov2": DINOv2(),
    "tdann": TDANN(),
}


def get_variant(name: str) -> BaseArch:
    if name in VARIANT_MAPPING:
        return VARIANT_MAPPING[name]
    else:
        raise NotImplementedError(f"Support for '{name}' has not been implemented yet.")
