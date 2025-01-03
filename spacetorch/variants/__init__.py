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
        "blocks.0": (257, 768),
        "blocks.1": (257, 768),
        "blocks.2": (257, 768),
        "blocks.3": (257, 768),
        "blocks.4": (257, 768),
        "blocks.5": (257, 768),
        "blocks.6": (257, 768),
        "blocks.7": (257, 768),
        "blocks.8": (257, 768),
        "blocks.9": (257, 768),
        "blocks.10": (257, 768),
        "blocks.11": (257, 768),
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
