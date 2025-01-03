from typing import Tuple
from typing_extensions import Literal

AggMode = Literal["mode", "mean", "circmean", "circstd"]

VVSRegion = Literal[
    "retina",
    "V1",
    "V2",
    "V4",
    "IT"
]

DeviceStr = Literal["cpu", "cuda"]

# shallow aliases
LayerString = str
Dims = Tuple[int, int, int]