from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Dict, Tuple, Union

import numpy as np
import torch
from spacetorch.types import Dims, VVSRegion

from spacetorch.utils.array_utils import FlatIndices, get_flat_indices
from spacetorch.constants import RETINA_SIZE, V1_SIZE, V2_SIZE, V4_SIZE, VTC_SIZE


TISSUE_SIZES: Dict[VVSRegion, float] = {
    "retina": RETINA_SIZE,
    "V1": V1_SIZE,
    "V2": V2_SIZE,
    "V4": V4_SIZE,
    "VTC": VTC_SIZE,
}

NEIGHBORHOOD_WIDTHS: Dict[VVSRegion, float] = {
    "retina": 0.0475,
    "V1": 3.7, # 1.626,
    "V2": 3.977,
    "V4": 2.545,
    "VTC": 31.818,
}

# at the the time models were trained, the positions were meant to match closer to estimates in macaque. We scale those initial positions by a fixed amount to make them a closer match to human data
RESCALE_POSITIONS_TO_HUMANS: Dict[VVSRegion, float] = {
    "retina": 0.6,
    "V1": 1.5,
    "V2": 1.75,
    "V4": 1.4,
    "VTC": 7,
}

BRAIN_MAPPING = {
    "resnet18": {
        "layer1.0": "retina",
        "layer1.1": "retina",
        "layer2.0": "V1",
        "layer2.1": "V1",
        "layer3.0": "V2",
        "layer3.1": "V4",
        "layer4.0": "VTC",
        "layer4.1": "VTC",
    },
    "vitb14": {
        "blocks.0.mlp.act": "retina",
        "blocks.1.mlp.act": "V1",
        "blocks.2.mlp.act": "V1",
        "blocks.3.mlp.act": "V2",
        "blocks.4.mlp.act": "V4",
        "blocks.5.mlp.act": "V4",
        "blocks.6.mlp.act": "V4",
        "blocks.7.mlp.act": "V4",
        "blocks.8.mlp.act": "VTC",
        "blocks.9.mlp.act": "VTC",
        "blocks.10.mlp.act": "VTC",
        "blocks.11.mlp.act": "VTC",
    }
}


@dataclass
class LayerPositions:

    # name should be the name of the layer these positions are for
    name: str

    # dims are in CHW format if conv, or a 1-tuple if fc
    dims: Union[Dims, Tuple[int]]

    # coordinates should be an N x 2 matrix with the x-coordinates of each unit in the
    # first column and the y-coordinates in the second column
    coordinates: np.ndarray

    # coordinates should be an P x Q matrix. Each row is one neighborhood consisting
    # of P indices. For all p_i \in P, 0 <= p_i <= len(coordinates)
    neighborhood_indices: np.ndarray

    # neighborhood_width is the width, in mm, of the neighborhoods
    neighborhood_width: float

    @property
    def flat_indices(self) -> FlatIndices:
        if len(self.dims) == 1:
            return np.arange(self.dims[0])
        elif len(self.dims) == 3:
            return get_flat_indices(self.dims)
        else:
            raise Exception(
                (
                    "Sorry, only FC (1-D shape) and conv (3-D shape) kernels are "
                    f"accepted, and dims was provided with {len(self.dims)} dimensions"
                )
            )

    def save(self, save_dir: Path):
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        path = save_dir / f"{self.name}.pkl"
        with path.open("wb") as stream:
            pickle.dump(self, stream)

    def save_np(self, save_dir: Path):
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        path = save_dir / f"{self.name}.npz"
        np.savez(path, **vars(self))

    @classmethod
    def load(cls, path: Path) -> "LayerPositions":
        path = Path(path)
        assert path.exists(), path

        if path.suffix == ".pkl":
            with path.open("rb") as stream:
                return pickle.load(stream)
        elif path.suffix == ".npz":
            state = np.load(path)

            def _scalar(x):
                return x[()]

            return cls(
                name=_scalar(state["name"]),
                dims=state["dims"],
                coordinates=state["coordinates"],
                neighborhood_indices=state["neighborhood_indices"],
                neighborhood_width=_scalar(state["neighborhood_width"]),
            )
        raise ValueError("suffix must be npz or pkl")

    def __len__(self) -> int:
        return len(self.coordinates)


@dataclass
class NetworkPositions:
    layer_positions: Dict[str, LayerPositions]
    version: int

    @classmethod
    def load_from_dir(cls, load_dir: Path):
        load_dir = Path(load_dir)
        assert load_dir.is_dir(), load_dir
        layer_files = list(load_dir.glob("*.pkl")) + list(load_dir.glob("*.npz"))

        d = {}
        for layer_file in layer_files:
            layer_name = layer_file.stem
            d[layer_name] = LayerPositions.load(layer_file)

        version_path = load_dir / "version.txt"
        version = 1.0
        if version_path.is_file():
            with version_path.open("r") as stream:
                version = int(stream.readline())

        return cls(version=version, layer_positions=d)

    def to_torch(self):
        """
        Converts each array or float in the original layer positions to be a torch
        Tensor of the appropriate type
        """
        for pos in self.layer_positions.values():
            pos.coordinates = torch.from_numpy(pos.coordinates.astype(np.float32))
            pos.neighborhood_indices = torch.from_numpy(
                pos.neighborhood_indices.astype(int)
            )
            pos.neighborhood_width = torch.tensor(pos.neighborhood_width)


def get_positions(
    cfg: str, rescale: bool = True
) -> Dict[str, LayerPositions]:
    load_dir = Path(cfg.spatial_loss.position_dir)

    netpos = NetworkPositions.load_from_dir(load_dir)
    layer_pos = dict(sorted(netpos.layer_positions.items()))

    rescale_lookup = {}

    for layer, brain_area in BRAIN_MAPPING[cfg.variant.architecture].items():
        rescale_lookup[layer] = RESCALE_POSITIONS_TO_HUMANS[brain_area]

    # perform rescaling
    for k, v in layer_pos.items():
        if rescale:
            v.coordinates = v.coordinates * rescale_lookup[k]
            v.neighborhood_width = v.neighborhood_width * rescale_lookup[k]
        layer_pos[k] = v

    return layer_pos
