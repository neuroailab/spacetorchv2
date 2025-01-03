import numpy as np
import xarray as xr

from spacetorch.utils.array_utils import flatten
from spacetorch.datasets import imagenet
from spacetorch.maps import TissueMap


def create_imnet_tissue(features: np.ndarray, positions: np.ndarray):
    """Creates a simple tissue map from generic features"""
    _data = xr.DataArray(
        data=flatten(features),
        dims=["image_idx", "unit_idx"],
    )

    responses = imagenet.ImageNetResponses(_data=_data)
    return TissueMap(positions, responses)