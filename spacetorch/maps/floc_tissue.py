import copy

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from pathlib import Path

from spacetorch.datasets import floc
from spacetorch.maps.it_map import ITMap
from spacetorch.variants.positions import LayerPositions
from spacetorch.utils.generic_utils import load_pickle, write_pickle
from spacetorch.feature_extractor import get_features_from_layer


def get_floc_responses(
    model,
    layers,
    verbose: bool = True,
) -> floc.fLocResponses:
    floc_features, _, floc_labels = get_features_from_layer(
        model,
        "fLoc",
        layers=layers,
        verbose=verbose,
        return_inputs_and_labels=True
    )

    return floc.fLocResponses(floc_features, floc_labels)


def get_floc_tissue(
    name: str,
    model: nn.Module,
    positions: LayerPositions,
    layer: str = "blocks.11",
    output_dir: str = "checkpoints/",
    **kwargs,
):
    """get_floc_tissue loads VTC-like tissue with responses to the fLoc stimuli

    Implements a caching mechanism, loading responses if possible from disk for the
    particular model and step requested.
    Once responses are loaded, they are paired with positions to create the tissue

    Inputs:
        model_name: the full name of the model config
        positions: the set of positions for the specified layer
        layer: the layer to get positions for
        step: either 'latest' or an epoch to load from
    """
    cache_id = name
    cache_loc = output_dir / "floc_responses.pkl"
    try:
        if cache_loc.exists():
            responses = load_pickle(cache_loc)
            print(f"Loaded {cache_id} from cache")
        else:
            responses = get_floc_responses(
                model, [layer], verbose=True, **kwargs
            )
            write_pickle(cache_loc, responses)

        tissue = ITMap(positions.coordinates, responses)
        tissue.cache_id = cache_id
        return tissue
    except AssertionError:
        return None


def add_custom_floc_legend(
    ax: plt.Axes, spacing: float = 2, include_nonsel: bool = True
):
    """Add the five categories (plus 'Non-selective' if desired) to the specified axis
    in a vertical list.
    """
    x, y = np.max(ax.get_xlim()), np.max(ax.get_ylim())
    contrasts_to_plot = copy.copy(floc.DOMAIN_CONTRASTS)
    if include_nonsel:
        contrasts_to_plot += [
            floc.Contrast(name="Non-selective", color="#BBB", on_categories=[])
        ]
    for idx, contrast in enumerate(contrasts_to_plot):
        ax.text(
            0.97 * x,
            0.9 * y - idx / spacing,
            contrast.name,
            fontdict={"color": contrast.color, "weight": 700},
        )
