import functools
from typing import Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from spacetorch.datasets import sine_gratings
from spacetorch.datasets import DatasetRegistry
from spacetorch.types import AggMode
from spacetorch.utils.generic_utils import load_pickle, write_pickle
from spacetorch.maps.smoother import Smoother, KernelParams
from spacetorch.maps.v1_map import V1Map
from spacetorch.feature_extractor import get_features_from_layer

METRIC_DICT = sine_gratings.SineGrating2019.get_metrics(as_dict=True)


def get_sine_responses(
    model,
    layers,
    verbose: bool = True,
    normalize_to_ringach_firing_rates: bool = True,
    reduce_along_hw: bool = False,
    is_llcnn: bool = False,
    is_lcnn: bool = False,
) -> sine_gratings.SineResponses:
    if is_llcnn or is_lcnn:
        dataset = DatasetRegistry.get("SineGrating2019_Unnormalized")
    else:
        dataset = DatasetRegistry.get("SineGrating2019")
    sine_features, _, sine_labels = get_features_from_layer(
        model,
        dataset,
        model_layer_strings=layers,
        verbose=verbose,
        return_inputs_and_labels=True,
        reduce_along_hw=reduce_along_hw,
    )

    return sine_gratings.SineResponses(sine_features, sine_labels, normalize_to_ringach_firing_rates=normalize_to_ringach_firing_rates, is_llcnn=is_llcnn, is_lcnn=is_lcnn)


def get_smoothed_map(
    tissue: V1Map,
    metric: sine_gratings.Metric,
    nb_width: float = 1.5,
    final_width=None,
    final_stride=None,
    agg: Optional[AggMode] = None,
    verbose: bool = False,
) -> np.ndarray:
    def extractor(tissue_map: V1Map, metric: sine_gratings.Metric):
        return tissue_map.get_preferences(metric)

    final_width = final_width or nb_width / 1.5
    final_stride = final_stride or nb_width / 30
    kernel_params = KernelParams(width=final_width, stride=final_stride)
    smoother = Smoother(kernel_params, verbose=verbose)

    if agg is None:
        agg = metric.agg_mode

    smoothed = smoother(
        tissue, functools.partial(extractor, metric=metric), agg, high=metric.high
    )

    return smoothed


def get_sine_tissue(
    name,
    model,
    positions,
    layer: str = "blocks.1",
    skip_cache=False,
    output_dir: str = "checkpoints/",
    normalize_to_ringach_firing_rates: bool = True,
    smooth_orientation_tuning_curves: bool = True,
):
    # check for cached responses
    cache_id = name
    cache_loc = output_dir / "cache" / "tuning_curve_responses.pkl"

    if cache_loc.exists() and not skip_cache:
        responses = load_pickle(cache_loc)
        print(f"Loaded {cache_id} from cache")
    else:
        responses = get_sine_responses(
            model,
            layers=[layer],
            verbose=True,
            normalize_to_ringach_firing_rates=normalize_to_ringach_firing_rates,
        )
        # write_pickle(cache_loc, responses)

    return V1Map(
        positions.coordinates,
        responses,
        output_dir=output_dir,
        skip_cache=skip_cache,
        smooth_orientation_tuning_curves=smooth_orientation_tuning_curves
    )


def add_sine_colorbar(fig, ax, metric: sine_gratings.Metric, **kwargs):
    norm = mpl.colors.Normalize(vmin=0, vmax=metric.high)
    mappable = plt.cm.ScalarMappable(cmap=metric.colormap, norm=norm)
    mappable.set_array([])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(
        mappable, cax=cax, extend="both", extendrect=True, **kwargs
    )

    cbar.set_ticks([norm.vmin, norm.vmax])
    cbar.set_ticklabels([metric.xticklabels[0], metric.xticklabels[-1]])
    cbar.ax.tick_params(axis="y", rotation=90)
    
    return cbar


def get_smoothness_curves(
    tissue: V1Map,
    metric: sine_gratings.Metric = METRIC_DICT["angles"],
    shuffle: bool = False,
) -> Tuple[np.ndarray, ...]:
    """
    Returns:
        (midpoints, curves)
    """
    hcw = 3.5
    analysis_width = hcw * (4 / 3)

    # compute largest possible distance given the window size
    max_dist = np.sqrt(2 * analysis_width**2) / 2
    tissue.set_unit_mask_by_ptp_percentile("angles", 75)
    bin_edges = np.linspace(0, max_dist, 10)
    distances, curves = tissue.metric_difference_over_distance(
        metric=metric,
        distance_cutoff=max_dist,
        bin_edges=bin_edges,
        num_samples=20,
        sample_size=1000,
        shuffle=shuffle,
        verbose=False,
    )

    return distances, curves
