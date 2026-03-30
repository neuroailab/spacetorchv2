import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from spacetorch.configs import get_cfg
from spacetorch.variants import get_variant
from spacetorch.variants.positions import get_positions
from spacetorch.maps import TissueMap
from spacetorch.maps.sine_tissue import get_sine_tissue, METRIC_DICT
from spacetorch.maps.screenshot_maps import (
    NauhausOrientationTissue,
    NauhausSFTissue,
    LivingstoneColorTissue,
)
from spacetorch.utils import array_utils, spatial_utils


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--layer", type=str)
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


args = get_parser().parse_args()


def get_curves(
    hypercolumn_width: float,
    tissue: TissueMap,
    name: str,
    metric_name: str,
    shuffle: bool = False,
    num_samples: int = 20,
    sample_size: int = 1000,
    verbose: bool = False,
):
    # make the analysis width slightly larger to capture full rise and fall
    hcw = hypercolumn_width
    analysis_width = hcw * (4 / 3)

    # compute largest possible distance given the window size
    max_dist = np.sqrt(2 * analysis_width**2) / 2

    # create 9 bins, going from 0 (closest) to max_dist
    bin_edges = np.linspace(0, max_dist, 10)
    midpoints = array_utils.midpoints_from_bin_edges(bin_edges)

    # convenience: store arguments shared by both conditional flows into a dict
    common = {
        "num_samples": num_samples,
        "sample_size": sample_size,
        "bin_edges": bin_edges,
        "shuffle": shuffle,
        "verbose": verbose,
    }

    if name == "macaque":
        _, curves = tissue[metric_name].difference_over_distance(**common)
    else:
        tissue.reset_unit_mask()
        tissue.set_unit_mask_by_ptp_percentile(metric_name, 75)
        _, curves = tissue.metric_difference_over_distance(
            distance_cutoff=max_dist, **common
        )

    # normalize midpoints to be a fraction of the hypercolumn width
    return midpoints / hcw, curves


def get_smoothness(
    hypercolumn_width: float,
    tissue: TissueMap,
    name: str
):
    smoothness_results = {"Smoothness": [], "Metric": []}

    curve_dict = {}
    for metric_name in METRIC_DICT.keys():
        distances, curves = get_curves(hypercolumn_width, tissue, name, metric_name)

        # compute smoothness for each curve
        smoos = [spatial_utils.smoothness(curve) for curve in curves]
        mean_smoothness = np.nanmean(smoos)

        smoothness_results["Smoothness"].append(mean_smoothness)
        smoothness_results["Metric"].append(metric_name)

        # for plotting, figure out what value we'd expect by chance
        _, chance_curves = get_curves(hypercolumn_width, tissue, name, metric_name, shuffle=True)
        chance_mean = np.nanmean(np.concatenate(chance_curves))
        norm_curves = [curve / chance_mean for curve in curves]

        curve_dict[metric_name] = {
            "Distances": distances * 100,  # convert to percentages
            "Curves": norm_curves,
        }

    return smoothness_results, curve_dict


def get_macaque_tissues():
    ori_tissue = NauhausOrientationTissue()
    sf_tissue = NauhausSFTissue()
    color_tissue = LivingstoneColorTissue()
    return {"angles": ori_tissue, "sfs": sf_tissue, "colors": color_tissue}


def main():
    cfg = get_cfg(args)
    variant = get_variant(cfg.variant.name)
    variant.set_eval_cfg(cfg)
    variant.set_eval_model(cfg)

    model = variant.eval_model
    is_tdann = "tdann" in cfg.name and not "tdann_logpolar" in cfg.name
    positions = get_positions(cfg, rescale=is_tdann)[args.layer]

    is_swinv2 = ("swinv2" in cfg.name)

    save_dir = Path(cfg.output_dir) / args.layer
    save_dir.mkdir(parents=True, exist_ok=True)

    # if (save_dir / "v1_smoothness.npy").exists():
    #     print(f"Found existing results in {save_dir}, skipping...")
    #     return

    v1_tissue = get_sine_tissue(
        cfg.name,
        model,
        positions,
        layer=args.layer,
        output_dir=save_dir,
        smooth_orientation_tuning_curves=False,
        is_swinv2=is_swinv2,
        skip_cache=True,
    )

    # macaque_tissue = get_macaque_tissues()
    # smoothness_results, curve_dict = get_smoothness(0.75, macaque_tissue, "macaque")

    try:
        column_spacing = np.load(save_dir / "pinwheel_density.npz", allow_pickle=True)["column_spacing_in_mm"].item()
    except:
        print("Please run v1_pinwheel_density.py first to compute hypercolumn spacing.")

    smoothness_results, curve_dict = get_smoothness(column_spacing, v1_tissue, "not macaque")
    np.save(save_dir / "v1_smoothness_right_hypercolumn.npy", smoothness_results)
    np.savez(save_dir / "v1_preference_right_hypercolumn.npz", **curve_dict)

    plt.figure(figsize=(1.5, 1.5))
    b = plt.bar(smoothness_results["Metric"], smoothness_results["Smoothness"], color="black")
    for rect in b:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.3f}', ha='center', va='bottom', fontsize=7)
    plt.ylabel("Smoothness")
    plt.yticks([0, 1])
    plt.savefig(save_dir / "v1_smoothness_right_hypercolumn.png", bbox_inches="tight", dpi=300)

    _, axs = plt.subplots(1, 3, figsize=(4.5, 1.5), constrained_layout=True)
    labels = {"angles": "Change in\npreferred orientation", "sfs": "Change in\nspatial frequency", "colors": "Fraction prefer.\nother color"}
    for i, (metric_name, res) in enumerate(curve_dict.items()):
        curves = np.stack(res["Curves"])
        mn_curve = np.mean(curves, axis=0)
        se = np.std(curves, axis=0)
        axs[i].plot(res["Distances"], mn_curve, color="black")
        axs[i].fill_between(res["Distances"], mn_curve - se, mn_curve + se, alpha=0.3, facecolor="black")
        axs[i].set_ylabel(labels[metric_name], fontsize=7)
        axs[i].set_yticks([0, 1])
        axs[i].set_xlabel("Pairwise Distance", fontsize=7)
        plt.savefig(save_dir / "v1_preference_right_hypercolumn.png", bbox_inches="tight", dpi=300)
    

if __name__ == "__main__":
    """
    Example usage:
    python3 scripts/v1_preference_and_smoothness.py --config configs/analysis_configs/vitb14_dinov2_imagenet_unoptimized.yaml --layer blocks.2
    """
    main()
