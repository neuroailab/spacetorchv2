import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from spacetorch.configs import get_cfg
from spacetorch.variants import get_variant
from spacetorch.variants.positions import get_positions
from spacetorch.datasets.floc import DOMAIN_CONTRASTS
from spacetorch.maps.floc_tissue import get_floc_tissue
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
contrasts = DOMAIN_CONTRASTS
contrast_dict = {c.name: c for c in contrasts}
contrast_order = ["Faces", "Bodies", "Characters", "Places", "Objects"]
ordered_contrasts = [contrast_dict[curr] for curr in contrast_order]
contrast_colors = {c.name: c.color for c in contrasts}

# randomly sample sets of units 25 times
num_samples = 25

# the distance curve will have N_BINS - 1 entries
N_BINS = 10

# for the human data, this is the farthest distance we'll consider
distance_cutoff = 60.0  # mm

human_bin_edges = np.linspace(0, distance_cutoff, N_BINS)
human_midpoints = array_utils.midpoints_from_bin_edges(human_bin_edges)


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

    # if (save_dir / "vtc_smoothness.npz").exists():
    #     print(f"Found existing results in {save_dir}, skipping...")
    #     return

    floc_tissue = get_floc_tissue(
        cfg.name,
        model,
        positions,
        layer=args.layer,
        output_dir=save_dir,
        is_swinv2=is_swinv2
    )

    smoothness_results = {"Smoothness": [], "Contrast": []}

    bin_edges = human_bin_edges

    curve_dict = {}
    for contrast in tqdm(contrasts):
        common = {
            "num_samples": num_samples,
            "bin_edges": bin_edges,
            "contrast": contrast,
            "distance_cutoff": distance_cutoff,
        }
        all_curves = []
        try:
            distances, curves = floc_tissue.category_smoothness(**common)
            _, chance_curves = floc_tissue.category_smoothness(**common, shuffle=True)
        except ValueError:
            continue

        # compute the value expected by chance
        chance_mean = np.nanmean(np.concatenate(chance_curves))

        smoo = [spatial_utils.smoothness(curve) for curve in curves]
        smoothness_results["Smoothness"].append(np.nanmean(smoo))
        smoothness_results["Contrast"].append(contrast.name)

        normalized_curves = [curve / chance_mean for curve in curves]
        all_curves.extend(normalized_curves)

        curve_dict[contrast.name] = {
            "Distances": distances,
            "Curves": all_curves,
        }

    np.savez(save_dir / "vtc_smoothness.npz", **smoothness_results)
    np.savez(save_dir / "vtc_delta_selectivity.npz", **curve_dict)

    fig, curves_row = plt.subplots(figsize=(4, 1), ncols=5, nrows=1, sharey=True)

    for ax, contrast_name in zip(curves_row, contrast_order):
        res = curve_dict[contrast_name]
        curves = np.stack(res["Curves"])
        mn_curve = np.nanmean(curves, axis=0)
        se = np.nanstd(curves, axis=0)

        dist = res["Distances"][: len(mn_curve)]
        line_handle = ax.plot(
            dist,
            mn_curve,
            color="black",
            mec="k",
            markevery=2,
            markersize=3,
            mew=0.5,
            lw=1,
            alpha=0.8,
        )

        ax.fill_between(
            dist,
            mn_curve - se,
            mn_curve + se,
            alpha=0.3,
            facecolor=line_handle[0].get_color(),
        )

        ax.legend().remove()
        sns.despine(fig, ax)

        if contrast_name == "Faces":
            ax.set_ylabel((r"$\Delta$ Selectivity" "\n" "(vs. Chance)"))
            ax.set_yticks([0, 1.0])
            ax.set_ylim([0, 1.5])
        else:
            ax.set_ylabel("")
            ax.set_yticks([])
        
    fig.savefig(save_dir / f"vtc_delta_selectivity.png", dpi=300, bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    """
    Example usage:
    python3 visualize/vtc_selectivity_smoothness.py --config configs/analysis_configs/vitb14_dinov2_imagenet_unoptimized.yaml --layer blocks.10
    """
    main()