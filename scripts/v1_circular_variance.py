import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from spacetorch.configs import get_cfg
from spacetorch.variants import get_variant
from spacetorch.variants.positions import get_positions
from spacetorch.maps import TissueMap
from spacetorch.maps.sine_tissue import get_sine_tissue
from spacetorch.datasets.ringach_2002 import load_ringach_data
from spacetorch.utils import array_utils


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--layer", type=str)
    parser.add_argument("--e", type=int)
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

RESP_THRESH = 1
SEL_THRESH = 0.6


def cv_bin(cv, bin_edges):
    """Bin circular variance into a histogram with specified bin edges"""
    counts, bin_edges = np.histogram(cv, bins=bin_edges)
    counts = counts / np.sum(counts) * 100
    return counts


def get_macaque_data():
    bin_edges = np.linspace(0.2, 1, 10)
    midpoints = array_utils.midpoints_from_bin_edges(bin_edges)

    circular_variance, per_units = [], []
    per_selective = 0.

    ringach_data = load_ringach_data(fieldname="orivar")
    ringach_counts = cv_bin(ringach_data, bin_edges)
    for x, y in zip(midpoints, ringach_counts):
        circular_variance.append(x)
        per_units.append(y)

    per_selective = np.mean(ringach_data < SEL_THRESH) * 100

    data = {
        "layer": "NA",
        "cv": circular_variance,
        "per_units": per_units,
        "per_selective": per_selective,
    }

    return data


def get_circular_variance(tissue: TissueMap, layer: str):
    bin_edges = np.linspace(0.2, 1, 10)
    midpoints = array_utils.midpoints_from_bin_edges(bin_edges)

    circular_variance, per_units = [], []
    per_selective = 0.

    x = tissue.responses.orientation_tuning_curves

    cv = tissue.responses.circular_variance
    mean_responses = tissue.responses._data.mean("image_idx").values
    cv = cv[~np.isnan(cv) & (mean_responses > RESP_THRESH)]

    counts = cv_bin(cv, bin_edges)
    for x, y in zip(midpoints, counts):
        circular_variance.append(x)
        per_units.append(y)

    per_selective = np.mean(cv < SEL_THRESH) * 100
    
    data = {
        "layer": layer,
        "cv": circular_variance,
        "per_units": per_units,
        "per_selective": per_selective,
    }

    return data


def main():
    cfg = get_cfg(args)
    variant = get_variant(cfg.variant.name)
    variant.set_eval_cfg(cfg)
    variant.set_eval_model(cfg)

    model = variant.eval_model
    positions = get_positions(cfg, rescale=False)[args.layer]


    save_dir = Path(cfg.output_dir) / args.layer
    save_dir.mkdir(parents=True, exist_ok=True)

    v1_tissue = get_sine_tissue(
        cfg.name,
        model,
        positions,
        layer=args.layer,
        output_dir=save_dir,
        # skip_cache=True,
    )

    data = get_circular_variance(v1_tissue, args.layer)
    print(data)
    np.savez(save_dir / "circular_variance.npz", **data)

    _, axs = plt.subplots(1, 2, figsize=(3, 1.5), constrained_layout=True)
    axs[0].plot(data["cv"], data["per_units"], color='black')
    axs[0].set_ylabel("% Units")
    axs[0].set_xlabel("CV")
    axs[0].set_yticks([0, 50, 100], [0, 50, 100])
    axs[0].set_xticks([0.4, 0.6, 0.8], [0.4, 0.6, 0.8])
    b = axs[1].bar([""], [data["per_selective"]], color='black')
    for rect in b:
        height = rect.get_height()
        axs[1].text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.3f}', ha='center', va='bottom', fontsize=7)
    axs[1].set_ylabel("% Selective")
    plt.savefig(save_dir / "circular_variance.png", bbox_inches="tight", dpi=300)
    

if __name__ == "__main__":
    """
    Example usage:
    python3 scripts/v1_circular_variance.py --config configs/analysis_configs/vitb14_dinov2_imagenet_unoptimized.yaml --layer blocks.2
    """
    main()
