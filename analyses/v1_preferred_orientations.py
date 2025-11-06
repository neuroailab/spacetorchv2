import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from spacetorch.configs import get_cfg
from spacetorch.variants import get_variant
from spacetorch.variants.positions import get_positions
from spacetorch.maps import TissueMap
from spacetorch.maps.sine_tissue import get_sine_tissue


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

xs = [0, 45, 90, 135, 180]


def card_ind(x):
    """define cardinality index as fraction of preferred orientations on the cardinals"""
    cardinality_index = (x[0] + x[2] + x[4]) / (x.sum())
    return cardinality_index


def get_macaque_data():
    # these values are ripped from Figure 5 of De Valois, Yund, Hepler '82 then passed through web plot digitizer
    foveal_dvd = np.array(
        [
            [-0.004290201486248257, 66.09515054010572],
            [1.0036007048188158, 49.24691641768176],
            [1.996537194514671, 76.96284379069945],
            [2.9885543553206158, 54.085037922316715],
            [3.9957097985137513, 66.09515054010573],
        ]
    )[:, 1]
    normed_dvd = foveal_dvd / foveal_dvd.sum() * 100

    orientation, per_units = [], []

    for x, y in zip(xs, normed_dvd):
        orientation.append(x)
        per_units.append(y)

    data = {
        "layer": "NA",
        "orientation": orientation,
        "per_units": per_units,
        "card_ind": card_ind(foveal_dvd),
    }

    return data


def get_preferred_orientations(tissue: TissueMap, layer: str):
    orientation, per_units = [], []

    tissue.reset_unit_mask()
    da = tissue.responses._data
    vals = da.groupby("angles").mean().argmax(axis=0).values
    counts = np.array(
        [
            np.sum(vals == 4),  # horizontal
            np.sum(vals == 2),  # +45 deg
            np.sum(vals == 0),  # vertical
            np.sum(vals == 6),  # -45 deg
            np.sum(vals == 4),  # horizontal
        ]
    )
    normed_counts = counts / counts.sum() * 100

    for x, y in zip(xs, normed_counts):
        orientation.append(x)
        per_units.append(y)
    
    data = {
        "layer": "NA",
        "orientation": orientation,
        "per_units": per_units,
        "card_ind": card_ind(counts),
    }

    return data


def main():
    cfg = get_cfg(args)
    variant = get_variant(cfg.variant.name)
    variant.set_eval_cfg(cfg)
    variant.set_eval_model(cfg)

    model = variant.eval_model
    is_tdann = "tdann" in cfg.name
    positions = get_positions(cfg, rescale=is_tdann)[args.layer]

    is_swinv2 = ("swinv2" in cfg.name)

    save_dir = Path(cfg.output_dir) / args.layer
    save_dir.mkdir(parents=True, exist_ok=True)

    if (save_dir / "preferred_orientations.npz").exists():
        print(f"Found existing results in {save_dir}, skipping...")
        return

    v1_tissue = get_sine_tissue(
        cfg.name,
        model,
        positions,
        layer=args.layer,
        output_dir=save_dir,
        smooth_orientation_tuning_curves=False,
        is_swinv2=is_swinv2
        # skip_cache=True
    )

    data = get_preferred_orientations(v1_tissue, args.layer)
    # data = get_macaque_data()
    print(data)
    np.savez(save_dir / "preferred_orientations.npz", **data)

    _, axs = plt.subplots(1, 2, figsize=(3, 1.5), constrained_layout=True)
    axs[0].plot(data["orientation"], data["per_units"], color='black')
    axs[0].set_ylabel("% Units")
    axs[0].set_xlabel("Preferred Orientations")
    axs[0].set_xticks([0, 50, 100, 150])
    b = axs[1].bar([""], [data["card_ind"]], color='black')
    for rect in b:
        height = rect.get_height()
        axs[1].text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.3f}', ha='center', va='bottom', fontsize=7)
    axs[1].set_ylabel("Cardinality Index")
    axs[1].set_yticks([0, 0.5, 1.0])
    plt.savefig(save_dir / f"preferred_orientations.png", bbox_inches="tight", dpi=300)
    

if __name__ == "__main__":
    """
    Example usage:
    python3 scripts/v1_preferred_orientations.py --config configs/analysis_configs/vitb14_dinov2_imagenet_unoptimized.yaml --layer blocks.2
    """
    main()
