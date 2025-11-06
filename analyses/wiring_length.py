import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path

from spacetorch.configs import get_cfg
from spacetorch.variants import get_variant
from spacetorch.variants.positions import get_positions
from spacetorch.wiring_length import Shifts, WireLengthExperiment


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--src_layer", type=str)
    parser.add_argument("--target_layer", type=str)
    parser.add_argument("--is_v1", action='store_true')
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

# constants
NUM_PATTERNS = 50

# for V1-like layers
V1__NB_WIDTH = 3.5
V1__KMEANS_THRESH = 0.9
V1__NUM_NBS = 20
V1__PCTILE = 95

# for VTC-like layers
VTC__KMEANS_THRESH = 10.0
VTC__PCTILE = 99


def process_model_v1(model, positions, src_layer, target_layer, is_swinv2=False):
    results = {"Wiring Length": [], "Pattern": [], "Window": [], "Shift Direction": []}

    wle = WireLengthExperiment(
        model=model,
        layer_positions=positions,
        source_layer=src_layer,
        target_layer=target_layer,
        num_patterns=NUM_PATTERNS,
        is_swinv2=is_swinv2
    )

    # run for a number of randomly-selected neighborhoods
    extent = 24.5 * 1.5
    for window in tqdm(range(V1__NUM_NBS), desc="V1 | Neighborhood"):
        start_x = np.random.uniform(low=0, high=extent - V1__NB_WIDTH)
        start_y = np.random.uniform(low=0, high=extent - V1__NB_WIDTH)
        lims_x = [start_x, start_x + V1__NB_WIDTH]
        lims_y = [start_y, start_y + V1__NB_WIDTH]

        for pattern in range(NUM_PATTERNS):
            for direction in Shifts:
                wl = wle.compute_wl(
                    pattern,
                    kmeans_dist_thresh=V1__KMEANS_THRESH,
                    active_pctile=V1__PCTILE,
                    lims=[lims_x, lims_y],
                    direction=direction,
                )

                results["Wiring Length"].append(wl)
                results["Pattern"].append(pattern)
                results["Window"].append(window)
                results["Shift Direction"].append(str(direction.value))

    return pd.DataFrame(results)


def process_model_vtc(model, positions, src_layer, target_layer, is_swinv2=False):
    results = {"Wiring Length": [], "Pattern": [], "Window": [], "Shift Direction": []}

    wle = WireLengthExperiment(
        model=model,
        layer_positions=positions,
        source_layer=src_layer,
        target_layer=target_layer,
        num_patterns=NUM_PATTERNS,
        is_swinv2=is_swinv2
    )

    for pattern in tqdm(range(NUM_PATTERNS), desc="VTC | Pattern"):
        for direction in Shifts:
            wl = wle.compute_wl(
                pattern,
                kmeans_dist_thresh=VTC__KMEANS_THRESH,
                active_pctile=VTC__PCTILE,
                lims=None,
                direction=direction,
            )

            results["Wiring Length"].append(wl)
            results["Pattern"].append(pattern)
            results["Window"].append(0)
            results["Shift Direction"].append(str(direction.value))

    return pd.DataFrame(results)


def main():
    cfg = get_cfg(args)
    variant = get_variant(cfg.variant.name)
    variant.set_eval_cfg(cfg)
    variant.set_eval_model(cfg)
    model = variant.eval_model

    is_tdann = "tdann" in cfg.name
    positions = get_positions(cfg, rescale=is_tdann)

    save_dir = Path(cfg.output_dir) / args.src_layer
    save_dir.mkdir(parents=True, exist_ok=True)

    is_swinv2 = ("swinv2" in cfg.name)

    if (save_dir / "wiring_length.csv").exists():
        print(f"Found existing results in {save_dir}, skipping...")
        return

    if args.is_v1:
        df = process_model_v1(model, positions, args.src_layer, args.target_layer, is_swinv2=is_swinv2)
    else:
        df = process_model_vtc(model, positions, args.src_layer, args.target_layer, is_swinv2=is_swinv2)

    df.to_csv(save_dir / "wiring_length.csv")

    fig, ax = plt.subplots(figsize=(2, 2), ncols=1)
    sns.barplot(
        data=df,
        y="Wiring Length",
        color="k",
        ax=ax
    )
    ax.set_xticks([])
    ax.set_ylabel("Wiring Length (mm)")
    sns.despine(fig, ax)
    fig.savefig(save_dir / "wiring_length.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
