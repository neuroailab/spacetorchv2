import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from spacetorch.visualize.common import ALL_MODELS


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preference", action='store_true')
    parser.add_argument("--smoothness", action='store_true')
    parser.add_argument("--circular_variance", action='store_true')
    parser.add_argument("--preferred_orientations", action='store_true')
    parser.add_argument("--pinwheel_density", action='store_true')
    return parser


args = get_parser().parse_args()

LOAD_DIR = Path("checkpoints")
SAVE_DIR = Path("checkpoints") / "figures"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def plot_preference():
    _, axs = plt.subplots(nrows=3, ncols=1, figsize=(2, 4.5))

    labels = {
        "angles": r"$\Delta$" " Preferred" "\n" "Orientation",
        "sfs": r"$\Delta$" " Spatial" "\n" "Frequency",
        "colors": "Fraction Pref." "\n" "Other Color"
    }

    for model, color in ALL_MODELS:
        print(model)
        data = np.load(LOAD_DIR / model / "figures" / "v1_preference.npz", allow_pickle=True)

        for i, (metric_name, res) in enumerate(data.items()):
            res = res.item()
            curves = np.stack(res["Curves"])
            mn_curve = np.mean(curves, axis=0)
            se = np.std(curves, axis=0)
            axs[i].plot(res["Distances"], mn_curve, color=color, zorder=3 if model == "macaque" else 2 if model == "tdann" else 1)
            axs[i].fill_between(res["Distances"], mn_curve - se, mn_curve + se, alpha=0.3, facecolor=color)
            axs[i].set_ylabel(labels[metric_name])
            axs[i].set_yticks([0, 1])
        
    axs[-1].set_xlabel("Pairwise Distance" "\n" r"$\%$ Column Spacing")
    sns.despine()
    
    plt.savefig(SAVE_DIR / "v1_preference.png", bbox_inches="tight", dpi=300, transparent=True)


def plot_smoothness():
    _, axs = plt.subplots(nrows=3, ncols=1, figsize=(2, 4.5))

    smoothness = {
        "angles": [],
        "sfs": [],
        "colors": []
    }

    model_names = []
    colors = []

    for model, color in ALL_MODELS:
        print(model)
        data = np.load(LOAD_DIR / model / "figures" / "v1_smoothness.npy", allow_pickle=True)
        smoothness["angles"].append(data.item()["Smoothness"][0])
        smoothness["sfs"].append(data.item()["Smoothness"][1])
        smoothness["colors"].append(data.item()["Smoothness"][2])
        model_names.append(model)
        colors.append(color)

    for i, (_, res) in enumerate(smoothness.items()):
        axs[i].bar(model_names, res, color=colors)
        axs[i].set_ylabel("Smoothness")
        axs[i].set_yticks([0, 1])
        axs[i].set_xlabel("")
        axs[i].set_xticks([])

    sns.despine()
    
    plt.savefig(SAVE_DIR / "v1_smoothness.png", bbox_inches="tight", dpi=300, transparent=True)


def plot_circular_variance():
    _, axs = plt.subplots(nrows=1, ncols=2, figsize=(5, 2), constrained_layout=True)

    per_selective = []
    model_names = []
    colors = []

    for model, color in ALL_MODELS:
        print(model)
        data = np.load(LOAD_DIR / model / "figures" / "circular_variance.npz", allow_pickle=True)

        axs[0].plot(data["cv"], data["per_units"], color=color, zorder=3 if model == "macaque" else 2 if model == "tdann" else 1)
        
        per_selective.append(data["per_selective"] if model != "macaque" else data["per_selective"] * 100)
        model_names.append(model)
        colors.append(color)
        
    axs[1].bar(model_names, per_selective, color=colors)

    axs[0].set_ylabel("% Units")
    axs[0].set_xlabel("Circular Variance")
    axs[0].set_yticks([0, 50, 100], [0, 50, 100])
    axs[0].set_xticks([0.4, 0.6, 0.8], [0.4, 0.6, 0.8])
    axs[1].set_ylabel("% Selective")
    axs[1].set_yticks([0, 20, 40], [0, 20, 40])
    axs[1].set_xticks([])

    sns.despine()
    
    plt.savefig(SAVE_DIR / "v1_circular_variance.png", bbox_inches="tight", dpi=300, transparent=True)


def plot_preferred_orientations():
    _, axs = plt.subplots(nrows=1, ncols=2, figsize=(5, 2), constrained_layout=True)

    card_ind = []
    model_names = []
    colors = []

    for model, color in ALL_MODELS:
        print(model)
        data = np.load(LOAD_DIR / model / "figures" / "preferred_orientations.npz", allow_pickle=True)

        axs[0].plot(data["orientation"], data["per_units"], color=color, zorder=3 if model == "macaque" else 2 if model == "tdann" else 1)
        
        card_ind.append(data["card_ind"])
        model_names.append(model)
        colors.append(color)
        
    axs[1].bar(model_names, card_ind, color=colors)

    axs[0].set_ylabel("% Units")
    axs[0].set_xlabel("Preferred Orientations")
    axs[0].set_yticks([0, 50, 100])
    axs[0].set_xticks([0, 50, 100, 150])
    axs[1].set_ylabel("Cardinality Index")
    axs[1].set_yticks([0, 0.5, 1])
    axs[1].set_xticks([])

    sns.despine()
    
    plt.savefig(SAVE_DIR / "v1_preferred_orientations.png", bbox_inches="tight", dpi=300, transparent=True)


def plot_pinwheel_density():
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(2, 2))

    density = []
    model_names = []
    colors = []

    for model, color in ALL_MODELS:
        print(model)

        if model == "macaque":
            data = np.pi
        else:
            data = np.load(LOAD_DIR / model / "figures" / "pinwheel_density.npy", allow_pickle=True)

        density.append(data)
        model_names.append(model)
        colors.append(color)
        
    ax.bar(model_names, density, color=colors)

    ax.set_ylabel(("Pinwheels /" "\n" r"Column Spacing$^2$"))
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_xlabel("")
    ax.set_xticks([])

    sns.despine()
    
    plt.savefig(SAVE_DIR / "v1_pinwheel_density.png", bbox_inches="tight", dpi=300, transparent=True)


def main():
    if args.preference:
        plot_preference()
    if args.smoothness:
        plot_smoothness()
    if args.circular_variance:
        plot_circular_variance()
    if args.preferred_orientations:
        plot_preferred_orientations()
    if args.pinwheel_density:
        plot_pinwheel_density()


if __name__ == "__main__":
    """
    Example usage:
    python3 visualize/v1_quant.py [--preference] [--smoothness] [--circular_variance] [--preferred_orientations] [--pinwheel_density]
    """
    main()
