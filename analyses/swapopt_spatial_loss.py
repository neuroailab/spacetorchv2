import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss_dir", type=str)
    parser.add_argument("--layers", nargs="+")
    return parser


args = get_parser().parse_args()
L = len(args.layers)


def main():
    # extract spatial losses for all layers
    plot_data = {"iteration": [], "loss": [], "layer": []}

    for layer in args.layers:
        loss_path = Path(args.loss_dir) / (layer + "_losses.npy")
        loss = np.load(loss_path)
        
        # Running average
        window_size = 40
        running_avg = np.convolve(loss, np.ones(window_size) / window_size, mode='valid')
        
        plot_data["iteration"].extend(range(window_size - 1, len(loss)))
        plot_data["loss"].extend(running_avg)
        plot_data["layer"].extend([layer] * len(running_avg))

    df = pd.DataFrame(plot_data)

    # Plot using Seaborn
    plt.figure(figsize=(3, 2))
    palette = sns.color_palette("magma_r", n_colors=L)

    g = sns.lineplot(x="iteration", y="loss", data=df, hue="layer", palette=palette, alpha=0.75)

    sns.move_legend(g, "upper left", bbox_to_anchor=(1.05, 1), title="", frameon=False, fontsize=7)

    plt.xticks([])
    plt.yticks([0.2, 0.3, 0.4, 0.5], [0.2, 0.3, 0.4, 0.5])
    plt.ylabel("Spatial Loss")
    plt.xlabel("Iterations")
    sns.despine()

    save_dir = Path(args.loss_dir) / "figures"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(save_dir / "swapopt_spatial_losses.png", bbox_inches="tight", dpi=300, transparent=True)


if __name__ == "__main__":
    main()