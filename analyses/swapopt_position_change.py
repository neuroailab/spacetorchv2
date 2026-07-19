import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from spacetorch.constants import V1_SIZE, V2_SIZE, V4_SIZE, VTC_SIZE
from spacetorch.variants.positions import BRAIN_MAPPING, LayerPositions


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions_dir", type=str)
    parser.add_argument("--swapopt_positions_dir", type=str)
    parser.add_argument("--architecture", type=str)
    parser.add_argument("--layers", nargs="+")
    return parser


args = get_parser().parse_args()
L = len(args.layers)


def get_normalization_constant(layer):
    '''
    Get rescaling factor for the simulated cortical sheet for a model layer based on the region in the brain it is assigned to
    '''
    region = BRAIN_MAPPING[args.architecture][layer]
    if region == "V1":
        return V1_SIZE / 1.5
    elif region == "V2":
        return V2_SIZE / 1.75
    elif region == "V4":
        return V4_SIZE / 1.4
    elif region == "VTC":
        return VTC_SIZE / 7
    else:
        return 1
    

def get_pos(f):
    def _scalar(x):
        return x[()]
    
    state = np.load(f)

    return LayerPositions(
        name=_scalar(state["name"]),
        dims=state["dims"],
        coordinates=state["coordinates"],
        neighborhood_indices=state["neighborhood_indices"],
        neighborhood_width=_scalar(state["neighborhood_width"]),
    )


def plot_distances(df):
    plt.figure(figsize=(3, 2))
    palette = sns.color_palette("magma_r", n_colors=L)

    flierprops = dict(markerfacecolor="pink", markersize=3, linestyle="none", markeredgewidth=0, zorder=1, alpha=0.5)
    sns.boxplot(x="layer", y="distance", data=df, hue="layer", palette=palette, width=0.5, linewidth=0.75, legend=False, flierprops=flierprops, zorder=2)

    plt.xticks([0, L-1], [args.layers[0], args.layers[-1]], rotation=30)
    plt.yticks([0, 1.4142135624], [0, "V2"])  # from 0 to sqrt(2), the length of the diagonal of the rescaled simulated cortical sheet
    plt.ylabel("$\ell_2$ distance")
    plt.xlabel("")
    sns.despine()

    save_dir = Path(args.swapopt_positions_dir) / "figures"
    save_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_dir / "swapped_unit_distances.png", bbox_inches="tight", dpi=300, transparent=True)


def main():
    # compute Euclidean distances between normalized pre- and post- swapopt unit positions
    l2_distances = []
    layers = []
    for layer in args.layers:
        normalization_constant = get_normalization_constant(layer)

        # pre-swapopt positions
        old_positions_path = Path(args.positions_dir) / (layer + ".npz")
        with open(old_positions_path, 'rb') as f:
            old_positions = get_pos(f).coordinates / normalization_constant
        
        # post-swapopt positions
        new_positions_path = Path(args.swapopt_positions_dir) / (layer + ".npz")
        with open(new_positions_path, 'rb') as f:
            new_positions = get_pos(f).coordinates / normalization_constant
        
        # Euclidean distance
        l2_distance = np.linalg.norm(new_positions - old_positions, ord=2, axis=1)
        l2_distances.append(l2_distance)
        layers.append([layer for _ in range(len(l2_distance))])

    # convert to a Pandas dataframe to visualize using Seaborn
    df = pd.DataFrame({"layer": np.concatenate(layers), "distance": np.concatenate(l2_distances)})

    # plot
    plot_distances(df)


if __name__ == "__main__":
    main()