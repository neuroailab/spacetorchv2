import argparse
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from spacetorch.configs import get_cfg
from spacetorch.variants import get_variant
from spacetorch.maps import TissueMap
from spacetorch.maps.sine_tissue import get_sine_tissue
from spacetorch.colormaps import nauhaus_colormaps
from spacetorch.utils import array_utils, plot_utils


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
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


def cv_bin(cv, bin_edges):
    """Bin circular variance into a histogram with specified bin edges"""
    counts, bin_edges = np.histogram(cv, bins=bin_edges)
    counts = counts / np.sum(counts) * 100
    return counts


def card_ind(x):
    """define cardinality index as fraction of preferred orientations on the cardinals"""
    cardinality_index = (x[0] + x[2] + x[4]) / (x.sum())
    return cardinality_index


def plot_tuning_curves(tissue: TissueMap, save_dir: Path):
    _, axes = plt.subplots(1, 4, figsize=(6, 1))

    cvs = tissue.responses.circular_variance
    sort_ind = np.argsort(cvs)
    if len(sort_ind) > 200_000:
        indices = (100, 70_000, 300_000, 760_000)
    else:
        indices = (100, 1_000, 20_000, 50_000)

    for ax_col, idx in zip(axes, indices):
        tc_ind = sort_ind[idx]
        tc = tissue.orientation_tc_fits[tc_ind].fit
        plot_utils.plot_tuning_curve(
            ax_col, tc, mode="line", plot_params={"lw": 1, "c": "k"}
        )
        mx = np.max(tc)
        ax_col.set_ylim([-mx * 0.1, mx * 1.2])
        ax_col.set_title(f"CV = {cvs[tc_ind]:.2f}", fontdict={"size": 10})
        ax_col.set_yticks([])
        ax_col.set_xticks([])

    axes[0].set_xticks([0, 25, 50])
    axes[0].set_xticklabels([0, 90, 180])
    for ax in axes[1:]:
        ax.set_xticks([])
        ax.set_xticklabels([])

    plt.subplots_adjust(wspace=0.3)
    plt.savefig(save_dir / "tc.png", dpi=300, bbox_inches="tight", transparent=True)


def plot_brainscore(save_dir: Path):
    df = pd.read_csv(save_dir / "brainscore_choices.csv", sep=",")
    
    fig = plt.figure(figsize=(2.5, 2))
    colors = sns.color_palette("magma_r", n_colors=10)
    print([to_hex(c) for c in colors])

    sns.barplot(x="model_identifier", y="raw", data=df, palette=colors)
    ceiling = df["ceiling"].iloc[0]
    plt.axhline(ceiling, color="red", linestyle="--", label="Ceiling")

    plt.yticks([0, 0.5, 1])
    plt.ylabel("Variance Explained")
    plt.xticks([])
    plt.xlabel("")

    sns.despine()
    fig.savefig(save_dir / "v1_pred.png", dpi=300, bbox_inches="tight", transparent=True)


def plot_mean_responses(tissue: TissueMap, save_dir: Path):
    otc = np.array(tissue.responses.orientation_tuning_curves.T)
    otc = np.reshape(otc, (8, 256, -1))
    otc = np.swapaxes(otc, 1, 2)
    otc = np.reshape(otc, (8, -1))
    colors = [nauhaus_colormaps["angles"](o / 180) for o in [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]]
    _, axs = plt.subplots(1, 1, figsize=(10, 3))
    for i, _o in enumerate(otc):
        axs.plot(_o[10_000:12_000], color=colors[i])
    axs.set_xticks([])
    axs.set_xlabel("Units")
    axs.set_ylabel("Mean response")
    sns.despine(ax=axs)
    plt.savefig(save_dir / "mean_responses.png", dpi=300, bbox_inches="tight", transparent=True)


def get_cv(tissue: TissueMap):
    bin_edges = np.linspace(0.2, 1, 10)
    midpoints = array_utils.midpoints_from_bin_edges(bin_edges)

    circular_variance, per_units = [], []
    per_selective = 0.

    x = tissue.responses.orientation_tuning_curves

    cv = tissue.responses.circular_variance
    mean_responses = tissue.responses._data.mean("image_idx").values
    cv = cv[~np.isnan(cv) & (mean_responses > 1)]

    counts = cv_bin(cv, bin_edges)
    for x, y in zip(midpoints, counts):
        circular_variance.append(x)
        per_units.append(y / 100)

    per_selective = np.mean(cv < 0.6)

    data = {
        "cv": circular_variance,
        "per_units": per_units,
        "per_selective": per_selective,
    }

    return data


def get_preferred_orientations(tissue: TissueMap):
    xs = [0, 45, 90, 135, 180]
    orientation, per_units = [], []

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
    normed_counts = counts / counts.sum()

    for x, y in zip(xs, normed_counts):
        orientation.append(x)
        per_units.append(y)
    
    data = {
        "orientation": orientation,
        "per_units": per_units,
        "card_ind": card_ind(counts),
    }

    return data


def main():
    cfg = get_cfg(args)
    # variant = get_variant(cfg.variant.name)
    # variant.set_eval_cfg(cfg)
    # variant.set_eval_model(cfg)

    # model = variant.eval_model

    # layers = [
    #     ("blocks.2.norm1", "A"),
    #     ("blocks.2.attn.q", "B"),
    #     ("blocks.2.attn.k", "C"),
    #     ("blocks.2.attn.v", "D"),
    #     ("blocks.2.attn", "E"),
    #     ("blocks.2.norm2", "F"),
    #     ("blocks.2.mlp.fc1", "G"),
    #     ("blocks.2.mlp.act", "H"),
    #     ("blocks.2.mlp.fc2", "I"),
    #     ("blocks.2", "J"),
    # ]
    # labels = [label for _, label in layers]

    plot_brainscore(Path(cfg.output_dir))

    # fig1, axs1 = plt.subplots(1, 2, figsize=(7, 2))
    # fig2, axs2 = plt.subplots(1, 2, figsize=(7, 2))

    # colors = sns.color_palette("magma_r", n_colors=10)
    # per_selective_all, card_ind_all = [], []

    # for i, (layer, label) in enumerate(layers):
    #     save_dir = Path(cfg.output_dir) / layer
    #     save_dir.mkdir(parents=True, exist_ok=True)

    #     tissue = get_sine_tissue(
    #         cfg.name,
    #         model,
    #         argparse.Namespace(**{"coordinates": []}),
    #         layer=layer,
    #         output_dir=save_dir,
    #     )

        # plot_mean_responses(tissue, save_dir)
        # plot_tuning_curves(tissue, save_dir)

        # data1 = get_cv(tissue)
        # sns.lineplot(x=data1["cv"], y=data1["per_units"], color=colors[i], label=label, ax=axs1[0], zorder=(10-i), marker="o", legend=False)
        # per_selective_all.append(data1["per_selective"])

        # data2 = get_preferred_orientations(tissue)
        # sns.lineplot(x=data2["orientation"], y=data2["per_units"], color=colors[i], label=label, ax=axs2[0], zorder=(10-i), marker="o", legend=False)
        # card_ind_all.append(data2["card_ind"])

    # axs1[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    # axs1[0].set_ylim([-0.03, 0.9])
    # axs1[0].set_ylabel("Fraction of Units")
    # axs1[0].set_xlabel("Circular Variance")
    # sns.barplot(x=labels, y=per_selective_all, palette=colors, ax=axs1[1])
    # axs1[1].set_ylabel("Fraction of Orientation\nSelective Units")
    # axs1[1].set_ylim([-0.03, 0.55])
    # axs1[1].set_yticks([0, 0.2, 0.4])
    # axs1[1].set_xticks([])

    # sns.despine(fig=fig1)
    # fig1.subplots_adjust(wspace=0.5)
    # fig1.savefig(Path(cfg.output_dir) / "cv_choices.png", dpi=300, bbox_inches="tight", transparent=True)


    # axs2[0].set_yticks([0, 0.2, 0.4])
    # axs2[0].set_ylim([-0.03, 0.5])
    # axs2[0].set_ylabel("Fraction of Units")
    # axs2[0].set_xlabel("Preferred Orientations")
    # axs2[0].set_xticks([0, 45, 90, 135, 180])
    # sns.barplot(x=labels, y=card_ind_all, palette=colors, ax=axs2[1])
    # axs2[1].set_ylabel("Cardinality Index")
    # axs2[1].set_ylim([-0.03, 1.1])
    # axs2[1].set_yticks([0, 0.5, 1])
    # axs2[1].set_xticks([])

    # sns.despine(fig=fig2)
    # fig2.subplots_adjust(wspace=0.5)
    # fig2.savefig(Path(cfg.output_dir) / "pref_orientation_choices.png", dpi=300, bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    main()
