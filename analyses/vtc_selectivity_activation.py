import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from matplotlib import patches
from spacetorch.configs import get_cfg
from spacetorch.variants import get_variant
from spacetorch.variants.positions import get_positions
from spacetorch.datasets.floc import DOMAIN_CONTRASTS
from spacetorch.maps.floc_tissue import get_floc_tissue


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


def add_scale_bar(
    ax, width, height=None, patch_kwargs=None, flipud=False, y_start=None
):
    """
    Adds a rectangular scale bar in the bottom right corner, just above x-axis
    Inputs
        width (float): width of bar
        height (float): height of bar, defaults to 0.01 * np.ptp(ylims)
        patch_kwargs (dict): other inputs
        flipud (bool): set to True for imshow
    """

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    if patch_kwargs is None:
        patch_kwargs = {"facecolor": "#222222"}

    if height is None:
        height = 0.025 * np.ptp(ylim)

    x_offset = np.ptp(xlim) * 0.04
    start_point = xlim[1] - (width + x_offset)
    y_start = y_start or ylim[0]
    if flipud:
        y_start += height
    xy = (start_point, y_start)
    rect = patches.Rectangle(xy, width, height, clip_on=False, **patch_kwargs)
    ax.add_patch(rect)


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

    if (save_dir / "vtc_selectivity_activation.png").exists():
        print(f"Found existing results in {save_dir}, skipping...")
        return

    floc_tissue = get_floc_tissue(
        cfg.name,
        model,
        positions,
        layer=args.layer,
        output_dir=save_dir,
        is_swinv2=is_swinv2
    )

    fig, axs = plt.subplots(figsize=(8, 2.2), nrows=2, ncols=5, gridspec_kw={"wspace": 0.5})

    most_sel_indices = {}
    for ax, contrast_name in zip(axs[1], contrast_order):
        contrast = contrast_dict[contrast_name]
        sel = floc_tissue.responses.selectivity(
            on_categories=contrast.on_categories, off_categories=contrast.off_categories
        )
        unit = np.argsort(sel)[-5]
        unit_resp = floc_tissue.responses._data[:, unit]

        resp_df = {"Category": [], "Response": []}
        for con in contrasts:
            on_indices = floc_tissue.responses._data.categories.isin(con.on_categories)
            responses = floc_tissue.responses._data[on_indices, unit]
            responses = np.random.choice(responses, size=(50,), replace=False)

            for val in responses:
                resp_df["Category"].append(con.name)
                resp_df["Response"].append(float(val))
        
        sns.barplot(
            data=resp_df,
            x="Category",
            y="Response",
            palette=contrast_colors,
            ax=ax,
            errorbar=None,
            order=contrast_order,
            alpha=0.8,
        )
        mx = max(resp_df["Response"])

        if contrast_name == contrast_order[0]:
            ax.set_ylabel("Activation\n(a.u.)")
        else:
            ax.set_ylabel("")
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_xticklabels(["F", "B", "C", "P", "O"])
        ax.set_xlabel("")
        ax.legend().remove()
        sns.despine(fig, ax)
        most_sel_indices[contrast.name] = unit

    for contrast_name, ax in zip(contrast_order, axs[0]):
        contrast = contrast_dict[contrast_name]
        handle = floc_tissue.make_single_contrast_map(
            ax, contrast, final_psm=1e-4, rasterized=True, vmin=-5, vmax=5, linewidths=0
        )
        most_sel = most_sel_indices[contrast_name]
        ax.scatter(
            floc_tissue.positions[most_sel, 0],
            floc_tissue.positions[most_sel, 1],
            s=30,
            marker="*",
            linewidths=0,
            c="black",
            rasterized=True,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim([0, 70])
        ax.set_xlim([0, 70])
        ax.set_axis_off()
        add_scale_bar(ax, 10)

    # add a colorbar to the last axis
    cax = fig.add_axes([0.92, 0.6, 0.01, 0.2])
    cb = plt.colorbar(handle, cax=cax, ticks=[-5, 0, 5])
    cax.set_xlabel(r"$t$-value", x=2)
        
    fig.savefig(save_dir / f"vtc_selectivity_activation.png", dpi=300, bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    """
    Example usage:
    python3 visualize/vtc_selectivity_activation.py --config configs/analysis_configs/vitb14_dinov2_imagenet_unoptimized.yaml --layer blocks.10
    """
    main()