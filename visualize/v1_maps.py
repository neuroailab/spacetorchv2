import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import patches

from spacetorch.configs import get_cfg
from spacetorch.variants import get_variant
from spacetorch.variants.positions import get_positions
from spacetorch.maps.sine_tissue import get_sine_tissue, add_sine_colorbar, METRIC_DICT


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

    fig, axs = plt.subplots(ncols=1, nrows=3, figsize=(1, 3))

    # with open(save_dir / "positions.txt", "w") as f:
    #     for x, y in v1_tissue.positions:
    #         f.write(f"{x},{y}\n")

    v1_tissue.reset_unit_mask()
    # zoom_lims = [[50, 60], [50, 60]]
    # v1_tissue.set_mask_by_pct_limits(zoom_lims)

    for (_, metric), ax in zip(METRIC_DICT.items(), axs):
        _ = v1_tissue.make_parameter_map(
            ax, metric=metric, linewidths=0, edgecolor=(0, 0, 0, 0.5), rasterized=True, final_s=0.4
        )

        ax.axis("off")

        cbar = add_sine_colorbar(fig, ax, metric, label=metric.xlabel)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label(label="", fontsize=10)

    # add_scale_bar(axs[-1], width=2)
    
    plt.savefig(save_dir / f"v1_maps_full.png", dpi=300, bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    """
    Example usage:
    python3 visualize/v1_maps.py --config configs/analysis_configs/vitb14_dinov2_imagenet_unoptimized.yaml --layer blocks.1
    """
    main()