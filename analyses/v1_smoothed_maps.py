import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import patches

from spacetorch.configs import get_cfg
from spacetorch.variants import get_variant
from spacetorch.variants.positions import get_positions
from spacetorch.maps.sine_tissue import get_sine_tissue, METRIC_DICT, get_smoothed_map


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
    is_tdann = "tdann" in cfg.name
    positions = get_positions(cfg, rescale=is_tdann)[args.layer]

    save_dir = Path(cfg.output_dir) / args.layer
    save_dir.mkdir(parents=True, exist_ok=True)

    v1_tissue = get_sine_tissue(
        cfg.name,
        model,
        positions,
        layer=args.layer,
        output_dir=save_dir,
        smooth_orientation_tuning_curves=False,
        # skip_cache=True,
    )

    lims = [5, 95]
    v1_tissue.set_mask_by_pct_limits([lims, lims])

    for name, metric in METRIC_DICT.items():
        smoothed = get_smoothed_map(
            v1_tissue, metric, final_width=1.5, final_stride=0.15, verbose=True
        )

        _, ax = plt.subplots(1, 1, figsize=(1, 1))

        ax.imshow(
            smoothed, cmap=metric.colormap, interpolation="nearest"
        )

        ax.axis("off")

        total_px = smoothed.shape[0]
        total_mm = np.ptp(v1_tissue._positions) * 0.9
        px_per_mm = total_px / total_mm
        add_scale_bar(ax, 10 * px_per_mm, flipud=True)
        
        plt.savefig(save_dir / f"smoothed_map_{name}.png", dpi=300, bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    """
    Example usage:
    python3 visualize/v1_smoothed_opm.py --config configs/analysis_configs/vitb14_dinov2_imagenet_unoptimized.yaml --layer blocks.2
    """
    main()
