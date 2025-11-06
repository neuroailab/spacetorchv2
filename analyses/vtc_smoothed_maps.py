import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import patches
from tqdm import tqdm

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

    is_swinv2 = ("swinv2" in cfg.name)

    save_dir = Path(cfg.output_dir) / args.layer
    save_dir.mkdir(parents=True, exist_ok=True)

    counter = 0
    for _ in [2, 3, 4, 6, 8, 10, 12]:
        if (save_dir / f"vtc_map_smoothed_t{_}.png").exists():
            print(f"Found existing results in {save_dir}, skipping...")
            counter += 1
        else:
            break
        if counter == 7:
            return
        
    _t = [2, 3, 4, 6, 8, 10, 12][counter:]

    floc_tissue = get_floc_tissue(
        cfg.name,
        model,
        positions,
        layer=args.layer,
        output_dir=save_dir,
        is_swinv2=is_swinv2
    )

    for t in _t:
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(1, 1))

        ax.add_patch(patches.Rectangle((0, 0), height=70, width=70, facecolor="#ddd"))

        floc_tissue.patches = []
        for contrast in contrasts:
            floc_tissue.find_patches(
                contrast,
                t=t,
                output_dir=save_dir,
                # skip_cache=True
            )

        for patch in floc_tissue.patches:
            ax.add_patch(patch.to_mpl_poly(alpha=0.8, lw=0.5))

        ax.set_xlim([0, 70])
        ax.set_ylim([0, 70])
        ax.axis("off")

        add_scale_bar(ax, 10, y_start=0)
            
        fig.savefig(save_dir / f"vtc_map_smoothed_t{t}.png", dpi=300, bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    """
    Example usage:
    python3 visualize/vtc_smoothed_maps.py --config configs/analysis_configs/vitb14_dinov2_imagenet_unoptimized.yaml --layer blocks.10
    """
    main()