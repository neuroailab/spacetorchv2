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
    is_tdann = "tdann" in cfg.name and not "tdann_logpolar" in cfg.name
    positions = get_positions(cfg, rescale=is_tdann)[args.layer]

    save_dir = Path(cfg.output_dir) / args.layer
    save_dir.mkdir(parents=True, exist_ok=True)

    # if (save_dir / "vtc_map_contrasts.png").exists():
    #     print(f"Found existing results in {save_dir}, skipping...")
    #     return

    floc_tissue = get_floc_tissue(
        cfg.name,
        model,
        positions,
        layer=args.layer,
        output_dir=save_dir,
    )

    fig, axs = plt.subplots(1, 5, figsize=(10, 2))

    max_sel = 0
    for ax, contrast in zip(axs, DOMAIN_CONTRASTS):
        z = floc_tissue.responses.selectivity(on_categories=contrast.on_categories)
        max_sel = max(max_sel, max(z))
    print(max_sel)

    for ax, contrast in zip(axs, DOMAIN_CONTRASTS):
        z = floc_tissue.responses.selectivity(on_categories=contrast.on_categories)

        ax.scatter(x=positions.coordinates[:, 0], y=positions.coordinates[:, 1], s=(np.abs(z) / np.max(z)) * 5, c=z, cmap='seismic', vmax=max_sel, vmin=-max_sel)

        ax.set_title(contrast.name)
        
        ax.axis("off")

    fig.savefig(save_dir / f"vtc_map_contrasts.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()