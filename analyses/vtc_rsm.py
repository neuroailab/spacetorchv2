import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from spacetorch.configs import get_cfg
from spacetorch.variants import get_variant
from spacetorch.variants.positions import get_positions
from spacetorch.datasets.floc import DOMAIN_CONTRASTS
from spacetorch.maps.floc_tissue import get_floc_tissue
from spacetorch.utils.rsa import get_model_rsm


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

    floc_tissue = get_floc_tissue(
        cfg.name,
        model,
        positions,
        layer=args.layer,
        output_dir=save_dir,
    )

    rsm = get_model_rsm(floc_tissue, is_itn=False)

    fig, ax = plt.subplots(figsize=(1.3, 1))

    mappable = ax.imshow(
        rsm, cmap="seismic", vmin=-1, vmax=1
    )

    ax.set_xticks(np.arange(5))
    ax.set_yticks(np.arange(5))
    ax.set_xticklabels([""] * 5)
    ax.set_yticklabels([""] * 5)

    cax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
    cb = fig.colorbar(mappable, cax=cax, label="Correlation", ticks=[-1, 1])
    cb.set_label("Correlation", labelpad=-5)
        
    fig.savefig(save_dir / f"vtc_rsm.png", dpi=300, bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    """
    Example usage:
    python3 visualize/vtc_rsm.py --config configs/analysis_configs/vitb14_dinov2_imagenet_unoptimized.yaml --layer blocks.10
    """
    main()