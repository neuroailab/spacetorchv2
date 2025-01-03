import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from spacetorch.configs import get_cfg
from spacetorch.maps.pinwheel_detector import PinwheelDetector
from spacetorch.variants import get_variant
from spacetorch.variants.positions import get_positions
from spacetorch.maps import TissueMap
from spacetorch.maps.sine_tissue import get_sine_tissue


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


def get_pinwheel_density(tissue: TissueMap):
    density = []

    edge_size = np.ptp(tissue._positions)
    num_hcol = edge_size / 3.5

    tissue.reset_unit_mask()

    pindet = PinwheelDetector(tissue, size_mult=1.5)
    pos, neg = pindet.count_pinwheels(var_thresh=0.3)
    total = pos + neg

    density = total / (num_hcol**2)

    return density


def main():
    cfg = get_cfg(args)
    variant = get_variant(cfg.variant.name)
    variant.set_eval_cfg(cfg)
    variant.set_eval_model(cfg)

    model = variant.eval_model
    positions = get_positions(cfg, rescale=False)[args.layer]

    save_dir = Path(cfg.output_dir) / "figures"
    save_dir.mkdir(parents=True, exist_ok=True)

    v1_tissue = get_sine_tissue(
        cfg.name,
        model,
        positions,
        layer=args.layer,
        output_dir=save_dir.parent,
        skip_cache=True,
    )

    density = get_pinwheel_density(v1_tissue)
    print(density)
    np.save(save_dir / "pinwheel_density.npy", density)

    plt.figure(figsize=(1.5, 1.5))
    b = plt.bar(["Model"], [density], color="black")
    for rect in b:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.3f}', ha='center', va='bottom', fontsize=7)
    plt.ylabel(("Pinwheels /" "\n" r"Column Spacing$^2$"))
    plt.yticks([0, 1, 2, 3, 4])
    plt.savefig(save_dir / "v1_pinwheel_density.png", bbox_inches="tight", dpi=300)
    

if __name__ == "__main__":
    """
    Example usage:
    python3 scripts/v1_pinwheel_density.py --config configs/analysis_configs/vitb14_dinov2_imagenet_unoptimized.yaml --layer blocks.2
    """
    main()
