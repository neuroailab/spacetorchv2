import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from spacetorch.swapopt import Swapper
from spacetorch.utils.plot_utils import remove_spines
from spacetorch.configs import get_cfg


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--layer", type=str)
    parser.add_argument("--feature_path", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
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


def log(to_print: str):
    print(f"\nLOG: {to_print}")


def plot_metrics(swapper: Swapper, save_path: Path, layer: str):
    np.save(save_path / f'{layer}_num_swaps', np.array(swapper.metrics.num_swaps))
    np.save(save_path / f'{layer}_losses', np.array(swapper.metrics.losses))

    line_kwargs = {"lw": 1.5, "c": "k"}

    fig, axes = plt.subplots(figsize=(10, 5), ncols=2)
    axes[0].plot(swapper.metrics.num_swaps, **line_kwargs)
    axes[1].plot(swapper.metrics.losses, **line_kwargs)

    axes[0].set_ylabel("Number of Swaps")
    axes[1].set_ylabel("Neighborhood Loss")

    remove_spines(axes)
    fig.savefig(save_path / f'{layer}_plot.png', dpi=100, bbox_inches="tight", facecolor="white")


def main():
    args = get_parser().parse_args()
    cfg = get_cfg(args)
    layer = args.layer
    
    log(f"Constructing swapper for layer {layer}")
    swapper: Swapper = Swapper(
        cfg, feature_path=args.feature_path, layer=layer, dataset_name=args.dataset_name
    )

    # swapper can get blocked if running it would overwrite existing position files
    if not swapper.blocked:
        log("Swapping")
        swapper.swap()

        log("Saving positions back")
        swapper.save_positions()

        log("Saving metrics plot")
        plot_metrics(swapper, swapper.new_save_dir, layer)

        log("All done!")


if __name__ == "__main__":
    """
    Example usage:
    python3 scripts/perform_swapopt.py --config configs/vitb14_dinov2_imagenet.yaml --layer blocks.1 --feature checkpoints/vitb14_dinov2_imagenet/features/SineGrating2019.h5 --dataset_name SineGrating2019
    """
    main()
