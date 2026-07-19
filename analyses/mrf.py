import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from spacetorch.receptive_fields.rf_helper import *
from spacetorch.receptive_fields.visualize_helper import *
from spacetorch.configs import get_cfg
from spacetorch.variants import get_variant
from spacetorch.maps.tiled_sine_tissue import get_tiled_sine_responses


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


def main():
    cfg = get_cfg(args)
    variant = get_variant(cfg.variant.name)
    variant.set_eval_cfg(cfg)
    variant.set_eval_model(cfg)
    model = variant.eval_model

    save_dir = Path(cfg.output_dir) / args.layer
    save_dir.mkdir(parents=True, exist_ok=True)

    is_lcnn = ("lcnn" in cfg.name)
    is_llcnn = ("llcnn" in cfg.name)

    responses = get_tiled_sine_responses(
        model,
        layers=[args.layer],
        architecture=cfg.variant.architecture,
        normalize_to_ringach_firing_rates=("tdann" in cfg.name) or ("resnet" in cfg.name),
        is_lcnn=is_lcnn and not is_llcnn,
        is_llcnn=is_llcnn,
    )

    maps = responses.get_spatial_response_maps()

    np.save(save_dir / "mrfs.npy", maps)

    try:
        fig, axs = plt.subplots(10, 10, figsize=(10, 10))
        for i in range(10):
            for j in range(10):
                d = maps[i * 10 + j]
                minmax = max(d.max(), abs(d.min()))
                axs[i][j].imshow(d, cmap="seismic", vmin=-minmax, vmax=minmax)
                axs[i][j].axis("off")
        plt.savefig(save_dir / "mrf_maps.png", dpi=300, bbox_inches='tight')
    except:
        fig, axs = plt.subplots(1, len(maps), figsize=(len(maps), 1))
        for i in range(len(maps)):
            d = maps[i]
            minmax = max(d.max(), abs(d.min()))
            axs[i].imshow(d, cmap="seismic", vmin=-minmax, vmax=minmax)
            axs[i].axis("off")
        plt.savefig(save_dir / "mrf_maps.png", dpi=300, bbox_inches='tight')

    sizes = responses.get_crf_sizes_deg()

    plt.figure(figsize=(2.5, 2))
    sns.histplot(sizes, color="k")
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8])
    plt.savefig(save_dir / "mrf_sizes.png", dpi=300, bbox_inches='tight')

    np.save(save_dir / "mrf_sizes.npy", sizes)


if __name__ == "__main__":
    main()
