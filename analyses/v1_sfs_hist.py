import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path

from spacetorch.configs import get_cfg
from spacetorch.variants import get_variant
from spacetorch.variants.positions import get_positions
from spacetorch.colormaps import nauhaus_raw_colormaps
from spacetorch.maps.sine_tissue import get_sine_tissue, METRIC_DICT


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


def rgba_to_hex(rgba):
    rgb = tuple((rgba[:3] * 255).astype(int))
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def main():
    cfg = get_cfg(args)
    variant = get_variant(cfg.variant.name)
    variant.set_eval_cfg(cfg)
    variant.set_eval_model(cfg)

    model = variant.eval_model
    is_tdann = "tdann" in cfg.name
    positions = get_positions(cfg, rescale=is_tdann)[args.layer]

    save_dir = Path(cfg.output_dir) / (args.layer + "_feature")
    save_dir.mkdir(parents=True, exist_ok=True)

    v1_tissue = get_sine_tissue(
        cfg.name,
        model,
        positions,
        layer=args.layer,
        output_dir=save_dir,
        skip_cache=True,
    )

    fig = plt.figure(figsize=(2, 2))

    colors = nauhaus_raw_colormaps["sfs"]
    color_map = {tuple(rgba): rgba_to_hex(rgba) for rgba in colors}

    dist = v1_tissue.get_unit_colors(metric=METRIC_DICT["sfs"])
    data_map = [rgba_to_hex(np.array(rgba)) for rgba in dist]
    data_counts = Counter(data_map)

    data = list(data_counts.items())
    values, counts = zip(*data)
    print(values, counts)

    bar_colors = [color_map[tuple(v)] for v in values]

    sns.barplot(x=values, y=counts, palette=bar_colors)
    plt.ylabel("")
    plt.xlabel("")
    plt.yticks([])
    plt.xticks([])
    sns.despine()
    
    fig.savefig(save_dir / f"v1_sfs_hist.png", dpi=300, bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    main()
