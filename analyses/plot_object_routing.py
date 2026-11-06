"""Heatmap of object-to-object routing for one layer, with pictures for ticks.

Reads the ``category_maps.npz`` written by ``object_routing.py`` and draws the
16 x 16 routing matrix for a single block: cell (i, j) is the mean attention
between an object of category i and an object of category j.

What each cell is averaged over. The matrix is indexed by *one* object
property, and everything else is marginalized: heads within the block, the
other property, object position, and the direction of the pair (A-to-B and
B-to-A are averaged, so the matrix is symmetric by construction). So the shape
matrix is averaged over colour and the colour matrix over shape -- the two
panels are the two marginals, not two slices.

Tick labels are the objects themselves. Shapes are drawn as silhouettes
averaged over every colour the shape appears in, so a shape tick carries no
colour information; colours are drawn as swatches of their generating RGB, so
a colour tick carries no shape information. Both are built from the actual
stimuli, not redrawn by hand.

Example:
    python3 analyses/plot_object_routing.py \\
        --routing_dir checkpoints/vitb16a_supervised_imagenet/object_routing \\
        --block 5
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from spacetorch.maps.discrimination_tissue import (
    build_object_thumbnails,
    thumbnail_zoom,
)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--routing_dir",
        type=str,
        default="checkpoints/vitb16a_supervised_imagenet/object_routing",
        help="directory holding category_maps.npz",
    )
    parser.add_argument(
        "--block",
        type=int,
        default=None,
        help="block index to plot; defaults to the one with the strongest "
        "same-category routing",
    )
    parser.add_argument(
        "--property",
        type=str,
        choices=["shape", "color", "both"],
        default="both",
        help="which marginal to draw",
    )
    parser.add_argument(
        "--center",
        action="store_true",
        default=True,
        help="subtract the matrix mean so the diagonal is visible; raw "
        "attention values differ between blocks by more than the effect does",
    )
    parser.add_argument("--raw", dest="center", action="store_false")
    parser.add_argument(
        "--n_examples",
        type=int,
        default=60,
        help="stimuli averaged per shape when building the silhouette ticks",
    )
    parser.add_argument(
        "--tick_resolution",
        type=int,
        default=256,
        help="pixels per side of each tick thumbnail; the source object is only "
        "about 20 pixels across, so this buys clean edges rather than new detail",
    )
    parser.add_argument(
        "--tick_points",
        type=float,
        default=15.0,
        help="on-page size of each tick thumbnail, in points",
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def add_image_ticks(
    axis, images: np.ndarray, order: np.ndarray, points: float = 15.0
):
    """Replace the numeric ticks with thumbnails, below and to the left.

    ``imshow`` draws with the y axis inverted, so the bottom edge of the
    matrix is at ``len(order) - 0.5`` rather than ``-0.5``; anchoring the x
    ticks at -0.5 puts them on top of the first row instead of under the plot.
    """
    axis.set_xticks([])
    axis.set_yticks([])
    zoom = thumbnail_zoom(images.shape[1], points)
    bottom = len(order) - 0.5
    offset = 2.0 + points / 2.0  # clear the axis by half a thumbnail
    for position, category in enumerate(order):
        thumb = images[int(category)]
        for xy, box_xy in (
            ((position, bottom), (0, -offset)),  # below the x axis
            ((-0.5, position), (-offset, 0)),  # left of the y axis
        ):
            axis.add_artist(
                AnnotationBbox(
                    OffsetImage(thumb, zoom=zoom),
                    xy,
                    xybox=box_xy,
                    xycoords="data",
                    boxcoords="offset points",
                    frameon=False,
                    annotation_clip=False,
                )
            )


def plot_matrix(
    axis,
    matrix: np.ndarray,
    images: np.ndarray,
    order: np.ndarray,
    title: str,
    center: bool,
    points: float,
):
    values = matrix - np.nanmean(matrix) if center else matrix
    if center:
        limit = np.nanmax(np.abs(values))
        handle = axis.imshow(values, cmap="seismic", vmin=-limit, vmax=limit)
    else:
        handle = axis.imshow(values, cmap="viridis")

    add_image_ticks(axis, images, order, points=points)
    axis.set_title(title, fontsize=8)
    axis.set_xlabel("object B", labelpad=points + 8, fontsize=8)
    axis.set_ylabel("object A", labelpad=points + 8, fontsize=8)
    return handle


def main():
    args = get_parser().parse_args()
    routing_dir = Path(args.routing_dir)
    stored = np.load(routing_dir / "category_maps.npz", allow_pickle=True)

    layers = [str(name) for name in stored["layers"]]
    maps = {"shape": stored["shape_maps"], "color": stored["color_maps"]}
    orders = {"shape": stored["shape_order"], "color": stored["color_order"]}
    indices = {
        "shape": stored["shape_diagonal_index"],
        "color": stored["color_diagonal_index"],
    }

    block = args.block
    if block is None:
        block = int(np.argmax(np.abs(indices["shape"])))
        print(f"no --block given; using {block}, the strongest same-shape routing")
    assert 0 <= block < len(layers), f"block must be in 0..{len(layers) - 1}"

    shape_images, color_images = build_object_thumbnails(
        n_examples=args.n_examples,
        resolution=args.tick_resolution,
        cache=routing_dir / "tick_images.npz",
    )
    images = {"shape": shape_images, "color": color_images}

    properties = ["shape", "color"] if args.property == "both" else [args.property]
    fig, axs = plt.subplots(
        1,
        len(properties),
        figsize=(4.6 * len(properties), 4.4),
        dpi=args.dpi,
        constrained_layout=True,
    )
    axs = np.atleast_1d(axs)

    for prop, axis in zip(properties, axs):
        other = "colour" if prop == "shape" else "shape"
        handle = plot_matrix(
            axis,
            maps[prop][block],
            images[prop],
            orders[prop],
            title=(
                f"{prop} routing, {layers[block].split('.attn')[0]}\n"
                f"averaged over heads and over {other}"
                + ("  (hue order)" if prop == "color" else "")
            ),
            center=args.center,
            points=args.tick_points,
        )
        bar = fig.colorbar(handle, ax=axis, fraction=0.046)
        bar.ax.tick_params(labelsize=6)
        bar.set_label(
            "attention " + ("(mean subtracted)" if args.center else ""), fontsize=7
        )
        print(
            f"{prop}: diagonal {np.nanmean(np.diag(maps[prop][block])):.5f} "
            f"vs off-diagonal "
            f"{np.nanmean(maps[prop][block][~np.eye(len(maps[prop][block]), dtype=bool)]):.5f} "
            f"(index {indices[prop][block]:+.5f})"
        )

    output = Path(args.output) if args.output else routing_dir / f"routing_block{block}.svg"
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight", format="svg")
    print(f"saved {output}")


if __name__ == "__main__":
    main()
