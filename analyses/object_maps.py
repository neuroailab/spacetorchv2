"""Object shape and colour preference plotted over the cortical sheet.

The object counterpart of ``v1_maps.py``. There, every unit gets a preferred
orientation from full-field gratings and is drawn at its cortical position,
coloured by that preference. Here the preference comes from the discrimination
objects: each unit's tuning curve is built from the objects that actually fell
inside its receptive field, since an object covers four patches rather than the
whole image.

Three maps are produced per layer -- preferred shape, preferred colour, and an
object-versus-background map -- alongside the quantitative question a map
raises, which is whether the preference is spatially clustered at all. That is
the agreement-versus-distance curve: the chance two units prefer the same
category as a function of how far apart they sit, against a control that
shuffles the pairwise distances. A decaying curve is a topographic map; a flat
one at the population base rate means the colours on the map are arranged at
random however pretty they look.

Units are channel/position pairs, matching ``LayerPositions.coordinates``, not
the channel-level axes used by ``privileged_axes.py``.

Example:
    python3 analyses/object_maps.py \\
        --config configs/tdann/taskonly_seed0.yaml --layer layer2.0
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spacetorch.configs import get_cfg
from spacetorch.datasets import DatasetRegistry
from spacetorch.maps.discrimination_tissue import (
    PROPERTIES,
    build_object_thumbnails,
    category_legend,
    get_discrimination_tissue,
)
from spacetorch.variants import get_variant
from spacetorch.variants.positions import get_positions


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--layer", type=str, required=True)
    parser.add_argument(
        "--n_images",
        type=int,
        default=0,
        help="discrimination stimuli to sweep; 0, the default, uses the split. "
        "A given site is covered by an object in roughly 5% of images, so "
        "trimming this quickly makes the per-unit tuning curves noisy",
    )
    parser.add_argument(
        "--selectivity_percentile",
        type=float,
        default=75.0,
        help="keep only units above this percentile of tuning-curve range; a "
        "weakly driven unit still has an argmax and would paint noise",
    )
    parser.add_argument(
        "--zoom_pct",
        type=float,
        nargs=4,
        default=[30, 50, 30, 50],
        help="x0 x1 y0 y1 as percentages of sheet width, for the zoomed panel",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--chunk_size", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=20)
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


def add_scale_bar(axis, width: float, height=None):
    """Rectangular scale bar in the bottom right, as in ``v1_maps``."""
    from matplotlib import patches

    xlim, ylim = axis.get_xlim(), axis.get_ylim()
    if height is None:
        height = 0.025 * np.ptp(ylim)
    start = xlim[1] - (width + np.ptp(xlim) * 0.04)
    axis.add_patch(
        patches.Rectangle(
            (start, ylim[0]), width, height, clip_on=False, facecolor="#222222"
        )
    )


def add_category_legend(fig, axis, metric, thumbnails):
    """A strip of the category colours, with each category's picture on it.

    Shape indices are not a scale, so a colour bar running from ``0`` to
    ``15`` says nothing about what a patch of colour on the map means. The
    strip instead shows each shape's silhouette drawn on the colour that
    stands for it, which is directly readable against the map.
    """
    return category_legend(axis, metric, thumbnails.get(metric.name))


def main():
    cfg = get_cfg(args)
    variant = get_variant(cfg.variant.name)
    variant.set_eval_cfg(cfg)
    variant.set_eval_model(cfg)
    model = variant.eval_model

    is_tdann = "tdann" in cfg.name and "tdann_logpolar" not in cfg.name
    positions = get_positions(cfg, rescale=is_tdann)[args.layer]

    save_dir = Path(cfg.output_dir) / args.layer
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = DatasetRegistry.get("Discrimination_val_attributes")

    # silhouettes for the shape legend; the colour legend needs none, since a
    # colour band already is the colour it stands for
    shape_thumbnails, _ = build_object_thumbnails(
        dataset=dataset, cache=Path(cfg.output_dir) / "object_thumbnails.npz"
    )
    thumbnails = {"shape": shape_thumbnails}

    indices = (
        None
        if args.n_images <= 0
        else np.linspace(0, len(dataset) - 1, args.n_images).astype(int)
    )

    tissue = get_discrimination_tissue(
        model,
        positions,
        layer=args.layer,
        dataset=dataset,
        indices=indices,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        num_workers=args.num_workers,
    )
    print(f"{len(tissue.tuning)} units at {args.layer}", flush=True)

    # ---- full-sheet preference maps, one per property, plus objectness ----
    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(9, 3), constrained_layout=True)
    for prop, axis in zip(PROPERTIES, axs):
        tissue.set_unit_mask_by_selectivity_percentile(prop, args.selectivity_percentile)
        tissue.make_parameter_map(
            axis, prop=prop, linewidths=0, rasterized=True, final_s=3
        )
        axis.set_title(
            f"{prop} preference, top {100 - args.selectivity_percentile:.0f}% selective",
            fontsize=7,
        )
        axis.axis("off")
        add_category_legend(fig, axis, tissue.tuning.metrics[prop], thumbnails)

    tissue.reset_unit_mask()
    objectness = tissue.tuning.objectness
    limit = np.nanpercentile(np.abs(objectness), 99)
    handle = axs[2].scatter(
        tissue.positions[:, 0],
        tissue.positions[:, 1],
        c=objectness,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        s=3,
        linewidths=0,
        rasterized=True,
    )
    axs[2].set_title("object minus background", fontsize=7)
    axs[2].axis("off")
    fig.colorbar(handle, ax=axs[2], fraction=0.046).ax.tick_params(labelsize=6)
    fig.savefig(save_dir / "object_maps_full.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- zoomed panels, where any local structure is actually visible ----
    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(6, 3), constrained_layout=True)
    x0, x1, y0, y1 = args.zoom_pct
    for prop, axis in zip(PROPERTIES, axs):
        tissue.set_unit_mask_by_selectivity_percentile(prop, args.selectivity_percentile)
        keep = tissue.unit_mask
        tissue.set_mask_by_pct_limits([[x0, x1], [y0, y1]])
        tissue.unit_mask = np.intersect1d(tissue.unit_mask, keep)
        if len(tissue.unit_mask) == 0:
            print(
                f"zoom window {args.zoom_pct} holds no selective {prop} units; "
                "widen --zoom_pct or lower --selectivity_percentile"
            )
        tissue.make_parameter_map(axis, prop=prop, linewidths=0, final_s=6)
        axis.set_title(f"{prop}, zoom (n={len(tissue.unit_mask)})", fontsize=7)
        axis.axis("off")
    add_scale_bar(axs[-1], width=2)
    fig.savefig(save_dir / "object_maps_zoom.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- is the preference clustered at all? ----
    curves = {}
    fig, axis = plt.subplots(figsize=(3, 2.2), constrained_layout=True)
    for prop, color in zip(PROPERTIES, ["black", "tab:blue"]):
        tissue.set_unit_mask_by_selectivity_percentile(prop, args.selectivity_percentile)
        for shuffle, style in ((False, "-"), (True, ":")):
            midpoints, sampled = tissue.preference_agreement_over_distance(
                prop, num_samples=args.num_samples, shuffle=shuffle
            )
            mean = np.nanmean(sampled, axis=0)
            sd = np.nanstd(sampled, axis=0)
            curves[f"{prop}{'_shuffled' if shuffle else ''}"] = mean
            curves[f"{prop}{'_shuffled' if shuffle else ''}_sd"] = sd
            curves["distance_mm"] = midpoints
            axis.plot(
                midpoints, mean, style, color=color,
                label=f"{prop}{' (shuffled)' if shuffle else ''}", lw=1,
            )
            if not shuffle:
                axis.fill_between(midpoints, mean - sd, mean + sd, color=color, alpha=0.2)
    axis.set_xlabel("cortical distance (mm)")
    axis.set_ylabel("P(same preferred category)")
    axis.legend(fontsize=5, frameon=False)
    fig.savefig(save_dir / "object_map_clustering.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    tissue.reset_unit_mask()
    np.savez(
        save_dir / "object_maps.npz",
        layer=args.layer,
        coordinates=tissue.positions,
        objectness=objectness,
        **{f"preferred_{p}": tissue.tuning.preferences(p) for p in PROPERTIES},
        **{f"selectivity_{p}": tissue.tuning.selectivity(p) for p in PROPERTIES},
        **curves,
    )

    print("\nclustering, near versus far (mean over samples):")
    for prop in PROPERTIES:
        near, far = curves[prop][0], curves[prop][-1]
        near_s, far_s = curves[f"{prop}_shuffled"][0], curves[f"{prop}_shuffled"][-1]
        print(
            f"  {prop:6s} native {near:.3f} -> {far:.3f}   "
            f"shuffled {near_s:.3f} -> {far_s:.3f}"
        )
    print(f"\nSaved results to {save_dir}")


if __name__ == "__main__":
    main()
