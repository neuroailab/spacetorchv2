"""Mutual attention between the two objects and every unit, over the sheet.

One stimulus holds two objects at known token positions. For each unit the
quantity drawn here is the attention that unit's location *exchanges* with
those objects -- symmetric, so both the attention it sends and the attention
it receives count:

    mutual(q) = mean over heads, mean over object tokens p of
                (A[q, p] + A[p, q]) / 2

Every unit at token position q is painted with that value. The hypothesis the
picture is meant to test is that units responding to objects are the ones
exchanging attention: if so, the bright territory on the mutual-attention map
should coincide with the positive territory on the object-versus-background
map drawn beside it. Both maps show every unit at its cortical position, with
no smoothing and no thresholding, so what is plotted is the measurement
itself.

Four stimuli are shown, one per cell of the generator's agreement design --
the two objects sharing a shape, a colour, both, or neither. Nothing is
averaged over stimuli: each row is one image, so the attention pattern
belongs to a scene you can look at.

One property of the measurement is worth stating plainly, because it bounds
what the map can show. Attention is a function of token position only; it
does not distinguish channels. A unit here is a channel/position pair, so the
several hundred channels sharing one token position all receive the same
mutual-attention value and are painted identically. Any structure visible on
the sheet therefore reflects how the topographic loss arranged token
positions, not how it arranged features. If the sheet is organised by channel
rather than by position, this map is scattered by construction -- which is
itself worth knowing, and is why the objectness map sits next to it.

Example:
    python3 analyses/object_contact_map.py \\
        --config configs/vitb16a_supervised_imagenet/taskonly.yaml \\
        --layer blocks.5.attn
"""

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr

from spacetorch.configs import get_cfg
from spacetorch.datasets import DatasetRegistry
from spacetorch.datasets.discrimination import (
    ATTRIBUTE_INDEX,
    OBJECT_PATCHES,
    PATCH_GRID,
)
from spacetorch.feature_extractor import get_features_from_layer
from spacetorch.maps.discrimination_tissue import accumulate_object_tuning
from spacetorch.variants import get_variant
from spacetorch.variants.positions import get_positions

PATCH_PX = 16  # the generator's patch size, so an object is 32 x 32 pixels

# the four cells of the generator's agreement design
CONDITIONS: List[Tuple[str, int, int]] = [
    ("same shape, same colour", 1, 1),
    ("same shape, different colour", 1, 0),
    ("different shape, same colour", 0, 1),
    ("different shape, different colour", 0, 0),
]


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--layer",
        type=str,
        required=True,
        help="attention module, e.g. blocks.5.attn; its stored attention "
        "matrix is read as `<layer>.attn_matrix`",
    )
    parser.add_argument(
        "--n_tuning_images",
        type=int,
        default=2000,
        help="stimuli swept to measure each unit's object-versus-background "
        "response, the comparison map",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="rainbow",
        help="colormap for mutual attention. `rainbow` and `turbo` are ordered "
        "and so read as a magnitude; `hsv` is cyclic and will wrap the top of "
        "the range back onto the bottom, which for a magnitude is misleading",
    )
    parser.add_argument(
        "--percentile_clip",
        type=float,
        default=99.0,
        help="clip the colour scale at this percentile. Attention is heavily "
        "skewed -- the object's own tokens are orders of magnitude above the "
        "rest -- so without clipping every other unit renders as one colour",
    )
    parser.add_argument(
        "--point_size", type=float, default=2.0, help="scatter marker size"
    )
    parser.add_argument(
        "--mark_objects",
        action="store_true",
        help="outline the units whose token position is covered by an object. "
        "These are the source sites; they scatter over the whole sheet, since "
        "the channels of one token position need not be neighbours",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--chunk_size", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
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


def pick_condition_images(labels: np.ndarray) -> Dict[str, Optional[int]]:
    """One example stimulus per condition, as a row index into the dataset.

    Among the candidates the one with the most typical object separation is
    taken. Objects are placed anywhere on a 13x13 grid, so an arbitrary pick
    can easily be a touching pair or one at opposite corners, neither of
    which is representative.
    """
    distance_column = ATTRIBUTE_INDEX["distance_patches"]
    median = float(np.median(labels[:, distance_column]))

    chosen: Dict[str, Optional[int]] = {}
    for name, shape_flag, color_flag in CONDITIONS:
        match = np.flatnonzero(
            (labels[:, ATTRIBUTE_INDEX["shape_same"]] == shape_flag)
            & (labels[:, ATTRIBUTE_INDEX["color_same"]] == color_flag)
        )
        if len(match) == 0:
            chosen[name] = None
            continue
        offsets = np.abs(labels[match, distance_column] - median)
        chosen[name] = int(match[np.argmin(offsets)])
    return chosen


def object_tokens(label: np.ndarray, object_idx: int, grid: int) -> np.ndarray:
    """Token indices covered by one object, on a grid of any resolution.

    The generator placed each object in a 2x2 window of a 14x14 grid of
    16-pixel patches. A model with a different patch size has a different
    token grid, so the window is converted through fractional image
    coordinates rather than assumed to be 2x2. The +1 accounts for the CLS
    token.
    """
    block_y = label[ATTRIBUTE_INDEX[f"block{object_idx}_y"]]
    block_x = label[ATTRIBUTE_INDEX[f"block{object_idx}_x"]]

    def span(start: float) -> range:
        low = int(np.floor(start / PATCH_GRID * grid))
        high = int(np.ceil((start + OBJECT_PATCHES) / PATCH_GRID * grid))
        return range(max(low, 0), min(max(high, low + 1), grid))

    rows, cols = span(block_y), span(block_x)
    return np.array([1 + r * grid + c for r in rows for c in cols], dtype=int)


def mutual_attention(
    attention: np.ndarray, label: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Attention each token exchanges with the two objects, per patch token.

    Averaged over heads and symmetrised: a unit counts as connected to the
    objects whether it is attending to them or being attended by them. The
    CLS token is dropped, since it has no position on the sheet.
    """
    n_heads, n_tokens, _ = attention.shape
    n_patches = n_tokens - 1
    grid = int(round(math.sqrt(n_patches)))
    assert grid * grid == n_patches, (
        f"{n_patches} patch tokens is not a square grid; this analysis assumes "
        "one CLS token followed by a square patch grid"
    )

    objects = np.concatenate(
        [object_tokens(label, object_idx, grid) for object_idx in (0, 1)]
    )
    pooled = attention.mean(axis=0)  # over heads
    to_objects = pooled[:, objects].mean(axis=1)
    from_objects = pooled[objects, :].mean(axis=0)

    exchanged = 0.5 * (to_objects + from_objects)
    return exchanged[1:].reshape(grid, grid), objects


def object_cell_mask(
    objects: np.ndarray, grid: int, dims: Tuple[int, int, int]
) -> np.ndarray:
    """Units whose token position is covered by either object."""
    n_channels, height, width = dims
    covered = np.zeros(grid * grid, dtype=bool)
    covered[objects - 1] = True
    return np.tile(covered.reshape(grid, grid).reshape(-1), n_channels)


def spread_to_units(per_cell: np.ndarray, dims: Tuple[int, int, int]) -> np.ndarray:
    """Give every channel at a token position the same value.

    The tissue's unit ordering is channel major, then row, then column, which
    is what ``LayerPositions.coordinates`` uses, so tiling the flattened cell
    grid once per channel lines the values up with the positions.
    """
    n_channels, height, width = dims
    assert per_cell.shape == (height, width), (
        f"attention grid {per_cell.shape} does not match the feature map "
        f"{(height, width)}; the layer and the attention matrix disagree"
    )
    return np.tile(per_cell.reshape(-1), n_channels)


def draw_stimulus(axis, image: np.ndarray, label: np.ndarray) -> None:
    """The stimulus with its two objects boxed and named A and B."""
    from matplotlib import patches

    axis.imshow(image)
    for object_idx, style in ((0, "-"), (1, "--")):
        top = int(label[ATTRIBUTE_INDEX[f"block{object_idx}_y"]]) * PATCH_PX
        left = int(label[ATTRIBUTE_INDEX[f"block{object_idx}_x"]]) * PATCH_PX
        size = OBJECT_PATCHES * PATCH_PX
        axis.add_patch(
            patches.Rectangle(
                (left - 2, top - 2), size + 4, size + 4,
                fill=False, edgecolor="black", lw=0.8, ls=style,
            )
        )
        axis.text(
            left + size / 2, top - 5, "AB"[object_idx],
            ha="center", va="bottom", fontsize=7, fontweight="bold",
        )
    axis.set_xticks([])
    axis.set_yticks([])


def scatter_units(
    axis,
    coordinates: np.ndarray,
    values: np.ndarray,
    cmap: str,
    vmin: float,
    vmax: float,
    on_object: Optional[np.ndarray] = None,
):
    """Every unit at its cortical position, coloured by ``values``."""
    handle = axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=args.point_size,
        linewidths=0,
        rasterized=True,
    )
    if on_object is not None and on_object.any():
        sites = np.flatnonzero(on_object)
        axis.scatter(
            coordinates[sites, 0], coordinates[sites, 1],
            marker="o", s=args.point_size * 3, facecolors="none",
            edgecolors="black", linewidths=0.2, rasterized=True, zorder=2,
        )
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    return handle


def stimulus_image(dataset, row: int) -> np.ndarray:
    """The unmodified stimulus, for display beside its maps."""
    from PIL import Image

    return np.asarray(Image.open(dataset.file_list[row]).convert("RGB"))


def main():
    cfg = get_cfg(args)
    variant = get_variant(cfg.variant.name)
    variant.set_eval_cfg(cfg)
    variant.set_eval_model(cfg)
    model = variant.eval_model

    from spacetorch.variants import OUTPUT_DIMS_FOR_224_INPUTS

    dims = OUTPUT_DIMS_FOR_224_INPUTS[cfg.variant.architecture][args.layer]
    n_channels, height, width = dims
    positions = get_positions(cfg, rescale=False)[args.layer]
    coordinates = np.asarray(positions.coordinates)
    assert len(coordinates) == n_channels * height * width, (
        f"{len(coordinates)} positions but dims say {n_channels * height * width}"
    )

    save_dir = Path(cfg.output_dir) / args.layer / "contact_map"
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = DatasetRegistry.get("Discrimination_val_attributes")
    labels = dataset.labels

    # ---- the comparison map: does the unit answer to objects at all? ----
    print(f"measuring object versus background over {args.n_tuning_images} stimuli",
          flush=True)
    tuning = accumulate_object_tuning(
        model,
        args.layer,
        dataset,
        indices=np.linspace(0, len(dataset) - 1, args.n_tuning_images).astype(int),
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        num_workers=args.num_workers,
    )
    objectness = tuning.objectness
    object_limit = np.nanpercentile(np.abs(objectness), 99)

    # ---- attention for one stimulus per condition ----
    chosen = pick_condition_images(labels)
    rows = [chosen[name] for name, _, _ in CONDITIONS if chosen[name] is not None]
    print(f"attention for {len(rows)} example stimuli: {rows}", flush=True)

    attention_layer = f"{args.layer}.attn_matrix"
    features, _, example_labels = get_features_from_layer(
        model,
        torch.utils.data.Subset(dataset, rows),
        model_layer_strings=[attention_layer],
        batch_size=args.batch_size,
        return_inputs_and_labels=True,
        shuffle=False,
        verbose=False,
        num_workers=args.num_workers,
    )
    attention = np.asarray(features, dtype=np.float32)
    assert attention.ndim == 4, (
        f"expected (batch, heads, tokens, tokens) attention, got "
        f"{attention.shape}; `{attention_layer}` did not return an attention "
        "matrix"
    )

    # ---- the figure ----
    fig, axs = plt.subplots(
        len(CONDITIONS), 3,
        figsize=(9.6, 2.9 * len(CONDITIONS)),
        constrained_layout=True,
    )
    axs = np.atleast_2d(axs)

    summary, stored = [], {}
    position = 0
    for row_idx, (name, _, _) in enumerate(CONDITIONS):
        source_row = chosen[name]
        if source_row is None:
            for axis in axs[row_idx]:
                axis.axis("off")
            axs[row_idx, 0].set_title(f"{name}\nno stimulus in this cell", fontsize=7)
            continue

        label = labels[source_row]
        per_cell, objects = mutual_attention(attention[position], label)
        position += 1

        values = spread_to_units(per_cell, dims)
        on_object = object_cell_mask(objects, per_cell.shape[0], dims)
        stored[f"mutual_{source_row}"] = values

        # attention onto the objects themselves dwarfs everything else
        vmax = float(np.percentile(values, args.percentile_clip))
        vmin = float(values.min())

        draw_stimulus(axs[row_idx, 0], stimulus_image(dataset, source_row), label)
        axs[row_idx, 0].set_title(f"{name}\nimage {source_row}", fontsize=7)

        handle = scatter_units(
            axs[row_idx, 1], coordinates, values, args.cmap, vmin, vmax,
            on_object=on_object if args.mark_objects else None,
        )
        bar = fig.colorbar(handle, ax=axs[row_idx, 1], fraction=0.046)
        bar.ax.tick_params(labelsize=5)
        bar.set_label("mutual attention with the objects", fontsize=6)

        handle = scatter_units(
            axs[row_idx, 2], coordinates, objectness, "RdBu_r",
            -object_limit, object_limit,
        )
        bar = fig.colorbar(handle, ax=axs[row_idx, 2], fraction=0.046)
        bar.ax.tick_params(labelsize=5)
        bar.set_label("object minus background", fontsize=6)

        # does the bright territory coincide with the object-driven territory?
        finite = np.isfinite(objectness) & np.isfinite(values)
        correlation = float(pearsonr(values[finite], objectness[finite])[0])
        prefers = finite & (objectness > 0)
        contrast = float(values[prefers].mean() - values[finite & ~prefers].mean())

        axs[row_idx, 1].set_title(
            f"mutual attention, clipped at the {args.percentile_clip:g}th pct\n"
            f"r with objectness = {correlation:+.3f}",
            fontsize=7,
        )
        axs[row_idx, 2].set_title(
            f"object versus background\n"
            f"object-preferring units get {contrast:+.2e} more attention",
            fontsize=7,
        )
        summary.append(
            {
                "condition": name,
                "image": source_row,
                "correlation_with_objectness": correlation,
                "attention_contrast": contrast,
                "mean_mutual_attention": float(values.mean()),
                "max_mutual_attention": float(values.max()),
                "n_object_preferring": int(prefers.sum()),
            }
        )

    fig.suptitle(
        f"{cfg.name}, {args.layer}: attention exchanged with the two objects, "
        "every unit, unsmoothed",
        fontsize=9,
    )
    fig.savefig(save_dir / "mutual_attention_maps.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    import pandas as pd

    table = pd.DataFrame(summary)
    table.insert(0, "layer", args.layer)
    table.insert(0, "model", cfg.name)
    table.to_csv(save_dir / "mutual_attention.csv", index=False)

    np.savez(
        save_dir / "mutual_attention.npz",
        layer=args.layer,
        coordinates=coordinates,
        objectness=objectness,
        images=np.array(rows),
        **stored,
    )

    print("\nper condition:")
    print(table.drop(columns=["model", "layer"]).round(5).to_string(index=False))
    print(
        "\nA positive correlation means the units exchanging attention with the "
        "objects are the ones that respond to objects. Note that attention is a "
        "function of token position only, so every channel at a position shares "
        "one value; the correlation is therefore carried entirely by which "
        "positions are object-driven, not by which features are."
    )
    print(f"\nSaved results to {save_dir}")


if __name__ == "__main__":
    main()
