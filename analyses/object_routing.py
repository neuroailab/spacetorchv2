"""Do ViT heads preferentially link objects that share a shape or a colour?

The object analogue of ``axial_specificity.py``. That analysis measures each
peripheral site's preferred orientation, then asks whether the sites a small
central grating most strongly modulates are the ones tuned to the *same*
orientation -- a contact measure read against a map of feature preference.

Here the contact measure is attention itself. Each discrimination image holds
two objects whose shape and colour categories are known exactly, so for every
head we can read the attention flowing from one object's tokens to the other's
and ask whether it depends on the two objects sharing a category. The map of
feature preference is then the full category-by-category attention matrix: a
raised diagonal is a same-feature routing preference, and off-diagonal
structure shows which categories a head links instead.

Three things come out of it.

* ``routing_by_head.csv`` -- per block and head, mean pair attention split by
  shape agreement, colour agreement, the 2x2 conjunction, and distance
  tertile, with within-object and object-to-background routes as controls.
* ``category_maps.npz`` -- the 16x16 shape-by-shape and colour-by-colour
  attention maps per block, the thing to visualize. Colour categories are
  ordered by hue so that proximity on the map means perceptual similarity;
  shapes have no natural metric and stay in their generator order.
* a summary figure: the two maps, a block-by-head image of the same-minus-
  different index, and its depth profile.

Attention is measured in the intact network -- no ablation and no factorial
presence contrast -- so the quantity is the head's actual routing, not a
causal claim about what the routing is used for.

Example:
    python3 analyses/object_routing.py --config configs/vitb16a_supervised_imagenet/taskonly.yaml
"""

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spacetorch.configs import get_cfg
from spacetorch.datasets import DatasetRegistry
from spacetorch.datasets.discrimination import ATTRIBUTE_INDEX, PATCH_GRID, OBJECT_PATCHES
from spacetorch.feature_extractor import get_pooled_features_from_layer
from spacetorch.variants import get_variant
from spacetorch.variants.positions import BRAIN_MAPPING

# the routes read out of each attention matrix, in the order they are stored
ROUTES = [
    "a_to_b",
    "b_to_a",
    "within_a",
    "within_b",
    "objects_to_other",
    "cls_to_objects",
]


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--layers",
        type=str,
        nargs="+",
        default=None,
        help="attention modules to read; defaults to every block in "
        "BRAIN_MAPPING, as `blocks.N.attn.attn_matrix`",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--n_images",
        type=int,
        default=0,
        help="discrimination stimuli to measure; 0, the default, uses the split",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=64,
        help="attention matrices are heads x tokens x tokens per image, so the "
        "chunk has to stay small when every block is hooked at once",
    )
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


def resolve_attention_layers(cfg) -> List[str]:
    """One ``attn_matrix`` hook per transformer block.

    ``FeatureExtractor`` treats a layer string ending in ``.attn_matrix`` as a
    request to hook the module named by the prefix and read its stored
    attention tensor, so the string has to name the attention submodule
    itself.
    """
    if args.layers is not None:
        return list(args.layers)

    architecture = cfg.variant.architecture
    assert architecture in BRAIN_MAPPING, (
        f"{architecture} has no BRAIN_MAPPING entry; pass --layers explicitly"
    )

    layers = []
    for key in BRAIN_MAPPING[architecture]:
        block = key[: -len(".attn")] if key.endswith(".attn") else key
        assert block.startswith("blocks."), (
            f"{architecture} does not look like a plain ViT ({key}); this "
            "analysis reads `blocks.N.attn.attn_matrix`"
        )
        layers.append(f"{block}.attn.attn_matrix")
    return layers


def object_tokens(label: np.ndarray, object_idx: int, grid: int) -> np.ndarray:
    """Token indices covered by one object, on a grid of any resolution.

    The generator placed each object in a 2x2 window of a 14x14 grid of
    16-pixel patches. A model with a different patch size has a different token
    grid, so the window is converted through fractional image coordinates
    rather than assumed to be 2x2. The +1 accounts for the CLS token.
    """
    block_y = label[ATTRIBUTE_INDEX[f"block{object_idx}_y"]]
    block_x = label[ATTRIBUTE_INDEX[f"block{object_idx}_x"]]

    def span(start: float) -> range:
        low = int(np.floor(start / PATCH_GRID * grid))
        high = int(np.ceil((start + OBJECT_PATCHES) / PATCH_GRID * grid))
        return range(max(low, 0), min(max(high, low + 1), grid))

    rows, cols = span(block_y), span(block_x)
    return np.array([1 + r * grid + c for r in rows for c in cols], dtype=int)


def route_reducer(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Collapse each attention matrix to a handful of group-to-group routes.

    Returns (n_images, n_heads, len(ROUTES)). Every route is a mean attention
    *per key*, so groups of different size stay comparable -- summing instead
    would make a bigger group look more attended purely for being bigger.
    """
    attention = np.asarray(features, dtype=np.float32)
    assert attention.ndim == 4, (
        f"expected (batch, heads, tokens, tokens) attention, got {attention.shape}; "
        "the layer string must end in `.attn_matrix`"
    )
    n_images, n_heads, n_tokens, _ = attention.shape

    n_patches = n_tokens - 1
    grid = int(round(math.sqrt(n_patches)))
    assert grid * grid == n_patches, (
        f"{n_patches} patch tokens is not a square grid; this analysis assumes "
        "one CLS token followed by a square patch grid"
    )

    out = np.zeros((n_images, n_heads, len(ROUTES)), dtype=np.float32)
    for image_idx in range(n_images):
        first = object_tokens(labels[image_idx], 0, grid)
        second = object_tokens(labels[image_idx], 1, grid)
        both = np.concatenate([first, second])
        other = np.setdiff1d(np.arange(1, n_tokens), both)

        matrix = attention[image_idx]
        out[image_idx, :, 0] = matrix[:, first][:, :, second].mean(axis=(1, 2))
        out[image_idx, :, 1] = matrix[:, second][:, :, first].mean(axis=(1, 2))
        out[image_idx, :, 2] = matrix[:, first][:, :, first].mean(axis=(1, 2))
        out[image_idx, :, 3] = matrix[:, second][:, :, second].mean(axis=(1, 2))
        out[image_idx, :, 4] = matrix[:, both][:, :, other].mean(axis=(1, 2))
        out[image_idx, :, 5] = matrix[:, 0:1][:, :, both].mean(axis=(1, 2))

    return out


def stratified_indices(labels: np.ndarray, n_images: int, seed: int) -> np.ndarray:
    """Subsample preserving the shape/colour agreement design of the split."""
    if n_images <= 0 or n_images >= len(labels):
        return np.arange(len(labels))

    cells = (
        labels[:, ATTRIBUTE_INDEX["shape_same"]] * 2
        + labels[:, ATTRIBUTE_INDEX["color_same"]]
    )
    rng = np.random.default_rng(seed)

    chosen = []
    for cell in np.unique(cells):
        members = np.flatnonzero(cells == cell)
        take = int(round(n_images * len(members) / len(labels)))
        chosen.append(rng.choice(members, size=min(take, len(members)), replace=False))

    return np.sort(np.concatenate(chosen))


def hue_order(labels: np.ndarray) -> np.ndarray:
    """Colour categories sorted by hue, so the map has a meaningful axis.

    Without this the colour-by-colour matrix is indexed by an arbitrary label
    number and only its diagonal can be read. Sorted by hue, a head that links
    *similar* rather than identical colours shows up as a broad band.
    """
    import colorsys

    categories = np.unique(labels[:, ATTRIBUTE_INDEX["color0"]]).astype(int)
    hues = {}
    for category in categories:
        row = labels[labels[:, ATTRIBUTE_INDEX["color0"]] == category][0]
        rgb = [row[ATTRIBUTE_INDEX[f"color0_{c}"]] / 255.0 for c in ("r", "g", "b")]
        hues[category] = colorsys.rgb_to_hsv(*rgb)[0]
    return np.array(sorted(categories, key=lambda c: hues[c]))


def category_map(
    pair_attention: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    order: np.ndarray,
) -> np.ndarray:
    """Mean pair attention for every ordered pair of categories.

    Symmetrized, because the pair attention already averages both directions
    and object 0 versus object 1 is only an ordering convention.
    """
    index = {category: i for i, category in enumerate(order)}
    n = len(order)
    total = np.zeros((n, n))
    count = np.zeros((n, n))
    for value, a, b in zip(pair_attention, first, second):
        i, j = index.get(int(a)), index.get(int(b))
        if i is None or j is None:
            continue
        total[i, j] += value
        count[i, j] += 1
        if i != j:
            total[j, i] += value
            count[j, i] += 1
    with np.errstate(invalid="ignore"):
        return np.where(count > 0, total / count, np.nan)


def diagonal_index(matrix: np.ndarray) -> float:
    """Same-category attention minus different-category attention."""
    eye = np.eye(len(matrix), dtype=bool)
    return float(np.nanmean(matrix[eye]) - np.nanmean(matrix[~eye]))


def head_table(
    routes: np.ndarray, labels: np.ndarray, blocks: List[str]
) -> pd.DataFrame:
    """One row per block and head, with the routing contrasts."""
    shape_same = labels[:, ATTRIBUTE_INDEX["shape_same"]].astype(bool)
    color_same = labels[:, ATTRIBUTE_INDEX["color_same"]].astype(bool)
    distance = labels[:, ATTRIBUTE_INDEX["distance_patches"]]
    tertiles = np.quantile(distance, [1 / 3, 2 / 3])

    rows = []
    for block_idx, block in enumerate(blocks):
        block_routes = routes[block_idx]  # (n_images, n_heads, n_routes)
        pair = 0.5 * (block_routes[:, :, 0] + block_routes[:, :, 1])
        for head in range(pair.shape[1]):
            values = pair[:, head]
            row = {
                "layer": block,
                "block": block_idx,
                "head": head,
                "mean_pair_attention": values.mean(),
                "attention_shape_same": values[shape_same].mean(),
                "attention_shape_different": values[~shape_same].mean(),
                "attention_color_same": values[color_same].mean(),
                "attention_color_different": values[~color_same].mean(),
                "within_object": block_routes[:, head, 2:4].mean(),
                "objects_to_other": block_routes[:, head, 4].mean(),
                "cls_to_objects": block_routes[:, head, 5].mean(),
                "directional_asymmetry": float(
                    np.abs(block_routes[:, head, 0] - block_routes[:, head, 1]).mean()
                ),
            }
            row["shape_same_minus_different"] = (
                row["attention_shape_same"] - row["attention_shape_different"]
            )
            row["color_same_minus_different"] = (
                row["attention_color_same"] - row["attention_color_different"]
            )
            # the 2x2 conjunction, so a shape effect cannot be a colour effect
            # riding on the correlation between the two in this stimulus set
            for shape_flag in (False, True):
                for color_flag in (False, True):
                    cell = (shape_same == shape_flag) & (color_same == color_flag)
                    row[f"attention_shape{int(shape_flag)}_color{int(color_flag)}"] = (
                        values[cell].mean() if cell.any() else np.nan
                    )
            # distance is the obvious confound: nearby objects attend more
            for tertile in range(3):
                low = -np.inf if tertile == 0 else tertiles[tertile - 1]
                high = np.inf if tertile == 2 else tertiles[tertile]
                cell = (distance >= low) & (distance < high)
                row[f"shape_contrast_distance_tertile{tertile + 1}"] = (
                    values[cell & shape_same].mean() - values[cell & ~shape_same].mean()
                    if (cell & shape_same).any() and (cell & ~shape_same).any()
                    else np.nan
                )
            rows.append(row)

    return pd.DataFrame(rows)


def plot(
    table: pd.DataFrame,
    shape_maps: np.ndarray,
    color_maps: np.ndarray,
    blocks: List[str],
    save_path: Path,
) -> None:
    fig, axs = plt.subplots(1, 4, figsize=(11, 2.4), constrained_layout=True)

    # pick the block with the strongest same-shape routing to illustrate
    per_block = table.groupby("block").shape_same_minus_different.mean()
    best = int(per_block.abs().idxmax())

    for panel, maps, name in (
        (axs[0], shape_maps, "shape"),
        (axs[1], color_maps, "colour"),
    ):
        matrix = maps[best]
        centred = matrix - np.nanmean(matrix)
        limit = np.nanmax(np.abs(centred))
        image = panel.imshow(centred, cmap="RdBu_r", vmin=-limit, vmax=limit)
        panel.set_title(f"{name} map, block {best}", fontsize=6)
        panel.set_xlabel(f"object B {name}" + (" (hue order)" if name == "colour" else ""))
        panel.set_ylabel(f"object A {name}")
        panel.tick_params(labelsize=4)
        fig.colorbar(image, ax=panel, fraction=0.046).ax.tick_params(labelsize=4)

    # every head, so a few strongly tuned heads are not hidden by the mean
    grid = table.pivot_table(
        index="head", columns="block", values="shape_same_minus_different"
    )
    limit = np.nanmax(np.abs(grid.to_numpy()))
    image = axs[2].imshow(grid, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axs[2].set_xlabel("block")
    axs[2].set_ylabel("head")
    axs[2].set_title("same - different shape, per head", fontsize=6)
    fig.colorbar(image, ax=axs[2], fraction=0.046).ax.tick_params(labelsize=4)

    for column, color, label in (
        ("shape_same_minus_different", "black", "shape"),
        ("color_same_minus_different", "tab:blue", "colour"),
    ):
        summary = table.groupby("block")[column].agg(["mean", "std"])
        axs[3].errorbar(
            summary.index, summary["mean"], yerr=summary["std"],
            fmt="o-", ms=3, lw=1, color=color, label=label,
        )
    axs[3].axhline(0, color="gray", lw=0.5, ls=":")
    axs[3].set_xlabel("block")
    axs[3].set_ylabel("same - different attention")
    axs[3].legend(fontsize=5, frameon=False)

    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    cfg = get_cfg(args)
    variant = get_variant(cfg.variant.name)
    variant.set_eval_cfg(cfg)
    variant.set_eval_model(cfg)
    model = variant.eval_model

    save_dir = Path(args.output_dir or cfg.output_dir) / "object_routing"
    save_dir.mkdir(parents=True, exist_ok=True)

    layers = resolve_attention_layers(cfg)
    print(f"reading attention from {len(layers)} blocks: {layers[0]} ... {layers[-1]}")

    discrimination = DatasetRegistry.get("Discrimination_val_attributes")
    indices = stratified_indices(discrimination.labels, args.n_images, args.seed)
    print(f"measuring {len(indices)} of {len(discrimination)} stimuli", flush=True)

    reduced, labels = get_pooled_features_from_layer(
        model,
        discrimination,
        layers,
        route_reducer,
        indices=indices,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        num_workers=args.num_workers,
    )
    routes = np.stack([reduced[layer] for layer in layers])  # (blocks, images, heads, routes)

    table = head_table(routes, labels, layers)
    table.insert(0, "model", cfg.name)
    table.to_csv(save_dir / "routing_by_head.csv", index=False)

    shape_order = np.unique(labels[:, ATTRIBUTE_INDEX["shape0"]]).astype(int)
    color_order = hue_order(labels)
    shape_first = labels[:, ATTRIBUTE_INDEX["shape0"]]
    shape_second = labels[:, ATTRIBUTE_INDEX["shape1"]]
    color_first = labels[:, ATTRIBUTE_INDEX["color0"]]
    color_second = labels[:, ATTRIBUTE_INDEX["color1"]]

    shape_maps, color_maps = [], []
    for block_idx in range(len(layers)):
        pair = 0.5 * (routes[block_idx, :, :, 0] + routes[block_idx, :, :, 1])
        pooled = pair.mean(axis=1)  # average over heads of this block
        shape_maps.append(category_map(pooled, shape_first, shape_second, shape_order))
        color_maps.append(category_map(pooled, color_first, color_second, color_order))
    shape_maps = np.stack(shape_maps)
    color_maps = np.stack(color_maps)

    np.savez(
        save_dir / "category_maps.npz",
        model=cfg.name,
        layers=np.array(layers),
        shape_maps=shape_maps,
        color_maps=color_maps,
        shape_order=shape_order,
        color_order=color_order,
        shape_diagonal_index=np.array([diagonal_index(m) for m in shape_maps]),
        color_diagonal_index=np.array([diagonal_index(m) for m in color_maps]),
    )

    plot(table, shape_maps, color_maps, layers, save_dir / "object_routing.png")

    summary = table.groupby("block")[
        ["mean_pair_attention", "shape_same_minus_different", "color_same_minus_different"]
    ].mean()
    summary["shape_ratio"] = (
        table.groupby("block").attention_shape_same.mean()
        / table.groupby("block").attention_shape_different.mean()
    )
    print("\nper block, averaged over heads:")
    print(summary.round(5).to_string())
    print(f"\nSaved results to {save_dir}")


if __name__ == "__main__":
    main()
