"""Where a region's attention goes, head by head, for two matched stimuli.

A set of tokens is chosen as the *source* -- the four covering one object, or
the leading token of a contour -- and their attention rows are averaged.
What that average puts on a *target* region is the measurement; the map of it
over the whole image is the picture. Two stimuli are shown per dataset, one
of each class, so the question is whether a head redistributes the source's
attention differently depending on the class.

    discrimination, late block: the source is all four tokens of object A and
    the target is object B. If a head computes sameness in its query-key
    product, A should send more attention to B when the two objects match --
    the comparison living in the routing weight itself.

    contour, early block: the source is the token at one end of the contour
    and the target is the rest of the path. If early attention does anything
    like contour integration, one element should reach the others; in the
    contour-absent image the very same cells hold elements at scrambled
    orientations, so attention going there anyway means the effect is about
    position rather than alignment.

How many tokens make up the source is a real choice. An object is read from
all four of its tokens, which averages away whatever happens to sit beside
any one of them. A contour is read from a single leading token by default,
which asks the sharper question -- can one element on the path reach the
whole rest of it -- and accepts the extra noise; ``--contour_source_tokens 0``
recovers the half-and-half version.

Both comparisons are matched rather than arbitrary.

* The contour pair is one seed's present and absent image, which share every
  element position and differ only in the orientation of the path elements.
  The contour's cells are recovered by differencing the two images, so no
  assumption about the generator's geometry is needed, and the *same* source
  and target tokens are used for both.
* The discrimination pair is chosen to match on inter-object distance, since
  nearby objects attend to each other more whatever their categories.

Attention is read from ``blocks.N.attn.attn_matrix``, the post-softmax
matrix. The CLS column and the source tokens themselves are removed and the
row renormalized, so what is plotted is a distribution over the image
*outside* the source -- otherwise a source region's attention to itself, far
larger than anything else, sets the colour scale and everything else renders
as black. Enrichment is then attention mass on the target over what a
uniform row would put there, so 1.0 is chance and the number is comparable
across heads, stimuli and models.

Stimulus coordinates live on the generator's 14x14 patch grid. A model whose
token grid differs -- DINOv2 at patch 14 gives 16x16 -- is handled by mapping
through fractional image coordinates rather than assuming the grids agree.

Two modes:

    default   figures for one model: the stimulus and each head's attention
              map drawn as separate panels, magma, one shared colour scale
    --sweep   no figures; enrichment over many contour pairs and many layers,
              written to a CSV, for finding where an effect is largest

Example:
    python3 analyses/attention_maps.py \\
        --config configs/vitb16a_supervised_imagenet/taskonly.yaml

    python3 analyses/attention_maps.py --sweep --n_pairs 24 \\
        --config configs/vitb16a_mocov3_imagenet/taskonly.yaml
"""

import argparse
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
from PIL import Image

from spacetorch.configs import get_cfg
from spacetorch.datasets import DatasetRegistry
from spacetorch.datasets.discrimination import (
    ATTRIBUTE_INDEX,
    OBJECT_PATCHES,
    PATCH_GRID,
)
from spacetorch.feature_extractor import get_features_from_layer
from spacetorch.variants import get_variant


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/vitb16a_supervised_imagenet/taskonly.yaml",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        choices=["contour", "discrimination"],
        default=["contour", "discrimination"],
    )
    parser.add_argument(
        "--contour_layer",
        type=str,
        default="blocks.0.attn",
        help="contour integration, if it is anywhere, is early: sweeping "
        "every block, the contour-minus-absent effect is largest at block 0 "
        "in all three patch-16 models and is gone by block 7",
    )
    parser.add_argument(
        "--discrimination_layer",
        type=str,
        default="blocks.11.attn",
        help="the same/different routing peaks in the middle, not at the end: "
        "sweeping every block of the supervised ViT, the mean same-minus-"
        "different enrichment rises to 8.2 at block 5 and is 0.2 by block 11, "
        "turning negative at blocks 8-10",
    )
    parser.add_argument(
        "--contour_pair",
        type=int,
        default=-1,
        help="which contour pair to draw; -1 picks the longest, most tightly "
        "spaced, most central contour, where integration is most likely to "
        "be visible at all",
    )
    parser.add_argument(
        "--contour_source_tokens",
        type=int,
        default=1,
        help="how many tokens from one end of the contour form the source; "
        "the rest of the path is the target. 0 splits the path in half",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="skip the figures and measure enrichment over many contour pairs "
        "and many layers, to find where the effect is largest",
    )
    parser.add_argument(
        "--sweep_layers",
        type=str,
        nargs="+",
        default=None,
        help="layers for --sweep; defaults to every block of the model",
    )
    parser.add_argument(
        "--n_pairs",
        type=int,
        default=24,
        help="contour pairs measured per layer in --sweep, taken from the "
        "most favourable end of the design",
    )
    parser.add_argument(
        "--n_discrimination",
        type=int,
        default=24,
        help="distance-matched same/different image pairs measured per layer "
        "in --sweep",
    )
    parser.add_argument("--cmap", type=str, default="magma")
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="fix the shared colour scale, so figures from different models "
        "and layers are directly comparable. Default picks the "
        "--vmax_percentile of everything in the figure",
    )
    parser.add_argument(
        "--vmax_percentile",
        type=float,
        default=99.5,
        help="percentile of all plotted values used for the shared scale; a "
        "plain maximum lets the single most peaked head set the scale for "
        "every panel",
    )
    parser.add_argument(
        "--keep_source",
        dest="exclude_source",
        action="store_false",
        default=True,
        help="leave the source tokens in the distribution. Off by default: a "
        "region attends to itself far more than to anything else, so keeping "
        "it renders every other panel black",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
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


class Probe:
    """One stimulus, its class, and the regions the measurement is between.

    ``source_cells`` and ``target_cells`` are on the generator's 14x14 patch
    grid, not the model's token grid; the translation happens once the model
    is known.
    """

    def __init__(
        self,
        dataset,
        index: int,
        label: int,
        class_name: str,
        source_cells: Sequence[int],
        target_cells: Sequence[int],
        target_name: str,
    ):
        self.dataset = dataset
        self.index = index
        self.label = label
        self.class_name = class_name
        self.source_cells = np.array(sorted(set(int(c) for c in source_cells)))
        self.target_cells = np.array(
            sorted(set(int(c) for c in target_cells) - set(self.source_cells.tolist()))
        )
        self.target_name = target_name

    @property
    def image(self) -> np.ndarray:
        return np.asarray(
            Image.open(self.dataset.file_list[self.index]).convert("RGB")
        )


def cell_of(row: int, column: int, grid: int = PATCH_GRID) -> int:
    return row * grid + column


def cells_to_tokens(cells: np.ndarray, grid: int) -> np.ndarray:
    """Stimulus-grid cells to model token indices, CLS included in the offset.

    The stimuli are laid out on a 14x14 grid of 16-pixel patches. A model
    with a different patch size tiles the same image differently, so each
    stimulus cell is converted through fractional image coordinates to
    whichever model tokens overlap it.
    """
    tokens = set()
    for cell in np.atleast_1d(cells):
        row, column = divmod(int(cell), PATCH_GRID)
        rows = range(
            int(np.floor(row / PATCH_GRID * grid)),
            max(int(np.ceil((row + 1) / PATCH_GRID * grid)),
                int(np.floor(row / PATCH_GRID * grid)) + 1),
        )
        columns = range(
            int(np.floor(column / PATCH_GRID * grid)),
            max(int(np.ceil((column + 1) / PATCH_GRID * grid)),
                int(np.floor(column / PATCH_GRID * grid)) + 1),
        )
        for r in rows:
            for c in columns:
                if 0 <= r < grid and 0 <= c < grid:
                    tokens.add(1 + r * grid + c)
    return np.array(sorted(tokens), dtype=int)


def object_cells(label: np.ndarray, object_idx: int) -> np.ndarray:
    """The 2x2 token window one object occupies."""
    top = int(label[ATTRIBUTE_INDEX[f"block{object_idx}_y"]])
    left = int(label[ATTRIBUTE_INDEX[f"block{object_idx}_x"]])
    return np.array(
        [
            cell_of(top + dy, left + dx)
            for dy in range(OBJECT_PATCHES)
            for dx in range(OBJECT_PATCHES)
        ]
    )


def discrimination_probes(n_pairs: int = 1) -> Tuple[List[Probe], str]:
    """Same and different images, matched on inter-object distance.

    The attribute dataset is used rather than ``Discrimination_val`` because
    the object coordinates are needed to choose the source tokens, and
    because it is the version that leaves the image at 224x224 -- the resize
    and crop the plain loader applies would move the objects off the patch
    grid the coordinates refer to.
    """
    dataset = DatasetRegistry.get("Discrimination_val_attributes")
    labels = dataset.labels
    is_same = labels[:, ATTRIBUTE_INDEX["sd_label"]].astype(int)
    distance = labels[:, ATTRIBUTE_INDEX["distance_patches"]]

    same_rows = np.flatnonzero(is_same == 1)
    different_rows = np.flatnonzero(is_same == 0)

    # order the "same" images by how typical their separation is, then pair
    # each with the closest-matching "different" one: distance drives pair
    # attention on its own and would otherwise confound the comparison
    median = float(np.median(distance))
    order = same_rows[np.argsort(np.abs(distance[same_rows] - median))]

    probes, used = [], set()
    for same_row in order[:n_pairs]:
        candidates = [r for r in different_rows if r not in used]
        different_row = int(
            candidates[
                int(np.argmin(np.abs(distance[candidates] - distance[same_row])))
            ]
        )
        used.add(different_row)
        for row, name in ((different_row, "different"), (int(same_row), "same")):
            label = labels[row]
            probes.append(
                Probe(
                    dataset=dataset,
                    index=row,
                    label=int(is_same[row]),
                    class_name=name,
                    source_cells=object_cells(label, 0),
                    target_cells=object_cells(label, 1),
                    target_name="object B",
                )
            )
    return probes, "object A (4 tokens) -> object B"


def contour_cells_from_pair(dataset, pair: int) -> Tuple[np.ndarray, np.ndarray]:
    """Which tokens hold the contour, by differencing the pair.

    The present and absent images of a seed are identical except that the
    path elements are rotated, so the pixels that change are exactly the
    contour's elements. Reading the location off the images avoids having to
    reconstruct it from eccentricity, orientation and side, which would be a
    guess about the generator's conventions rather than a measurement.
    """
    present = np.asarray(
        Image.open(dataset.file_list[2 * pair]).convert("RGB"), dtype=float
    )
    absent = np.asarray(
        Image.open(dataset.file_list[2 * pair + 1]).convert("RGB"), dtype=float
    )
    difference = np.abs(present - absent).sum(axis=-1)

    per_cell = difference.reshape(
        PATCH_GRID,
        difference.shape[0] // PATCH_GRID,
        PATCH_GRID,
        difference.shape[1] // PATCH_GRID,
    ).sum(axis=(1, 3))
    assert per_cell.max() > 0, (
        f"contour pair {pair}: the two images are identical, so the contour "
        "cannot be located"
    )

    rows, columns = np.nonzero(per_cell > 0.05 * per_cell.max())
    return np.array([cell_of(r, c) for r, c in zip(rows, columns)]), per_cell


def split_contour(cells: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The leading token(s) of the path as source, the rest as target.

    Cells are ordered along the contour's own principal axis rather than by
    row or column, so "leading" means one end of the path whatever its
    orientation. ``--contour_source_tokens`` sets how many of them are the
    source; the default of one asks the sharpest version of the question --
    can a single element on the path reach the whole of the rest of it --
    at the cost of being hostage to whatever happens to sit beside that one
    token. A value of 0 splits the path in half instead, which averages that
    away but halves the target.
    """
    coordinates = np.array([divmod(int(c), PATCH_GRID) for c in cells], dtype=float)
    centred = coordinates - coordinates.mean(axis=0)
    # first principal axis of the element positions is the contour's direction
    _, _, components = np.linalg.svd(centred, full_matrices=False)
    projection = centred @ components[0]

    order = np.argsort(projection)
    n_source = (
        len(cells) // 2
        if args.contour_source_tokens <= 0
        else min(args.contour_source_tokens, len(cells) - 1)
    )
    return cells[order[:n_source]], cells[order[n_source:]]


def contour_pair_scores(dataset) -> np.ndarray:
    """Rank pairs by how favourable they are to integration.

    Long, tightly spaced, central contours first: that is where an effect
    should appear if it appears anywhere, so a sweep spends its budget there
    rather than on two-token paths out in the periphery.
    """
    length = dataset.factor("contour_length_achieved")[0::2]
    spacing = dataset.factor("relative_spacing")[0::2]
    eccentricity = dataset.factor("eccentricity_deg")[0::2]
    return length - 3.0 * spacing - 0.5 * eccentricity


def contour_probes(pair: int, dataset=None) -> Tuple[List[Probe], str]:
    """A contour-present image and its own absent partner, same tokens.

    The dataset is passed in rather than fetched, so that probes built for
    different pairs all index into one instance -- ``read_attention`` wraps
    them in a single ``Subset``, which is only meaningful if they do.
    """
    dataset = dataset if dataset is not None else DatasetRegistry.get("Contour_val")
    cells, _ = contour_cells_from_pair(dataset, pair)
    if len(cells) < 4:
        return [], ""

    source, target = split_contour(cells)
    probes = []
    for offset, name in ((1, "no contour"), (0, "contour")):
        index = 2 * pair + offset
        probes.append(
            Probe(
                dataset=dataset,
                index=index,
                label=int(dataset.labels[index]),
                class_name=name,
                source_cells=source,
                target_cells=target,
                target_name="rest of the contour",
            )
        )
    return probes, (
        f"{len(source)} leading contour token{'' if len(source) == 1 else 's'}"
        " -> the rest of the contour"
    )


def read_attention(model, probes: List[Probe], layers: List[str]) -> Dict[str, np.ndarray]:
    """Post-softmax attention for the probe stimuli, per layer."""
    dataset = probes[0].dataset
    assert all(probe.dataset is dataset for probe in probes), (
        "probes index into different dataset instances; they are wrapped in "
        "one Subset, so they must share one"
    )

    strings = [f"{layer}.attn_matrix" for layer in layers]
    features, _, _ = get_features_from_layer(
        model,
        torch.utils.data.Subset(dataset, [probe.index for probe in probes]),
        model_layer_strings=strings,
        batch_size=args.batch_size,
        return_inputs_and_labels=True,
        shuffle=False,
        verbose=False,
    )
    if len(strings) == 1:
        features = {strings[0]: features}

    out = {}
    for layer, string in zip(layers, strings):
        attention = np.asarray(features[string], dtype=np.float32)
        assert attention.ndim == 4, (
            f"expected (batch, heads, tokens, tokens), got {attention.shape}; "
            f"`{string}` did not return an attention matrix"
        )
        out[layer] = attention
    return out


def token_grid(attention: np.ndarray) -> int:
    """Side of the model's square patch-token grid."""
    n_patches = attention.shape[-1] - 1
    grid = int(round(math.sqrt(n_patches)))
    assert grid * grid == n_patches, (
        f"{n_patches} patch tokens is not a square grid; this analysis assumes "
        "one CLS token followed by a square patch grid"
    )
    return grid


def source_rows(attention: np.ndarray, source_tokens: np.ndarray) -> np.ndarray:
    """Each head's attention from the source region, averaged over its tokens.

    The CLS column goes, and by default the source tokens go too, before the
    row is renormalized: what is left is a distribution over the rest of the
    image, which is the thing worth looking at. Leaving the source in lets a
    region's attention to itself -- always the largest entry by a wide
    margin -- set the colour scale for every panel.
    """
    rows = attention[:, source_tokens, 1:].mean(axis=1)
    if args.exclude_source:
        rows = rows.copy()
        rows[:, source_tokens - 1] = 0.0
    return rows / np.maximum(rows.sum(axis=1, keepdims=True), 1e-12)


def enrichment(row: np.ndarray, target_tokens: np.ndarray) -> float:
    """Attention mass on the target over what a uniform row would put there."""
    if len(target_tokens) == 0:
        return float("nan")
    share = len(target_tokens) / np.count_nonzero(row >= 0)
    return float(row[target_tokens - 1].sum() / share)


def measure(
    attention: np.ndarray, probes: List[Probe], layer: str
) -> List[Dict[str, object]]:
    """Enrichment for every probe and head at one layer."""
    grid = token_grid(attention)
    records = []
    for index, probe in enumerate(probes):
        source = cells_to_tokens(probe.source_cells, grid)
        target = cells_to_tokens(probe.target_cells, grid)
        rows = source_rows(attention[index], source)
        for head in range(rows.shape[0]):
            records.append(
                {
                    "layer": layer,
                    "block": int(layer.split(".")[1]),
                    "class": probe.class_name,
                    "label": probe.label,
                    "image_index": probe.index,
                    "head": head,
                    "enrichment": enrichment(rows[head], target),
                    "target_mass": float(rows[head][target - 1].sum()),
                    "n_source_tokens": len(source),
                    "n_target_tokens": len(target),
                }
            )
    return records


def figure_for(
    probes: List[Probe], attention: np.ndarray, layer: str, title: str, save_path: Path
):
    """Two rows, one per class; the stimulus and then one panel per head.

    The stimulus is drawn on its own, unannotated, and each head's attention
    is its own panel. Every attention panel shares one colour scale, so a
    difference in brightness between the two rows is a difference in
    attention and not a difference in normalisation.
    """
    import pandas as pd

    grid = token_grid(attention)
    n_heads = attention.shape[1]

    maps, targets = [], []
    for index, probe in enumerate(probes):
        source = cells_to_tokens(probe.source_cells, grid)
        targets.append(cells_to_tokens(probe.target_cells, grid))
        maps.append(source_rows(attention[index], source))

    stacked = np.concatenate([row.ravel() for row in maps])
    vmax = args.vmax or float(np.percentile(stacked, args.vmax_percentile))

    fig, axs = plt.subplots(
        len(probes),
        n_heads + 1,
        figsize=(1.2 * (n_heads + 1), 1.45 * len(probes)),
        constrained_layout=True,
    )
    axs = np.atleast_2d(axs)

    records = []
    for panel, probe in enumerate(probes):
        axs[panel, 0].imshow(probe.image)
        axs[panel, 0].set_ylabel(probe.class_name, fontsize=8)
        axs[panel, 0].set_xticks([])
        axs[panel, 0].set_yticks([])
        if panel == 0:
            axs[panel, 0].set_title("stimulus", fontsize=7)

        for head in range(n_heads):
            axis = axs[panel, head + 1]

            white_red = LinearSegmentedColormap.from_list(
                "white_red",
                ["white", "red", "black"],
            )

            handle = axis.imshow(
                maps[panel][head].reshape(grid, grid),
                cmap=white_red,
                vmin=0.0,
                vmax=vmax,
                interpolation="nearest",
            )

            value = enrichment(maps[panel][head], targets[panel])
            axis.set_xlabel(f"{value:.2f}x", fontsize=6)
            axis.set_xticks([])
            axis.set_yticks([])
            if panel == 0:
                axis.set_title(f"head {head}", fontsize=7)

            records.append(
                {
                    "layer": layer,
                    "class": probe.class_name,
                    "label": probe.label,
                    "image_index": probe.index,
                    "head": head,
                    "enrichment": value,
                    "target_mass": float(maps[panel][head][targets[panel] - 1].sum()),
                }
            )

    bar = fig.colorbar(handle, ax=axs[:, 1:].ravel().tolist(), fraction=0.012, pad=0.01)
    bar.ax.tick_params(labelsize=5)
    bar.set_label("attention (shared scale)", fontsize=6)
    fig.suptitle(
        f"{title}, {layer}   "
        f"(numbers are attention on the {probes[0].target_name} over chance; "
        f"shared vmax {vmax:.4f})",
        fontsize=8,
    )
    fig.savefig(save_path, dpi=300, bbox_inches="tight", format="svg")
    plt.close(fig)
    return pd.DataFrame(records)


def contrast_table(table):
    """Per layer and head, the class difference in enrichment."""
    wide = table.pivot_table(
        index=["block", "layer", "head"], columns="class", values="enrichment"
    )
    classes = list(wide.columns)
    positive = "same" if "same" in classes else "contour"
    negative = [c for c in classes if c != positive][0]
    # `difference` and `head` collide with pandas methods, so attribute
    # access on them silently returns a bound method; always index by key
    wide["class_difference"] = wide[positive] - wide[negative]
    wide["ratio"] = wide[positive] / wide[negative].replace(0, np.nan)
    return wide.reset_index(), positive, negative


def run_sweep(model, cfg, save_dir: Path):
    """Enrichment over many stimuli and layers, to find where the effect lives."""
    import pandas as pd

    layers = args.sweep_layers
    if layers is None:
        from spacetorch.variants.positions import BRAIN_MAPPING

        keys = BRAIN_MAPPING[cfg.variant.architecture]
        layers = [k if k.endswith(".attn") else f"{k}.attn" for k in keys]
        layers = sorted(set(layers), key=lambda s: int(s.split(".")[1]))
    print(f"sweeping {len(layers)} layers: {layers[0]} ... {layers[-1]}")

    frames = []
    for name in args.datasets:
        if name == "contour":
            dataset = DatasetRegistry.get("Contour_val")
            ranked = np.argsort(-contour_pair_scores(dataset))
            probes = []
            for pair in ranked:
                built, title = contour_probes(int(pair), dataset)
                probes.extend(built)
                if len(probes) >= 2 * args.n_pairs:
                    break
            probes = probes[: 2 * args.n_pairs]
        else:
            probes, title = discrimination_probes(args.n_discrimination)

        print(f"{name}: {len(probes)} stimuli, {title}", flush=True)
        attention = read_attention(model, probes, layers)
        for layer in layers:
            records = measure(attention[layer], probes, layer)
            frame = pd.DataFrame(records)
            frame.insert(0, "dataset", name)
            frames.append(frame)
            del records

    table = pd.concat(frames, ignore_index=True)
    table.insert(0, "model", cfg.name)
    path = save_dir / "attention_sweep.csv"
    table.to_csv(path, index=False)

    for name in args.datasets:
        subset = table[table.dataset == name]
        wide, positive, negative = contrast_table(subset)
        print(f"\n=== {name}: {positive} minus {negative}, mean over stimuli ===")
        by_layer = wide.groupby("block")["class_difference"].agg(["mean", "max"])
        print(by_layer.round(4).to_string())
        best = wide.loc[wide["class_difference"].idxmax()]
        print(
            f"best: block {int(best['block'])} head {int(best['head'])}  "
            f"{positive} {best[positive]:.3f} vs {negative} {best[negative]:.3f}  "
            f"(difference {best['class_difference']:+.3f}, "
            f"ratio {best['ratio']:.2f}x)"
        )

    print(f"\nSaved sweep to {path}")
    return table


def main():
    cfg = get_cfg(args)
    variant = get_variant(cfg.variant.name)
    variant.set_eval_cfg(cfg)
    variant.set_eval_model(cfg)
    model = variant.eval_model

    save_dir = Path(args.output_dir or cfg.output_dir) / "attention_maps"
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.sweep:
        run_sweep(model, cfg, save_dir)
        return

    import pandas as pd

    tables = []
    for name in args.datasets:
        if name == "contour":
            dataset = DatasetRegistry.get("Contour_val")
            pair = args.contour_pair
            if pair < 0:
                pair = int(np.argmax(contour_pair_scores(dataset)))
                print(
                    f"contour: pair {pair} — length "
                    f"{dataset.factor('contour_length_achieved')[2 * pair]:g}, "
                    f"spacing {dataset.factor('relative_spacing')[2 * pair]:g}, "
                    f"eccentricity {dataset.factor('eccentricity_deg')[2 * pair]:g}"
                )
            probes, title = contour_probes(pair, dataset)
            assert probes, f"contour pair {pair} has too few contour tokens to split"
            layer = args.contour_layer
        else:
            probes, title = discrimination_probes(1)
            layer = args.discrimination_layer

        attention = read_attention(model, probes, [layer])[layer]
        print(f"{name}: attention {attention.shape} from {layer}, "
              f"{len(probes[0].source_cells)} source cells, "
              f"{len(probes[0].target_cells)} target cells")

        # the layer is in the filename so that runs at different blocks sit
        # side by side instead of overwriting each other
        table = figure_for(
            probes, attention, layer, title,
            save_dir / f"attention_{name}_{layer.replace('.', '_')}.svg",
        )
        table.insert(0, "dataset", name)
        table.insert(0, "model", cfg.name)
        tables.append(table)

        wide = table.pivot_table(index="head", columns="class", values="enrichment")
        print(f"\n{name}: target enrichment per head")
        print(wide.round(3).to_string())

    combined = pd.concat(tables, ignore_index=True)
    suffix = "_".join(sorted(set(combined.layer))).replace(".", "_")
    combined.to_csv(save_dir / f"attention_enrichment_{suffix}.csv", index=False)
    print(f"\nSaved figures and attention_enrichment.csv to {save_dir}")


if __name__ == "__main__":
    main()
