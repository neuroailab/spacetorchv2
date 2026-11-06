"""Object shape and colour preference laid out on the simulated cortical sheet.

``v1_map`` gives every unit a preferred orientation from full-field gratings and
plots it over the tissue. The objects in the discrimination stimuli support the
same construction, with one complication: a grating covers the whole image, so
every unit sees it, whereas an object covers four patches and moves from image
to image. A unit's object tuning therefore has to be estimated only from the
images in which an object actually fell inside its receptive field.

That is what ``accumulate_object_tuning`` does. Sweeping the stimuli once, it
adds each unit's response into a bin indexed by the shape and the colour of
whichever object covered that unit, so the per-unit tuning curve is over the
objects the unit actually saw. Units here are channel/position pairs, the
convention the tissue maps use, not the channel-level axes the privileged-axis
analyses use.

Because every object occupies a fixed fraction of the image and positions are
jittered over a 13 x 13 grid, a given site is covered in roughly 5% of images.
With the full 6,400-image split that leaves a few hundred samples per site and
roughly twenty per category, which is enough for an argmax but not for a
precise tuning curve -- hence the selectivity mask before anything is read off
the map.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

from spacetorch.datasets.discrimination import (
    ATTRIBUTE_INDEX,
    OBJECT_PATCHES,
    PATCH_GRID,
    parse_color_name,
)
from spacetorch.maps import TissueMap
from spacetorch.maps.smoother import KernelParams, Smoother
from spacetorch.privileged_axes import _fractional_slice
from spacetorch.utils.array_utils import lower_tri, midpoints_from_bin_edges
from spacetorch.utils.spatial_utils import agg_by_distance

PROPERTIES = ["shape", "color"]


@dataclass
class ObjectMetric:
    """How one object property is named, coloured, ordered and matched.

    ``ranks`` carries a position on a *cyclic* axis when the property has one.
    Hue does: the sixteen colour categories lie on a circle, so "the same
    colour" can sensibly admit a tolerance of one neighbouring category, the
    way the orientation analyses admit 22.5 degrees. Shape has no such axis --
    the shape clusters come from k-means on silhouettes and no ordering of
    them is canonical -- so ``ranks`` is None, matching is exact, and the map
    is a categorical patch map rather than a smooth one.

    This is what fixes each property's chance level by construction rather
    than by estimate, exactly as the orientation analyses declare 3 of 8
    bins = 0.375 from the binning alone.
    """

    name: str
    n_unique: int
    colors: np.ndarray  # (n_unique, 4) RGBA, one per category
    xlabel: str
    ranks: Optional[np.ndarray] = None  # position on a cyclic axis, if any
    tolerance: int = 0  # neighbouring ranks that still count as iso

    @property
    def declared_chance(self) -> float:
        """Iso fraction expected by chance, read off the binning."""
        return (2 * self.tolerance + 1) / self.n_unique

    @property
    def display_order(self) -> np.ndarray:
        """Categories in the order a legend should show them."""
        if self.ranks is None:
            return np.arange(self.n_unique)
        return np.argsort(self.ranks)

    def iso(self, preferences: np.ndarray, presented: int) -> np.ndarray:
        """Which preferences count as matching the presented category.

        Exact identity when the property is unordered; within ``tolerance``
        steps around the circle when it is cyclic.
        """
        if self.ranks is None:
            return np.asarray(preferences) == presented

        delta = np.abs(self.ranks[np.asarray(preferences)] - self.ranks[presented])
        return np.minimum(delta, self.n_unique - delta) <= self.tolerance


def shape_metric(n_shapes: int) -> ObjectMetric:
    """Shapes have no natural ordering, so a qualitative palette is used."""
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab20")
    colors = np.array([cmap(i / max(n_shapes - 1, 1)) for i in range(n_shapes)])
    return ObjectMetric("shape", n_shapes, colors, "preferred shape")


def hue_ranks(color_names: List[str]) -> np.ndarray:
    """Each colour category's position on the hue circle.

    The category indices are assigned by sorting colour *names*, which is
    arbitrary. Sorted by hue they become an axis along which "nearby" means
    perceptually similar, which is what lets the colour map be read as a
    smooth one and lets a one-category tolerance mean something.
    """
    import colorsys

    rgb = np.array([parse_color_name(name) for name in color_names]) / 255.0
    hues = np.array([colorsys.rgb_to_hsv(*triple)[0] for triple in rgb])
    order = np.argsort(hues)
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(len(order))
    return ranks


def color_metric(color_names: List[str], tolerance: int = 1) -> ObjectMetric:
    """Colour categories are drawn in their own colour, ordered by hue.

    The generator recorded each object's mean RGB, so a unit can be painted
    with the colour it actually prefers. That makes the map readable without a
    legend, and it means a smooth progression across the sheet is visible as a
    smooth progression of hue.

    Hue is a full 360-degree circle -- unlike orientation, which is axial over
    180 and needs angle doubling -- so the default tolerance of one
    neighbouring category puts the declared chance level at 3/16 = 0.1875.
    """
    rgb = np.array([parse_color_name(name) for name in color_names]) / 255.0
    colors = np.concatenate([rgb, np.ones((len(rgb), 1))], axis=1)
    return ObjectMetric(
        "color",
        len(color_names),
        colors,
        "preferred colour",
        ranks=hue_ranks(color_names),
        tolerance=tolerance,
    )


def _square_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Tight bounding box of a mask, padded out to a square.

    Squaring it before resizing keeps the aspect ratio, so a tall shape does
    not get stretched into a fat one on the way to the thumbnail.
    """
    rows, cols = np.nonzero(mask)
    top, bottom = rows.min(), rows.max() + 1
    left, right = cols.min(), cols.max() + 1

    side = max(bottom - top, right - left)
    centre_y, centre_x = (top + bottom) / 2, (left + right) / 2
    top = int(round(centre_y - side / 2))
    left = int(round(centre_x - side / 2))
    return top, top + side, left, left + side


def build_object_thumbnails(
    dataset=None,
    n_examples: int = 60,
    resolution: int = 256,
    cache=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """One RGBA thumbnail per shape and per colour, taken from the stimuli.

    Shape thumbnails average the foreground mask over many instances, which
    both suppresses the per-pixel colour noise and antialiases the edge. The
    instances have to be *aligned* first: the generator jitters each object
    inside its 2x2 token window, so averaging the raw crops averages a spread
    of offsets and smears the silhouette. Each instance is therefore cropped
    to its own square bounding box before being resized onto a common canvas.

    The object is only about twenty pixels across in the source image, so the
    resolution here buys clean, correctly-scaled edges rather than detail that
    was never recorded.

    Colour thumbnails are flat swatches of the generating RGB -- the colour
    with shape marginalized out, as the silhouette is the shape with colour
    marginalized out.

    These exist so that a figure can name a category by showing it. A
    sixteen-category qualitative palette is not readable as a colour bar, so
    both the routing heatmap's tick labels and the tissue maps' legends use
    the objects themselves instead of an index.
    """
    from pathlib import Path

    from spacetorch.datasets import DatasetRegistry

    cache = Path(cache) if cache is not None else None
    if cache is not None and cache.exists():
        stored = np.load(cache)
        # a cache written before the resolution was recorded, or at a
        # different resolution, is rebuilt rather than trusted
        if "resolution" in stored and int(stored["resolution"]) == resolution:
            return stored["shapes"], stored["colors"]

    from PIL import Image

    if dataset is None:
        dataset = DatasetRegistry.get("Discrimination_val_attributes")
    labels = dataset.labels
    n_shapes = int(labels[:, ATTRIBUTE_INDEX["shape0"]].max()) + 1

    patch_px = 16  # the generator's patch size, so an object is 32 x 32 pixels
    window = OBJECT_PATCHES * patch_px
    # average at a modest size, then upsample once: resizing every 20-pixel
    # instance straight to 256 would just interpolate noise
    working = 64
    masks = np.zeros((n_shapes, working, working))
    counts = np.zeros(n_shapes)

    for shape in range(n_shapes):
        # look at both object slots, so every shape is found quickly
        for object_idx in (0, 1):
            rows = np.flatnonzero(
                labels[:, ATTRIBUTE_INDEX[f"shape{object_idx}"]] == shape
            )
            for row in rows[: n_examples // 2]:
                image = np.asarray(
                    Image.open(dataset.file_list[row]).convert("RGB"), dtype=np.float32
                )
                top = int(labels[row, ATTRIBUTE_INDEX[f"block{object_idx}_y"]]) * patch_px
                left = int(labels[row, ATTRIBUTE_INDEX[f"block{object_idx}_x"]]) * patch_px
                crop = image[top : top + window, left : left + window]

                # the background is white by construction
                mask = (np.abs(crop - 255.0).sum(axis=-1) > 20).astype(np.float32)
                if not mask.any():
                    continue

                r0, r1, c0, c1 = _square_bbox(mask)
                padded = np.zeros((r1 - r0, c1 - c0), dtype=np.float32)
                sr0, sr1 = max(r0, 0), min(r1, mask.shape[0])
                sc0, sc1 = max(c0, 0), min(c1, mask.shape[1])
                padded[sr0 - r0 : sr1 - r0, sc0 - c0 : sc1 - c0] = mask[sr0:sr1, sc0:sc1]

                masks[shape] += np.asarray(
                    Image.fromarray(padded, mode="F").resize(
                        (working, working), Image.BILINEAR
                    )
                )
                counts[shape] += 1

    masks /= np.maximum(counts, 1)[:, None, None]

    shape_images = np.zeros((n_shapes, resolution, resolution, 4))
    for shape in range(n_shapes):
        alpha = np.asarray(
            Image.fromarray(masks[shape].astype(np.float32), mode="F").resize(
                (resolution, resolution), Image.BICUBIC
            )
        )
        # crisp the boundary: aligned instances agree almost everywhere, so
        # the soft edge is interpolation rather than genuine uncertainty
        alpha = np.clip((alpha - 0.45) * 6.0 + 0.5, 0.0, 1.0)
        shape_images[shape, ..., :3] = 0.15  # near-black ink
        shape_images[shape, ..., 3] = alpha

    rgb = np.array([parse_color_name(name) for name in dataset.color_names]) / 255.0
    color_images = np.ones((len(rgb), resolution, resolution, 4))
    color_images[..., :3] = rgb[:, None, None, :]

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache, shapes=shape_images, colors=color_images, resolution=resolution
        )

    return shape_images, color_images


def thumbnail_zoom(resolution: int, points: float) -> float:
    """``OffsetImage`` zoom that renders a thumbnail at a fixed size in points.

    ``OffsetImage`` draws the array at ``resolution * zoom`` points, so the
    zoom has to shrink as the resolution grows -- otherwise a higher
    resolution just draws a bigger picture instead of a sharper one of the
    same size. There is no dpi term: the dpi correction inside matplotlib
    converts points to device pixels and cancels out.
    """
    return points / resolution


def category_filmstrip(
    metric: "ObjectMetric", images: Optional[np.ndarray] = None, cell: int = 48
) -> np.ndarray:
    """A vertical strip with one cell per category, for use as a legend.

    Each cell is the colour the map paints that category, and -- when
    ``images`` is given -- the category's own thumbnail composited on top in
    whichever of black or white reads against that colour. A viewer can then
    match a colour on the map to the shape it stands for by eye, which a
    sixteen-entry qualitative colour bar labelled ``0`` and ``15`` does not
    allow.

    Cells follow the metric's display order, so the colour legend runs around
    the hue circle rather than through the arbitrary order the category
    indices happen to carry.
    """
    from PIL import Image

    strip = np.ones((metric.n_unique * cell, cell, 4))
    for position, category in enumerate(metric.display_order):
        tile = np.ones((cell, cell, 4))
        tile[..., :3] = metric.colors[category][:3]

        if images is not None:
            glyph = images[int(category)]
            alpha = np.asarray(
                Image.fromarray(glyph[..., 3].astype(np.float32), mode="F").resize(
                    (cell, cell), Image.BILINEAR
                )
            )[..., None]
            luminance = float(np.dot(metric.colors[category][:3], [0.299, 0.587, 0.114]))
            ink = 0.0 if luminance > 0.5 else 1.0
            tile[..., :3] = tile[..., :3] * (1 - alpha) + ink * alpha

        strip[position * cell : (position + 1) * cell] = tile
    return strip


def smoothing_kernel(
    positions: np.ndarray,
    width: Optional[float] = None,
    stride: Optional[float] = None,
) -> KernelParams:
    """The orientation-map kernel, expressed relative to the sheet.

    ``lateral.ipynb`` smooths orientation preference with a 1.5 mm window
    stepped by 0.15 mm over a 36.75 mm sheet. Those numbers are not portable:
    a layer whose positions span a different range would be smoothed by a
    quite different fraction of itself. The defaults here are that same
    kernel as a proportion -- 1.5/36.75 of the extent, stepped by a tenth of
    its own width -- so a familiar sheet reproduces the familiar picture and
    an unfamiliar one is still smoothed sensibly.
    """
    extent = float(max(np.ptp(positions[:, 0]), np.ptp(positions[:, 1])))
    width = 1.5 / 36.75 * extent if width is None else width
    stride = width / 10.0 if stride is None else stride
    return KernelParams(width=width, stride=stride)


def smoothed_preference_image(
    positions: np.ndarray,
    preferences: np.ndarray,
    metric: "ObjectMetric",
    kernel: Optional[KernelParams] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, Smoother]:
    """Modal preferred category in a sliding window, as an RGBA image.

    The categorical counterpart of ``get_smoothed_map``. Orientation is
    circular and smooths with ``circmean``; shape and colour categories have
    no metric, so the aggregate is the *mode* -- the category most units in
    the window prefer -- which is then painted with the map's palette. The
    result is an image rather than a scalar field with a colormap, since a
    category index is not a value a colour bar can interpolate.

    Windows holding no units come back as NaN and are left transparent.

    The :class:`Smoother` is returned alongside the image because the caller
    needs it: ``positions_to_pixel_coords`` maps unit positions onto this
    exact pixel grid, which is how boutons end up in the right place on top.
    """
    kernel = kernel or smoothing_kernel(positions)
    smoother = Smoother(kernel, verbose=verbose)
    modal = smoother(
        SimpleNamespace(positions=positions),
        lambda _: np.asarray(preferences, dtype=int),
        agg_mode="mode",
    )

    image = np.ones(modal.shape + (4,))
    valid = np.isfinite(modal)
    image[valid] = metric.colors[modal[valid].astype(int)]
    image[~valid, 3] = 0.0
    return image, smoother


def smoothed_scalar_image(
    positions: np.ndarray,
    values: np.ndarray,
    kernel: Optional[KernelParams] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, Smoother]:
    """Mean of a scalar over the same sliding window, as a 2-D array.

    The companion of :func:`smoothed_preference_image` for quantities that
    are continuous rather than categorical -- most usefully the indicator
    "does this unit prefer the presented category", whose window mean is the
    local density of iso-preferring units. That density is a far better
    substrate for boutons than a sixteen-category patch map: a reader can
    see whether ink concentrates on light or dark, which is a judgement the
    eye can make, whereas matching black dots against sixteen hues is not.
    """
    kernel = kernel or smoothing_kernel(positions)
    smoother = Smoother(kernel, verbose=verbose)
    image = smoother(
        SimpleNamespace(positions=positions),
        lambda _: np.asarray(values, dtype=float),
        agg_mode="mean",
    )
    return image, smoother


def category_legend(
    axis,
    metric: "ObjectMetric",
    images: Optional[np.ndarray] = None,
    size: str = "7%",
    pad: float = 0.04,
    fontsize: float = 6,
):
    """Attach a :func:`category_filmstrip` to the right of ``axis``.

    Drawn as an image in its own axes rather than with ``OffsetImage`` ticks,
    so the glyphs scale with the panel instead of needing a point size that
    happens to fit.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size=size, pad=pad)
    cax.imshow(
        category_filmstrip(metric, images), aspect="auto", origin="lower",
        interpolation="nearest",
    )
    cax.set_xticks([])
    cax.set_yticks([])
    for spine in cax.spines.values():
        spine.set_linewidth(0.3)
    cax.set_ylabel(metric.xlabel, fontsize=fontsize, rotation=270, labelpad=8)
    cax.yaxis.set_label_position("right")
    return cax


def object_cell_box(
    label: np.ndarray, object_idx: int, height: int, width: int
) -> Tuple[int, int, int, int]:
    """Feature-map cells covered by one object, on a map of any resolution."""
    block_y = label[ATTRIBUTE_INDEX[f"block{object_idx}_y"]]
    block_x = label[ATTRIBUTE_INDEX[f"block{object_idx}_x"]]
    r0, r1 = _fractional_slice(block_y / PATCH_GRID, (block_y + OBJECT_PATCHES) / PATCH_GRID, height)
    c0, c1 = _fractional_slice(block_x / PATCH_GRID, (block_x + OBJECT_PATCHES) / PATCH_GRID, width)
    return r0, r1, c0, c1


class ObjectTuningByUnit:
    """Per-unit tuning curves over object shape and colour.

    Arrays are stored as (C, H, W, n_categories) and flattened to the tissue's
    unit ordering -- channel major, then row, then column -- on access, which
    is the ordering ``LayerPositions.coordinates`` uses.
    """

    def __init__(
        self,
        sums: Dict[str, np.ndarray],
        counts: Dict[str, np.ndarray],
        on_object: np.ndarray,
        on_count: np.ndarray,
        off_object: np.ndarray,
        off_count: np.ndarray,
        color_names: List[str],
    ):
        self.dims = on_object.shape  # (C, H, W)
        self.color_names = color_names

        self.tuning: Dict[str, np.ndarray] = {}
        for prop in PROPERTIES:
            with np.errstate(invalid="ignore", divide="ignore"):
                self.tuning[prop] = sums[prop] / counts[prop][None, :, :, :]

        with np.errstate(invalid="ignore", divide="ignore"):
            on = on_object / np.maximum(on_count, 1)[None, :, :]
            off = off_object / np.maximum(off_count, 1)[None, :, :]
        self._objectness = (on - off).reshape(-1)

        self.metrics = {
            "shape": shape_metric(self.tuning["shape"].shape[-1]),
            "color": color_metric(color_names),
        }

    def __len__(self) -> int:
        return int(np.prod(self.dims))

    def _flat(self, array: np.ndarray) -> np.ndarray:
        return array.reshape(len(self), -1)

    def preferences(self, prop: str) -> np.ndarray:
        """Index of each unit's best-driving category."""
        curves = self._flat(self.tuning[prop])
        return np.nan_to_num(curves, nan=-np.inf).argmax(axis=1)

    def selectivity(self, prop: str) -> np.ndarray:
        """Peak-to-trough range of the tuning curve, as in ``get_peak_heights``."""
        curves = self._flat(self.tuning[prop])
        return np.nanmax(curves, axis=1) - np.nanmin(curves, axis=1)

    @property
    def objectness(self) -> np.ndarray:
        """Response with an object on the unit minus response without one."""
        return self._objectness


def accumulate_object_tuning(
    model,
    layer: str,
    dataset,
    indices: Optional[np.ndarray] = None,
    batch_size: int = 64,
    chunk_size: int = 200,
    num_workers: int = 8,
    verbose: bool = True,
) -> ObjectTuningByUnit:
    """One sweep of the stimuli, accumulating per-unit tuning curves.

    Feature maps are never held for the whole stimulus set -- only the running
    sums, which are (C, H, W, n_categories) and so are tiny next to the
    responses themselves.

    An object slot whose category comes back negative is treated as absent:
    it contributes to neither the tuning curves nor the object-versus-
    background contrast, and the cells it would have covered count as
    background. That is how a single-object localizer is swept with the same
    code as the intact two-object stimuli -- the erased object is marked with
    a negative category rather than needing a separate accumulator.
    """
    from torch.utils.data import Subset

    from spacetorch.feature_extractor import get_features_from_layer

    if indices is None:
        indices = np.arange(len(dataset))
    indices = np.asarray(indices)

    n_shapes = int(dataset.labels[:, ATTRIBUTE_INDEX["shape0"]].max()) + 1
    n_colors = len(dataset.color_names)
    sizes = {"shape": n_shapes, "color": n_colors}

    sums: Dict[str, np.ndarray] = {}
    counts: Dict[str, np.ndarray] = {}
    on_object = on_count = off_object = off_count = None

    for start in tqdm(
        range(0, len(indices), chunk_size), desc="chunks", disable=not verbose
    ):
        chunk = indices[start : start + chunk_size]
        features, _, labels = get_features_from_layer(
            model,
            Subset(dataset, chunk),
            model_layer_strings=[layer],
            batch_size=batch_size,
            return_inputs_and_labels=True,
            shuffle=False,
            verbose=False,
            num_workers=num_workers,
        )
        features = np.asarray(features, dtype=np.float32)
        assert features.ndim == 4, (
            f"expected (N, C, H, W) features, got {features.shape}; the tissue "
            "maps need a spatial layer"
        )
        _, n_channels, height, width = features.shape

        if on_object is None:
            for prop, size in sizes.items():
                sums[prop] = np.zeros((n_channels, height, width, size), dtype=np.float64)
                counts[prop] = np.zeros((height, width, size), dtype=np.float64)
            on_object = np.zeros((n_channels, height, width), dtype=np.float64)
            off_object = np.zeros((n_channels, height, width), dtype=np.float64)
            on_count = np.zeros((height, width), dtype=np.float64)
            off_count = np.zeros((height, width), dtype=np.float64)

        for image_idx in range(len(features)):
            label = labels[image_idx]
            response = features[image_idx]
            covered = np.zeros((height, width), dtype=bool)

            for object_idx in (0, 1):
                categories = {
                    prop: int(label[ATTRIBUTE_INDEX[f"{prop}{object_idx}"]])
                    for prop in PROPERTIES
                }
                if min(categories.values()) < 0:  # erased by a localizer
                    continue

                r0, r1, c0, c1 = object_cell_box(label, object_idx, height, width)
                patch = response[:, r0:r1, c0:c1]
                for prop in PROPERTIES:
                    category = categories[prop]
                    sums[prop][:, r0:r1, c0:c1, category] += patch
                    counts[prop][r0:r1, c0:c1, category] += 1
                covered[r0:r1, c0:c1] = True

            on_object += response * covered[None, :, :]
            off_object += response * (~covered)[None, :, :]
            on_count += covered
            off_count += ~covered

    return ObjectTuningByUnit(
        sums, counts, on_object, on_count, off_object, off_count, dataset.color_names
    )


class DiscriminationMap(TissueMap):
    """Object preference over the cortical sheet."""

    def __init__(
        self,
        positions: np.ndarray,
        tuning: ObjectTuningByUnit,
        unit_mask: Optional[np.ndarray] = None,
    ):
        super().__init__(positions, tuning, unit_mask=unit_mask)
        self.tuning = tuning
        self.rng = np.random.default_rng(0)

    def get_preferences(self, prop: str) -> np.ndarray:
        return self.tuning.preferences(prop)[self.unit_mask]

    def get_selectivity(self, prop: str) -> np.ndarray:
        return self.tuning.selectivity(prop)[self.unit_mask]

    def get_unit_colors(self, prop: str) -> np.ndarray:
        metric = self.tuning.metrics[prop]
        return metric.colors[self.get_preferences(prop)]

    def set_unit_mask_by_selectivity_percentile(self, prop: str, percentile: float):
        """Keep only the units whose tuning curve actually has a peak.

        A unit covered by few objects, or driven by none of them, still has an
        argmax; plotting it paints noise onto the map. The repository's V1 maps
        do the same thing with ``set_unit_mask_by_ptp_percentile``.
        """
        self.reset_unit_mask()
        selectivity = self.tuning.selectivity(prop)
        finite = np.isfinite(selectivity)
        cutoff = np.percentile(selectivity[finite], percentile)
        self.unit_mask = np.flatnonzero(finite & (selectivity >= cutoff))

    def make_parameter_map(
        self,
        axis,
        prop: str = "shape",
        scale_points: bool = True,
        final_s: Optional[float] = None,
        **kwargs,
    ):
        """Scatter every unit at its cortical position, coloured by preference.

        Returns None when the current mask is empty, which happens easily when
        a selectivity mask is intersected with a zoom window; the caller gets
        an empty panel rather than an exception from the point scaling.
        """
        if len(self.unit_mask) == 0:
            return None

        colors = self.get_unit_colors(prop)

        if scale_points:
            selectivity = self.get_selectivity(prop)
            selectivity = np.where(np.isfinite(selectivity), selectivity, 0.0)
            spread = np.ptp(selectivity)
            selectivity = (
                (selectivity - selectivity.min()) / spread + 0.5
                if spread > 0
                else np.ones_like(selectivity)
            )
        else:
            selectivity = np.ones(len(self.positions))

        return axis.scatter(
            self.positions[:, 0],
            self.positions[:, 1],
            s=final_s if final_s is not None else 0.4 * selectivity,
            c=colors,
            **kwargs,
        )

    def preference_agreement_over_distance(
        self,
        prop: str,
        distance_cutoff: float = 4.0,
        num_samples: int = 20,
        sample_size: int = 1000,
        num_bins: int = 20,
        shuffle: bool = False,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Chance that two units share a preferred category, versus how far apart.

        The categorical counterpart of ``metric_difference_over_distance``: for
        orientation one plots the mean angular difference, but shape and colour
        categories have no metric, so agreement is the 0/1 indicator that two
        units prefer the same one. A curve that starts high and decays is a
        topographic map; a flat curve at the population base rate is not.
        ``shuffle`` permutes the pairwise distances, which destroys any spatial
        arrangement while leaving the preferences untouched.
        """
        preferences = self.get_preferences(prop)
        positions = self.positions

        curves, bin_edges = [], None
        for _ in range(num_samples):
            picked = self.rng.choice(
                np.arange(len(positions)),
                size=(min(sample_size, len(positions)),),
                replace=False,
            )
            sample = preferences[picked][:, np.newaxis]
            agreement = lower_tri((sample == sample.T).astype(float))

            distances = lower_tri(squareform(pdist(positions[picked])))
            if shuffle:
                self.rng.shuffle(distances)

            keep = np.nonzero(distances <= distance_cutoff)[0]
            means, _, bin_edges = agg_by_distance(
                distances[keep], agreement[keep], bin_edges=bin_edges, num_bins=num_bins
            )
            curves.append(means)

        return midpoints_from_bin_edges(bin_edges), curves


def get_discrimination_tissue(
    model,
    positions,
    layer: str,
    dataset,
    indices: Optional[np.ndarray] = None,
    batch_size: int = 64,
    chunk_size: int = 200,
    num_workers: int = 8,
    verbose: bool = True,
) -> DiscriminationMap:
    """Measure object tuning and attach it to the layer's cortical positions."""
    tuning = accumulate_object_tuning(
        model,
        layer,
        dataset,
        indices=indices,
        batch_size=batch_size,
        chunk_size=chunk_size,
        num_workers=num_workers,
        verbose=verbose,
    )
    assert len(tuning) == len(positions.coordinates), (
        f"{len(tuning)} units measured but {len(positions.coordinates)} positions "
        f"for {layer}; the position file and the layer do not match"
    )
    return DiscriminationMap(positions.coordinates, tuning)
