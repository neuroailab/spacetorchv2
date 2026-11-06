"""
Privileged representational axes, following Khosla et al. (2024).

A representation ``Z`` of N units to M stimuli can be written in any basis:
``ZQ`` for orthogonal ``Q`` preserves every rotation-invariant property, which
includes linearly decodable content and representational geometry. The native
basis is therefore *privileged* only if something measurable about it survives
rotation that an arbitrary basis does not reproduce.

This module supplies the three ingredients that claim needs.

1.  **Per-axis selectivity.** Orientation selectivity from the first phase
    harmonic of the grating response (``GratingHarmonics``), and, for object
    properties, both a marginal explained variance (``eta_squared``) and a
    functional-localizer contrast of one category against the rest
    (``localizer_contrast``).

2.  **Graded axis perturbation.** ``rotation_path`` interpolates along the
    SO(n) manifold from the identity to a Haar-random rotation by taking
    fractional powers of a single draw, and ``axis_perturbation_curve``
    re-measures selectivity at each step. A monotone decline of the curve --
    not a single native-versus-rotated contrast -- is the evidence.

3.  **A ceiling.** ``optimal_rotation`` maximizes a differentiable surrogate of
    the same statistic over SO(n) by gradient ascent on the Lie algebra, so
    native selectivity can be reported as a fraction of the best any basis
    achieves.

For comparing two trained models, ``identity_statistics`` reports the physical
identity measures of the empirical reproducibility package -- matched
correlation, mutual-nearest-neighbour fraction and assignment stability, from a
one-to-one assignment fitted on one split of the stimuli and scored on a
held-out split -- and ``cycle_closure`` tests whether that correspondence
composes around a loop of three models. All of them are meant to be run against
a dense rotated basis, because raw correlation alone does not separate the two.
"""

from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

# usage strata, named as in the source tables: the most-used tenth of the axes,
# the next 40%, and the lower-used half
STRATUM_NAMES: List[str] = ["top10", "middle40", "bottom50"]
STRATUM_PERCENTILE_EDGES: List[float] = [50.0, 90.0]


# ---------------------------------------------------------------------------
# Pooling feature maps into per-axis responses
# ---------------------------------------------------------------------------


def channel_responses(features: np.ndarray) -> np.ndarray:
    """Collapse feature maps to one response per axis per image.

    Spatial positions are averaged, so an "axis" is a channel of the native
    basis, matching Khosla et al.'s definition of a unit in a convolutional
    network as a single filter with its response averaged over all locations.
    The tissue analyses elsewhere in this repository instead treat each
    channel/position pair as a unit; that convention cannot be used here,
    because a rotation control has to act on the basis the network itself
    factorizes.
    """
    features = np.asarray(features, dtype=np.float32)
    if features.ndim == 2:
        return features
    if features.ndim == 4:
        return features.mean(axis=(2, 3))
    raise ValueError(f"expected (N, C) or (N, C, H, W) features, got {features.shape}")


def region_responses(features: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Mean response of every axis inside a rectangular region of each image.

    Inputs:
        features: (N, C, H, W) feature maps
        boxes: (N, 4) fractional (top, left, bottom, right) extents in [0, 1]
    """
    features = np.asarray(features, dtype=np.float32)
    assert features.ndim == 4, f"expected (N, C, H, W) features, got {features.shape}"
    n_images, n_axes, height, width = features.shape
    assert len(boxes) == n_images, "one box is required per image"

    responses = np.zeros((n_images, n_axes), dtype=np.float32)
    for image_idx, (top, left, bottom, right) in enumerate(np.asarray(boxes)):
        r0, r1 = _fractional_slice(top, bottom, height)
        c0, c1 = _fractional_slice(left, right, width)
        responses[image_idx] = features[image_idx, :, r0:r1, c0:c1].mean(axis=(1, 2))

    return responses


def complement_responses(
    features: np.ndarray, boxes_per_object: List[np.ndarray]
) -> np.ndarray:
    """Mean response of every axis outside all of the supplied regions."""
    features = np.asarray(features, dtype=np.float32)
    n_images, n_axes, height, width = features.shape

    responses = np.zeros((n_images, n_axes), dtype=np.float32)
    for image_idx in range(n_images):
        keep = np.ones((height, width), dtype=bool)
        for boxes in boxes_per_object:
            top, left, bottom, right = boxes[image_idx]
            r0, r1 = _fractional_slice(top, bottom, height)
            c0, c1 = _fractional_slice(left, right, width)
            keep[r0:r1, c0:c1] = False
        if not keep.any():
            responses[image_idx] = np.nan
            continue
        responses[image_idx] = features[image_idx][:, keep].mean(axis=1)

    return responses


def _fractional_slice(low: float, high: float, size: int) -> Tuple[int, int]:
    """Convert a fractional extent to a half-open index range of at least one."""
    start = int(np.clip(np.floor(low * size), 0, size - 1))
    stop = int(np.clip(np.ceil(high * size), start + 1, size))
    return start, stop


# ---------------------------------------------------------------------------
# Per-axis selectivity
# ---------------------------------------------------------------------------


def _group_indicator(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """One-hot membership matrix and per-group counts."""
    _, inverse = np.unique(np.asarray(labels), return_inverse=True)
    n_groups = int(inverse.max()) + 1
    membership = np.zeros((len(inverse), n_groups), dtype=np.float64)
    membership[np.arange(len(inverse)), inverse] = 1.0
    return membership, membership.sum(axis=0)


def eta_squared(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Marginal explained-variance fraction of a categorical factor.

    .. math::
        \\eta^2(v; \\ell) = \\frac{\\sum_k n_k (\\bar v_k - \\bar v)^2}
                                  {\\sum_o (v_o - \\bar v)^2}

    An empirical one-factor association measured while the other stimulus
    properties vary freely, not an out-of-sample prediction score.
    """
    values = np.asarray(values, dtype=np.float64)
    squeeze = values.ndim == 1
    if squeeze:
        values = values[:, None]

    membership, counts = _group_indicator(labels)
    group_means = (membership.T @ values) / counts[:, None]

    grand_mean = values.mean(axis=0)
    between = (counts[:, None] * (group_means - grand_mean) ** 2).sum(axis=0)
    total = ((values - grand_mean) ** 2).sum(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        eta2 = np.where(total > 0, between / total, np.nan)

    return eta2[0] if squeeze else eta2


def localizer_contrast(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Standardized response difference for each category against all others.

    This is the functional-localizer logic Khosla et al. use to count
    category-selective units: contrast the response to one category with the
    response to everything else. They threshold a t-statistic at 10 over 515
    stimuli; a t-statistic scales with the square root of the sample size, and
    the panels here are roughly 25 times larger, so the effect size is returned
    instead. The two differ only by that factor and the effect size can carry a
    fixed threshold across stimulus sets.

    Returns:
        (n_categories, n_axes) array of Cohen's d, positive when the category
        drives the axis above the rest of the stimuli
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]

    membership, counts = _group_indicator(labels)
    n_samples = len(values)

    totals = values.sum(axis=0)
    squares = (values**2).sum(axis=0)
    group_sums = membership.T @ values
    group_squares = membership.T @ (values**2)

    inside_n = counts[:, None]
    outside_n = n_samples - inside_n

    inside_mean = group_sums / inside_n
    outside_mean = (totals - group_sums) / outside_n

    inside_var = np.maximum(group_squares / inside_n - inside_mean**2, 0.0)
    outside_var = np.maximum(
        (squares - group_squares) / outside_n - outside_mean**2, 0.0
    )
    # unbiased, and guarded for singleton categories
    inside_var = inside_var * inside_n / np.maximum(inside_n - 1, 1)
    outside_var = outside_var * outside_n / np.maximum(outside_n - 1, 1)

    pooled = 0.5 * (inside_var + outside_var)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(pooled > 0, (inside_mean - outside_mean) / np.sqrt(pooled), 0.0)


def peak_selectivity(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Strongest localizer contrast any single category produces, per axis.

    An axis counts as selective for a property when some category of that
    property drives it away from the rest, which is what "a face-selective
    unit" means. Taking the maximum over categories keeps the measure a single
    number per axis so it can be thresholded and counted.
    """
    return localizer_contrast(values, labels).max(axis=0)


def category_profile(values: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Per-category mean, count, and standard error for a single axis."""
    values = np.asarray(values, dtype=np.float64)
    categories, inverse = np.unique(np.asarray(labels), return_inverse=True)

    means = np.zeros(len(categories))
    counts = np.zeros(len(categories), dtype=int)
    sems = np.zeros(len(categories))
    for group in range(len(categories)):
        group_values = values[inverse == group]
        counts[group] = len(group_values)
        means[group] = group_values.mean()
        sems[group] = (
            group_values.std(ddof=1) / np.sqrt(len(group_values))
            if len(group_values) > 1
            else 0.0
        )

    return categories, means, counts, sems


def signed_contrast(
    values: np.ndarray, indicator: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean difference between the two levels of a binary indicator, and its d'."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    indicator = np.asarray(indicator).astype(bool)

    positive, negative = values[indicator], values[~indicator]
    delta = positive.mean(axis=0) - negative.mean(axis=0)

    pooled = 0.5 * (positive.var(axis=0, ddof=1) + negative.var(axis=0, ddof=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        dprime = np.where(pooled > 0, delta / np.sqrt(pooled), np.nan)

    return delta, dprime


class GratingHarmonics:
    """Orientation tuning from the first phase harmonic of a grating response.

    The harmonic is a fixed linear map of the response vector, so it is
    precomputed once as a pair of design matrices and can then be applied to a
    rotated response matrix as cheaply as to the native one. That matters
    because the rotation curve re-measures selectivity at every step.

    For a response :math:`z(\\theta, f, \\phi, \\chi)` sampled at P evenly
    spaced phases,

    .. math::
        \\alpha = \\frac{2}{P}\\sum_p z\\cos\\phi_p, \\quad
        \\beta  = \\frac{2}{P}\\sum_p z\\sin\\phi_p, \\quad
        M = \\sqrt{\\alpha^2 + \\beta^2}.

    Taking the harmonic rather than the mean matters: a simple-cell-like unit
    inverts with grating phase, so its mean over phase is flat in orientation
    even when it is sharply tuned.
    """

    def __init__(
        self,
        angles: np.ndarray,
        sfs: np.ndarray,
        phases: np.ndarray,
        colors: np.ndarray,
    ):
        self.angles = np.unique(angles)
        self.sfs = np.unique(sfs)
        self.colors = np.unique(colors)

        n_images = len(angles)
        n_cells = len(self.angles) * len(self.sfs) * len(self.colors)
        cos_design = np.zeros((n_cells, n_images))
        sin_design = np.zeros((n_cells, n_images))

        cell = 0
        for angle in self.angles:
            for sf in self.sfs:
                for color in self.colors:
                    mask = (angles == angle) & (sfs == sf) & (colors == color)
                    n_phases = int(mask.sum())
                    if n_phases:
                        cos_design[cell, mask] = 2.0 / n_phases * np.cos(phases[mask])
                        sin_design[cell, mask] = 2.0 / n_phases * np.sin(phases[mask])
                    cell += 1

        self.cos_design = cos_design
        self.sin_design = sin_design
        self.shape = (len(self.angles), len(self.sfs), len(self.colors))

    def modulation(self, responses: np.ndarray) -> Dict[str, np.ndarray]:
        """(n_axes, n_angles, n_sfs, n_colors) harmonic amplitude."""
        responses = np.asarray(responses, dtype=np.float64)
        n_axes = responses.shape[1]

        def reshape(design: np.ndarray) -> np.ndarray:
            return (design @ responses).T.reshape((n_axes,) + self.shape)

        cos_term, sin_term = reshape(self.cos_design), reshape(self.sin_design)
        return {
            "modulation": np.sqrt(cos_term**2 + sin_term**2),
            "cos_term": cos_term,
            "sin_term": sin_term,
        }

    def tuning(self, responses: np.ndarray) -> Dict[str, np.ndarray]:
        """Orientation selectivity index and preferred stimulus, per axis.

        The frequency/colour pair over which orientation moves the response
        most is selected first, and the index summarizes the concentration of
        the resulting profile:

        .. math::
            \\mathrm{OSI} = \\frac{|\\sum_\\theta M(\\theta)e^{2i\\theta}|}
                                  {\\sum_\\theta M(\\theta)}

        The doubled angle makes the measure axial, identifying orientations
        180 degrees apart.
        """
        harmonic = self.modulation(responses)
        modulation = harmonic["modulation"]
        n_axes, _, n_sfs, n_colors = modulation.shape

        spread = modulation.max(axis=1) - modulation.min(axis=1)
        best = spread.reshape(n_axes, -1).argmax(axis=1)
        best_sf, best_color = np.unravel_index(best, (n_sfs, n_colors))

        axis_idx = np.arange(n_axes)
        profile = modulation[axis_idx, :, best_sf, best_color]
        cos_profile = harmonic["cos_term"][axis_idx, :, best_sf, best_color]
        sin_profile = harmonic["sin_term"][axis_idx, :, best_sf, best_color]

        angles_rad = np.deg2rad(self.angles)
        numerator = np.abs((profile * np.exp(2j * angles_rad)[None, :]).sum(axis=1))
        denominator = profile.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            osi = np.where(denominator > 0, numerator / denominator, np.nan)

        preferred = profile.argmax(axis=1)
        return {
            "orientation_selectivity_index": osi,
            "preferred_orientation_deg": self.angles[preferred],
            "preferred_spatial_frequency": self.sfs[best_sf],
            "preferred_color": self.colors[best_color],
            "preferred_phase_rad": np.arctan2(
                sin_profile[axis_idx, preferred], cos_profile[axis_idx, preferred]
            ),
            "peak_phase_modulation": profile[axis_idx, preferred],
        }

    def osi(self, responses: np.ndarray) -> np.ndarray:
        return self.tuning(responses)["orientation_selectivity_index"]


def axial_difference(first_deg: np.ndarray, second_deg: np.ndarray) -> np.ndarray:
    """Orientation difference in degrees, identifying angles 180 apart."""
    difference = np.abs(np.asarray(first_deg) - np.asarray(second_deg)) % 180.0
    return np.minimum(difference, 180.0 - difference)


def uniform_orientation_baseline(angles_deg: np.ndarray, tolerance_deg: float) -> float:
    """Chance agreement between two independently drawn preferred orientations.

    Enumerated over the discrete bins the grating bank actually samples rather
    than assumed continuous, so it stays correct if the stimulus set changes.
    With the eight bins of the 2019 set and a one-bin tolerance this is 0.375.
    """
    angles_deg = np.unique(angles_deg)
    grid_a, grid_b = np.meshgrid(angles_deg, angles_deg)
    return float((axial_difference(grid_a, grid_b) <= tolerance_deg + 1e-9).mean())


# ---------------------------------------------------------------------------
# Whole-population summaries
# ---------------------------------------------------------------------------


def gini_coefficient(values: np.ndarray) -> float:
    """Inequality of a non-negative distribution, 0 (flat) to 1 (a spike).

    A threshold-free companion to the selective-unit count: it asks whether
    selectivity is concentrated on a few coordinates using every axis, without
    choosing how many count as "the top".
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan

    values = values - min(values.min(), 0.0)
    total = values.sum()
    if total <= 0:
        return np.nan

    ordered = np.sort(values)
    n_values = len(ordered)
    index = np.arange(1, n_values + 1)
    return float(
        (2.0 * (index * ordered).sum()) / (n_values * total) - (n_values + 1) / n_values
    )


def selectivity_summary(
    scores: np.ndarray, thresholds: Sequence[float], prefix: str = ""
) -> Dict[str, float]:
    """Summaries of a per-axis selectivity distribution, over all axes.

    ``n_selective@t`` is the quantity Khosla et al. track across rotations, and
    is reported at a ladder of thresholds rather than one, because the count is
    only a valid concentration measure when the bar is stringent. Simulations
    with eight selective axes planted among 256 make the dependence explicit
    (native versus fully rotated counts, as the planted signal grows from 4% to
    40% of total response variance):

    ======  ==========  =========  ========  =======
    d >     4% signal   6.5%       16%       40%
    ======  ==========  =========  ========  =======
    1       8 / 0       8 / 6      8 / 66    8 / 144
    2       0 / 0       8 / 0      8 / 4     8 / 53
    3       0 / 0       0 / 0      8 / 0     8 / 4
    ======  ==========  =========  ========  =======

    At a lenient bar, rotation *creates* moderately selective axes faster than
    it destroys sharp ones and the count inverts; at ``d > 3`` the count falls
    in every regime. Hence the stringent default, and hence reporting the whole
    ladder so the dependence is visible rather than hidden in one number.

    ``gini`` needs no threshold and fell with rotation in all four regimes
    above, which makes it the robust companion statistic. ``max`` agrees with
    it. ``mean`` and ``median`` describe the bulk and typically move the
    opposite way; none of these is rotation-invariant, because selectivity is
    normalized within each axis and a nuisance-only axis scores near zero until
    signal is mixed into it.
    """
    scores = np.asarray(scores, dtype=np.float64)
    finite = scores[np.isfinite(scores)]
    if len(finite) == 0:
        return {}

    summary = {
        f"{prefix}gini": gini_coefficient(finite),
        f"{prefix}max": float(finite.max()),
        f"{prefix}mean": float(finite.mean()),
        f"{prefix}median": float(np.median(finite)),
    }
    for threshold in thresholds:
        summary[f"{prefix}n_selective@{threshold:g}"] = float(
            (finite > threshold).sum()
        )
    return summary


def usage_strata(responses: np.ndarray) -> Dict[str, np.ndarray]:
    """Rank axes by response variance and bin them into three usage strata."""
    responses = np.asarray(responses, dtype=np.float64)
    relevance = responses.var(axis=0)

    order = relevance.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(relevance))
    percentile = 100.0 * ranks / max(len(relevance) - 1, 1)

    stratum = np.full(len(relevance), STRATUM_NAMES[2], dtype=object)
    stratum[percentile >= STRATUM_PERCENTILE_EDGES[0]] = STRATUM_NAMES[1]
    stratum[percentile >= STRATUM_PERCENTILE_EDGES[1]] = STRATUM_NAMES[0]

    return {
        "relevance": relevance,
        "relevance_percentile": percentile,
        "relevance_stratum": stratum,
    }


def top_axes(scores: np.ndarray, n_top: int = 16) -> np.ndarray:
    """Indices of the most selective axes, best first, ignoring NaNs.

    Used only to pick exemplars to display. No reported statistic depends on
    it; those all run over the full population.
    """
    scores = np.asarray(scores, dtype=np.float64)
    filled = np.where(np.isfinite(scores), scores, -np.inf)
    return filled.argsort()[::-1][:n_top]


def subsample_axes(n_axes: int, max_axes: Optional[int]) -> Optional[np.ndarray]:
    """Evenly spaced axis subset, or None to use every axis.

    An escape hatch for the cross-model alignment curve, which re-solves an
    assignment at every rotation step. The within-model selectivity curves use
    every axis.
    """
    if max_axes is None or max_axes <= 0 or n_axes <= max_axes:
        return None
    return np.unique(np.linspace(0, n_axes - 1, max_axes).astype(int))


# ---------------------------------------------------------------------------
# Axis perturbation along SO(n)
# ---------------------------------------------------------------------------


def random_orthogonal(n_axes: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-distributed random rotation, with determinant forced to +1."""
    gaussian = rng.standard_normal((n_axes, n_axes))
    q, r = np.linalg.qr(gaussian)
    q = q * np.sign(np.diag(r))[None, :]
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def rotation_path(
    n_axes: int, alphas: Sequence[float], rng: np.random.Generator
) -> Iterator[Tuple[float, np.ndarray]]:
    """Interpolate from the identity to one Haar rotation along SO(n).

    Yields ``(alpha, Q**alpha)`` for each requested alpha, where alpha = 0 is
    the native basis and alpha = 1 a uniformly drawn rotation. Following
    Khosla et al., the fractional powers come from a single real Schur
    decomposition ``Q = W T W^T``: for an orthogonal matrix ``T`` decomposes
    into two-dimensional rotation blocks, and raising the matrix to a power
    means scaling each block's angle. Doing the decomposition once keeps the
    whole path roughly as expensive as one rotation.
    """
    from scipy.linalg import schur

    rotation = random_orthogonal(n_axes, rng)
    triangular, basis = schur(rotation, output="real")

    # collect the plane rotations: adjacent 2x2 blocks, plus any pair of -1
    # eigenvalues, which together form a rotation by pi in their own plane
    planes: List[Tuple[int, int, float]] = []
    negatives: List[int] = []
    index = 0
    while index < n_axes:
        if index + 1 < n_axes and abs(triangular[index + 1, index]) > 1e-10:
            angle = np.arctan2(triangular[index + 1, index], triangular[index, index])
            planes.append((index, index + 1, angle))
            index += 2
            continue
        if triangular[index, index] < 0:
            negatives.append(index)
        index += 1

    for first, second in zip(negatives[::2], negatives[1::2]):
        planes.append((first, second, np.pi))

    for alpha in alphas:
        if alpha == 0:
            yield 0.0, np.eye(n_axes)
            continue
        powered = np.eye(n_axes)
        for first, second, angle in planes:
            cosine, sine = np.cos(alpha * angle), np.sin(alpha * angle)
            powered[first, first] = cosine
            powered[second, second] = cosine
            powered[second, first] = sine
            powered[first, second] = -sine
        yield float(alpha), basis @ powered @ basis.T


def axis_perturbation_curve(
    responses: np.ndarray,
    statistic: Callable[[np.ndarray], Dict[str, float]],
    alphas: Sequence[float],
    n_rotations: int = 10,
    seed: int = 0,
    center: bool = True,
    verbose: bool = True,
) -> Dict[float, Dict[str, Tuple[float, float]]]:
    """Re-measure a summary statistic along the rotation path.

    ``statistic`` maps an (n_samples, n_axes) response matrix to a dictionary
    of scalars. It is evaluated once in the native basis and then at each
    alpha for each of ``n_rotations`` independent draws of the rotation.

    Responses are centered first, so the comparison concerns response
    variation rather than where the mean response vector happens to point.

    Returns:
        ``{alpha: {name: (mean, sd over draws)}}``, with alpha 0 holding the
        native value and an sd of zero.
    """
    responses = np.asarray(responses, dtype=np.float64)
    if center:
        responses = responses - responses.mean(axis=0, keepdims=True)

    native = statistic(responses)
    curve: Dict[float, Dict[str, Tuple[float, float]]] = {
        0.0: {key: (float(value), 0.0) for key, value in native.items()}
    }

    positive = [alpha for alpha in alphas if alpha > 0]
    samples: Dict[float, Dict[str, List[float]]] = {
        alpha: {key: [] for key in native} for alpha in positive
    }

    rng = np.random.default_rng(seed)
    for draw in range(n_rotations):
        if verbose:
            print(f"  rotation draw {draw + 1}/{n_rotations}", flush=True)
        for alpha, rotation in rotation_path(responses.shape[1], positive, rng):
            for key, value in statistic(responses @ rotation).items():
                samples[alpha][key].append(value)

    for alpha in positive:
        curve[alpha] = {
            key: (float(np.nanmean(values)), float(np.nanstd(values, ddof=1)))
            if len(values) > 1
            else (float(np.nanmean(values)), 0.0)
            for key, values in samples[alpha].items()
        }

    return curve


def optimal_rotation(
    responses: np.ndarray,
    objective: Callable,
    n_steps: int = 40,
    learning_rate: float = 1e-3,
    seed: int = 0,
    monitor: Optional[Callable[[np.ndarray], float]] = None,
    monitor_every: int = 5,
) -> Tuple[np.ndarray, float]:
    """Largest value of a differentiable objective over any basis.

    Every rotation can be written as the matrix exponential of a
    skew-symmetric matrix, which turns the constrained search over SO(n) into
    an unconstrained one over that matrix's free parameters. ``objective``
    takes a rotated torch response matrix and returns a scalar to maximize;
    because a count of selective axes is not differentiable, callers should
    pass a smooth surrogate as ``objective`` and the real statistic as
    ``monitor``. The rotation returned is the one that maximized ``monitor``,
    not the surrogate, and the identity is always among the candidates -- so
    the result is a genuine ceiling that cannot come out below the native
    basis. Reporting the surrogate's optimum instead silently allows a
    "ceiling" beneath the value it is supposed to bound.

    Returns:
        (best rotation as a numpy array, best monitored value)
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    matrix = torch.as_tensor(np.asarray(responses), dtype=torch.float32, device=device)
    matrix = matrix - matrix.mean(dim=0, keepdim=True)

    n_axes = matrix.shape[1]
    torch.manual_seed(seed)
    free = torch.zeros((n_axes, n_axes), device=device, requires_grad=True)
    optimizer = torch.optim.Adam([free], lr=learning_rate)

    identity = np.eye(n_axes)
    score = monitor if monitor is not None else None
    best_rotation = identity
    best_value = score(identity) if score else -np.inf

    for step in range(n_steps):
        optimizer.zero_grad()
        rotation = torch.matrix_exp(free - free.T)
        value = objective(matrix @ rotation)
        (-value).backward()
        optimizer.step()

        last = step == n_steps - 1
        if score is not None and (step % monitor_every == 0 or last):
            candidate = rotation.detach().cpu().numpy()
            current = score(candidate)
            if current > best_value:
                best_value, best_rotation = current, candidate
        elif score is None:
            current = float(value.detach().cpu())
            if current > best_value:
                best_value = current
                best_rotation = rotation.detach().cpu().numpy()

    return best_rotation, best_value


def torch_eta_squared(matrix, membership, counts):
    """Differentiable eta-squared, for the SO(n) ceiling search."""
    group_means = (membership.T @ matrix) / counts[:, None]
    grand_mean = matrix.mean(dim=0)
    between = (counts[:, None] * (group_means - grand_mean) ** 2).sum(dim=0)
    total = ((matrix - grand_mean) ** 2).sum(dim=0)
    return between / total.clamp_min(1e-12)


def soft_count(scores, threshold: float, temperature: float = 0.05):
    """Smooth surrogate for the number of axes above a threshold."""
    import torch

    return torch.sigmoid((scores - threshold) / temperature).sum()


# ---------------------------------------------------------------------------
# Cross-model axis identity
# ---------------------------------------------------------------------------


def correlation_matrix(responses_a: np.ndarray, responses_b: np.ndarray) -> np.ndarray:
    """Pearson correlation between every axis of one model and of another."""
    assert len(responses_a) == len(responses_b), "responses must share their stimuli"

    def standardize(responses: np.ndarray) -> np.ndarray:
        responses = np.asarray(responses, dtype=np.float64)
        centered = responses - responses.mean(axis=0, keepdims=True)
        scale = np.linalg.norm(centered, axis=0)
        return np.divide(centered, scale, out=np.zeros_like(centered), where=scale > 0)

    return standardize(responses_a).T @ standardize(responses_b)


def one_to_one_assignment(corr: np.ndarray) -> np.ndarray:
    """Assignment maximizing total matched correlation, one partner per axis.

    The empirical reproducibility package selects its physical-axis map this
    way -- ``argmax_pi sum_i C[i, pi(i)]`` on training data -- rather than by
    letting several axes claim the same partner, and every discrete identity
    statistic it reports is computed from this assignment.
    """
    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(-np.asarray(corr))
    assignment = np.full(corr.shape[0], -1, dtype=int)
    assignment[rows] = cols
    return assignment


def mutual_nearest_fraction(corr: np.ndarray) -> float:
    """Fraction of axes that are each other's single best partner.

    A much harsher test than matched correlation: it asks whether the
    correspondence is unambiguous in both directions, which is where the
    reference package finds native channels beating a dense rotated basis even
    when raw correlation does not separate them (conv1: .475 native vs .197
    dense).
    """
    corr = np.asarray(corr)
    forward = corr.argmax(axis=1)
    backward = corr.argmax(axis=0)
    return float((backward[forward] == np.arange(len(forward))).mean())


def assignment_stability(corr_fit: np.ndarray, corr_test: np.ndarray) -> float:
    """Fraction of axes assigned the same partner on fitting and held-out data.

    A stable assignment means the correspondence is a property of the axes
    rather than of the particular stimuli it was fitted on.
    """
    return float(
        (one_to_one_assignment(corr_fit) == one_to_one_assignment(corr_test)).mean()
    )


def cycle_closure(assignments: Sequence[np.ndarray]) -> float:
    """Fraction of axes returned to themselves by a chain of assignments.

    With three models, ``i -> pi_AB -> j -> pi_BC -> k -> pi_CA -> i'`` closes
    when ``i' == i``. Transitivity is a far stronger claim than pairwise
    similarity: an arbitrary pairwise match need not compose.
    """
    index = np.arange(len(assignments[0]))
    current = index.copy()
    for assignment in assignments:
        valid = current >= 0
        nxt = np.full_like(current, -1)
        nxt[valid] = assignment[current[valid]]
        current = nxt
    return float((current == index).mean())


def identity_statistics(
    fit_a: np.ndarray,
    fit_b: np.ndarray,
    test_a: np.ndarray,
    test_b: np.ndarray,
) -> Dict[str, object]:
    """Matched correlation plus the discrete identity measures, for one pair.

    Raw matched correlation is reported alongside the discrete measures rather
    than on its own, because the reference package shows it is the statistic
    least able to separate a native basis from a dense rotated one -- a rotated
    basis can beat the native channels on correlation at conv1 while losing
    decisively on mutual-nearest-neighbour matching and cycle closure.
    """
    corr_fit = correlation_matrix(fit_a, fit_b)
    corr_test = correlation_matrix(test_a, test_b)

    assignment = one_to_one_assignment(corr_fit)
    index = np.arange(len(assignment))

    semi = corr_fit.argmax(axis=1)
    return {
        "matched_correlation": float(np.nanmean(corr_test[index, assignment])),
        "semi_matched_correlation": float(np.nanmean(corr_test[index, semi])),
        "mutual_nearest_fraction": mutual_nearest_fraction(corr_fit),
        "assignment_stability": assignment_stability(corr_fit, corr_test),
        "assignment": assignment,
    }
