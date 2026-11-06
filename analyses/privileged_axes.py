"""Are orientation, shape, colour and position carried by privileged axes?

Implements the framework of Khosla et al. (2024) for these four stimulus
properties, using the measurement conventions of the privileged visual
computations note and the empirical reproducibility package. No natural-image
set is involved: the correspondence between models is fitted and scored on
splits of the same stimuli the selectivity is measured on.

Four conventions are inherited from those packages, and each one matters:

* the state read is the **argument of the nonlinearity** -- the pre-GELU MLP
  gate in a transformer, the pre-ReLU channel in a convnet -- because
  correspondence "is usually strongest at a preactivation or normalized state
  rather than after the pointwise nonlinearity";
* orientation is probed with a **Gabor patch on one token**, read at that
  token, not with full-field gratings averaged over every token;
* cross-model identity is judged by **mutual-nearest-neighbour matching,
  assignment stability and cycle closure** against a dense rotated basis, not
  by raw matched correlation, which that package reports a rotated basis can
  win;
* every comparison is also made **within usedness strata**, since downstream
  usedness is its strongest predictor of which axes recur.

Two questions are answered, and they are independent.

**Is a property's selectivity axis-dependent?** (their Figure 6) This needs one
model. Count the axes selective for the property, then repeat the count along a
path of rotations ``Q**alpha`` running from the native basis to a Haar-random
one. Every rotated basis holds identical decodable information, so a decline in
the count as alpha grows means the property is expressed by *coordinates*
rather than merely by the subspace they span. The count is reported against two
references: the fully rotated basis, and the best basis found by gradient
ascent on SO(n), which says whether the native axes are merely better than
arbitrary or close to the best available.

**Do two trained models share those axes?** (their Figure 3) Match the axes of
one model to the other on a fitting split, score the matched pairs on a
held-out split, and repeat with the first model's basis replaced by a dense
rotation.

Controls, both cheap and both worth running: ``--shuffle_control`` permutes the
property labels against the responses, which must abolish the effect; and
pointing ``--configs`` at an ``unoptimized.yaml`` gives an untrained network,
which should show no privilege at all.

Every layer the architecture exposes (every key in its BRAIN_MAPPING entry) is
extracted and compared -- there is no ``--areas`` selector. All ``--configs``
are assumed to share an architecture (same as the row-alignment assertion on
``object_labels`` further down already assumes), so the layer list is resolved
once, from the first config.
"""

import argparse
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from spacetorch.configs import get_cfg
from spacetorch.datasets import DatasetRegistry
from spacetorch.datasets.discrimination import DiscriminationResponses
from spacetorch.datasets.gabor_patches import GaborPatches
from spacetorch.feature_extractor import get_pooled_features_from_layer
from spacetorch.privileged_axes import (
    STRATUM_NAMES,
    GratingHarmonics,
    axial_difference,
    axis_perturbation_curve,
    channel_responses,
    complement_responses,
    correlation_matrix,
    cycle_closure,
    identity_statistics,
    one_to_one_assignment,
    optimal_rotation,
    peak_selectivity,
    random_orthogonal,
    region_responses,
    rotation_path,
    selectivity_summary,
    soft_count,
    subsample_axes,
    top_axes,
    torch_eta_squared,
    uniform_orientation_baseline,
    usage_strata,
)
from spacetorch.variants import get_variant
from spacetorch.variants.positions import BRAIN_MAPPING

OBJECT_PROPERTIES = ["shape", "color", "position"]
PROPERTIES = ["orientation"] + OBJECT_PROPERTIES
PROPERTY_COLORS = {
    "orientation": "tab:red",
    "shape": "black",
    "color": "tab:blue",
    "position": "tab:green",
}

# Where each architecture's pre-nonlinearity state lives, relative to the layer
# BRAIN_MAPPING names. Both reference packages measure this state rather than
# the post-activation one: "approximate weak zippering ... is usually strongest
# at a preactivation or normalized state rather than after the pointwise
# nonlinearity", with the ViT object being the pre-GELU MLP gate and the
# convolutional object the pre-ReLU channel.
#
# For a ResNet BasicBlock/Bottleneck the exact argument of the ReLU is
# ``bn2 + identity``, which is not a module; ``bn2``/``bn3`` is the residual
# branch's own pre-ReLU output and is the closest hookable state. That
# difference is worth remembering when reading ResNet numbers.
PREACTIVATION_LAYER = {
    "resnet18": lambda layer: f"{layer}.bn2",
    "resnet18_logpolar": lambda layer: f"{layer}.bn2",
    "resnet50": lambda layer: f"{layer}.bn3",
    "vitb16": lambda layer: f"{layer}.mlp.fc1",
    "vitb14": lambda layer: f"{layer}.mlp.fc1",
    # the *_a architectures name the attention output; the MLP gate of the same
    # block is the reference packages' object
    "vitb16a": lambda layer: f"{layer[:-len('.attn')]}.mlp.fc1",
    "vitb14a": lambda layer: f"{layer[:-len('.attn')]}.mlp.fc1",
    "vitb16a_mae": lambda layer: f"{layer[:-len('.attn')]}.mlp.fc1",
    "swin_tiny": lambda layer: f"{layer}.mlp.fc1",
    "swin_tiny_a": lambda layer: f"{layer[:-len('.attn_output')]}.mlp.fc1",
    "alexnet": lambda layer: layer.rsplit(".", 1)[0] + ".0",
    "alexnet_rotnet": lambda layer: layer.rsplit(".", 1)[0] + ".0",
}

RESULT_COLUMNS = [
    "group",
    "area",
    "analysis",
    "model_a",
    "model_b",
    "layer_a",
    "layer_b",
    "property",
    "stimuli",
    "statistic",
    "alpha",
    "value",
    "sd",
    "n_axes",
]

STRATA = ["all", "top10", "bottom50"]


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=str, required=True)
    parser.add_argument("--configs", type=str, nargs="+", required=True)
    parser.add_argument(
        "--names",
        type=str,
        nargs="+",
        default=None,
        help="labels for the models; defaults to the config paths, which matters "
        "because several configs share a `name` field",
    )
    parser.add_argument("--output_dir", type=str, default="paper")
    parser.add_argument(
        "--state",
        type=str,
        choices=["preactivation", "native"],
        default="preactivation",
        help="which state to read. 'preactivation' maps each BRAIN_MAPPING "
        "layer to the argument of its nonlinearity (pre-GELU MLP gate for a "
        "transformer, pre-ReLU channel for a convnet), which is the state both "
        "reference packages measure; 'native' keeps the BRAIN_MAPPING layer",
    )
    parser.add_argument(
        "--n_dense_rotations",
        type=int,
        default=3,
        help="dense-rotated-basis controls for the discrete identity statistics",
    )
    parser.add_argument(
        "--n_images",
        type=int,
        default=0,
        help="discrimination stimuli to measure; 0, the default, uses the split",
    )
    parser.add_argument("--position_bins", type=int, default=4)
    parser.add_argument(
        "--osi_thresholds",
        type=float,
        nargs="+",
        default=[0.5, 0.6, 0.7, 0.8],
        help="orientation-selectivity bars to count axes at; the last is the "
        "headline one used for figures and for the preferred-orientation subset",
    )
    parser.add_argument(
        "--selectivity_thresholds",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 3.0, 4.0],
        help="object-property selectivity bars, in pooled standard deviations. "
        "The headline is the last but one (d > 3): simulations show the count "
        "only falls under rotation when the bar is stringent, while a lenient "
        "bar inverts because mixing manufactures moderately selective axes",
    )
    parser.add_argument(
        "--orientation_tolerance_deg",
        type=float,
        default=22.5,
        help="preferred orientations agree within this axial difference",
    )
    parser.add_argument(
        "--n_alphas",
        type=int,
        default=6,
        help="rotation steps between the native basis and a full rotation",
    )
    parser.add_argument("--n_rotations", type=int, default=5)
    parser.add_argument(
        "--ceiling_steps",
        type=int,
        default=40,
        help="Adam steps for the SO(n) ceiling search; 0 skips it",
    )
    parser.add_argument("--ceiling_lr", type=float, default=1e-3)
    parser.add_argument(
        "--shuffle_control",
        action="store_true",
        help="also run every within-model curve with the property labels "
        "permuted, which must abolish the effect",
    )
    parser.add_argument(
        "--max_alignment_axes",
        type=int,
        default=0,
        help="cap on axes entering the cross-model comparison, which re-solves "
        "an assignment per dense-rotation draw; 0 uses every axis",
    )
    parser.add_argument("--n_top", type=int, default=16, help="exemplar axes to log")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--chunk_size", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stage", type=str, choices=["extract", "compare", "all"], default="all"
    )
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
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
ALPHAS = np.linspace(0.0, 1.0, args.n_alphas + 1)
HEADLINE = {
    "orientation": args.osi_thresholds[-1],
    **{
        prop: args.selectivity_thresholds[
            min(2, len(args.selectivity_thresholds) - 1)
        ]
        for prop in OBJECT_PROPERTIES
    },
}


# --------------------------------------------------------------------- models


def model_labels() -> List[str]:
    """Unique label per config.

    Config ``name`` fields are not usable: all five TDANN seeds declare
    ``name: tdann_optimized``, so keying results on it would collapse them.
    """
    if args.names is not None:
        assert len(args.names) == len(args.configs), "one name per config"
        return list(args.names)
    return [f"{Path(c).parent.name}/{Path(c).stem}" for c in args.configs]


def resolve_layers(cfg) -> List[str]:
    """Every layer this architecture exposes, i.e. every BRAIN_MAPPING key.

    No area filtering: the whole point is to extract and compare all of them,
    not just the ones some brain-area assignment happens to pick out. With
    ``--state preactivation`` each key is then mapped to the argument of its
    nonlinearity.
    """
    architecture = cfg.variant.architecture
    assert architecture in BRAIN_MAPPING, (
        f"{architecture} has no BRAIN_MAPPING entry, so its layers can't be "
        "enumerated -- add one, or extract this architecture by hand"
    )
    layers = list(BRAIN_MAPPING[architecture].keys())
    if args.state == "native":
        return layers

    assert architecture in PREACTIVATION_LAYER, (
        f"{architecture} has no pre-activation mapping; add one to "
        "PREACTIVATION_LAYER or pass --state native"
    )
    return [PREACTIVATION_LAYER[architecture](layer) for layer in layers]


def stratified_indices(labels: np.ndarray, n_images: int, seed: int) -> np.ndarray:
    """Discrimination subsample preserving the shape/colour agreement design.

    The split is deliberately unbalanced -- half the images match on both
    properties, the rest split between shape-only, colour-only and neither --
    so a subsample has to respect those proportions.
    """
    from spacetorch.datasets.discrimination import ATTRIBUTE_INDEX

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


def outgoing_weight_norm(model, layer: str) -> Optional[np.ndarray]:
    """Squared norm of each axis's outgoing weight column, if resolvable.

    Downstream usedness is the reference packages' strongest single predictor
    of which axes recur, and it weights an axis's response variance by how much
    the next layer actually reads from it. For a transformer MLP gate the
    reader is ``fc2``; for a pre-ReLU convolutional channel there is no single
    reader, so this returns None and the relevance falls back to variance
    alone.
    """
    from spacetorch.utils.torch_utils import resolve_sequential_module_from_str

    if not layer.endswith("mlp.fc1"):
        return None
    try:
        reader = resolve_sequential_module_from_str(
            model, layer[: -len("fc1")] + "fc2"
        )
    except Exception:
        return None
    weight = reader.weight.detach().cpu().numpy()  # (out_features, n_axes)
    return (weight**2).sum(axis=0)


def extract(label: str, config: str, cache_paths: Dict[str, Path]):
    """Run one model over both stimulus sets, caching every layer at once.

    All requested layers are hooked in a single pass. Reading and decoding the
    images dominates the runtime, so many layers together cost barely more
    than one.

    ``cache_paths`` is keyed directly by layer name (already filtered to
    whichever layers still need (re-)extracting), so there is no separate
    area-to-layer lookup to keep in sync.
    """
    cfg = get_cfg(SimpleNamespace(config=config, opts=args.opts))
    layers = list(cache_paths.keys())
    print(f"\n=== {label}: layers = {', '.join(layers)}", flush=True)

    variant = get_variant(cfg.variant.name)
    variant.set_eval_cfg(cfg)
    variant.set_eval_model(cfg)
    model = variant.eval_model

    gabors = GaborPatches()
    discrimination = DatasetRegistry.get("Discrimination_val_attributes")
    discrimination_indices = stratified_indices(
        discrimination.labels, args.n_images, args.seed
    )

    def stimulated_site(features, labels):
        """Read the Gabor response where the patch actually is.

        A full-field grating can be read as a spatial average, but a patch at
        one token cannot: averaging over 196 tokens mixes the one stimulated
        site with 195 unstimulated ones.
        """
        box = np.tile(np.asarray(gabors.patch_box), (len(features), 1))
        return region_responses(features, box)

    def object_pooled(features, labels):
        boxes = [
            discrimination.object_boxes(0, labels),
            discrimination.object_boxes(1, labels),
        ]
        regions = [region_responses(features, box) for box in boxes]
        # object 0, object 1, background, and the whole image
        return np.stack(
            [
                regions[0],
                regions[1],
                complement_responses(features, boxes),
                channel_responses(features),
            ],
            axis=1,
        )

    common = dict(
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        num_workers=args.num_workers,
    )
    print("-- gabor patches", flush=True)
    grating_responses, grating_labels = get_pooled_features_from_layer(
        model, gabors, layers, stimulated_site, **common
    )
    print("-- discrimination stimuli", flush=True)
    object_responses, object_labels = get_pooled_features_from_layer(
        model, discrimination, layers, object_pooled,
        indices=discrimination_indices, **common
    )

    outgoing = {layer: outgoing_weight_norm(model, layer) for layer in layers}

    for layer, cache_path in cache_paths.items():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            label=label,
            area=layer,
            layer=layer,
            architecture=cfg.variant.architecture,
            outgoing_weight_norm=(
                outgoing[layer] if outgoing[layer] is not None else np.array([])
            ),
            grating=grating_responses[layer].astype(np.float32),
            grating_labels=grating_labels,
            objects=object_responses[layer].astype(np.float32),
            object_labels=object_labels,
            object_indices=discrimination_indices,
        )
        print(
            f"cached {cache_path} ({grating_responses[layer].shape[1]} axes)", flush=True
        )

    del model, variant
    torch.cuda.empty_cache()


# ------------------------------------------------------------------- measuring


class ModelMeasurement:
    """Per-axis selectivity of one model at one layer, and the responses behind it."""

    def __init__(self, cache: Dict[str, np.ndarray]):
        self.label = str(cache["label"])
        self.area = str(cache["area"])
        self.layer = str(cache["layer"])
        self.architecture = str(cache["architecture"])

        self.grating = cache["grating"].astype(np.float64)
        self.grating_labels = cache["grating_labels"]
        self.object_labels = cache["object_labels"]

        # (n_images, {object 0, object 1, background, whole image}, n_axes)
        objects = cache["objects"].astype(np.float64)
        self.occurrences = np.concatenate([objects[:, 0], objects[:, 1]], axis=0)
        self.image_responses = objects[:, 3]

        self.harmonics = GratingHarmonics(
            self.grating_labels[:, 0],
            self.grating_labels[:, 1],
            self.grating_labels[:, 2],
            self.grating_labels[:, 3],
        )
        self.discrimination = DiscriminationResponses.from_pooled(
            [objects[:, 0], objects[:, 1]],
            objects[:, 2],
            self.object_labels,
            position_bins=args.position_bins,
        )

        # weight response variance by how much the next layer reads the axis,
        # which is the reference packages' notion of downstream usedness
        outgoing = cache.get("outgoing_weight_norm")
        self.usage = usage_strata(self.occurrences)
        if outgoing is not None and np.size(outgoing) == self.occurrences.shape[1]:
            self.usage = usage_strata(
                self.occurrences * np.sqrt(np.asarray(outgoing))[None, :]
            )
            self.usedness_definition = "variance_times_outgoing_norm"
        else:
            self.usedness_definition = "variance"
        self._tuning: Optional[Dict[str, np.ndarray]] = None

    @property
    def n_axes(self) -> int:
        return self.occurrences.shape[1]

    @property
    def tuning(self) -> Dict[str, np.ndarray]:
        if self._tuning is None:
            self._tuning = self.discrimination.property_tuning()
        return self._tuning

    @property
    def object_labels_by_property(self) -> Dict[str, np.ndarray]:
        return {
            "shape": self.discrimination.shape_labels,
            "color": self.discrimination.color_labels,
            "position": self.discrimination.position_labels(),
        }

    def selectivity(self, prop: str) -> np.ndarray:
        """Per-axis selectivity for one property, in the native basis."""
        if prop == "orientation":
            return self.harmonics.osi(self.grating)
        return peak_selectivity(self.occurrences, self.object_labels_by_property[prop])

    def thresholds(self, prop: str) -> List[float]:
        return (
            args.osi_thresholds if prop == "orientation" else args.selectivity_thresholds
        )

    def responses_for(self, prop: str) -> np.ndarray:
        return self.grating if prop == "orientation" else self.occurrences

    def tuning_table(self) -> pd.DataFrame:
        orientation = self.harmonics.tuning(self.grating)
        table = pd.DataFrame(
            {
                "group": args.group,
                "area": self.area,
                "model": self.label,
                "layer": self.layer,
                "axis": np.arange(self.n_axes),
                "relevance": self.usage["relevance"],
                "relevance_percentile": self.usage["relevance_percentile"],
                "relevance_stratum": self.usage["relevance_stratum"],
            }
        )
        for key in (
            "orientation_selectivity_index",
            "preferred_orientation_deg",
            "preferred_spatial_frequency",
            "preferred_color",
            "peak_phase_modulation",
        ):
            table[key] = orientation[key]
        table["usedness_definition"] = self.usedness_definition
        for source in (
            self.tuning,
            self.discrimination.relation_tuning(),
            self.discrimination.feature_relation_regression(),
        ):
            for key, values in source.items():
                table[key] = values
        for prop in OBJECT_PROPERTIES:
            table[f"{prop}_selectivity"] = self.selectivity(prop)
        return table


def make_statistic(measurement: ModelMeasurement, prop: str, labels: np.ndarray):
    """Closure re-measuring a property's selectivity summary on rotated axes."""
    thresholds = measurement.thresholds(prop)

    if prop == "orientation":
        harmonics = measurement.harmonics

        def statistic(rotated: np.ndarray) -> Dict[str, float]:
            return selectivity_summary(harmonics.osi(rotated), thresholds)

    else:

        def statistic(rotated: np.ndarray) -> Dict[str, float]:
            return selectivity_summary(peak_selectivity(rotated, labels), thresholds)

    return statistic


def ceiling(
    measurement: ModelMeasurement, prop: str, labels: np.ndarray
) -> Dict[str, float]:
    """Best selective-axis count any basis achieves, via ascent on SO(n).

    The count is not differentiable, so a sigmoid-softened count is optimized
    while the hard statistic is monitored, and the identity is always among the
    candidates so the result really does bound the native basis. Orientation is
    excluded: its index selects the best frequency/colour cell per axis, an
    argmax that does not pass a gradient.
    """
    if args.ceiling_steps <= 0 or prop == "orientation":
        return {}

    responses = measurement.occurrences
    thresholds = measurement.thresholds(prop)
    centered = responses - responses.mean(axis=0, keepdims=True)
    headline = HEADLINE[prop]

    unique, inverse = np.unique(labels, return_inverse=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    membership = torch.zeros((len(inverse), len(unique)), device=device)
    membership[torch.arange(len(inverse)), torch.as_tensor(inverse, device=device)] = 1.0
    counts = membership.sum(dim=0)

    def objective(rotated):
        # eta-squared is the differentiable stand-in for the localizer contrast
        return soft_count(torch_eta_squared(rotated, membership, counts), 0.25)

    def monitor(rotation: np.ndarray) -> float:
        summary = selectivity_summary(
            peak_selectivity(centered @ rotation, labels), [headline]
        )
        return summary.get(f"n_selective@{headline:g}", -np.inf)

    rotation, _ = optimal_rotation(
        responses, objective, n_steps=args.ceiling_steps,
        learning_rate=args.ceiling_lr, seed=args.seed, monitor=monitor,
    )
    hard = selectivity_summary(peak_selectivity(centered @ rotation, labels), thresholds)
    return {f"ceiling_{key}": value for key, value in hard.items()}


def within_model_rows(measurement: ModelMeasurement) -> List[dict]:
    """Selectivity against rotation, for every property."""
    rows: List[dict] = []
    rng = np.random.default_rng(args.seed)

    def add(prop, statistic, alpha, value, sd, stimuli, analysis="rotation_curve"):
        rows.append(
            {
                "group": args.group,
                "area": measurement.area,
                "analysis": analysis,
                "model_a": measurement.label,
                "model_b": "",
                "layer_a": measurement.layer,
                "layer_b": "",
                "property": prop,
                "stimuli": stimuli,
                "statistic": statistic,
                "alpha": alpha,
                "value": value,
                "sd": sd,
                "n_axes": measurement.n_axes,
            }
        )

    for prop in PROPERTIES:
        stimuli = "gabor_patches" if prop == "orientation" else "discrimination"
        labels = (
            None if prop == "orientation" else measurement.object_labels_by_property[prop]
        )
        print(f"-- {measurement.label} | {prop}", flush=True)

        curve = axis_perturbation_curve(
            measurement.responses_for(prop),
            make_statistic(measurement, prop, labels),
            ALPHAS,
            n_rotations=args.n_rotations,
            seed=args.seed,
        )
        for alpha, summary in curve.items():
            for statistic, (value, sd) in summary.items():
                add(prop, statistic, alpha, value, sd, stimuli)

        for statistic, value in ceiling(measurement, prop, labels).items():
            add(prop, statistic, np.nan, value, 0.0, stimuli, analysis="ceiling")

        if args.shuffle_control and labels is not None:
            shuffled = labels[rng.permutation(len(labels))]
            curve = axis_perturbation_curve(
                measurement.occurrences,
                make_statistic(measurement, prop, shuffled),
                ALPHAS,
                n_rotations=args.n_rotations,
                seed=args.seed,
                verbose=False,
            )
            for alpha, summary in curve.items():
                for statistic, (value, sd) in summary.items():
                    add(prop, statistic, alpha, value, sd, stimuli, "shuffled_labels")

    return rows


# ------------------------------------------------------- cross-model identity


def _halves(n_stimuli: int) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic fit/score split of the stimuli."""
    order = np.random.default_rng(args.seed).permutation(n_stimuli)
    return order[: n_stimuli // 2], order[n_stimuli // 2 :]


def _stratum_mask(usage: Dict[str, np.ndarray], stratum: str) -> np.ndarray:
    if stratum == "all":
        return np.ones(len(usage["relevance"]), dtype=bool)
    return np.asarray(usage["relevance_stratum"]) == stratum


def _centered(matrix: np.ndarray) -> np.ndarray:
    return matrix - matrix.mean(axis=0, keepdims=True)


def identity_rows(
    first: ModelMeasurement, second: ModelMeasurement
) -> Tuple[List[dict], pd.DataFrame]:
    """Discrete physical-identity statistics for one model pair.

    Raw matched correlation alone does not separate a native basis from a
    dense rotated one -- the reference package reports a rotated basis winning
    on correlation at conv1 while losing decisively on mutual-nearest-neighbour
    matching and cycle closure. So every statistic here is computed both in the
    native basis and after replacing the first model's basis by a dense
    rotation, and the comparison is repeated within usedness strata, since
    usedness is that package's strongest predictor of which axes recur.
    """
    rows: List[dict] = []
    detail_rows = []
    rng = np.random.default_rng(args.seed)

    subset_a = subsample_axes(first.n_axes, args.max_alignment_axes)
    subset_b = subsample_axes(second.n_axes, args.max_alignment_axes)
    axes_a = np.arange(first.n_axes) if subset_a is None else subset_a
    axes_b = np.arange(second.n_axes) if subset_b is None else subset_b

    def add(stimuli, stratum, statistic, basis, value, sd, n_axes):
        rows.append(
            {
                "group": args.group,
                "area": first.area,
                "analysis": f"identity_{basis}",
                "model_a": first.label,
                "model_b": second.label,
                "layer_a": first.layer,
                "layer_b": second.layer,
                "property": stratum,
                "stimuli": stimuli,
                "statistic": statistic,
                "alpha": 0.0 if basis == "native" else 1.0,
                "value": value,
                "sd": sd,
                "n_axes": n_axes,
            }
        )

    for stimuli, responses_a, responses_b in (
        ("gabor_patches", first.grating, second.grating),
        ("discrimination", first.image_responses, second.image_responses),
    ):
        matrix_a = _centered(responses_a[:, axes_a])
        matrix_b = _centered(responses_b[:, axes_b])
        fit, test = _halves(len(matrix_a))

        usage_a = usage_strata(matrix_a)
        rotated = [
            matrix_a @ random_orthogonal(matrix_a.shape[1], rng)
            for _ in range(args.n_dense_rotations)
        ]
        usage_rotated = [usage_strata(r) for r in rotated]

        for stratum in STRATA:
            keep = _stratum_mask(usage_a, stratum)
            if keep.sum() < 4:
                continue
            native = identity_statistics(
                matrix_a[np.ix_(fit, keep)], matrix_b[fit],
                matrix_a[np.ix_(test, keep)], matrix_b[test],
            )
            dense = []
            for rotation, usage in zip(rotated, usage_rotated):
                dense_keep = _stratum_mask(usage, stratum)
                if dense_keep.sum() < 4:
                    continue
                dense.append(
                    identity_statistics(
                        rotation[np.ix_(fit, dense_keep)], matrix_b[fit],
                        rotation[np.ix_(test, dense_keep)], matrix_b[test],
                    )
                )

            for statistic in (
                "matched_correlation",
                "semi_matched_correlation",
                "mutual_nearest_fraction",
                "assignment_stability",
            ):
                add(stimuli, stratum, statistic, "native",
                    native[statistic], 0.0, int(keep.sum()))
                if dense:
                    values = [d[statistic] for d in dense]
                    add(stimuli, stratum, statistic, "dense",
                        float(np.mean(values)),
                        float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                        int(keep.sum()))

        if stimuli == "gabor_patches":
            partners = identity_statistics(
                matrix_a[fit], matrix_b[fit], matrix_a[test], matrix_b[test]
            )["assignment"]
            osi_a = first.selectivity("orientation")[axes_a]
            osi_b = second.selectivity("orientation")[axes_b][partners]
            pref_a = first.harmonics.tuning(first.grating)[
                "preferred_orientation_deg"
            ][axes_a]
            pref_b = second.harmonics.tuning(second.grating)[
                "preferred_orientation_deg"
            ][axes_b][partners]
            detail_rows.append(
                pd.DataFrame(
                    {
                        "group": args.group,
                        "area": first.area,
                        "model_a": first.label,
                        "model_b": second.label,
                        "axis_a": axes_a,
                        "axis_b": axes_b[partners],
                        "relevance_stratum_a": usage_a["relevance_stratum"],
                        "osi_a": osi_a,
                        "osi_b": osi_b,
                        "preferred_a": pref_a,
                        "preferred_b": pref_b,
                        "orientation_difference_deg": axial_difference(pref_a, pref_b),
                    }
                )
            )

            for stratum in STRATA:
                keep = _stratum_mask(usage_a, stratum)
                selective = (
                    keep
                    & (osi_a >= HEADLINE["orientation"])
                    & (osi_b >= HEADLINE["orientation"])
                )
                if selective.sum() < 2:
                    continue
                agreement = float(
                    (
                        axial_difference(pref_a[selective], pref_b[selective])
                        <= args.orientation_tolerance_deg + 1e-9
                    ).mean()
                )
                rows.append(
                    {
                        "group": args.group,
                        "area": first.area,
                        "analysis": "preferred_orientation",
                        "model_a": first.label,
                        "model_b": second.label,
                        "layer_a": first.layer,
                        "layer_b": second.layer,
                        "property": stratum,
                        "stimuli": "gabor_patches",
                        "statistic": "agreement",
                        "alpha": 0.0,
                        "value": agreement,
                        "sd": 0.0,
                        "n_axes": int(selective.sum()),
                    }
                )

    detail = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()
    return rows, detail


def cycle_rows(triple: List[ModelMeasurement]) -> List[dict]:
    """Transitivity of the physical map around a loop of three models.

    Pairwise similarity can be high without the correspondence composing;
    closure asks whether following A to B to C and back returns each axis to
    itself. This is the statistic on which the reference package finds native
    channels beating a dense rotated basis most decisively, so it is run
    against the same dense control and within usedness strata.
    """
    rows: List[dict] = []
    rng = np.random.default_rng(args.seed)
    first = triple[0]

    for stimuli, attribute in (
        ("gabor_patches", "grating"),
        ("discrimination", "image_responses"),
    ):
        subsets = [subsample_axes(m.n_axes, args.max_alignment_axes) for m in triple]
        matrices = [
            _centered(
                getattr(m, attribute)[:, np.arange(m.n_axes) if sub is None else sub]
            )
            for m, sub in zip(triple, subsets)
        ]
        fit, _ = _halves(len(matrices[0]))
        usage = usage_strata(matrices[0])

        def closure(mats) -> np.ndarray:
            """Per-axis indicator of whether the A->B->C->A loop closes."""
            assignments = [
                one_to_one_assignment(
                    correlation_matrix(mats[i][fit], mats[(i + 1) % 3][fit])
                )
                for i in range(3)
            ]
            index = np.arange(len(assignments[0]))
            current = index.copy()
            for assignment in assignments:
                current = assignment[current]
            return current == index

        native = closure(matrices)
        dense = []
        for _ in range(args.n_dense_rotations):
            rotated = [matrices[0] @ random_orthogonal(matrices[0].shape[1], rng)]
            rotated += matrices[1:]
            dense.append(closure(rotated))

        for stratum in STRATA:
            keep = _stratum_mask(usage, stratum)
            if keep.sum() < 4:
                continue
            for basis, values in (("native", [native]), ("dense", dense)):
                scores = [float(v[keep].mean()) for v in values]
                rows.append(
                    {
                        "group": args.group,
                        "area": first.area,
                        "analysis": f"cycle_{basis}",
                        "model_a": first.label,
                        "model_b": "+".join(m.label for m in triple[1:]),
                        "layer_a": first.layer,
                        "layer_b": "",
                        "property": stratum,
                        "stimuli": stimuli,
                        "statistic": "cycle_closure",
                        "alpha": 0.0 if basis == "native" else 1.0,
                        "value": float(np.mean(scores)),
                        "sd": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                        "n_axes": int(keep.sum()),
                    }
                )

    return rows


# -------------------------------------------------------------------- plotting


def plot(results: pd.DataFrame, area: str, save_path: Path) -> None:
    area_results = results[results.area == area]
    fig, axs = plt.subplots(1, 3, figsize=(8.5, 2.2), constrained_layout=True)

    # (a) selective axes against rotation, per property, averaged over models
    panel = area_results[
        (area_results.analysis == "rotation_curve")
        & (area_results.statistic.str.startswith("n_selective@"))
    ]
    for prop in PROPERTIES:
        subset = panel[
            (panel["property"] == prop)
            & (panel.statistic == f"n_selective@{HEADLINE[prop]:g}")
        ]
        if subset.empty:
            continue
        curve = subset.groupby("alpha").value.mean()
        native = curve.get(0.0, np.nan)
        if not np.isfinite(native) or native <= 0:
            continue
        axs[0].plot(
            curve.index, 100 * curve / native, "o-", ms=3,
            color=PROPERTY_COLORS[prop], label=prop,
        )
    axs[0].set_xlabel(r"Axis rotation ($\alpha$)")
    axs[0].set_ylabel("Selective axes (% of native)")
    axs[0].axhline(100, color="gray", lw=0.4, ls=":")
    axs[0].legend(fontsize=5, frameon=False)

    # (b) the same, but with the labels shuffled: this must be flat
    shuffled = area_results[
        (area_results.analysis == "shuffled_labels")
        & (area_results.statistic.str.startswith("n_selective@"))
    ]
    if not shuffled.empty:
        for prop in OBJECT_PROPERTIES:
            subset = shuffled[
                (shuffled["property"] == prop)
                & (shuffled.statistic == f"n_selective@{HEADLINE[prop]:g}")
            ]
            if subset.empty:
                continue
            curve = subset.groupby("alpha").value.mean()
            native = curve.get(0.0, np.nan)
            if np.isfinite(native) and native > 0:
                axs[1].plot(
                    curve.index, 100 * curve / native, "o-", ms=3,
                    color=PROPERTY_COLORS[prop], label=prop,
                )
        axs[1].set_xlabel(r"Axis rotation ($\alpha$)")
        axs[1].set_ylabel("Shuffled labels (% of native)")
        axs[1].legend(fontsize=5, frameon=False)
    else:
        axs[1].axis("off")

    # (c) native versus dense rotated basis on the discrete identity measures,
    # restricted to the most-used decile where the reference effect is largest
    statistics = ["matched_correlation", "mutual_nearest_fraction",
                  "assignment_stability", "cycle_closure"]
    labels, native_vals, dense_vals = [], [], []
    for statistic in statistics:
        kind = "cycle" if statistic == "cycle_closure" else "identity"
        nat = area_results[
            (area_results.analysis == f"{kind}_native")
            & (area_results.statistic == statistic)
            & (area_results["property"] == "top10")
        ]
        den = area_results[
            (area_results.analysis == f"{kind}_dense")
            & (area_results.statistic == statistic)
            & (area_results["property"] == "top10")
        ]
        if nat.empty:
            continue
        labels.append(statistic.replace("_", "\n"))
        native_vals.append(nat.value.mean())
        dense_vals.append(den.value.mean() if not den.empty else np.nan)
    if labels:
        x = np.arange(len(labels))
        axs[2].bar(x - 0.2, native_vals, width=0.4, color="black", label="native")
        axs[2].bar(x + 0.2, dense_vals, width=0.4, color="firebrick", label="dense")
        axs[2].set_xticks(x)
        axs[2].set_xticklabels(labels, fontsize=4)
        axs[2].set_ylabel("Top-decile identity")
        axs[2].legend(fontsize=5, frameon=False)
    else:
        axs[2].axis("off")

    fig.suptitle(f"{args.group} | {area}", fontsize=7, y=1.06)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------------ main


def main():
    labels = model_labels()

    output_dir = Path(args.output_dir) / args.group
    cache_dir = output_dir / "responses"
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_cfg = get_cfg(SimpleNamespace(config=args.configs[0], opts=args.opts))
    layers = resolve_layers(reference_cfg)

    cache_paths = {
        (label, layer): cache_dir / f"{label.replace('/', '__')}__{layer}.npz"
        for label in labels
        for layer in layers
    }

    if args.stage in ("extract", "all"):
        for label, config in zip(labels, args.configs):
            if args.only is not None and label != args.only:
                continue
            todo = {
                layer: cache_paths[(label, layer)]
                for layer in layers
                if args.overwrite or not cache_paths[(label, layer)].exists()
            }
            if not todo:
                print(f"cached, skipping: {label}")
                continue
            extract(label, config, todo)
        if args.stage == "extract":
            return

    results: List[dict] = []
    tuning_tables, detail_tables = [], []

    for layer in layers:
        print(f"\n########## {args.group} | {layer}", flush=True)
        measurements = []
        for label in labels:
            path = cache_paths[(label, layer)]
            assert path.exists(), f"missing cache {path}; run --stage extract first"
            measurements.append(ModelMeasurement(dict(np.load(path))))

        reference = measurements[0].object_labels
        for measurement in measurements[1:]:
            assert np.array_equal(measurement.object_labels, reference), (
                "cached models saw different discrimination stimuli; re-extract "
                "with --overwrite so responses are row-aligned"
            )

        for measurement in measurements:
            print(
                f"-- {measurement.label}: {measurement.n_axes} axes at {measurement.layer}"
            )
            tuning_tables.append(measurement.tuning_table())
            results.extend(within_model_rows(measurement))
            for prop in PROPERTIES:
                scores = measurement.selectivity(prop)
                best = top_axes(scores, args.n_top)
                print(f"   top {prop} axes {best[:8]} (max {np.nanmax(scores):.3f})")

        for first, second in itertools.combinations(measurements, 2):
            print(f"-- identity {first.label} -> {second.label}", flush=True)
            pair_rows, detail = identity_rows(first, second)
            results.extend(pair_rows)
            if not detail.empty:
                detail_tables.append(detail)

        for triple in itertools.combinations(measurements, 3):
            print(f"-- cycle {'/'.join(m.label for m in triple)}", flush=True)
            results.extend(cycle_rows(list(triple)))

    frame = pd.DataFrame(results)[RESULT_COLUMNS]
    frame.to_csv(output_dir / "privileged_axes_results.csv", index=False)
    pd.concat(tuning_tables, ignore_index=True).to_csv(
        output_dir / "axis_tuning.csv", index=False
    )
    if detail_tables:
        pd.concat(detail_tables, ignore_index=True).to_csv(
            output_dir / "matched_axes.csv", index=False
        )

    with open(output_dir / "privileged_axes_summary.json", "w") as handle:
        json.dump(
            {
                "group": args.group,
                "models": labels,
                "layers": layers,
                "state": args.state,
                "stimuli": ["gabor_patches", "discrimination"],
                "alphas": ALPHAS.tolist(),
                "n_rotations": args.n_rotations,
                "n_dense_rotations": args.n_dense_rotations,
                "osi_thresholds": list(args.osi_thresholds),
                "selectivity_thresholds": list(args.selectivity_thresholds),
                "headline_thresholds": HEADLINE,
                "orientation_uniform_baseline": uniform_orientation_baseline(
                    measurements[0].grating_labels[:, 0], args.orientation_tolerance_deg
                ),
            },
            handle,
            indent=2,
        )

    for layer in layers:
        plot(frame, layer, output_dir / f"privileged_axes_{layer}.png")

    curve = frame[frame.analysis == "rotation_curve"]
    print("\nnative vs fully rotated, averaged over models:")
    for prop in PROPERTIES:
        for statistic in (f"n_selective@{HEADLINE[prop]:g}", "gini"):
            rows = curve[(curve["property"] == prop) & (curve.statistic == statistic)]
            if rows.empty:
                continue
            a = rows[rows.alpha == 0].value.mean()
            b = rows[rows.alpha == 1].value.mean()
            print(f"  {prop:12s} {statistic:16s} {a:8.3f} -> {b:8.3f}")
    print(f"\nSaved results to {output_dir}")


if __name__ == "__main__":
    main()
