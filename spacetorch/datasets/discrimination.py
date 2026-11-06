from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pickle

import numpy as np
import torch
import torchvision
import xarray as xr
from PIL import Image
from torch.utils.data import Dataset

from spacetorch.privileged_axes import (
    category_profile,
    complement_responses,
    eta_squared,
    region_responses,
    signed_contrast,
)


DISCRIMINATION_TRANSFORMS = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
    ]
)

# The stimuli are already 224x224, and the generative metadata locates each
# object on the 14x14 patch grid of that exact image. Resizing or cropping
# would silently invalidate those coordinates, so the attribute dataset uses
# the images as supplied and only normalizes them.
DISCRIMINATION_TRANSFORMS_EXACT = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


NUM_TRAIN_IMAGES = 6400
NUM_VALIDATION_IMAGES = 6400

# the stimuli were generated against a 14 x 14 grid of 16-pixel patches, and
# every object occupies one 2 x 2 window of it
PATCH_GRID = 14
OBJECT_PATCHES = 2


class DiscriminationData(torchvision.datasets.ImageFolder):
    """Same/Different Task (Lepori et al., 2024)"""

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


# Column layout of the label vector returned by DiscriminationAttributes. The
# names mirror the object metadata table of the source analyses, extended with
# the object colors in RGB so that color can be treated either categorically or
# as a continuous descriptor.
ATTRIBUTE_COLUMNS: List[str] = [
    "index",
    "sd_label",
    "shape0",
    "shape1",
    "color0",
    "color1",
    "shape_same",
    "color_same",
    "block0_y",
    "block0_x",
    "block1_y",
    "block1_x",
    "distance_patches",
    "color0_r",
    "color0_g",
    "color0_b",
    "color1_r",
    "color1_g",
    "color1_b",
]

ATTRIBUTE_INDEX: Dict[str, int] = {name: i for i, name in enumerate(ATTRIBUTE_COLUMNS)}


def parse_color_name(color_name: str) -> Tuple[int, int, int]:
    """Recover the generating RGB triple from a color name.

    Color names look like ``mean76-175-80_var10``: the mean foreground RGB
    followed by the per-pixel noise variance used when the object was drawn.
    """
    mean_part = color_name.split("_")[0]
    assert mean_part.startswith("mean"), f"unexpected color name {color_name}"
    red, green, blue = mean_part[len("mean") :].split("-")
    return int(red), int(green), int(blue)


class DiscriminationAttributes(Dataset):
    """Discrimination stimuli labelled with their generative object attributes.

    ``DiscriminationData`` exposes only the binary same/different class, which
    is enough to decode the task but not to ask what a single axis of the
    representation is tuned to. The generator, however, saved everything needed
    for that question in ``datadict.pkl``: for each image, the shape index, the
    color, and the four patch tokens occupied by each of the two objects. This
    dataset reads that file and returns those attributes as the target, in the
    same way ``SineGrating2019`` returns the grating parameters.

    The two objects are ordered by their top-left token so that "object 0" means
    the same thing in every image, and the derived equality indicators
    (``shape_same``, ``color_same``) rather than the raw category numbers define
    the relation conditions.
    """

    def __init__(
        self,
        discrimination_dir: str,
        transforms: Optional[torchvision.transforms.Compose] = None,
    ):
        self.transforms = transforms
        self.root = Path(discrimination_dir)
        self.split = self.root.name

        with open(self.root / "datadict.pkl", "rb") as handle:
            datadict = pickle.load(handle)

        # datadict also describes split-out "different-shape" / "different-color"
        # directories that are not shipped with the dataset; keep the entries
        # whose image is actually on disk
        entries = []
        for key, record in datadict.items():
            path = self.root.parent / key
            if path.exists():
                entries.append((key, path, record))
        entries.sort(key=lambda entry: entry[0])
        assert len(entries), f"no discrimination images found under {self.root}"

        self.keys = [key for key, _, _ in entries]
        self.file_list = [path for _, path, _ in entries]

        # a stable category ordering, shared across splits by construction since
        # both splits use the same sixteen colors
        self.color_names = sorted(
            {record["c1"] for _, _, record in entries}
            | {record["c2"] for _, _, record in entries}
        )
        color_to_index = {name: i for i, name in enumerate(self.color_names)}

        self.labels = np.zeros([len(entries), len(ATTRIBUTE_COLUMNS)], dtype=float)
        for image_idx, (_, _, record) in enumerate(entries):
            first = _object_record(record, "1", color_to_index)
            second = _object_record(record, "2", color_to_index)
            # order the two objects by top-left token position
            first, second = sorted([first, second], key=lambda obj: (obj["block_y"], obj["block_x"]))

            distance = np.hypot(
                second["block_y"] - first["block_y"], second["block_x"] - first["block_x"]
            )

            self.labels[image_idx] = [
                image_idx,
                float(record["sd-label"]),
                first["shape"],
                second["shape"],
                first["color"],
                second["color"],
                float(first["shape"] == second["shape"]),
                float(first["color"] == second["color"]),
                first["block_y"],
                first["block_x"],
                second["block_y"],
                second["block_x"],
                distance,
                *first["rgb"],
                *second["rgb"],
            ]

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = Image.open(self.file_list[idx]).convert("RGB")
        target = self.labels[idx, :]
        if self.transforms:
            img = self.transforms(img)

        return img, target

    def object_boxes(self, object_idx: int, labels: Optional[np.ndarray] = None) -> np.ndarray:
        """Fractional (top, left, bottom, right) extent of one object per image.

        Coordinates are fractions of the image, so they can be mapped onto a
        feature map of any spatial resolution.
        """
        labels = self.labels if labels is None else np.asarray(labels)
        block_y = labels[:, ATTRIBUTE_INDEX[f"block{object_idx}_y"]]
        block_x = labels[:, ATTRIBUTE_INDEX[f"block{object_idx}_x"]]
        return np.stack(
            [
                block_y / PATCH_GRID,
                block_x / PATCH_GRID,
                (block_y + OBJECT_PATCHES) / PATCH_GRID,
                (block_x + OBJECT_PATCHES) / PATCH_GRID,
            ],
            axis=1,
        )


def _object_record(record: dict, suffix: str, color_to_index: Dict[str, int]) -> dict:
    tokens = sorted(record[f"pos{suffix}"])
    color_name = record[f"c{suffix}"]
    return {
        "shape": float(record[f"s{suffix}"]),
        "color": float(color_to_index[color_name]),
        "rgb": [float(channel) for channel in parse_color_name(color_name)],
        "block_y": float(tokens[0] // PATCH_GRID),
        "block_x": float(tokens[0] % PATCH_GRID),
    }


class DiscriminationResponses:
    """Per-axis object tuning on the discrimination stimuli.

    Each object occurrence contributes one response per axis: the mean over the
    feature-map region the object covers, the analogue of averaging a
    transformer gate over the object's four tokens. Because every image supplies
    two occurrences, an axis's shape or color tuning is measured while the other
    property, the object's position, and the identity of the partner object all
    vary freely.

    The resulting marginal explained variances are empirical one-factor
    associations. Shape and color are not fully crossed in this stimulus set --
    most images match on both properties -- so the two scores should be read
    side by side rather than as an additive decomposition.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        dataset: DiscriminationAttributes,
        position_bins: int = 4,
    ):
        labels = np.asarray(labels)
        boxes = [dataset.object_boxes(0, labels), dataset.object_boxes(1, labels)]
        self._setup(
            [region_responses(features, box) for box in boxes],
            complement_responses(features, boxes),
            labels,
            position_bins,
        )

    @classmethod
    def from_pooled(
        cls,
        object_responses: List[np.ndarray],
        background: np.ndarray,
        labels: np.ndarray,
        position_bins: int = 4,
    ) -> "DiscriminationResponses":
        """Build from responses that were already pooled during extraction.

        The orbit analysis reduces feature maps to object-region means chunk by
        chunk, so by the time tuning is measured the maps no longer exist. This
        entry point takes the pooled arrays directly and gives identical
        results to the feature-map constructor.
        """
        responses = cls.__new__(cls)
        responses._setup(object_responses, background, np.asarray(labels), position_bins)
        return responses

    def _setup(
        self,
        object_responses: List[np.ndarray],
        background: np.ndarray,
        labels: np.ndarray,
        position_bins: int,
    ) -> None:
        self.labels = labels
        self.position_bins = position_bins
        self.background = background

        # image-level endpoint: the axis averaged over both object regions
        self.image_responses = 0.5 * (object_responses[0] + object_responses[1])

        # one row per object occurrence, both objects of an image stacked
        self._data = xr.DataArray(
            data=np.concatenate(object_responses, axis=0),
            coords={
                "shapes": ("occurrence_idx", self._stack("shape0", "shape1")),
                "colors": ("occurrence_idx", self._stack("color0", "color1")),
                "block_y": ("occurrence_idx", self._stack("block0_y", "block1_y")),
                "block_x": ("occurrence_idx", self._stack("block0_x", "block1_x")),
            },
            dims=["occurrence_idx", "axis_idx"],
        )

    def _stack(self, first_column: str, second_column: str) -> np.ndarray:
        return np.concatenate(
            [
                self.labels[:, ATTRIBUTE_INDEX[first_column]],
                self.labels[:, ATTRIBUTE_INDEX[second_column]],
            ]
        )

    def __len__(self) -> int:
        return self._data.sizes["axis_idx"]

    @property
    def occurrence_responses(self) -> np.ndarray:
        """(2 * n_images, n_axes) responses, one row per object occurrence."""
        return self._data.values

    @property
    def shape_labels(self) -> np.ndarray:
        """Shape category of each object occurrence."""
        return self._data.shapes.values

    @property
    def color_labels(self) -> np.ndarray:
        """Color category of each object occurrence."""
        return self._data.colors.values

    @property
    def shape_same(self) -> np.ndarray:
        return self.labels[:, ATTRIBUTE_INDEX["shape_same"]].astype(bool)

    @property
    def color_same(self) -> np.ndarray:
        return self.labels[:, ATTRIBUTE_INDEX["color_same"]].astype(bool)

    @property
    def is_same(self) -> np.ndarray:
        return self.labels[:, ATTRIBUTE_INDEX["sd_label"]].astype(bool)

    @property
    def distance_patches(self) -> np.ndarray:
        return self.labels[:, ATTRIBUTE_INDEX["distance_patches"]]

    def position_labels(self, axis: Optional[str] = None) -> np.ndarray:
        """Coarse positional category of each object occurrence.

        The 13 x 13 possible top-left corners are binned onto a
        ``position_bins`` grid, so that every cell holds enough occurrences for
        a stable category mean. ``axis`` selects the row or column marginal.
        """
        rows = np.digitize(
            self._data.block_y.values,
            np.linspace(0, PATCH_GRID - OBJECT_PATCHES, self.position_bins + 1)[1:-1],
        )
        cols = np.digitize(
            self._data.block_x.values,
            np.linspace(0, PATCH_GRID - OBJECT_PATCHES, self.position_bins + 1)[1:-1],
        )
        if axis == "y":
            return rows
        if axis == "x":
            return cols
        return rows * self.position_bins + cols

    def property_tuning(self) -> Dict[str, np.ndarray]:
        """Marginal explained variance of each object property, per axis."""
        responses = self.occurrence_responses
        shapes = self.shape_labels
        colors = self.color_labels

        return {
            "shape_eta2": eta_squared(responses, shapes),
            "color_eta2": eta_squared(responses, colors),
            "position_eta2": eta_squared(responses, self.position_labels()),
            "position_y_eta2": eta_squared(responses, self.position_labels("y")),
            "position_x_eta2": eta_squared(responses, self.position_labels("x")),
            "object_identity_eta2": eta_squared(
                responses, shapes * 100.0 + colors
            ),
            "mean_object_response": responses.mean(axis=0),
            "object_minus_background": (
                responses.mean(axis=0) - np.nanmean(self.background, axis=0)
            ),
        }

    def relation_tuning(self) -> Dict[str, np.ndarray]:
        """Signed same/different response contrasts, per axis."""
        delta, dprime = signed_contrast(self.image_responses, self.is_same)
        return {
            "relation_delta_same_minus_different": delta,
            "relation_dprime": dprime,
        }

    def feature_relation_regression(self) -> Dict[str, np.ndarray]:
        """Fixed-design regression of the image-level response on the relation.

        .. math::
            e_i = \\beta_0 + \\beta_s s_i + \\beta_c c_i + \\beta_{sc} s_i c_i
                  + \\beta_d \\tilde d_i + \\beta_{d^2} \\tilde d_i^2 + \\epsilon_i

        with :math:`s_i` and :math:`c_i` the shape and color agreement
        indicators and :math:`\\tilde d_i` the standardized separation of the
        two objects. ``beta_shape`` is therefore the shape-match contrast at
        color mismatch; at color match it is ``beta_shape + beta_shape_color``.
        The distance terms are included so that neither match contrast can be
        produced by the objects simply sitting closer together.
        """
        shape_same = self.shape_same.astype(float)
        color_same = self.color_same.astype(float)
        distance = self.distance_patches
        distance = (distance - distance.mean()) / distance.std()

        design = np.stack(
            [
                np.ones_like(shape_same),
                shape_same,
                color_same,
                shape_same * color_same,
                distance,
                distance**2,
            ],
            axis=1,
        )
        coefficients, *_ = np.linalg.lstsq(design, self.image_responses, rcond=None)

        names = [
            "beta_intercept",
            "beta_shape",
            "beta_color",
            "beta_shape_color",
            "beta_distance",
            "beta_distance2",
        ]
        return dict(zip(names, coefficients))

    def category_profile(self, axis_idx: int, prop: str) -> Tuple[np.ndarray, ...]:
        """Per-category response profile of a single axis.

        Returns:
            (categories, means, counts, sems), with the responses standardized
            within the axis for display only.
        """
        responses = self.occurrence_responses[:, axis_idx]
        standardized = (responses - responses.mean()) / responses.std()

        labels = {
            "shape": self.shape_labels,
            "color": self.color_labels,
            "position": self.position_labels(),
        }[prop]

        return category_profile(standardized, labels)
