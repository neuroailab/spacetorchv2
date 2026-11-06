"""Contour-detection stimuli: a field of Gabor elements, with or without a path.

Each stimulus is a field of oriented elements on a grid. In the *present*
image a subset of them are aligned into a smooth contour; in the *absent*
image the same field is drawn with those elements randomly oriented. The task
a probe is asked to perform is simply whether a contour is there.

The two images of a pair are generated from one seed, so they share the same
background layout and differ only in the alignment of the contour elements.
That is what makes the pair a controlled comparison, and it is why ``seeds``
is exposed separately: it is the grouping variable a cross-validated probe
should split on, so that a pair is never divided between training and test.
Not because the background leaks the answer -- it appears once with each
label, so it is orthogonal to it -- but because a probe whose training set
contains the partner image is being asked to detect a contour in a field it
has already seen, which is the weaker of the two questions one could ask.

Each split directory -- ``train`` and ``val`` -- holds its own
``metadata.csv`` and its own ``with_contour`` / ``without_contour`` folders.
Seeds do not overlap between the splits, so no background is shared across
them and a probe fit on one is genuinely tested on novel fields by the other.

The metadata carries the generative parameters, recorded here per image so
that accuracy can be broken down by them. The design is fully crossed with
one seed per cell: six relative spacings by four eccentricities by five
contour lengths by four orientations is 480 cells, which is exactly the
number of pairs in each split. Every marginal is therefore balanced and
chance is exactly 0.5 overall and within any level.

Two of the recorded columns are not free factors and are kept for
completeness rather than for slicing. ``skew_angle_deg`` is a one-to-one
function of ``relative_spacing``, so breaking accuracy down by it would
reproduce the spacing breakdown under a different name. And
``contour_length_achieved`` differs from ``contour_length_target`` for about
one stimulus in ten, the placement having failed to fit the requested number
of elements; the achieved length is what is actually in the image, so that
is the one to slice on when the distinction matters.
"""

from pathlib import Path
from typing import List

import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import Dataset

# The stimuli are already 224x224 and the element grid is laid out in those
# exact pixels, so they are used as supplied and only normalized. Resizing or
# cropping would change the element spacing the generator recorded.
CONTOUR_TRANSFORMS = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

CONTOUR_TRANSFORMS_UNNORMALIZED = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor()]
)

PRESENT_DIR = "with_contour"
ABSENT_DIR = "without_contour"

# generative parameters kept per image, for breaking accuracy down by them
FACTOR_COLUMNS: List[str] = [
    "seed",
    "relative_spacing",
    "eccentricity_deg",
    "contour_length_target",
    "contour_length_achieved",
    "orientation_deg",
    "contour_side",
    "skew_angle_deg",
    "contour_spacing_deg",
    "n_background_cells",
]

# the crossed factors worth reporting accuracy against. Ordered by how much
# the contour-integration literature expects them to matter: wider spacing
# and shorter paths break integration, eccentricity degrades it, and
# orientation should not matter much at all
CONDITION_FACTORS: List[str] = [
    "relative_spacing",
    "eccentricity_deg",
    "contour_length_achieved",
    "orientation_deg",
]


class ContourData(Dataset):
    """Contour-present and contour-absent stimuli, labelled 1 and 0.

    Both members of every pair are returned, in metadata order, so index
    ``2 * i`` is the present image of pair ``i`` and ``2 * i + 1`` is its
    absent partner. Keeping them adjacent makes it obvious in any downstream
    split that the two belong together.
    """

    def __init__(self, root, transforms=CONTOUR_TRANSFORMS):
        import pandas as pd

        self.root = Path(root)
        self.transforms = transforms

        metadata = pd.read_csv(self.root / "metadata.csv")
        assert len(metadata), f"no rows in {self.root / 'metadata.csv'}"
        missing = [c for c in FACTOR_COLUMNS if c not in metadata.columns]
        assert not missing, (
            f"{self.root / 'metadata.csv'} is missing {missing}; it predates "
            "the split layout and needs regenerating"
        )
        self.split = str(metadata["split"].iloc[0]) if "split" in metadata else ""

        files: List[Path] = []
        labels: List[int] = []
        factors: List[np.ndarray] = []
        for _, row in metadata.iterrows():
            values = np.array([row[column] for column in FACTOR_COLUMNS], dtype=float)
            for present, directory, column in (
                (1, PRESENT_DIR, "filename_present"),
                (0, ABSENT_DIR, "filename_absent"),
            ):
                path = self.root / directory / row[column]
                assert path.exists(), f"missing stimulus {path}"
                files.append(path)
                labels.append(present)
                factors.append(values)

        self.file_list = files
        self.labels = np.array(labels, dtype=int)
        self.factors = np.stack(factors)
        self.factor_columns = list(FACTOR_COLUMNS)

        # the grouping variable for cross-validation: a seed's two images
        # share a background and must never be split across folds
        self.seeds = self.factors[:, FACTOR_COLUMNS.index("seed")].astype(int)

    def __len__(self) -> int:
        return len(self.file_list)

    def factor(self, name: str) -> np.ndarray:
        """One generative parameter, aligned with ``labels``."""
        assert name in self.factor_columns, (
            f"{name} is not a recorded factor; have {self.factor_columns}"
        )
        return self.factors[:, self.factor_columns.index(name)]

    def __getitem__(self, index: int):
        image = Image.open(self.file_list[index]).convert("RGB")
        if self.transforms is not None:
            image = self.transforms(image)
        return image, int(self.labels[index])
