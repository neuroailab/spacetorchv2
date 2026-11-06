"""Localized Gabor patches for measuring orientation tuning at one site.

Full-field gratings drive every position identically, so a channel's tuning
has to be read as a spatial average. That is fine for a convolutional map but
wrong for a transformer, where averaging over 196 tokens mixes a tuned token
with 195 untuned ones and washes the tuning out.

This bank instead places a single 16x16 Gabor patch on the centre token of an
otherwise flat ImageNet-mean background, so the response can be read at the
stimulated site alone. It reproduces the stimulus set of the privileged visual
computations note: twelve orientations, six frequencies from one to six cycles
per patch, four phases, and three normalized colour directions, giving 864
images.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset

# the background the patch sits on, and the channel statistics used to
# normalize it afterwards
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMAGE_SIZE = 224
PATCH_SIZE = 16
# the 14 x 14 token grid's centre cell, which is also the centre of the image
PATCH_ROW = IMAGE_SIZE // PATCH_SIZE // 2
PATCH_COL = IMAGE_SIZE // PATCH_SIZE // 2

ORIENTATIONS = np.arange(12) * 15.0
CYCLES_PER_PATCH = np.arange(1, 7, dtype=float)
PHASES = np.arange(4) * (np.pi / 2)

# luminance, red-green and blue-yellow, each scaled to unit norm so that the
# three directions inject the same contrast energy
COLOR_DIRECTIONS = np.array(
    [[1.0, 1.0, 1.0], [1.0, -1.0, 0.0], [-1.0, -1.0, 2.0]], dtype=np.float32
)
COLOR_DIRECTIONS /= np.linalg.norm(COLOR_DIRECTIONS, axis=1, keepdims=True)

ENVELOPE_SIGMA_PX = 4.0
CONTRAST = 0.5


def gabor_patch(
    orientation_deg: float, cycles_per_patch: float, phase_rad: float
) -> np.ndarray:
    """A single-channel Gabor on [-1, 1], sized ``PATCH_SIZE``."""
    coords = np.arange(PATCH_SIZE, dtype=np.float32) - (PATCH_SIZE - 1) / 2.0
    x, y = np.meshgrid(coords, coords)

    theta = np.deg2rad(orientation_deg)
    rotated = x * np.cos(theta) + y * np.sin(theta)

    envelope = np.exp(-(x**2 + y**2) / (2.0 * ENVELOPE_SIGMA_PX**2))
    carrier = np.cos(2.0 * np.pi * (cycles_per_patch / PATCH_SIZE) * rotated + phase_rad)
    return (envelope * carrier).astype(np.float32)


class GaborPatches(Dataset):
    """Gabor patches at the centre token, labelled by their parameters.

    Labels are ``(orientation_deg, cycles_per_patch, phase_rad, colour index)``,
    in the same column order ``GratingHarmonics`` expects, so the phase-harmonic
    orientation machinery applies unchanged.
    """

    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self.labels = np.array(
            [
                [orientation, cycles, phase, color_idx]
                for orientation in ORIENTATIONS
                for cycles in CYCLES_PER_PATCH
                for phase in PHASES
                for color_idx in range(len(COLOR_DIRECTIONS))
            ],
            dtype=float,
        )

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def patch_box(self) -> Tuple[float, float, float, float]:
        """Fractional (top, left, bottom, right) extent of the stimulated site.

        Passed to ``region_responses`` so the response is read where the patch
        actually is, on a feature map of any resolution.
        """
        top = PATCH_ROW * PATCH_SIZE / IMAGE_SIZE
        left = PATCH_COL * PATCH_SIZE / IMAGE_SIZE
        return (top, left, top + PATCH_SIZE / IMAGE_SIZE, left + PATCH_SIZE / IMAGE_SIZE)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        orientation, cycles, phase, color_idx = self.labels[idx]
        patch = gabor_patch(orientation, cycles, phase)
        direction = COLOR_DIRECTIONS[int(color_idx)]

        image = np.tile(IMAGENET_MEAN, (IMAGE_SIZE, IMAGE_SIZE, 1))
        top = PATCH_ROW * PATCH_SIZE
        left = PATCH_COL * PATCH_SIZE
        image[top : top + PATCH_SIZE, left : left + PATCH_SIZE] += (
            CONTRAST * patch[:, :, None] * direction[None, None, :]
        )
        image = np.clip(image, 0.0, 1.0)

        if self.normalize:
            image = (image - IMAGENET_MEAN) / IMAGENET_STD

        tensor = torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32))
        return tensor, self.labels[idx]
