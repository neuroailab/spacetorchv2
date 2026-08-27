from typing import Optional, Type
import math
import numpy as np
import torch
import torchvision
from scipy import ndimage
from einops import reduce, rearrange
from torch.utils.data import Dataset
import xarray as xr

from spacetorch.constants import DVA_PER_IMAGE
from spacetorch.datasets.ringach_2002 import load_ringach_data
from spacetorch.utils.array_utils import flatten
from spacetorch.variants import OUTPUT_DIMS_FOR_224_INPUTS


ComposedTransforms = Type[torchvision.transforms.transforms.Compose]


PIXELS_PER_DEG = 28.0
PATCH_DIAMETER_DEG = 0.5
PATCH_DIAMETER_PX = int(PATCH_DIAMETER_DEG * PIXELS_PER_DEG)
GRID_SPACING_PX = PATCH_DIAMETER_PX // 2


def make_sine_patch(
    image_size: int,
    cx: int,
    cy: int,
    diameter_px: int,
    sf_cpd: float,
    angle_deg: float,
    phase: float,
    pixels_per_deg: float = PIXELS_PER_DEG,
    gray_value: float = 128.0,
    contrast: float = 1.0,
) -> np.ndarray:
    """
    Returns a (H, W, 3) uint8 image with a circular sine grating patch
    at position (cx, cy), with grating phase defined relative to patch center.
    """
    img = np.full((image_size, image_size, 3), gray_value, dtype=np.float32)

    radius_px = diameter_px / 2.0
    yy, xx = np.mgrid[0:image_size, 0:image_size].astype(np.float32)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = dist <= radius_px

    angle_rad = np.deg2rad(angle_deg)
    sf_cpp = sf_cpd / pixels_per_deg
    grating = np.cos(
        2 * np.pi * sf_cpp * ((xx - cx) * np.cos(angle_rad) + (yy - cy) * np.sin(angle_rad))
        + phase
    )

    amplitude = gray_value * contrast
    luminance = np.clip(gray_value + amplitude * grating, 0, 255)

    for c in range(3):
        img[:, :, c] = np.where(mask, luminance, gray_value)

    return img.astype(np.uint8)


class TiledSineGrating2019(Dataset):
    """
    Tiled sine grating patch dataset for classical RF mapping.

    A small circular sine grating patch (default 0.5 deg diameter) is
    presented at each position on a regular grid covering the full 224x224
    image, at 4 orientations, 4 spatial frequencies, and 4 phases.
    Labels are (cx_px, cy_px, sf, angle, phase).
    """

    SF_CPDS = np.array([18.6, 31.6, 57.8, 96.9])
    ANGLES  = np.array([0.0, 45.0, 90.0, 135.0])
    PHASES  = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])

    def __init__(
        self,
        transforms: Optional[ComposedTransforms] = None,
        patch_diameter_deg: float = PATCH_DIAMETER_DEG,
        grid_spacing_px: Optional[int] = None,
        image_size: int = 224,
        pixels_per_deg: float = PIXELS_PER_DEG,
        gray_value: float = 128.0,
        contrast: float = 1.0,
        sfs: Optional[np.ndarray] = None,
        angles: Optional[np.ndarray] = None,
        phases: Optional[np.ndarray] = None,
    ):
        self.transforms = transforms
        self.image_size = image_size
        self.pixels_per_deg = pixels_per_deg
        self.gray_value = gray_value
        self.contrast = contrast
        self.patch_diameter_px = int(patch_diameter_deg * pixels_per_deg)

        spacing = grid_spacing_px if grid_spacing_px is not None else self.patch_diameter_px // 2
        half = self.patch_diameter_px // 2

        self.grid_positions = [
            (cx, cy)
            for cy in range(half, image_size - half + 1, spacing)
            for cx in range(half, image_size - half + 1, spacing)
        ]

        self.sfs = sfs if sfs is not None else self.SF_CPDS
        self.angles = angles if angles is not None else self.ANGLES
        self.phases = phases if phases is not None else self.PHASES

        self.index_map = [
            (cx, cy, sf, angle, phase)
            for (cx, cy) in self.grid_positions
            for sf in self.sfs
            for angle in self.angles
            for phase in self.phases
        ]

        self.labels = np.array(
            [(cx, cy, sf, angle, phase) for (cx, cy, sf, angle, phase) in self.index_map],
            dtype=np.float32,
        )

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        cx, cy, sf, angle, phase = self.index_map[idx]

        img = make_sine_patch(
            image_size=self.image_size,
            cx=int(cx), cy=int(cy),
            diameter_px=self.patch_diameter_px,
            sf_cpd=sf,
            angle_deg=angle,
            phase=phase,
            pixels_per_deg=self.pixels_per_deg,
            gray_value=self.gray_value,
            contrast=self.contrast,
        )

        target = self.labels[idx]  # (5,): cx, cy, sf, angle, phase

        if self.transforms:
            img = self.transforms(img)

        return img, target

    @property
    def n_positions(self) -> int:
        return len(self.grid_positions)

    @property
    def grid_shape(self) -> tuple:
        """(n_rows, n_cols) of the position grid, for reshaping response maps."""
        half = self.patch_diameter_px // 2
        spacing = self.patch_diameter_px // 2
        n = len(range(half, self.image_size - half + 1, spacing))
        return (n, n)


def get_closest_factors(num):
    num_root = int(math.sqrt(num))
    while num % num_root != 0:
        num_root -= 1
    return num_root, int(num / num_root)


class TiledSineResponses:
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,          # (N, 5): cx_px, cy_px, sf, angle, phase
        grid_shape: tuple,
        architecture: str,
        layer: str,
        normalize_to_ringach_firing_rates: bool = True,
        is_llcnn: bool = False,
        is_lcnn: bool = False,
    ):
        self.DVA_PER_IMAGE = DVA_PER_IMAGE
        self.grid_shape = grid_shape
        self.pixels_per_deg = PIXELS_PER_DEG

        if is_llcnn:
            f_shape = features.shape
            kh, kw = get_closest_factors(f_shape[1])
            features = (torch.from_numpy(features)
                .permute(0, 2, 3, 1)
                .reshape(f_shape[0], f_shape[2], f_shape[3], kh, kw)
                .permute(0, 1, 3, 2, 4)
                .reshape(f_shape[0], f_shape[2]*kh, f_shape[3]*kw)
                .numpy())
            features = rearrange(features, 'b h w -> b (h w)')
            print(features.shape)

        if is_lcnn:
            features = reduce(features, 'b c h w -> b c', 'mean')

        if normalize_to_ringach_firing_rates:
            if np.min(features) < 0:
                features -= np.min(features)
            max_per_unit = np.max(features, axis=0)
            median_of_peak_neuron_firing_rates = np.median(load_ringach_data("maxdc"))
            ringach_to_features_ratio = (
                median_of_peak_neuron_firing_rates / np.median(max_per_unit)
            )
            features *= ringach_to_features_ratio

        self._data = xr.DataArray(
            data=flatten(features),
            coords={
                "cx": ("image_idx", labels[:, 0]),
                "cy": ("image_idx", labels[:, 1]),
                "sfs": ("image_idx", labels[:, 2]),
                "angles": ("image_idx", labels[:, 3]),
                "phases": ("image_idx", labels[:, 4]),
            },
            dims=["image_idx", "unit_idx"],
        )

        self.unit_indices = self.get_indices_of_central_units(architecture, layer)

    def __len__(self) -> int:
        return len(self.unit_indices)

    @property
    def baseline_subtracted_data(self) -> np.ndarray:
        data = self._data.values
        return data - data.mean(axis=0)

    def get_indices_of_central_units(self, architecture: str, layer: str) -> np.ndarray:
        C, H, W = OUTPUT_DIMS_FOR_224_INPUTS[architecture][layer]

        if architecture == "lcnn":
            return np.array([27, 28, 36, 37])

        if architecture == "llcnn":
            center_idx = 84 * 168 + 84

            offsets = np.array([
                dh * 168 + dw
                for dh in [-2, -1, 0, 1, 2]
                for dw in [-2, -1, 0, 1, 2]
            ])

            return center_idx + offsets

        center_h, center_w = H // 2, W // 2
        center_offset = center_h * W + center_w
        return center_offset + (H * W) * np.arange(C)

    def get_spatial_response_maps(self) -> np.ndarray:
        """
        Returns (n_units, n_rows, n_cols) spatial response maps,
        averaged over SF, angle, and phase at each grid position.
        """
        data = self.baseline_subtracted_data

        _, cx_idx = np.unique(self._data.cx.values, return_inverse=True)
        _, cy_idx = np.unique(self._data.cy.values, return_inverse=True)

        n_rows, n_cols = self.grid_shape
        n_units = len(self.unit_indices)

        unit_data = data[:, self.unit_indices].astype(np.float32)

        sums = np.zeros((n_rows, n_cols, n_units), dtype=np.float32)
        counts = np.zeros((n_rows, n_cols), dtype=np.int32)

        np.add.at(sums, (cy_idx, cx_idx), unit_data)
        np.add.at(counts, (cy_idx, cx_idx), 1)

        maps = sums / counts[:, :, None]
        return maps.transpose(2, 0, 1)

    def get_crf_sizes_deg(self, noise_threshold=0.05, center_pos=(16, 16)) -> np.ndarray:
        """
        Estimates CRF diameter in degrees for each central unit.
        Counts positions where energy >= threshold * peak energy,
        then back-computes the diameter of a circle with equivalent area.
        Returns: (n_units,) array of CRF diameters in degrees.
        """
        maps = self.get_spatial_response_maps()

        unique_cx = np.unique(self._data.cx.values)
        spacing_deg = (unique_cx[1] - unique_cx[0]) / self.pixels_per_deg
        n_units = maps.shape[0]
        crf_diameters = np.full(n_units, np.nan)

        for u in range(len(self.unit_indices)):
            m = maps[u].copy()

            thresh = noise_threshold * m.max()
            binary = m > thresh

            if not binary[center_pos[0], center_pos[1]]:
                continue

            # keep only the contiguous suprathreshold region containing the center position
            labeled, _ = ndimage.label(binary)
            center_label = labeled[center_pos[0], center_pos[1]]
            region_mask = labeled == center_label

            n_above = region_mask.sum()
            area_deg2 = n_above * spacing_deg ** 2
            crf_diameters[u] = 2 * np.sqrt(area_deg2 / np.pi)

        return crf_diameters[~np.isnan(crf_diameters)]

