from dataclasses import dataclass
from typing import Optional, Type, List, Callable, Union
import glob
import math
import numpy as np
import torch
import torchvision
from einops import reduce, rearrange
from skimage import io
from torch.utils.data import Dataset
import xarray as xr
from scipy.signal import argrelmax

from spacetorch.colormaps import nauhaus_colormaps
from spacetorch.constants import DVA_PER_IMAGE
from spacetorch.datasets.ringach_2002 import load_ringach_data
from spacetorch.types import AggMode
from spacetorch.utils.array_utils import flatten
from spacetorch.variants import OUTPUT_DIMS_FOR_224_INPUTS

ComposedTransforms = Type[torchvision.transforms.transforms.Compose]

# diameters in degrees of visual angle, including 0.0 for blank baseline
EXPANDING_DIAMETERS = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                       0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]


@dataclass
class Metric:
    name: str
    n_unique: int
    high: float
    xticklabels: Union[List[str], np.ndarray]
    xlabel: str
    agg_mode: AggMode
    colormap: Callable


def make_circular_mask(image_size: int, diameter_deg: float,
                       pixels_per_deg: float = 28.0) -> np.ndarray:
    """
    Returns a (H, W) float mask with 1 inside the aperture and 0 outside.
    diameter_deg=0 returns an all-zero mask (blank gray stimulus).
    """
    if diameter_deg == 0.0:
        return np.zeros((image_size, image_size), dtype=np.float32)

    radius_px = (diameter_deg / 2.0) * pixels_per_deg
    cx, cy = image_size / 2.0, image_size / 2.0
    y = np.arange(image_size, dtype=np.float32)
    x = np.arange(image_size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return (dist <= radius_px).astype(np.float32)


def apply_aperture_mask_np(img: np.ndarray, diameter_deg: float,
                            pixels_per_deg: float = 28.0,
                            gray_value: float = 128.0) -> np.ndarray:
    """
    Applies a circular aperture mask to a (H, W, C) uint8 numpy image.
    Outside the aperture is set to gray_value (mean luminance).
    """
    H, W = img.shape[:2]
    mask = make_circular_mask(H, diameter_deg, pixels_per_deg)  # (H, W)
    mask = mask[:, :, None]  # (H, W, 1) for broadcasting over channels
    out = img * mask + gray_value * (1 - mask)
    return out.astype(np.uint8)


class ExpandingSineGrating2019(Dataset):
    """
    Expanding aperture sine gratings dataset.

    Each original grating image is repeated once per diameter in
    EXPANDING_DIAMETERS, with the aperture mask applied on the fly.
    Labels are (angle, sf, phase, color, diameter).
    """

    metrics: List[Metric] = [
        Metric(
            name="angles",
            n_unique=8,
            high=180,
            xticklabels=[f"{x:.0f}" for x in np.linspace(0, 180, 9)],
            xlabel=r"Orientation ($^\circ$)",
            agg_mode="circmean",
            colormap=nauhaus_colormaps["angles"],
        ),
        Metric(
            name="sfs",
            n_unique=8,
            high=110,
            xticklabels=[
                f"{x:.0f}" for x in np.linspace(5.5, 110.0, 9) / DVA_PER_IMAGE
            ],
            xlabel="Spatial Frequency (cpd)",
            agg_mode="mean",
            colormap=nauhaus_colormaps["sfs"],
        ),
        Metric(
            name="colors",
            n_unique=2,
            high=1,
            xticklabels=["B/W", "Color"],
            xlabel="",
            agg_mode="mean",
            colormap=nauhaus_colormaps["colors"],
        ),
        Metric(
            name="diameters",
            n_unique=len(EXPANDING_DIAMETERS),
            high=8,
            xticklabels=[f"{x:.2f}" for x in EXPANDING_DIAMETERS],
            xlabel="Stimulus diameter (deg)",
            agg_mode="mean",
            colormap=nauhaus_colormaps["diameters"],
        ),
    ]

    def __init__(
        self,
        sine_dir: str,
        transforms: Optional[ComposedTransforms] = None,
        diameters: Optional[List[float]] = None,
        pixels_per_deg: float = 28.0,
        gray_value: float = 128.0,
    ):
        self.transforms = transforms
        self.pixels_per_deg = pixels_per_deg
        self.gray_value = gray_value
        self.diameters = diameters if diameters is not None else EXPANDING_DIAMETERS

        self.file_list = sorted(glob.glob(f"{sine_dir}/*.jpg"))
        n_images = len(self.file_list)
        n_diameters = len(self.diameters)

        # labels: (n_images * n_diameters, 5)
        # columns: angle, sf, phase, color, diameter
        base_labels = np.zeros([n_images, 4], dtype=float)
        for img_idx, fname in enumerate(self.file_list):
            parts = fname.split("/")[-1].split("_")
            angle = float(parts[1][:-3])
            sf = float(parts[2][:-2])
            phase = float(parts[3][:-5])
            color_string = parts[4].split(".jpg")[0]
            color = 0.0 if color_string == "bw" else 1.0
            base_labels[img_idx] = [angle, sf, phase, color]

        # tile: for each image repeat across all diameters
        # resulting order: img0_diam0, img0_diam1, ..., img1_diam0, img1_diam1, ...
        self.labels = np.zeros([n_images * n_diameters, 5], dtype=float)
        self.index_map = []  # maps flat index -> (img_idx, diameter)

        for img_idx in range(n_images):
            for d_idx, diam in enumerate(self.diameters):
                flat_idx = img_idx * n_diameters + d_idx
                self.labels[flat_idx, :4] = base_labels[img_idx]
                self.labels[flat_idx, 4] = diam
                self.index_map.append((img_idx, diam))

    @classmethod
    def get_metrics(cls, as_dict: bool = False):
        if as_dict:
            return {m.name: m for m in cls.metrics}
        return cls.metrics

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_idx, diam = self.index_map[idx]
        img = io.imread(self.file_list[img_idx])  # (H, W, C) uint8

        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)

        # apply aperture mask before any transforms
        img = apply_aperture_mask_np(img, diam, self.pixels_per_deg, self.gray_value)

        target = self.labels[idx]  # (5,): angle, sf, phase, color, diameter

        if self.transforms:
            img = self.transforms(img)

        return img, target


def get_closest_factors(num):
    num_root = int(math.sqrt(num))
    while num % num_root != 0:
        num_root -= 1
    return num_root, int(num / num_root)


class ExpandingSineResponses:
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        normalize_to_ringach_firing_rates: bool = True,
        is_llcnn: bool = False,
        is_lcnn: bool = False,
    ):
        self.DVA_PER_IMAGE = DVA_PER_IMAGE

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

        if is_lcnn:
            features = reduce(features, 'b c h w -> b c', 'mean')

        if normalize_to_ringach_firing_rates:
            if np.min(features) < 0:
                features -= np.min(features)
            max_per_unit: np.ndarray = np.max(features, axis=0)
            median_of_peak_neuron_firing_rates = np.median(load_ringach_data("maxdc"))
            ringach_to_features_ratio: float = (
                median_of_peak_neuron_firing_rates / np.median(max_per_unit)
            )
            features *= ringach_to_features_ratio

        self._data = xr.DataArray(
            data=flatten(features),
            coords={
                "angles":    ("image_idx", labels[:, 0]),
                "sfs":       ("image_idx", labels[:, 1]),
                "phases":    ("image_idx", labels[:, 2]),
                "colors":    ("image_idx", labels[:, 3]),
                "diameters": ("image_idx", labels[:, 4]),
            },
            dims=["image_idx", "unit_idx"],
        )

        self._convert_sfs_to_cpd()
        self._compute_circular_variance()

    def __len__(self) -> int:
        return self._data.sizes["unit_idx"]

    @property
    def orientation_tuning_curves(self) -> xr.DataArray:
        x = self._data.where(self._data.diameters == 8.0, drop=True)
        return x.groupby("angles").mean().T

    @property
    def circular_variance(self) -> np.ndarray:
        return self._data.circular_variance.values

    @property
    def baseline_subtracted_data(self) -> xr.DataArray:
        baseline_response = self._data.where(self._data.diameters == 0.0, drop=True).mean("image_idx")
        return self._data - baseline_response
    
    def get_indices_of_central_units(self, architecture: str, layer: str) -> np.ndarray:
        C, H, W = OUTPUT_DIMS_FOR_224_INPUTS[architecture][layer]
        
        if architecture == "lcnn":
            return np.array([27, 28, 36, 37])

        if architecture == "llcnn":
            kh, kw = get_closest_factors(C)

            tiled_H = H * kh
            tiled_W = W * kw

            center_h = tiled_H // 2
            center_w = tiled_W // 2

            center_idx = center_h * tiled_W + center_w

            offsets = np.array([
                dh * tiled_W + dw
                for dh in [-2, -1, 0, 1, 2]
                for dw in [-2, -1, 0, 1, 2]
            ])

            return center_idx + offsets
        
        center_h, center_w = H // 2, W // 2
        center_offset = center_h * W + center_w
        center_indices = center_offset + (H * W) * np.arange(C)
        return center_indices
    
    def get_indices_of_peripheral_units(self, architecture: str, layer: str) -> np.ndarray:
        C, H, W = OUTPUT_DIMS_FOR_224_INPUTS[architecture][layer]
        center_h, center_w = H // 2, W // 2
        center_offset = center_h * W + math.ceil(0.2 * center_w)
        center_indices = center_offset + (H * W) * np.arange(C)
        return center_indices

    def get_preferences(self, metric: str = "angles") -> xr.DataArray:
        x = self._data.where(self._data.diameters == 8.0, drop=True)
        mean_response = x.groupby(metric).mean()
        return mean_response.argmax(axis=0)

    def get_orientation_selective_indices(self, candidate_indices: np.ndarray) -> np.ndarray:
        """
        From a set of candidate unit indices, returns those that are orientation
        selective, defined as:
        1. circular variance < 0.6
        2. positive response at preferred angle at diameter 8.0
        """
        # criterion 1: low circular variance
        cv = self.circular_variance[candidate_indices]
        low_cv_mask = cv < 0.6

        # criterion 2: positive response at preferred angle at diameter=8.0
        pref_indices = self.get_preferences("angles")
        angle_vals = np.linspace(0, 180, 8, endpoint=False)
        pref_angles = angle_vals[pref_indices.values]

        data = self.baseline_subtracted_data.values
        angles_vals = self._data.angles.values
        diams_vals = self._data.diameters.values

        positive_pref_mask = np.zeros(len(candidate_indices), dtype=bool)
        for i, unit_idx in enumerate(candidate_indices):
            pref_angle = pref_angles[unit_idx]
            mask = (angles_vals == pref_angle) & (diams_vals == 8.0)
            positive_pref_mask[i] = data[mask, unit_idx].mean() > 0

        os_mask = low_cv_mask & positive_pref_mask
        return candidate_indices[os_mask]

    def get_response_at_diameter(self, architecture: str, layer: str) -> np.ndarray:
        pref_indices = self.get_preferences("angles")
        pref_angles = np.linspace(0, 180, 8, endpoint=False)[pref_indices.values]

        data = self.baseline_subtracted_data.values
        diams_vals = self._data.diameters.values
        angles_vals = self._data.angles.values
        unique_diams = np.unique(diams_vals)

        center_indices = self.get_indices_of_central_units(architecture=architecture, layer=layer)
        os_indices = self.get_orientation_selective_indices(center_indices)

        curves = np.zeros((len(unique_diams), len(os_indices)))
        for i, unit_idx in enumerate(os_indices):
            pref_i = pref_angles[unit_idx]
            for j, diam in enumerate(unique_diams):
                mask = (angles_vals == pref_i) & (diams_vals == diam)
                curves[j, i] = data[mask, unit_idx].mean()

        return curves, os_indices, pref_angles, unique_diams

    def get_rf_sizes(self, architecture: str, layer: str) -> np.ndarray:
        """
        Returns the aperture diameter at peak response for each unit.
        """
        curves, _, _, unique_diams = self.get_response_at_diameter(architecture=architecture, layer=layer)
        n_units = curves.shape[1]
        peak_diams = np.full(n_units, np.nan)

        for u in range(n_units):
            curve = curves[:, u]

            local_peaks, = argrelmax(curve, order=1)
            local_peaks = local_peaks[local_peaks > 0]

            if len(local_peaks) > 0:
                peak_diams[u] = EXPANDING_DIAMETERS[local_peaks[0]]
            else:
                # no local peak -> response asymptoted without declining; use the SMALLEST diameter reaching 95% of
                # the eventual peak
                global_idx = curve.argmax()
                if global_idx > 0:
                    peak_value = curve[global_idx]
                    threshold = 0.95 * peak_value
                    above_thresh = np.where(curve[1:] >= threshold)[0] + 1  # skip idx 0
                    if len(above_thresh) > 0:
                        peak_diams[u] = EXPANDING_DIAMETERS[above_thresh[0]]

        return peak_diams[~np.isnan(peak_diams)]

    def get_surround_sizes(self, architecture: str, layer: str, threshold: float = 0.1) -> np.ndarray:
        """
        Suppression index = diameter at which response asymptotes after the peak.
        Defined as the smallest diameter beyond the peak where the response
        stays within `threshold` * peak_response of its value for all larger diameters.
        
        Returns: (n_units,) array of diameters in degrees
        """
        curves, _, _, unique_diams = self.get_response_at_diameter(architecture=architecture, layer=layer)
        
        peak_idx = curves.argmax(axis=0)
        peak_response = curves.max(axis=0)
        n_units = curves.shape[1]

        asymptote_diameters = np.full(n_units, np.nan)
        asymptote_diameters = []

        for u in range(n_units):
            p_idx = peak_idx[u]
            p_resp = peak_response[u]
    
            post_peak_curve = curves[p_idx:, u]
            post_peak_diams = EXPANDING_DIAMETERS[p_idx:]
            final_value = post_peak_curve[-1]
    
            for i in range(len(post_peak_curve)):
                remaining = post_peak_curve[i:]
                if np.all(np.abs(remaining - final_value) <= threshold * p_resp):
                    asymptote_diameters[u] = post_peak_diams[i]
                    break

        return asymptote_diameters[~np.isnan(asymptote_diameters)]

    def get_peak_heights(self, metric: str = "angles"):
        tuning_curve = self._data.groupby(metric).mean()
        return np.ptp(tuning_curve.data, axis=0)

    def _convert_sfs_to_cpd(self):
        cpd: np.ndarray = self._data["sfs"] / self.DVA_PER_IMAGE
        self._data = self._data.assign_coords({"sfs": ("image_idx", cpd.data)})

    def _compute_circular_variance(self):
        n_angles = 8
        angles = np.linspace(0, np.pi, n_angles + 1)[:-1]

        otc = np.array(self.orientation_tuning_curves)

        # shift each unit's tuning curve so its minimum is exactly 0
        otc = otc - np.min(otc, axis=1, keepdims=True)

        numerator = np.sum(
            otc * np.exp(angles * 2 * 1j), axis=1
        )
        denominator = np.sum(otc, axis=1)
        R = numerator / denominator
        CV = 1 - np.abs(R)

        self._data = self._data.assign_coords(
            {"circular_variance": ("unit_idx", CV.real)}
        )

    def get_peripheral_tuning_to_central_grating(self):
        angle_vals = np.linspace(0, 180, 8, endpoint=False)
        angle_rad = np.linspace(0, np.pi, 8, endpoint=False)
        data = self._data
        n_units = data.sizes["unit_idx"]

        # get baseline response at diameter=0
        blank_data = data.where(data.diameters == 0.0, drop=True)
        baseline = blank_data.values.mean(axis=0)

        # compute preferred angle and CV at diameter=8.0 (classical RF stimulation)
        full_diam_data = data.where(data.diameters == 8.0, drop=True)
        tuning_full = np.zeros((8, n_units))
        for i, ang in enumerate(angle_vals):
            mask = full_diam_data.angles.values == ang
            tuning_full[i] = full_diam_data.values[mask].mean(axis=0)

        normed_tuning = tuning_full - tuning_full.min(axis=0, keepdims=True)
        numerator = np.sum(normed_tuning * np.exp(2j * angle_rad[:, None]), axis=0)
        denominator = np.sum(normed_tuning, axis=0)
        cv = 1 - np.abs(numerator / (denominator + 1e-8))
        pref_angles = angle_vals[tuning_full.argmax(axis=0)]

        # compute modulation at diameter=0.5 (small central grating), baseline-subtracted
        small_diam_data = data.where(data.diameters == 0.5, drop=True)
        tuning_small = np.zeros((8, n_units))
        for i, ang in enumerate(angle_vals):
            mask = small_diam_data.angles.values == ang
            tuning_small[i] = small_diam_data.values[mask].mean(axis=0)

        modulation = tuning_small - baseline[np.newaxis, :]

        return cv, pref_angles, modulation