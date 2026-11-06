"""
Orientation specificity of lateral spread for the LLCNN (Qian et al., 2024), the counterpart of orientation_specificity.py
(bouton map) and quantifying_specificity.py (numbers) for a model without saved positions.

Layout. specificity.py tiles a layer's C channels into kh x kw blocks of one (H * kh, W * kw) sheet (layer1: 9 channels
-> 3 x 3 blocks of a 168 x 168 sheet over the 56 x 56 feature map), so a unit's place on that sheet is both its tissue
position and, by block, its RF cell. The LLCNN's lateral connections are a Gaussian pool over 11 x 11 sheet units after
each conv (layer1 only), i.e. ~+-0.24 deg per pool; at layer1.1.conv2 the drive from the 0.5 deg grating ends ~1.5 deg away.

Bouton map (OUT_DIR / "specificity_llcnn.png"), as orientation_specificity.py: the top (1 - m) of units with RF center
>= map_far_deg, in black over every tuned unit colored by its smoothed preferred orientation. Smoothing is V1Map's
smooth_orientation_tuning_curves=True (as get_sine_tissue would do, which doesn't know the sheet layout): each unit's
full-field SineGrating2019 tuning curve is interpolated over orientation and fit with a von Mises, and the fit's peak is
a continuous preferred orientation. The fits are cached in the layer's cache/ directory. The numbers below still use the
argmax preferences from axial_specificity.npz, like the other models.

Numbers, as quantifying_specificity.py with RF center >= far_deg (see that script for the definitions):
OUT_DIR / "llcnn_specificity_by_grating.csv" (one row per grating) and OUT_DIR / "llcnn_specificity.csv" (mean and SD).
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage
from scipy.spatial import cKDTree
import argparse

from spacetorch.configs import get_cfg
from spacetorch.variants import get_variant
from spacetorch.colormaps import nauhaus_colormaps
from spacetorch.maps.sine_tissue import get_sine_responses
from spacetorch.maps.v1_map import V1Map

angle_vals = np.linspace(0, 180, 8, endpoint=False)

CONFIG, LAYER = "/ccn2/u/ynshah/spacetorchv2/configs/llcnn/spatial.yaml", "layer1.1.conv2"
LLCNN_CHANNELS = {"layer1": 9, "layer2": 144, "layer3": 256, "layer4": 529}

map_far_deg = 0.0    # bouton map: RF center >= this far from the grating (0 = include units that see it directly)
far_deg = 2.0        # numbers: as quantifying_specificity.py
m = 0.95             # boutons = top (1 - m) of candidates
cv_max = 1.0         # keep only units with circular variance <= this (1 = keep all)
smooth_deg = 1.0     # scale of the large-scale density kept by the patchiness null
n_null = 20          # null maps for clark_evans
MALACH_MEAN, MALACH_SD = 0.30, 0.10
rng = np.random.default_rng(0)

OUT_DIR = Path("/ccn2/u/ynshah/spacetorchv2/paper")


def closest_factors(num):
    root = int(np.sqrt(num))
    while num % root:
        root -= 1
    return root, num // root


def load_tissue(layer, n_units):
    # tissue positions (mm, sheet row/col jittered by 0.1 and scaled to 36.75) and each unit's RF cell (h, w)
    kh, kw = closest_factors(LLCNN_CHANNELS[layer.split(".")[0]])
    H = W = int(np.sqrt(n_units // (kh * kw)))
    row, col = np.divmod(np.arange(n_units), W * kw)
    positions = np.stack([row, col], axis=1) + np.random.default_rng(0).normal(0, 0.1, (n_units, 2))
    return positions * 36.75 / np.ptp(positions), row // kh, col // kw, H, W


def rf_distance_deg(cell_h, cell_w, H, W, modulation):
    # RF center of a unit is ~the center of its grid cell (h, w) on the H x W feature map of the 8 deg image
    gx = (np.arange(H * W) % W + 0.5) * 8.0 / W - 4.0
    gy = (np.arange(H * W) // W + 0.5) * 8.0 / H - 4.0
    cell = cell_h * W + cell_w
    # locate the grating in feature-map coordinates: drive-weighted centroid of the central (< 1 deg) cells at >= half
    # max drive
    drive = np.bincount(cell, weights=np.abs(modulation).mean(axis=0), minlength=H * W) / np.bincount(cell, minlength=H * W)
    near = np.hypot(gx, gy) < 1.0
    wts = np.where(near & (drive >= 0.5 * drive[near].max()), drive, 0)
    x0, y0 = (gx * wts).sum() / wts.sum(), (gy * wts).sum() / wts.sum()
    return np.hypot(gx[cell] - x0, gy[cell] - y0)


def axial_diff(a, b):
    d = np.abs(a - b) % 180
    return np.minimum(d, 180 - d)


def top_units(score, units):
    # the top (1 - m) of units by score
    return units[score[units] > np.quantile(score[units], m)] if len(units) else units


def specificity(pref, theta):
    if len(pref) == 0:
        return {"iso_index": 0.0, "frac_far": np.nan, "frac_orth": np.nan}
    delta = axial_diff(pref, theta)
    return {"iso_index": np.mean(np.cos(2 * np.deg2rad(delta))), "frac_far": np.mean(delta >= 45),
            "frac_orth": np.mean(delta >= 67.5)}


def mean_nn(pts):
    return cKDTree(pts).query(pts, k=2)[0][:, 1].mean()


def patchiness(positions, boutons, candidates, mm_per_deg):
    if len(boutons) < 10:
        return {"clark_evans": np.nan}
    # null boutons: drawn from the candidates in proportion to the boutons' density smoothed over smooth_deg
    bin_mm = 0.05 * mm_per_deg
    ij = np.floor((positions - positions.min(axis=0)) / bin_mm).astype(int)
    shape = tuple(ij.max(axis=0) + 1)
    flat = np.ravel_multi_index(tuple(ij.T), shape)

    def smoothed(units):
        counts = np.bincount(flat[units], minlength=np.prod(shape)).reshape(shape).astype(float)
        return ndimage.gaussian_filter(counts, smooth_deg * mm_per_deg / bin_mm)

    rate = (smoothed(boutons) / np.maximum(smoothed(candidates), 1e-12)).ravel()[flat[candidates]]
    null_nn = [mean_nn(positions[rng.choice(candidates, len(boutons), replace=False, p=rate / rate.sum())])
               for _ in range(n_null)]
    return {"clark_evans": mean_nn(positions[boutons]) / np.mean(null_nn)}


args = argparse.Namespace(config=CONFIG, layer=LAYER, opts=None)
cfg = get_cfg(args)
variant = get_variant(cfg.variant.name)
variant.set_eval_cfg(cfg)

x = np.load(Path(cfg.output_dir) / args.layer / "axial_specificity.npz")
cv, pref_angles, modulation = x["cv"], x["pref_angles"], x["modulation"]
positions, cell_h, cell_w, H, W = load_tissue(args.layer, cv.shape[0])
mm_per_deg = np.ptp(positions, axis=0).max() / 8.0
rf_dist = rf_distance_deg(cell_h, cell_w, H, W, modulation)
tuned = cv <= cv_max
# units the grating doesn't reach have modulation ~1e-8 (float32 rounding), so require real drive
driven = modulation > 1e-4 * np.abs(modulation).max()

# smoothed preferences: von Mises fits to each unit's full-field tuning curve (SineResponses tiles llcnn channels into
# the same sheet as load_tissue, so units line up with positions)
variant.set_eval_model(cfg)
responses = get_sine_responses(variant.eval_model, [args.layer], is_llcnn=True)
tissue = V1Map(positions, responses, output_dir=Path(cfg.output_dir) / args.layer, skip_cache=False,
               smooth_orientation_tuning_curves=True)
smoothed_pref = tissue.orientation_preferences
assert len(smoothed_pref) == len(cv)

# --- bouton map at theta = 0 ---
angle = 0
map_boutons = top_units(modulation[angle], np.flatnonzero(driven[angle] & (rf_dist >= map_far_deg) & tuned))
fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.scatter(*positions[tuned].T, s=5, c=smoothed_pref[tuned], cmap=nauhaus_colormaps["angles"], vmin=0, vmax=180,
           linewidths=0, zorder=1)
ax.scatter(*positions[map_boutons].T, s=0.4, facecolor="black", zorder=2, alpha=0.5)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
sns.despine(ax=ax, left=True, bottom=True)
plt.tight_layout()
plt.savefig(OUT_DIR / "specificity_llcnn.png", transparent=True, dpi=300)
print(f"map: {len(map_boutons)} boutons with RF >= {map_far_deg} deg, all within {rf_dist[map_boutons].max():.2f} deg")

# --- numbers ---
far = (rf_dist >= far_deg) & tuned
oriented = modulation - modulation.mean(axis=0, keepdims=True)
plain = [top_units(modulation[a], np.flatnonzero(far & driven[a])) for a in range(len(angle_vals))]
times = np.bincount(np.concatenate(plain).astype(int), minlength=len(cv))

rows = []
for angle, theta in enumerate(angle_vals):
    candidates = np.flatnonzero(far & driven[angle])
    boutons = top_units(oriented[angle], candidates)
    rows.append({
        "model": "llcnn", "theta": theta, "n_candidates": len(candidates), "n_boutons": len(boutons),
        "always_on_frac": np.mean(times[plain[angle]] >= 6) if len(plain[angle]) else np.nan,
        **specificity(pref_angles[boutons], theta),
        **patchiness(positions, plain[angle], candidates, mm_per_deg),
    })

by_grating = pd.DataFrame(rows)
by_grating.to_csv(OUT_DIR / "llcnn_specificity_by_grating.csv", index=False)

metrics = ["n_boutons", "always_on_frac", "frac_far", "frac_orth", "iso_index", "clark_evans"]
g = by_grating.groupby("model")[metrics]
summary = g.mean().join(g.std()[[c for c in metrics if c != "n_boutons"]], rsuffix="_sd")
summary["chance_frac_far"] = 5 / 8
summary["malach_z"] = (summary["frac_far"] - MALACH_MEAN) / MALACH_SD
summary["within_malach"] = summary["malach_z"].abs() <= 1
summary["orth_below_15pct"] = summary["frac_orth"] < 0.15
summary.to_csv(OUT_DIR / "llcnn_specificity.csv")
pd.set_option("display.width", 250)
print(summary.round(3).to_string())
