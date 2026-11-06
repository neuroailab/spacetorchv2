"""
Iso-orientation specificity and patchiness of lateral spread from a 0.5 deg central grating, per model, compared with
Malach et al. 1993 (PNAS 90:10469).

Boutons. For each grating orientation theta, candidates are driven units (modulation > 1e-4 of the model's max) whose RF
center is >= far_deg from the grating and whose circular variance is <= cv_max. far_deg = 2 is beyond the CNNs'
feed-forward reach (their drive past ~2 deg is float noise), so CNNs have no boutons. Two bouton sets, each the top
(1 - m) of candidates:
    plain      -- by modulation[theta]. Includes always-on units: a few ViT channels respond to any central grating at
                  nearly every token, and since each channel's units sit together on the sheet they light up as clumps.
                  always_on_frac = fraction of plain boutons that are plain boutons for >= 6 of the 8 gratings.
    oriented   -- by modulation[theta] - mean over the 8 thetas, which removes the always-on units: what this grating
                  in particular recruits.
Specificity uses the oriented boutons; patchiness uses the plain boutons.

Specificity (oriented boutons), from each bouton's own full-field preferred orientation pref and delta = |pref - theta| (axial, 0-90 deg):
    frac_far   -- fraction of boutons with delta >= 45. Malach: 30 +- 10% of patches.
    frac_orth  -- fraction with delta >= 67.5. Malach: < 15%.
    iso_index  -- mean cos(2 * delta): 1 = all iso-oriented, 0 = no relation (also used when there are no boutons).
Preferences are argmaxes over the same 8 orientations as theta, so delta takes only 0, 22.5, 45, 67.5 or 90 and boutons
with no relation to theta give frac_far = 5/8 (chance_frac_far), not 0.5.
Against Malach: malach_z = (frac_far - 0.30) / 0.10, i.e. distance from Malach's mean in units of Malach's spread (their
coefficient of variation is 0.10 / 0.30 = 0.33); within_malach = |malach_z| <= 1. A model without boutons is not within.

Patchiness (plain boutons; dense clumps separated by gaps), from bouton positions on the cortical sheet:
    clark_evans -- mean nearest-neighbour distance between boutons / the same for null maps with the same number of
                   boutons. Null boutons are drawn from the candidates with probability proportional to the boutons'
                   own density smoothed over smooth_deg, so a null map keeps the real map's large-scale gradients (e.g.
                   more boutons near the injection site) but none of its clumping below that scale.
                   1 = scattered, < 1 = patchy. Barely changes for smooth_deg between 0.5 and 2 deg.
Outputs: OUT_DIR / "quantifying_specificity.csv" (one row per model: mean and SD over the 8 gratings),
OUT_DIR / "quantifying_specificity_by_grating.csv" (one row per model and grating) and OUT_DIR / "quantifying_specificity.png"
(per model: frac_far against Malach's 30 +- 10%, and clark_evans).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

from spacetorch.configs import get_cfg
from spacetorch.variants import get_variant
from spacetorch.variants.positions import get_positions

angle_vals = np.linspace(0, 180, 8, endpoint=False)

far_deg = 2        # boutons must have their RF center >= this far from the 0.5 deg grating
m = 0.95             # boutons = top (1 - m) of candidates by orientation-specific modulation
cv_max = 1.0         # keep only units with circular variance <= this (1 = keep all)
smooth_deg = 1.0     # scale of the large-scale density kept by the patchiness null; the tissue spans the 8 deg image
n_null = 20          # null maps for clark_evans
MALACH_MEAN, MALACH_SD = 0.30, 0.10
rng = np.random.default_rng(0)

OUT_DIR = Path("/ccn2/u/ynshah/spacetorchv2/paper")

models_and_layers = [
    ("/ccn2/u/ynshah/spacetorchv2/configs/alexnet_supervised_imagenet/swapopt_lwx1.yaml", "_feature_blocks.2.1"),
    ("/ccn2/u/ynshah/spacetorchv2/configs/alexnet_rotnet_imagenet/swapopt_lwx1.yaml", "_feature_blocks.2.1"),
    ("/ccn2/u/ynshah/spacetorchv2/configs/alexnet_deepcluster_imagenet/swapopt_lwx1.yaml", "features.6"),
    ("/ccn2/u/ynshah/spacetorchv2/configs/tdann/swapopt_lwx1_seed3.yaml", "layer2.0"),
    ("/ccn2/u/ynshah/spacetorchv2/configs/tdann/supervised_swapopt_lwx1.yaml", "layer2.0"),
    ("/ccn2/u/ynshah/spacetorchv2/configs/resnet50_dino_imagenet/swapopt_lwx1.yaml", "layer2.0"),
    ("/ccn2/u/ynshah/spacetorchv2/configs/resnet50_mocov3_imagenet/swapopt_lwx1.yaml", "layer2.0"),
    # ("/ccn2/u/ynshah/spacetorchv2/configs/vitb16a_supervised_imagenet/unoptimized.yaml", "blocks.1.attn"),
    ("/ccn2/u/ynshah/spacetorchv2/configs/vitb16a_supervised_imagenet/taskonly.yaml", "blocks.1.attn"),
    ("/ccn2/u/ynshah/spacetorchv2/configs/vitb16a_supervised_imagenet/swapopt_lwx1.yaml", "blocks.1.attn"),
    ("/ccn2/u/ynshah/spacetorchv2/configs/vitb16a_simclr_imagenet/swapopt_lwx1.yaml",     "blocks.2.attn"),
    ("/ccn2/u/ynshah/spacetorchv2/configs/vitb16a_mocov3_imagenet/swapopt_lwx1.yaml",     "blocks.2.attn"),
    ("/ccn2/u/ynshah/spacetorchv2/configs/vitb14a_dinov2_imagenet/swapopt_lwx1.yaml",     "blocks.2.attn"),
]
MODEL_NAMES = [
    "alexnet_supervised", "alexnet_rotnet", "alexnet_deepcluster", "tdann", "tdann_supervised", "resnet_dino",
    "resnet_mocov3", "taskonly", "supervised", "simclr", "mocov3", "dinov2",
]

# unoptimized.yaml has relative paths
CFG_OPTS = {"random_init_vit": [
    "output_dir=/data/ynshah/spacetorchv2/checkpoints/vitb16a_supervised_imagenet_unoptimized",
    "spatial_loss.position_dir=/data/ynshah/spacetorchv2/checkpoints/vitb16a_supervised_imagenet/positions",
]}


def rf_distance_deg(dims, modulation):
    # units are flattened (C, H, W); RF center of unit (c, h, w) is ~grid cell (h, w) of the 8 deg image
    C, H, W = dims
    h, w = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    x = ((w + 0.5) * 8.0 / W - 4.0).ravel()
    y = ((h + 0.5) * 8.0 / H - 4.0).ravel()
    # CNN stride/padding shifts true RF centers off the grid cells (up to ~0.15 deg), so locate the grating
    # in feature-map coordinates: drive-weighted centroid of the central (< 1 deg) cells at >= half max drive
    drive = np.abs(modulation).mean(axis=0).reshape(C, H * W).mean(axis=0)
    near = np.hypot(x, y) < 1.0
    wts = np.where(near & (drive >= 0.5 * drive[near].max()), drive, 0)
    x0, y0 = (x * wts).sum() / wts.sum(), (y * wts).sum() / wts.sum()
    return np.tile(np.hypot(x - x0, y - y0), C)


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


rows = []
for model_name, (config_path, layer) in zip(MODEL_NAMES, models_and_layers):
    args = argparse.Namespace(config=config_path, layer=layer, opts=CFG_OPTS.get(model_name))

    cfg = get_cfg(args)
    variant = get_variant(cfg.variant.name)
    variant.set_eval_cfg(cfg)

    save_dir = Path(cfg.output_dir) / args.layer
    try:
        x = np.load(save_dir / "axial_specificity.npz")
    except FileNotFoundError:
        print(f"Could not load {save_dir / 'axial_specificity.npz'} -- skipping")
        continue
    cv, pref_angles, modulation = x["cv"], x["pref_angles"], x["modulation"]

    is_tdann = "tdann" in cfg.name and not "tdann_logpolar" in cfg.name
    positions_obj = get_positions(cfg, rescale=is_tdann)[args.layer]
    positions = positions_obj.coordinates
    assert cv.shape[0] == positions.shape[0]
    mm_per_deg = np.ptp(positions, axis=0).max() / 8.0

    rf_dist = rf_distance_deg(positions_obj.dims, modulation)
    far = (rf_dist >= far_deg) & (cv <= cv_max)
    oriented = modulation - modulation.mean(axis=0, keepdims=True)
    # CNN units beyond their feed-forward reach have modulation ~1e-12 (float noise), so require real drive
    driven = modulation > 1e-4 * np.abs(modulation).max()

    # always-on units: plain top-(1 - m) boutons for >= 6 of the 8 gratings
    plain = [top_units(modulation[a], np.flatnonzero(far & driven[a])) for a in range(len(angle_vals))]
    times = np.bincount(np.concatenate(plain).astype(int), minlength=len(cv))

    for angle, theta in enumerate(angle_vals):
        candidates = np.flatnonzero(far & driven[angle])
        boutons = top_units(oriented[angle], candidates)
        row = {
            "model": model_name, "theta": theta, "n_candidates": len(candidates), "n_boutons": len(boutons),
            "always_on_frac": np.mean(times[plain[angle]] >= 6) if len(plain[angle]) else np.nan,
            **specificity(pref_angles[boutons], theta),
            **patchiness(positions, plain[angle], candidates, mm_per_deg),
        }
        rows.append(row)
        print(f"{model_name:20s} theta={theta:5.1f}  n={len(boutons):5d}  frac_far={row['frac_far']:.2f}  "
              f"iso={row['iso_index']:+.2f}  clark_evans={row['clark_evans']:.2f}")


by_grating = pd.DataFrame(rows)
by_grating.to_csv(OUT_DIR / "quantifying_specificity_by_grating.csv", index=False)

# one row per model: mean and SD over the 8 gratings, plus the comparison with Malach
metrics = ["n_boutons", "always_on_frac", "frac_far", "frac_orth", "iso_index", "clark_evans"]
g = by_grating.groupby("model", sort=False)[metrics]
summary = g.mean().join(g.std()[[c for c in metrics if c != "n_boutons"]], rsuffix="_sd")
summary["chance_frac_far"] = 5 / 8
summary["malach_z"] = (summary["frac_far"] - MALACH_MEAN) / MALACH_SD
summary["within_malach"] = summary["malach_z"].abs() <= 1
summary["orth_below_15pct"] = summary["frac_orth"] < 0.15
summary.to_csv(OUT_DIR / "quantifying_specificity.csv")
pd.set_option("display.width", 250)
print(summary[["n_boutons", "always_on_frac", "frac_far", "frac_far_sd", "malach_z", "within_malach", "frac_orth",
               "iso_index", "clark_evans", "clark_evans_sd"]].round(3).to_string())

# per model: fraction of boutons >= 45 deg from the grating (Malach's 30 +- 10% shaded, chance dashed) and patchiness,
# mean +- SD over gratings; models without boutons are left out
plotted = summary.dropna(subset=["frac_far"])
fig, axes = plt.subplots(1, 2, figsize=(2 * (0.4 * len(plotted) + 1), 2.6))
xs = np.arange(len(plotted))
axes[0].axhspan(MALACH_MEAN - MALACH_SD, MALACH_MEAN + MALACH_SD, color="0.9", zorder=0)
axes[0].axhline(5 / 8, color="0.5", ls="--", lw=0.6)
axes[0].errorbar(xs, plotted["frac_far"], yerr=plotted["frac_far_sd"], fmt="o", ms=4, color="black")
axes[0].set_ylim(0, 1)
axes[0].set_ylabel("boutons $\\geq$45° from grating", fontsize=8)
axes[1].axhline(1, color="0.5", ls="--", lw=0.6)
axes[1].errorbar(xs, plotted["clark_evans"], yerr=plotted["clark_evans_sd"], fmt="o", ms=4, color="black")
axes[1].set_ylabel("Clark-Evans ratio\n(< 1 = patchy)", fontsize=8)
for ax in axes:
    ax.set_xticks(xs)
    ax.set_xticklabels(plotted.index, rotation=60, ha="right", fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(OUT_DIR / "quantifying_specificity.png", dpi=300)
