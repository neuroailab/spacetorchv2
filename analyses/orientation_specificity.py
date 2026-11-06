import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from spacetorch.configs import get_cfg
from spacetorch.variants import get_variant
from spacetorch.variants.positions import get_positions
from spacetorch.colormaps import nauhaus_colormaps
from spacetorch.maps.sine_tissue import get_sine_tissue, METRIC_DICT, get_smoothed_map
from spacetorch.maps.smoother import Smoother, KernelParams

angle_metric = METRIC_DICT["angles"]

far_deg = 0.0  # boutons must have their RF center >= this far from the 0.5 deg grating (0 = include units that see it directly)

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

models_and_layers = [
    # ("/ccn2/u/ynshah/spacetorchv2/configs/alexnet_supervised_imagenet/swapopt_lwx1.yaml", "_feature_blocks.2.1"),
    # ("/ccn2/u/ynshah/spacetorchv2/configs/alexnet_rotnet_imagenet/swapopt_lwx1.yaml", "_feature_blocks.2.1"),
    # ("/ccn2/u/ynshah/spacetorchv2/configs/alexnet_deepcluster_imagenet/swapopt_lwx1.yaml", "features.6"),
    # ("/ccn2/u/ynshah/spacetorchv2/configs/tdann/swapopt_lwx1_seed3.yaml", "layer2.0"),
    # ("/ccn2/u/ynshah/spacetorchv2/configs/tdann/supervised_swapopt_lwx1.yaml", "layer2.0"),
    # ("/ccn2/u/ynshah/spacetorchv2/configs/resnet50_dino_imagenet/swapopt_lwx1.yaml", "layer2.0"),
    # ("/ccn2/u/ynshah/spacetorchv2/configs/resnet50_mocov3_imagenet/swapopt_lwx1.yaml", "layer2.0"),
    # ("/ccn2/u/ynshah/spacetorchv2/configs/vitb16a_supervised_imagenet/unoptimized.yaml", "blocks.1.attn"),
    ("/ccn2/u/ynshah/spacetorchv2/configs/vitb16a_supervised_imagenet/taskonly.yaml", "blocks.1.attn"),
    # ("/ccn2/u/ynshah/spacetorchv2/configs/vitb16a_supervised_imagenet/swapopt_lwx1.yaml", "blocks.1.attn"),
    # ("/ccn2/u/ynshah/spacetorchv2/configs/vitb16a_simclr_imagenet/swapopt_lwx1.yaml",     "blocks.2.attn"),
    # ("/ccn2/u/ynshah/spacetorchv2/configs/vitb16a_mocov3_imagenet/swapopt_lwx1.yaml",     "blocks.2.attn"),
    # ("/ccn2/u/ynshah/spacetorchv2/configs/vitb14a_dinov2_imagenet/swapopt_lwx1.yaml",     "blocks.2.attn"),
]

MODEL_COLORS = {
    # "alexnet_supervised": "#d10808",
    # "alexnet_rotnet": "#d10808",
    # "alexnet_deepcluster": "#d10808",
    # "tdann": "#d10808",
    # "tdann_supervised": "#d10808",
    # "resnet_dino": "#d10808",
    # "resnet_mocov3": "#d10808",
    # "random_init_vit": "#000000",
    "taskonly": (0.6, 0.6, 0.6),
    # "supervised": "#072f5f",
    # "simclr":     "#1261a0",
    # "mocov3":    "#3895d3",
    # "dinov2":    "#58cced",
}

# unoptimized.yaml has relative paths
CFG_OPTS = {"random_init_vit": [
    "output_dir=/data/ynshah/spacetorchv2/checkpoints/vitb16a_supervised_imagenet_unoptimized",
    "spatial_loss.position_dir=/data/ynshah/spacetorchv2/checkpoints/vitb16a_supervised_imagenet/positions",
]}

def robust_modulation_mask(mod_vals, m, gap_ratio_cap=5000):
    sorted_vals = np.sort(mod_vals)
    n = len(sorted_vals)
    cut_idx = int(m * n)
    threshold = sorted_vals[cut_idx]

    bulk = sorted_vals[:cut_idx]                     # everything below threshold
    selected = sorted_vals[cut_idx:]                  # candidate "modulated" units

    # typical spacing in the bulk (robust to its own outliers via median diff)
    typical_gap = np.median(np.diff(bulk))
    if typical_gap == 0:
        # many tied values (e.g. random-init ViT): use the typical nonzero spacing instead
        diffs = np.diff(bulk)
        typical_gap = np.median(diffs[diffs > 0]) if np.any(diffs > 0) else np.finfo(float).eps

    # spread of the selected group, normalized by how many units are in it
    selected_span = selected[-1] - selected[0]
    expected_span = typical_gap * len(selected)

    print(selected_span, expected_span, selected_span / expected_span)

    if selected_span > gap_ratio_cap * expected_span:
        # top bin is outlier-dominated — selected group isn't smoothly
        # continuous with the rest of the distribution
        return None  # signal "don't trust this m"

    return mod_vals > threshold

for model_name, (config_path, layer) in zip(MODEL_COLORS.keys(), models_and_layers):
    args = argparse.Namespace(config=config_path, layer=layer, opts=CFG_OPTS.get(model_name))

    cfg = get_cfg(args)
    variant = get_variant(cfg.variant.name)
    variant.set_eval_cfg(cfg)
    variant.set_eval_model(cfg)

    save_dir = Path(cfg.output_dir) / args.layer
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        x = np.load(save_dir / "axial_specificity.npz")
    except:
        print(f"Could not load {save_dir / 'axial_specificity.npz'} -- skipping")
        continue

    cv, modulation = x["cv"], x["modulation"]

    model = variant.eval_model
    is_tdann = "tdann" in cfg.name and not "tdann_logpolar" in cfg.name
    positions_obj = get_positions(cfg, rescale=is_tdann)[args.layer]

    positions = positions_obj.coordinates

    assert cv.shape[0] == positions.shape[0], (
        f"cv has {cv.shape[0]} units but tissue has {positions.shape[0]} positions -- "
        "these must come from the exact same unit ordering."
    )

    angle_vals = np.linspace(0, 180, 8, endpoint=False)
    cv_mask = cv <= 1
    center = np.array([np.ptp(positions[:, 0]), np.ptp(positions[:, 1])]) / 2 + positions.min(axis=0)
    radius = 0

    distances = np.linalg.norm(positions - center, axis=1)
    peripheral_mask = distances > radius

    angle = 0
    target_angle = angle_vals[angle]

    smoother = Smoother(KernelParams(width=1.5, stride=0.15))

    # CNN units outside the grating's RF have modulation ~1e-12 (float noise), not exactly 0, so over all units
    # the top 5% is outlier-dominated; rank only units that are driven and whose RF is >= far_deg from the grating
    rf_dist = rf_distance_deg(positions_obj.dims, modulation)
    pool = (modulation[angle] > 1e-4 * np.abs(modulation).max()) & (rf_dist >= far_deg)
    # the gap check guarded against CNN float-noise units, which `pool` now excludes; with far_deg = 0 the units under
    # the grating are legitimately far above the rest (feed-forward drive), so don't let them trip it
    pool_mask = robust_modulation_mask(modulation[angle][pool], 0.95, gap_ratio_cap=np.inf)

    if pool_mask is None:
        print(f"m={0.95} is too high for {model_name} at angle {target_angle} -- skipping")
        continue

    modulation_mask = np.zeros_like(pool)
    modulation_mask[pool] = pool_mask
    print(np.sum(modulation_mask))

    v1_tissue = get_sine_tissue(
        cfg.name, model, positions_obj, layer=args.layer,
        output_dir=save_dir, smooth_orientation_tuning_curves=False, skip_cache=True,
    )

    v1_tissue.set_mask_by_pct_limits([[0, 100], [0, 100]])
    smoothed = get_smoothed_map(
        v1_tissue, angle_metric, final_width=1.5, final_stride=0.15, verbose=True
    )

    combined_mask = cv_mask & modulation_mask & peripheral_mask

    xlim = (0.0 * 36.75, 1 * 36.75)
    ylim = (0.0 * 36.75, 1 * 36.75)
    zoom_mask = (
        (positions[:, 0] >= xlim[0]) & (positions[:, 0] <= xlim[1]) &
        (positions[:, 1] >= ylim[0]) & (positions[:, 1] <= ylim[1])
    )
    highlight_mask = zoom_mask & combined_mask

    bouton_positions = positions[highlight_mask]
    px, py = smoother.positions_to_pixel_coords(bouton_positions, v1_tissue.positions)
    # star where the grating lands on the tissue (units whose RF center is within 0.5 deg of it)
    star_px, star_py = smoother.positions_to_pixel_coords(np.median(positions[rf_dist < 0.5], axis=0)[None], v1_tissue.positions)

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.imshow(smoothed, cmap=nauhaus_colormaps["angles"], vmin=0, vmax=180, interpolation="nearest")
    ax.scatter(px, py, s=0.4, facecolor="black", zorder=2, alpha=0.5)
    ax.scatter(star_px, star_py, s=200, marker="*", c=["#FFABFF"], zorder=10)
    ax.set_xticks([])
    ax.set_yticks([])
    sns.despine(ax=ax, left=True, bottom=True)

    plt.tight_layout()
    plt.savefig(
        f"/ccn2/u/ynshah/spacetorchv2/paper/specificity_{model_name}.png",
        transparent=True, dpi=300,
    )
