import argparse
import numpy as np
import pickle as pkl
from pathlib import Path
from scipy import ndimage
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

from spacetorch.configs import get_cfg
from spacetorch.variants import get_variant
from spacetorch.datasets import DatasetRegistry
from spacetorch.feature_extractor import get_features_from_layer


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--layer", type=str)
    parser.add_argument("--probe", type=str, choices=["linear", "mlp"], default="linear")
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--n_jobs", type=int, default=5, help="Parallel jobs for GridSearchCV")
    parser.add_argument("--tol", type=float, default=0.05,
                         help="Distance from corner-sampled background color, above which a pixel counts as foreground")
    parser.add_argument("--mode", type=str, choices=["joint", "separate", "both"], default="both",
                         help="'joint' fits one 4-dim probe for both objects at once; "
                              "'separate' fits two independent 2-dim probes, one per object; "
                              "'both' does both and logs each independently")
    return parser


args = get_parser().parse_args()


def get_two_object_bboxes_normalized(img, tol=0.05, min_area_frac=0.001):
    """
    Detects background as the single most common pixel color (assumes background
    covers the majority of the image, which holds for these stimuli), then
    finds connected components of pixels that differ from it.

    Returns [(x0, y0, x1, y1), (x0, y0, x1, y1)] in PIXEL coords, sorted
    left-to-right by x0, or None if fewer than 2 valid components are found
    (caller should skip that sample).
    """
    img = np.asarray(img, dtype=np.float32)
    H, W, C = img.shape

    # background color = the pixel value at a guaranteed-background
    # location: the four corners should always be background for
    # centered objects; average them for robustness
    corner_pixels = np.stack([img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]])
    bg_color = corner_pixels.mean(axis=0)  # (C,)

    dist_from_bg = np.linalg.norm(img - bg_color[None, None, :], axis=-1)
    non_bg = dist_from_bg > tol

    labeled, n_components = ndimage.label(non_bg)
    min_area = min_area_frac * H * W

    components = []
    for label_id in range(1, n_components + 1):
        mask = labeled == label_id
        area = mask.sum()
        if area < min_area:
            continue
        ys, xs = np.where(mask)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        components.append((area, bbox))

    if len(components) < 2:
        return None

    components.sort(key=lambda c: -c[0])
    top_two = components[:2]
    top_two.sort(key=lambda c: c[1][0])  # sort by x0, left-to-right

    return [top_two[0][1], top_two[1][1]]


def build_position_targets(inputs, tol=0.05):
    """
    inputs: (N, H, W, C) array, native scale (normalized or [0,1] both fine
    since detection is relative to each image's own corner color).

    Returns:
        valid_idx: indices of samples where both objects were detected
        Y: (n_valid, 8) array of
           [x0_a, y0_a, x1_a, y1_a, x0_b, y0_b, x1_b, y1_b], each coordinate
           normalized to [0, 1] by image width/height.
    """
    valid_idx, targets = [], []
    for i, img in enumerate(inputs):
        H, W = img.shape[:2]
        bboxes = get_two_object_bboxes_normalized(img, tol=tol)
        if bboxes is None:
            continue
        (x0_a, y0_a, x1_a, y1_a), (x0_b, y0_b, x1_b, y1_b) = bboxes
        valid_idx.append(i)
        targets.append([
            x0_a / W, y0_a / H, x1_a / W, y1_a / H,
            x0_b / W, y0_b / H, x1_b / W, y1_b / H,
        ])
    return np.array(valid_idx), np.array(targets, dtype=np.float32)


def fit_regression_probe_with_cv(probe_type: str, X_train: np.ndarray, Y_train: np.ndarray, cv: int, n_jobs: int):
    if probe_type == "linear":
        estimator = Ridge()
        param_grid = {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}
    elif probe_type == "mlp":
        estimator = MLPRegressor(max_iter=300, early_stopping=True, validation_fraction=0.1)
        param_grid = {
            "hidden_layer_sizes": [(256,), (512,), (256, 256)],
            "learning_rate_init": [1e-4, 1e-3, 1e-2],
            "alpha": [1e-4, 1e-3, 1e-2],
        }

    search = GridSearchCV(
        estimator,
        param_grid,
        cv=cv,
        scoring="r2",  # for multi-dim Y, this is the uniform-average r2 across dims
        n_jobs=n_jobs,
        verbose=2,
        refit=True,
    )
    search.fit(X_train, Y_train)

    print(f"Best params: {search.best_params_}")
    print(f"Best CV R^2: {search.best_score_:.4f}")

    return search.best_estimator_, search.best_params_, search.best_score_, search.cv_results_


def fit_and_eval(name, X_train, Y_train, X_val, Y_val, dim_names, probe_type, cv, n_jobs):
    """
    Fits one probe (via its own independent GridSearchCV) on (X_train, Y_train),
    evaluates on both train (refit) and val, and logs everything under `name`
    so joint vs. per-object results never share a hyperparameter search.
    """
    print(f"\n=== Fitting probe: {name} (target dims: {dim_names}) ===")
    probe, best_params, best_cv_r2, cv_results = fit_regression_probe_with_cv(
        probe_type, X_train, Y_train, cv=cv, n_jobs=n_jobs
    )

    train_preds = probe.predict(X_train)
    train_r2 = r2_score(Y_train, train_preds)
    train_mse = mean_squared_error(Y_train, train_preds)
    train_r2_per_dim = {
        dn: r2_score(Y_train[:, i], train_preds[:, i]) for i, dn in enumerate(dim_names)
    }
    print(f"[{name}] Train R^2 (refit): {train_r2:.4f}, MSE: {train_mse:.4f}")
    print(f"[{name}] Train R^2 per dim: {train_r2_per_dim}")

    val_preds = probe.predict(X_val)
    val_r2 = r2_score(Y_val, val_preds)
    val_mse = mean_squared_error(Y_val, val_preds)
    val_r2_per_dim = {
        dn: r2_score(Y_val[:, i], val_preds[:, i]) for i, dn in enumerate(dim_names)
    }
    print(f"[{name}] Val R^2: {val_r2:.4f}, MSE: {val_mse:.4f}")
    print(f"[{name}] Val R^2 per dim: {val_r2_per_dim}")

    return probe, {
        "best_params": best_params,
        "best_cv_r2": best_cv_r2,
        "cv_results": cv_results,
        "train_r2": train_r2,
        "train_mse": train_mse,
        "train_r2_per_dim": train_r2_per_dim,
        "train_preds": train_preds,
        "val_r2": val_r2,
        "val_mse": val_mse,
        "val_r2_per_dim": val_r2_per_dim,
        "val_preds": val_preds,
    }


def main():
    cfg = get_cfg(args)
    variant = get_variant(cfg.variant.name)
    variant.set_eval_cfg(cfg)
    variant.set_eval_model(cfg)
    model = variant.eval_model

    save_dir = Path(cfg.output_dir) / args.layer / "position_decoding"
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- train ---
    train_dataset = DatasetRegistry.get("Discrimination_train")
    train_features, train_inputs, train_labels = get_features_from_layer(
        model=model,
        dataset=train_dataset,
        verbose=True,
        batch_size=128,
        model_layer_strings=[args.layer],
        return_inputs_and_labels=True,
    )

    train_inputs = np.asarray(train_inputs)
    train_inputs_hwc = train_inputs.transpose(0, 2, 3, 1)
    train_valid_idx, Y_train_all = build_position_targets(train_inputs_hwc, args.tol)
    print(f"Train: {len(train_valid_idx)}/{len(train_inputs)} samples had both objects detected")

    X_train_full = np.asarray(train_features).reshape(len(train_labels), -1)
    X_train = X_train_full[train_valid_idx]

    # --- validate (features extracted once, reused across all probes) ---
    val_dataset = DatasetRegistry.get("Discrimination_val")
    val_features, val_inputs, val_labels = get_features_from_layer(
        model=model,
        dataset=val_dataset,
        verbose=True,
        batch_size=128,
        model_layer_strings=[args.layer],
        return_inputs_and_labels=True,
    )

    val_inputs = np.asarray(val_inputs)
    val_inputs_hwc = val_inputs.transpose(0, 2, 3, 1)
    val_valid_idx, Y_val_all = build_position_targets(val_inputs_hwc, args.tol)
    print(f"Val: {len(val_valid_idx)}/{len(val_inputs)} samples had both objects detected")

    X_val_full = np.asarray(val_features).reshape(len(val_labels), -1)
    X_val = X_val_full[val_valid_idx]

    results = {"train_valid_idx": train_valid_idx, "val_valid_idx": val_valid_idx}
    probes = {}

    bbox_dim_names = ["x0", "y0", "x1", "y1"]
    joint_dim_names = [f"{d}_a" for d in bbox_dim_names] + [f"{d}_b" for d in bbox_dim_names]

    # --- joint: one probe predicts all 8 dims (bbox_a + bbox_b) at once ---
    if args.mode in ("joint", "both"):
        probe_joint, res_joint = fit_and_eval(
            "joint", X_train, Y_train_all, X_val, Y_val_all,
            dim_names=joint_dim_names,
            probe_type=args.probe, cv=args.cv_folds, n_jobs=args.n_jobs,
        )
        probes["joint"] = probe_joint
        results["joint"] = res_joint

    # --- separate: independent probe per object (4-dim bbox each), own GridSearchCV each ---
    if args.mode in ("separate", "both"):
        probe_a, res_a = fit_and_eval(
            "object_a", X_train, Y_train_all[:, 0:4], X_val, Y_val_all[:, 0:4],
            dim_names=bbox_dim_names,
            probe_type=args.probe, cv=args.cv_folds, n_jobs=args.n_jobs,
        )
        probe_b, res_b = fit_and_eval(
            "object_b", X_train, Y_train_all[:, 4:8], X_val, Y_val_all[:, 4:8],
            dim_names=bbox_dim_names,
            probe_type=args.probe, cv=args.cv_folds, n_jobs=args.n_jobs,
        )
        probes["object_a"] = probe_a
        probes["object_b"] = probe_b
        results["object_a"] = res_a
        results["object_b"] = res_b

    out_path = save_dir / f"{args.probe}_position_results_{args.mode}.pkl"
    with open(out_path, "wb") as f:
        pkl.dump(results, f)
    print(f"\nSaved results to {out_path}")

    probes_path = save_dir / f"{args.probe}_position_probes_{args.mode}.pkl"
    with open(probes_path, "wb") as f:
        pkl.dump(probes, f)
    print(f"Saved probes to {probes_path}")


if __name__ == "__main__":
    main()
