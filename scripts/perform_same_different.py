import argparse
import numpy as np
import pickle as pkl
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

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
    return parser


args = get_parser().parse_args()


def fit_probe_with_cv(probe_type: str, X_train: np.ndarray, y_train: np.ndarray, cv: int, n_jobs: int):
    if probe_type == "linear":
        estimator = LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="auto")
        param_grid = {
            "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        }
    elif probe_type == "mlp":
        estimator = MLPClassifier(max_iter=300, early_stopping=True, validation_fraction=0.1)
        param_grid = {
            "hidden_layer_sizes": [(256,), (512,), (256, 256)],
            "learning_rate_init": [1e-4, 1e-3, 1e-2],
            "alpha": [1e-4, 1e-3, 1e-2],
        }

    search = GridSearchCV(
        estimator,
        param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=n_jobs,
        verbose=2,
        refit=True,
    )
    search.fit(X_train, y_train)

    print(f"Best params: {search.best_params_}")
    print(f"Best CV accuracy: {search.best_score_:.4f}")

    return search.best_estimator_, search.best_params_, search.best_score_, search.cv_results_


def main():
    cfg = get_cfg(args)
    variant = get_variant(cfg.variant.name)
    variant.set_eval_cfg(cfg)
    variant.set_eval_model(cfg)
    model = variant.eval_model

    save_dir = Path(cfg.output_dir) / args.layer / "discrimination"
    save_dir.mkdir(parents=True, exist_ok=True)

    # train
    train_dataset = DatasetRegistry.get("Discrimination_train")
    train_features, _, train_labels = get_features_from_layer(
        model=model,
        dataset=train_dataset,
        verbose=True,
        batch_size=128,
        model_layer_strings=[args.layer],
        return_inputs_and_labels=True,
    )

    X_train = train_features.reshape(len(train_labels), -1)
    y_train = np.array(train_labels)
    print("Unique train labels:", np.unique(y_train))
    assert len(X_train) == 6400, f"Expected 6400 train images, got {len(X_train)}"

    # re-train on full dataset
    probe, best_params, best_cv_acc, cv_results = fit_probe_with_cv(
        args.probe, X_train, y_train, cv=args.cv_folds, n_jobs=args.n_jobs
    )

    train_preds = probe.predict(X_train)
    train_correct = (train_preds == y_train).astype(int)
    train_acc = accuracy_score(y_train, train_preds)
    print(f"Train accuracy (refit on full train): {train_acc:.4f}")

    # validate
    val_dataset = DatasetRegistry.get("Discrimination_val")
    val_features, _, val_labels = get_features_from_layer(
        model=model,
        dataset=val_dataset,
        verbose=True,
        batch_size=128,
        model_layer_strings=[args.layer],
        return_inputs_and_labels=True,
    )

    X_val = val_features.reshape(len(val_labels), -1)
    y_val = np.array(val_labels)
    print("Unique val labels:", np.unique(y_val))

    val_preds = probe.predict(X_val)
    val_correct = (val_preds == y_val).astype(int)
    val_acc = accuracy_score(y_val, val_preds)
    print(f"Val accuracy: {val_acc:.4f}")

    results = {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "best_cv_acc": best_cv_acc,
        "best_params": best_params,
        "cv_results": cv_results,
        "train_labels": y_train,
        "train_preds": train_preds,
        "train_correct": train_correct,
        "val_labels": y_val,
        "val_preds": val_preds,
        "val_correct": val_correct,
    }

    out_path = save_dir / f"{args.probe}_results.pkl"
    with open(out_path, "wb") as f:
        pkl.dump(results, f)
    print(f"Saved results to {out_path}")

    probe_path = save_dir / f"{args.probe}_probe.pkl"
    with open(probe_path, "wb") as f:
        pkl.dump(probe, f)
    print(f"Saved probe to {probe_path}")


if __name__ == "__main__":
    main()