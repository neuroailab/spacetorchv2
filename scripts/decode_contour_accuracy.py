"""Decode contour presence from a model layer, the way decode_accuracy.py does.

The counterpart of ``decode_accuracy.py`` for the contour stimuli: fit a
probe on a layer's responses to the training split and ask, on the held-out
split, whether it can tell a field containing a smooth contour from the same
field with the contour elements scrambled. The protocol is the same -- grid
search on train, refit, score on val -- with two differences that follow from
how the stimuli are built.

* Folds of the hyperparameter search are grouped by seed. The two images of
  a pair share the entire background and differ only in whether eleven-odd
  elements are aligned, so an ungrouped inner fold would tune ``C`` against
  fields the probe had already seen. Train and val already have disjoint
  seeds, so the headline number is unaffected either way; this only keeps
  the model selection honest.
* Accuracy is reported per condition as well as overall. The design is
  fully crossed with one seed per cell -- six relative spacings by four
  eccentricities by five contour lengths by four orientations is 480 cells,
  the number of pairs in each split -- so every marginal is balanced and
  chance is exactly 0.5 within any level, not just overall.

The per-condition table is the point of the analysis rather than a
diagnostic. Contour integration is expected to fail as elements are pushed
apart and as the path gets shorter, to degrade with eccentricity, and to be
roughly indifferent to the contour's orientation; a decoder that is flat in
spacing and length is not reading out integration, however high its overall
accuracy.

Results land in ``<output_dir>/<layer>/contour/``:
    <probe>_results.pkl        predictions, labels and factors per image
    <probe>_probe.pkl          the fitted probe
    <probe>_condition_accuracy.csv
        tidy, one row per (split, factor, level) and per two-way cell of the
        spacing-by-eccentricity and spacing-by-length interactions

Example:
    python3 scripts/decode_contour_accuracy.py \\
        --config configs/tdann/swapopt_lwx1_seed0.yaml --layer layer2.0
"""

import argparse
import itertools
import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from spacetorch.configs import get_cfg
from spacetorch.datasets import DatasetRegistry
from spacetorch.datasets.contour import CONDITION_FACTORS
from spacetorch.feature_extractor import get_features_from_layer
from spacetorch.variants import get_variant

# two-way breakdowns worth having: both are predictions of the association
# field, where spacing sets whether neighbouring elements link at all and
# length sets how much evidence accumulates along the path
INTERACTIONS = [
    ("relative_spacing", "eccentricity_deg"),
    ("relative_spacing", "contour_length_achieved"),
]


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--layer", type=str)
    parser.add_argument("--probe", type=str, choices=["linear", "mlp"], default="linear")
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument(
        "--n_jobs", type=int, default=5, help="Parallel jobs for GridSearchCV"
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="z-score each feature before fitting. Off by default so the "
        "numbers stay comparable with decode_accuracy.py, but worth turning "
        "on if lbfgs fails to converge",
    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=0,
        help="randomly subsample this many features before fitting; 0, the "
        "default, uses all of them. A ViT block has 150,528 responses per "
        "image against 960 images, so this is a speed knob",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    return parser


args = get_parser().parse_args()


def build_estimator(probe_type: str):
    """The estimator and the grid searched over it."""
    if probe_type == "linear":
        estimator = LogisticRegression(max_iter=1000, solver="lbfgs")
        param_grid = {"C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
        prefix = "logisticregression__"
    else:
        estimator = MLPClassifier(
            max_iter=300, early_stopping=True, validation_fraction=0.1
        )
        param_grid = {
            "hidden_layer_sizes": [(256,), (512,), (256, 256)],
            "learning_rate_init": [1e-4, 1e-3, 1e-2],
            "alpha": [1e-4, 1e-3, 1e-2],
        }
        prefix = "mlpclassifier__"

    if args.standardize:
        estimator = make_pipeline(StandardScaler(), estimator)
        param_grid = {prefix + key: value for key, value in param_grid.items()}

    return estimator, param_grid


def fit_probe_with_cv(X, y, groups, cv: int, n_jobs: int):
    """Grid search with folds that keep a seed's two images together."""
    estimator, param_grid = build_estimator(args.probe)
    search = GridSearchCV(
        estimator,
        param_grid,
        cv=GroupKFold(n_splits=cv),
        scoring="accuracy",
        n_jobs=n_jobs,
        verbose=2,
        refit=True,
    )
    search.fit(X, y, groups=groups)

    print(f"Best params: {search.best_params_}")
    print(f"Best CV accuracy: {search.best_score_:.4f}")
    return search


def extract(model, dataset_name: str):
    """Layer responses, labels and design factors for one split."""
    dataset = DatasetRegistry.get(dataset_name)
    features, _, labels = get_features_from_layer(
        model=model,
        dataset=dataset,
        verbose=True,
        batch_size=args.batch_size,
        model_layer_strings=[args.layer],
        return_inputs_and_labels=True,
        shuffle=False,
    )

    X = np.asarray(features).reshape(len(dataset), -1)
    y = np.asarray(dataset.labels)
    assert np.array_equal(np.asarray(labels).ravel(), y), (
        f"{dataset_name}: labels from the feature extractor do not match the "
        "dataset's order; pass shuffle=False"
    )
    print(f"{dataset_name}: {X.shape[0]} images, {X.shape[1]} features, "
          f"{len(np.unique(dataset.seeds))} seeds, "
          f"class balance {y.mean():.3f}")
    return dataset, X, y


def condition_table(dataset, correct: np.ndarray, split: str) -> pd.DataFrame:
    """Accuracy within every level of every crossed factor, plus two-ways.

    Tidy rather than wide: one row per cell, with the second factor blank on
    the marginal rows, so a breakdown that is added later does not change the
    schema of the file.
    """
    rows = []
    for factor in CONDITION_FACTORS:
        values = dataset.factor(factor)
        for level in np.unique(values):
            cell = values == level
            rows.append(
                {
                    "split": split,
                    "factor_a": factor,
                    "level_a": level,
                    "factor_b": "",
                    "level_b": np.nan,
                    "n": int(cell.sum()),
                    "accuracy": float(correct[cell].mean()),
                }
            )

    for first, second in INTERACTIONS:
        a_values, b_values = dataset.factor(first), dataset.factor(second)
        for a_level, b_level in itertools.product(
            np.unique(a_values), np.unique(b_values)
        ):
            cell = (a_values == a_level) & (b_values == b_level)
            if not cell.any():
                continue
            rows.append(
                {
                    "split": split,
                    "factor_a": first,
                    "level_a": a_level,
                    "factor_b": second,
                    "level_b": b_level,
                    "n": int(cell.sum()),
                    "accuracy": float(correct[cell].mean()),
                }
            )

    return pd.DataFrame(rows)


def print_marginals(table: pd.DataFrame, split: str) -> None:
    marginals = table[(table.split == split) & (table.factor_b == "")]
    for factor in CONDITION_FACTORS:
        cells = marginals[marginals.factor_a == factor]
        parts = [
            f"{row.level_a:g}: {row.accuracy:.3f} (n={row.n})"
            for row in cells.itertuples()
        ]
        print(f"  {factor:26s} " + "   ".join(parts))


def main():
    cfg = get_cfg(args)
    variant = get_variant(cfg.variant.name)
    variant.set_eval_cfg(cfg)
    variant.set_eval_model(cfg)
    model = variant.eval_model

    save_dir = Path(cfg.output_dir) / args.layer / "contour"
    save_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, X_train, y_train = extract(model, "Contour_train")
    val_dataset, X_val, y_val = extract(model, "Contour_val")

    assert not set(train_dataset.seeds) & set(val_dataset.seeds), (
        "train and val share seeds, so a val background was seen in training"
    )

    if 0 < args.max_features < X_train.shape[1]:
        rng = np.random.default_rng(args.seed)
        keep = np.sort(
            rng.choice(X_train.shape[1], size=args.max_features, replace=False)
        )
        X_train, X_val = X_train[:, keep], X_val[:, keep]
        print(f"subsampled to {X_train.shape[1]} features")

    search = fit_probe_with_cv(
        X_train, y_train, train_dataset.seeds, cv=args.cv_folds, n_jobs=args.n_jobs
    )
    probe = search.best_estimator_

    train_preds = probe.predict(X_train)
    train_correct = (train_preds == y_train).astype(int)
    train_acc = accuracy_score(y_train, train_preds)
    print(f"Train accuracy (refit on full train): {train_acc:.4f}")

    val_preds = probe.predict(X_val)
    val_correct = (val_preds == y_val).astype(int)
    val_acc = accuracy_score(y_val, val_preds)
    print(f"Val accuracy: {val_acc:.4f}  (chance 0.5)")

    conditions = pd.concat(
        [
            condition_table(train_dataset, train_correct, "train"),
            condition_table(val_dataset, val_correct, "val"),
        ],
        ignore_index=True,
    )
    conditions.insert(0, "layer", args.layer)
    conditions.insert(0, "model", cfg.name)
    conditions.insert(2, "probe", args.probe)

    condition_path = save_dir / f"{args.probe}_condition_accuracy.csv"
    conditions.to_csv(condition_path, index=False)

    print("\nval accuracy by condition:")
    print_marginals(conditions, "val")
    print("\ntrain accuracy by condition:")
    print_marginals(conditions, "train")

    results = {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "best_cv_acc": search.best_score_,
        "best_params": search.best_params_,
        "cv_results": search.cv_results_,
        "train_labels": y_train,
        "train_preds": train_preds,
        "train_correct": train_correct,
        "train_seeds": train_dataset.seeds,
        "train_factors": train_dataset.factors,
        "val_labels": y_val,
        "val_preds": val_preds,
        "val_correct": val_correct,
        "val_seeds": val_dataset.seeds,
        "val_factors": val_dataset.factors,
        "factor_columns": val_dataset.factor_columns,
        "condition_accuracy": conditions,
        "n_features": int(X_train.shape[1]),
    }

    out_path = save_dir / f"{args.probe}_results.pkl"
    with open(out_path, "wb") as f:
        pkl.dump(results, f)
    print(f"\nSaved results to {out_path}")
    print(f"Saved per-condition accuracy to {condition_path}")

    probe_path = save_dir / f"{args.probe}_probe.pkl"
    with open(probe_path, "wb") as f:
        pkl.dump(probe, f)
    print(f"Saved probe to {probe_path}")


if __name__ == "__main__":
    main()
