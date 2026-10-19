"""
Conventional ML baselines for comparison against the FT-Transformer:
  - Logistic Regression
  - Random Forest
  - XGBoost

Each is trained on the same preprocessed splits produced by data_preprocessing.py,
with the same class weights as the FT-Transformer (train.class_weights), and
evaluated by the same code, so comparisons are apples-to-apples. All three
handle both the binary and the multi-class task.
"""

import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from . import config
from .evaluate import evaluate_predictions, summary_line
from .train import class_weights


def _get_models(seed: int, multiclass: bool):
    return {
        # no n_jobs: lbfgs fits all classes in one job, and n_jobs=-1 only
        # spawns an idle pool of worker processes
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=None, n_jobs=-1, random_state=seed
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            device=config.DEVICE.type,
            eval_metric="mlogloss" if multiclass else "logloss",
            n_jobs=-1, random_state=seed
        ),
    }


def run_baselines(data: dict, seed: int = config.RANDOM_STATE) -> dict:
    """
    Trains each baseline model on the training split and evaluates on the
    held-out test split (binary: decision threshold tuned on the validation
    split, as for the FT-Transformer). Returns {model_name: metrics_dict}.
    """
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    class_names = data["class_names"]

    # per-row weights from the shared per-class weights; for "balanced" this
    # equals class_weight="balanced" in sklearn
    sample_weight = class_weights(y_train, len(class_names))[y_train]

    results = {}
    for name, model in _get_models(seed, multiclass=len(class_names) > 2).items():
        print(f"[baseline] Training {name}...")
        start = time.time()
        model.fit(X_train, y_train, sample_weight=sample_weight)
        train_time = time.time() - start
        if isinstance(model, XGBClassifier):
            # test data is a CPU array; predict on CPU instead of copying it to the GPU
            model.set_params(device="cpu")

        metrics, _ = evaluate_predictions(y_val, model.predict_proba(X_val),
                                          y_test, model.predict_proba(X_test), class_names)
        metrics["train_time_sec"] = round(train_time, 2)
        results[name] = metrics
        print(f"[baseline] {name}: {summary_line(metrics)} ({train_time:.1f}s)")

    return results


if __name__ == "__main__":
    from .data_preprocessing import load_dataset
    data = load_dataset()
    results = run_baselines(data)
    for name, metrics in results.items():
        print(name, metrics)
