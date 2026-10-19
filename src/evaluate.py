"""
Evaluation utilities: metric computation, decision-threshold tuning,
confusion matrix plotting, ROC curve plotting, and final comparison table
generation (FT-Transformer vs baselines vs base-paper reported IDS-MTran
numbers). Every model is evaluated by the same code.

Binary task (NSL-KDD): the decision threshold is the one that maximises F1
on the validation split, and that threshold is then applied to the test
split. The default 0.5 threshold is a poor fit for class-weighted models on
imbalanced data, where it trades away a lot of precision for recall.

Multi-class task (CIC-DDoS2019, UNSW-NB15): the prediction is the most
probable class. Precision, recall and F1 are macro averages (every class
counts equally, so rare attack types matter as much as common ones), plus a
support-weighted F1 and per-class F1.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # file-only backend; the default Tk backend can crash on cleanup
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, precision_recall_curve,
    confusion_matrix, roc_curve, classification_report
)
from sklearn.preprocessing import label_binarize

from . import config
from .train import autocast


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Binary metrics; y_prob is P(attack)."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
            # PR-AUC (average precision): threshold-independent and, unlike
            # ROC-AUC, sensitive to false positives on the rare attack class
            metrics["pr_auc"] = average_precision_score(y_true, y_prob)
        except ValueError:
            metrics["roc_auc"] = metrics["pr_auc"] = float("nan")
    return metrics


def tune_threshold(y_val, p_val) -> float:
    """Decision threshold on P(attack) that maximises F1 on the validation split."""
    precision, recall, thresholds = precision_recall_curve(y_val, p_val)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    # the last precision/recall pair has no matching threshold
    return float(thresholds[np.argmax(f1[:-1])])


def _evaluate_binary(y_val, p_val, y_test, p_test):
    """
    Test metrics at the validation-tuned threshold. Also records the tuned
    threshold and the F1 the default 0.5 threshold would have given, so the
    report can show both.
    """
    threshold = tune_threshold(y_val, p_val)
    preds = (p_test >= threshold).astype(int)
    metrics = compute_metrics(y_test, preds, p_test)
    metrics["threshold"] = threshold
    metrics["f1_at_0.5"] = f1_score(y_test, (p_test >= 0.5).astype(int), zero_division=0)
    return metrics, preds


def _evaluate_multiclass(y_test, P_test, class_names):
    """Metrics for the most-probable-class prediction; P_test is (rows, classes)."""
    labels = np.arange(len(class_names))
    P_test = P_test / P_test.sum(axis=1, keepdims=True)  # guard float32 rounding
    preds = P_test.argmax(axis=1)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, labels=labels, average="macro", zero_division=0),
        "recall": recall_score(y_test, preds, labels=labels, average="macro", zero_division=0),
        "f1": f1_score(y_test, preds, labels=labels, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_test, preds, labels=labels, average="weighted", zero_division=0),
        # one-vs-rest per class, then averaged over classes
        "roc_auc": roc_auc_score(y_test, P_test, multi_class="ovr", average="macro", labels=labels),
        "pr_auc": average_precision_score(label_binarize(y_test, classes=labels), P_test,
                                          average="macro"),
        "per_class_f1": dict(zip(class_names, f1_score(
            y_test, preds, labels=labels, average=None, zero_division=0).tolist())),
        "per_class_support": dict(zip(class_names, np.bincount(
            y_test, minlength=len(class_names)).tolist())),
    }
    return metrics, preds


def evaluate_predictions(y_val, P_val, y_test, P_test, class_names):
    """
    Test metrics for any model, given its class-probability matrices
    (rows x classes) on the validation and test splits. Returns
    (metrics, test predictions).
    """
    if len(class_names) == 2:
        return _evaluate_binary(y_val, P_val[:, 1], y_test, P_test[:, 1])
    return _evaluate_multiclass(y_test, P_test, class_names)


def summary_line(metrics: dict) -> str:
    """One-line summary of a model's test metrics for the console."""
    head = f"threshold={metrics['threshold']:.3f} " if "threshold" in metrics else ""
    f1_name = "f1" if "threshold" in metrics else "macro_f1"
    return (f"{head}acc={metrics['accuracy']:.4f} {f1_name}={metrics['f1']:.4f} "
            f"pr_auc={metrics['pr_auc']:.4f} roc_auc={metrics['roc_auc']:.4f}")


@torch.no_grad()
def predict_proba(model, X: np.ndarray) -> np.ndarray:
    """Class probabilities (rows x classes), computed in batches so large test sets fit."""
    device = config.DEVICE
    model.eval()
    probs = []
    for start in range(0, len(X), config.EVAL_BATCH_SIZE):
        batch = torch.from_numpy(X[start:start + config.EVAL_BATCH_SIZE]).float().to(device)
        with autocast():
            logits = model(batch)
        probs.append(torch.softmax(logits.float(), dim=1).cpu())
    return torch.cat(probs).numpy()


def evaluate_ft_transformer(model, data: dict, plot: bool = True) -> dict:
    y_test, class_names = data["y_test"], data["class_names"]
    P_test = predict_proba(model, data["X_test"])
    metrics, preds = evaluate_predictions(data["y_val"], predict_proba(model, data["X_val"]),
                                          y_test, P_test, class_names)

    binary = len(class_names) == 2
    how = (f"threshold {metrics['threshold']:.3f}, tuned on validation" if binary
           else "most probable class")
    report = classification_report(y_test, preds, labels=np.arange(len(class_names)),
                                   target_names=class_names, digits=4, zero_division=0)
    print(f"[eval] FT-Transformer classification report ({how}):\n", report)

    if plot:
        cm = confusion_matrix(y_test, preds, labels=np.arange(len(class_names)))
        _plot_confusion_matrix(cm, class_names, f"FT-Transformer ({how})",
                               "ft_transformer_confusion_matrix.png")
        if binary:
            _plot_roc_curve(y_test, P_test[:, 1], "FT-Transformer", "ft_transformer_roc_curve.png")

    return metrics


def _plot_confusion_matrix(cm, class_names, title, filename):
    if len(class_names) == 2:
        values, fmt, size = cm, "d", (5, 4)
    else:
        # row-normalised (each row = recall of that class); counts would be
        # unreadable next to classes with tens of rows
        values = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        fmt, size = ".2f", (0.7 * len(class_names) + 3, 0.6 * len(class_names) + 2.5)
    plt.figure(figsize=size)
    sns.heatmap(values, annot=True, fmt=fmt, cmap="Blues", annot_kws={"fontsize": 8},
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {title}", fontsize=10)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    path = os.path.join(config.OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[eval] Saved confusion matrix to {path}")


def _plot_roc_curve(y_true, y_prob, title, filename):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {title}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(config.OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[eval] Saved ROC curve to {path}")


def _table_columns() -> dict:
    """Metric key -> column name in the comparison tables, for config.TASK."""
    if config.TASK == "binary":
        return {
            "threshold": "Threshold", "accuracy": "Accuracy", "precision": "Precision",
            "recall": "Recall", "f1": "F1-Score", "f1_at_0.5": "F1 @0.5",
            "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "train_time_sec": "Train time (s)",
        }
    return {
        "accuracy": "Accuracy", "precision": "Precision (macro)", "recall": "Recall (macro)",
        "f1": "F1 (macro)", "f1_weighted": "F1 (weighted)", "roc_auc": "ROC-AUC (macro OvR)",
        "pr_auc": "PR-AUC (macro)", "train_time_sec": "Train time (s)",
    }


def _mean_std(values, digits: int = 4) -> str:
    values = pd.Series(values, dtype=float)
    if len(values) == 1:
        return f"{values.iloc[0]:.{digits}f}"
    return f"{values.mean():.{digits}f} ± {values.std():.{digits}f}"


def build_comparison_table(runs: list, seeds: list) -> pd.DataFrame:
    """
    Combines every model's per-seed results (runs[i] = {model: metrics} for
    seeds[i]) with the base paper's reported IDS-MTran numbers. Writes:
      per_seed_metrics.csv - one row per (seed, model), raw numbers
      model_comparison.csv - one row per model, "mean ± std" over the seeds
                             (sample standard deviation), ready for the report
      per_class_f1.csv     - multi-class only: F1 of every class per model
    """
    columns = _table_columns()
    per_seed = pd.DataFrame([
        {"Seed": seed, "Model": model, **{col: metrics.get(key) for key, col in columns.items()}}
        for seed, run in zip(seeds, runs) for model, metrics in run.items()
    ])
    per_seed_path = os.path.join(config.OUTPUT_DIR, "per_seed_metrics.csv")
    per_seed.to_csv(per_seed_path, index=False)

    rows = [
        {"Model": model, "Task": config.TASK, "Seeds": len(group),
         **{col: _mean_std(group[col], 1 if col == "Train time (s)" else 4)
            for col in columns.values()}}
        for model, group in per_seed.groupby("Model", sort=False)
    ]

    # reported numbers for the dataset being run (see config.py)
    base = config.IDS_MTRAN_REPORTED_METRICS[config.DATASET_NAME]
    rows.append({
        "Model": "IDS-MTran (reported, base paper)", "Task": base["task"], "Seeds": "",
        **{col: f"{base[key]:.4f}" if base.get(key) is not None else ""
           for key, col in columns.items()},
    })

    df = pd.DataFrame(rows)
    path = os.path.join(config.OUTPUT_DIR, "model_comparison.csv")
    # utf-8-sig so Excel shows "±" correctly
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[eval] Saved per-seed metrics to {per_seed_path}")
    print(f"[eval] Saved comparison table (mean ± std over seeds {seeds}) to {path}")
    print(df.to_string(index=False))

    if config.TASK == "multiclass":
        _build_per_class_table(runs)
    return df


def _build_per_class_table(runs: list):
    """F1 of every class (rows) for every model (columns), mean ± std over seeds."""
    models = list(runs[0])
    first = runs[0][models[0]]
    table = pd.DataFrame({
        "Class": list(first["per_class_f1"]),
        "Test rows": list(first["per_class_support"].values()),
        **{model: [_mean_std([run[model]["per_class_f1"][cls] for run in runs])
                   for cls in first["per_class_f1"]]
           for model in models},
    })
    path = os.path.join(config.OUTPUT_DIR, "per_class_f1.csv")
    table.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[eval] Saved per-class F1 (mean ± std over seeds) to {path}")
    print(table.to_string(index=False))


def save_all_metrics(runs: list, seeds: list):
    def summarise(values):
        values = np.array(values, dtype=float)
        return {"mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}

    summary = {}
    for model, first in runs[0].items():
        summary[model] = {}
        for key, value in first.items():
            if isinstance(value, dict):  # per-class values
                summary[model][key] = {cls: summarise([run[model][key][cls] for run in runs])
                                       for cls in value}
            else:
                summary[model][key] = summarise([run[model][key] for run in runs])
    all_results = {
        "task": config.TASK,
        "seeds": seeds,
        "summary": summary,
        "per_seed": {str(seed): run for seed, run in zip(seeds, runs)},
        "IDS-MTran (reported)": config.IDS_MTRAN_REPORTED_METRICS[config.DATASET_NAME],
    }
    path = os.path.join(config.OUTPUT_DIR, "all_metrics.json")
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"[eval] Saved raw metrics JSON to {path}")
