"""
Charts of the 5-seed results for the report / webpage.

Reads outputs/<dataset>/per_seed_metrics.csv and all_metrics.json for the
three default runs and writes to outputs/figures/results/:
  main_metric.png         F1 (NSL-KDD, binary) / macro F1 (multi-class), mean ± std,
                          with IDS-MTran's reported value as a reference
  pr_auc.png              PR-AUC (macro for multi-class), mean ± std
  training_time.png       mean training time per model (log scale)
  <dataset>_per_class_f1.png   multi-class: F1 of every class for every model

Means and spreads are over seeds 42-46 (each seed = a different split and
initialisation); whiskers are ±1 sample standard deviation.

Usage:  python scripts/plot_results.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config  # noqa: E402
import plot_style as ps  # noqa: E402

TITLES = {"nsl_kdd": "NSL-KDD (binary)", "cic_ddos2019": "CIC-DDoS2019 (12 classes)",
          "unsw_nb15": "UNSW-NB15 (10 classes)"}
MODELS = list(ps.MODEL_COLORS)
FIG_DIR = os.path.join(config.PROJECT_ROOT, "outputs", "figures", "results")


def load(dataset):
    out = os.path.join(config.PROJECT_ROOT, "outputs", dataset)
    per_seed = pd.read_csv(os.path.join(out, "per_seed_metrics.csv"))
    per_seed["Model"] = per_seed["Model"].str.replace(" (this work)", "", regex=False)
    with open(os.path.join(out, "all_metrics.json")) as f:
        metrics = json.load(f)
    return per_seed, metrics


def column(per_seed, *candidates):
    return next(c for c in candidates if c in per_seed.columns)


def dot_plot(data, metric_cols, ylabel, title, filename, reference=None):
    """One panel per dataset: each model's mean ± std, IDS-MTran as a grey reference."""
    fig, axes = plt.subplots(1, len(data), figsize=(4.1 * len(data), 3.9))
    for ax, (dataset, (per_seed, metrics)) in zip(axes, data.items()):
        col = column(per_seed, *metric_cols)
        stats = per_seed.groupby("Model")[col].agg(["mean", "std"]).reindex(MODELS)
        ref = metrics["IDS-MTran (reported)"].get(reference) if reference else None
        values = list(stats["mean"] - stats["std"]) + list(stats["mean"] + stats["std"])
        values += [ref] if ref is not None else []
        # close values (NSL-KDD) need 4 decimals to tell them apart
        fmt = "{:.4f}" if max(values) - min(values) < 0.1 else "{:.3f}"
        x = np.arange(len(MODELS))
        for i, model in enumerate(MODELS):
            m, s = stats.loc[model]
            ax.errorbar(i, m, yerr=s, fmt="o", ms=8, color=ps.MODEL_COLORS[model],
                        mec=ps.SURFACE, mew=2, elinewidth=2, capsize=4, zorder=3)
            ax.annotate(fmt.format(m), (i, m + s), xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=8, color=ps.INK_2)
        if ref is not None:
            ax.axhline(ref, color=ps.MUTED, lw=1.2, zorder=1)
            ax.annotate(f"IDS-MTran reported: {fmt.format(ref)}", (-0.45, ref),
                        xytext=(0, -11), textcoords="offset points", ha="left",
                        fontsize=8, color=ps.INK_2)
        span = max(values) - min(values)
        ax.set_ylim(min(values) - 0.12 * span, max(values) + 0.18 * span)
        ax.set_xlim(-0.5, len(MODELS) - 0.5)
        ax.set_xticks(x, ["FT-Trans.", "Log. Reg.", "Rand. Forest", "XGBoost"], fontsize=8.5)
        ax.grid(axis="x", visible=False)
        ax.set_title(TITLES[dataset], fontsize=10)
        ax.set_ylabel(ylabel(dataset), fontsize=9)
    handles = [Line2D([], [], marker="o", ls="", ms=8, color=c, label=m)
               for m, c in ps.MODEL_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(title, x=0.01, ha="left", fontsize=11.5, fontweight="semibold")
    fig.tight_layout()
    ps.save(fig, os.path.join(FIG_DIR, filename))


def training_time(data):
    fig, axes = plt.subplots(1, len(data), figsize=(4.1 * len(data), 2.9), sharey=True)
    for ax, (dataset, (per_seed, _)) in zip(axes, data.items()):
        t = per_seed.groupby("Model")["Train time (s)"].mean().reindex(MODELS)
        y = np.arange(len(MODELS))[::-1]
        ax.barh(y, t.values, height=0.55, color=[ps.MODEL_COLORS[m] for m in MODELS])
        ax.set_xlim(0, t.max() * 1.25)
        for yi, v in zip(y, t.values):
            ax.text(v + t.max() * 0.02, yi, f"{v:,.0f} s" if v >= 10 else f"{v:.1f} s",
                    va="center", fontsize=8.5, color=ps.INK_2)
        ax.set_yticks(y, MODELS, fontsize=8.5)
        ax.grid(axis="y", visible=False)
        ax.set_title(TITLES[dataset], fontsize=10)
        ax.set_xlabel("Mean training time per seed (seconds)", fontsize=8.5)
    fig.suptitle("Training cost", x=0.01, ha="left", fontsize=11.5, fontweight="semibold")
    fig.tight_layout()
    ps.save(fig, os.path.join(FIG_DIR, "training_time.png"))


def per_class_heatmap(dataset, metrics):
    summary = metrics["summary"]
    models = {m: summary[m + (" (this work)" if m == "FT-Transformer" else "")] for m in MODELS}
    classes = list(models["FT-Transformer"]["per_class_f1"])
    support = metrics["per_seed"][str(metrics["seeds"][0])]["FT-Transformer (this work)"]["per_class_support"]
    F = np.array([[models[m]["per_class_f1"][c]["mean"] for m in MODELS] for c in classes])
    fig, ax = plt.subplots(figsize=(6.6, 0.42 * len(classes) + 1.6))
    im = ax.imshow(F, cmap=ps.BLUE_RAMP, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(MODELS)), MODELS, fontsize=8.5)
    ax.set_yticks(range(len(classes)), [f"{c} ({support[c]:,})" for c in classes], fontsize=8.5)
    ax.xaxis.tick_top()
    ax.grid(False)
    for i in range(len(classes)):
        for j in range(len(MODELS)):
            ax.text(j, i, f"{F[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if F[i, j] > 0.55 else ps.INK)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label="F1 (mean of 5 seeds)")
    ax.set_title(f"{TITLES[dataset]}: F1 per class (test rows in brackets)", pad=28)
    ps.save(fig, os.path.join(FIG_DIR, f"{dataset}_per_class_f1.png"))


def main():
    ps.apply()
    os.makedirs(FIG_DIR, exist_ok=True)
    data = {d: load(d) for d in ["nsl_kdd", "cic_ddos2019", "unsw_nb15"]}

    def f1_label(d):
        return "F1 (attack class)" if d == "nsl_kdd" else "Macro F1"

    def pr_label(d):
        return "PR-AUC" if d == "nsl_kdd" else "Macro PR-AUC"

    dot_plot(data, ["F1-Score", "F1 (macro)"], f1_label,
             "Detection quality: F1 (mean ± std over 5 seeds)", "main_metric.png", reference="f1")
    dot_plot(data, ["PR-AUC", "PR-AUC (macro)"], pr_label,
             "Threshold-free quality: PR-AUC (mean ± std over 5 seeds)", "pr_auc.png")
    training_time(data)
    for dataset in ["cic_ddos2019", "unsw_nb15"]:
        per_class_heatmap(dataset, data[dataset][1])


if __name__ == "__main__":
    main()
