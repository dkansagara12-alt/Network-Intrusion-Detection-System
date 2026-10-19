"""
Exploratory data analysis and visualisation for the three datasets.

Writes to outputs/figures/eda/:
  <dataset>_class_distribution.png   rows per class after cleaning (log scale)
  <dataset>_top_features.png         most informative features (mutual information)
  <dataset>_correlation.png          correlation between those features
  <dataset>_pca_variance.png         cumulative explained variance (PCA)
  <dataset>_pca_projection.png       2-D PCA projection, normal vs attack
  <dataset>_pca_by_class.png         multi-class only: one panel per class
and outputs/eda_summary.csv (rows before/after cleaning, features, classes).

Statistics that need a model (mutual information, PCA) use a stratified
sample of at most --per-class rows per class, so rare classes are visible.

Usage:  python scripts/eda.py [--datasets ...] [--per-class 3000]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config  # noqa: E402
import plot_style as ps  # noqa: E402

TITLES = {"nsl_kdd": "NSL-KDD", "cic_ddos2019": "CIC-DDoS2019", "unsw_nb15": "UNSW-NB15"}
FIG_DIR = os.path.join(config.PROJECT_ROOT, "outputs", "figures", "eda")


def stratified_sample(y, per_class, rng):
    picked = [rng.choice(np.flatnonzero(y == c), min(per_class, (y == c).sum()), replace=False)
              for c in np.unique(y)]
    return np.sort(np.concatenate(picked))


def class_distribution(name, counts, class_names):
    order = np.argsort(counts)
    fig, ax = plt.subplots(figsize=(7, 0.36 * len(counts) + 1.4))
    ax.barh(np.array(class_names)[order], counts[order], height=0.6, color=ps.SERIES[0])
    ax.set_xscale("log")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(right=counts.max() * 6)
    for i, v in enumerate(counts[order]):
        ax.text(v * 1.15, i, f"{v:,}", va="center", fontsize=8.5, color=ps.INK_2)
    ax.set_title(f"{TITLES[name]}: rows per class after cleaning (log scale)")
    ax.set_xlabel("Rows")
    ps.save(fig, os.path.join(FIG_DIR, f"{name}_class_distribution.png"))


def top_features(name, X_num, y, n=12):
    mi = mutual_info_classif(X_num.values, y, random_state=0)
    top = pd.Series(mi, index=X_num.columns).sort_values(ascending=False).head(n)
    fig, ax = plt.subplots(figsize=(7, 0.34 * n + 1.3))
    ax.barh(top.index[::-1], top.values[::-1], height=0.6, color=ps.SERIES[0])
    ax.grid(axis="y", visible=False)
    for i, v in enumerate(top.values[::-1]):
        ax.text(v + top.max() * 0.01, i, f"{v:.2f}", va="center", fontsize=8.5, color=ps.INK_2)
    ax.set_xlim(right=top.max() * 1.12)
    ax.set_title(f"{TITLES[name]}: most informative features")
    ax.set_xlabel("Mutual information with the class label (nats)")
    ps.save(fig, os.path.join(FIG_DIR, f"{name}_top_features.png"))
    return list(top.index)


def correlation(name, X_num, features):
    corr = X_num[features].corr().fillna(0)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(corr.values, cmap=ps.DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(features)), features, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(features)), features, fontsize=8)
    ax.grid(False)
    for i in range(len(features)):
        for j in range(len(features)):
            v = corr.values[i, j]
            label = f"{v:.1f}".replace("-0.0", "0.0")
            ax.text(j, i, label, ha="center", va="center", fontsize=6.5,
                    color="white" if abs(v) > 0.6 else ps.INK_2)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Pearson correlation")
    ax.set_title(f"{TITLES[name]}: correlation between the top features")
    ps.save(fig, os.path.join(FIG_DIR, f"{name}_correlation.png"))


def pca_figures(name, X_all, y, class_names):
    Z = StandardScaler().fit_transform(X_all.values)
    pca = PCA(random_state=0).fit(Z)
    cum = np.cumsum(pca.explained_variance_ratio_)
    k90, k95 = int(np.searchsorted(cum, 0.90) + 1), int(np.searchsorted(cum, 0.95) + 1)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(np.arange(1, len(cum) + 1), cum, color=ps.SERIES[0])
    for k, level in [(k90, 0.90), (k95, 0.95)]:
        ax.plot([k], [cum[k - 1]], "o", ms=7, color=ps.SERIES[0], mec=ps.SURFACE, mew=2)
        ax.annotate(f"{level:.0%} with {k} components", (k, cum[k - 1]), xytext=(10, -14),
                    textcoords="offset points", fontsize=8.5, color=ps.INK_2)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(f"Number of principal components (of {X_all.shape[1]} features)")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title(f"{TITLES[name]}: PCA explained variance")
    ps.save(fig, os.path.join(FIG_DIR, f"{name}_pca_variance.png"))

    P = pca.transform(Z)[:, :2]
    # clip the view to the central 99% of points per axis so a few extreme
    # flows don't squeeze everything else into a corner
    lo, hi = np.percentile(P, 0.5, axis=0), np.percentile(P, 99.5, axis=0)
    pad = (hi - lo) * 0.05
    xlim, ylim = (lo[0] - pad[0], hi[0] + pad[0]), (lo[1] - pad[1], hi[1] + pad[1])
    pc = f"PC1 ({pca.explained_variance_ratio_[0]:.0%})", f"PC2 ({pca.explained_variance_ratio_[1]:.0%})"
    # equal numbers of normal and attack points, so neither hides the other
    rng = np.random.default_rng(1)
    normal_idx, attack_idx = np.flatnonzero(y == 0), np.flatnonzero(y != 0)
    n = min(len(normal_idx), len(attack_idx))
    normal_idx, attack_idx = rng.choice(normal_idx, n, False), rng.choice(attack_idx, n, False)
    fig, ax = plt.subplots(figsize=(6.4, 5))
    ax.scatter(*P[attack_idx].T, s=7, alpha=0.45, color=ps.SERIES[1], lw=0, label="Attack (all types)")
    ax.scatter(*P[normal_idx].T, s=7, alpha=0.45, color=ps.SERIES[0], lw=0,
               label=class_names[0].capitalize())
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel(pc[0]); ax.set_ylabel(pc[1])
    ax.legend(loc="upper right", markerscale=2.5)
    ax.set_title(f"{TITLES[name]}: 2-D PCA projection ({n:,} normal + {n:,} attack flows)")
    ps.save(fig, os.path.join(FIG_DIR, f"{name}_pca_projection.png"))

    if len(class_names) > 2:  # small multiples: each class over the rest in grey
        cols = 4
        rows = int(np.ceil(len(class_names) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.7 * rows), sharex=True, sharey=True)
        for c, ax in zip(range(len(class_names)), axes.flat):
            ax.scatter(*P[y != c].T, s=3, color=ps.NEUTRAL, alpha=0.35, lw=0)
            ax.scatter(*P[y == c].T, s=6, color=ps.SERIES[0], alpha=0.7, lw=0)
            ax.set_title(f"{class_names[c]} ({(y == c).sum():,})", fontsize=9.5)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)
            ax.tick_params(labelsize=7)
        for ax in list(axes.flat)[len(class_names):]:
            ax.set_visible(False)
        fig.suptitle(f"{TITLES[name]}: PCA projection per class (blue = class, grey = all others)",
                     x=0.01, ha="left", fontsize=11, fontweight="semibold")
        fig.tight_layout()
        ps.save(fig, os.path.join(FIG_DIR, f"{name}_pca_by_class.png"))
    return k90, k95


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=config.DATASETS)
    parser.add_argument("--per-class", type=int, default=3000)
    args = parser.parse_args()
    ps.apply()
    os.makedirs(FIG_DIR, exist_ok=True)
    summary_path = os.path.join(config.PROJECT_ROOT, "outputs", "eda_summary.csv")
    summary = (pd.read_csv(summary_path, index_col="Dataset").to_dict("index")
               if os.path.exists(summary_path) else {})

    for name in args.datasets:
        config.set_dataset(name)
        from src.data_preprocessing import load_clean_dataset
        clean = load_clean_dataset()
        X, y, class_names = clean["X"], np.asarray(clean["y"]), clean["class_names"]
        onehot = [c for cat in clean.get("categorical_columns", []) for c in X.columns
                  if c.startswith(cat + "_")]
        numeric = [c for c in X.columns if c not in onehot]

        counts = np.bincount(y, minlength=len(class_names))
        class_distribution(name, counts, class_names)

        rng = np.random.default_rng(0)
        idx = stratified_sample(y, args.per_class, rng)
        X_s = X.iloc[idx].copy()
        X_s = X_s.fillna(X_s.median()).loc[:, X_s.nunique() > 1]
        X_num = X_s[[c for c in numeric if c in X_s.columns]]
        features = top_features(name, X_num, y[idx])
        correlation(name, X_num, features)
        k90, k95 = pca_figures(name, X_s, y[idx], class_names)

        meta = clean.get("meta", {})
        summary[TITLES[name]] = {
            "Task": config.TASK,
            "Rows (raw)": meta.get("rows_raw"),
            "Duplicates removed": meta.get("rows_raw", 0) - meta.get("rows_clean", 0),
            "Rows (clean)": len(X),
            "Numerical features": len(numeric),
            "Categorical features": len(clean.get("categorical_columns", [])),
            "One-hot columns": len(onehot),
            "Classes": len(class_names),
            "Normal share": round(float((y == 0).mean()), 4),
            "Rarest class": f"{class_names[int(counts.argmin())]} ({int(counts.min()):,})",
            "PCA comps for 90% / 95% var.": f"{k90} / {k95}",
        }
        pd.DataFrame.from_dict(summary, orient="index").rename_axis("Dataset").to_csv(summary_path)
        print(f"[eda] {name} done; summary in {summary_path}")


if __name__ == "__main__":
    main()
