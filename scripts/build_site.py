"""
Builds the offline project webpage docs/index.html from site/template.html.

The template holds the text; this script fills in everything that comes
from the experiments, so the page always matches the latest outputs/:
  <!--TABLE:comparison:<dataset>-->   mean ± std comparison table
  <!--TABLE:per_class:<dataset>-->    per-class F1 table (multi-class)
  <!--TABLE:eda_summary-->            dataset summary from scripts/eda.py
  <!--DEMO:<dataset>-->                output of predict.py on demo/<dataset>_samples.csv
  {{dataset|Model|Column}}            one cell of a comparison table ("0.9915 ± 0.0007")
  {{dataset|Model|Column|mean}}       just the mean of that cell
Figures from outputs/figures/ and the FT-Transformer confusion matrices are
copied to docs/figures/. The page has no external dependencies (works offline
and on GitHub Pages).

Usage:  python scripts/build_site.py
"""

import html
import os
import re
import shutil
import subprocess
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT, DOCS = os.path.join(ROOT, "outputs"), os.path.join(ROOT, "docs")
DATASETS = ["nsl_kdd", "cic_ddos2019", "unsw_nb15"]


def read_csv(*parts):
    return pd.read_csv(os.path.join(OUT, *parts), encoding="utf-8-sig", dtype=str).fillna("")


def table(df, caption=None, highlight_best=()):
    """HTML table; in highlight_best columns the highest mean is bold."""
    best = {}
    for col in highlight_best:
        if col in df.columns:
            means = pd.to_numeric(df[col].str.split(" ").str[0], errors="coerce")
            mask = ~df["Model"].str.contains("IDS-MTran") if "Model" in df.columns else True
            best[col] = means[mask].idxmax()
    rows = []
    for i, row in df.iterrows():
        cells = []
        for col, v in row.items():
            v = html.escape(str(v))
            if best.get(col) == i:
                v = f"<strong>{v}</strong>"
            cells.append(f"<td>{v}</td>")
        cls = ' class="reference"' if "IDS-MTran" in str(row.iloc[0]) else ""
        rows.append(f"<tr{cls}>{''.join(cells)}</tr>")
    head = "".join(f"<th>{html.escape(c)}</th>" for c in df.columns)
    cap = f"<caption>{caption}</caption>" if caption else ""
    return (f'<div class="table-wrap"><table>{cap}<thead><tr>{head}</tr></thead>'
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def comparison(dataset):
    df = read_csv(dataset, "model_comparison.csv").drop(columns=["Seeds"])
    df["Model"] = df["Model"].str.replace(" (this work)", "", regex=False)
    metric_cols = [c for c in df.columns if c not in ("Model", "Task", "Threshold", "Train time (s)")]
    return table(df, highlight_best=metric_cols)


def demo_output(dataset):
    sample = os.path.join("demo", f"{dataset}_samples.csv")
    run = subprocess.run([sys.executable, "predict.py", "--dataset", dataset, "--input", sample,
                          "--show", "8"], cwd=ROOT, capture_output=True, text=True,
                         env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if run.returncode != 0:
        raise RuntimeError(run.stderr)
    command = f"$ python predict.py --dataset {dataset} --input {sample}\n"
    return f"<pre><code>{html.escape(command + run.stdout.strip())}</code></pre>"


def cell(dataset, model, column, part=None):
    df = read_csv(dataset, "model_comparison.csv")
    row = df[df["Model"].str.startswith(model)]
    value = row.iloc[0][column]
    return value.split(" ")[0] if part == "mean" else value


def main():
    with open(os.path.join(ROOT, "site", "template.html"), encoding="utf-8") as f:
        page = f.read()

    for dataset in DATASETS:
        page = page.replace(f"<!--TABLE:comparison:{dataset}-->", comparison(dataset))
        if os.path.exists(os.path.join(OUT, dataset, "per_class_f1.csv")):
            per_class = read_csv(dataset, "per_class_f1.csv")
            per_class.columns = [c.replace(" (this work)", "") for c in per_class.columns]
            page = page.replace(f"<!--TABLE:per_class:{dataset}-->", table(per_class))
        if f"<!--DEMO:{dataset}-->" in page:
            page = page.replace(f"<!--DEMO:{dataset}-->", demo_output(dataset))
    page = page.replace("<!--TABLE:eda_summary-->", table(read_csv("eda_summary.csv")))
    page = re.sub(r"\{\{(\w+)\|([^|}]+)\|([^|}]+)(?:\|(mean))?\}\}",
                  lambda m: cell(*m.groups()), page)
    leftover = re.findall(r"<!--(?:TABLE|DEMO):[^>]*-->|\{\{[^}]*\}\}", page)
    if leftover:
        raise RuntimeError(f"unfilled placeholders: {leftover}")

    fig_dir = os.path.join(DOCS, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    for sub in ("eda", "results"):
        for name in os.listdir(os.path.join(OUT, "figures", sub)):
            shutil.copy(os.path.join(OUT, "figures", sub, name), os.path.join(fig_dir, name))
    for dataset in DATASETS:
        shutil.copy(os.path.join(OUT, dataset, "ft_transformer_confusion_matrix.png"),
                    os.path.join(fig_dir, f"{dataset}_confusion_matrix.png"))
    shutil.copy(os.path.join(ROOT, "site", "style.css"), os.path.join(DOCS, "style.css"))
    missing = [f for f in re.findall(r'src="figures/([^"]+)"', page)
               if not os.path.exists(os.path.join(fig_dir, f))]
    if missing:
        raise RuntimeError(f"figures referenced but not found: {missing}")

    with open(os.path.join(DOCS, "index.html"), "w", encoding="utf-8") as f:
        f.write(page)
    print(f"[site] wrote {os.path.join(DOCS, 'index.html')} "
          f"({len(os.listdir(fig_dir))} figures)")


if __name__ == "__main__":
    main()
