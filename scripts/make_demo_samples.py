"""
Builds the demo inputs in demo/ and the inference metadata predict.py needs.

For each dataset: loads and cleans the data, re-creates the seed-42 split
(identical to the training run's, so the saved scaler/feature list are
re-written unchanged - this is checked), picks up to N test flows per class,
and writes them in the dataset's own column format (one-hot columns turned
back into their category) with a true_label column.

Finally checks that predict.py's preprocessing of those raw rows reproduces
exactly the feature matrix used for evaluation.

Usage:  python scripts/make_demo_samples.py [--per-class 20] [--datasets ...]
"""

import argparse
import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config  # noqa: E402

DEMO_DIR = os.path.join(config.PROJECT_ROOT, "demo")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--datasets", nargs="+", default=config.DATASETS)
    args = parser.parse_args()
    os.makedirs(DEMO_DIR, exist_ok=True)

    for dataset in args.datasets:
        config.set_dataset(dataset)
        from src.data_preprocessing import load_clean_dataset, split_and_scale
        from src.inference import FlowClassifier

        clean = load_clean_dataset()
        saved = joblib.load(os.path.join(config.OUTPUT_DIR, "feature_names.joblib"))
        data = split_and_scale(**clean, seed=config.RANDOM_STATE, save_intermediate=False)
        if data["feature_names"] != saved:
            raise RuntimeError(f"{dataset}: features differ from the trained run; retrain first.")
        # same split as training -> re-saving writes identical artifacts + inference_meta
        split_and_scale(**clean, seed=config.RANDOM_STATE, save_intermediate=True)

        rng = np.random.default_rng(0)
        y_test = clean["y"][data["test_index"]]
        picked = []  # positions within the test split
        for c in range(len(data["class_names"])):
            members = np.flatnonzero(y_test == c)
            picked.extend(rng.choice(members, min(args.per_class, len(members)), replace=False))
        picked = np.sort(np.array(picked))

        rows = clean["X"].iloc[data["test_index"][picked]].copy()
        for cat in clean.get("categorical_columns", []):
            block = [c for c in rows.columns if c.startswith(cat + "_")]
            rows[cat] = rows[block].idxmax(axis=1).str[len(cat) + 1:]
            rows = rows.drop(columns=block)
        rows["true_label"] = [data["class_names"][i] for i in y_test[picked]]
        path = os.path.join(DEMO_DIR, f"{dataset}_samples.csv")
        rows.reset_index(drop=True).to_csv(path, index=False)
        print(f"[demo] {dataset}: wrote {len(rows)} flows "
              f"({rows['true_label'].nunique()} classes) to {path}")

        # predict.py must turn these raw rows into exactly the evaluated features
        encoded = FlowClassifier(dataset).encode(rows.drop(columns=["true_label"]))
        if not np.allclose(encoded, data["X_test"][picked], atol=1e-5):
            raise RuntimeError(f"{dataset}: inference preprocessing does not match training")
        print(f"[demo] {dataset}: inference preprocessing matches the evaluated test features")


if __name__ == "__main__":
    main()
