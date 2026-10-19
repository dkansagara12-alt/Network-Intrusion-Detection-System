"""
Demo: classify network flows with a trained FT-Transformer.

Usage:
    python predict.py --dataset nsl_kdd      --input demo/nsl_kdd_samples.csv
    python predict.py --dataset cic_ddos2019 --input demo/cic_ddos2019_samples.csv
    python predict.py --dataset unsw_nb15    --input demo/unsw_nb15_samples.csv --output predictions.csv

The input is a CSV (or Parquet) of flow records in the dataset's own column
format. The demo/ samples come from each dataset's held-out test split (seed
42), so the model never saw them during training; their "true_label" column
is only used to score the predictions.
"""

import argparse
import os

import pandas as pd

from src import config


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=config.DATASETS, required=True)
    parser.add_argument("--input", required=True, help="CSV or Parquet file of flow records")
    parser.add_argument("--seed", type=int, help="which trained seed to use (default: 42)")
    parser.add_argument("--output", help="optional CSV to write the predictions to")
    parser.add_argument("--show", type=int, default=15, help="rows to print (default 15)")
    args = parser.parse_args()

    from src.inference import FlowClassifier

    if os.path.splitext(args.input)[1].lower() in (".parquet", ".pq"):
        flows = pd.read_parquet(args.input)
    else:
        flows = pd.read_csv(args.input, low_memory=False)
    classifier = FlowClassifier(args.dataset, seed=args.seed)
    rule = (f"threshold {classifier.threshold:.3f} on P(attack)" if classifier.threshold is not None
            else "most probable class")
    print(f"Model: FT-Transformer, {args.dataset} ({config.TASK}, "
          f"{len(classifier.class_names)} classes), seed {classifier.seed}, decision: {rule}")
    print(f"Classifying {len(flows)} flow(s) from {args.input}\n")

    result = classifier.predict(flows)
    if "true_label" in flows.columns:
        result.insert(0, "true_label", flows["true_label"].values)
        result["correct"] = result["true_label"] == result["predicted"]

    pd.set_option("display.width", 120)
    print(result.head(args.show).to_string())
    if len(result) > args.show:
        print(f"... ({len(result) - args.show} more rows)")

    if "correct" in result:
        print(f"\nCorrect: {result['correct'].sum()}/{len(result)} "
              f"({result['correct'].mean():.1%})")
        per_class = result.groupby("true_label")["correct"].agg(["sum", "count"])
        per_class["recall"] = (per_class["sum"] / per_class["count"]).map("{:.0%}".format)
        print("\nPer true class:")
        print(per_class.rename(columns={"sum": "correct", "count": "flows"}).to_string())

    if args.output:
        result.to_csv(args.output)
        print(f"\nSaved predictions to {args.output}")


if __name__ == "__main__":
    main()
