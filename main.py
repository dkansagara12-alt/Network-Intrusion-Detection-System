"""
End-to-end pipeline for the "Intelligent NIDS Using FT-Transformer" project.

Usage:
    python main.py --dataset nsl_kdd                        # binary
    python main.py --dataset unsw_nb15                      # multi-class (10 classes)
    python main.py --dataset cic_ddos2019                   # multi-class (12 classes)
    python main.py --dataset unsw_nb15 --task binary        # -> outputs/unsw_nb15_binary/
    python main.py --dataset nsl_kdd --nsl-split official   # KDDTrain+ -> KDDTest+
    python main.py --dataset nsl_kdd --num-seeds 1          # quick single run
    python main.py --dataset unsw_nb15 --resume             # continue an interrupted run

Each dataset's task (config.DATASET_TASKS) matches what IDS-MTran reports.
Every run is repeated for config.NUM_SEEDS seeds (42, 43, ...). Each seed
changes the train/val/test split and the models' random initialisation; the
comparison table reports mean ± std over the seeds.

Outputs (saved to outputs/<dataset>/):
    - model_comparison.csv                  (mean ± std: FT-Transformer vs baselines vs IDS-MTran)
    - per_seed_metrics.csv                  (raw numbers for every seed and model)
    - per_class_f1.csv                      (multi-class only: F1 of every class)
    - all_metrics.json
    - ft_transformer_confusion_matrix.png   (first seed)
    - ft_transformer_roc_curve.png          (binary only, first seed)
    - scaler.joblib, feature_names.joblib, imputer_medians.joblib        (first seed)
Checkpoints (saved to checkpoints/<dataset>/):
    - ft_transformer_seed<seed>.pt
"""

import argparse
import json
import os
import time

from src import config


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=config.DATASETS, default=config.DATASET_NAME)
    parser.add_argument("--nsl-split", choices=["random", "official"],
                        default=config.NSL_KDD_SPLIT,
                        help="NSL-KDD protocol: random split of train+test (as in "
                             "IDS-MTran) or the official KDDTrain+/KDDTest+ split")
    parser.add_argument("--num-seeds", type=int, default=config.NUM_SEEDS,
                        help="number of seeds to repeat the experiment with")
    parser.add_argument("--resume", action="store_true",
                        help="skip seeds already finished by an interrupted run "
                             "(read from outputs/<dataset>/seed_progress.json)")
    parser.add_argument("--task", choices=["binary", "multiclass"],
                        help="override the dataset's default task (config.DATASET_TASKS); "
                             "results then go to outputs/<dataset>_<task>/")
    args = parser.parse_args()
    config.NSL_KDD_SPLIT = args.nsl_split
    task = args.task or config.DATASET_TASKS[args.dataset]
    run_name = args.dataset
    if args.dataset == "nsl_kdd" and args.nsl_split == "official":
        run_name += "_official"
    if task != config.DATASET_TASKS[args.dataset]:
        run_name += f"_{task}"
    config.set_dataset(args.dataset, run_name=run_name, task=task)
    seeds = [config.RANDOM_STATE + i for i in range(args.num_seeds)]

    # imported after set_dataset so every module sees the chosen dataset
    from src.data_preprocessing import load_clean_dataset, split_and_scale
    from src.train import train_model
    from src.evaluate import evaluate_ft_transformer, build_comparison_table, save_all_metrics
    from src.baseline_models import run_baselines

    # results are saved after every seed, so an interrupted run can --resume
    progress_path = os.path.join(config.OUTPUT_DIR, "seed_progress.json")
    done = {}  # seed -> {model name: metrics}
    if args.resume and os.path.exists(progress_path):
        with open(progress_path) as f:
            done = {int(seed): run for seed, run in json.load(f).items()}
        print(f"Resuming: seeds {sorted(done)} already finished")

    if any(seed not in done for seed in seeds):
        print("=" * 70)
        print(f"Loading and cleaning {config.DATASET_NAME}")
        print("=" * 70)
        clean = load_clean_dataset()

    for i, seed in enumerate(seeds):
        if seed in done:
            continue
        print("\n" + "#" * 70)
        print(f"# SEED {seed} ({i + 1}/{len(seeds)})")
        print("#" * 70)
        first = i == 0

        print(f"\nSTEP 1: Train/val/test split and scaling (seed {seed})")
        data = split_and_scale(**clean, seed=seed, save_intermediate=first)

        print(f"\nSTEP 2: Training FT-Transformer (seed {seed})")
        start = time.time()
        model, history = train_model(data, seed=seed)
        ft_train_time = time.time() - start

        print(f"\nSTEP 3: Evaluating FT-Transformer on test set (seed {seed})")
        ft_metrics = evaluate_ft_transformer(model, data, plot=first)
        ft_metrics["train_time_sec"] = round(ft_train_time, 2)

        print(f"\nSTEP 4: Training & evaluating baseline ML models (seed {seed})")
        baseline_results = run_baselines(data, seed=seed)

        done[seed] = {"FT-Transformer (this work)": ft_metrics, **baseline_results}
        with open(progress_path, "w") as f:
            json.dump(done, f, indent=2, default=str)

    runs = [done[seed] for seed in seeds]  # one {model name: metrics} dict per seed

    print("\n" + "=" * 70)
    print(f"STEP 5: Building final comparison table ({len(seeds)} seed(s))")
    print("=" * 70)
    build_comparison_table(runs, seeds)
    save_all_metrics(runs, seeds)

    print(f"\nDone. Check {config.OUTPUT_DIR} for plots, tables, and metrics.")


if __name__ == "__main__":
    main()
