"""
Central configuration for the NIDS FT-Transformer project.
Pick the dataset with `python main.py --dataset <name>` (or edit DATASET_NAME).
"""

import os
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Which dataset to run. This project targets the same three datasets as the
# base paper (IDS-MTran): "nsl_kdd", "cic_ddos2019", "unsw_nb15".
DATASETS = ["nsl_kdd", "cic_ddos2019", "unsw_nb15"]
DATASET_NAME = "nsl_kdd"

# Classification task per dataset, matching what IDS-MTran reports:
# binary (normal vs attack) for NSL-KDD, multi-class (normal + each attack
# type) for CIC-DDoS2019 and UNSW-NB15. Override with main.py --task.
DATASET_TASKS = {"nsl_kdd": "binary", "cic_ddos2019": "multiclass", "unsw_nb15": "multiclass"}

DATA_ROOT = os.path.join(PROJECT_ROOT, "data")


def set_dataset(name: str, run_name: str = None, task: str = None):
    """
    Point DATA_DIR / OUTPUT_DIR / CHECKPOINT_DIR at one dataset's folders and
    set TASK (default: DATASET_TASKS[name]). run_name (default: the dataset
    name) names the outputs/ and checkpoints/ subfolders, so variants such as
    the official NSL-KDD split don't overwrite each other.
    """
    global DATASET_NAME, DATA_DIR, OUTPUT_DIR, CHECKPOINT_DIR, TASK
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {DATASETS}.")
    DATASET_NAME = name
    TASK = task or DATASET_TASKS[name]
    # Each dataset lives in its own subfolder under data/, e.g. data/nsl_kdd/
    DATA_DIR = os.path.join(DATA_ROOT, name)
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", run_name or name)
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", run_name or name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


set_dataset(DATASET_NAME)

# ---------------------------------------------------------------------------
# Task setup
# ---------------------------------------------------------------------------
LABEL_COLUMN_CANDIDATES = ["Label", " Label", "label", "Attack", "class"]

TEST_SIZE = 0.15
VAL_SIZE = 0.15           # taken out of the remaining training data
RANDOM_STATE = 42         # base seed: seeds the splits AND model training
NUM_SEEDS = 5             # each run repeats with seeds RANDOM_STATE, +1, ...; tables report mean ± std

# NSL-KDD evaluation protocol:
#   "random"   - merge KDDTrain+ and KDDTest+ (148,517 rows) and split randomly.
#                This is what IDS-MTran did, so it is the default.
#   "official" - train on KDDTrain+, test on KDDTest+ (contains 17 attack types
#                never seen in training - a harder generalisation test).
NSL_KDD_SPLIT = "random"

# CIC-DDoS2019 files to load (glob pattern inside data/cic_ddos2019/).
# In the Kaggle parquet mirror (dhoogla/cicddos2019) the "-testing" files hold
# UNB's 01-12 *training day* (DrDoS_DNS/LDAP/MSSQL/NTP/NetBIOS/SNMP/UDP, Syn,
# TFTP, UDP-lag, WebDDoS) - the day IDS-MTran's 12 labels come from. The
# "-training" files hold the 03-11 day (incl. Portmap), which the paper did
# not use. Use "*.parquet" for both days, or "*.csv" for the original CSVs.
CIC_DDOS2019_FILE_PATTERN = "*-testing.parquet"

# ---------------------------------------------------------------------------
# FT-Transformer hyperparameters
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FT_TOKEN_DIM = 64         # embedding dim per feature token
FT_NUM_LAYERS = 3         # transformer encoder layers
FT_NUM_HEADS = 8
FT_FFN_DIM = 128
FT_DROPOUT = 0.1
FT_ATTN_DROPOUT = 0.1

# Class weights used by every model (FT-Transformer loss, baselines' sample
# weights), per task: "balanced" = n / (k * n_c), i.e. sklearn's
# class_weight="balanced"; "sqrt" = its square root (milder for very rare
# classes); "none".
#   binary:     "balanced" - the validation-tuned threshold then corrects the
#               resulting bias towards the attack class.
#   multiclass: "sqrt" - chosen on the CIC-DDoS2019 validation split (seed 42):
#               FT-Transformer macro F1 0.667 (sqrt) vs 0.655 (balanced) vs
#               0.647 (none, never predicts WebDDoS); RF/XGBoost within 0.01
#               of each other across the three. "balanced" gives WebDDoS
#               (~35 training rows) a weight of ~500 and floods its predictions.
CLASS_WEIGHTING = {"binary": "balanced", "multiclass": "sqrt"}

USE_AMP = True            # bf16 mixed precision on GPU (~2x faster); no effect on CPU
BATCH_SIZE = 512
EVAL_BATCH_SIZE = 4096    # inference only; lower it if evaluation runs out of memory
NUM_EPOCHS = 100          # upper bound, as in IDS-MTran; early stopping ends runs sooner
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE = 5

# ---------------------------------------------------------------------------
# Reported metrics from the base paper (for comparison only, not retrained)
# Xi, C., Wang, H., & Wang, X. (2024). IDS-MTran, Scientific Reports 14, 23239.
# Values are fractions (paper reports percentages). ROC-AUC is not reported.
#
# The tasks match DATASET_TASKS: binary for NSL-KDD (Table 1), multi-class for
# CIC-DDoS2019 (Table 3, 12 labels) and UNSW-NB15 (Table 5, 10 labels). The
# paper does not say how its multi-class precision/recall/F1 are averaged;
# its recall differs from its accuracy, so they are not support-weighted
# averages. They are compared against this project's macro averages.
# ---------------------------------------------------------------------------
IDS_MTRAN_REPORTED_METRICS = {
    "nsl_kdd": {
        "task": "binary",
        "accuracy": 0.9925, "precision": 0.9907, "recall": 0.9902, "f1": 0.9905,
        "roc_auc": None,
    },
    "cic_ddos2019": {
        "task": "multi-class (12 labels)",
        "accuracy": 0.9907, "precision": 0.9981, "recall": 0.9942, "f1": 0.9961,
        "roc_auc": None,
    },
    "unsw_nb15": {
        "task": "multi-class (10 labels)",
        "accuracy": 0.997, "precision": 0.985, "recall": 0.998, "f1": 0.991,
        "roc_auc": None,
    },
    "source": "Xi, C., Wang, H., & Wang, X. (2024). A Novel Multi-Scale Network "
              "Intrusion Detection Model with Transformer. Scientific Reports, 14, 23239."
}
