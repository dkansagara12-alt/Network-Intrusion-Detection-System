"""
Data loading & preprocessing for NSL-KDD.

NSL-KDD is distributed as headerless text files. Each row has 41 features +
a class label + a difficulty score. This loader:

  1. Loads exactly two files: KDDTrain+.txt (125,973 rows) and KDDTest+.txt
     (22,544 rows). The other files in the release must NOT be merged in:
       - KDDTrain+_20Percent / KDDTest-21 are subsets of these two files
       - the .arff files are copies of the .txt files (with labels collapsed
         to normal/anomaly, so drop_duplicates() would not catch them)
     Merging them put ~48% of test rows verbatim into the training set.
  2. Assigns the standard 41 NSL-KDD column names (no header row in the
     raw files) and drops the difficulty-level column (a KDD scoring
     artifact, not a traffic feature).
  3. One-hot encodes the three categorical columns (protocol_type, service,
     flag) since our FT-Transformer's feature tokenizer expects purely
     numeric input.
  4. Binary-encodes the label: 'normal' -> 0, any attack name -> 1.
  5. Splits according to config.NSL_KDD_SPLIT:
       "random"   - merge both files (148,517 rows, as IDS-MTran did) and
                    split randomly
       "official" - train on KDDTrain+, test on KDDTest+
"""

import glob
import os
import numpy as np
import pandas as pd

from .. import config
from ..data_preprocessing import split_and_scale

# Standard NSL-KDD 41 feature names, in official column order.
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "class",
]
CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]
TRAIN_FILE = "KDDTrain+.txt"
TEST_FILE = "KDDTest+.txt"


def _locate(data_dir: str, filename: str) -> str:
    matches = glob.glob(os.path.join(data_dir, "**", filename), recursive=True)
    if not matches:
        raise FileNotFoundError(
            f"{filename} not found under {data_dir}. NSL-KDD needs both "
            f"{TRAIN_FILE} and {TEST_FILE}."
        )
    # prefer the copy closest to data_dir if the release was unzipped twice
    path = min(matches, key=lambda p: (p.count(os.sep), p))
    if len(matches) > 1:
        print(f"[nsl_kdd] {len(matches)} copies of {filename} found, using {path}")
    return path


def _read_one_file(path: str) -> pd.DataFrame:
    # KDDTrain+.txt / KDDTest+.txt are plain comma-separated with no header row.
    df = pd.read_csv(path, header=None, encoding="latin1", low_memory=False)

    # Standard releases have a trailing "difficulty level" column (43rd col).
    num_cols = df.shape[1]
    if num_cols == len(NSL_KDD_COLUMNS) + 1:
        df.columns = NSL_KDD_COLUMNS + ["difficulty_level"]
        df = df.drop(columns=["difficulty_level"])
    elif num_cols == len(NSL_KDD_COLUMNS):
        df.columns = NSL_KDD_COLUMNS
    else:
        raise ValueError(
            f"Unexpected column count ({num_cols}) in {path}. "
            f"Expected {len(NSL_KDD_COLUMNS)} (standard) or "
            f"{len(NSL_KDD_COLUMNS) + 1} (with difficulty_level). "
            f"Check that this is a genuine NSL-KDD file."
        )
    return df


def _binary_encode_labels(series: pd.Series) -> pd.Series:
    # NSL-KDD label strings look like "normal", "neptune", "smurf.",
    # "ipsweep" etc. Some releases have a trailing "." - strip it.
    normalized = series.astype(str).str.strip().str.rstrip(".").str.lower()
    return (normalized != "normal").astype(int)


def load_clean() -> dict:
    """
    NSL-KDD: load KDDTrain+ / KDDTest+ -> dedup -> one-hot encode
    categoricals -> binary label. Returns the keyword arguments for
    split_and_scale(), which does the (per-seed) split and scaling.
    """
    if config.TASK != "binary":
        raise NotImplementedError("The NSL-KDD loader supports the binary task only.")
    train = _read_one_file(_locate(config.DATA_DIR, TRAIN_FILE))
    test = _read_one_file(_locate(config.DATA_DIR, TEST_FILE))
    train["_source"], test["_source"] = "train", "test"
    raw = pd.concat([train, test], ignore_index=True)
    print(f"[nsl_kdd] Loaded {TRAIN_FILE} ({len(train)} rows) + "
          f"{TEST_FILE} ({len(test)} rows)")

    official = config.NSL_KDD_SPLIT == "official"
    # duplicates are judged on features + the binary label (two identical
    # flows named e.g. "neptune" and "smurf" are the same "attack" row).
    # official split: only drop duplicates within each file, so KDDTest+ stays
    # the standard test set; random split: drop them across the merged data,
    # so no row can land in both train and test
    raw["_attack"] = _binary_encode_labels(raw["class"])
    features = [c for c in NSL_KDD_COLUMNS if c != "class"]
    dedup_cols = features + ["_attack"] + (["_source"] if official else [])
    before = len(raw)
    raw = raw.drop_duplicates(subset=dedup_cols).reset_index(drop=True)
    print(f"[nsl_kdd] Removed {before - len(raw)} duplicate row(s), {len(raw)} remain")

    y = raw["_attack"]
    print(f"[nsl_kdd] Binary class distribution:\n{y.value_counts()} (0=NORMAL, 1=ATTACK)")

    test_mask = (raw["_source"] == "test").values if official else None
    X = raw.drop(columns=["class", "_source", "_attack"])
    X = pd.get_dummies(X, columns=CATEGORICAL_COLUMNS, prefix=CATEGORICAL_COLUMNS,
                       dtype=np.float32)
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    print(f"[nsl_kdd] Split protocol: {config.NSL_KDD_SPLIT}")
    meta = {"rows_raw": before, "rows_clean": len(raw)}
    return dict(X=X, y=y.values, class_names=["NORMAL", "ATTACK"], test_mask=test_mask,
                categorical_columns=CATEGORICAL_COLUMNS, meta=meta, tag="nsl_kdd")


if __name__ == "__main__":
    data = split_and_scale(**load_clean())
    print("NSL-KDD preprocessing complete.")
