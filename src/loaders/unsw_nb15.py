"""
Data loading & preprocessing for UNSW-NB15 (full raw release).

Uses the four raw files UNSW-NB15_1.csv ... UNSW-NB15_4.csv (2,540,047 rows,
no header row), which is the version IDS-MTran evaluated on. Column names
come from NUSW-NB15_features.csv. This loader:

  1. Drops identifier columns: srcip, sport, dstip, dsport, stime, ltime.
     These describe the testbed, not the traffic behaviour - every attack in
     this dataset comes from 175.45.176.0-3, so leaving srcip in would let a
     model "detect" attacks by IP address.
  2. Picks the target for config.TASK and drops the other label column
     (each one gives the other away):
       multiclass - attack_cat: Normal + 9 attack categories, as in
                    IDS-MTran (the raw data spells Backdoor two ways)
       binary     - the 0/1 'label' column (0 = normal, 1 = attack)
  3. Drops duplicate rows (after removing identifiers): 512,757 rows in
     the multi-class version (522,732 in binary), mostly repeated 'Generic'
     attack flows; attack rows go from 321,283 to 88,424 (binary: 78,449).
  4. One-hot encodes proto, state and service.

The pre-split UNSW_NB15_training-set.csv / testing-set.csv files are not
used. (Note: in this copy their names are swapped - "training-set" has
82,332 rows and "testing-set" 175,341, the reverse of the official sizes.)
"""

import os
import numpy as np
import pandas as pd

from .. import config
from ..data_preprocessing import split_and_scale

RAW_FILES = [f"UNSW-NB15_{i}.csv" for i in range(1, 5)]
FEATURES_FILE = "NUSW-NB15_features.csv"
IDENTIFIER_COLUMNS = ["srcip", "sport", "dstip", "dsport", "stime", "ltime"]
CATEGORICAL_COLUMNS = ["proto", "state", "service"]


def _column_names(data_dir: str):
    spec = pd.read_csv(os.path.join(data_dir, FEATURES_FILE), encoding="latin1")
    # normalise e.g. "ct_src_ ltm" -> "ct_src_ltm", "Sload" -> "sload"
    return [str(n).strip().lower().replace(" ", "") for n in spec["Name"]]


def load_clean() -> dict:
    """
    UNSW-NB15: load the 4 raw files -> drop identifiers -> pick the target
    (attack_cat or label) -> dedup -> one-hot encode categoricals. Returns
    the keyword arguments for split_and_scale(), which does the (per-seed)
    split and scaling.
    """
    names = _column_names(config.DATA_DIR)
    frames = []
    for fname in RAW_FILES:
        path = os.path.join(config.DATA_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. UNSW-NB15 needs all of {RAW_FILES}.")
        frames.append(pd.read_csv(path, header=None, names=names, low_memory=False))
        print(f"[unsw_nb15] Loaded {fname} ({len(frames[-1])} rows)")
    raw = pd.concat(frames, ignore_index=True)
    print(f"[unsw_nb15] Merged shape: {raw.shape}")

    multiclass = config.TASK == "multiclass"
    # normal traffic has no attack category in the raw files
    raw["attack_cat"] = (raw["attack_cat"].fillna("Normal").astype(str).str.strip()
                         .replace({"Backdoors": "Backdoor"}))
    target = "attack_cat" if multiclass else "label"
    raw = raw.drop(columns=IDENTIFIER_COLUMNS + ["label" if multiclass else "attack_cat"])
    for col in CATEGORICAL_COLUMNS:
        raw[col] = raw[col].astype(str).str.strip().str.lower()
    # convert before deduplicating: ct_ftp_cmd mixes numbers with text ("0",
    # blanks), so "0" and 0 would otherwise count as different values and
    # leave duplicates behind. Blanks / missing values become NaN and are
    # imputed from training medians in split_and_scale.
    numeric = [c for c in raw.columns if c not in CATEGORICAL_COLUMNS + [target]]
    raw[numeric] = raw[numeric].apply(pd.to_numeric, errors="coerce")
    raw[numeric] = raw[numeric].replace([np.inf, -np.inf], np.nan)

    before = len(raw)
    raw = raw.drop_duplicates().reset_index(drop=True)
    print(f"[unsw_nb15] Removed {before - len(raw)} duplicate row(s), {len(raw)} remain")

    if multiclass:
        class_names = ["Normal"] + sorted(set(raw[target]) - {"Normal"})
        y = raw[target].map({name: i for i, name in enumerate(class_names)})
        print(f"[unsw_nb15] Class distribution:\n{raw[target].value_counts().to_string()}")
    else:
        class_names = ["NORMAL", "ATTACK"]
        y = raw[target].astype(int)
        print(f"[unsw_nb15] Binary class distribution:\n{y.value_counts()} (0=NORMAL, 1=ATTACK)")

    X = raw.drop(columns=[target])
    X = pd.get_dummies(X, columns=CATEGORICAL_COLUMNS, prefix=CATEGORICAL_COLUMNS,
                       dtype=np.float32)

    meta = {"rows_raw": before, "rows_clean": len(raw)}
    return dict(X=X, y=y.values, class_names=class_names,
                categorical_columns=CATEGORICAL_COLUMNS, meta=meta, tag="unsw_nb15")


if __name__ == "__main__":
    data = split_and_scale(**load_clean())
    print("UNSW-NB15 preprocessing complete.")
