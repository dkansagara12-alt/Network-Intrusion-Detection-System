"""
Data loading & preprocessing.

load_dataset() dispatches to the right loader for config.DATASET_NAME:
    "nsl_kdd"      -> src.loaders.nsl_kdd   (headerless txt, categorical one-hot)
    "unsw_nb15"    -> src.loaders.unsw_nb15 (raw 4-file release, identifiers dropped)
    "cic_ddos2019" -> the CICFlowMeter-style loader in this module

Every loader returns its cleaned feature table (load_clean_dataset()); the
train/val/test split and all fitted preprocessing statistics are then produced
the same way for every dataset by split_and_scale(), once per seed.

CICFlowMeter quirks handled by the CIC loader:
  - Leading/trailing whitespace in column names (' Label' etc.)
  - Inf / -Inf values in Flow Bytes/s and Flow Packets/s
  - Duplicate rows
  - Mixed dtypes / stray header rows re-embedded mid-file (rare CSV corruption)
"""

import glob
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from . import config


# ---------------------------------------------------------------------------
# Shared split -> impute -> drop constant columns -> scale
# ---------------------------------------------------------------------------
def split_and_scale(X: pd.DataFrame, y, class_names, test_mask=None,
                    categorical_columns=(), seed: int = None, meta: dict = None,
                    save_intermediate: bool = True, tag: str = "data") -> dict:
    """
    Splits into train/val/test, then imputes NaNs with training medians, drops
    columns that are constant in the training split, and standardises with a
    scaler fitted on the training split. None of these statistics see the
    val/test rows, so preprocessing cannot leak test information.

    X must be fully numeric (NaN allowed); y holds class indices into
    class_names (binary: 0 = normal, 1 = attack). If test_mask is given (boolean array),
    those rows form the test set (e.g. the official KDDTest+ file); otherwise a
    stratified random split of config.TEST_SIZE is used.

    categorical_columns names the original categorical features whose one-hot
    columns are prefixed "<name>_" in X. The column indices of each one-hot
    block are returned as "cat_groups", so the FT-Transformer can give each
    categorical feature a single embedding token.

    seed (default config.RANDOM_STATE) sets the random split, so each seed of
    a multi-seed run sees a different train/val/test partition. meta (loader
    statistics such as row counts before/after deduplication) is passed through.
    """
    seed = config.RANDOM_STATE if seed is None else seed
    y = np.asarray(y, dtype=np.int64)
    idx = np.arange(len(X))

    if test_mask is None:
        train_full_idx, test_idx = train_test_split(
            idx, test_size=config.TEST_SIZE, stratify=y, random_state=seed
        )
    else:
        test_mask = np.asarray(test_mask, dtype=bool)
        train_full_idx, test_idx = idx[~test_mask], idx[test_mask]

    val_ratio = config.VAL_SIZE / (1 - config.TEST_SIZE)
    train_idx, val_idx = train_test_split(
        train_full_idx, test_size=val_ratio, stratify=y[train_full_idx],
        random_state=seed
    )

    X_train, X_val, X_test = X.iloc[train_idx], X.iloc[val_idx], X.iloc[test_idx]

    # impute with training medians
    medians = X_train.median()
    X_train, X_val, X_test = (d.fillna(medians) for d in (X_train, X_val, X_test))

    # drop columns that carry no signal in the training split
    # (constant, or entirely NaN so no median exists)
    nunique = X_train.nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        print(f"[{tag}] Dropped {len(constant_cols)} constant column(s): {constant_cols}")
    feature_names = [c for c in X.columns if c not in constant_cols]
    cat_groups = [[i for i, f in enumerate(feature_names) if f.startswith(cat + "_")]
                  for cat in categorical_columns]
    cat_groups = [g for g in cat_groups if g]

    X_train, X_val, X_test = (d[feature_names].values.astype(np.float32)
                              for d in (X_train, X_val, X_test))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    if save_intermediate:
        joblib.dump(scaler, os.path.join(config.OUTPUT_DIR, "scaler.joblib"))
        joblib.dump(feature_names, os.path.join(config.OUTPUT_DIR, "feature_names.joblib"))
        joblib.dump(medians[feature_names], os.path.join(config.OUTPUT_DIR, "imputer_medians.joblib"))
        # everything predict.py needs to rebuild the model and encode new flows
        joblib.dump({"dataset": config.DATASET_NAME, "task": config.TASK,
                     "class_names": list(class_names),
                     "categorical_columns": list(categorical_columns),
                     "cat_groups": cat_groups, "seed": seed},
                    os.path.join(config.OUTPUT_DIR, "inference_meta.joblib"))
        print(f"[{tag}] Saved scaler, feature_names, imputer medians and inference "
              f"metadata to {config.OUTPUT_DIR}")

    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    print(f"[{tag}] Final feature count: {len(feature_names)} columns "
          f"({len(feature_names) - sum(map(len, cat_groups))} numerical + "
          f"{len(cat_groups)} one-hot encoded categorical features)")
    # share of non-normal rows (class 0 is the normal class in every dataset)
    print(f"[{tag}] Train: {X_train.shape} (attack share {(y_train > 0).mean():.3f}), "
          f"Val: {X_val.shape}, Test: {X_test.shape} (attack share {(y_test > 0).mean():.3f}), "
          f"{len(class_names)} classes")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "feature_names": feature_names,
        "cat_groups": cat_groups,
        "class_names": list(class_names),
        "scaler": scaler,
        "test_index": test_idx,  # row positions in X, e.g. to pick demo samples
        "meta": meta or {},
    }


# ---------------------------------------------------------------------------
# CICFlowMeter-style loader (CIC-DDoS2019)
# ---------------------------------------------------------------------------
# Labels that only exist in CIC-IDS2017. Guards against the 2017 dataset being
# loaded by mistake as CIC-DDoS2019 (both use the same CICFlowMeter format).
_CIC_IDS2017_ONLY_LABELS = {"DOS HULK", "PORTSCAN", "HEARTBLEED", "FTP-PATATOR",
                            "SSH-PATATOR", "BOT", "INFILTRATION"}


def _find_data_files(data_dir: str, pattern: str):
    files = sorted(glob.glob(os.path.join(data_dir, "**", pattern), recursive=True))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found under {data_dir}. "
            f"Place the CIC-DDoS2019 files there, or change "
            f"config.CIC_DDOS2019_FILE_PATTERN."
        )
    return files


def _read_one_file(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path, low_memory=False, encoding="latin1")
    elif ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")


def _load_and_merge(data_dir: str) -> pd.DataFrame:
    files = _find_data_files(data_dir, config.CIC_DDOS2019_FILE_PATTERN)
    print(f"[data] Found {len(files)} data file(s):")
    for f in files:
        print(f"   - {os.path.basename(f)}")

    frames = []
    for f in files:
        try:
            df = _read_one_file(f)
        except Exception as e:
            print(f"[data] WARNING: failed to read {f}: {e}")
            continue
        # strip whitespace from column names (classic CICFlowMeter CSV issue,
        # harmless no-op for already-clean parquet columns)
        df.columns = [str(c).strip() for c in df.columns]
        frames.append(df)

    merged = pd.concat(frames, axis=0, ignore_index=True)
    print(f"[data] Merged shape before cleaning: {merged.shape}")
    return merged


def _find_label_column(df: pd.DataFrame) -> str:
    for cand in config.LABEL_COLUMN_CANDIDATES:
        if cand in df.columns:
            return cand
        if cand.strip() in df.columns:
            return cand.strip()
    # fallback: last column, common convention in this dataset
    print("[data] WARNING: could not find a known label column name, "
          "falling back to the last column.")
    return df.columns[-1]


def _binary_encode_labels(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.upper()
    return (normalized != "BENIGN").astype(int)


def _cic_class_name(label) -> str:
    """'DrDoS_DNS' -> 'DNS', 'UDP-lag' -> 'UDPLag', so both days' names agree."""
    name = str(label).strip().removeprefix("DrDoS_")
    return {"UDP-lag": "UDPLag", "BENIGN": "Benign"}.get(name, name)


def _clean(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    before = len(df)

    # drop identifier columns first so rows that differ only in IP / flow ID /
    # timestamp are caught by the duplicate check below
    drop_like = ["Unnamed: 0", "Flow ID", "Source IP", "Src IP", "Source Port",
                 "Src Port", "Destination IP", "Dst IP", "Timestamp", "SimillarHTTP"]
    df = df.drop(columns=[c for c in drop_like if c in df.columns])

    # drop exact duplicate rows (before splitting, so no row lands in both
    # train and test)
    df = df.drop_duplicates()

    labels = df[label_col]
    features = df.drop(columns=[label_col]).apply(pd.to_numeric, errors="coerce")

    # inf/-inf -> NaN; imputed later from training medians in split_and_scale
    features = features.replace([np.inf, -np.inf], np.nan)

    df_clean = features.copy()
    df_clean[label_col] = labels.values
    df_clean = df_clean.dropna(subset=[label_col])

    after = len(df_clean)
    print(f"[data] Rows before cleaning: {before}, after cleaning: {after}")
    return df_clean


def _load_cic_clean() -> dict:
    raw = _load_and_merge(config.DATA_DIR)
    label_col = _find_label_column(raw)
    print(f"[data] Using label column: '{label_col}'")
    print(f"[data] Class distribution (raw):\n{raw[label_col].value_counts().to_string()}")

    found = set(raw[label_col].astype(str).str.strip().str.upper().unique())
    wrong = found & _CIC_IDS2017_ONLY_LABELS
    if wrong:
        raise ValueError(
            f"{config.DATA_DIR} contains CIC-IDS2017 labels {sorted(wrong)}, not "
            f"CIC-DDoS2019. Move those files out and add the CIC-DDoS2019 files."
        )

    cleaned = _clean(raw, label_col)
    X = cleaned.drop(columns=[label_col])
    if config.TASK == "multiclass":
        names = cleaned[label_col].map(_cic_class_name)
        class_names = ["Benign"] + sorted(set(names) - {"Benign"})
        y = names.map({name: i for i, name in enumerate(class_names)})
        print(f"[data] Class distribution:\n{names.value_counts().to_string()}")
    else:
        class_names = ["BENIGN", "ATTACK"]
        y = _binary_encode_labels(cleaned[label_col])
        print(f"[data] Binary class distribution:\n{y.value_counts()} (0=BENIGN, 1=ATTACK)")

    meta = {"rows_raw": len(raw), "rows_clean": len(cleaned)}
    return dict(X=X, y=y.values, class_names=class_names, meta=meta, tag="data")


def load_clean_dataset() -> dict:
    """
    Loads and cleans config.DATASET_NAME (the slow part) and returns the
    keyword arguments for split_and_scale(), so a multi-seed run loads the
    data once and only re-splits it per seed.
    """
    if config.DATASET_NAME == "nsl_kdd":
        from .loaders.nsl_kdd import load_clean
        return load_clean()
    if config.DATASET_NAME == "unsw_nb15":
        from .loaders.unsw_nb15 import load_clean
        return load_clean()
    return _load_cic_clean()


def load_dataset(save_intermediate: bool = True, seed: int = None):
    """
    Loads config.DATASET_NAME and returns a dict with numpy arrays and metadata
    (feature names, scaler), with the same interface for every dataset, so
    train.py / evaluate.py / baseline_models.py stay dataset-agnostic.
    """
    return split_and_scale(**load_clean_dataset(), seed=seed,
                           save_intermediate=save_intermediate)


if __name__ == "__main__":
    data = load_dataset()
    print("Preprocessing complete.")
