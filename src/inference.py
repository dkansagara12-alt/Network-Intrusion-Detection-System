"""
Inference: classify new network flows with a trained FT-Transformer.

Uses the artifacts a training run saves (outputs/<dataset>/: scaler, feature
list, imputer medians, inference metadata, all_metrics.json; checkpoints/
<dataset>/ft_transformer_seed<seed>.pt) and applies exactly the training
preprocessing to flow records in the dataset's own column format:
  one-hot encode categoricals onto the training columns -> impute with the
  training medians -> scale -> FT-Transformer -> class.
Identifier and label columns in the input are ignored.

Binary models use the decision threshold that was tuned on the validation
split for that seed; multi-class models predict the most probable class.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
import torch

from . import config
from .evaluate import predict_proba
from .ft_transformer import FTTransformer

FT_MODEL_NAME = "FT-Transformer (this work)"


class FlowClassifier:
    def __init__(self, dataset: str, seed: int = None):
        config.set_dataset(dataset)
        out = config.OUTPUT_DIR
        self.meta = joblib.load(os.path.join(out, "inference_meta.joblib"))
        self.seed = self.meta["seed"] if seed is None else seed
        self.class_names = self.meta["class_names"]
        self.feature_names = joblib.load(os.path.join(out, "feature_names.joblib"))
        self.medians = joblib.load(os.path.join(out, "imputer_medians.joblib"))
        self.scaler = joblib.load(os.path.join(out, "scaler.joblib"))
        self.onehot_columns = [self.feature_names[i] for g in self.meta["cat_groups"] for i in g]

        self.threshold = None
        if len(self.class_names) == 2:
            with open(os.path.join(out, "all_metrics.json")) as f:
                runs = json.load(f)["per_seed"]
            self.threshold = runs[str(self.seed)][FT_MODEL_NAME]["threshold"]

        self.model = FTTransformer(
            num_features=len(self.feature_names), d_token=config.FT_TOKEN_DIM,
            num_layers=config.FT_NUM_LAYERS, num_heads=config.FT_NUM_HEADS,
            ffn_dim=config.FT_FFN_DIM, num_classes=len(self.class_names),
            cat_groups=self.meta["cat_groups"],
        ).to(config.DEVICE)
        checkpoint = os.path.join(config.CHECKPOINT_DIR, f"ft_transformer_seed{self.seed}.pt")
        self.model.load_state_dict(torch.load(checkpoint, map_location=config.DEVICE))
        self.model.eval()

    def encode(self, flows: pd.DataFrame) -> np.ndarray:
        """Raw flow records -> the scaled feature matrix the model was trained on."""
        df = flows.copy()
        df.columns = [str(c).strip() for c in df.columns]
        cats = [c for c in self.meta["categorical_columns"] if c in df.columns]
        for col in cats:
            df[col] = df[col].astype(str).str.strip()
            if self.meta["dataset"] == "unsw_nb15":  # the UNSW loader lower-cases these
                df[col] = df[col].str.lower()
        X = pd.get_dummies(df, columns=cats, prefix=cats, dtype=np.float32)

        missing = [c for c in self.feature_names
                   if c not in X.columns and c not in self.onehot_columns]
        if missing:
            print(f"[inference] WARNING: {len(missing)} feature column(s) missing from the "
                  f"input, filled with training medians: {missing[:5]}...")
        # columns not seen in training (e.g. identifiers, labels) are dropped;
        # a category never seen in training leaves its one-hot block all zero
        X = X.reindex(columns=self.feature_names)
        X[self.onehot_columns] = X[self.onehot_columns].fillna(0.0)
        X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        X = X.fillna(self.medians)
        return self.scaler.transform(X.values.astype(np.float32)).astype(np.float32)

    def predict(self, flows: pd.DataFrame) -> pd.DataFrame:
        """One row per flow: predicted class and the model's probability for it."""
        P = predict_proba(self.model, self.encode(flows))
        if self.threshold is not None:
            pred = (P[:, 1] >= self.threshold).astype(int)
        else:
            pred = P.argmax(axis=1)
        return pd.DataFrame({
            "predicted": [self.class_names[i] for i in pred],
            "probability": P[np.arange(len(pred)), pred].round(4),
        }, index=flows.index)
