"""
Unit tests for the shared split -> impute -> drop constants -> scale step,
on small synthetic data (no dataset files needed).

Run:  python -m unittest discover tests      (or: python -m pytest tests)
"""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from src import config
from src.data_preprocessing import split_and_scale


def synthetic(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "a": rng.normal(10, 3, n),
        "b": rng.exponential(5, n),
        "always_7": np.full(n, 7.0),
        "proto_tcp": rng.integers(0, 2, n).astype(np.float32),
    })
    X["proto_udp"] = 1.0 - X["proto_tcp"]
    y = (rng.random(n) < 0.3).astype(int)
    return X, y


class SplitAndScaleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.saved_output_dir = config.OUTPUT_DIR
        config.OUTPUT_DIR = self.tmp.name

    def tearDown(self):
        config.OUTPUT_DIR = self.saved_output_dir
        self.tmp.cleanup()

    def split(self, X, y, **kwargs):
        return split_and_scale(X, y, class_names=["NORMAL", "ATTACK"],
                               save_intermediate=False, tag="test", **kwargs)

    def test_splits_are_disjoint_and_complete(self):
        X, y = synthetic()
        X["row_id"] = np.arange(len(X), dtype=float)  # survives scaling as a unique key
        d = self.split(X, y)
        col = d["feature_names"].index("row_id")
        ids = [np.round(d["scaler"].inverse_transform(d[k])[:, col]).astype(int)
               for k in ("X_train", "X_val", "X_test")]
        tr, va, te = (set(i) for i in ids)
        self.assertFalse(tr & va or tr & te or va & te, "a row appears in two splits")
        self.assertEqual(len(tr | va | te), len(X))

    def test_split_proportions_and_stratification(self):
        X, y = synthetic()
        d = self.split(X, y)
        n = len(X)
        self.assertAlmostEqual(len(d["y_test"]) / n, config.TEST_SIZE, delta=0.01)
        self.assertAlmostEqual(len(d["y_val"]) / n, config.VAL_SIZE, delta=0.01)
        for part in ("y_train", "y_val", "y_test"):
            self.assertAlmostEqual(d[part].mean(), y.mean(), delta=0.02)

    def test_scaler_is_fitted_on_training_rows_only(self):
        X, y = synthetic()
        d = self.split(X, y)
        np.testing.assert_allclose(d["X_train"].mean(axis=0), 0, atol=1e-4)
        np.testing.assert_allclose(d["X_train"].std(axis=0), 1, atol=1e-3)
        # val/test are transformed with the training statistics, not their own
        self.assertGreater(np.abs(d["X_test"].mean(axis=0)).max(), 1e-4)

    def test_missing_values_get_training_median(self):
        X, y = synthetic()
        X.loc[:9, "a"] = np.nan
        d = self.split(X, y)
        self.assertFalse(np.isnan(d["X_train"]).any() or np.isnan(d["X_test"]).any())

    def test_constant_columns_are_dropped(self):
        X, y = synthetic()
        d = self.split(X, y)
        self.assertNotIn("always_7", d["feature_names"])
        self.assertEqual(d["X_train"].shape[1], len(d["feature_names"]))

    def test_categorical_groups_point_at_one_hot_columns(self):
        X, y = synthetic()
        d = self.split(X, y, categorical_columns=["proto"])
        self.assertEqual(len(d["cat_groups"]), 1)
        names = [d["feature_names"][i] for i in d["cat_groups"][0]]
        self.assertEqual(sorted(names), ["proto_tcp", "proto_udp"])

    def test_predefined_test_set_is_respected(self):
        X, y = synthetic()
        mask = np.zeros(len(X), dtype=bool)
        mask[-300:] = True
        d = self.split(X, y, test_mask=mask)
        np.testing.assert_array_equal(d["test_index"], np.arange(len(X) - 300, len(X)))

    def test_seed_controls_the_split(self):
        X, y = synthetic()
        a, b, c = (self.split(X, y, seed=s)["test_index"] for s in (1, 1, 2))
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))

    def test_artifacts_are_saved_when_asked(self):
        X, y = synthetic()
        split_and_scale(X, y, class_names=["NORMAL", "ATTACK"], save_intermediate=True, tag="test")
        for f in ("scaler.joblib", "feature_names.joblib", "imputer_medians.joblib",
                  "inference_meta.joblib"):
            self.assertTrue(os.path.exists(os.path.join(config.OUTPUT_DIR, f)), f)


if __name__ == "__main__":
    unittest.main()
