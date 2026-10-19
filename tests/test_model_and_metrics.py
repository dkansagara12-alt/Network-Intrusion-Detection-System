"""
Unit tests for the FT-Transformer, class weighting and evaluation code
(synthetic inputs, no dataset files needed).
"""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from src import config
from src.evaluate import (build_comparison_table, evaluate_predictions, tune_threshold)
from src.ft_transformer import FTTransformer
from src.train import class_weights


class FTTransformerTest(unittest.TestCase):
    def test_one_token_per_feature_with_categorical_groups(self):
        # 10 columns: 5 numerical + one-hot blocks of 3 and 2 columns -> 7 tokens
        model = FTTransformer(num_features=10, d_token=16, num_heads=4,
                              cat_groups=[[3, 4, 5], [8, 9]])
        x = torch.randn(6, 10)
        self.assertEqual(model.tokenizer.num_tokens, 7)
        self.assertEqual(tuple(model.tokenizer(x).shape), (6, 7, 16))

    def test_output_has_one_logit_per_class(self):
        for classes in (2, 10):
            model = FTTransformer(num_features=8, d_token=16, num_heads=4, num_classes=classes)
            self.assertEqual(tuple(model(torch.randn(4, 8)).shape), (4, classes))

    def test_categorical_token_depends_only_on_category(self):
        model = FTTransformer(num_features=4, d_token=8, num_heads=2, cat_groups=[[1, 2, 3]])
        onehot = torch.tensor([[0.5, 0, 1, 0], [-2.0, 0, 1, 0], [0.5, 1, 0, 0]])
        tokens = model.tokenizer(onehot)
        cat_token = tokens[:, -1]  # the categorical feature's token
        torch.testing.assert_close(cat_token[0], cat_token[1])  # same category
        self.assertFalse(torch.allclose(cat_token[0], cat_token[2]))  # different category

    def test_checkpoint_round_trip(self):
        kwargs = dict(num_features=6, d_token=8, num_heads=2, num_classes=3, cat_groups=[[4, 5]])
        a, b = FTTransformer(**kwargs).eval(), FTTransformer(**kwargs).eval()
        b.load_state_dict(a.state_dict())
        x = torch.randn(5, 6)
        torch.testing.assert_close(a(x), b(x))


class ClassWeightsTest(unittest.TestCase):
    def setUp(self):
        self.saved = config.TASK

    def tearDown(self):
        config.TASK = self.saved

    def test_balanced_and_sqrt_weights(self):
        y = np.array([0] * 90 + [1] * 9 + [2] * 1)
        config.TASK = "binary"  # "balanced"
        balanced = class_weights(y, 3)
        np.testing.assert_allclose(balanced, 100 / (3 * np.array([90, 9, 1])), rtol=1e-6)
        config.TASK = "multiclass"  # "sqrt"
        np.testing.assert_allclose(class_weights(y, 3), np.sqrt(balanced), rtol=1e-6)


class EvaluationTest(unittest.TestCase):
    def test_threshold_maximises_validation_f1(self):
        y = np.array([0, 0, 0, 0, 1, 1])
        p = np.array([0.1, 0.2, 0.3, 0.65, 0.7, 0.9])
        self.assertAlmostEqual(tune_threshold(y, p), 0.7)  # separates the classes perfectly

    def test_binary_metrics_for_a_perfect_model(self):
        y = np.array([0, 1, 0, 1, 1])
        P = np.column_stack([1 - y, y]).astype(float)
        metrics, preds = evaluate_predictions(y, P, y, P, ["NORMAL", "ATTACK"])
        np.testing.assert_array_equal(preds, y)
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
            self.assertAlmostEqual(metrics[key], 1.0)

    def test_multiclass_macro_and_per_class_metrics(self):
        y = np.array([0, 0, 0, 0, 1, 1, 2, 2])
        pred = np.array([0, 0, 0, 0, 1, 1, 2, 0])  # one class-2 row wrong
        P = np.eye(3)[pred] * 0.9 + 0.1 / 3
        metrics, _ = evaluate_predictions(y, P, y, P, ["N", "A", "B"])
        self.assertAlmostEqual(metrics["accuracy"], 7 / 8)
        self.assertAlmostEqual(metrics["per_class_f1"]["A"], 1.0)
        self.assertAlmostEqual(metrics["per_class_f1"]["B"], 2 / 3)
        self.assertAlmostEqual(metrics["f1"], np.mean(list(metrics["per_class_f1"].values())))
        self.assertEqual(metrics["per_class_support"], {"N": 4, "A": 2, "B": 2})

    def test_comparison_table_reports_mean_and_std(self):
        tmp = tempfile.TemporaryDirectory()
        saved = (config.OUTPUT_DIR, config.TASK, config.DATASET_NAME)
        config.OUTPUT_DIR, config.TASK, config.DATASET_NAME = tmp.name, "binary", "nsl_kdd"
        try:
            runs = [{"Model A": {"accuracy": a, "precision": 1, "recall": 1, "f1": a,
                                 "roc_auc": 1, "pr_auc": 1, "threshold": 0.5,
                                 "f1_at_0.5": a, "train_time_sec": 1.0}} for a in (0.9, 0.8)]
            build_comparison_table(runs, [1, 2])
            table = pd.read_csv(os.path.join(tmp.name, "model_comparison.csv"), encoding="utf-8-sig")
            self.assertEqual(table.loc[0, "Accuracy"], "0.8500 ± 0.0707")
            self.assertIn("IDS-MTran", table.loc[1, "Model"])
        finally:
            config.OUTPUT_DIR, config.TASK, config.DATASET_NAME = saved
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
