"""
Validation checks on the real datasets and trained artifacts. Each test is
skipped when the files it needs are not present (e.g. on a fresh clone
without the data). The UNSW-NB15 checks load 2.5M rows and only run with
the environment variable RUN_SLOW_TESTS=1.

These guard the problems found while building the project: duplicated
NSL-KDD files leaking test rows into training, identifier/label columns
leaking into the features, and CIC-IDS2017 data posing as CIC-DDoS2019.
"""

import os
import unittest

import numpy as np
import pandas as pd

from src import config
from src.data_preprocessing import load_clean_dataset, split_and_scale

IDENTIFIERS = {"srcip", "sport", "dstip", "dsport", "stime", "ltime", "Flow ID", "Source IP",
               "Src IP", "Destination IP", "Dst IP", "Timestamp", "id"}
LABELS = {"label", "attack_cat", "class", "Label", "difficulty_level"}


def has_data(dataset, filename):
    return os.path.exists(os.path.join(config.DATA_ROOT, dataset, filename))


def load(dataset):
    config.set_dataset(dataset)
    return load_clean_dataset()


class DatasetChecks:
    """Checks shared by every dataset; subclasses set DATASET and EXPECTED_CLASSES."""

    @classmethod
    def setUpClass(cls):
        cls.clean = load(cls.DATASET)

    @classmethod
    def tearDownClass(cls):
        config.set_dataset(config.DATASETS[0])

    def test_no_identifier_or_label_columns_in_features(self):
        leaked = (IDENTIFIERS | LABELS) & set(self.clean["X"].columns)
        self.assertFalse(leaked, f"leaking columns: {leaked}")

    def test_class_labels(self):
        y = np.asarray(self.clean["y"])
        self.assertEqual(len(self.clean["class_names"]), self.EXPECTED_CLASSES)
        self.assertEqual(set(np.unique(y)), set(range(self.EXPECTED_CLASSES)))

    def test_no_duplicate_rows_left(self):
        X = self.clean["X"].copy()
        X["_y"] = self.clean["y"]
        self.assertEqual(int(X.duplicated().sum()), 0)

    def test_no_test_flow_also_appears_in_training_with_its_label(self):
        d = split_and_scale(**self.clean, save_intermediate=False)
        X = self.clean["X"].reset_index(drop=True).fillna(-999)
        y = np.asarray(self.clean["y"])
        key = pd.util.hash_pandas_object(X, index=False).values
        test = np.zeros(len(X), dtype=bool)
        test[d["test_index"]] = True
        seen = set(zip(key[~test], y[~test]))  # training + validation flows
        leaked = sum((k, label) in seen for k, label in zip(key[test], y[test]))
        # identical features with a *different* label can occur (e.g. CIC-DDoS2019
        # reflection attacks); the same flow with the same label must not
        self.assertEqual(leaked, 0)


@unittest.skipUnless(has_data("nsl_kdd", "KDDTrain+.txt"), "NSL-KDD files not present")
class NSLKDDTest(DatasetChecks, unittest.TestCase):
    DATASET, EXPECTED_CLASSES = "nsl_kdd", 2

    def test_only_the_two_main_files_are_used(self):
        # KDDTrain+ (125,973) + KDDTest+ (22,544) = 148,517 rows before dedup
        self.assertEqual(self.clean["meta"]["rows_raw"], 148517)


@unittest.skipUnless(has_data("cic_ddos2019", "DNS-testing.parquet"), "CIC-DDoS2019 not present")
class CICDDoS2019Test(DatasetChecks, unittest.TestCase):
    DATASET, EXPECTED_CLASSES = "cic_ddos2019", 12

    def test_it_is_really_cic_ddos2019(self):
        names = set(self.clean["class_names"])
        self.assertTrue({"DNS", "LDAP", "NTP", "WebDDoS"} <= names)
        self.assertFalse({"PortScan", "DoS Hulk", "Bot"} & names)  # CIC-IDS2017 labels


@unittest.skipUnless(has_data("unsw_nb15", "UNSW-NB15_1.csv") and os.environ.get("RUN_SLOW_TESTS"),
                     "UNSW-NB15 not present or RUN_SLOW_TESTS not set")
class UNSWNB15Test(DatasetChecks, unittest.TestCase):
    DATASET, EXPECTED_CLASSES = "unsw_nb15", 10


@unittest.skipUnless(os.path.exists(os.path.join(config.PROJECT_ROOT, "demo", "nsl_kdd_samples.csv"))
                     and os.path.exists(os.path.join(config.PROJECT_ROOT, "checkpoints", "nsl_kdd",
                                                     "ft_transformer_seed42.pt")),
                     "demo samples or trained checkpoint not present")
class DemoInferenceTest(unittest.TestCase):
    def test_demo_predictions(self):
        from src.inference import FlowClassifier
        for dataset, min_accuracy in [("nsl_kdd", 0.95), ("cic_ddos2019", 0.6), ("unsw_nb15", 0.6)]:
            path = os.path.join(config.PROJECT_ROOT, "demo", f"{dataset}_samples.csv")
            if not os.path.exists(path):
                continue
            flows = pd.read_csv(path)
            clf = FlowClassifier(dataset)
            self.assertTrue(np.isfinite(clf.encode(flows.drop(columns=["true_label"]))).all())
            pred = clf.predict(flows)["predicted"]
            self.assertTrue(set(pred) <= set(clf.class_names))
            self.assertGreaterEqual((pred == flows["true_label"]).mean(), min_accuracy, dataset)
        config.set_dataset(config.DATASETS[0])


if __name__ == "__main__":
    unittest.main()
