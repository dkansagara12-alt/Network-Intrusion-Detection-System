# Network Intrusion Detection with the FT-Transformer

23AID205 project. The FT-Transformer (a transformer for tabular data) is compared
with Logistic Regression, Random Forest and XGBoost, and with the published
results of IDS-MTran (Xi et al., 2024), on three intrusion-detection datasets.
Each dataset uses the same task as the paper:

| Dataset | Task | Classes |
|---|---|---|
| **NSL-KDD** | binary | normal vs attack |
| **CIC-DDoS2019** | multi-class | Benign + 11 DDoS attack types |
| **UNSW-NB15** | multi-class | Normal + 9 attack categories |

**Project webpage:** open [`docs/index.html`](docs/index.html) in a browser (works
offline). With GitHub Pages enabled for the `docs/` folder it is also at
<https://dkansagara12-alt.github.io/AI-ML/>. It covers the motivation,
methodology, EDA, results, discussion and references.

**Repository:** <https://github.com/dkansagara12-alt/AI-ML>

## Results

Mean ± standard deviation over 5 seeds (each seed = a different train/val/test
split and model initialisation). F1 for NSL-KDD, macro F1 for the multi-class
datasets. Full tables: `outputs/<dataset>/model_comparison.csv`.

| Model | NSL-KDD (F1) | CIC-DDoS2019 (macro F1) | UNSW-NB15 (macro F1) |
|---|---|---|---|
| FT-Transformer | 0.9916 ± 0.0008 | 0.6840 ± 0.0103 | 0.5859 ± 0.0088 |
| Logistic Regression | 0.9557 ± 0.0012 | 0.6152 ± 0.0044 | 0.4400 ± 0.0060 |
| Random Forest | 0.9958 ± 0.0002 | 0.6890 ± 0.0047 | 0.5135 ± 0.0059 |
| XGBoost | 0.9961 ± 0.0006 | 0.7149 ± 0.0103 | 0.6042 ± 0.0053 |
| IDS-MTran (paper, reported) | 0.9905 | 0.9961 | 0.9910 |

XGBoost is the strongest and cheapest model on all three datasets. The
FT-Transformer matches IDS-MTran's reported result on NSL-KDD, clearly beats
Logistic Regression, beats Random Forest on UNSW-NB15, and has the best macro
recall on UNSW-NB15. IDS-MTran's multi-class scores could not be reproduced by
any model here (see section 7). Accuracy is at least 0.95 for every model on every
dataset, so macro F1 is the more informative number.

## 1. Setup

```bash
python -m venv venv
venv\Scripts\activate            # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

For GPU training, install the CUDA build of PyTorch first (the default pip
wheel on Windows is CPU-only), e.g.
`pip install torch --index-url https://download.pytorch.org/whl/cu130`.

Then download the datasets: see [`data/README.md`](data/README.md).

## 2. Reproduce the experiments

```bash
python main.py --dataset nsl_kdd           # binary
python main.py --dataset cic_ddos2019      # 12 classes
python main.py --dataset unsw_nb15         # 10 classes
```

Each run loads and cleans the dataset once, then for each of 5 seeds (42–46):
splits and scales the data, trains the FT-Transformer, trains the three
baselines, and evaluates all four on the test split. It ends with a
mean ± std comparison table that includes IDS-MTran's reported numbers.

Time for 5 seeds on an RTX 4060 laptop GPU: NSL-KDD ~10 min, CIC-DDoS2019
~45 min, UNSW-NB15 ~3.5 h. If a run is interrupted, rerun the same command with
`--resume`: it skips the seeds already saved.

Options: `--num-seeds 1` for a quick run; `--task binary|multiclass` to
override the dataset's task (results go to `outputs/<dataset>_<task>/`);
`--nsl-split official` trains on KDDTrain+ and tests on KDDTest+, whose test set
contains 17 attack types never seen in training.

## 3. Demo: classify flow records

```bash
python predict.py --dataset unsw_nb15 --input demo/unsw_nb15_samples.csv
python predict.py --dataset cic_ddos2019 --input demo/cic_ddos2019_samples.csv
python predict.py --dataset nsl_kdd --input demo/nsl_kdd_samples.csv
```

`predict.py` loads the trained FT-Transformer (seed 42) and the saved
preprocessing, encodes the flow records exactly as during training, and prints
the predicted class with its probability. The `demo/` samples are flows from each
dataset's test split (up to 20 per class) in the dataset's own column format; their
`true_label` column is only used to score the predictions. Any CSV/Parquet file
with the dataset's columns works, and `--output predictions.csv` saves the
results. `python scripts/make_demo_samples.py` rebuilds the samples and checks
that inference reproduces the evaluated features exactly.

## 4. Tests and validation

```bash
python -m unittest discover tests
```

(`python -m pytest tests` works too if pytest is installed.)

- `tests/test_preprocessing.py`: splits are disjoint, complete and stratified;
  imputation and scaling use training rows only; constant columns are dropped;
  categorical groups are correct; seeds control the split.
- `tests/test_model_and_metrics.py`: one token per feature, output shapes,
  checkpoint round trip, class weights, threshold tuning, binary and multi-class
  metrics, the mean ± std table.
- `tests/test_data_validation.py` (real data, skipped if it is not present):
  no identifier or label column in the features, no duplicate rows, no test
  flow that also appears in training with the same label, the right files and
  classes, and correct demo predictions. The UNSW-NB15 checks load 2.5M rows
  and only run with `RUN_SLOW_TESTS=1`.

## 5. EDA, charts and webpage

```bash
python scripts/eda.py            # outputs/figures/eda/ + outputs/eda_summary.csv
python scripts/plot_results.py   # outputs/figures/results/
python scripts/build_site.py     # docs/index.html from site/template.html
```

`build_site.py` fills the webpage's tables, figures, demo output and key numbers
from the latest `outputs/`, so after a rerun one command updates the page. Edit
the text in `site/template.html`, not in `docs/index.html`.

## 6. Method in brief

**Preprocessing** (`src/loaders/`, `src/data_preprocessing.py`):
- drop identifier columns (IPs, ports, timestamps, flow and row IDs) and the
  other label column;
- remove duplicate rows **before** splitting;
- one-hot encode categorical columns;
- stratified 70/15/15 train/validation/test split;
- median imputation, constant-column removal and standard scaling, all fitted on
  the **training split only**.

**FT-Transformer** (`src/ft_transformer.py`): every feature becomes one 64-d
token. Numerical features use a learned linear projection, and categorical
features a learned embedding of their one-hot block. A [CLS] token is
prepended, followed by 3 pre-norm Transformer blocks (8 heads) and an MLP head.
It is trained with AdamW (lr 1e-4) for up to 100 epochs with early stopping
on validation loss and bf16 mixed precision.

**Baselines** (`src/baseline_models.py`): Logistic Regression, Random Forest
(200 trees) and XGBoost (300 trees, depth 8).

**Class imbalance:** every model uses the same class weights
(`CLASS_WEIGHTING`). Binary uses "balanced" weights, and the decision threshold
is tuned on the validation split. Multi-class uses square-root-balanced weights,
chosen on the CIC-DDoS2019 validation split.

**Metrics** (`src/evaluate.py`):
- binary: precision/recall/F1 for the attack class, plus ROC-AUC and PR-AUC;
- multi-class: macro-averaged precision/recall/F1 (every class counts
  equally), weighted F1, one-vs-rest ROC-AUC and PR-AUC, and per-class F1
  (`per_class_f1.csv`);
- the validation and test splits are never used for training, and no decision
  is made on the test split.

**Outputs per dataset** (`outputs/<dataset>/`): `model_comparison.csv`,
`per_seed_metrics.csv`, `per_class_f1.csv`, `all_metrics.json`, the
FT-Transformer confusion matrix, and the preprocessing artifacts used by
`predict.py`. Model weights are in `checkpoints/<dataset>/ft_transformer_seed<seed>.pt`.
The earlier binary runs of CIC-DDoS2019 and UNSW-NB15 are kept in
`outputs/*_binary/`.

## 7. Comparing against IDS-MTran

IDS-MTran's numbers are taken from the paper; it is not re-trained here (no code
is published). Points for the report:
- The tasks match the paper. NSL-KDD (binary) is the closest comparison.
- The paper does not state its train/test split or how its multi-class
  precision/recall/F1 are averaged. Its recall differs from its accuracy, so they
  are not support-weighted averages; they are shown next to this project's macro
  averages.
- For UNSW-NB15 the paper uses all 2.54M raw rows without removing duplicates.
  Removing them deletes about three quarters of the attack rows, mostly repeated
  "Generic" flows, which are easy to classify and inflate scores.
- CIC-DDoS2019: the Kaggle mirror has no SSDP flows and includes Syn, so one of
  the 12 classes differs from the paper's. 3,941 flows have exactly the same
  features as a flow of a different class, so those cannot all be classified
  correctly.

## 8. Data problems found and fixed

1. **NSL-KDD:** the original loader merged 16 files (ARFF copies, subsets and a
   duplicated folder), so 45% of test rows were also in training. Only
   `KDDTrain+` and `KDDTest+` are used now.
2. **CIC-DDoS2019:** the folder meant for it contained CIC-IDS2017. The loader
   now refuses CIC-IDS2017 labels.
3. **UNSW-NB15:** an early run used the row ID as a feature; the files are
   sorted by class, so it leaked the label. All identifiers are dropped now.
   A mixed-type column (`ct_ftp_cmd`) also hid 1,371 duplicate rows until
   values were converted to numbers before deduplication.

The validation tests in `tests/test_data_validation.py` guard all of these.

## 9. Project structure

```
nids_project/
├── main.py                   # train + evaluate one dataset (5 seeds)
├── predict.py                # demo: classify flow records with a trained model
├── src/
│   ├── config.py             # paths, tasks, hyperparameters, IDS-MTran numbers
│   ├── data_preprocessing.py # dataset dispatch, CIC-DDoS2019 loader, split/scale
│   ├── loaders/              # NSL-KDD and UNSW-NB15 loaders
│   ├── ft_transformer.py     # FT-Transformer (PyTorch)
│   ├── train.py              # training loop, early stopping, class weights
│   ├── baseline_models.py    # Logistic Regression, Random Forest, XGBoost
│   ├── evaluate.py           # metrics, thresholds, plots, comparison tables
│   └── inference.py          # loads a trained model for predict.py
├── scripts/                  # EDA, result charts, demo samples, webpage build
├── tests/                    # unit tests and data validation
├── demo/                     # sample flows from each test split
├── site/                     # webpage template + stylesheet
├── docs/                     # built webpage (offline / GitHub Pages)
├── outputs/                  # tables, metrics, figures
├── checkpoints/              # trained FT-Transformer weights
└── data/                     # datasets (not in git, see data/README.md)
```

## Contributors

| Name | Roll number |
|---|---|
| Dhruv Jayesh Kansagara | CB.AI.U4AID25114 |
| Harish Kumar | CB.AI.U4AID25117 |
| Prithve KC | CB.AI.U4AID25122 |
| Raghav Vatsan | CB.AI.U4AID25151 |

## References

See the webpage's reference list. The main ones: Gorishniy et al., "Revisiting
Deep Learning Models for Tabular Data" (NeurIPS 2021); Xi et al., "A novel
multi-scale network intrusion detection model with transformer" (Scientific
Reports, 2024); and the NSL-KDD, CIC-DDoS2019 and UNSW-NB15 dataset papers.
