# Datasets

The datasets are too large for GitHub (about 1 GB), so they are not in the
repository. Download them and place the files as shown below.

| Folder | Files | Source |
|---|---|---|
| `data/nsl_kdd/` | `KDDTrain+.txt`, `KDDTest+.txt` | NSL-KDD, Canadian Institute for Cybersecurity: <https://www.unb.ca/cic/datasets/nsl.html> (also mirrored at <https://github.com/HoaNP/NSL-KDD-DataSet>, the link used by the IDS-MTran paper) |
| `data/cic_ddos2019/` | the 17 `*.parquet` files (only `*-testing.parquet` are used) | Cleaned CIC-DDoS2019 mirror on Kaggle: <https://www.kaggle.com/datasets/dhoogla/cicddos2019> (35 MB; direct download: `https://www.kaggle.com/api/v1/datasets/download/dhoogla/cicddos2019`). Original: <https://www.unb.ca/cic/datasets/ddos-2019.html> |
| `data/unsw_nb15/` | `UNSW-NB15_1.csv` … `UNSW-NB15_4.csv`, `NUSW-NB15_features.csv` | UNSW Canberra: <https://research.unsw.edu.au/projects/unsw-nb15-dataset> (the CSV files of the full release) |

Expected layout:

```
data/
├── nsl_kdd/
│   ├── KDDTrain+.txt
│   └── KDDTest+.txt
├── cic_ddos2019/
│   ├── DNS-testing.parquet
│   ├── ...                       (17 parquet files)
│   └── UDPLag-training.parquet
└── unsw_nb15/
    ├── NUSW-NB15_features.csv
    ├── UNSW-NB15_1.csv
    ├── UNSW-NB15_2.csv
    ├── UNSW-NB15_3.csv
    └── UNSW-NB15_4.csv
```

Notes:
- In the Kaggle CIC-DDoS2019 mirror the `-testing` files hold UNB's 12 January
  "training day" (the attacks IDS-MTran used) and the `-training` files hold the
  11 March day. The project uses `*-testing.parquet`
  (`CIC_DDOS2019_FILE_PATTERN` in `src/config.py`).
- Other files that come with these downloads (`.arff` copies, `KDDTrain+_20Percent`,
  `KDDTest-21`, `UNSW_NB15_training-set.csv` / `testing-set.csv`) are ignored by the
  loaders. Merging the NSL-KDD copies and subsets was the cause of a 45% train/test
  leak earlier in the project.
- `python -m unittest tests.test_data_validation` checks the data once it is in place.

Please cite the original dataset papers (see the README's references) when
using the data.
