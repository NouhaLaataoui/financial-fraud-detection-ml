# Datasets — Download and Preparation Guide

This project studies financial fraud detection using three public benchmark datasets. Download the data separately: raw datasets and generated preprocessing caches are excluded from this repository.

The instructions below describe the inputs documented in the graduation report. Check the loading cells in the actual notebooks for their precise paths and preprocessing implementation.

## 1. Credit Card Fraud Detection — Phase 1

**Download:** [ULB Credit Card Fraud Detection on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

**Required file:** `creditcard.csv`

- Original data: 284,807 transactions, including 492 fraud cases.
- Target: `Class` — `1` indicates fraud; `0` indicates a legitimate transaction.
- Source predictors: anonymised PCA components `V1`–`V28`, `Amount` and `Time`.
- The study removes 1,081 exact duplicate rows, retaining 283,726 transactions. Recalculate class counts after deduplication.
- The report log-transforms Amount and uses Time for exploration only.

### Preparation

1. Download and extract `creditcard.csv`.
2. Place it in the local Phase 1 data folder.
3. Update the Phase 1 notebook's input path.
4. Check the target and duplicate counts before splitting.
5. Fit any learned preprocessing only on training data.

## 2. IEEE-CIS Fraud Detection — Phase 2

**Download:** [IEEE-CIS Fraud Detection competition on Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection)

**Required files:**

- `train_transaction.csv`
- `train_identity.csv`

The labelled transaction data contains 590,540 rows, with approximately 3.499% fraud. The target is `isFraud`: `1` for fraud and `0` for legitimate transactions. Identity records are available for approximately 24.42% of transactions.

### Preparation

1. Sign in to Kaggle and follow the competition's access requirements.
2. Download and extract the two labelled training CSV files.
3. Place them in the local Phase 2 data folder and update notebook paths.
4. Left-join identity information to transaction data using `TransactionID`, retaining transactions without an identity record.
5. Verify that the join does not create duplicate transaction rows.
6. Split the labelled data before fitting imputation, encoding, frequency mappings or scaling.
7. Fit preprocessing on training data only and apply it unchanged to validation and test partitions.
8. Apply undersampling and BorderlineSMOTE only to the training partition where the selected pipeline requires them.

The study creates its validation and test partitions from the labelled training data. The competition's unlabelled `test_transaction.csv` and `test_identity.csv` are not substitutes for the labelled test partition used for evaluation.

## 3. Elliptic Bitcoin Dataset — Phase 3

**Download:** [Elliptic Dataset on Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)

**Required tabular inputs:**

- `elliptic_txs_features.csv`
- `elliptic_txs_classes.csv`

The download also includes `elliptic_txs_edgelist.csv`. It describes graph edges; the reported experiments use tabular models rather than graph neural networks.

The full dataset contains 203,769 transactions across 49 time steps. The study uses only the 46,564 labelled transactions, of which 4,545 are illicit (approximately 9.76%). It describes 165 anonymised features, comprising 94 local and 71 aggregated features.

### Preparation

1. Download and extract the dataset.
2. Place the CSV files in the local Phase 3 data folder and update notebook paths.
3. Check whether the feature CSV has headers before selecting columns.
4. Join features and classes by transaction identifier.
5. Exclude transactions with unknown labels; do not treat them as legitimate.
6. Map source labels explicitly:

| Source label | Meaning | Binary target |
| --- | --- | ---: |
| `1` | Illicit | 1 |
| `2` | Licit | 0 |
| `unknown` | Unlabelled | Exclude |

7. Preserve the time-step column for chronological partitioning, separately from the notebook's predictor selection.
8. For the temporal experiment, use steps 1–29 as the historical period and steps 30–49 for future evaluation. Keep preprocessing fitting, calibration and threshold selection within historical data. Refer to the code for the historical validation allocation.

## Suggested local folder layout

These subfolder names are suggestions. Adapt them to the actual notebook paths; existing data does not need to be moved if you configure the paths correctly.

```text
Data/
├── README.md
├── creditcard/
│   └── creditcard.csv
├── ieee_cis/
│   ├── train_transaction.csv
│   └── train_identity.csv
└── elliptic/
    ├── elliptic_txs_features.csv
    ├── elliptic_txs_classes.csv
    └── elliptic_txs_edgelist.csv
```

`Data/` is capitalised to match this repository. File and folder names may be case-sensitive on other operating systems.

## Configure input paths

For a notebook located directly inside `Notebooks/`, the following illustrates how to find the repository and data folders:

```python
from pathlib import Path

current_dir = Path.cwd()
project_root = current_dir.parent if current_dir.name == "Notebooks" else current_dir
data_dir = project_root / "Data"

creditcard_path = data_dir / "creditcard" / "creditcard.csv"
ieee_transaction_path = data_dir / "ieee_cis" / "train_transaction.csv"
ieee_identity_path = data_dir / "ieee_cis" / "train_identity.csv"
elliptic_features_path = data_dir / "elliptic" / "elliptic_txs_features.csv"
elliptic_classes_path = data_dir / "elliptic" / "elliptic_txs_classes.csv"
```

This is an example, not a confirmed configuration API in the supplied notebooks. Check `Path.cwd()` and adapt the loading cells and variable names accordingly. For nested notebooks, set the repository root explicitly.

## Before running experiments

- Extract archives and verify the expected CSV filenames.
- Check row counts, target values, class balance and identifier uniqueness.
- Remove exact duplicates before partitioning where the phase requires it.
- Use the reported stratified 70/15/15 training/validation/test split for random benchmarks and primary seed 42.
- Keep validation/test sets at their original class distribution for baseline evaluation.
- Select thresholds and fit calibration on validation data, not test data.
- Rebuild caches when data, splits or preprocessing settings change.
- Consult the root README for environment setup, models, metrics and execution order.

## Version control and data access

Keep raw datasets local and share source links rather than redistributing CSV files. Dataset access and reuse remain subject to the providers' terms; a code license does not override dataset terms.

The repository currently excludes `/Data/` in `.gitignore`. The author can publish this documentation alone using:

```bash
git add -f Data/README.md
git commit -m "Add dataset download instructions"
git push origin main
```

Do not force-add the entire Data folder. Do not publish Kaggle credentials, API tokens, local preprocessing caches or environment secrets.
