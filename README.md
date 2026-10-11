# Financial Fraud Detection with Machine Learning

## A Comparative Study of Fraud Detection Models under Class Imbalance and Data Constraints

**Author:** Nouha Laataoui  
**Degree:** MSc Data Analytics, Intelligence and Security — Rennes School of Business  
**Supervisor:** Professor Nadjib Brahimi  
**Academic year:** 2025–2026

## Project overview

This graduation project compares machine learning approaches to financial fraud detection across three settings: credit card transactions, e-commerce payments and Bitcoin transactions. It examines how class imbalance, data availability, heterogeneous features, probability calibration and temporal distribution shifts affect model performance.

Four model families are evaluated: **Logistic Regression, Random Forest, XGBoost and LightGBM**. Beyond predictive accuracy, the study considers interpretability, training cost, robustness and the reliability of predicted probabilities.

The experimental workflow includes leakage-safe preprocessing, validation-based threshold selection, imbalance handling, bootstrap confidence intervals, statistical comparisons, SHAP explanations and temporal evaluation.

> **Documentation status:** This README is based on the graduation report. The code was not inspected when this document was prepared. The directory layout and notebook names below are suggested conventions, not confirmed existing files. Execution requires the author's actual notebooks/scripts and datasets; the report alone is not executable. Adapt paths and names to the uploaded code before treating this as a fully verified reproduction guide.

## Contents

- [Datasets](#datasets)
- [Methodology](#methodology)
- [Key findings](#key-findings)
- [Suggested repository structure](#suggested-repository-structure)
- [Step-by-step usage guide](#step-by-step-usage-guide)
- [Reproducibility notes](#reproducibility-notes)
- [Limitations](#limitations)
- [Citation and reuse](#citation-and-reuse)

## Datasets

Download the datasets separately from the sources cited in the report. Raw datasets are not bundled with this documentation. Check each provider's access conditions and redistribution terms before sharing data.

| Phase | Dataset | Domain | Observations used | Positive-class rate | Feature information |
| --- | --- | --- | ---: | ---: | --- |
| 1 | Credit Card Fraud Detection | Banking | 283,726 after duplicate removal | Approximately 0.17% | 30 source predictors: V1–V28, Amount and Time; Time used for exploration only |
| 2 | IEEE-CIS Fraud Detection | E-commerce | 590,540 labelled transactions | 3.499% | Transaction and identity tables; 627 features after preprocessing, as reported |
| 3 | Elliptic Bitcoin Dataset | Cryptocurrency | 46,564 labelled transactions | 9.76% illicit | 165 anonymised features: 94 local and 71 aggregated; 49 time steps |

### Phase 1 — Credit Card Fraud Detection

**Source:** [ULB Credit Card Fraud Detection on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- Expected input: `creditcard.csv`.
- Original dataset: 284,807 transactions, including 492 fraud cases.
- Target: `Class`, with `1` for fraud and `0` for legitimate transactions.
- Remove exact duplicate rows before partitioning; the report retains 283,726 rows.
- V1–V28 are anonymised PCA components. The report applies a log transformation to Amount and retains Time for exploratory analysis only.
- Recalculate the positive-class count after deduplication rather than reusing the original count of 492.

### Phase 2 — IEEE-CIS Fraud Detection

**Source:** [IEEE-CIS Fraud Detection competition on Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection)

- Expected labelled inputs: `train_transaction.csv` and `train_identity.csv`.
- Join on `TransactionID`, retaining transaction rows without identity records through a left join.
- Target: `isFraud`, with `1` for fraud and `0` for legitimate transactions.
- Identity records cover approximately 24.42% of transactions. Missing identity information is therefore a central feature of this phase.
- Fit missingness rules, imputation, encoding, frequency mappings and scaling on the training partition only.
- The competition's unlabelled test files are not substitutes for the labelled held-out test partition used in this study.
- Kaggle may require sign-in and acceptance of competition rules before download.

### Phase 3 — Elliptic Bitcoin Dataset

**Source:** [Elliptic Dataset on Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)

- Expected inputs: `elliptic_txs_features.csv` and `elliptic_txs_classes.csv`.
- The download also provides `elliptic_txs_edgelist.csv`; the reported models use tabular features, rather than a graph neural network.
- Match feature rows and labels using transaction identifiers. Confirm headers and column order when loading the CSV files.
- Original dataset: 203,769 transactions. Use only the 46,564 labelled transactions; exclude unknown labels.
- Map the source labels explicitly: `1` (illicit) → positive class `1`; `2` (licit) → negative class `0`. Do not map unknown transactions to legitimate transactions.
- There are 4,545 illicit transactions in the labelled subset.
- Preserve the time-step column for chronological evaluation, separately from the predictor selection defined in the notebooks.

## Methodology

### Evaluation protocol

1. Inspect the data, labels, duplicates, missingness and class distribution.
2. Create stratified **70% training / 15% validation / 15% test** partitions for random-split benchmarks.
3. Fit preprocessing exclusively on the training partition and apply the frozen transformations to validation and test data.
4. Apply resampling only to training data, where required.
5. Train each model and choose its classification threshold by maximising F1 on validation data.
6. Apply the selected threshold unchanged to the held-out test set.
7. Evaluate discrimination, calibration, computation time and robustness.
8. For Elliptic, additionally train on time steps **1–29** and evaluate on steps **30–49**. Threshold selection and calibration must use historical validation data without accessing the future test period; follow the actual code for the historical validation allocation.

### Model configuration reported in the study

| Model | Main settings |
| --- | --- |
| Logistic Regression | `C=0.1`, `class_weight='balanced'`, `solver='liblinear'` |
| Random Forest | 100 trees, `max_depth=10`, `min_samples_split=50`, `min_samples_leaf=20` |
| XGBoost | 150 estimators, `max_depth=4`, `learning_rate=0.1`; training-derived `scale_pos_weight` |
| LightGBM | Phases 1 and 3: 150 estimators, `learning_rate=0.05`, `num_leaves=31`; Phase 2: 1,000 estimators, `num_leaves=127` |

These are the main settings described in the report, not a complete estimator specification. Consult the code for remaining parameters and any pipeline-specific overrides.

### Imbalance strategies

Phases 1 and 3 use class weighting. Phase 2 compares:

| Pipeline | Strategy | Training rows reported |
| --- | --- | ---: |
| A | Class weighting without resampling | 413,378 |
| B | Random undersampling to a balanced 1:1 ratio | 28,928 |
| C | Random undersampling followed by BorderlineSMOTE; final fraud prevalence 20% | 180,800 |
| D | LightGBM `is_unbalance` / Balanced Random Forest variants | Depends on estimator |

### Metrics and additional analyses

- **Primary metric:** PR-AUC, calculated by trapezoidal integration of the precision–recall curve.
- **Also reported:** Average Precision, F1, precision, recall and ROC-AUC. Average Precision and trapezoidal PR-AUC are distinct computations; label them separately.
- **Calibration:** Expected Calibration Error with 15 equal-width bins, Brier score and reliability diagrams; post-hoc isotonic calibration in Phase 2.
- **Uncertainty:** 95% percentile bootstrap intervals with 300 resamples.
- **Robustness:** repeated seeds; paired t-tests and Wilcoxon signed-rank tests, with Bonferroni correction.
- **Sensitivity:** training fractions of 1%, 5%, 10%, 25%, 50% and 100%; evaluation fraud prevalence of 0.1%, 0.5%, 1% and 5%.
- **Explainability:** SHAP summary plots and feature importance.
- **Efficiency:** loading/preprocessing caches and training/inference timings.

## Key findings

The following values are results reported in the dissertation, not independently rerun for this README.

| Experiment | Reported result | Interpretation |
| --- | --- | --- |
| Phase 1, XGBoost | PR-AUC **0.8088** | Strong discrimination under extreme class imbalance |
| Phase 2, LightGBM Pipeline A | PR-AUC **0.8171** | Higher than XGBoost Pipeline A (**0.7325**) in this experiment |
| Phase 2, LightGBM Pipeline C | PR-AUC **0.8075** | Class weighting outperforms RUS + BorderlineSMOTE in the repeated A/C comparison; Wilcoxon **p=0.0020** |
| Phase 2, isotonic calibration of Pipeline A | ECE **0.0582 → 0.0005**; PR-AUC **0.8171 → 0.8161** | Calibration improves with a small change in discrimination |
| Phase 3, LightGBM | Random-split PR-AUC **0.9848**; temporal PR-AUC **0.8679** | Approximately **0.117** absolute degradation under future-period evaluation |

Pipeline D achieves a reported single-run Phase 2 PR-AUC of **0.8231**, but was not evaluated over repeated seeds. Consequently, Pipeline A should not be described as having the highest single-run PR-AUC across all four pipelines; its advantage over Pipeline C is supported by repeated experiments.

Across these experiments, boosting models generally perform strongly, while data complexity, calibration and temporal shift materially affect their usefulness. Cross-dataset differences do not establish a causal effect of fraud prevalence alone.

## Suggested repository structure

The following paths are proposed. Rename the actual files consistently or adapt this guide to their existing names.

```text
financial-fraud-detection-ml/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── phase1_credit_card_fraud.ipynb
│   ├── phase2_ieee_cis.ipynb
│   └── phase3_elliptic_bitcoin.ipynb
├── src/                         # Optional: reusable code extracted from notebooks
├── data/
│   ├── README.md                # Dataset sources and access instructions
│   └── raw/
│       ├── creditcard/creditcard.csv
│       ├── ieee_cis/train_transaction.csv
│       ├── ieee_cis/train_identity.csv
│       ├── elliptic/elliptic_txs_features.csv
│       ├── elliptic/elliptic_txs_classes.csv
│       └── elliptic/elliptic_txs_edgelist.csv
├── results/
│   ├── figures/
│   └── metrics/
├── cache/                       # Local generated files, excluded from Git
└── docs/
    └── graduation_report.pdf    # If the author chooses to include the report
```

## Step-by-step usage guide

### Step 1 — Obtain the repository and code

Once the repository is published, replace `YOUR-USERNAME` with its owner's actual GitHub username:

```bash
git clone https://github.com/YOUR-USERNAME/financial-fraud-detection-ml.git
cd financial-fraud-detection-ml
```

Alternatively, download and extract the repository ZIP. Verify that the actual notebooks and dependency file are included. If only this README or the report is present, the experimental code must first be added by the author.

### Step 2 — Create an isolated Python environment

The study reports **Python 3.10**. A machine with **16 GB RAM** is the reported reference environment; Phase 2 is the most memory-intensive phase. A GPU is not required by the reported workflow.

Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell activation is blocked by local policy, use Command Prompt and activate with `.venv\Scripts\activate.bat`, or invoke `.venv\Scripts\python.exe` directly.

macOS / Linux:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Step 3 — Install dependencies

If the repository contains the author's verified dependency file:

```bash
python -m pip install -r requirements.txt
python -m pip check
```

The report's Appendix E lists these versions:

| Component | Reported version |
| --- | --- |
| Python | 3.10 |
| NumPy | 1.26.4 |
| pandas | 2.0 |
| scikit-learn | 1.3 |
| XGBoost | 2.0 |
| LightGBM | 4.6.0 |
| imbalanced-learn | 0.11 |
| SHAP | 0.46.0 |
| matplotlib | 3.7 |
| seaborn | 0.12 |
| joblib | 1.3 |

The report does not specify every patch version, and its main text lists NumPy 1.24 instead of Appendix E's 1.26.4. The actual working environment should determine the final dependency pins.

If `requirements.txt` is absent, the following is a **starting environment specification to verify against the code**, not a validated lockfile:

```bash
python -m pip install "numpy==1.26.4" "pandas==2.0.*" "scikit-learn==1.3.*" "xgboost==2.0.*" "lightgbm==4.6.0" "imbalanced-learn==0.11.*" "shap==0.46.0" "matplotlib==3.7.*" "seaborn==0.12.*" "joblib==1.3.*" scipy jupyterlab ipykernel
python -m pip check
```

After successfully running the notebooks, record the working package versions in the repository. Additional imports in the actual code may require additional dependencies.

### Step 4 — Download and arrange the data

1. Open the three dataset source links above.
2. Download and extract the required CSV files.
3. Place them in the proposed `data/raw/` folders, or the locations expected by the actual notebooks.
4. Check that downloads are complete and that CSV files are not still inside ZIP archives.
5. Confirm the target columns and Elliptic label mapping before training.

Never commit Kaggle credentials, API tokens or raw datasets into the repository.

### Step 5 — Adapt file paths and outputs

Open the configuration/loading cells in each notebook. Replace machine-specific paths with paths to your local data. For example, **when the current working directory is the repository root**:

```python
from pathlib import Path

PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "data" / "raw"
CREDITCARD_PATH = DATA_DIR / "creditcard" / "creditcard.csv"
IEEE_TRANSACTION_PATH = DATA_DIR / "ieee_cis" / "train_transaction.csv"
IEEE_IDENTITY_PATH = DATA_DIR / "ieee_cis" / "train_identity.csv"
ELLIPTIC_FEATURES_PATH = DATA_DIR / "elliptic" / "elliptic_txs_features.csv"
ELLIPTIC_CLASSES_PATH = DATA_DIR / "elliptic" / "elliptic_txs_classes.csv"

for directory in ("results/figures", "results/metrics", "cache"):
    (PROJECT_ROOT / directory).mkdir(parents=True, exist_ok=True)
```

Check `Path.cwd()` inside the notebook: if it points to `notebooks/`, use its parent as `PROJECT_ROOT`. This snippet illustrates path configuration; it does not imply the existing code uses these variable names.

Set the primary seed to **42** and preserve the reported split proportions. Clear or rebuild caches if data, preprocessing or split settings change.

### Step 6 — Start Jupyter with the correct environment

```bash
python -m ipykernel install --user --name fraud-detection --display-name "Python 3.10 - Fraud Detection"
python -m jupyterlab
```

Select **Python 3.10 - Fraud Detection** as the notebook kernel. Start with one phase and restart its kernel before running all cells from top to bottom, so execution does not depend on stale variables.

### Step 7 — Run Phase 1: credit card benchmark

Open the actual Phase 1 notebook (suggested name: `notebooks/phase1_credit_card_fraud.ipynb`).

1. Load the data and inspect `Class`.
2. Remove exact duplicates and check the resulting row/label counts.
3. Create stratified training, validation and test partitions.
4. Apply the reported Amount transformation and training-fitted preprocessing; keep Time exploratory.
5. Train the four model families with the code's class-weighting settings.
6. Select thresholds on validation data and evaluate on test data.
7. Generate precision–recall curves, calibration plots and SHAP analyses where implemented.
8. Save model metrics and figures.

### Step 8 — Run Phase 2: IEEE-CIS benchmark

Open the actual Phase 2 notebook (suggested name: `notebooks/phase2_ieee_cis.ipynb`). Use the final improved experimental configuration if both original and improved versions are supplied.

1. Left-join labelled transaction and identity data on `TransactionID` and check that the join does not multiply transaction rows.
2. Split the labelled data before fitting preprocessing.
3. Fit imputers, missingness rules and encoders on training data only; transform validation and test data without refitting.
4. Train Pipeline A first as the baseline.
5. Run Pipelines B and C on the same partitions, applying RUS/BorderlineSMOTE only to training data.
6. Evaluate Pipeline D separately where implemented.
7. Fit isotonic calibration using validation data, never the test set; check the notebook's allocation of validation data for calibration and threshold selection.
8. Run the repeated-seed A/B/C comparisons and statistical tests.
9. Export discrimination, calibration, timing and interpretability results.

The reported 627-feature matrix is a reference check. A mismatch should prompt inspection of preprocessing settings, category handling and input data; do not force the dimensionality by arbitrary column removal.

### Step 9 — Run Phase 3: Elliptic benchmark

Open the actual Phase 3 notebook (suggested name: `notebooks/phase3_elliptic_bitcoin.ipynb`).

1. Load features and labels, join by transaction identifier and exclude unknown labels.
2. Verify the binary illicit/licit mapping and labelled subset size.
3. Run the stratified random-split benchmark.
4. Run the temporal experiment using steps 1–29 for historical training and steps 30–49 for future evaluation.
5. Keep preprocessing fitting, calibration and threshold selection within historical data.
6. Compare random and temporal PR-AUC and document the absolute difference.
7. Export learning curves and sensitivity analyses where implemented.

### Step 10 — Inspect and interpret the outputs

Depending on the supplied code, retain:

- Metrics by phase, model, pipeline, seed and split strategy.
- Validation-selected thresholds and test precision/recall/F1.
- Precision–recall curves and bootstrap confidence intervals.
- Calibration diagrams, ECE and Brier scores.
- SHAP plots, learning curves and prevalence sensitivity results.
- Training/inference timings and random-versus-temporal comparisons.

Suggested export locations are `results/metrics/` and `results/figures/`. Configure explicit saving in the notebooks if they currently only display outputs.

Compare results with the report using matching settings. Differences can arise from package versions, data processing, seed configuration and threshold/calibration choices. Hardware timings are reference measurements, not expected guarantees.

### Step 11 — Reuse or extend the code

To apply the workflow to another dataset:

1. Define its positive class and remove target-derived predictors.
2. Choose a split that matches the intended deployment; use chronological evaluation where time matters.
3. Adapt preprocessing to its feature types and fit it on training data only.
4. Start with a simple baseline and class weighting before adding resampling.
5. Select thresholds and fit calibration on held-out validation data.
6. Evaluate the untouched test set with both discrimination and calibration metrics.
7. Record all data, configuration and environment changes.

Possible extensions include cost-sensitive threshold selection, rolling temporal evaluation, graph-based models for Elliptic and drift monitoring. These are extension ideas, not implemented features confirmed by this README.

## Reproducibility notes

- **Primary seed:** 42.
- **Repeated experiments:** Appendix A specifies 5 runs for Phases 1 and 3, and 10 for the improved Phase 2 framework. The main methodology describes 10 runs more broadly; use Appendix A as the documented phase-level breakdown and confirm exact seeds from the code.
- **Phase 2 Pipeline D:** single configuration, without repeated-seed evaluation.
- **Bootstrap:** 300 resamples, 95% percentile intervals. Handle resamples lacking both classes explicitly.
- **Statistical thresholds:** Phase 2 Bonferroni alpha approximately 0.0083 for six model comparisons and 0.0167 for three pipeline comparisons.
- **Leakage prevention:** no full-dataset fitting of preprocessing, no validation/test resampling, and no test-based threshold or calibration fitting.
- **Cache validity:** separate caches by data version, split/seed and preprocessing configuration. Build pickle/joblib caches locally from trusted data.
- **Reference hardware:** Windows 10, Intel Core i7, 16 GB RAM.
- **Reference optimisations:** the report describes approximately 15× faster loading with pickle and a reduction from 170.9 s to 0.09 s for cached preprocessing retrieval. These measure repeated cache reuse, not first-run training acceleration.

For publication, exclude local raw data, virtual environments, generated caches and credentials from version control. A suitable starting `.gitignore` includes:

```gitignore
.venv/
__pycache__/
.ipynb_checkpoints/
data/raw/
cache/
.env
kaggle.json
```

## Troubleshooting

| Problem | Check or action |
| --- | --- |
| `FileNotFoundError` | Check extracted CSV names, configured paths and notebook working directory |
| `ModuleNotFoundError` | Install dependencies in the environment used by the selected kernel |
| Dependency/API errors | Compare installed versions with the author's working environment; restart the kernel after changes |
| Phase 2 runs out of memory | Use the documented dtype/caching optimisations, release unused objects and run phases separately; reduced samples are exploratory runs, not full reproduction |
| Elliptic labels are incorrect | Check source label mapping and exclusion of unknown transactions |
| Results differ from the report | Check data cleaning, splits, seeds, preprocessing, calibration and metric definitions |
| Cached results do not change | Invalidate the relevant cache and rerun from a clean kernel |

## Limitations

This is an academic benchmark study, not a production fraud-detection service. Anonymisation limits domain interpretation, the labelled Elliptic subset introduces selection constraints, and random-split performance may overstate future-period performance. The datasets differ in more than class prevalence, so cross-phase comparisons do not isolate a causal imbalance effect. Reported model settings are representative configurations rather than evidence of exhaustive hyperparameter optimisation.

The report also contains some inconsistent descriptions of counts, repetition settings and feature breakdowns. For exact reproduction, the final code, data checks and recorded outputs must resolve these differences; this README avoids treating ambiguous details as verified implementation facts.

## Citation and reuse

If you refer to this work, cite:

> Laataoui, N. (2026). *A Comparative Study of Fraud Detection Models under Class Imbalance and Data Constraints*. MSc graduation project, Rennes School of Business. Supervisor: Nadjib Brahimi.

Credit the original dataset providers separately. Consult the repository's `LICENSE` file for code reuse permissions. This README does not grant a license to the code or datasets.
