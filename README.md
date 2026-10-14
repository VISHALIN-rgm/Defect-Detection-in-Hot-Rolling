# Defect Detection in Hot Rolling

Binary classification of hot rolling steel coils: predict whether a coil has an
**Alpha defect** from 49 process measurements. The project trains a model on the
provided training data and writes predictions for the test set.

## Problem statement

| `Y` | Meaning |
|-----|---------|
| `1` | Coil has an Alpha defect |
| `0` | Coil has no defect |

Each coil is identified by `CoilID` and described by numeric features `X1`–`X49`.
The goal is a `Y` prediction for every coil in the test set.

**Evaluation emphasis:** recall first (ideally no missed defects / 0 false
negatives), with precision also important (target ≈ 90%).

## Repository contents

| File | Purpose |
|------|---------|
| `Datasets.zip` | Dataset archive (contains a `dataset/` folder) |
| `solution.py` | Full ML pipeline (preprocessing, 4 models, threshold tuning, ensemble) |
| `quick_predict.py` | Fast, model-free baseline submission |
| `data_utils.py` | Shared, robust data loader (extracts the zip if needed) |
| `expected_submission.csv` | Submission format example; also the default output of both scripts |
| `requirements.txt` | Python dependencies |

### Dataset

`Datasets.zip` contains:

- `dataset/train.csv` – `CoilID`, `X1`…`X49`, `Y` (labelled)
- `dataset/test.csv` – `CoilID`, `X1`…`X49` (no `Y`)
- `dataset/sample_submission.csv` – submission format example

The scripts extract the archive automatically on first run. They also work if
`train.csv` / `test.csv` are placed directly in the repository root.

## Setup

Python 3.9+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run from the repository root.

```bash
# Full pipeline: trains XGBoost, Random Forest, Gradient Boosting and
# Logistic Regression, tunes thresholds, ensembles, writes the submission.
python solution.py

# Quick baseline (no model training)
python quick_predict.py
```

Both scripts write `expected_submission.csv`, **overwriting** the existing file.
Copy it first if you want to keep the original example.

## Submission format

CSV with a header and one row per test coil:

```csv
CoilID,Y
711,0
1542,1
```

## Notes: class imbalance and recall

- Defects are the minority class. `solution.py` handles this with
  `scale_pos_weight` (XGBoost) and `class_weight='balanced'` (Random Forest,
  Logistic Regression).
- Because missing a defect is costly, decision thresholds are tuned to maximize
  recall while keeping precision ≥ 85% on the training data, and the ensemble
  uses a lowered threshold (0.45).
- Metrics printed by `solution.py` are computed on the training data, so they
  are optimistic; use cross-validation or a hold-out split for an honest estimate.
- `quick_predict.py` is only a rough baseline and predicts many false positives.
