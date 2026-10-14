# Defect Detection in Hot Rolling

This project predicts whether a hot-rolled steel coil has an Alpha defect. It is a
binary classification task:

- `Y = 1`: defect present
- `Y = 0`: no defect

The repository includes a full model-training pipeline and a lightweight heuristic
baseline to generate a submission.

## Dataset

`Datasets.zip` contains:

- `dataset/train.csv`: 1,352 labeled coils, with `CoilID`, 49 feature columns
  (`X1`–`X49`), and target `Y`
- `dataset/test.csv`: 339 coils with the same identifier and feature columns, but
  without `Y`
- `dataset/sample_submission.csv`: a small example of the submission format

The training data is imbalanced: 66 of 1,352 coils are labeled as defects (about
4.9%). Both scripts can extract the archive automatically when the CSV files are
not already available. They also support `train.csv` and `test.csv` placed directly
in the repository root.

## Setup

Use Python 3.9 or newer, then install the dependencies from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Run

Run commands from the repository root.

The full pipeline preprocesses the data, trains and evaluates several classifiers,
and writes `expected_submission.csv`:

```bash
python solution.py
```

The quick baseline uses a simple feature-sum heuristic and writes the same output
file; it is useful for checking the data/submission workflow, not as a substitute
for model validation:

```bash
python quick_predict.py
```

Either script overwrites `expected_submission.csv`. The checked-in file is an
example submission and is not model output from a run.

## Submission format

The output CSV must contain exactly one row per test coil and these columns in this
order:

```csv
CoilID,Y
```

Keep the test set's `CoilID` values and order. `Y` must be a binary prediction (`0`
or `1`). The scripts generate this format automatically.

## Evaluation note

Recall is important because a false negative means a defective coil was missed.
However, the positive class is only about 4.9% of the training set, so a model can
appear accurate while missing defects. Consider recall together with precision and
the false-positive/false-negative counts; the scripts' local threshold and
training-set metrics do not guarantee a particular score on unseen data.
