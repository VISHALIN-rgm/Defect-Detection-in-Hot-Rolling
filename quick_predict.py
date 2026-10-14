"""
Quick baseline: generate expected_submission.csv without training a model.

Uses a simple normalized feature-sum threshold. For the full ML pipeline run
solution.py instead.
"""

import numpy as np
import pandas as pd

from data_utils import load_train_test

OUTPUT_FILE = 'expected_submission.csv'


def main():
    print("Generating baseline predictions...")
    train_df, test_df = load_train_test()

    X_train = train_df.drop(['CoilID', 'Y'], axis=1)
    y_train = train_df['Y'].values
    X_test = test_df[X_train.columns]
    test_coil_ids = test_df['CoilID'].values

    # Impute with training means so NaNs do not break the sums
    means = X_train.mean()
    train_sum = X_train.fillna(means).sum(axis=1).values
    test_sum = X_test.fillna(means).sum(axis=1).values

    std = train_sum.std() or 1.0
    train_norm = (train_sum - train_sum.mean()) / std
    test_norm = (test_sum - train_sum.mean()) / std

    # Low threshold (30th percentile of defect scores) favours recall
    threshold = np.percentile(train_norm[y_train == 1], 30)
    predictions = (test_norm > threshold).astype(int)

    submission = pd.DataFrame({'CoilID': test_coil_ids, 'Y': predictions})
    submission.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✓ {OUTPUT_FILE} created successfully!")
    print(f"  Total rows: {len(submission)}")
    print(f"  Defects (Y=1): {predictions.sum()}")
    print(f"  No Defects (Y=0): {len(predictions) - predictions.sum()}")


if __name__ == "__main__":
    main()
