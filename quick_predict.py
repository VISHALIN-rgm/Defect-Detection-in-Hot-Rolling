"""
Generate sample expected_submission.csv file
"""

import pandas as pd
import numpy as np
import warnings
from zipfile import ZipFile
import os

warnings.filterwarnings('ignore')

print("Generating sample predictions...")

# Load data from either the repository root or the archive's dataset/ directory.
try:
    if not (os.path.exists('train.csv') and os.path.exists('test.csv')) and not (
        os.path.exists(os.path.join('dataset', 'train.csv')) and
        os.path.exists(os.path.join('dataset', 'test.csv'))
    ):
        with ZipFile('Datasets.zip', 'r') as zip_ref:
            zip_ref.extractall('.')
    
    data_dir = '.' if os.path.exists('train.csv') and os.path.exists('test.csv') else 'dataset'
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    
    # Simple baseline: use statistical model with feature analysis
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    
    # Get basic statistics to inform predictions
    defect_rate = train_df['Y'].mean()
    
    # Create predictions based on ensemble of simple rules
    X_train = train_df.drop(['CoilID', 'Y'], axis=1)
    y_train = train_df['Y'].values
    
    X_test = test_df.drop(['CoilID'], axis=1)
    test_coil_ids = test_df['CoilID'].values
    
    # Simple predictive features: sum of process parameters
    train_feature_sum = X_train.sum(axis=1).values
    test_feature_sum = X_test.sum(axis=1).values
    
    # Normalize
    train_feature_sum_norm = (train_feature_sum - train_feature_sum.mean()) / train_feature_sum.std()
    test_feature_sum_norm = (test_feature_sum - train_feature_sum.mean()) / train_feature_sum.std()
    
    # Generate predictions with threshold optimization
    threshold = np.percentile(train_feature_sum_norm[y_train == 1], 30)
    predictions = (test_feature_sum_norm > threshold).astype(int)
    
    # Create submission
    submission = pd.DataFrame({
        'CoilID': test_coil_ids,
        'Y': predictions
    })
    
    submission.to_csv('expected_submission.csv', index=False)
    
    print(f"\n✓ expected_submission.csv created successfully!")
    print(f"  Total rows: {len(submission)}")
    print(f"  Defects (Y=1): {sum(predictions)}")
    print(f"  No Defects (Y=0): {len(predictions) - sum(predictions)}")
    print(f"\nFirst 10 predictions:")
    print(submission.head(10).to_string(index=False))
    
except Exception as e:
    print(f"Error: {e}")
    raise
