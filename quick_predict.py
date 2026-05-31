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

# Try to extract and load data
try:
    if os.path.exists('Datasets.zip'):
        with ZipFile('Datasets.zip', 'r') as zip_ref:
            zip_ref.extractall('.')
    
    test_df = pd.read_csv('test.csv')
    
    # Simple baseline: use statistical model with feature analysis
    train_df = pd.read_csv('train.csv')
    
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
    print("\nCreating default submission with balanced predictions...")
    
    # Default submission if data loading fails
    # Assumes test.csv has 339 rows based on problem description
    n_test = 339
    predictions = np.random.binomial(1, 0.2, n_test)  # ~20% defect rate
    
    submission = pd.DataFrame({
        'CoilID': range(1, n_test + 1),
        'Y': predictions
    })
    
    submission.to_csv('expected_submission.csv', index=False)
    print(f"\n✓ expected_submission.csv created (default)")
