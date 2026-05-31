"""
Defect Detection in Hot Rolling - Machine Learning Solution
================================================================================
Problem: Detect Alpha defects (1) vs No Defect (0) in hot rolling steel
Metric: Recall ≥ 100%, Precision ≥ 90% (0 false negatives allowed)
Score: Max 100 (based on false negatives and false positives)
================================================================================
"""

import pandas as pd
import numpy as np
import warnings
from zipfile import ZipFile
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_prepare_data():
    """Load data from zip file and prepare for modeling"""
    
    print("=" * 80)
    print("STEP 1: LOADING AND PREPARING DATA")
    print("=" * 80)
    
    # Extract zip file if not already extracted
    if not os.path.exists('train.csv') or not os.path.exists('test.csv'):
        print("\nExtracting Datasets.zip...")
        with ZipFile('Datasets.zip', 'r') as zip_ref:
            zip_ref.extractall('.')
    
    # Load datasets
    print("Loading training data...")
    train_df = pd.read_csv('train.csv')
    print(f"Train shape: {train_df.shape}")
    print(f"Train columns: {train_df.columns.tolist()}")
    
    print("\nLoading test data...")
    test_df = pd.read_csv('test.csv')
    print(f"Test shape: {test_df.shape}")
    print(f"Test columns: {test_df.columns.tolist()}")
    
    # Display basic statistics
    print("\n" + "=" * 80)
    print("DATA STATISTICS")
    print("=" * 80)
    print(f"\nClass distribution in training data:")
    print(train_df['Y'].value_counts())
    print(f"\nClass imbalance ratio: {train_df['Y'].value_counts()[0]} : {train_df['Y'].value_counts()[1]}")
    
    print(f"\nMissing values in train data:")
    missing_train = train_df.isnull().sum()
    print(missing_train[missing_train > 0])
    
    print(f"\nMissing values in test data:")
    missing_test = test_df.isnull().sum()
    print(missing_test[missing_test > 0])
    
    return train_df, test_df


# ============================================================================
# STEP 2: FEATURE ENGINEERING AND SCALING
# ============================================================================

def preprocess_features(train_df, test_df):
    """Preprocess features: handle missing values and scale"""
    
    print("\n" + "=" * 80)
    print("STEP 2: FEATURE ENGINEERING AND PREPROCESSING")
    print("=" * 80)
    
    # Separate features and target
    X_train = train_df.drop(['CoilID', 'Y'], axis=1)
    y_train = train_df['Y'].values
    X_test = test_df.drop(['CoilID'], axis=1)
    test_coil_ids = test_df['CoilID'].values
    
    print(f"\nFeatures shape: {X_train.shape}")
    print(f"Target distribution: Defect={sum(y_train==1)}, No Defect={sum(y_train==0)}")
    
    # Handle missing values with mean imputation
    print("\nHandling missing values...")
    feature_means = X_train.mean()
    X_train_filled = X_train.fillna(feature_means)
    X_test_filled = X_test.fillna(feature_means)
    
    print(f"Missing values after imputation - Train: {X_train_filled.isnull().sum().sum()}, Test: {X_test_filled.isnull().sum().sum()}")
    
    # Feature scaling using StandardScaler
    print("\nScaling features...")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_filled)
    X_test_scaled = scaler.transform(X_test_filled)
    
    print(f"Scaled train data shape: {X_train_scaled.shape}")
    print(f"Scaled test data shape: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, test_coil_ids, X_train.columns.tolist()


# ============================================================================
# STEP 3: FEATURE IMPORTANCE ANALYSIS
# ============================================================================

def analyze_feature_importance(X_train, y_train):
    """Analyze and visualize feature importance"""
    
    print("\n" + "=" * 80)
    print("STEP 3: FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)
    
    from sklearn.ensemble import RandomForestClassifier
    
    # Train a quick RF model for feature importance
    rf_quick = RandomForestClassifier(n_estimators=50, max_depth=10, 
                                     class_weight='balanced', random_state=42, n_jobs=-1)
    rf_quick.fit(X_train, y_train)
    
    # Get feature importances
    importances = rf_quick.feature_importances_
    feature_names = [f'X{i}' for i in range(1, len(importances) + 1)]
    
    # Sort and display top features
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 15 Most Important Features:")
    print(feature_importance_df.head(15).to_string(index=False))
    
    return feature_importance_df


# ============================================================================
# STEP 4: MODEL TRAINING
# ============================================================================

def train_models(X_train, y_train):
    """Train multiple models with optimized hyperparameters"""
    
    print("\n" + "=" * 80)
    print("STEP 4: TRAINING MULTIPLE MODELS")
    print("=" * 80)
    
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from xgboost import XGBClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_validate
    
    models = {}
    
    # Calculate class weights for imbalanced data
    n_neg = sum(y_train == 0)
    n_pos = sum(y_train == 1)
    class_weight_ratio = n_neg / n_pos
    
    print(f"\nClass weight ratio: {class_weight_ratio:.2f}")
    
    # 1. XGBoost with optimized parameters
    print("\n1. Training XGBoost...")
    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=class_weight_ratio,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    models['XGBoost'] = xgb_model
    print("   ✓ XGBoost trained")
    
    # 2. Random Forest
    print("2. Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    models['RandomForest'] = rf_model
    print("   ✓ Random Forest trained")
    
    # 3. Gradient Boosting
    print("3. Training Gradient Boosting...")
    gb_model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        min_samples_split=5,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    models['GradientBoosting'] = gb_model
    print("   ✓ Gradient Boosting trained")
    
    # 4. Logistic Regression (for comparison)
    print("4. Training Logistic Regression...")
    lr_model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    lr_model.fit(X_train, y_train)
    models['LogisticRegression'] = lr_model
    print("   ✓ Logistic Regression trained")
    
    return models


# ============================================================================
# STEP 5: MODEL EVALUATION AND THRESHOLD TUNING
# ============================================================================

def evaluate_models(models, X_train, y_train):
    """Evaluate models and tune threshold for optimal recall"""
    
    print("\n" + "=" * 80)
    print("STEP 5: MODEL EVALUATION AND THRESHOLD TUNING")
    print("=" * 80)
    
    from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
    from sklearn.model_selection import cross_val_predict
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n{model_name}:")
        print("-" * 40)
        
        # Get predictions
        y_pred = model.predict(X_train)
        
        # Calculate metrics
        recall = recall_score(y_train, y_pred)
        precision = precision_score(y_train, y_pred)
        f1 = f1_score(y_train, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
        
        print(f"  Recall:      {recall:.4f} (True Positive Rate)")
        print(f"  Precision:   {precision:.4f} (Positive Predictive Value)")
        print(f"  F1-Score:    {f1:.4f}")
        print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        
        # Get probability predictions for threshold tuning
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_train)[:, 1]
            
            # Find optimal threshold to maximize recall while maintaining precision > 90%
            best_threshold = 0.5
            best_recall = 0
            
            for threshold in np.arange(0.1, 0.9, 0.01):
                y_pred_threshold = (y_proba >= threshold).astype(int)
                if sum(y_pred_threshold) > 0:
                    rec = recall_score(y_train, y_pred_threshold)
                    prec = precision_score(y_train, y_pred_threshold)
                    
                    # Find threshold with highest recall and precision >= 85%
                    if prec >= 0.85 and rec > best_recall:
                        best_recall = rec
                        best_threshold = threshold
            
            print(f"  Optimal Threshold: {best_threshold:.3f}")
            y_pred_tuned = (y_proba >= best_threshold).astype(int)
            recall_tuned = recall_score(y_train, y_pred_tuned)
            precision_tuned = precision_score(y_train, y_pred_tuned)
            print(f"  Tuned Recall:    {recall_tuned:.4f}")
            print(f"  Tuned Precision: {precision_tuned:.4f}")
            
            results[model_name] = {
                'model': model,
                'threshold': best_threshold,
                'recall': recall_tuned,
                'precision': precision_tuned
            }
        else:
            results[model_name] = {
                'model': model,
                'threshold': 0.5,
                'recall': recall,
                'precision': precision
            }
    
    return results


# ============================================================================
# STEP 6: ENSEMBLE PREDICTION
# ============================================================================

def ensemble_predict(models, X_test, results):
    """Create ensemble predictions from multiple models"""
    
    print("\n" + "=" * 80)
    print("STEP 6: ENSEMBLE PREDICTION")
    print("=" * 80)
    
    # Collect predictions from all models
    all_predictions = []
    weights = []
    
    for model_name, model_info in results.items():
        model = model_info['model']
        threshold = model_info['threshold']
        
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= threshold).astype(int)
        else:
            y_pred = model.predict(X_test)
        
        # Weight based on precision and recall
        weight = (model_info['precision'] + model_info['recall']) / 2
        
        all_predictions.append(y_proba if hasattr(model, 'predict_proba') else y_pred)
        weights.append(weight)
        
        print(f"{model_name}: weight={weight:.4f}")
    
    # Weighted average ensemble
    weights = np.array(weights) / sum(weights)
    ensemble_proba = np.zeros_like(all_predictions[0])
    
    for i, pred in enumerate(all_predictions):
        if len(pred.shape) == 1:  # Binary predictions
            ensemble_proba += weights[i] * pred
        else:  # Probabilities
            ensemble_proba += weights[i] * pred
    
    # Use optimized threshold
    ensemble_threshold = 0.45
    ensemble_pred = (ensemble_proba >= ensemble_threshold).astype(int)
    
    print(f"\nEnsemble threshold: {ensemble_threshold}")
    print(f"Ensemble predictions distribution: {np.bincount(ensemble_pred)}")
    
    return ensemble_pred


# ============================================================================
# STEP 7: GENERATE SUBMISSION
# ============================================================================

def create_submission(predictions, test_coil_ids, output_file='expected_submission.csv'):
    """Create submission file in required format"""
    
    print("\n" + "=" * 80)
    print("STEP 7: CREATING SUBMISSION FILE")
    print("=" * 80)
    
    submission_df = pd.DataFrame({
        'CoilID': test_coil_ids,
        'Y': predictions
    })
    
    submission_df.to_csv(output_file, index=False)
    
    print(f"\nSubmission file created: {output_file}")
    print(f"Total predictions: {len(submission_df)}")
    print(f"Defect predictions (Y=1): {sum(predictions)}")
    print(f"No defect predictions (Y=0): {len(predictions) - sum(predictions)}")
    
    print("\nFirst 10 rows of submission:")
    print(submission_df.head(10).to_string(index=False))
    
    return submission_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline"""
    
    print("\n" + "=" * 80)
    print("DEFECT DETECTION IN HOT ROLLING - ML SOLUTION")
    print("=" * 80)
    
    # Step 1: Load data
    train_df, test_df = load_and_prepare_data()
    
    # Step 2: Preprocess features
    X_train, X_test, y_train, test_coil_ids, feature_names = preprocess_features(train_df, test_df)
    
    # Step 3: Analyze features
    feature_importance_df = analyze_feature_importance(X_train, y_train)
    
    # Step 4: Train models
    models = train_models(X_train, y_train)
    
    # Step 5: Evaluate and tune
    results = evaluate_models(models, X_train, y_train)
    
    # Step 6: Ensemble prediction
    ensemble_predictions = ensemble_predict(models, X_test, results)
    
    # Step 7: Create submission
    submission_df = create_submission(ensemble_predictions, test_coil_ids)
    
    print("\n" + "=" * 80)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review expected_submission.csv")
    print("2. Submit predictions for evaluation")
    print("3. Check F1 score and adjust parameters if needed")
    
    return submission_df


if __name__ == "__main__":
    submission = main()
