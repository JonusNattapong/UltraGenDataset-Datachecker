"""
Advanced usage example for the Dataset Checker library
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from dataset_checker import DatasetChecker
from dataset_checker.checks import data_distribution

def load_and_prepare_data():
    """Load diabetes dataset and introduce some quality issues"""
    # Load the diabetes dataset
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df['target'] = diabetes.target
    
    # Introduce quality issues for demonstration
    
    # 1. Add missing values
    rows_to_nullify = np.random.choice(df.index, size=int(0.1 * len(df)), replace=False)
    cols_to_nullify = np.random.choice(df.columns, size=3, replace=False)
    
    for col in cols_to_nullify:
        df.loc[np.random.choice(rows_to_nullify, size=int(0.7 * len(rows_to_nullify)), replace=False), col] = np.nan
    
    # 2. Add outliers
    for col in df.select_dtypes(include=['float64']).columns:
        if col != 'target':  # Don't modify the target
            # Add some extreme values
            outlier_indices = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
            df.loc[outlier_indices, col] = df[col].max() * np.random.uniform(3, 5, size=len(outlier_indices))
    
    # 3. Add duplicates
    duplicate_rows = df.sample(int(0.05 * len(df))).copy()
    df = pd.concat([df, duplicate_rows], ignore_index=True)
    
    # 4. Create a classification version of the target
    df['diabetes_class'] = pd.cut(df['target'], 
                                 bins=[0, 100, 200, 400], 
                                 labels=['low', 'medium', 'high'])
    
    # 5. Create some derived features with incorrect format
    df['bmi_to_bp'] = df['bmi'] / df['bp']
    
    return df

def evaluate_model_performance(X_train, X_test, y_train, y_test):
    """Evaluate model performance before data cleaning"""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return rmse, model

def main():
    # Set style for plots
    sns.set(style="whitegrid")
    
    # Load data with quality issues
    print("Loading dataset with synthetic quality issues...")
    df = load_and_prepare_data()
    print(f"Dataset shape: {df.shape}")
    
    # Initial train-test split (keeping the test set clean for fair comparison)
    X = df.drop(['target', 'diabetes_class'], axis=1)
    y = df['target']
    X_train_dirty, X_test, y_train_dirty, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Make a copy of the dirty training data
    dirty_df = pd.concat([X_train_dirty, y_train_dirty.rename('target')], axis=1)
    
    # Evaluate model on dirty data
    print("\nEvaluating model on dirty data...")
    rmse_dirty, _ = evaluate_model_performance(X_train_dirty, X_test, y_train_dirty, y_test)
    print(f"RMSE with dirty data: {rmse_dirty:.2f}")
    
    # Initialize DatasetChecker
    print("\nInitializing DatasetChecker...")
    checker = DatasetChecker(dirty_df, name="diabetes_dataset")
    
    # Run comprehensive quality check
    print("\nRunning comprehensive quality check...")
    report = checker.run_quality_check()
    
    # Print summary
    print("\n" + "=" * 60)
    print("DATASET QUALITY REPORT SUMMARY")
    print("=" * 60)
    print(report.summary())
    
    # Print recommendations
    print("\n" + "=" * 60)
    print("QUALITY IMPROVEMENT RECOMMENDATIONS")
    print("=" * 60)
    print(report.view_recommendations())
    
    # Save report as HTML and JSON
    report.to_html('diabetes_quality_report.html')
    report.to_json('diabetes_quality_report.json')
    print("\nQuality reports saved to 'diabetes_quality_report.html' and 'diabetes_quality_report.json'")
    
    # Fix dataset issues (Multi-step cleaning pipeline)
    print("\nCleaning the dataset (multi-step pipeline)...")
    
    # Step 1: Fix missing values
    print("Step 1: Fixing missing values...")
    checker.fix_missing_values(strategy='auto')
    
    # Step 2: Remove outliers
    print("Step 2: Removing outliers...")
    checker.check_outliers(method='zscore', threshold=3.0)
    checker.remove_outliers(method='zscore')
    
    # Step 3: Fix data format issues
    print("Step 3: Checking and fixing format issues...")
    # Create custom format rules
    format_rules = {col: 'float' for col in checker.data.columns if col != 'target'}
    format_rules['target'] = 'float'
    
    # Check formats
    checker.check_data_format(format_rules=format_rules)
    
    # Step 4: Remove duplicates
    print("Step 4: Removing duplicates...")
    checker.check_duplicates()
    checker.remove_duplicates()
    
    # Step 5: Transform skewed distributions
    print("Step 5: Transforming skewed distributions...")
    
    # Check distributions first
    dist_result = checker.check_data_distribution()
    
    # Find columns with significant skewness
    skewed_cols = [col for col, skew in dist_result['skewness'].items() 
                  if abs(skew) > 1.0 and col != 'target']
    
    if skewed_cols:
        print(f"  Found {len(skewed_cols)} significantly skewed columns: {', '.join(skewed_cols)}")
        # Transform skewed columns
        checker.data = data_distribution.transform_non_normal(checker.data, columns=skewed_cols)
    else:
        print("  No significantly skewed columns found")
    
    # Get the cleaned dataset
    clean_df = checker.data
    print(f"\nCleaned dataset shape: {clean_df.shape}")
    
    # Run quality check on clean dataset
    print("\nRunning quality check on cleaned dataset...")
    clean_report = checker.run_quality_check()
    
    # Print improvement summary
    print("\n" + "=" * 60)
    print("QUALITY IMPROVEMENT SUMMARY")
    print("=" * 60)
    print(f"Original quality score: {report.overall_score:.2f}")
    print(f"Improved quality score: {clean_report.overall_score:.2f}")
    print(f"Improvement: {(clean_report.overall_score - report.overall_score) * 100:.1f}%")
    
    # Prepare clean data for modeling
    X_train_clean = clean_df.drop('target', axis=1)
    y_train_clean = clean_df['target']
    
    # Evaluate model on clean data
    print("\nEvaluating model on cleaned data...")
    rmse_clean, model_clean = evaluate_model_performance(X_train_clean, X_test, y_train_clean, y_test)
    print(f"RMSE with cleaned data: {rmse_clean:.2f}")
    print(f"RMSE improvement: {(rmse_dirty - rmse_clean) / rmse_dirty * 100:.1f}%")
    
    # Visualize feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train_clean.columns,
        'importance': model_clean.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance After Data Cleaning')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("\nFeature importance plot saved to 'feature_importance.png'")
    
    # Create a comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Missing values before/after
    ax1 = axes[0, 0]
    missing_before = report.results['missing_values']['missing_percentage']
    missing_after = clean_report.results['missing_values']['missing_percentage']
    
    if missing_before:
        pd.DataFrame({
            'Before': pd.Series(missing_before),
            'After': pd.Series(missing_after)
        }).fillna(0).plot(kind='bar', ax=ax1)
        ax1.set_title('Missing Values Before vs After')
        ax1.set_ylabel('Missing (%)')
    else:
        ax1.text(0.5, 0.5, "No missing values found", ha='center', va='center')
        ax1.set_title('Missing Values')
    
    # Plot 2: Outliers before/after
    ax2 = axes[0, 1]
    outliers_before = {k: len(v) for k, v in report.results['outliers']['outliers_by_column'].items()}
    outliers_after = {k: len(v) for k, v in clean_report.results['outliers']['outliers_by_column'].items()}
    
    if outliers_before:
        pd.DataFrame({
            'Before': pd.Series(outliers_before),
            'After': pd.Series(outliers_after)
        }).fillna(0).plot(kind='bar', ax=ax2)
        ax2.set_title('Outliers Before vs After')
        ax2.set_ylabel('Count')
    else:
        ax2.text(0.5, 0.5, "No outliers found", ha='center', va='center')
        ax2.set_title('Outliers')
    
    # Plot 3: RMSE comparison
    ax3 = axes[1, 0]
    ax3.bar(['Dirty Data', 'Clean Data'], [rmse_dirty, rmse_clean])
    ax3.set_title('Model RMSE (Lower is Better)')
    ax3.set_ylabel('RMSE')
    
    # Plot 4: Overall quality score
    ax4 = axes[1, 1]
    ax4.bar(['Before Cleaning', 'After Cleaning'], [report.overall_score, clean_report.overall_score])
    ax4.set_title('Overall Quality Score')
    ax4.set_ylabel('Score (Higher is Better)')
    ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('cleaning_comparison.png')
    print("Comparison visualization saved to 'cleaning_comparison.png'")
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("The Dataset Checker library successfully identified and fixed several issues in the dataset.")
    print("This resulted in:")
    print(f"1. Quality score increase from {report.overall_score:.2f} to {clean_report.overall_score:.2f}")
    print(f"2. Model performance improvement with {(rmse_dirty - rmse_clean) / rmse_dirty * 100:.1f}% lower RMSE")
    print(f"3. Dataset size reduction from {len(dirty_df)} to {len(clean_df)} rows (removed duplicates and outliers)")
    print("=" * 60)

if __name__ == "__main__":
    main()
