"""
Basic usage example for the Dataset Checker library
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataset_checker import DatasetChecker

def create_sample_dataset():
    """Create a sample dataset with various quality issues for demonstration"""
    # Create a dataframe with some common issues
    np.random.seed(42)
    
    # Sample size
    n = 1000
    
    # Create data with various issues
    data = {
        # Numeric columns
        'age': np.random.normal(35, 10, n),
        'income': np.random.lognormal(10, 1, n),  # right-skewed
        'height': np.random.normal(170, 10, n),
        
        # Categorical column
        'gender': np.random.choice(['Male', 'Female', 'Other'], n, p=[0.48, 0.48, 0.04]),
        
        # Columns with missing values
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD', None], n, p=[0.3, 0.3, 0.2, 0.1, 0.1]),
        'satisfaction': np.random.choice([1, 2, 3, 4, 5, None], n, p=[0.1, 0.2, 0.3, 0.2, 0.1, 0.1]),
        
        # Date column
        'registration_date': pd.date_range(start='2020-01-01', periods=n),
        
        # Columns with format issues
        'email': [f'user{i}@example.com' if i % 10 != 0 else f'user{i}@invalid' for i in range(n)],
        'phone': [f'123-456-{i:04d}' if i % 15 != 0 else f'invalid-phone-{i}' for i in range(n)],
        
        # Target column with imbalance
        'churn': np.random.choice([0, 1], n, p=[0.85, 0.15])
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some outliers
    df.loc[np.random.choice(n, 20, replace=False), 'age'] = np.random.uniform(80, 100, 20)
    df.loc[np.random.choice(n, 15, replace=False), 'income'] = np.random.uniform(1000000, 2000000, 15)
    
    # Add some duplicates
    duplicate_rows = df.sample(50).copy()
    df = pd.concat([df, duplicate_rows], ignore_index=True)
    
    return df

def main():
    # Create sample dataset with issues
    print("Creating sample dataset...")
    df = create_sample_dataset()
    print(f"Dataset shape: {df.shape}")
    
    # Initialize DatasetChecker
    print("\nInitializing DatasetChecker...")
    checker = DatasetChecker(df, name="sample_dataset")
    
    # Run comprehensive quality check
    print("\nRunning comprehensive quality check...")
    report = checker.run_quality_check()
    
    # Print summary
    print("\n" + "=" * 50)
    print("DATASET QUALITY REPORT SUMMARY")
    print("=" * 50)
    print(report.summary())
    
    # Print recommendations
    print("\n" + "=" * 50)
    print("QUALITY IMPROVEMENT RECOMMENDATIONS")
    print("=" * 50)
    print(report.view_recommendations())
    
    # Visualize issues
    print("\nCreating visualizations...")
    plt.figure(figsize=(15, 10))
    
    # Plot missing values
    plt.subplot(2, 2, 1)
    report.plot_missing_values(figsize=(6, 4))
    
    # Plot outliers
    plt.subplot(2, 2, 2)
    report.plot_outliers(figsize=(6, 4))
    
    # Plot class balance
    plt.subplot(2, 2, 3)
    report.plot_class_balance(figsize=(6, 4))
    
    plt.tight_layout()
    plt.savefig('dataset_quality_visualizations.png')
    print("Visualizations saved to 'dataset_quality_visualizations.png'")
    
    # Generate HTML report
    print("\nGenerating HTML report...")
    report.to_html('dataset_quality_report.html')
    print("HTML report saved to 'dataset_quality_report.html'")
    
    # Fix dataset issues
    print("\nFixing dataset issues...")
    
    # Fix missing values
    print("- Fixing missing values...")
    df_fixed = checker.fix_missing_values(strategy='auto')
    
    # Remove outliers
    print("- Removing outliers...")
    df_fixed = checker.remove_outliers(method='zscore')
    
    # Remove duplicates
    print("- Removing duplicates...")
    df_fixed = checker.remove_duplicates()
    
    # Run quality check on fixed dataset
    print("\nRunning quality check on fixed dataset...")
    report_fixed = checker.run_quality_check()
    
    # Print comparison
    print("\n" + "=" * 50)
    print("BEFORE vs AFTER FIXING")
    print("=" * 50)
    print(f"Original dataset shape: {df.shape}")
    print(f"Fixed dataset shape: {df_fixed.shape}")
    print(f"Original quality score: {report.overall_score:.2f}")
    print(f"Fixed quality score: {report_fixed.overall_score:.2f}")
    print("=" * 50)

if __name__ == "__main__":
    main()
