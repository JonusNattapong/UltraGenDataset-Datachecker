"""
Functions for detecting and handling outliers in datasets
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Any
from scipy import stats
from sklearn.ensemble import IsolationForest

def check_outliers(data: pd.DataFrame, 
                  columns: Optional[List[str]] = None, 
                  method: str = 'zscore', 
                  threshold: float = 3.0) -> Dict:
    """
    Check for outliers in the dataset.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The dataset to check
    columns : List[str], optional
        List of columns to check, if None, all numerical columns are checked
    method : str, optional
        Method to use for outlier detection, by default 'zscore'
        Options: 'zscore', 'iqr', 'isolation_forest'
    threshold : float, optional
        Threshold to use for outlier detection, by default 3.0
        
    Returns
    -------
    Dict
        Dictionary containing outlier statistics
    """
    # Validate method
    valid_methods = ['zscore', 'iqr', 'isolation_forest']
    if method not in valid_methods:
        raise ValueError(f"Invalid method: {method}. Valid options are: {', '.join(valid_methods)}")
    
    # If no columns specified, use all numeric columns
    if columns is None:
        numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        columns = [col for col in data.columns if data[col].dtype.name in numeric_dtypes]
    else:
        # Check that all specified columns exist and are numeric
        for col in columns:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in dataset")
            if not pd.api.types.is_numeric_dtype(data[col]):
                raise ValueError(f"Column '{col}' is not numeric")
    
    outliers_by_column = {}
    total_outliers = 0
    
    if method == 'zscore':
        for col in columns:
            # Calculate Z-scores
            z_scores = np.abs(stats.zscore(data[col], nan_policy='omit'))
            # Find indices of outliers
            outlier_indices = np.where(z_scores > threshold)[0]
            
            if len(outlier_indices) > 0:
                outliers_by_column[col] = outlier_indices
                total_outliers += len(outlier_indices)
                
    elif method == 'iqr':
        for col in columns:
            # Calculate IQR
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            
            # Define bounds
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            # Find outliers
            outlier_indices = data.index[(data[col] < lower_bound) | (data[col] > upper_bound)].tolist()
            
            if len(outlier_indices) > 0:
                outliers_by_column[col] = outlier_indices
                total_outliers += len(outlier_indices)
                
    elif method == 'isolation_forest':
        # Fit Isolation Forest on all numeric columns together
        clf = IsolationForest(contamination=0.1, random_state=42)
        
        # Handle missing values - replace with mean
        data_for_if = data[columns].copy()
        data_for_if = data_for_if.fillna(data_for_if.mean())
        
        # Detect outliers
        outlier_predictions = clf.fit_predict(data_for_if)
        outlier_indices = np.where(outlier_predictions == -1)[0]
        
        # Map back to individual columns
        for col in columns:
            col_outliers = []
            for idx in outlier_indices:
                if not pd.isna(data.iloc[idx][col]):
                    col_outliers.append(idx)
            
            if len(col_outliers) > 0:
                outliers_by_column[col] = np.array(col_outliers)
                # Don't increment total_outliers here since we're counting unique rows
        
        total_outliers = len(outlier_indices)
    
    # Calculate score based on percentage of outliers
    total_values = len(data) * len(columns)
    score = 1.0 - (total_outliers / total_values) if total_values > 0 else 1.0
    
    result = {
        'total_outliers': int(total_outliers),
        'outliers_by_column': outliers_by_column,
        'method': method,
        'threshold': threshold,
        'score': float(score)
    }
    
    return result

def remove_outliers(data: pd.DataFrame, 
                   outlier_result: Dict,
                   strategy: str = 'remove') -> pd.DataFrame:
    """
    Remove or handle outliers in the dataset.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The dataset to fix
    outlier_result : Dict
        Dictionary containing outlier detection results, as returned by check_outliers()
    strategy : str, optional
        Strategy to use for handling outliers, by default 'remove'
        Options: 'remove', 'cap', 'mean', 'median'
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with outliers handled according to the strategy
    """
    # Make a copy to avoid modifying the original DataFrame
    fixed_data = data.copy()
    
    # Validate strategy
    valid_strategies = ['remove', 'cap', 'mean', 'median']
    if strategy not in valid_strategies:
        raise ValueError(f"Invalid strategy: {strategy}. Valid options are: {', '.join(valid_strategies)}")
    
    # Extract outlier information
    outliers_by_column = outlier_result.get('outliers_by_column', {})
    
    if not outliers_by_column:
        # No outliers detected, return original data
        return fixed_data
    
    # Get all unique outlier indices across all columns
    all_outlier_indices = set()
    for indices in outliers_by_column.values():
        all_outlier_indices.update(indices)
    
    if strategy == 'remove':
        # Remove rows with outliers
        fixed_data = fixed_data.drop(index=list(all_outlier_indices))
        
    else:
        # Handle outliers by column
        for col, indices in outliers_by_column.items():
            if strategy == 'cap':
                # Cap outliers at thresholds
                q1 = fixed_data[col].quantile(0.25)
                q3 = fixed_data[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                # Apply capping
                fixed_data.loc[fixed_data.index[indices], col] = fixed_data.loc[fixed_data.index[indices], col].clip(lower=lower_bound, upper=upper_bound)
                
            elif strategy == 'mean':
                # Replace with mean
                col_mean = fixed_data[col].mean()
                fixed_data.loc[fixed_data.index[indices], col] = col_mean
                
            elif strategy == 'median':
                # Replace with median
                col_median = fixed_data[col].median()
                fixed_data.loc[fixed_data.index[indices], col] = col_median
    
    return fixed_data
