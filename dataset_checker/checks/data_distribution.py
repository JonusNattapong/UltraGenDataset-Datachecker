"""
Functions for analyzing distributions of features in datasets
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from typing import Dict, List, Union, Optional, Any

def check_distribution(data: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict:
    """
    Analyze distributions of features in the dataset.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The dataset to check
    columns : List[str], optional
        List of columns to analyze, if None, all numerical columns are analyzed
        
    Returns
    -------
    Dict
        Dictionary containing distribution statistics
    """
    # If no columns specified, use all numeric columns
    if columns is None:
        numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        columns = [col for col in data.columns if data[col].dtype.name in numeric_dtypes]
    else:
        # Check that all specified columns exist
        for col in columns:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in dataset")
        
        # Filter to only include numeric columns
        columns = [col for col in columns if pd.api.types.is_numeric_dtype(data[col])]
    
    if not columns:
        return {
            'columns': [],
            'stats': {},
            'skewness': {},
            'kurtosis': {},
            'normality': {},
            'score': 1.0
        }
    
    # Calculate basic statistics
    stats_dict = {}
    for col in columns:
        # Get series without NaN values
        series = data[col].dropna()
        
        # Skip if empty after dropping NaN
        if len(series) == 0:
            continue
            
        # Calculate descriptive statistics
        stats_dict[col] = {
            'mean': float(series.mean()),
            'std': float(series.std()),
            'min': float(series.min()),
            'max': float(series.max()),
            '25%': float(series.quantile(0.25)),
            '50%': float(series.median()),
            '75%': float(series.quantile(0.75))
        }
    
    # Calculate skewness and kurtosis
    skewness = {}
    kurtosis = {}
    normality = {}
    
    for col in columns:
        # Get series without NaN values
        series = data[col].dropna()
        
        # Skip if empty after dropping NaN
        if len(series) == 0:
            continue
            
        # Skip if constant (zero variance)
        if series.std() == 0:
            skewness[col] = 0.0
            kurtosis[col] = 0.0
            normality[col] = {'statistic': 0.0, 'p_value': 1.0, 'normal': True}
            continue
        
        # Calculate skewness
        skew = float(stats.skew(series))
        skewness[col] = skew
        
        # Calculate excess kurtosis (normal distribution has kurtosis=3, excess kurtosis=0)
        kurt = float(stats.kurtosis(series))
        kurtosis[col] = kurt
        
        # Test for normality (Shapiro-Wilk test)
        # For large datasets, limit to 5000 samples for performance
        test_series = series.sample(min(5000, len(series)), random_state=42) if len(series) > 5000 else series
        
        try:
            statistic, p_value = stats.shapiro(test_series)
            normal = p_value > 0.05  # Consider normal if p-value > 0.05
        except:
            # Fallback if Shapiro-Wilk fails (can happen with large datasets)
            statistic, p_value = 0.0, 0.0
            normal = False
            
        normality[col] = {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'normal': normal
        }
    
    # Calculate distribution score
    # Score is based on normality and presence of extreme skewness/kurtosis
    if skewness:
        # Average normalized skewness penalty (0 for normal, higher for skewed)
        skew_penalties = [min(abs(skew) / 3, 1) for skew in skewness.values()]
        
        # Average normalized kurtosis penalty
        kurt_penalties = [min(abs(kurt) / 5, 1) for kurt in kurtosis.values()]
        
        # Combine penalties, weighting skewness more heavily
        avg_penalty = 0.7 * sum(skew_penalties) / len(skew_penalties) + \
                      0.3 * sum(kurt_penalties) / len(kurt_penalties)
        
        # Calculate score (1.0 = perfect normal distribution, 0.0 = extreme deviation)
        score = 1.0 - avg_penalty
    else:
        score = 1.0
    
    result = {
        'columns': columns,
        'stats': stats_dict,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'normality': normality,
        'score': float(score)
    }
    
    return result

def transform_non_normal(data: pd.DataFrame, 
                        columns: Optional[List[str]] = None,
                        method: str = 'auto') -> pd.DataFrame:
    """
    Apply transformations to make distributions more normal.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The dataset to transform
    columns : List[str], optional
        List of columns to transform, if None, all numerical columns with significant skewness are transformed
    method : str, optional
        Transformation method to apply, by default 'auto'
        Options: 'auto', 'log', 'sqrt', 'boxcox', 'yeojohnson'
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with transformed distributions
    """
    # Make a copy to avoid modifying the original DataFrame
    transformed_data = data.copy()
    
    # Validate method
    valid_methods = ['auto', 'log', 'sqrt', 'boxcox', 'yeojohnson']
    if method not in valid_methods:
        raise ValueError(f"Invalid method: {method}. Valid options are: {', '.join(valid_methods)}")
    
    # If no columns specified, find columns with significant skewness
    if columns is None:
        # First, check distributions of all numeric columns
        dist_result = check_distribution(data)
        
        # Select columns with significant skewness (|skew| > 0.5)
        columns = [col for col, skew in dist_result['skewness'].items() if abs(skew) > 0.5]
    
    # No columns to transform
    if not columns:
        return transformed_data
    
    # Apply transformations to each column
    for col in columns:
        # Skip if column doesn't exist or isn't numeric
        if col not in transformed_data.columns or not pd.api.types.is_numeric_dtype(transformed_data[col]):
            continue
        
        # Get series without NaN values
        series = transformed_data[col].dropna()
        
        # Skip if empty after dropping NaN
        if len(series) == 0:
            continue
        
        # Skip if column has zero or negative values for log/sqrt transformations
        has_non_positive = (series <= 0).any()
        
        # Determine transformation method
        if method == 'auto':
            # Calculate skewness
            skew = stats.skew(series)
            
            if skew > 1.0:  # Right-skewed
                if not has_non_positive:
                    col_method = 'log'
                else:
                    col_method = 'yeojohnson'
            elif skew < -1.0:  # Left-skewed
                col_method = 'yeojohnson'
            else:  # Moderate skewness
                if not has_non_positive:
                    col_method = 'boxcox'
                else:
                    col_method = 'yeojohnson'
        else:
            col_method = method
            
        # Apply the transformation
        if col_method == 'log':
            if has_non_positive:
                # Can't apply log to non-positive values
                continue
                
            transformed_data[col] = np.log1p(transformed_data[col])
            
        elif col_method == 'sqrt':
            if has_non_positive:
                # Can't apply sqrt to negative values
                continue
                
            transformed_data[col] = np.sqrt(transformed_data[col])
            
        elif col_method == 'boxcox':
            if has_non_positive:
                # Can't apply Box-Cox to non-positive values
                continue
                
            try:
                transformed_data.loc[transformed_data[col].notna(), col], _ = stats.boxcox(series)
            except:
                # Fallback if Box-Cox fails
                continue
                
        elif col_method == 'yeojohnson':
            try:
                transformed_data.loc[transformed_data[col].notna(), col], _ = stats.yeojohnson(series)
            except:
                # Fallback if Yeo-Johnson fails
                continue
    
    return transformed_data
