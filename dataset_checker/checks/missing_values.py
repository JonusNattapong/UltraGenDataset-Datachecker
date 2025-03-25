"""
Functions for checking and handling missing values in datasets
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Any

def check_missing(data: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict:
    """
    Check for missing values in the dataset.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The dataset to check
    columns : List[str], optional
        List of columns to check, if None, all columns are checked
        
    Returns
    -------
    Dict
        Dictionary containing missing value statistics
    """
    if columns is None:
        columns = data.columns
    else:
        # Check that all specified columns exist in the DataFrame
        for col in columns:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in dataset")
    
    # Calculate missing value statistics
    missing_count = {}
    missing_percentage = {}
    total_missing = 0
    columns_with_missing = 0
    
    for col in columns:
        count = data[col].isna().sum()
        percentage = (count / len(data)) * 100
        
        if count > 0:
            missing_count[col] = count
            missing_percentage[col] = percentage
            total_missing += count
            columns_with_missing += 1
    
    # Calculate score based on missing values (1.0 = no missing, 0.0 = all missing)
    total_cells = len(data) * len(columns)
    score = 1.0 - (total_missing / total_cells) if total_cells > 0 else 1.0
    
    result = {
        'total_missing': int(total_missing),
        'columns_with_missing': int(columns_with_missing),
        'missing_count': missing_count,
        'missing_percentage': missing_percentage,
        'score': float(score)
    }
    
    return result

def fix_missing(data: pd.DataFrame, 
               strategy: str = 'auto',
               fill_values: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Fix missing values in the dataset.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The dataset to fix
    strategy : str, optional
        Strategy to use for filling missing values, by default 'auto'
        Options: 'auto', 'mean', 'median', 'mode', 'constant'
    fill_values : Dict[str, Any], optional
        Dictionary mapping column names to fill values for the 'constant' strategy
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with missing values fixed
    """
    # Make a copy to avoid modifying the original DataFrame
    fixed_data = data.copy()
    
    # Define valid strategies
    valid_strategies = ['auto', 'mean', 'median', 'mode', 'constant', 'drop']
    if strategy not in valid_strategies:
        raise ValueError(f"Invalid strategy: {strategy}. Valid options are: {', '.join(valid_strategies)}")
    
    # Initialize fill_values dictionary if it doesn't exist
    if fill_values is None:
        fill_values = {}
    
    # Get columns with missing values
    missing_columns = [col for col in fixed_data.columns if fixed_data[col].isna().any()]
    
    # Strategy 'drop': Drop rows with any missing values
    if strategy == 'drop':
        return fixed_data.dropna()
    
    # Process each column with missing values
    for col in missing_columns:
        # If a specific fill value is provided for this column, use it
        if col in fill_values:
            fixed_data[col] = fixed_data[col].fillna(fill_values[col])
            continue
        
        # Auto strategy: determine the best strategy based on data type and distribution
        if strategy == 'auto':
            if pd.api.types.is_numeric_dtype(fixed_data[col]):
                # For numeric data, use median for skewed distributions, mean otherwise
                skewness = fixed_data[col].skew()
                if abs(skewness) > 1.0:  # Significantly skewed
                    fixed_data[col] = fixed_data[col].fillna(fixed_data[col].median())
                else:
                    fixed_data[col] = fixed_data[col].fillna(fixed_data[col].mean())
            else:
                # For categorical/object data, use mode
                mode_value = fixed_data[col].mode()[0] if not fixed_data[col].mode().empty else None
                fixed_data[col] = fixed_data[col].fillna(mode_value)
        
        # Specific strategies
        elif strategy == 'mean' and pd.api.types.is_numeric_dtype(fixed_data[col]):
            fixed_data[col] = fixed_data[col].fillna(fixed_data[col].mean())
        
        elif strategy == 'median' and pd.api.types.is_numeric_dtype(fixed_data[col]):
            fixed_data[col] = fixed_data[col].fillna(fixed_data[col].median())
        
        elif strategy == 'mode':
            mode_value = fixed_data[col].mode()[0] if not fixed_data[col].mode().empty else None
            fixed_data[col] = fixed_data[col].fillna(mode_value)
            
    return fixed_data
