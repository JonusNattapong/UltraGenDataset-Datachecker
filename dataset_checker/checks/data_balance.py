"""
Functions for checking class balance in datasets
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Any
from collections import Counter

def check_balance(data: pd.DataFrame, target_column: str) -> Dict:
    """
    Check for class balance issues in the dataset.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The dataset to check
    target_column : str
        Name of the target column for classification datasets
        
    Returns
    -------
    Dict
        Dictionary containing balance statistics
    """
    # Check that the target column exists
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    
    # Count classes
    class_counts = dict(Counter(data[target_column].dropna()))
    
    # Sort by frequency (most common first)
    class_counts = {k: v for k, v in sorted(class_counts.items(), key=lambda item: item[1], reverse=True)}
    
    # Calculate class percentages
    total_samples = sum(class_counts.values())
    class_percentages = {k: (v / total_samples) * 100 for k, v in class_counts.items()}
    
    # Calculate imbalance metrics
    n_classes = len(class_counts)
    
    if n_classes <= 1:
        imbalance_ratio = 1.0
        entropy = 0.0
    else:
        # Imbalance ratio: ratio of the most frequent class to the least frequent class
        most_common_class = next(iter(class_counts))
        least_common_class = list(class_counts.keys())[-1]
        imbalance_ratio = class_counts[most_common_class] / class_counts[least_common_class]
        
        # Class entropy
        probabilities = np.array(list(class_percentages.values())) / 100
        entropy = -np.sum(probabilities * np.log2(probabilities))
        
        # Normalize entropy to [0, 1] where 1 is perfectly balanced
        max_entropy = np.log2(n_classes)
        entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    
    # Calculate score based on entropy (1.0 = perfectly balanced, 0.0 = completely imbalanced)
    score = entropy
    
    result = {
        'target_column': target_column,
        'n_classes': n_classes,
        'class_counts': class_counts,
        'class_percentages': class_percentages,
        'imbalance_ratio': float(imbalance_ratio),
        'entropy': float(entropy),
        'score': float(score)
    }
    
    return result

def balance_dataset(data: pd.DataFrame, 
                   target_column: str,
                   method: str = 'undersample',
                   sampling_strategy: Union[str, Dict] = 'auto') -> pd.DataFrame:
    """
    Balance the dataset by oversampling minority classes or undersampling majority classes.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The dataset to balance
    target_column : str
        Name of the target column for classification datasets
    method : str, optional
        Method to use for balancing, by default 'undersample'
        Options: 'undersample', 'oversample', 'smote'
    sampling_strategy : Union[str, Dict], optional
        Sampling strategy, by default 'auto'
        'auto': All classes will have the same number of samples as the majority class (for oversample)
                or minority class (for undersample)
        'not minority': Only oversample the minority class
        'not majority': Only undersample the majority class
        Dict: Specify the desired number of samples for each class
        
    Returns
    -------
    pandas.DataFrame
        Balanced DataFrame
    """
    # Make a copy to avoid modifying the original DataFrame
    balanced_data = data.copy()
    
    # Get class counts
    class_counts = dict(Counter(balanced_data[target_column].dropna()))
    n_classes = len(class_counts)
    
    # Cannot balance if there's only one class
    if n_classes <= 1:
        return balanced_data
    
    # Validate method
    valid_methods = ['undersample', 'oversample', 'smote']
    if method not in valid_methods:
        raise ValueError(f"Invalid method: {method}. Valid options are: {', '.join(valid_methods)}")
    
    # If sampling_strategy is 'auto', determine appropriate strategy
    if sampling_strategy == 'auto':
        if method == 'undersample':
            # Undersample to the size of the minority class
            min_class = min(class_counts, key=class_counts.get)
            min_count = class_counts[min_class]
            sampling_strategy = {cls: min(count, min_count) for cls, count in class_counts.items()}
        else:
            # Oversample to the size of the majority class
            max_class = max(class_counts, key=class_counts.get)
            max_count = class_counts[max_class]
            sampling_strategy = {cls: max(count, max_count) for cls, count in class_counts.items()}
    
    # If SMOTE is requested but not available, fall back to regular oversampling
    if method == 'smote':
        try:
            from imblearn.over_sampling import SMOTE
            
            # Extract features and target
            X = balanced_data.drop(columns=[target_column])
            y = balanced_data[target_column]
            
            # Apply SMOTE
            smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            # Combine back into DataFrame
            balanced_data = pd.DataFrame(X_resampled, columns=X.columns)
            balanced_data[target_column] = y_resampled
            
        except ImportError:
            # Fall back to regular oversampling
            print("SMOTE not available. Falling back to regular oversampling.")
            method = 'oversample'
    
    # Apply undersampling
    if method == 'undersample':
        result_data = []
        
        for cls in class_counts:
            # Get all samples of this class
            cls_data = balanced_data[balanced_data[target_column] == cls]
            
            # Determine how many samples to keep
            n_samples = sampling_strategy.get(cls, len(cls_data))
            
            # Random undersampling
            if len(cls_data) > n_samples:
                cls_data = cls_data.sample(n=n_samples, random_state=42)
            
            result_data.append(cls_data)
        
        # Combine all classes
        balanced_data = pd.concat(result_data, ignore_index=True)
    
    # Apply oversampling
    elif method == 'oversample':
        result_data = []
        
        for cls in class_counts:
            # Get all samples of this class
            cls_data = balanced_data[balanced_data[target_column] == cls]
            
            # Determine how many samples to generate
            n_samples = sampling_strategy.get(cls, len(cls_data))
            
            # Random oversampling
            if len(cls_data) < n_samples:
                # Number of complete copies
                n_copies = n_samples // len(cls_data)
                
                # Number of additional samples needed
                n_additional = n_samples % len(cls_data)
                
                # Create multiple copies
                for _ in range(n_copies):
                    result_data.append(cls_data)
                
                # Add the remaining samples
                if n_additional > 0:
                    result_data.append(cls_data.sample(n=n_additional, random_state=42))
            else:
                result_data.append(cls_data)
        
        # Combine all classes
        balanced_data = pd.concat(result_data, ignore_index=True)
    
    return balanced_data
