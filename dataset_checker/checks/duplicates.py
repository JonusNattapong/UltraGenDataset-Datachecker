"""
Functions for detecting and handling duplicate records in datasets
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Any
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def check_duplicates(data: pd.DataFrame, 
                    columns: Optional[List[str]] = None, 
                    fuzzy: bool = False,
                    threshold: float = 0.9) -> Dict:
    """
    Check for duplicate records in the dataset.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The dataset to check
    columns : List[str], optional
        List of columns to check, if None, all columns are checked
    fuzzy : bool, optional
        Whether to perform fuzzy matching for duplicates, by default False
    threshold : float, optional
        Similarity threshold for fuzzy matching, by default 0.9
        
    Returns
    -------
    Dict
        Dictionary containing duplicate statistics
    """
    # If no columns specified, use all columns
    if columns is None:
        columns = data.columns.tolist()
    else:
        # Check that all specified columns exist in the DataFrame
        for col in columns:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in dataset")
    
    # Check for exact duplicates
    if not fuzzy:
        # Find duplicates
        duplicate_mask = data.duplicated(subset=columns, keep='first')
        duplicate_indices = np.where(duplicate_mask)[0]
        
        # Get first occurrences of duplicates
        if len(duplicate_indices) > 0:
            # For each duplicate index, find its first occurrence
            duplicate_pairs = {}
            for idx in duplicate_indices:
                duplicate_row = data.iloc[idx][columns]
                
                # Find the first occurrence of this row
                for i, row in data[columns].iterrows():
                    if i >= idx:
                        break
                    
                    if row.equals(duplicate_row):
                        duplicate_pairs[idx] = i
                        break
        else:
            duplicate_pairs = {}
            
    # Check for fuzzy duplicates
    else:
        duplicate_indices = []
        duplicate_pairs = {}
        
        # Convert DataFrame to a list of strings for text columns
        text_columns = [col for col in columns if data[col].dtype == 'object']
        
        if text_columns:
            # Concatenate text columns for each row
            text_data = data[text_columns].fillna('').astype(str).apply(lambda x: ' '.join(x), axis=1)
            
            # Generate TF-IDF matrix
            vectorizer = TfidfVectorizer(strip_accents='unicode', lowercase=True)
            try:
                tfidf_matrix = vectorizer.fit_transform(text_data)
                
                # Compute cosine similarity matrix
                similarity_matrix = cosine_similarity(tfidf_matrix)
                
                # Find indices of similar rows
                for i in range(len(data) - 1):
                    for j in range(i + 1, len(data)):
                        if similarity_matrix[i, j] >= threshold:
                            duplicate_indices.append(j)
                            duplicate_pairs[j] = i
            
            except Exception as e:
                # Handle cases where text vectorization fails
                print(f"Warning: Fuzzy matching failed - {str(e)}")
                pass
        
        # For numerical columns, check for near-duplicates
        num_columns = [col for col in columns if pd.api.types.is_numeric_dtype(data[col])]
        
        if num_columns:
            for i in range(len(data) - 1):
                for j in range(i + 1, len(data)):
                    # Skip if already identified as a duplicate
                    if j in duplicate_indices:
                        continue
                    
                    # Calculate similarity for numerical features
                    row_i = data.iloc[i][num_columns].fillna(0).values
                    row_j = data.iloc[j][num_columns].fillna(0).values
                    
                    # Skip if any row has all zeros (to avoid division by zero)
                    if np.sum(row_i) == 0 or np.sum(row_j) == 0:
                        continue
                    
                    # Calculate cosine similarity
                    similarity = np.dot(row_i, row_j) / (np.linalg.norm(row_i) * np.linalg.norm(row_j))
                    
                    if similarity >= threshold:
                        duplicate_indices.append(j)
                        duplicate_pairs[j] = i
        
        # Convert to array for consistency
        duplicate_indices = np.array(list(set(duplicate_indices)))
                    
    # Calculate duplicate statistics
    total_duplicates = len(duplicate_indices)
    duplicate_percentage = (total_duplicates / len(data)) * 100 if len(data) > 0 else 0
    
    # Calculate score based on percentage of duplicates (1.0 = no duplicates, 0.0 = all duplicates)
    score = 1.0 - (duplicate_percentage / 100)
    
    result = {
        'total_duplicates': int(total_duplicates),
        'duplicate_percentage': float(duplicate_percentage),
        'duplicate_indices': duplicate_indices.tolist() if isinstance(duplicate_indices, np.ndarray) else duplicate_indices,
        'duplicate_pairs': duplicate_pairs,
        'fuzzy_match': fuzzy,
        'score': float(score)
    }
    
    return result

def remove_duplicates(data: pd.DataFrame, 
                     columns: Optional[List[str]] = None,
                     keep: str = 'first') -> pd.DataFrame:
    """
    Remove duplicate records from the dataset.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The dataset to fix
    columns : List[str], optional
        List of columns to check, if None, all columns are checked
    keep : str, optional
        Which duplicates to keep, by default 'first'
        Options: 'first', 'last', False (don't keep any)
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with duplicates removed
    """
    # Make a copy to avoid modifying the original DataFrame
    fixed_data = data.copy()
    
    # If no columns specified, use all columns
    if columns is None:
        columns = fixed_data.columns.tolist()
    
    # Remove duplicates
    fixed_data = fixed_data.drop_duplicates(subset=columns, keep=keep)
    
    return fixed_data
