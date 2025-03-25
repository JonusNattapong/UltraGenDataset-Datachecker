"""
Functions for checking and validating data formats in datasets
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Union, Optional, Any
from datetime import datetime

def check_format(data: pd.DataFrame, 
                format_rules: Optional[Dict[str, str]] = None) -> Dict:
    """
    Check for data format issues in the dataset.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The dataset to check
    format_rules : Dict[str, str], optional
        Dictionary mapping column names to expected formats
        e.g. {'date_col': 'date', 'email_col': 'email'}
        
    Returns
    -------
    Dict
        Dictionary containing format validation results
    """
    # If no format rules specified, infer them
    if format_rules is None:
        format_rules = _infer_format_rules(data)
    
    # Check that all specified columns exist in the DataFrame
    for col in format_rules:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in dataset")
    
    # Check for format issues
    format_issues_by_column = {}
    expected_formats = {}
    total_format_issues = 0
    
    for col, format_type in format_rules.items():
        # Store expected format
        expected_formats[col] = format_type
        
        # Get validation function for this format type
        validate_func = _get_validation_function(format_type)
        
        if validate_func is not None:
            # Count format issues
            issues = sum(~data[col].apply(lambda x: validate_func(x) if pd.notnull(x) else True))
            if issues > 0:
                format_issues_by_column[col] = int(issues)
                total_format_issues += issues
    
    # Calculate score based on percentage of format issues
    total_values = sum(len(data) for col in format_rules)
    score = 1.0 - (total_format_issues / total_values) if total_values > 0 else 1.0
    
    result = {
        'total_format_issues': int(total_format_issues),
        'format_issues_by_column': format_issues_by_column,
        'expected_formats': expected_formats,
        'score': float(score)
    }
    
    return result

def _infer_format_rules(data: pd.DataFrame) -> Dict[str, str]:
    """
    Infer format rules for columns based on column names and data types.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The dataset to analyze
        
    Returns
    -------
    Dict[str, str]
        Dictionary mapping column names to inferred formats
    """
    format_rules = {}
    
    for col in data.columns:
        col_lower = col.lower()
        
        # Try to infer format from column name
        if 'email' in col_lower:
            format_rules[col] = 'email'
        elif 'phone' in col_lower or 'tel' in col_lower:
            format_rules[col] = 'phone'
        elif 'date' in col_lower or 'time' in col_lower:
            format_rules[col] = 'date'
        elif 'url' in col_lower or 'website' in col_lower or 'link' in col_lower:
            format_rules[col] = 'url'
        elif 'zip' in col_lower or 'postal' in col_lower:
            format_rules[col] = 'zipcode'
        elif 'ip' in col_lower and ('address' in col_lower or col_lower == 'ip'):
            format_rules[col] = 'ip'
        
        # If no rule inferred from name, try to infer from data
        else:
            # Sample non-null values
            sample = data[col].dropna().sample(min(10, len(data[col].dropna()))).astype(str) if not data[col].dropna().empty else []
            
            if sample.empty:
                continue
                
            # Check for date patterns
            date_count = sum(1 for x in sample if _is_potential_date(x))
            if date_count >= len(sample) * 0.5:
                format_rules[col] = 'date'
                continue
                
            # Check for email patterns
            email_count = sum(1 for x in sample if _validate_email(x))
            if email_count >= len(sample) * 0.5:
                format_rules[col] = 'email'
                continue
                
            # Check for URL patterns
            url_count = sum(1 for x in sample if _validate_url(x))
            if url_count >= len(sample) * 0.5:
                format_rules[col] = 'url'
                continue
    
    return format_rules

def _get_validation_function(format_type: str):
    """
    Get the appropriate validation function for a given format type.
    
    Parameters
    ----------
    format_type : str
        Type of format to validate
        
    Returns
    -------
    function or None
        Validation function for the specified format type
    """
    format_validators = {
        'email': _validate_email,
        'phone': _validate_phone,
        'date': _validate_date,
        'url': _validate_url,
        'zipcode': _validate_zipcode,
        'ip': _validate_ip,
        'number': _validate_number,
        'integer': _validate_integer,
        'float': _validate_float,
        'boolean': _validate_boolean
    }
    
    return format_validators.get(format_type.lower())

def _validate_email(value) -> bool:
    """Validate email format"""
    if not isinstance(value, str):
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, value))

def _validate_phone(value) -> bool:
    """Validate phone number format"""
    if not isinstance(value, str) and not isinstance(value, (int, float)):
        return False
    
    # Convert to string if numeric
    if isinstance(value, (int, float)):
        value = str(int(value))
    
    # Remove common separators
    value = re.sub(r'[\s\-\.()]+', '', value)
    
    # Check that it contains only digits and has a reasonable length
    return value.isdigit() and 7 <= len(value) <= 15

def _validate_date(value) -> bool:
    """Validate date format"""
    if isinstance(value, (pd.Timestamp, datetime)):
        return True
    
    if not isinstance(value, str):
        return False
    
    # Try common date formats
    date_formats = [
        '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d',
        '%d-%m-%Y', '%m-%d-%Y', '%Y%m%d',
        '%d %b %Y', '%d %B %Y'
    ]
    
    for fmt in date_formats:
        try:
            datetime.strptime(value, fmt)
            return True
        except ValueError:
            continue
            
    return False

def _is_potential_date(value) -> bool:
    """Check if a value is potentially a date"""
    # Check for numeric-only dates like YYYYMMDD
    if value.isdigit() and 8 <= len(value) <= 10:
        return True
        
    # Check for dates with common separators
    if re.search(r'\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4}', value):
        return True
        
    # Check for dates with month names
    month_pattern = r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)'
    if re.search(rf'\d{{1,2}}[\s-]*{month_pattern}[\s-]*\d{{2,4}}', value.lower()) or \
       re.search(rf'{month_pattern}[\s-]*\d{{1,2}}[\s,-]*\d{{2,4}}', value.lower()):
        return True
        
    return False

def _validate_url(value) -> bool:
    """Validate URL format"""
    if not isinstance(value, str):
        return False
    
    # Simple URL validation pattern
    pattern = r'^(https?|ftp):\/\/[^\s/$.?#].[^\s]*$'
    return bool(re.match(pattern, value))

def _validate_zipcode(value) -> bool:
    """Validate ZIP/postal code format"""
    if isinstance(value, (int, float)):
        value = str(int(value))
    
    if not isinstance(value, str):
        return False
    
    # Check for various postal code formats
    # US ZIP code (5 digits or ZIP+4)
    us_pattern = r'^\d{5}(-\d{4})?$'
    
    # Canadian postal code (A1A 1A1 format)
    ca_pattern = r'^[A-Za-z]\d[A-Za-z][ -]?\d[A-Za-z]\d$'
    
    # UK postal code
    uk_pattern = r'^[A-Za-z]{1,2}\d[A-Za-z\d]?[ ]?\d[A-Za-z]{2}$'
    
    # Generic - any sequence of 4-10 alphanumeric characters
    generic_pattern = r'^[A-Za-z0-9]{4,10}$'
    
    return bool(re.match(us_pattern, value) or 
                re.match(ca_pattern, value) or 
                re.match(uk_pattern, value) or 
                re.match(generic_pattern, value))

def _validate_ip(value) -> bool:
    """Validate IP address format"""
    if not isinstance(value, str):
        return False
    
    # IPv4 pattern
    ipv4_pattern = r'^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$'
    ipv4_match = re.match(ipv4_pattern, value)
    
    if ipv4_match:
        return all(0 <= int(g) <= 255 for g in ipv4_match.groups())
    
    # IPv6 pattern (simplified)
    ipv6_pattern = r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
    return bool(re.match(ipv6_pattern, value))

def _validate_number(value) -> bool:
    """Validate that value is a number"""
    return isinstance(value, (int, float)) and not pd.isna(value)

def _validate_integer(value) -> bool:
    """Validate that value is an integer"""
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return value.is_integer()
    return False

def _validate_float(value) -> bool:
    """Validate that value is a float"""
    return isinstance(value, float) and not pd.isna(value)

def _validate_boolean(value) -> bool:
    """Validate that value is a boolean"""
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return value in (0, 1)
    if isinstance(value, str):
        return value.lower() in ('true', 'false', 'yes', 'no', 't', 'f', 'y', 'n', '1', '0')
    return False

def fix_format_issues(data: pd.DataFrame, 
                     format_rules: Dict[str, str],
                     strategy: str = 'auto') -> pd.DataFrame:
    """
    Fix format issues in the dataset.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The dataset to fix
    format_rules : Dict[str, str]
        Dictionary mapping column names to expected formats
    strategy : str, optional
        Strategy to use for fixing format issues, by default 'auto'
        Options: 'auto', 'convert', 'remove'
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with format issues fixed
    """
    # Make a copy to avoid modifying the original DataFrame
    fixed_data = data.copy()
    
    # Validate strategy
    valid_strategies = ['auto', 'convert', 'remove']
    if strategy not in valid_strategies:
        raise ValueError(f"Invalid strategy: {strategy}. Valid options are: {', '.join(valid_strategies)}")
    
    # Process each column with format rules
    for col, format_type in format_rules.items():
        # Get validation function for this format type
        validate_func = _get_validation_function(format_type)
        
        if validate_func is None:
            continue
            
        # Create a mask for invalid values
        invalid_mask = ~fixed_data[col].apply(lambda x: validate_func(x) if pd.notnull(x) else True)
        
        # No invalid values, continue to next column
        if not invalid_mask.any():
            continue
            
        # Handle based on strategy and format type
        if strategy == 'remove' or (strategy == 'auto' and format_type in ['email', 'url', 'ip']):
            # For these formats, invalid values are likely errors and should be set to NaN
            fixed_data.loc[invalid_mask, col] = np.nan
            
        elif strategy == 'convert' or strategy == 'auto':
            # Try to convert values to the correct format
            if format_type == 'date':
                fixed_data.loc[:, col] = pd.to_datetime(fixed_data[col], errors='coerce')
                
            elif format_type == 'number' or format_type == 'float':
                fixed_data.loc[:, col] = pd.to_numeric(fixed_data[col], errors='coerce')
                
            elif format_type == 'integer':
                # Convert to float first to handle strings, then to integer
                numeric_vals = pd.to_numeric(fixed_data[col], errors='coerce')
                fixed_data.loc[:, col] = numeric_vals.fillna(numeric_vals).astype('Int64')
                
            elif format_type == 'boolean':
                # Map various boolean representations to True/False
                bool_map = {
                    'true': True, 'false': False,
                    'yes': True, 'no': False,
                    't': True, 'f': False,
                    'y': True, 'n': False,
                    '1': True, '0': False,
                    1: True, 0: False
                }
                
                fixed_data.loc[:, col] = fixed_data[col].map(
                    lambda x: bool_map.get(str(x).lower(), np.nan) if pd.notnull(x) else np.nan
                )
                
            elif format_type == 'phone':
                # Standardize phone numbers by removing non-digit characters
                fixed_data.loc[:, col] = fixed_data[col].astype(str).apply(
                    lambda x: re.sub(r'[\s\-\.()]+', '', x) if pd.notnull(x) and isinstance(x, str) else x
                )
                
                # Set invalid phone numbers to NaN
                fixed_data.loc[invalid_mask, col] = np.nan
                
    return fixed_data
