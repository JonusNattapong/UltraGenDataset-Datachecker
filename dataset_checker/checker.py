"""
DatasetChecker: Main class for checking dataset quality
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Any, Tuple
import logging

from .report import QualityReport
from .checks import (
    missing_values,
    outliers,
    duplicates, 
    data_format,
    data_balance,
    data_distribution
)

class DatasetChecker:
    """
    A comprehensive class for checking and improving dataset quality.
    
    This class provides methods to detect and handle common data quality issues
    such as missing values, outliers, duplicates, and data format inconsistencies.
    It also provides tools for analyzing data distributions and balance.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The dataset to check
    name : str, optional
        A name for the dataset, by default "dataset"
    """
    
    def __init__(self, data: pd.DataFrame, name: str = "dataset"):
        """
        Initialize a new DatasetChecker instance.
        
        Parameters
        ----------
        data : pandas.DataFrame
            The dataset to check
        name : str, optional
            A name for the dataset, by default "dataset"
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")
        
        self.data = data
        self.name = name
        self.original_data = data.copy()
        self.results = {}
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up a logger for the DatasetChecker"""
        logger = logging.getLogger(f"dataset_checker.{self.name}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
        
    def run_quality_check(self, 
                         check_missing: bool = True,
                         check_outliers: bool = True,
                         check_duplicates: bool = True,
                         check_format: bool = True,
                         check_balance: bool = True,
                         check_distribution: bool = True) -> QualityReport:
        """
        Run a comprehensive quality check on the dataset.
        
        Parameters
        ----------
        check_missing : bool, optional
            Whether to check for missing values, by default True
        check_outliers : bool, optional
            Whether to check for outliers, by default True
        check_duplicates : bool, optional
            Whether to check for duplicates, by default True
        check_format : bool, optional
            Whether to check for data format issues, by default True
        check_balance : bool, optional
            Whether to check for class balance issues, by default True
        check_distribution : bool, optional
            Whether to check data distributions, by default True
            
        Returns
        -------
        QualityReport
            A report object containing the results of all checks
        """
        self.logger.info(f"Running quality check on dataset '{self.name}'")
        self.results = {}
        
        if check_missing:
            self.check_missing_values()
            
        if check_outliers:
            self.check_outliers()
            
        if check_duplicates:
            self.check_duplicates()
            
        if check_format:
            self.check_data_format()
            
        if check_balance:
            self.check_data_balance()
            
        if check_distribution:
            self.check_data_distribution()
            
        return QualityReport(self.results, self.data, self.name)
        
    def check_missing_values(self, columns: Optional[List[str]] = None) -> Dict:
        """
        Check for missing values in the dataset.
        
        Parameters
        ----------
        columns : List[str], optional
            List of columns to check, if None, all columns are checked
            
        Returns
        -------
        Dict
            Dictionary containing missing value statistics
        """
        self.logger.info("Checking for missing values")
        result = missing_values.check_missing(self.data, columns)
        self.results['missing_values'] = result
        return result
        
    def check_outliers(self, 
                      columns: Optional[List[str]] = None, 
                      method: str = 'zscore', 
                      threshold: float = 3.0) -> Dict:
        """
        Check for outliers in the dataset.
        
        Parameters
        ----------
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
        self.logger.info(f"Checking for outliers using method '{method}'")
        result = outliers.check_outliers(self.data, columns, method, threshold)
        self.results['outliers'] = result
        return result
        
    def check_duplicates(self, 
                        columns: Optional[List[str]] = None, 
                        fuzzy: bool = False,
                        threshold: float = 0.9) -> Dict:
        """
        Check for duplicate records in the dataset.
        
        Parameters
        ----------
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
        self.logger.info("Checking for duplicates")
        result = duplicates.check_duplicates(self.data, columns, fuzzy, threshold)
        self.results['duplicates'] = result
        return result
        
    def check_data_format(self, 
                         format_rules: Optional[Dict[str, str]] = None) -> Dict:
        """
        Check for data format issues in the dataset.
        
        Parameters
        ----------
        format_rules : Dict[str, str], optional
            Dictionary mapping column names to expected formats
            e.g. {'date_col': 'date', 'email_col': 'email'}
            
        Returns
        -------
        Dict
            Dictionary containing format validation results
        """
        self.logger.info("Checking data formats")
        result = data_format.check_format(self.data, format_rules)
        self.results['data_format'] = result
        return result
        
    def check_data_balance(self, 
                          target_column: str) -> Dict:
        """
        Check for class balance issues in the dataset.
        
        Parameters
        ----------
        target_column : str
            Name of the target column for classification datasets
            
        Returns
        -------
        Dict
            Dictionary containing balance statistics
        """
        self.logger.info(f"Checking class balance for '{target_column}'")
        result = data_balance.check_balance(self.data, target_column)
        self.results['data_balance'] = result
        return result
        
    def check_data_distribution(self, 
                              columns: Optional[List[str]] = None) -> Dict:
        """
        Analyze distributions of features in the dataset.
        
        Parameters
        ----------
        columns : List[str], optional
            List of columns to analyze, if None, all numerical columns are analyzed
            
        Returns
        -------
        Dict
            Dictionary containing distribution statistics
        """
        self.logger.info("Analyzing data distributions")
        result = data_distribution.check_distribution(self.data, columns)
        self.results['data_distribution'] = result
        return result
        
    def fix_missing_values(self, 
                          strategy: str = 'auto',
                          fill_values: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Fix missing values in the dataset.
        
        Parameters
        ----------
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
        self.logger.info(f"Fixing missing values using strategy '{strategy}'")
        if 'missing_values' not in self.results:
            self.check_missing_values()
            
        self.data = missing_values.fix_missing(self.data, strategy, fill_values)
        return self.data
        
    def remove_outliers(self, 
                       method: str = 'zscore',
                       threshold: float = 3.0,
                       columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Remove outliers from the dataset.
        
        Parameters
        ----------
        method : str, optional
            Method to use for outlier detection, by default 'zscore'
            Options: 'zscore', 'iqr', 'isolation_forest'
        threshold : float, optional
            Threshold to use for outlier detection, by default 3.0
        columns : List[str], optional
            List of columns to check, if None, all numerical columns are checked
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with outliers removed
        """
        self.logger.info(f"Removing outliers using method '{method}'")
        if 'outliers' not in self.results:
            self.check_outliers(columns, method, threshold)
            
        self.data = outliers.remove_outliers(self.data, 
                                            self.results['outliers'],
                                            method)
        return self.data
        
    def remove_duplicates(self, 
                         columns: Optional[List[str]] = None,
                         keep: str = 'first') -> pd.DataFrame:
        """
        Remove duplicate records from the dataset.
        
        Parameters
        ----------
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
        self.logger.info("Removing duplicates")
        if 'duplicates' not in self.results:
            self.check_duplicates(columns)
            
        self.data = duplicates.remove_duplicates(self.data, columns, keep)
        return self.data
        
    def reset_data(self) -> None:
        """
        Reset the dataset to its original state.
        """
        self.data = self.original_data.copy()
        self.logger.info("Reset dataset to original state")
