"""
QualityReport: Class for generating dataset quality reports
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Union, Optional, Any, Tuple
import os
import jinja2
import json
from datetime import datetime

class QualityReport:
    """
    Class for generating and visualizing dataset quality reports.
    
    Parameters
    ----------
    results : Dict
        Dictionary containing results from quality checks
    data : pandas.DataFrame
        The dataset that was checked
    dataset_name : str
        Name of the dataset
    """
    
    def __init__(self, results: Dict, data: pd.DataFrame, dataset_name: str = "dataset"):
        """
        Initialize a new QualityReport instance.
        
        Parameters
        ----------
        results : Dict
            Dictionary containing results from quality checks
        data : pandas.DataFrame
            The dataset that was checked
        dataset_name : str
            Name of the dataset
        """
        self.results = results
        self.data = data
        self.dataset_name = dataset_name
        self.timestamp = datetime.now()
        self._calculate_overall_score()
        
    def _calculate_overall_score(self) -> None:
        """Calculate overall quality score based on all checks"""
        scores = []
        weights = {
            'missing_values': 1.0,
            'outliers': 0.8,
            'duplicates': 0.9,
            'data_format': 0.7,
            'data_balance': 0.6,
            'data_distribution': 0.5
        }
        
        total_weight = 0
        
        for check_name, result in self.results.items():
            if 'score' in result:
                scores.append(result['score'] * weights.get(check_name, 1.0))
                total_weight += weights.get(check_name, 1.0)
        
        if scores and total_weight > 0:
            self.overall_score = sum(scores) / total_weight
        else:
            self.overall_score = None

    def summary(self) -> str:
        """
        Get a summary of the quality check results.
        
        Returns
        -------
        str
            A summary string of the quality check results
        """
        summary_lines = [
            f"Dataset Quality Report for '{self.dataset_name}'",
            f"Generated on: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Dataset Shape: {self.data.shape[0]} rows × {self.data.shape[1]} columns",
            ""
        ]
        
        if self.overall_score is not None:
            quality_level = "Excellent" if self.overall_score >= 0.9 else \
                            "Good" if self.overall_score >= 0.8 else \
                            "Fair" if self.overall_score >= 0.7 else \
                            "Poor" if self.overall_score >= 0.6 else \
                            "Problematic"
            summary_lines.append(f"Overall Quality Score: {self.overall_score:.2f} - {quality_level}")
            summary_lines.append("")
        
        summary_lines.append("Quality Check Results:")
        
        if 'missing_values' in self.results:
            res = self.results['missing_values']
            summary_lines.append(f"- Missing Values: {res['total_missing']} missing values across {res['columns_with_missing']} columns")
            
        if 'outliers' in self.results:
            res = self.results['outliers']
            summary_lines.append(f"- Outliers: {res['total_outliers']} outliers detected across {len(res['outliers_by_column'])} columns")
            
        if 'duplicates' in self.results:
            res = self.results['duplicates']
            summary_lines.append(f"- Duplicates: {res['total_duplicates']} duplicate rows ({res['duplicate_percentage']:.2f}% of data)")
            
        if 'data_format' in self.results:
            res = self.results['data_format']
            summary_lines.append(f"- Format Issues: {res['total_format_issues']} format issues across {len(res['format_issues_by_column'])} columns")
            
        if 'data_balance' in self.results:
            res = self.results['data_balance']
            summary_lines.append(f"- Class Balance: Imbalance ratio of {res['imbalance_ratio']:.2f} for target '{res['target_column']}'")
            
        if 'data_distribution' in self.results:
            res = self.results['data_distribution']
            n_skewed = sum(1 for v in res['skewness'].values() if abs(v) > 0.5)
            summary_lines.append(f"- Distributions: {n_skewed} columns with significant skewness")
            
        return "\n".join(summary_lines)
        
    def plot_missing_values(self, figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot missing values in the dataset.
        
        Parameters
        ----------
        figsize : Tuple[int, int], optional
            Figure size, by default (10, 6)
        """
        if 'missing_values' not in self.results:
            print("Missing values check not run")
            return
            
        missing_data = self.results['missing_values']['missing_percentage']
        
        plt.figure(figsize=figsize)
        ax = sns.barplot(x=list(missing_data.keys()), y=list(missing_data.values()))
        plt.title('Missing Values by Column')
        plt.xlabel('Column')
        plt.ylabel('Missing (%)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
    def plot_outliers(self, figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot outliers in the dataset.
        
        Parameters
        ----------
        figsize : Tuple[int, int], optional
            Figure size, by default (10, 6)
        """
        if 'outliers' not in self.results:
            print("Outliers check not run")
            return
            
        outliers_by_column = self.results['outliers']['outliers_by_column']
        
        plt.figure(figsize=figsize)
        columns = list(outliers_by_column.keys())
        counts = [len(indices) for indices in outliers_by_column.values()]
        
        ax = sns.barplot(x=columns, y=counts)
        plt.title('Outliers by Column')
        plt.xlabel('Column')
        plt.ylabel('Number of Outliers')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
    def plot_class_balance(self, figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot class balance for the target column.
        
        Parameters
        ----------
        figsize : Tuple[int, int], optional
            Figure size, by default (10, 6)
        """
        if 'data_balance' not in self.results:
            print("Class balance check not run")
            return
            
        target_column = self.results['data_balance']['target_column']
        class_counts = self.results['data_balance']['class_counts']
        
        plt.figure(figsize=figsize)
        ax = sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
        plt.title(f'Class Distribution for {target_column}')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
    def to_html(self, output_path: str) -> None:
        """
        Generate an HTML report of the quality check results.
        
        Parameters
        ----------
        output_path : str
            Path to save the HTML report
        """
        # HTML report template
        template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dataset Quality Report - {{ report.dataset_name }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .container { max-width: 1200px; margin: 0 auto; }
                .header { background-color: #f5f5f5; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
                .section { margin-bottom: 30px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .score { font-size: 24px; font-weight: bold; }
                .good { color: green; }
                .fair { color: orange; }
                .poor { color: red; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Dataset Quality Report</h1>
                    <p>Dataset: <strong>{{ report.dataset_name }}</strong></p>
                    <p>Generated on: {{ report.timestamp }}</p>
                    <p>Dataset Shape: {{ report.data_shape[0] }} rows × {{ report.data_shape[1] }} columns</p>
                    {% if report.overall_score %}
                    <p>Overall Quality Score: 
                        <span class="score {% if report.overall_score >= 0.8 %}good{% elif report.overall_score >= 0.6 %}fair{% else %}poor{% endif %}">
                            {{ "%.2f"|format(report.overall_score) }}
                        </span>
                    </p>
                    {% endif %}
                </div>

                {% if 'missing_values' in report.results %}
                <div class="section">
                    <h2>Missing Values</h2>
                    <p>Total missing values: {{ report.results.missing_values.total_missing }}</p>
                    <p>Columns with missing values: {{ report.results.missing_values.columns_with_missing }}</p>
                    <table>
                        <tr>
                            <th>Column</th>
                            <th>Missing Count</th>
                            <th>Missing Percentage</th>
                        </tr>
                        {% for col, pct in report.results.missing_values.missing_percentage.items() %}
                        <tr>
                            <td>{{ col }}</td>
                            <td>{{ report.results.missing_values.missing_count[col] }}</td>
                            <td>{{ "%.2f"|format(pct) }}%</td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>
                {% endif %}

                {% if 'outliers' in report.results %}
                <div class="section">
                    <h2>Outliers</h2>
                    <p>Total outliers: {{ report.results.outliers.total_outliers }}</p>
                    <p>Method used: {{ report.results.outliers.method }}</p>
                    <table>
                        <tr>
                            <th>Column</th>
                            <th>Number of Outliers</th>
                            <th>Outlier Percentage</th>
                        </tr>
                        {% for col, indices in report.results.outliers.outliers_by_column.items() %}
                        <tr>
                            <td>{{ col }}</td>
                            <td>{{ indices|length }}</td>
                            <td>{{ "%.2f"|format(indices|length * 100 / report.data_shape[0]) }}%</td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>
                {% endif %}

                {% if 'duplicates' in report.results %}
                <div class="section">
                    <h2>Duplicates</h2>
                    <p>Total duplicate rows: {{ report.results.duplicates.total_duplicates }}</p>
                    <p>Percentage of duplicates: {{ "%.2f"|format(report.results.duplicates.duplicate_percentage) }}%</p>
                </div>
                {% endif %}

                {% if 'data_format' in report.results %}
                <div class="section">
                    <h2>Data Format Issues</h2>
                    <p>Total format issues: {{ report.results.data_format.total_format_issues }}</p>
                    <table>
                        <tr>
                            <th>Column</th>
                            <th>Expected Format</th>
                            <th>Issues Found</th>
                        </tr>
                        {% for col, issues in report.results.data_format.format_issues_by_column.items() %}
                        <tr>
                            <td>{{ col }}</td>
                            <td>{{ report.results.data_format.expected_formats[col] }}</td>
                            <td>{{ issues }}</td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>
                {% endif %}

                {% if 'data_balance' in report.results %}
                <div class="section">
                    <h2>Class Balance</h2>
                    <p>Target column: {{ report.results.data_balance.target_column }}</p>
                    <p>Imbalance ratio: {{ "%.2f"|format(report.results.data_balance.imbalance_ratio) }}</p>
                    <table>
                        <tr>
                            <th>Class</th>
                            <th>Count</th>
                            <th>Percentage</th>
                        </tr>
                        {% for class, count in report.results.data_balance.class_counts.items() %}
                        <tr>
                            <td>{{ class }}</td>
                            <td>{{ count }}</td>
                            <td>{{ "%.2f"|format(count * 100 / report.data_shape[0]) }}%</td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>
                {% endif %}

                {% if 'data_distribution' in report.results %}
                <div class="section">
                    <h2>Data Distributions</h2>
                    <table>
                        <tr>
                            <th>Column</th>
                            <th>Mean</th>
                            <th>Std Dev</th>
                            <th>Min</th>
                            <th>25%</th>
                            <th>Median</th>
                            <th>75%</th>
                            <th>Max</th>
                            <th>Skewness</th>
                            <th>Kurtosis</th>
                        </tr>
                        {% for col in report.results.data_distribution.columns %}
                        <tr>
                            <td>{{ col }}</td>
                            <td>{{ "%.2f"|format(report.results.data_distribution.stats[col].mean) }}</td>
                            <td>{{ "%.2f"|format(report.results.data_distribution.stats[col].std) }}</td>
                            <td>{{ "%.2f"|format(report.results.data_distribution.stats[col].min) }}</td>
                            <td>{{ "%.2f"|format(report.results.data_distribution.stats[col]['25%']) }}</td>
                            <td>{{ "%.2f"|format(report.results.data_distribution.stats[col]['50%']) }}</td>
                            <td>{{ "%.2f"|format(report.results.data_distribution.stats[col]['75%']) }}</td>
                            <td>{{ "%.2f"|format(report.results.data_distribution.stats[col].max) }}</td>
                            <td class="{% if report.results.data_distribution.skewness[col]|abs > 1 %}poor{% elif report.results.data_distribution.skewness[col]|abs > 0.5 %}fair{% else %}good{% endif %}">
                                {{ "%.2f"|format(report.results.data_distribution.skewness[col]) }}
                            </td>
                            <td>{{ "%.2f"|format(report.results.data_distribution.kurtosis[col]) }}</td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>
                {% endif %}
            </div>
        </body>
        </html>
        """
        
        # Prepare data for the template
        template_data = {
            'report': {
                'dataset_name': self.dataset_name,
                'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'data_shape': self.data.shape,
                'overall_score': self.overall_score,
                'results': self.results
            }
        }
        
        # Render HTML
        template = jinja2.Template(template_str)
        html_content = template.render(**template_data)
        
        # Save to file
        with open(output_path, 'w') as f:
            f.write(html_content)
            
        print(f"Report saved to {output_path}")
        
    def to_json(self, output_path: str) -> None:
        """
        Save the report results to a JSON file.
        
        Parameters
        ----------
        output_path : str
            Path to save the JSON file
        """
        # Create a serializable copy of the results
        serializable_results = {
            'dataset_name': self.dataset_name,
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'data_shape': self.data.shape,
            'overall_score': self.overall_score,
            'results': {}
        }
        
        # Process each result type to ensure it's serializable
        for check_name, result in self.results.items():
            serializable_results['results'][check_name] = {}
            for key, value in result.items():
                # Handle non-serializable objects
                if isinstance(value, dict):
                    serializable_results['results'][check_name][key] = {
                        k: list(v) if isinstance(v, np.ndarray) else v
                        for k, v in value.items()
                    }
                elif isinstance(value, np.ndarray):
                    serializable_results['results'][check_name][key] = list(value)
                else:
                    serializable_results['results'][check_name][key] = value
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        print(f"Report saved to {output_path}")

    def view_recommendations(self) -> str:
        """
        Get recommendations for improving dataset quality based on the results.
        
        Returns
        -------
        str
            A string containing recommendations
        """
        recommendations = [
            f"Quality Improvement Recommendations for '{self.dataset_name}':",
            ""
        ]
        
        if 'missing_values' in self.results:
            res = self.results['missing_values']
            if res['total_missing'] > 0:
                recommendations.append("Missing Values Recommendations:")
                if res['total_missing'] / (self.data.shape[0] * self.data.shape[1]) > 0.2:
                    recommendations.append("- Your dataset has a high percentage of missing values. Consider acquiring more complete data.")
                
                high_missing_cols = [col for col, pct in res['missing_percentage'].items() if pct > 30]
                if high_missing_cols:
                    recommendations.append(f"- Consider dropping columns with excessive missing values: {', '.join(high_missing_cols)}")
                    
                low_missing_cols = [col for col, pct in res['missing_percentage'].items() if 0 < pct <= 10]
                if low_missing_cols:
                    recommendations.append(f"- For columns with few missing values, consider imputation: {', '.join(low_missing_cols)}")
                
                recommendations.append("")
        
        if 'outliers' in self.results:
            res = self.results['outliers']
            if res['total_outliers'] > 0:
                recommendations.append("Outliers Recommendations:")
                
                high_outlier_cols = [col for col, indices in res['outliers_by_column'].items() 
                                    if len(indices) > 0.05 * self.data.shape[0]]
                if high_outlier_cols:
                    recommendations.append(f"- Columns with many outliers that may need transformation: {', '.join(high_outlier_cols)}")
                
                low_outlier_cols = [col for col, indices in res['outliers_by_column'].items() 
                                   if 0 < len(indices) <= 0.01 * self.data.shape[0]]
                if low_outlier_cols:
                    recommendations.append(f"- Columns with few outliers that could be capped or removed: {', '.join(low_outlier_cols)}")
                
                recommendations.append("")
        
        if 'duplicates' in self.results:
            res = self.results['duplicates']
            if res['total_duplicates'] > 0:
                recommendations.append("Duplicates Recommendations:")
                if res['duplicate_percentage'] > 5:
                    recommendations.append("- Your dataset has a significant number of duplicates. You should remove them.")
                else:
                    recommendations.append("- Remove the small number of duplicate rows found.")
                recommendations.append("")
        
        if 'data_format' in self.results:
            res = self.results['data_format']
            if res['total_format_issues'] > 0:
                recommendations.append("Data Format Recommendations:")
                for col, issues in res['format_issues_by_column'].items():
                    if issues > 0:
                        recommendations.append(f"- Fix format issues in column '{col}' (expected format: {res['expected_formats'].get(col, 'unknown')})")
                recommendations.append("")
        
        if 'data_balance' in self.results:
            res = self.results['data_balance']
            if res['imbalance_ratio'] > 1.5:
                recommendations.append("Class Balance Recommendations:")
                if res['imbalance_ratio'] > 10:
                    recommendations.append("- Your dataset has severe class imbalance. Consider:")
                    recommendations.append("  * Collecting more data for minority classes")
                    recommendations.append("  * Using oversampling techniques like SMOTE")
                    recommendations.append("  * Using class weights in your model")
                elif res['imbalance_ratio'] > 3:
                    recommendations.append("- Your dataset has moderate class imbalance. Consider using class weights or oversampling.")
                else:
                    recommendations.append("- Your dataset has slight class imbalance. Monitor model performance on minority classes.")
                recommendations.append("")
        
        if 'data_distribution' in self.results:
            res = self.results['data_distribution']
            skewed_cols = [col for col, val in res['skewness'].items() if abs(val) > 1]
            if skewed_cols:
                recommendations.append("Data Distribution Recommendations:")
                recommendations.append(f"- Consider applying transformations to highly skewed columns: {', '.join(skewed_cols)}")
                recommendations.append("  * Log transformation for right-skewed data")
                recommendations.append("  * Square transformation for left-skewed data")
                recommendations.append("  * Box-Cox transformation for general use")
                recommendations.append("")
        
        if len(recommendations) <= 2:
            recommendations.append("No specific recommendations - your dataset appears to be of good quality!")
            
        return "\n".join(recommendations)
