# Dataset Checker

A comprehensive Python library for checking and improving the quality of datasets.

## Features

- **Missing Values Detection**: Identify and handle missing values in your dataset
- **Outlier Detection**: Find and manage statistical outliers
- **Duplicate Detection**: Identify identical or nearly identical records
- **Data Format Validation**: Ensure data formats are consistent
- **Data Balance Analysis**: Check for class imbalance issues
- **Data Distribution Analysis**: Analyze distributions of your features
- **Comprehensive Reporting**: Generate detailed quality reports

## Installation

```bash
pip install dataset-checker
```

## Quick Start

```python
import pandas as pd
from dataset_checker import DatasetChecker

# Load your dataset
df = pd.read_csv('your_dataset.csv')

# Create a checker instance
checker = DatasetChecker(df)

# Run a comprehensive quality check
report = checker.run_quality_check()

# View report summary
print(report.summary())

# Generate a comprehensive report
report.to_html('dataset_quality_report.html')
```

## Documentation

For complete documentation, visit the docs directory or check the function-specific help:

```python
help(DatasetChecker)
```

## Examples

See the `examples` directory for detailed usage examples.

## License

This project is licensed under the MIT License - see the LICENSE file for details.