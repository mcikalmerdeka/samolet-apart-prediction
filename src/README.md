# Source Code Structure

This directory contains the source code for the SAMOLET Apartment Price Prediction System.

## Directory Structure

```
src/
├── __init__.py          # Package initialization
├── core/                # Core functionality
│   ├── __init__.py
│   ├── preprocessing.py       # Data preprocessing utilities
│   ├── feature_selection.py   # Feature selection methods
│   ├── statistics.py          # Statistical analysis tools
│   ├── visualization.py       # Plotting and visualization
│   └── regression_evals_and_tuning.py  # ML model evaluation & tuning
├── config/              # Configuration
│   ├── __init__.py
│   ├── settings.py           # Paths, constants, settings
│   └── logging_config.py     # Logging configuration
└── utils/               # Additional utilities
    └── __init__.py
```

## Module Descriptions

### `src/core/`
Core functionality for the machine learning pipeline:
- **preprocessing.py**: Data cleaning, encoding (ordinal, one-hot, mean), scaling, outlier removal
- **feature_selection.py**: Correlation analysis, VIF calculation, feature importance
- **statistics.py**: Descriptive statistics for numerical/categorical/date columns
- **visualization.py**: Dynamic plots (histograms, boxplots, heatmaps, countplots)
- **regression_evals_and_tuning.py**: Model evaluation, hyperparameter tuning, cross-validation

### `src/config/`
Configuration management:
- **settings.py**: Centralized paths, constants, default values, and model metadata
- **logging_config.py**: Logging setup with console and file handlers
- All file paths defined here for easy maintenance

### `src/utils/`
Additional utility functions (currently empty, reserved for future use).

## Usage

Import from the src package:

```python
# Import core functionality
from src.core import feature_encoding, feature_scaling, eval_regression

# Import configuration
from src.config import MODEL_PATH, ORDINAL_CATEGORIES

# Import logging
from src.config import setup_logger
logger = setup_logger("samolet_price_predictor")
```

## Entry Points

- **main.py** (root): Gradio interface for interactive predictions
- **notebook.ipynb** (root): Jupyter notebook for model training and analysis

Both entry points use the `src/` package for all functionality.
