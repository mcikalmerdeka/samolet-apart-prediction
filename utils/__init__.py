"""
Utils package for data preprocessing, statistics, visualization, and ML utilities.
The package is my own personal library for data analysis and machine learning projects.
"""

# Core preprocessing functions (always available)
from .preprocessing import (
    check_data_information,
    drop_columns,
    change_binary_dtype,
    handle_missing_values,
    filter_outliers,
    feature_scaling,
    feature_encoding
)

# Feature selection functions (always available)
from .feature_selection import (
    calculate_correlation_tabular,
    analyze_categorical_relationships,
    calculate_vif,
    calculate_feature_importance
)
# Statistics functions (always available)
from .statistics import (
    describe_numerical_combined,
    describe_categorical_combined,
    describe_date_columns,
    identify_distribution_types
)

# Visualization functions (always available)
from .visualization import (
    plot_dynamic_hisplots_kdeplots,
    plot_dynamic_boxplots_violinplots,
    plot_dynamic_countplot,
    plot_correlation_heatmap
)

# ML regression functions
from .regression_evals_and_tuning import (
    eval_regression,
    tune_pipelines,
    tune_single_model,
    tune_all_models,
    get_model_pipeline,
    get_hyperparameters
)

# Define what's exported with `from utils import *`
__all__ = [
    # Preprocessing
    'check_data_information',
    'drop_columns',
    'change_binary_dtype',
    'handle_missing_values',
    'filter_outliers',
    'feature_scaling',
    'feature_encoding',

    # Feature selection
    'calculate_correlation_tabular',
    'analyze_categorical_relationships',
    'calculate_vif',
    'calculate_feature_importance',
    
    # Statistics
    'describe_numerical_combined',
    'describe_categorical_combined',
    'describe_date_columns',
    'identify_distribution_types',
    
    # Visualization
    'plot_dynamic_hisplots_kdeplots',
    'plot_dynamic_boxplots_violinplots',
    'plot_dynamic_countplot',
    'plot_correlation_heatmap',
    
    # ML regression
    'eval_regression',
    'tune_pipelines',
    'tune_single_model',
    'tune_all_models',
    'get_model_pipeline',
    'get_hyperparameters',
]
