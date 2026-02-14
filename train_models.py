"""
Training Script for Apartment Price Prediction
Uses utils.regression_evals_and_tuning functions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.append('utils')

# Import the regression utilities
from regression_evals_and_tuning import (
    eval_regression,
    tune_pipelines,
    tune_single_model,
    tune_all_models,
    get_model_pipeline,
    get_hyperparameters
)

# ============================================================
# STAGE 1: LOAD AND PREPARE DATA
# ============================================================

print("="*60)
print("STAGE 1: DATA PREPARATION")
print("="*60)

# Load the dataset
df_raw = pd.read_csv('data/case_data.csv')

# Define features and target
target = 'TotalCost'  # Can also use 'PricePerMeter'

# Numerical features
numerical_features = [
    'FloorsTotal', 'Phase', 'Floor', 'Section',
    'CeilingHeight', 'TotalArea', 'AreaWithoutBalcony',
    'LivingArea', 'KitchenArea', 'HallwayArea', 'PlotArea'
]

# Categorical features
categorical_features = [
    'District', 'Class', 'BuildingType', 'PropertyType',
    'PropertyCategory', 'Apartments', 'Finishing',
    'ApartmentOption', 'Mortgage', 'Subsidies', 'Layout',
    'Developer_encoded', 'Complex_encoded'
]

print(f"Target: {target}")
print(f"Numerical features: {len(numerical_features)}")
print(f"Categorical features: {len(categorical_features)}")

# ============================================================
# STAGE 2: DATA PREPROCESSING
# ============================================================

print("\n" + "="*60)
print("STAGE 2: DATA PREPROCESSING")
print("="*60)

# Remove rows with missing target values
df_model = df_raw.dropna(subset=[target]).copy()

# Remove outliers (1st and 99th percentile)
q_low = df_model[target].quantile(0.01)
q_high = df_model[target].quantile(0.99)
df_model = df_model[(df_model[target] >= q_low) & (df_model[target] <= q_high)]

print(f"Dataset shape after filtering: {df_model.shape}")
print(f"Target range: {df_model[target].min():,.0f} - {df_model[target].max():,.0f}")

# Encode categorical variables
df_encoded = df_model.copy()
label_encoders = {}

for col in categorical_features:
    if col in df_encoded.columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le

# Select features
feature_cols = [col for col in numerical_features + categorical_features 
                if col in df_encoded.columns]
X = df_encoded[feature_cols]
y = df_encoded[target]

print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"\nTarget statistics:")
print(f"Train - Mean: {y_train.mean():,.0f}, Std: {y_train.std():,.0f}")
print(f"Test - Mean: {y_test.mean():,.0f}, Std: {y_test.std():,.0f}")

# ============================================================
# STAGE 3: MODEL TRAINING AND EVALUATION
# ============================================================

print("\n" + "="*60)
print("STAGE 3: MODEL TRAINING")
print("="*60)

# Option 1: Train a single model (Random Forest)
print("\n" + "="*60)
print("OPTION 1: Tune Single Model (Random Forest)")
print("="*60)

rf_model, rf_time = tune_single_model(
    model_name='randomforest',
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    search_method='random',  # Use 'random' for faster search
    n_iter=20,
    scoring='neg_mean_squared_error',
    display=True,
    progress_bar=True
)

print(f"\nRandom Forest tuning completed in {rf_time:.2f} seconds")
print(f"Best parameters: {rf_model.best_params_}")
print(f"Best CV score (neg MSE): {rf_model.best_score_:.4f}")

# Evaluate the tuned Random Forest
print("\n" + "="*60)
print("EVALUATING TUNED RANDOM FOREST")
print("="*60)

rf_metrics = eval_regression(
    model=rf_model.best_estimator_,
    X_test=X_test,
    y_test=y_test,
    X_train=X_train,
    y_train=y_train,
    n_splits=5,
    n_repeats=2
)

# Option 2: Train multiple models
print("\n" + "="*60)
print("OPTION 2: Tune Multiple Models")
print("="*60)

# Select models to tune
models_to_tune = ['ridge', 'randomforest', 'xgboost']

fitted_models, fit_times = tune_all_models(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    models=models_to_tune,
    search_method='random',
    n_iter=15,
    scoring='neg_mean_squared_error',
    display=True,
    progress_bar=True
)

print('\n' + '='*60)
print('SUMMARY OF ALL MODELS')
print('='*60)
for name, model in fitted_models.items():
    print(f'\n{name.upper()}:')
    print(f'  Best CV Score: {model.best_score_:.4f}')
    print(f'  Best Parameters: {model.best_params_}')

# ============================================================
# STAGE 4: MODEL COMPARISON
# ============================================================

print("\n" + "="*60)
print("STAGE 4: MODEL COMPARISON")
print("="*60)

comparison_results = []

for name, model in fitted_models.items():
    y_pred = model.best_estimator_.predict(X_test)
    
    # Calculate metrics
    from sklearn.metrics import mean_absolute_error, r2_score
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    comparison_results.append({
        'Model': name,
        'Best_CV_Score': model.best_score_,
        'Test_MAE': mae,
        'Test_R2': r2
    })
    
    print(f'\n{name.upper()}:')
    print(f'  Mean Absolute Error: {mae:,.0f}')
    print(f'  R2 Score: {r2:.4f}')

# Create comparison dataframe
comparison_df = pd.DataFrame(comparison_results)
print('\n' + '='*60)
print('COMPARISON TABLE')
print('='*60)
print(comparison_df.to_string(index=False))

# ============================================================
# STAGE 5: DETAILED EVALUATION OF BEST MODEL
# ============================================================

print("\n" + "="*60)
print("STAGE 5: DETAILED EVALUATION OF BEST MODEL")
print("="*60)

# Find best model by test R2 score
best_model_name = comparison_df.loc[comparison_df['Test_R2'].idxmax(), 'Model']
best_model = fitted_models[best_model_name]

print(f'\nBest Model: {best_model_name.upper()}')
print(f'Test R2 Score: {comparison_df.loc[comparison_df["Test_R2"].idxmax(), "Test_R2"]:.4f}')

# Detailed evaluation
best_metrics = eval_regression(
    model=best_model.best_estimator_,
    X_test=X_test,
    y_test=y_test,
    X_train=X_train,
    y_train=y_train
)

print(f'\nBest parameters for {best_model_name}:')
for param, value in best_model.best_params_.items():
    print(f'  {param}: {value}')

# ============================================================
# STAGE 6: VISUALIZATION
# ============================================================

print("\n" + "="*60)
print("STAGE 6: VISUALIZATION")
print("="*60)

y_pred_best = best_model.best_estimator_.predict(X_test)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Predictions vs Actual
axes[0, 0].scatter(y_test, y_pred_best, alpha=0.5)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Price')
axes[0, 0].set_ylabel('Predicted Price')
axes[0, 0].set_title(f'{best_model_name.upper()}: Predictions vs Actual')

# Plot 2: Residuals
residuals = y_test - y_pred_best
axes[0, 1].scatter(y_pred_best, residuals, alpha=0.5)
axes[0, 1].axhline(y=0, color='r', linestyle='--')
axes[0, 1].set_xlabel('Predicted Price')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('Residual Plot')

# Plot 3: Distribution of residuals
axes[1, 0].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
axes[1, 0].set_xlabel('Residuals')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Distribution of Residuals')
axes[1, 0].axvline(x=0, color='r', linestyle='--')

# Plot 4: Model Comparison
if len(comparison_results) > 1:
    x_pos = range(len(comparison_results))
    models = [r['Model'] for r in comparison_results]
    r2_scores = [r['Test_R2'] for r in comparison_results]
    
    axes[1, 1].bar(x_pos, r2_scores, alpha=0.7)
    axes[1, 1].set_xlabel('Models')
    axes[1, 1].set_ylabel('R2 Score')
    axes[1, 1].set_title('Model Comparison (R2 Score)')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(models, rotation=45)

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved as 'model_evaluation.png'")
plt.show()

# Print prediction statistics
print(f'\nPrediction Statistics:')
print(f'Mean Actual: {y_test.mean():,.0f}')
print(f'Mean Predicted: {y_pred_best.mean():,.0f}')
print(f'Median Actual: {y_test.median():,.0f}')
print(f'Median Predicted: {np.median(y_pred_best):,.0f}')

# ============================================================
# STAGE 7: FEATURE IMPORTANCE (if applicable)
# ============================================================

print("\n" + "="*60)
print("STAGE 7: FEATURE IMPORTANCE")
print("="*60)

# Check if best model has feature_importances_ attribute
if hasattr(best_model.best_estimator_.named_steps[list(best_model.best_estimator_.named_steps.keys())[-1]], 'feature_importances_'):
    importances = best_model.best_estimator_.named_steps[list(best_model.best_estimator_.named_steps.keys())[-1]].feature_importances_
    
    # Create feature importance dataframe
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Plot top 15 features
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance_df.head(15), x='importance', y='feature')
    plt.title(f'Top 15 Feature Importances - {best_model_name.upper()}')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("Feature importance plot saved as 'feature_importance.png'")
    plt.show()
    
    print('\nTop 10 Most Important Features:')
    print(feature_importance_df.head(10).to_string(index=False))
else:
    print(f"\n{best_model_name.upper()} does not provide feature importances.")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
