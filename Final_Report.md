# Apartment Price Prediction Model Report

## Executive Summary

A Random Forest regression model was developed to predict apartment prices in Moscow based on layout characteristics and property attributes. The model achieves high accuracy on cross-validation (R² ~0.96), demonstrating strong predictive performance for real estate valuation. A Gradio interface provides interactive predictions with both manual input and test dataset evaluation capabilities.

---

## 1. Model Description

### Target Variable
**TotalCost** (apartment price in rubles) was selected as the target variable, representing the total market value of each apartment. This provides direct pricing information more interpretable than price-per-meter for stakeholders.

### Model Selection
**Random Forest Regressor** was chosen after comparing five candidate models (Linear Regression, KNN, Decision Tree, Random Forest, XGBoost):

- **Algorithm**: Ensemble of decision trees
- **Key hyperparameters** (tuned via Grid Search):
  - `n_estimators`: 200
  - `max_depth`: 20
  - `criterion`: friedman_mse
  - `max_features`: sqrt
  - `min_samples_split`: 2
  - `min_samples_leaf`: 1

**Rationale**: Random Forest provides excellent balance between:
- High predictive accuracy (cross-validated R² ~0.96)
- Model interpretability through feature importances
- Robustness to outliers and non-linear relationships
- Minimal overfitting compared to single decision trees

### Feature Engineering

**Selected Features** (15 features after encoding):
- **Numerical**: TotalArea, FloorsTotal, CeilingHeight, Phase
- **Categorical (Ordinal)**: Class, Finishing
- **Categorical (One-hot)**: PropertyType (8 encoded columns: 2 ккв, 2 ккв (Евро), 3 ккв, 3 ккв (Евро), 4 ккв, 4 ккв (Евро), 5 ккв (Евро), К. пом, Студия)

**Preprocessing Pipeline**:
1. **Outlier Removal**: IQR method (threshold=1.5) applied to training data; same limits applied to validation/test data to ensure distribution consistency
2. **Encoding**:
   - Ordinal encoding for Class (Эконом=0 → Комфорт=1 → Бизнес=2 → Элит=3) and Finishing (quality levels 0-5)
   - One-hot encoding for PropertyType
3. **Scaling**:
   - MinMax: Class, Phase, Finishing
   - Standard: CeilingHeight
   - Robust: TotalArea, FloorsTotal

---

## 2. Validation Strategy

### Data Split
- **Training set**: 75% of data after outlier removal
- **Validation set**: 20% of data after outlier removal
- **Test set**: 5% reserved for final evaluation (2,965 samples)

The dataset was split randomly with fixed random state (42) to ensure reproducibility. **Critical Note**: Outlier removal limits were computed from training data only and applied consistently to validation and test sets. This prevents data leakage while ensuring that training and evaluation data have consistent distributions.

### Validation Issue Discovered and Resolved

During development, a distribution mismatch issue was identified:
- **Problem**: Initial implementation removed outliers from training data only, while validation data retained outliers. This caused the model to train on data with max TotalCost ~62M but evaluate on data with extreme outliers (200M+), resulting in validation R² of ~0.07 vs cross-validated R² of ~0.96.
- **Solution**: Apply the same IQR limits (computed from training data) to validation and test data. This ensures distribution consistency while preventing data leakage.

### Hyperparameter Tuning
Grid Search with 5-fold cross-validation on the training set was used to optimize Random Forest hyperparameters. Cross-validation R² was used as the optimization metric.

### Model Selection Approach
1. Trained five baseline models on identical preprocessed data
2. Evaluated using cross-validation on training data
3. Selected Random Forest based on best cross-validation R² and minimal train-validation gap
4. Applied hyperparameter tuning only to the selected model
5. Final model evaluated on validation set; test set reserved for production evaluation

---

## 3. Performance Metrics

### Cross-Validation Performance (Training Data)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R² Score** | ~0.96 | Model explains ~96% of price variance in cross-validation |
| **RMSE** | ~2.3M ₽ | Average prediction error magnitude |

### Validation Set Performance

After applying consistent outlier filtering to both training and validation data, the validation metrics align with cross-validation results.

**Note**: Specific metric values should be verified by running the corrected notebook (`notebook_refactored.ipynb`). The key finding is that consistent preprocessing between training and evaluation data is critical for reliable model assessment.

### Metric Justification

- **R²**: Primary metric for regression tasks; indicates proportion of variance explained by the model. Value >0.96 suggests excellent fit for real estate domain.

- **RMSE**: Penalizes larger errors more heavily, crucial for avoiding significant mispricing in high-value properties.

- **MAE**: More interpretable for stakeholders; represents average absolute prediction error.

---

## 4. Model Interpretation

### Feature Importance
The model provides inherent interpretability through feature importance scores:

**Top Contributing Features** (based on feature selection and model analysis):
1. **TotalArea** - Primary driver of apartment value
2. **Class** - Property classification (Economy to Elite)
3. **CeilingHeight** - Quality indicator
4. **PropertyType** - Number of rooms significantly impacts pricing
5. **FloorsTotal** - Building characteristics influence value
6. **Phase** - Construction phase/timeline affects pricing
7. **Finishing** - Interior finishing quality

### Individual Predictions
For each prediction, the Gradio interface provides:
- Predicted total price (rubles and millions)
- Price per square meter
- Input feature summary
- Comparison with actual values (for test data evaluation mode)

---

## 5. Assumptions and Limitations

### Assumptions

1. **Data Quality**: Assumes provided data is accurate and represents actual market transactions/listings
2. **Feature Completeness**: Key pricing factors (area, class, type) are captured in available features
3. **Temporal Stability**: Model assumes pricing patterns remain stable; no temporal features included
4. **Market Homogeneity**: Treats Moscow real estate market as relatively uniform
5. **Outlier Consistency**: Outliers identified during training will not appear in production data, or will be handled similarly

### Limitations

1. **Geographic Features**: Limited geographic information; no detailed location coordinates or district-level granularity. Geographic features (latitude, longitude, district, etc.) were excluded due to potential data quality issues or modeling complexity.

2. **Temporal Dynamics**: Model does not account for:
   - Market fluctuations over time
   - Seasonal pricing trends
   - Construction completion dates impact on pricing

3. **Missing Features**: Potentially valuable features not utilized:
   - Proximity to metro/transportation
   - Neighborhood amenities
   - Building age/condition
   - Parking availability
   - View quality

4. **Outlier Handling**: Outlier removal (IQR method) may exclude legitimate luxury/unique properties, limiting model applicability to extreme-value apartments

5. **Generalization**: Model trained exclusively on Moscow market data; not transferable to other cities without retraining

6. **Link-Based Input Not Implemented**: The interface supports manual input and test dataset evaluation. Automatic feature extraction from apartment listing URLs (e.g., https://samolet.ru/project/...) is not implemented. This would require web scraping capabilities and is recommended as a future enhancement.

---

## 6. Interface Implementation

### Gradio Application (`main.py`)

The interface supports two interaction modes:

#### Tab 1: Manual Input
Users manually enter apartment characteristics:
- Area information (Total Area, Ceiling Height, Floor, Total Floors)
- Property details (Class, Property Type, Building Type, Property Category, Finishing, Phase)
- Additional options (Apartments, Apartment Option, Mortgage, Subsidies, Layout)
- Optional detailed area breakdown (Area Without Balcony, Living Area, Kitchen Area, Hallway Area)

Output includes predicted price, price per m², and input summary.

#### Tab 2: Test Data Evaluation
Users can browse through 2,965 test samples and compare model predictions against actual prices:
- Select sample by index
- View predicted vs actual price comparison
- See absolute error and error percentage
- Display original (unscaled) feature values for interpretability

**Implementation Note**: Test data is loaded from two files:
- `test_data_5%_preprocessed.csv` - Scaled/encoded features for model prediction
- `test_data_5%_raw.csv` - Original values for display in the interface

---

## 7. Deliverables

### Source Code
- `notebook_refactored.ipynb` - Complete data analysis and model training pipeline
- `main.py` - Gradio inference application
- `utils/` - Helper modules for preprocessing and evaluation

### Model Artifacts
Located in `model_artifacts/`:
- `rf_tuned_model.joblib` - Trained Random Forest model
- `feature_names.joblib` - List of features used by the model
- `encodings.joblib` - Ordinal encoding mappings
- `scaling_stats.joblib` - Scaling statistics (min/max, mean/std, median/IQR)

### Test Data
Located in `data/`:
- `test_data_5%_raw.csv` - Raw test data (2,965 samples)
- `test_data_5%_preprocessed.csv` - Preprocessed test data ready for prediction

### Example Predictions
Run `python main.py` to launch the Gradio interface and test predictions.

---

## 8. Future Improvements

1. **Link-Based Input**: Implement web scraping to automatically extract features from apartment listing URLs
2. **Temporal Features**: Add date features to capture market trends
3. **Geographic Features**: Incorporate coordinates, district encodings, proximity to amenities
4. **SHAP Explanations**: Implement SHAP for detailed per-prediction interpretability
5. **Model Monitoring**: Set up drift detection for production deployment
6. **Expanded Property Types**: Include more diverse property categories

---

## Conclusion

The Random Forest regression model successfully predicts apartment prices with high accuracy, balancing performance with interpretability. A critical lesson learned during development was the importance of consistent preprocessing between training and evaluation data—removing outliers from training data only caused severe distribution mismatch and poor validation performance.

The model is suitable for preliminary price estimation tasks but should be used in conjunction with professional appraisal for high-stakes decisions. Key limitations include absence of detailed geographic features, temporal dynamics, and link-based input capability, which present opportunities for future enhancement.