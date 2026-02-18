# SAMOLET Apartment Price Prediction System

Machine learning model for predicting apartment prices based on layout characteristics and property attributes for **SAMOLET Group** (ПАО «ГК «Самолет»), one of Russia's largest residential real estate developers (MOEX: SMLT).

## 🎯 Model Performance

- **R² Score**: 0.9786 (97.86% variance explained, 5-fold CV)
- **RMSE**: ~1.75M ₽ (~7-8% error)
- **MAE**: ~884K ₽ (~3-4% error)
- **Algorithm**: Random Forest Regressor (tuned via Grid Search)
- **Generalization**: Minimal overfitting (train R² 0.9966 vs test R² 0.9786)

## 📁 Project Structure

```
.
├── main.py                      # Gradio web interface (entry point)
├── notebook.ipynb               # Jupyter notebook for training/analysis
├── pyproject.toml               # Project config & dependencies
├── requirements.txt            # Pip/uv install targets
├── README.md                    # This file
├── Final_Report.md              # Detailed technical report
│
├── src/                         # Source code package
│   ├── core/                    # Core ML functionality
│   │   ├── preprocessing.py   # Data preprocessing & encoding
│   │   ├── feature_selection.py # Feature selection methods
│   │   ├── statistics.py        # Statistical analysis
│   │   ├── visualization.py     # Plotting utilities
│   │   └── regression_evals_and_tuning.py  # ML evaluation
│   ├── scripts/                 # Web scraping scripts
│   │   ├── web_scraper.py       # Basic requests-based scraper
│   │   ├── browser_scraper.py   # Playwright automation
│   │   ├── crawl4ai_scraper.py # Open-source AI crawler
│   │   ├── firecrawl_scraper.py # Cloud-based service
│   │   └── WEB_SCRAPING_README.md # Scraping documentation
│   ├── config/                  # Configuration
│   │   ├── settings.py          # Paths, constants, settings
│   │   └── logging_config.py    # Logging configuration
│   └── utils/                   # Additional utilities
│
├── models/                      # Trained model artifacts
│   ├── rf_tuned_model.joblib
│   ├── feature_encoders.joblib
│   ├── scalers.joblib           # Fitted sklearn scalers (inference)
│   ├── scaling_stats.joblib
│   ├── feature_names.joblib
│   ├── categorical_values.joblib
│   └── ...
│
├── data/                        # Data files
│   ├── test_data_5%_preprocessed.csv
│   └── test_data_5%_raw.csv
│
├── output/                      # Plots and web scraping outputs(e.g. feature importance)
│   ├── gini_importance.png
│   ├── permutation_importance.png
│   ├── firecrawl_output_{hash}_{flat_id}_{timestamp}.txt
│   ├── browser_output_{hash}_{flat_id}_{timestamp}.txt
│   ├── webscraper_output_{hash}_{flat_id}_{timestamp}.txt
```

## 🚀 Quick Start

### Option 1: Web Interface (Recommended)

```bash
# Run with Python (after installing dependencies)
python main.py

# Or with uv (no need to activate venv)
uv run main.py
```

Then open http://127.0.0.1:7860 in your browser.

**Features:**

- **Manual Input**: Enter apartment characteristics and get price predictions
- **Link-Based Input**: Documentation of web scraping attempts (currently blocked by anti-bot protection)
- **Test Data Evaluation**: Compare predictions against actual prices from test dataset

### Option 2: Jupyter Notebook

```bash
jupyter notebook notebook.ipynb
```

Contains complete model training pipeline, EDA, and analysis.

## 📊 Features Used

Feature importance (Gini + Permutation) ranks **District** and **TotalArea** as top drivers:

- **District**: 113 unique locations, mean-encoded (sklearn TargetEncoder); dominant predictor (~67% R² impact when removed)
- **TotalArea**: Primary size driver (~32% R² impact when removed)
- **Class**: Эконом → Комфорт → Бизнес → Элит
- **CeilingHeight**, **PropertyType**, **FloorsTotal**, **Phase**, **Finishing**: Additional predictors

See `output/gini_importance.png` and `output/permutation_importance.png` for full rankings.

## 🔧 Installation

Open the project folder (e.g. after extracting the zip or copying the root directory), then install dependencies using either method below.

### Option A: Using uv (recommended)

```bash
# From project root - install from pyproject.toml
uv sync

# Or install from requirements.txt
uv add -r .\requirements.txt
```

### Option B: Using pip

```bash
pip install -r requirements.txt
```

### Required Packages

Core dependencies:

- pandas, numpy, scikit-learn
- gradio (for web interface)
- joblib (for model serialization)
- beautifulsoup4 (for HTML parsing)

Optional for web scraping:

- playwright, crawl4ai, firecrawl-py

## 📈 Model Details

### Preprocessing Pipeline

1. **Outlier Removal**: IQR method (threshold=1.5); limits from training data only, applied to val/test
2. **Feature Encoding** (sklearn transformers, saved in `feature_encoders.joblib`):
   - Ordinal: Class, Finishing (OrdinalEncoder)
   - One-hot: PropertyType, BuildingType, etc. (OneHotEncoder, drop first)
   - Mean encoding: District (TargetEncoder with 5-fold cross-fitting, no leakage)
3. **Scaling**: Fitted sklearn scalers saved in `scalers.joblib`; inference uses same MinMax/Standard scalers as training (no manual replication)

### Validation Strategy

- Train/Validation/Test split: 75%/20%/5% (after outlier removal)
- 5-fold cross-validation for hyperparameter tuning (Grid Search)
- Outlier limits computed on training data only; applied consistently to val/test

## 🏢 About SAMOLET Group

- **Company**: ПАО «ГК «Самолет» (SAMOLET Group)
- **Ticker**: SMLT (MOEX)
- **Founded**: 2012
- **Headquarters**: Moscow, Russia
- **Business**: Full-cycle residential real estate development
- **Coverage**: Multiple Russian regions, primarily Moscow area

## 📄 Documentation

- **Final_Report.md**: Complete technical documentation
- **src/README.md**: Source code structure
- **scripts/WEB_SCRAPING_README.md**: Scraping implementation details

## ⚠️ Known Limitations

1. **Geographic**: Model trained on Moscow region data (113 districts); retraining needed for other SAMOLET regions
2. **Temporal**: No time-based features; doesn't account for market trends
3. **Link-Based Input**: Web scraping blocked by anti-bot protection (four approaches documented)
4. **Outliers**: IQR method may exclude legitimate luxury properties

For full assumptions, limitations, and metric interpretation in the real estate domain, see **Final_Report.md**.
