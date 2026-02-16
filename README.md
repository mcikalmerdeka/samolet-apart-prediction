# SAMOLET Apartment Price Prediction System

Machine learning model for predicting apartment prices based on layout characteristics and property attributes for **SAMOLET Group** (ПАО «ГК «Самолет»), one of Russia's largest residential real estate developers (MOEX: SMLT).

## 🎯 Model Performance

- **R² Score**: 0.9787 (97.87% variance explained)
- **RMSE**: ~1.75M ₽ (~7-8% error)
- **MAE**: ~880K ₽ (~3-4% error)
- **Algorithm**: Random Forest Regressor (tuned)

## 📁 Project Structure

```
.
├── main.py                      # Gradio web interface (entry point)
├── notebook.ipynb              # Jupyter notebook for training/analysis
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
│   ├── scaling_stats.joblib
│   └── ...
│
├── data/                        # Data files
│   ├── test_data_5%_preprocessed.csv
│   └── test_data_5%_raw.csv
│
└── other/                       # Archive/older versions
    ├── main_v1.py
    └── notebook_v1.ipynb
```

## 🚀 Quick Start

### Option 1: Web Interface (Recommended)

```bash
python main.py
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

- **TotalArea**: Primary price driver
- **District**: 113 unique locations with mean encoding
- **Class**: Эконом → Комфорт → Бизнес → Элит
- **CeilingHeight**: Quality indicator
- **PropertyType**: Room count and type
- **FloorsTotal**: Building characteristics
- **Phase**: Construction timeline
- **Finishing**: Interior quality level

## 🔧 Installation

```bash
# Clone repository
git clone <repo-url>
cd samolet-price-prediction

# Install dependencies
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

## 📝 Usage Examples

### Using the Package

```python
# Import from src package
from src.core import feature_encoding, feature_scaling
from src.config import MODEL_PATH, ORDINAL_CATEGORIES

# Load model
import joblib
model = joblib.load(MODEL_PATH)

# Make predictions
# (See main.py for complete example)
```

### Web Scraping (Documentation Only)

Four scraping approaches were implemented but blocked by SAMOLET's anti-bot protection:

1. **web_scraper.py**: HTTP 403 Forbidden
2. **browser_scraper.py**: Browser fingerprint detection
3. **crawl4ai_scraper.py**: IP-based blocking with "Guru meditation" error
4. **firecrawl_scraper.py**: Requires API key (most likely to succeed)

See `scripts/WEB_SCRAPING_README.md` for detailed documentation.

## 📈 Model Details

### Preprocessing Pipeline

1. **Outlier Removal**: IQR method (threshold=1.5)
2. **Feature Encoding**:
   - Ordinal: Class, Finishing (sklearn OrdinalEncoder)
   - One-hot: PropertyType, BuildingType, etc. (sklearn OneHotEncoder)
   - Mean encoding: District (sklearn TargetEncoder with 5-fold CV)
3. **Scaling**:
   - MinMax: Class, Phase, Finishing
   - Standard: TotalArea, CeilingHeight, FloorsTotal, District

### Validation Strategy

- Train/Validation/Test split: 75%/20%/5%
- 5-fold cross-validation for hyperparameter tuning
- Consistent outlier limits across all splits

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

1. **Geographic**: Model trained primarily on Moscow region data (113 districts)
2. **Temporal**: No time-based features; doesn't account for market trends
3. **Link-Based Input**: Web scraping blocked by anti-bot protection
4. **Outliers**: IQR method may exclude legitimate luxury properties

## 🔮 Future Improvements

1. Implement FireCrawl API for link-based extraction
2. Add temporal features for market trend analysis
3. Enhanced geographic features (coordinates, proximity metrics)
4. SHAP explanations for interpretability
5. Model monitoring and drift detection

## 📧 Contact

Developed for SAMOLET Group Data Science team.

---

**Note**: This model is for demonstration and preliminary estimation purposes. Use professional appraisal for high-stakes decisions.
