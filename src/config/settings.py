"""
Configuration module for SAMOLET Apartment Price Prediction System
Contains paths, constants, and settings used across the application
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Model artifact paths
MODEL_PATH = MODELS_DIR / "rf_tuned_model.joblib"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.joblib"
FEATURE_ENCODERS_PATH = MODELS_DIR / "feature_encoders.joblib"
SCALING_STATS_PATH = MODELS_DIR / "scaling_stats.joblib"
CATEGORICAL_VALUES_PATH = MODELS_DIR / "categorical_values.joblib"

# Test data paths
TEST_DATA_PREPROCESSED_PATH = DATA_DIR / "test_data_5%_preprocessed.csv"
TEST_DATA_RAW_PATH = DATA_DIR / "test_data_5%_raw.csv"

# Feature configuration
ORDINAL_CATEGORIES = {
    "Class": ["Эконом", "Комфорт", "Бизнес", "Элит"],
    "Finishing": ["Нет данных", "Без отделки", "Подчистовая", "Чистовая", "С мебелью (частично)", "С мебелью"]
}

# Categorical choices for UI
CLASS_CHOICES = ["Эконом", "Комфорт", "Бизнес", "Элит"]
BUILDING_TYPE_CHOICES = ["Монолит", "Панель", "Кирпич", "Кирпич-монолит"]
PROPERTY_TYPE_CHOICES = ["1 ккв", "2 ккв", "3 ккв", "4 ккв", "5 ккв", "Студия", "Апартаменты"]
PROPERTY_CATEGORY_CHOICES = ["Многокв. дом", "Апартаменты"]
APARTMENTS_CHOICES = ["Нет", "Да"]
FINISHING_CHOICES = ["Нет данных", "Без отделки", "Подчистовая", "Чистовая", "С мебелью (частично)", "С мебелью"]
APARTMENT_OPTION_CHOICES = ["Новостройка", "Вторичка"]
MORTGAGE_CHOICES = ["Да", "Нет"]
SUBSIDIES_CHOICES = ["Да", "Нет"]
LAYOUT_CHOICES = ["Да", "Нет", "Евро"]

# Gradio interface settings
GRADIO_SERVER_NAME = "127.0.0.1"
GRADIO_SERVER_PORT = 7860
GRADIO_SHARE = False

# Default prediction parameters
DEFAULT_TOTAL_AREA = 50.0
DEFAULT_CEILING_HEIGHT = 2.7
DEFAULT_FLOOR = 5
DEFAULT_FLOORS_TOTAL = 17
DEFAULT_PHASE = 1
DEFAULT_CLASS = "Комфорт"
DEFAULT_PROPERTY_TYPE = "2 ккв"
DEFAULT_BUILDING_TYPE = "Монолит"
DEFAULT_FINISHING = "Чистовая"

# Model metadata
MODEL_VERSION = "2.0"
MODEL_TYPE = "Random Forest Regressor"
MODEL_R2_SCORE = 0.9787
COMPANY_NAME = "SAMOLET Group"
COMPANY_TICKER = "SMLT"
