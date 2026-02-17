"""
Gradio Inference Script for Apartment Price Prediction (v2 - Using SKLearn Encoders)
This script creates an interactive UI for predicting apartment prices using the trained Random Forest model.

Developed for: SAMOLET Group (ПАО «ГК «Самолет»)
Company: One of Russia's largest residential real estate developers (MOEX: SMLT)
Founded: 2012, Headquartered in Moscow
Operations: Full-cycle development across multiple Russian regions

Dataset Context: Model trained on SAMOLET's property portfolio, primarily concentrated in 
Moscow region and surrounding areas (113 unique districts/locations).
"""

import gradio as gr
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import logging

# Setup application logger - this initializes handlers
from src.config import setup_logger
logger = setup_logger("samolet_price_predictor")

# Import core modules for applying encoders
from src.core import feature_encoding, feature_scaling
from src.config import (
    MODEL_PATH,
    FEATURE_NAMES_PATH,
    FEATURE_ENCODERS_PATH,
    SCALERS_PATH,
    SCALING_STATS_PATH,
    CATEGORICAL_VALUES_PATH,
    TEST_DATA_PREPROCESSED_PATH,
    TEST_DATA_RAW_PATH,
    ORDINAL_CATEGORIES
)

# Import sklearn for unpickling
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')

try:
    logger.info("Loading model artifacts...")
    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded from {MODEL_PATH}")
    
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    logger.info(f"Loaded {len(feature_names)} feature names")
    
    feature_encoders = joblib.load(FEATURE_ENCODERS_PATH)
    logger.debug(f"Feature encoders loaded: ordinal={len([k for k in feature_encoders.keys() if k.startswith('ordinal_')])}, onehot={len([k for k in feature_encoders.keys() if k.startswith('onehot_')])}")
    
    scalers = joblib.load(SCALERS_PATH)
    logger.info(f"Sklearn scalers loaded: {list(scalers.keys())}")
    
    scaling_stats = joblib.load(SCALING_STATS_PATH)
    logger.info("Scaling statistics loaded (backup)")
    
    categorical_values = joblib.load(CATEGORICAL_VALUES_PATH)
    logger.info(f"Loaded {len(categorical_values['District'])} unique districts/locations")
    
    test_data_preprocessed = pd.read_csv(TEST_DATA_PREPROCESSED_PATH)
    test_data_raw = pd.read_csv(TEST_DATA_RAW_PATH)
    logger.info(f"Test data loaded: {len(test_data_preprocessed)} samples")
    
    logger.info("✅ All model artifacts loaded successfully")
    logger.info("💼 Model trained on SAMOLET Group property portfolio (primarily Moscow region)")
    
    # Also print to console for user feedback
    print(f"✅ Model loaded successfully")
    print(f"✅ Expected features: {len(feature_names)} features")
    print(f"✅ Test data loaded: {len(test_data_preprocessed)} samples")
    print(f"✅ Available locations: {len(categorical_values['District'])} unique districts/locations")
    print(f"\n💼 Model trained on SAMOLET Group property portfolio")
    print(f"   (primarily Moscow region and surrounding areas)")
except Exception as e:
    logger.error(f"Failed to load model artifacts: {e}", exc_info=True)
    print(f"❌ Error loading model artifacts: {e}")
    raise

# Preprocessing function (matching notebook preprocessing)
def preprocess_input(
    total_area: float,
    floors_total: int,
    phase: int,
    floor: int,
    ceiling_height: float,
    class_option: str,
    building_type: str,
    property_type: str,
    property_category: str,
    apartments: str,
    finishing: str,
    apartment_option: str,
    mortgage: str,
    subsidies: str,
    layout: str,
    district: str,
    area_without_balcony: float = None,
    living_area: float = None,
    kitchen_area: float = None,
    hallway_area: float = None
):
    """
    Preprocess input data following the same pipeline as in the notebook.
    Uses fitted sklearn encoders for proper transformation.
    
    Returns:
        DataFrame with processed features ready for model prediction
    """
    
    # Create input dataframe with all original features
    input_data = pd.DataFrame([{
        'TotalArea': total_area,
        'FloorsTotal': floors_total,
        'Phase': phase,
        'Floor': floor,
        'CeilingHeight': ceiling_height,
        'AreaWithoutBalcony': area_without_balcony if area_without_balcony else total_area * 0.95,
        'LivingArea': living_area if living_area else total_area * 0.6,
        'KitchenArea': kitchen_area if kitchen_area else total_area * 0.15,
        'HallwayArea': hallway_area if hallway_area else total_area * 0.1,
        'Class': class_option,
        'BuildingType': building_type,
        'PropertyType': property_type,
        'PropertyCategory': property_category,
        'Apartments': apartments,
        'Finishing': finishing,
        'ApartmentOption': apartment_option,
        'Mortgage': mortgage,
        'Subsidies': subsidies,
        'Layout': layout,
        'District': district
    }])
    
    # 1. Feature Encoding using fitted sklearn encoders
    ordinal_cols = ["Class", "Finishing"]
    nominal_cols = ["BuildingType", "PropertyType", "PropertyCategory", "Apartments", 
                    "ApartmentOption", "Mortgage", "Subsidies", "Layout"]
    mean_encoding_cols = ["District"]
    
    input_encoded, _ = feature_encoding(
        data=input_data,
        ordinal_columns=ordinal_cols,
        nominal_columns=nominal_cols,
        mean_encoding_columns=mean_encoding_cols,
        ordinal_categories=ORDINAL_CATEGORIES,
        encoders=feature_encoders,
        handle_unknown='ignore'  # Handle unknown categories gracefully
    )
    
    # 2. Select only the features used in training
    # Ensure all expected features exist
    for feature in feature_names:
        if feature not in input_encoded.columns:
            input_encoded[feature] = 0
    
    # Keep only the features in the correct order
    input_encoded = input_encoded[feature_names]
    
    # 3. Feature Scaling using fitted sklearn scalers
    input_scaled = input_encoded.copy()
    
    # Apply MinMax scaling
    if 'minmax' in scalers:
        minmax_cols = scalers['minmax'].feature_names_in_
        cols_to_scale = [col for col in minmax_cols if col in input_scaled.columns]
        if cols_to_scale:
            input_scaled[cols_to_scale] = scalers['minmax'].transform(input_scaled[cols_to_scale])
    
    # Apply Standard scaling
    if 'standard' in scalers:
        standard_cols = scalers['standard'].feature_names_in_
        cols_to_scale = [col for col in standard_cols if col in input_scaled.columns]
        if cols_to_scale:
            input_scaled[cols_to_scale] = scalers['standard'].transform(input_scaled[cols_to_scale])
    
    return input_scaled


def predict_price(
    total_area: float,
    floors_total: int,
    phase: int,
    floor: int,
    ceiling_height: float,
    class_option: str,
    building_type: str,
    property_type: str,
    property_category: str,
    apartments: str,
    finishing: str,
    apartment_option: str,
    mortgage: str,
    subsidies: str,
    layout: str,
    district: str,
    area_without_balcony: float = None,
    living_area: float = None,
    kitchen_area: float = None,
    hallway_area: float = None
):
    """
    Predict apartment price based on input features.
    
    Returns:
        Formatted prediction string with price in rubles
    """
    
    try:
        logger.info(f"Starting prediction: {district}, {property_type}, {total_area}m²")
        
        # Preprocess the input
        logger.debug("Preprocessing input data...")
        processed_data = preprocess_input(
            total_area=total_area,
            floors_total=floors_total,
            phase=phase,
            floor=floor,
            ceiling_height=ceiling_height,
            class_option=class_option,
            building_type=building_type,
            property_type=property_type,
            property_category=property_category,
            apartments=apartments,
            finishing=finishing,
            apartment_option=apartment_option,
            mortgage=mortgage,
            subsidies=subsidies,
            layout=layout,
            district=district,
            area_without_balcony=area_without_balcony,
            living_area=living_area,
            kitchen_area=kitchen_area,
            hallway_area=hallway_area
        )
        
        # Make prediction
        logger.debug("Making prediction with trained model...")
        prediction = model.predict(processed_data)[0]
        logger.info(f"Prediction successful: {prediction:,.0f} ₽")
        
        # Format output
        prediction_millions = prediction / 1_000_000
        price_per_sqm = prediction / total_area
        
        result = f"""
        ### 🏠 Predicted Apartment Price
        
        **Total Price:** {prediction:,.0f} ₽ ({prediction_millions:.2f} million ₽)
        
        **Price per m²:** {price_per_sqm:,.0f} ₽/m²
        
        ---
        **Input Summary:**
        - Area: {total_area} m²
        - Floor: {floor} / {floors_total}
        - District: {district}
        - Class: {class_option}
        - Finishing: {finishing}
        """
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        return f"❌ Error during prediction: {str(e)}\n\nDetails: {repr(e)}"


def predict_from_test_data(sample_index: int):
    """
    Load a test sample and make prediction, showing comparison with actual value.
    
    Args:
        sample_index: Index of the test sample to use (0 to len(test_data)-1)
    
    Returns:
        Formatted string with prediction vs actual comparison
    """
    
    try:
        logger.info(f"Evaluating test sample #{sample_index + 1}")
        sample_preprocessed = test_data_preprocessed.iloc[sample_index]
        sample_raw = test_data_raw.iloc[sample_index]
        
        X_sample = sample_preprocessed[feature_names].values.reshape(1, -1)
        actual_price = sample_preprocessed['TotalCost']
        
        logger.debug(f"Sample features: District={sample_raw['District']}, Area={sample_raw['TotalArea']}m²")
        predicted_price = model.predict(X_sample)[0]
        logger.info(f"Test prediction: predicted={predicted_price:,.0f} ₽, actual={actual_price:,.0f} ₽")
        
        error = predicted_price - actual_price
        error_percent = (error / actual_price) * 100
        abs_error = abs(error)
        
        total_area = sample_raw['TotalArea']
        price_per_sqm = predicted_price / total_area
        
        result = f"""
        ### 📊 Test Sample #{sample_index + 1} / {len(test_data_preprocessed)}
        
        ---
        
        #### Prediction Results:
        
        | Metric | Value |
        |--------|-------|
        | **Predicted Price** | {predicted_price:,.0f} ₽ ({predicted_price/1_000_000:.2f}M ₽) |
        | **Actual Price** | {actual_price:,.0f} ₽ ({actual_price/1_000_000:.2f}M ₽) |
        | **Absolute Error** | {abs_error:,.0f} ₽ ({abs_error/1_000_000:.2f}M ₽) |
        | **Error %** | {error_percent:.2f}% |
        | **Price per m²** | {price_per_sqm:,.0f} ₽/m² |
        
        ---
        
        #### Sample Features (Original Values):
        - **TotalArea**: {sample_raw['TotalArea']} m²
        - **District**: {sample_raw['District']}
        - **Class**: {sample_raw['Class']}
        - **CeilingHeight**: {sample_raw['CeilingHeight']} m
        - **Floor**: {sample_raw['Floor']} / {sample_raw['FloorsTotal']}
        - **Phase**: {sample_raw['Phase']}
        - **Finishing**: {sample_raw['Finishing']}
        - **PropertyType**: {sample_raw['PropertyType']}
        - **BuildingType**: {sample_raw['BuildingType']}
        
        """
        
        return result
        
    except Exception as e:
        logger.error(f"Test prediction failed for sample #{sample_index}: {e}", exc_info=True)
        return f"❌ Error during prediction: {str(e)}"


# Define the Gradio interface
def create_gradio_interface():
    """
    Create and configure the Gradio interface for apartment price prediction.
    """
    
    # Define choices for categorical inputs (from the dataset)
    class_choices = ["Эконом", "Комфорт", "Бизнес", "Элит"]
    building_type_choices = ["Монолит", "Панель", "Кирпич", "Кирпич-монолит"]
    property_type_choices = ["1 ккв", "2 ккв", "3 ккв", "4 ккв", "5 ккв", "Студия", "Апартаменты"]
    property_category_choices = ["Многокв. дом", "Апартаменты"]
    apartments_choices = ["Нет", "Да"]
    finishing_choices = ["Нет данных", "Без отделки", "Подчистовая", "Чистовая", "С мебелью (частично)", "С мебелью"]
    apartment_option_choices = ["Новостройка", "Вторичка"]
    mortgage_choices = ["Да", "Нет"]
    subsidies_choices = ["Да", "Нет"]
    layout_choices = ["Да", "Нет", "Евро"]
    district_choices = categorical_values['District']  # Load from artifacts
    
    # Create the interface with organized input sections
    with gr.Blocks(title="🏠 Apartment Price Predictor", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("""
        # 🏠 Apartment Price Prediction System (v2)
        
        **Developed for SAMOLET Group** - One of Russia's largest residential real estate developers
        
        This tool predicts apartment prices based on layout characteristics and property features.
        The model uses **Random Forest Regressor** with proper sklearn preprocessing pipeline.
        
        **Model Coverage**: Trained on SAMOLET's property portfolio, primarily from Moscow region 
        and surrounding areas (113 unique locations in dataset).
        ---
        """)
        
        with gr.Tabs():
            # Tab 1: Manual Input
            with gr.Tab("🔧 Manual Input"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📐 Area & Layout")
                        
                        total_area = gr.Number(
                            label="Total Area (m²)",
                            value=50.0,
                            minimum=20.0,
                            maximum=200.0,
                            info="Total area of the apartment"
                        )
                        
                        ceiling_height = gr.Number(
                            label="Ceiling Height (m)",
                            value=2.7,
                            minimum=2.5,
                            maximum=3.5,
                            info="Height of ceilings"
                        )
                        
                        floor = gr.Slider(
                            label="Floor",
                            minimum=0,
                            maximum=40,
                            value=5,
                            step=1,
                            info="Floor number"
                        )
                        
                        floors_total = gr.Slider(
                            label="Total Floors in Building",
                            minimum=1,
                            maximum=40,
                            value=17,
                            step=1,
                            info="Total number of floors"
                        )
                        
                        with gr.Accordion("📊 Optional: Detailed Area Breakdown", open=False):
                            area_without_balcony = gr.Number(
                                label="Area Without Balcony (m²)",
                                value=None,
                                minimum=0,
                                info="Leave empty to auto-estimate"
                            )
                            living_area = gr.Number(
                                label="Living Area (m²)",
                                value=None,
                                minimum=0,
                                info="Leave empty to auto-estimate"
                            )
                            kitchen_area = gr.Number(
                                label="Kitchen Area (m²)",
                                value=None,
                                minimum=0,
                                info="Leave empty to auto-estimate"
                            )
                            hallway_area = gr.Number(
                                label="Hallway Area (m²)",
                                value=None,
                                minimum=0,
                                info="Leave empty to auto-estimate"
                            )
                    
                    with gr.Column():
                        gr.Markdown("### 🏢 Property Details")
                        
                        district = gr.Dropdown(
                            choices=district_choices,
                            label="District / Location (Район)",
                            value=district_choices[0] if district_choices else None,
                            info="Development location - affects price significantly"
                        )
                        
                        class_option = gr.Dropdown(
                            choices=class_choices,
                            label="Class",
                            value="Комфорт",
                            info="Property class category"
                        )
                        
                        property_type = gr.Dropdown(
                            choices=property_type_choices,
                            label="Property Type",
                            value="2 ккв",
                            info="Number of rooms"
                        )
                        
                        building_type = gr.Dropdown(
                            choices=building_type_choices,
                            label="Building Type",
                            value="Монолит",
                            info="Construction type"
                        )
                        
                        property_category = gr.Dropdown(
                            choices=property_category_choices,
                            label="Property Category",
                            value="Многокв. дом"
                        )
                        
                        finishing = gr.Dropdown(
                            choices=finishing_choices,
                            label="Finishing",
                            value="Чистовая",
                            info="Level of interior finishing"
                        )
                        
                        phase = gr.Slider(
                            label="Phase",
                            minimum=1,
                            maximum=6,
                            value=1,
                            step=1,
                            info="Construction phase"
                        )
                        
                        with gr.Accordion("🔧 Additional Options", open=False):
                            apartments = gr.Dropdown(
                                choices=apartments_choices,
                                label="Apartments",
                                value="Нет"
                            )
                            
                            apartment_option = gr.Dropdown(
                                choices=apartment_option_choices,
                                label="Apartment Option",
                                value="Новостройка"
                            )
                            
                            mortgage = gr.Dropdown(
                                choices=mortgage_choices,
                                label="Mortgage Available",
                                value="Да"
                            )
                            
                            subsidies = gr.Dropdown(
                                choices=subsidies_choices,
                                label="Subsidies Available",
                                value="Нет"
                            )
                            
                            layout = gr.Dropdown(
                                choices=layout_choices,
                                label="Layout",
                                value="Да"
                            )
                
                # Prediction button and output for manual input
                predict_btn = gr.Button("🔮 Predict Price", variant="primary", size="lg")
                output_manual = gr.Markdown(label="Prediction Result")
                
                predict_btn.click(
                    fn=predict_price,
                    inputs=[
                        total_area, floors_total, phase, floor, ceiling_height,
                        class_option, building_type, property_type, property_category,
                        apartments, finishing, apartment_option, mortgage, subsidies, layout,
                        district, area_without_balcony, living_area, kitchen_area, hallway_area
                    ],
                    outputs=output_manual
                )
            
            # Tab 2: Link-Based Input
            with gr.Tab("🔗 Link-Based Input"):
                gr.Markdown("""
                ### Web Scraping Feature (Coming Soon)
                
                This feature would allow you to provide a direct link to a SAMOLET property listing,
                and the system would automatically extract apartment characteristics from the webpage.
                
                **Example Link:**
                ```
                https://samolet.ru/project/oktyabrskaya-98/flats/308985/
                ```
                
                **Current Status:** 🔴 **Not Available**
                
                Due to SAMOLET's advanced anti-bot protection, automated web scraping is currently blocked:
                
                **What was tried:**
                1. **web_scraper.py** - Basic requests-based scraper
                   - Result: HTTP 403 Forbidden (blocked by anti-bot protection)
                
                2. **browser_scraper.py** - Playwright browser automation
                   - Result: HTTP 403 Forbidden (browser fingerprint detected)
                
                3. **crawl4ai_scraper.py** - Open-source AI-powered crawling
                   - Result: HTTP 403 Forbidden ("Access to samolet.ru is forbidden")
                   - IP-based blocking detected
                
                4. **firecrawl_scraper.py** - Cloud-based scraping service
                   - Requires API key and credits
                   - More likely to succeed but has usage costs
                
                **Why it doesn't work:**
                - SAMOLET uses Cloudflare/CDN protection that detects and blocks automated access
                - Browser fingerprinting detects headless/automated browsers
                - IP addresses get temporarily or permanently blocked
                - 403 Forbidden errors with "Guru meditation" responses
                
                **Alternative:**
                Please use the **"Manual Input"** tab to enter apartment characteristics directly.
                The model can still provide accurate predictions with manual input of:
                - Total Area
                - District/Location
                - Property Type (rooms)
                - Class
                - Building Type
                - Finishing level
                - Floor information
                - Ceiling Height
                """)
                
                with gr.Row():
                    with gr.Column():
                        gr.Textbox(
                            label="Property URL (Disabled)",
                            value="https://samolet.ru/project/...",
                            interactive=False,
                            info="🔗 Link-based extraction is currently unavailable due to anti-bot protection"
                        )
                        
                    with gr.Column():
                        gr.Button(
                            "🚫 Extract Data from Link",
                            variant="secondary",
                            interactive=False,
                            size="lg"
                        )
            
            # Tab 3: Test Data Evaluation
            with gr.Tab("📊 Test Data Evaluation"):
                gr.Markdown(f"""
                ### Evaluate Model on Test Dataset
                
                Browse through **{len(test_data_preprocessed)} test samples** and compare predicted vs actual prices.
                The test data has been preprocessed and scaled using the same pipeline as training.
                """)
                
                with gr.Row():
                    sample_slider = gr.Slider(
                        label="Select Test Sample",
                        minimum=0,
                        maximum=len(test_data_preprocessed) - 1,
                        value=0,
                        step=1,
                        info=f"Choose a sample from 0 to {len(test_data_preprocessed) - 1}"
                    )
                
                evaluate_btn = gr.Button("📈 Evaluate Sample", variant="primary", size="lg")
                output_test = gr.Markdown(label="Evaluation Result")
                
                evaluate_btn.click(
                    fn=predict_from_test_data,
                    inputs=[sample_slider],
                    outputs=output_test
                )
        gr.Markdown("""
        ---
        ### ℹ️ Model Information
        
        - **Client:** SAMOLET Group (ПАО «ГК «Самолет») - Russia's leading residential developer
        - **Model:** Random Forest Regressor (Tuned) - R² = 0.9786
        - **Features Used:** TotalArea, Class, CeilingHeight, FloorsTotal, Phase, Finishing, District (mean-encoded), PropertyType (one-hot)
        - **Preprocessing:** sklearn OrdinalEncoder, OneHotEncoder, TargetEncoder (mean encoding), StandardScaler, MinMaxScaler
        - **Training Data:** SAMOLET property portfolio, primarily Moscow region (113 locations)
        """)
    
    return demo


# Main execution
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Starting SAMOLET Apartment Price Prediction Application")
    logger.info("=" * 60)
    logger.info(f"Server: 127.0.0.1:7860")
    logger.info(f"Model: Random Forest Regressor (R² = 0.9786)")
    logger.info("=" * 60)
    
    demo = create_gradio_interface()
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )
