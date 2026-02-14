"""
Gradio Inference Script for Apartment Price Prediction
This script creates an interactive UI for predicting apartment prices using the trained Random Forest model.
"""

import gradio as gr
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# Import sklearn for unpickling
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')

# Load the trained model and feature names
MODEL_PATH = Path('model_artifacts/rf_tuned_model.joblib')
FEATURE_NAMES_PATH = Path('model_artifacts/feature_names.joblib')
SCALING_STATS_PATH = Path('model_artifacts/scaling_stats.joblib')
ENCODINGS_PATH = Path('model_artifacts/encodings.joblib')
TEST_DATA_PREPROCESSED_PATH = Path('data/test_data_5%_preprocessed.csv')
TEST_DATA_RAW_PATH = Path('data/test_data_5%_raw.csv')

try:
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    scaling_stats = joblib.load(SCALING_STATS_PATH)
    encodings = joblib.load(ENCODINGS_PATH)
    
    test_data_preprocessed = pd.read_csv(TEST_DATA_PREPROCESSED_PATH)
    test_data_raw = pd.read_csv(TEST_DATA_RAW_PATH)
    print(f"Model loaded successfully")
    print(f"Expected features: {len(feature_names)} features")
    print(f"Test data loaded: {len(test_data_preprocessed)} samples")
except Exception as e:
    print(f"Error loading model artifacts: {e}")
    raise

# Preprocessing functions (matching notebook preprocessing)
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
    area_without_balcony: float = None,
    living_area: float = None,
    kitchen_area: float = None,
    hallway_area: float = None
):
    """
    Preprocess input data following the same pipeline as in the notebook.
    
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
        'Layout': layout
    }])
    
    # 1. Feature Encoding
    # Ordinal encoding for Class and Finishing (using loaded encodings)
    input_data['Class'] = input_data['Class'].map(encodings['class_mapping'])
    input_data['Finishing'] = input_data['Finishing'].map(encodings['finishing_mapping'])
    
    # One-hot encoding for PropertyType (and other nominals if needed)
    nominal_columns = ["BuildingType", "PropertyType", "PropertyCategory", "Apartments", 
                       "ApartmentOption", "Mortgage", "Subsidies", "Layout"]
    
    input_data = pd.get_dummies(input_data, columns=nominal_columns, prefix=nominal_columns)
    
    # 2. Select only the features used in training
    # Extract base features and PropertyType columns
    selected_features = []
    for feature in feature_names:
        if feature in input_data.columns:
            selected_features.append(feature)
        else:
            # If a one-hot encoded column is missing, add it with value 0
            input_data[feature] = 0
            selected_features.append(feature)
    
    # Ensure columns are in the same order as training
    input_data = input_data[feature_names]
    
    # 3. Feature Scaling (using actual statistics from training data)
    # MinMax scaling for Class, Phase, Finishing
    for col in ['Class', 'Phase', 'Finishing']:
        if col in input_data.columns and col in scaling_stats['minmax']:
            min_val = scaling_stats['minmax'][col]['min']
            max_val = scaling_stats['minmax'][col]['max']
            input_data[col] = (input_data[col] - min_val) / (max_val - min_val)
    
    # Standard scaling for CeilingHeight
    if 'CeilingHeight' in input_data.columns:
        mean_val = scaling_stats['standard']['CeilingHeight']['mean']
        std_val = scaling_stats['standard']['CeilingHeight']['std']
        input_data['CeilingHeight'] = (input_data['CeilingHeight'] - mean_val) / std_val
    
    # Robust scaling for TotalArea and FloorsTotal
    for col in ['TotalArea', 'FloorsTotal']:
        if col in input_data.columns and col in scaling_stats['robust']:
            median_val = scaling_stats['robust'][col]['median']
            iqr_val = scaling_stats['robust'][col]['iqr']
            input_data[col] = (input_data[col] - median_val) / iqr_val
    
    return input_data


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
        # Preprocess the input
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
            area_without_balcony=area_without_balcony,
            living_area=living_area,
            kitchen_area=kitchen_area,
            hallway_area=hallway_area
        )
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        
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
        - Class: {class_option}
        - Finishing: {finishing}
        """
        
        return result
        
    except Exception as e:
        return f"❌ Error during prediction: {str(e)}"


def predict_from_test_data(sample_index: int):
    """
    Load a test sample and make prediction, showing comparison with actual value.
    
    Args:
        sample_index: Index of the test sample to use (0 to len(test_data)-1)
    
    Returns:
        Formatted string with prediction vs actual comparison
    """
    
    try:
        sample_preprocessed = test_data_preprocessed.iloc[sample_index]
        sample_raw = test_data_raw.iloc[sample_index]
        
        X_sample = sample_preprocessed[feature_names].values.reshape(1, -1)
        actual_price = sample_preprocessed['TotalCost']
        
        predicted_price = model.predict(X_sample)[0]
        
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
    
    # Create the interface with organized input sections
    with gr.Blocks(title="🏠 Apartment Price Predictor", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("""
        # 🏠 Apartment Price Prediction System
        
        This tool predicts apartment prices in Moscow based on layout characteristics and property features.
        The model is trained on real estate data and uses a **Random Forest Regressor** with optimized hyperparameters.
        
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
                        area_without_balcony, living_area, kitchen_area, hallway_area
                    ],
                    outputs=output_manual
                )
            
            # Tab 2: Test Data Evaluation
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
        
        - **Model:** Random Forest Regressor (Tuned)
        - **Features Used:** TotalArea, Class, CeilingHeight, FloorsTotal, Phase, Finishing, PropertyType
        
        The model was trained on Moscow apartment data with preprocessing including ordinal encoding, one-hot encoding, and feature scaling.
        """)
    
    return demo


# Main execution
if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )
