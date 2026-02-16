"""
Integration Script: FireCrawl Scraper + Price Prediction
Connects web scraping with the apartment price prediction model

Setup:
    1. Create .env file: echo "FIRECRAWL_API_KEY=your-key" > .env
    2. Install: pip install firecrawl-py python-dotenv
    3. Run: python firecrawl_predict.py --url "https://samolet.ru/..."

Workflow:
    1. Scrape apartment data from URL using FireCrawl
    2. Extract structured features
    3. Preprocess for model input
    4. Run price prediction
    5. Display results with comparison
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Optional
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded .env file")
except ImportError:
    pass  # python-dotenv is optional, will try environment variables

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from firecrawl_scraper import FireCrawlScraper, ApartmentData, save_raw_output
    from main import predict_price, preprocess_input
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure firecrawl_scraper.py and main.py are in the same directory")
    sys.exit(1)


def map_scraped_to_prediction(data: ApartmentData) -> Dict:
    """
    Map scraped apartment data to prediction model inputs
    
    Args:
        data: Scraped ApartmentData
        
    Returns:
        Dictionary of parameters for predict_price function
    """
    # Default values for missing fields
    prediction_params = {
        'total_area': data.total_area or 50.0,
        'floors_total': data.floors_total or 17,
        'phase': data.phase or 1,
        'floor': data.floor or 5,
        'ceiling_height': data.ceiling_height or 2.7,
        'class_option': map_class(data.class_option),
        'building_type': map_building_type(data.building_type),
        'property_type': map_property_type(data.property_type),
        'property_category': 'Многокв. дом',  # Default
        'apartments': 'Нет',  # Default
        'finishing': map_finishing(data.finishing),
        'apartment_option': 'Новостройка',  # Default
        'mortgage': 'Да',  # Default
        'subsidies': 'Нет',  # Default
        'layout': 'Да',  # Default
        'district': data.district or 'Москва',  # Default or extracted
        'area_without_balcony': data.area_without_balcony,
        'living_area': data.living_area,
        'kitchen_area': data.kitchen_area,
        'hallway_area': data.hallway_area,
    }
    
    return prediction_params


def map_class(class_value: Optional[str]) -> str:
    """Map scraped class to model categories"""
    if not class_value:
        return 'Комфорт'  # Default
    
    class_lower = class_value.lower()
    class_mapping = {
        'эконом': 'Эконом',
        'комфорт': 'Комфорт',
        'бизнес': 'Бизнес',
        'элит': 'Элит',
        'elite': 'Элит',
        'business': 'Бизнес',
        'comfort': 'Комфорт',
        'econom': 'Эконом',
    }
    
    for key, value in class_mapping.items():
        if key in class_lower:
            return value
    
    return 'Комфорт'  # Default


def map_building_type(building_type: Optional[str]) -> str:
    """Map scraped building type to model categories"""
    if not building_type:
        return 'Монолит'  # Default
    
    type_lower = building_type.lower()
    type_mapping = {
        'монолит': 'Монолит',
        'панель': 'Панель',
        'кирпич': 'Кирпич',
        'кирпич-монолит': 'Кирпич-монолит',
        'brick': 'Кирпич',
        'panel': 'Панель',
        'monolith': 'Монолит',
    }
    
    for key, value in type_mapping.items():
        if key in type_lower:
            return value
    
    return 'Монолит'  # Default


def map_property_type(property_type: Optional[str]) -> str:
    """Map scraped property type to model categories"""
    if not property_type:
        return '2 ккв'  # Default
    
    type_lower = property_type.lower()
    
    # Check for studio
    if 'студия' in type_lower or 'studio' in type_lower:
        return 'Студия'
    
    # Check for apartments
    if 'апартамент' in type_lower or 'apartment' in type_lower:
        return 'Апартаменты'
    
    # Extract number of rooms
    import re
    match = re.search(r'(\d+)', str(property_type))
    if match:
        rooms = int(match.group(1))
        if 1 <= rooms <= 5:
            return f'{rooms} ккв'
    
    return '2 ккв'  # Default


def map_finishing(finishing: Optional[str]) -> str:
    """Map scraped finishing to model categories"""
    if not finishing:
        return 'Чистовая'  # Default
    
    finishing_lower = finishing.lower()
    finishing_mapping = {
        'чистовая': 'Чистовая',
        'подчистовая': 'Подчистовая',
        'без отделки': 'Без отделки',
        'нет данных': 'Нет данных',
        'с мебелью': 'С мебелью',
        'white box': 'Подчистовая',
        'clean': 'Чистовая',
        'rough': 'Без отделки',
        'none': 'Без отделки',
    }
    
    for key, value in finishing_mapping.items():
        if key in finishing_lower:
            return value
    
    return 'Чистовая'  # Default


def scrape_and_predict(url: str, use_cache: bool = True, save_output: bool = False) -> Optional[Dict]:
    """
    Complete pipeline: Scrape → Extract → Predict
    
    Args:
        url: Apartment listing URL
        use_cache: Whether to use cached results
        save_output: Whether to save raw output to file
        
    Returns:
        Dictionary with scraped data and prediction, or None if failed
    """
    print("\n" + "="*60)
    print("  🔥 FireCrawl → Price Prediction Pipeline")
    print("="*60)
    
    # Step 1: Initialize scraper
    api_key = os.getenv('FIRECRAWL_API_KEY')
    if not api_key:
        print("\n❌ FireCrawl API key not found!")
        print("Set FIRECRAWL_API_KEY environment variable")
        return None
    
    try:
        scraper = FireCrawlScraper(api_key=api_key, save_output=save_output)
    except Exception as e:
        print(f"\n❌ Failed to initialize scraper: {e}")
        return None
    
    # Step 2: Scrape data
    print(f"\n📥 Step 1: Scraping data from URL...")
    scraped_data = scraper.scrape_with_extraction(url, use_cache=use_cache)
    
    if not scraped_data:
        print("\n❌ Failed to scrape data")
        return None
    
    print(f"✅ Scraped successfully!")
    
    # Save raw output if requested
    if save_output and scraped_data.raw_content:
        save_raw_output(url, scraped_data, scraped_data.raw_content)
    
    # Step 3: Map to prediction inputs
    print(f"\n🔄 Step 2: Mapping scraped data to model inputs...")
    prediction_params = map_scraped_to_prediction(scraped_data)
    
    # Show mapped values
    print("\n  Mapped inputs:")
    for key, value in prediction_params.items():
        if value is not None and key not in ['area_without_balcony', 'living_area', 'kitchen_area', 'hallway_area']:
            print(f"    • {key}: {value}")
    
    # Step 4: Run prediction
    print(f"\n🤖 Step 3: Running price prediction...")
    try:
        prediction_result = predict_price(**prediction_params)
        
        # Parse prediction result (it's a formatted string)
        # Extract predicted price from the result
        import re
        price_match = re.search(r'\*\*Total Price:\*\* ([\d\s,]+) ₽', prediction_result)
        predicted_price = None
        if price_match:
            price_str = price_match.group(1).replace(' ', '').replace(',', '')
            predicted_price = float(price_str)
        
        # Step 5: Compile results
        result = {
            'url': url,
            'scraped_data': scraped_data.to_dict(),
            'prediction_params': prediction_params,
            'prediction_result': prediction_result,
            'predicted_price': predicted_price,
            'listed_price': scraped_data.price,
            'price_difference': None,
            'price_difference_percent': None,
        }
        
        # Calculate price difference if both prices available
        if predicted_price and scraped_data.price:
            diff = predicted_price - scraped_data.price
            diff_percent = (diff / scraped_data.price) * 100
            result['price_difference'] = diff
            result['price_difference_percent'] = diff_percent
        
        return result
        
    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_results(result: Dict):
    """Print formatted results"""
    print("\n" + "="*60)
    print("  📊 PREDICTION RESULTS")
    print("="*60)
    
    scraped = result['scraped_data']
    
    # Scraped info
    print(f"\n🏠 Scraped Information:")
    print(f"  Project: {scraped.get('project_name', 'N/A')}")
    print(f"  Total Area: {scraped.get('total_area')} m²" if scraped.get('total_area') else "  Total Area: N/A")
    print(f"  Floor: {scraped.get('floor')}/{scraped.get('floors_total')}" if scraped.get('floor') else "  Floor: N/A")
    print(f"  Property Type: {scraped.get('property_type', 'N/A')}")
    print(f"  District: {scraped.get('district', 'N/A')}")
    
    # Prediction
    print(f"\n{result['prediction_result']}")
    
    # Price comparison
    if result['listed_price'] and result['predicted_price']:
        print(f"\n📈 Price Comparison:")
        print(f"  Listed Price: {result['listed_price']:,.0f} ₽")
        print(f"  Predicted Price: {result['predicted_price']:,.0f} ₽")
        print(f"  Difference: {result['price_difference']:+,.0f} ₽ ({result['price_difference_percent']:+.2f}%)")
        
        if abs(result['price_difference_percent']) < 10:
            print(f"  ✅ Price is within expected range")
        elif result['price_difference'] > 0:
            print(f"  ⚠️  Listed price is LOWER than predicted (potential deal)")
        else:
            print(f"  ⚠️  Listed price is HIGHER than predicted")


def save_prediction_results(result: Dict, output_dir: str = "prediction_results"):
    """Save prediction results to JSON"""
    from datetime import datetime
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    flat_id = result['url'].rstrip('/').split('/')[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"prediction_{flat_id}_{timestamp}.json"
    filepath = output_path / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Results saved to: {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(
        description='Scrape apartment from URL and predict price',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Setup (create .env file):
  echo "FIRECRAWL_API_KEY=your-api-key-here" > .env
  pip install firecrawl-py python-dotenv

Examples:
  # Basic usage:
  python firecrawl_predict.py --url "https://samolet.ru/project/oktyabrskaya-98/flats/308985/"
  
  # Save results:
  python firecrawl_predict.py --url "..." --save
  
  # Force refresh (no cache):
  python firecrawl_predict.py --url "..." --no-cache

Prerequisites:
  1. FireCrawl API key in .env file (get at firecrawl.dev)
  2. Model artifacts in model_artifacts/ directory
  3. Trained model from main.py
        """
    )
    
    parser.add_argument('--url', type=str, required=True,
                        help='Apartment listing URL to scrape and predict')
    parser.add_argument('--save', action='store_true',
                        help='Save results to JSON file')
    parser.add_argument('--save-output', action='store_true',
                        help='Save raw scraper output to .txt file for review')
    parser.add_argument('--no-cache', action='store_true',
                        help='Skip cache, fetch fresh data')
    
    args = parser.parse_args()
    
    # Run pipeline
    result = scrape_and_predict(args.url, use_cache=not args.no_cache, save_output=args.save_output)
    
    if result:
        print_results(result)
        
        if args.save:
            save_prediction_results(result)
        
        print("\n✅ Pipeline completed successfully!")
    else:
        print("\n❌ Pipeline failed")
        print("\nTroubleshooting:")
        print("  • Check FireCrawl API key and credits")
        print("  • Verify model artifacts are present")
        print("  • Ensure URL is accessible")


if __name__ == "__main__":
    main()
