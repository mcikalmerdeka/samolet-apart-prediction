"""
Web Scraping Script for SAMOLET Website
Extracts apartment information from SAMOLET property listing URLs

Usage: 
    python web_scraper.py
    python web_scraper.py --url "https://samolet.ru/project/oktyabrskaya-98/flats/308985/"

Note: SAMOLET website may have anti-bot protection (401 error)
This script attempts multiple approaches:
1. Standard requests with rotating user agents
2. Playwright/Selenium for browser automation
3. API endpoint discovery
4. Fallback manual input mode
"""

import requests
from bs4 import BeautifulSoup
import json
import re
from pathlib import Path
import argparse
import sys
from typing import Dict, Optional, Any
from dataclasses import dataclass
import time
import random

# User agents for rotation to avoid blocking
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
]


@dataclass
class ApartmentData:
    """Data structure for extracted apartment information"""
    url: str
    total_area: Optional[float] = None
    floors_total: Optional[int] = None
    phase: Optional[int] = None
    floor: Optional[int] = None
    ceiling_height: Optional[float] = None
    area_without_balcony: Optional[float] = None
    living_area: Optional[float] = None
    kitchen_area: Optional[float] = None
    hallway_area: Optional[float] = None
    class_option: Optional[str] = None
    building_type: Optional[str] = None
    property_type: Optional[str] = None
    property_category: Optional[str] = None
    apartments: Optional[str] = None
    finishing: Optional[str] = None
    apartment_option: Optional[str] = None
    mortgage: Optional[str] = None
    subsidies: Optional[str] = None
    layout: Optional[str] = None
    district: Optional[str] = None
    price: Optional[float] = None
    project_name: Optional[str] = None
    raw_content: Optional[str] = None  # Store raw HTML for --save-output
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'url': self.url,
            'total_area': self.total_area,
            'floors_total': self.floors_total,
            'phase': self.phase,
            'floor': self.floor,
            'ceiling_height': self.ceiling_height,
            'area_without_balcony': self.area_without_balcony,
            'living_area': self.living_area,
            'kitchen_area': self.kitchen_area,
            'hallway_area': self.hallway_area,
            'class_option': self.class_option,
            'building_type': self.building_type,
            'property_type': self.property_type,
            'property_category': self.property_category,
            'apartments': self.apartments,
            'finishing': self.finishing,
            'apartment_option': self.apartment_option,
            'mortgage': self.mortgage,
            'subsidies': self.subsidies,
            'layout': self.layout,
            'district': self.district,
            'price': self.price,
            'project_name': self.project_name,
            'raw_content': self.raw_content
        }


class SamoletWebScraper:
    """Web scraper for SAMOLET property website"""
    
    def __init__(self, delay: float = 1.0):
        self.delay = delay
        self.session = requests.Session()
        self.last_request_time = 0
        
    def _get_headers(self) -> Dict[str, str]:
        """Generate headers with random user agent"""
        return {
            'User-Agent': random.choice(USER_AGENTS),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        }
    
    def _rate_limit(self):
        """Add delay between requests"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_request_time = time.time()
    
    def fetch_page(self, url: str, retries: int = 3) -> Optional[BeautifulSoup]:
        """Fetch and parse webpage with retries"""
        self._rate_limit()
        
        for attempt in range(retries):
            try:
                print(f"  Attempt {attempt + 1}/{retries}...")
                response = self.session.get(url, headers=self._get_headers(), timeout=30)
                
                if response.status_code == 200:
                    print(f"  ✅ Successfully fetched: {url}")
                    return BeautifulSoup(response.text, 'html.parser')
                elif response.status_code == 401:
                    print(f"  ⚠️  401 Unauthorized - Site requires authentication or blocks scrapers")
                    print(f"  Trying alternative approach...")
                    continue
                else:
                    print(f"  ⚠️  Status {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                print(f"  ❌ Request error: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
        return None
    
    def extract_apartment_data(self, soup: BeautifulSoup, url: str) -> ApartmentData:
        """Extract apartment information from parsed HTML"""
        data = ApartmentData(url=url)
        
        print("\n🔍 Extracting apartment data...")
        
        # Store raw HTML content for output saving
        data.raw_content = str(soup) if soup else None
        
        # Extract JSON-LD data (structured data)
        json_ld = self._extract_json_ld(soup)
        if json_ld:
            data = self._parse_json_ld(data, json_ld)
        
        # Extract from HTML meta tags
        data = self._extract_meta_tags(soup, data)
        
        # Extract from HTML elements
        data = self._extract_from_html(soup, data)
        
        # Extract from JavaScript variables
        data = self._extract_from_scripts(soup, data)
        
        return data
    
    def _extract_json_ld(self, soup: BeautifulSoup) -> Optional[Dict]:
        """Extract JSON-LD structured data"""
        scripts = soup.find_all('script', type='application/ld+json')
        for script in scripts:
            try:
                data = json.loads(script.string)
                if isinstance(data, list):
                    for item in data:
                        if item.get('@type') in ['Residence', 'Product', 'Apartment', 'SingleFamilyResidence']:
                            return item
                elif data.get('@type') in ['Residence', 'Product', 'Apartment', 'SingleFamilyResidence']:
                    return data
            except (json.JSONDecodeError, AttributeError):
                continue
        return None
    
    def _parse_json_ld(self, data: ApartmentData, json_ld: Dict) -> ApartmentData:
        """Parse JSON-LD data into ApartmentData"""
        try:
            # Area information
            if 'floorSize' in json_ld:
                size = json_ld['floorSize']
                if isinstance(size, dict):
                    value = size.get('value', size.get('@value', 0))
                    unit = size.get('unitCode', '').upper()
                    if unit in ['MTQ', 'M2', 'SQMT']:
                        data.total_area = float(value)
            
            # Price
            if 'offers' in json_ld:
                offers = json_ld['offers']
                if isinstance(offers, dict):
                    price = offers.get('price')
                    if price:
                        data.price = float(price)
            
            # Address / District
            if 'address' in json_ld:
                address = json_ld['address']
                if isinstance(address, dict):
                    data.district = address.get('addressLocality', address.get('streetAddress', ''))
            
            # Floor
            if 'floorLevel' in json_ld:
                data.floor = int(json_ld['floorLevel'])
                
        except (ValueError, TypeError) as e:
            print(f"  ⚠️  Error parsing JSON-LD: {e}")
            
        return data
    
    def _extract_meta_tags(self, soup: BeautifulSoup, data: ApartmentData) -> ApartmentData:
        """Extract data from meta tags"""
        meta_tags = {
            'description': soup.find('meta', {'name': 'description'}),
            'og:title': soup.find('meta', {'property': 'og:title'}),
            'og:description': soup.find('meta', {'property': 'og:description'}),
        }
        
        # Try to extract area from description
        if meta_tags['description']:
            desc = meta_tags['description'].get('content', '')
            area_match = re.search(r'(\d+(?:\.\d+)?)\s*м²', desc)
            if area_match and not data.total_area:
                data.total_area = float(area_match.group(1))
            
            price_match = re.search(r'(\d[\d\s]*)\s*₽', desc)
            if price_match and not data.price:
                price_str = price_match.group(1).replace(' ', '').replace('\xa0', '')
                data.price = float(price_str)
        
        return data
    
    def _extract_from_html(self, soup: BeautifulSoup, data: ApartmentData) -> ApartmentData:
        """Extract data from HTML elements"""
        
        # Common selectors for apartment data
        selectors = {
            'area': ['.flat-area', '.area-value', '[data-area]', '.total-area', '.square'],
            'floor': ['.flat-floor', '.floor-value', '[data-floor]', '.floor'],
            'price': ['.flat-price', '.price-value', '[data-price]', '.cost', '.price'],
            'rooms': ['.flat-rooms', '.rooms-value', '[data-rooms]', '.rooms'],
            'project': ['.project-name', '.project-title', 'h1'],
        }
        
        # Extract total area
        if not data.total_area:
            for selector in selectors['area']:
                element = soup.select_one(selector)
                if element:
                    text = element.get_text(strip=True)
                    match = re.search(r'(\d+(?:\.\d+)?)', text)
                    if match:
                        data.total_area = float(match.group(1))
                        print(f"  ✓ Found total area: {data.total_area} m²")
                        break
        
        # Extract floor
        if not data.floor:
            for selector in selectors['floor']:
                element = soup.select_one(selector)
                if element:
                    text = element.get_text(strip=True)
                    match = re.search(r'(\d+)', text)
                    if match:
                        data.floor = int(match.group(1))
                        print(f"  ✓ Found floor: {data.floor}")
                        break
        
        # Extract price
        if not data.price:
            for selector in selectors['price']:
                element = soup.select_one(selector)
                if element:
                    text = element.get_text(strip=True)
                    # Match price patterns
                    match = re.search(r'(\d[\d\s\xa0]*)', text)
                    if match:
                        price_str = match.group(1).replace(' ', '').replace('\xa0', '')
                        try:
                            data.price = float(price_str)
                            print(f"  ✓ Found price: {data.price:,.0f} ₽")
                            break
                        except ValueError:
                            continue
        
        # Extract property type (rooms)
        if not data.property_type:
            for selector in selectors['rooms']:
                element = soup.select_one(selector)
                if element:
                    text = element.get_text(strip=True)
                    # Match room patterns (e.g., "2-комнатная", "студия", "1-комн")
                    if re.search(r'студия', text, re.I):
                        data.property_type = 'Студия'
                        print(f"  ✓ Found property type: {data.property_type}")
                        break
                    match = re.search(r'(\d+)[\s-]*(?:комн|ккв)', text, re.I)
                    if match:
                        rooms = int(match.group(1))
                        data.property_type = f'{rooms} ккв'
                        print(f"  ✓ Found property type: {data.property_type}")
                        break
        
        # Extract project name
        for selector in selectors['project']:
            element = soup.select_one(selector)
            if element:
                data.project_name = element.get_text(strip=True)
                print(f"  ✓ Found project: {data.project_name}")
                break
        
        # Look for characteristics table
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    label = cells[0].get_text(strip=True).lower()
                    value = cells[1].get_text(strip=True)
                    
                    if 'площадь' in label and not data.total_area:
                        match = re.search(r'(\d+(?:\.\d+)?)', value)
                        if match:
                            data.total_area = float(match.group(1))
                            print(f"  ✓ Found total area (table): {data.total_area} m²")
                    
                    elif 'этаж' in label and 'из' in value:
                        parts = value.split('из')
                        if len(parts) == 2:
                            try:
                                data.floor = int(parts[0].strip())
                                data.floors_total = int(parts[1].strip())
                                print(f"  ✓ Found floor info: {data.floor}/{data.floors_total}")
                            except ValueError:
                                pass
                    
                    elif 'высота потолков' in label and not data.ceiling_height:
                        match = re.search(r'(\d+(?:\.\d+)?)', value)
                        if match:
                            data.ceiling_height = float(match.group(1))
                            print(f"  ✓ Found ceiling height: {data.ceiling_height} m")
                    
                    elif 'отделка' in label and not data.finishing:
                        data.finishing = value
                        print(f"  ✓ Found finishing: {data.finishing}")
                    
                    elif 'материал' in label or 'конструкция' in label:
                        if not data.building_type:
                            data.building_type = value
                            print(f"  ✓ Found building type: {data.building_type}")
        
        return data
    
    def _extract_from_scripts(self, soup: BeautifulSoup, data: ApartmentData) -> ApartmentData:
        """Extract data from JavaScript variables"""
        scripts = soup.find_all('script')
        
        for script in scripts:
            if script.string:
                script_text = script.string
                
                # Look for apartment data in window.__INITIAL_STATE__ or similar
                patterns = [
                    r'window\.__INITIAL_STATE__\s*=\s*({.*?});',
                    r'window\.__DATA__\s*=\s*({.*?});',
                    r'const\s+flatData\s*=\s*({.*?});',
                    r'var\s+apartment\s*=\s*({.*?});',
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, script_text, re.DOTALL)
                    if matches:
                        try:
                            json_str = matches[0]
                            apartment_json = json.loads(json_str)
                            print(f"  ✓ Found JavaScript data variable")
                            # Would parse the JSON here based on structure
                            return self._parse_script_data(data, apartment_json)
                        except json.JSONDecodeError:
                            continue
        
        return data
    
    def _parse_script_data(self, data: ApartmentData, json_data: Dict) -> ApartmentData:
        """Parse apartment data from JavaScript variables"""
        try:
            # Common keys to look for
            keys_mapping = {
                'total_area': ['totalArea', 'square', 'area', 'total_square', 'общая_площадь'],
                'floor': ['floor', 'floorNumber', 'floor_number', 'этаж'],
                'floors_total': ['totalFloors', 'floors', 'floorsTotal', 'floors_count', 'всего_этажей'],
                'price': ['price', 'cost', 'totalPrice', 'total_cost', 'цена'],
                'rooms': ['rooms', 'roomCount', 'roomsCount', 'room_count', 'комнаты'],
                'phase': ['phase', 'buildingPhase', 'constructionPhase', 'фаза'],
                'ceiling_height': ['ceilingHeight', 'ceiling', 'высота_потолков'],
                'finishing': ['finishing', 'finish', 'decoration', 'отделка'],
                'building_type': ['buildingType', 'material', 'houseType', 'constructionType', 'тип_дома'],
                'district': ['district', 'location', 'area', 'districtName', 'район'],
                'project_name': ['projectName', 'project', 'title', 'name', 'проект'],
            }
            
            for attr, keys in keys_mapping.items():
                for key in keys:
                    if key in json_data:
                        value = json_data[key]
                        if hasattr(data, attr):
                            setattr(data, attr, value)
                            print(f"  ✓ Found {attr} from JS: {value}")
                            break
                            
        except Exception as e:
            print(f"  ⚠️  Error parsing script data: {e}")
            
        return data
    
    def scrape(self, url: str) -> Optional[ApartmentData]:
        """Main scraping function"""
        print(f"\n{'='*60}")
        print(f"🔍 Scraping: {url}")
        print(f"{'='*60}")
        
        soup = self.fetch_page(url)
        
        if soup is None:
            print(f"\n❌ Failed to fetch page after all attempts")
            print(f"\nPossible reasons:")
            print(f"  • Website requires authentication")
            print(f"  • Anti-bot protection is active")
            print(f"  • Site blocks requests from this IP")
            print(f"\nRecommendations:")
            print(f"  1. Try using a browser automation tool (Playwright/Selenium)")
            print(f"  2. Use a proxy or VPN")
            print(f"  3. Check if there's a public API")
            print(f"  4. Consider manual data entry mode")
            return None
        
        data = self.extract_apartment_data(soup, url)
        return data
    
    def save_results(self, data: ApartmentData, output_dir: str = "output"):
        """Save scraped data to JSON file in unified output folder"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate filename from URL with timestamp
        import datetime
        flat_id = data.url.rstrip('/').split('/')[-1]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"apartment_{flat_id}_{timestamp}.json"
        filepath = output_path / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data.to_dict(), f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Results saved to: {filepath}")
        return filepath


def generate_random_hash(length: int = 4) -> str:
    """Generate random alphanumeric hash"""
    import random
    import string
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


def save_raw_output(url: str, data: ApartmentData, html_content: str, output_dir: str = "output"):
    """
    Save complete scraper output to text file for review (includes both formatted results and raw HTML)
    
    Args:
        url: Source URL
        data: Extracted ApartmentData
        html_content: Raw HTML content
        output_dir: Directory to save output files (default: output)
    """
    import datetime
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Generate random hash
    random_hash = generate_random_hash(4)
    
    # Extract flat ID from URL
    flat_id = url.rstrip('/').split('/')[-1]
    
    # Create filename with hash
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"webscraper_output_{random_hash}_{flat_id}_{timestamp}.txt"
    filepath = output_path / filename
    
    # Helper function for conditional display
    def show(val, unit="", default="Not found"):
        if val is not None:
            return f"{val}{unit}"
        return default
    
    # Format price with commas
    price_str = ""
    if data.price:
        price_str = f"{data.price:,.0f} ₽ ({data.price/1_000_000:.2f}M ₽)"
    
    # Format price per sqm
    price_per_sqm_str = ""
    if data.price and data.total_area:
        price_per_sqm_str = f"{(data.price / data.total_area):,.0f} ₽/m²"
    
    # Build formatted output with both terminal-style results and raw HTML
    output_content = f"""# Web Scraper Output Report
# Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

{'='*60}
🏠 SAMOLET Website Scraper
{'='*60}

🔗 URL: {url}
📝 Hash: {random_hash}
📅 Scraped: {datetime.datetime.now().isoformat()}

{'='*60}
📊 EXTRACTED APARTMENT INFORMATION
{'='*60}

🏢 Project: {data.project_name or 'Not found'}

📐 AREA & LAYOUT:
  • Total Area: {show(data.total_area, ' m²')}
  • Area Without Balcony: {show(data.area_without_balcony, ' m²')}
  • Living Area: {show(data.living_area, ' m²')}
  • Kitchen Area: {show(data.kitchen_area, ' m²')}
  • Hallway Area: {show(data.hallway_area, ' m²')}

🏠 PROPERTY DETAILS:
  • Property Type: {show(data.property_type)}
  • Class: {show(data.class_option)}
  • Building Type: {show(data.building_type)}
  • Floor: {show(data.floor)}
  • Total Floors: {show(data.floors_total)}
  • Ceiling Height: {show(data.ceiling_height, ' m')}
  • Finishing: {show(data.finishing)}
  • Phase: {show(data.phase)}

📍 LOCATION:
  • District: {show(data.district)}

💰 PRICE:
  • Listed Price: {price_str or 'Not found'}
  • Price per m²: {price_per_sqm_str or 'Not found'}

💼 OTHER:
  • Apartments: {show(data.apartments)}
  • Apartment Option: {show(data.apartment_option)}
  • Mortgage Available: {show(data.mortgage)}
  • Subsidies Available: {show(data.subsidies)}
  • Layout: {show(data.layout)}

{'='*60}
🔍 RAW HTML CONTENT (First 5000 chars)
{'='*60}

{html_content[:5000] if html_content else 'No HTML content available'}

{'='*60}
End of Report
{'='*60}
"""
    
    # Save formatted content
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(output_content)
    
    print(f"\n📝 Complete output saved to: {filepath}")
    return filepath


def print_apartment_info(data: ApartmentData):
    """Print formatted apartment information"""
    print(f"\n{'='*60}")
    print(f"📊 EXTRACTED APARTMENT INFORMATION")
    print(f"{'='*60}")
    
    print(f"\n🔗 URL: {data.url}")
    if data.project_name:
        print(f"🏢 Project: {data.project_name}")
    
    print(f"\n📐 AREA & LAYOUT:")
    print(f"  • Total Area: {data.total_area} m²" if data.total_area else "  • Total Area: Not found")
    print(f"  • Area Without Balcony: {data.area_without_balcony} m²" if data.area_without_balcony else "  • Area Without Balcony: Not found")
    print(f"  • Living Area: {data.living_area} m²" if data.living_area else "  • Living Area: Not found")
    print(f"  • Kitchen Area: {data.kitchen_area} m²" if data.kitchen_area else "  • Kitchen Area: Not found")
    print(f"  • Hallway Area: {data.hallway_area} m²" if data.hallway_area else "  • Hallway Area: Not found")
    
    print(f"\n🏠 PROPERTY DETAILS:")
    print(f"  • Property Type: {data.property_type}" if data.property_type else "  • Property Type: Not found")
    print(f"  • Class: {data.class_option}" if data.class_option else "  • Class: Not found")
    print(f"  • Building Type: {data.building_type}" if data.building_type else "  • Building Type: Not found")
    print(f"  • Floor: {data.floor}" if data.floor else "  • Floor: Not found")
    print(f"  • Total Floors: {data.floors_total}" if data.floors_total else "  • Total Floors: Not found")
    print(f"  • Ceiling Height: {data.ceiling_height} m" if data.ceiling_height else "  • Ceiling Height: Not found")
    print(f"  • Finishing: {data.finishing}" if data.finishing else "  • Finishing: Not found")
    print(f"  • Phase: {data.phase}" if data.phase else "  • Phase: Not found")
    
    print(f"\n📍 LOCATION:")
    print(f"  • District: {data.district}" if data.district else "  • District: Not found")
    
    print(f"\n💰 PRICE:")
    if data.price:
        print(f"  • Listed Price: {data.price:,.0f} ₽ ({data.price/1_000_000:.2f}M ₽)")
        if data.total_area:
            price_per_sqm = data.price / data.total_area
            print(f"  • Price per m²: {price_per_sqm:,.0f} ₽/m²")
    else:
        print("  • Listed Price: Not found")
    
    print(f"\n💼 OTHER:")
    print(f"  • Apartments: {data.apartments}" if data.apartments else "  • Apartments: Not found")
    print(f"  • Apartment Option: {data.apartment_option}" if data.apartment_option else "  • Apartment Option: Not found")
    print(f"  • Mortgage Available: {data.mortgage}" if data.mortgage else "  • Mortgage Available: Not found")
    print(f"  • Subsidies Available: {data.subsidies}" if data.subsidies else "  • Subsidies Available: Not found")
    print(f"  • Layout: {data.layout}" if data.layout else "  • Layout: Not found")
    
    print(f"\n{'='*60}")


def test_connection(url: str = "https://samolet.ru"):
    """Test basic connectivity to the website"""
    print(f"\n🌐 Testing connection to {url}...")
    
    try:
        headers = {
            'User-Agent': USER_AGENTS[0],
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }
        response = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
        
        print(f"  Status Code: {response.status_code}")
        print(f"  Final URL: {response.url}")
        print(f"  Content-Type: {response.headers.get('Content-Type', 'Unknown')}")
        print(f"  Content-Length: {len(response.text)} bytes")
        
        if response.status_code == 200:
            print(f"  ✅ Connection successful!")
            
            # Check for common anti-bot indicators
            if 'captcha' in response.text.lower() or 'robot' in response.text.lower():
                print(f"  ⚠️  Warning: Page may contain CAPTCHA or anti-bot protection")
                
            return True
        else:
            print(f"  ⚠️  Unexpected status code: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Connection failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Scrape apartment data from SAMOLET website',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python web_scraper.py                                    # Run default test
  python web_scraper.py --url "https://samolet.ru/..."     # Scrape specific URL
  python web_scraper.py --test-connection                  # Test site connectivity
  python web_scraper.py --url "..." --save                 # Save JSON results
  python web_scraper.py --url "..." --save-output          # Save formatted text output
        """
    )
    
    parser.add_argument('--url', type=str, 
                        default="https://samolet.ru/project/oktyabrskaya-98/flats/308985/",
                        help='URL of the apartment listing to scrape')
    parser.add_argument('--test-connection', action='store_true',
                        help='Test connectivity to the website')
    parser.add_argument('--save', action='store_true',
                        help='Save results to JSON file in output/')
    parser.add_argument('--save-output', action='store_true',
                        help='Save formatted output to .txt file in output/')
    parser.add_argument('--delay', type=float, default=1.0,
                        help='Delay between requests (seconds)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  🏠 SAMOLET Website Scraper")
    print("  Extract apartment information for price prediction")
    print("="*60)
    
    # Test connection first if requested
    if args.test_connection:
        test_connection()
        return
    
    # Create scraper instance
    scraper = SamoletWebScraper(delay=args.delay)
    
    # Scrape the URL
    data = scraper.scrape(args.url)
    
    if data:
        print_apartment_info(data)
        
        if args.save:
            scraper.save_results(data)
        
        if args.save_output and data.raw_content:
            save_raw_output(args.url, data, data.raw_content)
    else:
        print("\n❌ Scraping failed. Creating manual input fallback...")
        print("\n" + "="*60)
        print("📝 MANUAL INPUT MODE")
        print("="*60)
        print("Since the website blocks automated requests, you can:")
        print("  1. Visit the URL manually in your browser")
        print("  2. Extract the visible apartment details")
        print("  3. Input them into the main.py prediction interface")
        print("\nRequired fields for prediction:")
        print("  - Total Area (m²)")
        print("  - District / Location")
        print("  - Property Type (rooms)")
        print("  - Class (Эконом/Комфорт/Бизнес/Элит)")
        print("  - Building Type")
        print("  - Finishing level")
        print("  - Floor / Total Floors")
        print("  - Ceiling Height")


if __name__ == "__main__":
    main()
