"""
Browser Automation Scraper for SAMOLET Website
Uses Playwright to simulate real browser behavior and bypass anti-bot protection

Requirements:
    pip install playwright beautifulsoup4
    playwright install chromium

Usage:
    python browser_scraper.py
    python browser_scraper.py --url "https://samolet.ru/project/oktyabrskaya-98/flats/308985/"
    python browser_scraper.py --save --headless

This approach is more likely to succeed than requests-based scraping because:
1. Uses real browser engine (Chromium)
2. Executes JavaScript like a real user
3. Handles cookies and session storage
4. Mimics human behavior patterns
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass
import argparse


try:
    from playwright.async_api import async_playwright, Page
except ImportError:
    print("❌ Playwright not installed!")
    print("Please run: pip install playwright")
    print("Then: playwright install chromium")
    exit(1)

from bs4 import BeautifulSoup


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
    raw_content: Optional[str] = None  # Store raw HTML/content
    extraction_timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.extraction_timestamp is None:
            import datetime
            self.extraction_timestamp = datetime.datetime.now().isoformat()
    
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
            'raw_content': self.raw_content,
            'extraction_timestamp': self.extraction_timestamp
        }


class BrowserScraper:
    """Browser-based scraper using Playwright"""
    
    def __init__(self, headless: bool = False, delay: float = 2.0):
        self.headless = headless
        self.delay = delay
    
    async def _setup_browser(self, playwright):
        """Initialize browser with anti-detection measures"""
        browser = await playwright.chromium.launch(
            headless=self.headless,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-web-security',
                '--disable-features=IsolateOrigins,site-per-process',
            ]
        )
        
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            locale='ru-RU',
            timezone_id='Europe/Moscow',
        )
        
        # Bypass webdriver detection
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5]
            });
            window.chrome = { runtime: {} };
        """)
        
        return browser, context
    
    async def _human_like_delay(self, page: Page, min_delay: float = 1.0, max_delay: float = 3.0):
        """Add human-like random delay"""
        import random
        delay = random.uniform(min_delay, max_delay)
        await asyncio.sleep(delay)
    
    async def _scroll_page(self, page: Page):
        """Simulate human scrolling"""
        await page.evaluate("""
            async () => {
                await new Promise((resolve) => {
                    let totalHeight = 0;
                    const distance = 100;
                    const timer = setInterval(() => {
                        const scrollHeight = document.body.scrollHeight;
                        window.scrollBy(0, distance);
                        totalHeight += distance;
                        
                        if(totalHeight >= scrollHeight || totalHeight > 3000) {
                            clearInterval(timer);
                            resolve();
                        }
                    }, 100);
                });
            }
        """)
        await asyncio.sleep(1)
        # Scroll back up a bit
        await page.evaluate("window.scrollTo(0, 0)")
        await asyncio.sleep(0.5)
    
    async def fetch_page(self, url: str) -> Optional[str]:
        """Fetch page content using browser"""
        async with async_playwright() as playwright:
            browser, context = await self._setup_browser(playwright)
            
            try:
                page = await context.new_page()
                
                print(f"  🌐 Navigating to: {url}")
                
                # Navigate with timeout
                response = await page.goto(
                    url, 
                    wait_until='networkidle',
                    timeout=60000
                )
                
                if response:
                    print(f"  ✅ Page loaded (Status: {response.status})")
                
                # Wait for content to load
                await asyncio.sleep(self.delay)
                
                # Simulate human behavior
                print(f"  👤 Simulating human behavior...")
                await self._scroll_page(page)
                await self._human_like_delay(page)
                
                # Get page content
                content = await page.content()
                
                # Get page title
                title = await page.title()
                print(f"  📄 Page title: {title}")
                
                # Try to extract data from page context
                page_data = await self._extract_from_page_context(page)
                
                await context.close()
                await browser.close()
                
                return content, page_data
                
            except Exception as e:
                print(f"  ❌ Browser error: {e}")
                await context.close()
                await browser.close()
                return None, None
    
    async def _extract_from_page_context(self, page: Page) -> Dict:
        """Extract data from JavaScript variables in page context"""
        data = {}
        
        try:
            # Look for common data variables
            variables = [
                'window.__INITIAL_STATE__',
                'window.__DATA__',
                'window.APP_DATA',
                'window.flatData',
                'window.apartment',
            ]
            
            for var in variables:
                try:
                    result = await page.evaluate(f'() => {{ try {{ return {var}; }} catch(e) {{ return null; }} }}')
                    if result:
                        print(f"  ✓ Found JavaScript variable: {var}")
                        data[var] = result
                except:
                    continue
                    
        except Exception as e:
            print(f"  ⚠️  Error extracting page context: {e}")
        
        return data
    
    def parse_content(self, html: str, url: str, js_data: Dict = None) -> ApartmentData:
        """Parse HTML content and extract apartment data"""
        data = ApartmentData(url=url)
        soup = BeautifulSoup(html, 'html.parser')
        
        print(f"\n🔍 Extracting apartment data...")
        
        # Extract from JavaScript data if available
        if js_data:
            data = self._parse_js_data(data, js_data)
        
        # Extract from HTML
        data = self._extract_from_html(soup, data)
        
        # Extract from JSON-LD
        data = self._extract_json_ld(soup, data)
        
        return data
    
    def _parse_js_data(self, data: ApartmentData, js_data: Dict) -> ApartmentData:
        """Parse data from JavaScript variables"""
        try:
            # Look through all JS variables
            for var_name, var_data in js_data.items():
                if isinstance(var_data, dict):
                    # Try to find apartment data structure
                    self._extract_from_dict(data, var_data)
                    
                    # Check nested structures
                    for key, value in var_data.items():
                        if isinstance(value, dict):
                            self._extract_from_dict(data, value)
                        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                            for item in value:
                                self._extract_from_dict(data, item)
                                
        except Exception as e:
            print(f"  ⚠️  Error parsing JS data: {e}")
        
        return data
    
    def _extract_from_dict(self, data: ApartmentData, d: Dict):
        """Extract apartment fields from a dictionary"""
        keys_mapping = {
            'total_area': ['totalArea', 'square', 'area', 'total_square', 'общая_площадь', 'площадь'],
            'floor': ['floor', 'floorNumber', 'floor_number', 'этаж', 'floors'],
            'floors_total': ['totalFloors', 'floorsTotal', 'floors_count', 'всего_этажей', 'houseFloors'],
            'price': ['price', 'cost', 'totalPrice', 'total_cost', 'цена', 'flatPrice'],
            'property_type': ['rooms', 'roomCount', 'roomsCount', 'room_count', 'комнаты', 'flatType', 'type'],
            'phase': ['phase', 'buildingPhase', 'constructionPhase', 'фаза', 'buildingPhaseNumber'],
            'ceiling_height': ['ceilingHeight', 'ceiling', 'высота_потолков', 'ceilingHeightM'],
            'finishing': ['finishing', 'finish', 'decoration', 'отделка', 'renovation', 'renovationType'],
            'building_type': ['buildingType', 'material', 'houseType', 'constructionType', 'тип_дома', 'wallMaterial'],
            'district': ['district', 'location', 'area', 'districtName', 'район', 'settlement'],
            'project_name': ['projectName', 'project', 'title', 'name', 'проект', 'projectTitle'],
            'class_option': ['class', 'propertyClass', 'houseClass', 'flatClass', 'класс'],
        }
        
        for attr, keys in keys_mapping.items():
            for key in keys:
                if key in d and hasattr(data, attr):
                    current_val = getattr(data, attr)
                    if current_val is None:  # Only set if not already found
                        value = d[key]
                        if value is not None and value != '':
                            setattr(data, attr, value)
                            if attr != 'url':  # Don't print URL multiple times
                                print(f"  ✓ Found {attr}: {value}")
                            break
    
    def _extract_json_ld(self, soup: BeautifulSoup, data: ApartmentData) -> ApartmentData:
        """Extract JSON-LD structured data"""
        scripts = soup.find_all('script', type='application/ld+json')
        
        for script in scripts:
            try:
                json_data = json.loads(script.string)
                
                if isinstance(json_data, list):
                    for item in json_data:
                        if item.get('@type') in ['Residence', 'Product', 'Apartment', 'SingleFamilyResidence']:
                            self._parse_json_ld_item(data, item)
                elif json_data.get('@type') in ['Residence', 'Product', 'Apartment', 'SingleFamilyResidence']:
                    self._parse_json_ld_item(data, json_data)
                    
            except (json.JSONDecodeError, AttributeError):
                continue
        
        return data
    
    def _parse_json_ld_item(self, data: ApartmentData, item: Dict):
        """Parse a single JSON-LD item"""
        try:
            if 'floorSize' in item:
                size = item['floorSize']
                if isinstance(size, dict):
                    value = size.get('value', size.get('@value', 0))
                    if not data.total_area:
                        data.total_area = float(value)
                        print(f"  ✓ Found area (JSON-LD): {data.total_area}")
            
            if 'offers' in item and not data.price:
                offers = item['offers']
                if isinstance(offers, dict):
                    price = offers.get('price')
                    if price:
                        data.price = float(price)
                        print(f"  ✓ Found price (JSON-LD): {data.price}")
            
            if 'floorLevel' in item and not data.floor:
                data.floor = int(item['floorLevel'])
                print(f"  ✓ Found floor (JSON-LD): {data.floor}")
                
        except (ValueError, TypeError) as e:
            pass
    
    def _extract_from_html(self, soup: BeautifulSoup, data: ApartmentData) -> ApartmentData:
        """Extract data from HTML elements"""
        
        # Project name
        if not data.project_name:
            h1 = soup.find('h1')
            if h1:
                data.project_name = h1.get_text(strip=True)
                print(f"  ✓ Found project (H1): {data.project_name}")
        
        # Common Russian real estate website patterns
        
        # Area patterns
        area_patterns = [
            r'(\d+(?:\.\d+)?)\s*м²',
            r'(\d+(?:\.\d+)?)\s*м2',
            r'(\d+(?:\.\d+)?)\s*кв\.м',
        ]
        
        if not data.total_area:
            for pattern in area_patterns:
                text = soup.get_text()
                match = re.search(pattern, text)
                if match:
                    data.total_area = float(match.group(1))
                    print(f"  ✓ Found area (regex): {data.total_area}")
                    break
        
        # Price patterns
        if not data.price:
            price_patterns = [
                r'(\d[\d\s\xa0]*)\s*₽',
                r'(\d[\d\s\xa0]*)\s*руб',
            ]
            for pattern in price_patterns:
                text = soup.get_text()
                match = re.search(pattern, text)
                if match:
                    price_str = match.group(1).replace(' ', '').replace('\xa0', '').replace(',', '')
                    try:
                        data.price = float(price_str)
                        print(f"  ✓ Found price (regex): {data.price:,.0f}")
                        break
                    except ValueError:
                        continue
        
        # Floor patterns (e.g., "3 этаж из 17")
        if not data.floor or not data.floors_total:
            floor_patterns = [
                r'(\d+)\s*этаж\s*из\s*(\d+)',
                r'(\d+)\s*/\s*(\d+)\s*этаж',
                r'этаж\s*(\d+)\s*из\s*(\d+)',
            ]
            for pattern in floor_patterns:
                text = soup.get_text()
                match = re.search(pattern, text, re.I)
                if match:
                    if not data.floor:
                        data.floor = int(match.group(1))
                    if not data.floors_total:
                        data.floors_total = int(match.group(2))
                    print(f"  ✓ Found floor info (regex): {data.floor}/{data.floors_total}")
                    break
        
        # Look for characteristics tables
        tables = soup.find_all('table', class_=re.compile(r'charact|spec|detail', re.I))
        if not tables:
            tables = soup.find_all('table')
        
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th', 'dt', 'dd'])
                if len(cells) >= 2:
                    label = cells[0].get_text(strip=True).lower()
                    value = cells[1].get_text(strip=True)
                    
                    if any(x in label for x in ['площадь', 'площадь общая']) and not data.total_area:
                        match = re.search(r'(\d+(?:\.\d+)?)', value)
                        if match:
                            data.total_area = float(match.group(1))
                            print(f"  ✓ Found area (table): {data.total_area}")
                    
                    elif any(x in label for x in ['этаж', 'этажность']):
                        if 'из' in value:
                            parts = value.split('из')
                            if len(parts) == 2:
                                try:
                                    if not data.floor:
                                        data.floor = int(parts[0].strip())
                                    if not data.floors_total:
                                        data.floors_total = int(parts[1].strip())
                                    print(f"  ✓ Found floor (table): {data.floor}/{data.floors_total}")
                                except ValueError:
                                    pass
                    
                    elif any(x in label for x in ['высота потолков', 'потолки']) and not data.ceiling_height:
                        match = re.search(r'(\d+(?:\.\d+)?)', value)
                        if match:
                            data.ceiling_height = float(match.group(1))
                            print(f"  ✓ Found ceiling height (table): {data.ceiling_height}")
                    
                    elif any(x in label for x in ['отделка', 'отделка квартиры']) and not data.finishing:
                        data.finishing = value
                        print(f"  ✓ Found finishing (table): {data.finishing}")
                    
                    elif any(x in label for x in ['материал стен', 'конструкция', 'тип дома']) and not data.building_type:
                        data.building_type = value
                        print(f"  ✓ Found building type (table): {data.building_type}")
                    
                    elif any(x in label for x in ['комнат', 'тип', 'планировка']) and not data.property_type:
                        if 'студия' in value.lower():
                            data.property_type = 'Студия'
                            print(f"  ✓ Found property type (table): {data.property_type}")
                        else:
                            match = re.search(r'(\d+)', value)
                            if match:
                                rooms = int(match.group(1))
                                data.property_type = f'{rooms} ккв'
                                print(f"  ✓ Found property type (table): {data.property_type}")
                    
                    elif any(x in label for x in ['класс', 'класс дома']) and not data.class_option:
                        data.class_option = value
                        print(f"  ✓ Found class (table): {data.class_option}")
                    
                    elif any(x in label for x in ['район', 'округ', 'населенный пункт']) and not data.district:
                        data.district = value
                        print(f"  ✓ Found district (table): {data.district}")
        
        # Look for div-based characteristics
        char_divs = soup.find_all(['div', 'section'], class_=re.compile(r'charact|spec|property|detail', re.I))
        for div in char_divs:
            labels = div.find_all(['span', 'div', 'dt'], class_=re.compile(r'label|title|name', re.I))
            values = div.find_all(['span', 'div', 'dd'], class_=re.compile(r'value|text|data', re.I))
            
            for label, value in zip(labels, values):
                label_text = label.get_text(strip=True).lower()
                value_text = value.get_text(strip=True)
                
                if 'площадь' in label_text and not data.total_area:
                    match = re.search(r'(\d+(?:\.\d+)?)', value_text)
                    if match:
                        data.total_area = float(match.group(1))
                        print(f"  ✓ Found area (div): {data.total_area}")
        
        return data
    
    async def scrape(self, url: str) -> Optional[ApartmentData]:
        """Main scraping function"""
        print(f"\n{'='*60}")
        print(f"🔍 Browser Scraping: {url}")
        print(f"{'='*60}")
        
        html, js_data = await self.fetch_page(url)
        
        if html is None:
            print(f"\n❌ Failed to fetch page")
            return None
        
        data = self.parse_content(html, url, js_data)
        
        # Store raw HTML for output saving
        if data:
            data.raw_content = html
        
        return data
    
    def save_results(self, data: ApartmentData, output_dir: str = "firecrawl_outputs"):
        """Save scraped data to JSON file in unified output folder"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        flat_id = data.url.rstrip('/').split('/')[-1]
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"apartment_browser_{flat_id}_{timestamp}.json"
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


def save_raw_output(url: str, data: ApartmentData, html_content: str, output_dir: str = "firecrawl_outputs"):
    """
    Save complete scraper output to text file for review
    
    Args:
        url: Source URL
        data: Extracted ApartmentData
        html_content: Raw HTML content
        output_dir: Directory to save output files (default: firecrawl_outputs)
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
    filename = f"browser_output_{random_hash}_{flat_id}_{timestamp}.txt"
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
    output_content = f"""# Browser Scraper Output Report
# Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

{'='*60}
🌐 Browser Automation Scraper (Playwright)
{'='*60}

🔗 URL: {url}
📝 Hash: {random_hash}
📅 Scraped: {data.extraction_timestamp}

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


async def main_async():
    parser = argparse.ArgumentParser(
        description='Scrape SAMOLET website using browser automation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python browser_scraper.py                                    # Run with visible browser
  python browser_scraper.py --headless                         # Run in headless mode
  python browser_scraper.py --url "https://samolet.ru/..."     # Scrape specific URL
  python browser_scraper.py --url "..." --save --headless      # Save JSON results
  python browser_scraper.py --url "..." --save-output          # Save formatted .txt output

Note: First run requires installing Playwright browsers:
  playwright install chromium
        """
    )
    
    parser.add_argument('--url', type=str,
                        default="https://samolet.ru/project/oktyabrskaya-98/flats/308985/",
                        help='URL of the apartment listing to scrape')
    parser.add_argument('--headless', action='store_true',
                        help='Run browser in headless mode (no visible window)')
    parser.add_argument('--save', action='store_true',
                        help='Save results to JSON file in firecrawl_outputs/')
    parser.add_argument('--save-output', action='store_true',
                        help='Save formatted output to .txt file in firecrawl_outputs/')
    parser.add_argument('--delay', type=float, default=2.0,
                        help='Delay between actions (seconds)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  🏠 SAMOLET Browser Automation Scraper")
    print("  Uses Playwright to simulate real browser")
    print("="*60)
    
    # Create scraper
    scraper = BrowserScraper(headless=args.headless, delay=args.delay)
    
    # Scrape
    data = await scraper.scrape(args.url)
    
    if data:
        print_apartment_info(data)
        
        if args.save:
            scraper.save_results(data)
        
        if args.save_output and data.raw_content:
            save_raw_output(args.url, data, data.raw_content)
            
        print("\n✅ Scraping completed successfully!")
        print("\n📋 Next steps:")
        print("  1. Review the extracted data above")
        print("  2. Use the data in main.py for price prediction")
        print("  3. Missing fields can be filled manually or with defaults")
    else:
        print("\n❌ Scraping failed")
        print("\nPossible solutions:")
        print("  • Try running with --headless for faster execution")
        print("  • Check if the URL is accessible in a regular browser")
        print("  • The site may have advanced anti-bot protection")
        print("  • Consider using manual data input mode")


def main():
    """Entry point"""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
