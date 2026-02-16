"""
FireCrawl Integration Script for SAMOLET Website
Extracts apartment information using FireCrawl AI scraping API

Prerequisites:
    1. Sign up at https://www.firecrawl.dev/
    2. Get API key from dashboard
    3. Install dependencies: pip install firecrawl-py python-dotenv
    4. Create .env file with: FIRECRAWL_API_KEY=your-api-key

Usage:
    python firecrawl_scraper.py
    python firecrawl_scraper.py --url "https://samolet.ru/project/oktyabrskaya-98/flats/308985/"
    python firecrawl_scraper.py --url "..." --extract --schema apartment_schema.json
    python firecrawl_scraper.py --batch urls.txt

Configuration (.env file):
    Create a .env file in the same directory:
    FIRECRAWL_API_KEY=your-api-key-here

Features:
    - Handles JavaScript-heavy sites
    - Bypasses basic anti-bot protection
    - Supports structured data extraction with LLM
    - Caching to reduce API costs
    - Batch processing multiple URLs
    - Loads API key from .env file (cleaner than env vars)

Note: Uses FireCrawl SDK v1.x API (Firecrawl class with .scrape() method)
      Updated from deprecated v0.x API (FirecrawlApp class with .scrape_url() method)
"""

import os
import json
import asyncio
import argparse
from pathlib import Path
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import time

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file from current directory
    print("✅ Loaded .env file")
except ImportError:
    print("⚠️  python-dotenv not installed, using environment variables only")
    print("   Install: pip install python-dotenv")

# Try to import FireCrawl
try:
    from firecrawl import Firecrawl
except ImportError:
    print("❌ FireCrawl not installed!")
    print("Please run: pip install firecrawl-py")
    print("Get API key at: https://www.firecrawl.dev/")
    exit(1)


@dataclass
class ApartmentData:
    """Data structure for extracted apartment information"""
    url: str
    extraction_timestamp: str = None
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
    raw_content: Optional[str] = None
    extraction_confidence: Optional[float] = None
    
    def __post_init__(self):
        if self.extraction_timestamp is None:
            self.extraction_timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FireCrawlScraper:
    """FireCrawl-based scraper for SAMOLET website"""
    
    def __init__(self, api_key: Optional[str] = None, output_dir: str = "output", save_output: bool = False):
        """
        Initialize FireCrawl scraper
        
        Args:
            api_key: FireCrawl API key (or from FIRECRAWL_API_KEY env var)
            output_dir: Directory to save scraped results (default: output)
            save_output: Whether to save raw output to file
        """
        self.api_key = api_key or os.getenv('FIRECRAWL_API_KEY')
        if not self.api_key:
            raise ValueError(
                "FireCrawl API key required. "
                "Set FIRECRAWL_API_KEY environment variable or pass api_key parameter.\n"
                "Get your key at: https://www.firecrawl.dev/"
            )
        
        self.app = Firecrawl(api_key=self.api_key)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.save_output = save_output
        
    def _get_output_path(self, url: str) -> Path:
        """Generate output file path for URL"""
        import hashlib
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        flat_id = url.rstrip('/').split('/')[-1]
        return self.output_dir / f"{flat_id}_{url_hash}.json"
    
    def _load_from_cache(self, url: str, max_age_hours: int = 24) -> Optional[ApartmentData]:
        """Load cached result if not expired"""
        cache_path = self._get_output_path(url)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            
            # Check cache age
            timestamp = datetime.fromisoformat(cached.get('extraction_timestamp', '2000-01-01'))
            age_hours = (datetime.now() - timestamp).total_seconds() / 3600
            
            if age_hours > max_age_hours:
                print(f"  🕐 Cache expired ({age_hours:.1f}h old)")
                return None
            
            print(f"  💾 Using cached result ({age_hours:.1f}h old)")
            return ApartmentData(**cached)
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"  ⚠️  Cache load error: {e}")
            return None
    
    def _save_to_cache(self, data: ApartmentData):
        """Save result to cache/output folder"""
        cache_path = self._get_output_path(data.url)
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data.to_dict(), f, ensure_ascii=False, indent=2)
            print(f"  💾 Saved to: {cache_path}")
        except Exception as e:
            print(f"  ⚠️  Save error: {e}")
    
    def scrape_basic(self, url: str, use_cache: bool = True) -> Optional[ApartmentData]:
        """
        Basic scraping - get page content
        
        Args:
            url: URL to scrape
            use_cache: Whether to use cached results
            
        Returns:
            ApartmentData with raw content
        """
        print(f"\n🔍 Basic scraping: {url}")
        
        # Check cache
        if use_cache:
            cached = self._load_from_cache(url)
            if cached:
                return cached
        
        try:
            print(f"  🌐 Calling FireCrawl API...")
            start_time = time.time()
            
            # Scrape the URL
            result = self.app.scrape(
                url,
                formats=['markdown', 'html']
            )
            
            elapsed = time.time() - start_time
            print(f"  ✅ Scraped in {elapsed:.2f}s")
            
            # Debug: Print result structure
            print(f"  📝 Result type: {type(result)}")
            if hasattr(result, '__dict__'):
                print(f"  📝 Result keys: {list(result.__dict__.keys())}")
            elif isinstance(result, dict):
                print(f"  📝 Result keys: {list(result.keys())}")
            
            # Create data object
            data = ApartmentData(url=url)
            
            # Store raw content - handle both dict and object results
            if isinstance(result, dict):
                # Dictionary result
                if 'markdown' in result:
                    data.raw_content = result['markdown']
                elif 'html' in result:
                    data.raw_content = result['html']
                
                # Try to extract metadata if available
                if 'metadata' in result:
                    metadata = result['metadata']
                    data.project_name = metadata.get('title', '')
            else:
                # Object result (v1.x API)
                if hasattr(result, 'markdown') and result.markdown:
                    data.raw_content = result.markdown
                    print(f"  📝 Got markdown: {len(result.markdown)} chars")
                elif hasattr(result, 'html') and result.html:
                    data.raw_content = result.html
                    print(f"  📝 Got HTML: {len(result.html)} chars")
                else:
                    print(f"  ⚠️  No markdown or HTML content available")
                
                if hasattr(result, 'metadata') and result.metadata:
                    if hasattr(result.metadata, 'title'):
                        data.project_name = result.metadata.title
                        # Parse title for structured data
                        data = self._parse_title_data(data, result.metadata.title)
            
            # Parse markdown content if available
            if data.raw_content:
                data = self._parse_markdown_content(data, data.raw_content)
                # Save complete output for review (if enabled)
                if self.save_output:
                    print(f"  💾 Saving output ({len(data.raw_content)} chars)...")
                    save_raw_output(url, data, data.raw_content, output_dir=str(self.output_dir))
                else:
                    print(f"  ℹ️  Output saving disabled (use --save-output to enable)")
            else:
                print(f"  ⚠️  No content to save")
            
            # Cache and return
            if use_cache:
                self._save_to_cache(data)
            
            return data
            
        except Exception as e:
            print(f"  ❌ FireCrawl error: {e}")
            return None
    
    def scrape_with_extraction(
        self, 
        url: str, 
        schema: Optional[Dict] = None,
        prompt: Optional[str] = None,
        use_cache: bool = True
    ) -> Optional[ApartmentData]:
        """
        Advanced scraping with LLM-based extraction
        
        Args:
            url: URL to scrape
            schema: JSON schema for structured extraction (optional)
            prompt: Natural language prompt for extraction (optional)
            use_cache: Whether to use cached results
            
        Returns:
            ApartmentData with extracted fields
        """
        print(f"\n🤖 AI-powered extraction: {url}")
        
        # Check cache
        if use_cache:
            cached = self._load_from_cache(url)
            if cached:
                return cached
        
        try:
            print(f"  🌐 Calling FireCrawl with extraction...")
            start_time = time.time()
            
            # Prepare extraction parameters
            extract_params = {
                'prompt': prompt or self._get_default_extraction_prompt(),
                'schema': schema or self._get_default_schema(),
            }
            
            # Use extract method (if available in your FireCrawl version)
            # or scrape with extraction params
            # Note: FireCrawl v1.x uses .extract() method or formats=['extract']
            try:
                # Try the extract method first (v1.x API)
                result = self.app.extract(
                    [url],
                    prompt=extract_params.get('prompt'),
                    schema=extract_params.get('schema')
                )
            except AttributeError:
                # Fallback to scrape with extract format
                result = self.app.scrape(
                    url,
                    formats=['extract']
                )
            
            elapsed = time.time() - start_time
            print(f"  ✅ Extracted in {elapsed:.2f}s")
            
            # Parse extracted data
            data = self._parse_extracted_data(url, result)
            
            # Save complete output for review (if enabled)
            if self.save_output:
                print(f"  💾 Saving output ({len(data.raw_content) if data.raw_content else 0} chars)...")
                save_raw_output(url, data, data.raw_content or "", output_dir=str(self.output_dir))
            else:
                print(f"  ℹ️  Output saving disabled (use --save-output to enable)")
            
            # Cache and return
            if use_cache:
                self._save_to_cache(data)
            
            return data
            
        except Exception as e:
            print(f"  ❌ Extraction error: {e}")
            print(f"  🔄 Falling back to basic scraping...")
            return self.scrape_basic(url, use_cache)
    
    def _get_default_extraction_prompt(self) -> str:
        """Default prompt for apartment extraction"""
        return """
        Extract apartment information from this real estate listing page.
        Focus on:
        - Total area in square meters (look for "м²", "кв.м", "площадь")
        - Floor number and total floors (e.g., "3 этаж из 17")
        - Number of rooms (e.g., "2-комнатная", "студия", "1-комнатная")
        - Ceiling height in meters
        - Building type (e.g., "монолит", "панель", "кирпич")
        - Finishing level (e.g., "чистовая", "подчистовая", "без отделки")
        - Price in rubles (₽)
        - Property class (e.g., "эконом", "комфорт", "бизнес", "элит")
        - District or location
        - Construction phase
        - Project name
        
        Return data in a structured format with these exact field names.
        """
    
    def _get_default_schema(self) -> Dict:
        """Default JSON schema for extraction"""
        return {
            "type": "object",
            "properties": {
                "total_area": {"type": "number", "description": "Total area in square meters"},
                "floor": {"type": "integer", "description": "Floor number"},
                "floors_total": {"type": "integer", "description": "Total floors in building"},
                "property_type": {"type": "string", "description": "Property type like '2-комнатная', 'студия', '1-комнатная'"},
                "ceiling_height": {"type": "number", "description": "Ceiling height in meters"},
                "building_type": {"type": "string", "description": "Building construction type"},
                "finishing": {"type": "string", "description": "Interior finishing level"},
                "price": {"type": "number", "description": "Price in rubles"},
                "class_option": {"type": "string", "description": "Property class category"},
                "district": {"type": "string", "description": "District or location"},
                "phase": {"type": "integer", "description": "Construction phase"},
                "project_name": {"type": "string", "description": "Project or building name"},
            },
            "required": ["total_area", "floor", "property_type"]
        }
    
    def _parse_extracted_data(self, url: str, result) -> ApartmentData:
        """Parse extraction result into ApartmentData"""
        data = ApartmentData(url=url)
        
        # Handle different result types (dict or object)
        extracted = {}
        if isinstance(result, dict):
            # Dictionary result (v0.x API or some v1.x responses)
            extracted = result.get('extract', {})
            if not extracted and 'data' in result:
                extracted = result['data']
        else:
            # Object result (v1.x ExtractResponse)
            # Try to convert to dict or access attributes
            try:
                if hasattr(result, 'data'):
                    extracted = result.data
                elif hasattr(result, '__dict__'):
                    extracted = result.__dict__
                else:
                    # Try to get data as string and parse
                    extracted = str(result)
            except Exception:
                extracted = {}
        
        # Map extracted fields to ApartmentData
        field_mapping = {
            'total_area': ['total_area', 'area', 'square', 'общая_площадь', 'площадь'],
            'floor': ['floor', 'floor_number', 'этаж', 'floorNumber'],
            'floors_total': ['floors_total', 'total_floors', 'всего_этажей', 'totalFloors'],
            'property_type': ['property_type', 'rooms', 'type', 'комнат', 'room_count'],
            'ceiling_height': ['ceiling_height', 'ceiling', 'высота_потолков'],
            'building_type': ['building_type', 'material', 'конструкция', 'тип_дома'],
            'finishing': ['finishing', 'finish', 'отделка', 'renovation'],
            'price': ['price', 'cost', 'цена', 'total_price'],
            'class_option': ['class_option', 'class', 'класс', 'property_class'],
            'district': ['district', 'location', 'район', 'area'],
            'phase': ['phase', 'фаза', 'building_phase'],
            'project_name': ['project_name', 'project', 'проект', 'title'],
        }
        
        # Ensure extracted is a dict
        if not isinstance(extracted, dict):
            print(f"  ⚠️  Unexpected extraction result type: {type(result)}")
            data.raw_content = str(result)
            return data
        
        for attr, possible_keys in field_mapping.items():
            for key in possible_keys:
                if key in extracted and extracted[key] is not None:
                    value = extracted[key]
                    # Convert types
                    if attr in ['floor', 'floors_total', 'phase'] and value is not None:
                        try:
                            value = int(float(str(value).replace(',', '.')))
                        except (ValueError, TypeError):
                            continue
                    elif attr in ['total_area', 'ceiling_height', 'price'] and value is not None:
                        try:
                            value = float(str(value).replace(' ', '').replace('\xa0', '').replace(',', '.'))
                        except (ValueError, TypeError):
                            continue
                    
                    setattr(data, attr, value)
                    print(f"  ✓ Extracted {attr}: {value}")
                    break
        
        # Set confidence based on how many fields were extracted
        extracted_count = sum(1 for attr in field_mapping.keys() 
                            if getattr(data, attr) is not None)
        data.extraction_confidence = extracted_count / len(field_mapping)
        
        # Store raw extraction result for output saving
        try:
            data.raw_content = json.dumps(extracted, ensure_ascii=False, indent=2)
        except:
            data.raw_content = str(result)
        
        return data
    
    def _parse_title_data(self, data: ApartmentData, title: str) -> ApartmentData:
        """Parse metadata title to extract apartment information"""
        import re
        
        print(f"  🔍 Parsing title: {title[:80]}...")
        
        # Parse property type (rooms)
        # Match patterns like "3-комнатную", "3-комнатная", "3 комнатная", etc.
        room_patterns = [
            (r'(\d+)-комнатн', 'numbered'),  # 3-комнатную, 3-комнатная, etc.
            (r'(\d+)\s*ккв', 'numbered'),
            (r'(\d+)\s*к\.кв', 'numbered'),
            (r'студия', 'studio'),
            (r'однокомнатн', '1'),
            (r'двухкомнатн', '2'),
            (r'трехкомнатн', '3'),
            (r'четырехкомнатн', '4'),
        ]
        
        for pattern, ptype in room_patterns:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                if ptype == 'studio':
                    data.property_type = 'Студия'
                elif ptype == 'numbered':
                    rooms = int(match.group(1))
                    if 1 <= rooms <= 5:
                        data.property_type = f'{rooms} ккв'
                else:
                    data.property_type = f'{ptype} ккв'
                
                print(f"  ✓ Found property type (title): {data.property_type}")
                break
        
        # Parse price
        price_match = re.search(r'(\d[\d\s]*)\s*руб', title)
        if price_match:
            price_str = price_match.group(1).replace(' ', '').replace('\xa0', '')
            try:
                data.price = float(price_str)
                print(f"  ✓ Found price (title): {data.price:,.0f} ₽")
            except ValueError:
                pass
        
        # Parse project name (ЖК «Name»)
        project_match = re.search(r'ЖК\s*«([^»]+)»', title)
        if project_match:
            data.project_name = project_match.group(1)
            print(f"  ✓ Found project name (title): {data.project_name}")
        
        return data
    
    def _parse_markdown_content(self, data: ApartmentData, content: str) -> ApartmentData:
        """Parse markdown content to extract structured data"""
        import re
        
        print(f"  🔍 Parsing markdown content ({len(content)} chars)...")
        
        # Area patterns (e.g., "78.5 м²", "Площадь: 45 кв.м")
        area_patterns = [
            r'(\d+(?:\.\d+)?)\s*м²',
            r'(\d+(?:\.\d+)?)\s*м2',
            r'(\d+(?:\.\d+)?)\s*кв\.\s*м',
            r'площадь[:\s]*(\d+(?:\.\d+)?)',
            r'общая\s+площадь[:\s]*(\d+(?:\.\d+)?)',
        ]
        
        if not data.total_area:
            for pattern in area_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    try:
                        data.total_area = float(match.group(1))
                        print(f"  ✓ Found area (content): {data.total_area} m²")
                        break
                    except ValueError:
                        pass
        
        # Floor patterns (e.g., "5 этаж из 24", "Этаж: 3/17")
        floor_patterns = [
            r'(\d+)\s*этаж\s*из\s*(\d+)',
            r'этаж[:\s]*(\d+)\s*/\s*(\d+)',
            r'(\d+)\s*/\s*(\d+)\s*этаж',
        ]
        
        if not data.floor or not data.floors_total:
            for pattern in floor_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    try:
                        if not data.floor:
                            data.floor = int(match.group(1))
                        if not data.floors_total:
                            data.floors_total = int(match.group(2))
                        print(f"  ✓ Found floor (content): {data.floor}/{data.floors_total}")
                        break
                    except (ValueError, IndexError):
                        pass
        
        # Ceiling height (e.g., "2.7 м", "высота потолков 2.75м")
        ceiling_patterns = [
            r'высота\s*потолков[:\s]*(\d+(?:\.\d+)?)',
            r'потолки[:\s]*(\d+(?:\.\d+)?)',
            r'потолок[:\s]*(\d+(?:\.\d+)?)\s*м',
        ]
        
        if not data.ceiling_height:
            for pattern in ceiling_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    try:
                        data.ceiling_height = float(match.group(1))
                        print(f"  ✓ Found ceiling height (content): {data.ceiling_height} m")
                        break
                    except ValueError:
                        pass
        
        # Building type (e.g., "монолит", "панель")
        building_patterns = {
            'монолит': 'Монолит',
            'панель': 'Панель',
            'кирпич-монолит': 'Кирпич-монолит',
            'кирпич': 'Кирпич',
        }
        
        if not data.building_type:
            for pattern, btype in building_patterns.items():
                if re.search(pattern, content, re.IGNORECASE):
                    data.building_type = btype
                    print(f"  ✓ Found building type (content): {data.building_type}")
                    break
        
        # Finishing (e.g., "чистовая", "подчистовая", "без отделки")
        finishing_patterns = {
            'чистовая': 'Чистовая',
            'подчистовая': 'Подчистовая',
            r'без\s*отделки': 'Без отделки',
            r'с\s*мебелью': 'С мебелью',
            r'white\s*box': 'Подчистовая',
        }
        
        if not data.finishing:
            for pattern, ftype in finishing_patterns.items():
                if re.search(pattern, content, re.IGNORECASE):
                    data.finishing = ftype
                    print(f"  ✓ Found finishing (content): {data.finishing}")
                    break
        
        # Price (if not found in title)
        if not data.price:
            price_patterns = [
                r'(\d[\d\s\xa0]*)\s*₽',
                r'цена[:\s]*(\d[\d\s\xa0]*)',
                r'стоимость[:\s]*(\d[\d\s\xa0]*)',
            ]
            for pattern in price_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    try:
                        price_str = match.group(1).replace(' ', '').replace('\xa0', '').replace(',', '')
                        data.price = float(price_str)
                        print(f"  ✓ Found price (content): {data.price:,.0f} ₽")
                        break
                    except ValueError:
                        pass
        
        return data
    
    async def scrape_batch(
        self, 
        urls: List[str], 
        use_extraction: bool = True,
        max_concurrent: int = 3
    ) -> List[ApartmentData]:
        """
        Scrape multiple URLs with rate limiting
        
        Args:
            urls: List of URLs to scrape
            use_extraction: Whether to use AI extraction
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of ApartmentData objects
        """
        print(f"\n📦 Batch processing {len(urls)} URLs...")
        results = []
        
        # Process in batches
        for i in range(0, len(urls), max_concurrent):
            batch = urls[i:i + max_concurrent]
            print(f"\n  Processing batch {i//max_concurrent + 1}/{(len(urls)-1)//max_concurrent + 1} ({len(batch)} URLs)")
            
            for url in batch:
                if use_extraction:
                    data = self.scrape_with_extraction(url)
                else:
                    data = self.scrape_basic(url)
                
                if data:
                    results.append(data)
                
                # Small delay between requests
                await asyncio.sleep(0.5)
        
        print(f"\n✅ Batch complete: {len(results)}/{len(urls)} successful")
        return results


def print_apartment_info(data: ApartmentData):
    """Print formatted apartment information"""
    print(f"\n{'='*60}")
    print(f"📊 EXTRACTED APARTMENT INFORMATION")
    print(f"{'='*60}")
    
    print(f"\n🔗 URL: {data.url}")
    print(f"⏱️  Extracted: {data.extraction_timestamp}")
    if data.extraction_confidence:
        print(f"🎯 Confidence: {data.extraction_confidence:.1%}")
    
    if data.project_name:
        print(f"🏢 Project: {data.project_name}")
    
    print(f"\n📐 AREA & LAYOUT:")
    print(f"  • Total Area: {data.total_area} m²" if data.total_area else "  • Total Area: Not found")
    print(f"  • Area Without Balcony: {data.area_without_balcony} m²" if data.area_without_balcony else "  • Area Without Balcony: Not found")
    print(f"  • Living Area: {data.living_area} m²" if data.living_area else "  • Living Area: Not found")
    print(f"  • Kitchen Area: {data.kitchen_area} m²" if data.kitchen_area else "  • Kitchen Area: Not found")
    
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
    
    print(f"\n{'='*60}")


def generate_random_hash(length: int = 4) -> str:
    """Generate random alphanumeric hash"""
    import random
    import string
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


def save_results(data: ApartmentData, output_dir: str = "scraped_data"):
    """Save results to JSON file"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    flat_id = data.url.rstrip('/').split('/')[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"firecrawl_{flat_id}_{timestamp}.json"
    filepath = output_path / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data.to_dict(), f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Results saved to: {filepath}")
    return filepath


def save_raw_output(url: str, data: ApartmentData, raw_content: str, output_dir: str = "output") -> Path:
    """
    Save complete scraper output to text file for review (includes both formatted results and raw content)
    
    Args:
        url: Source URL
        data: Extracted ApartmentData
        raw_content: Raw content from FireCrawl (markdown or HTML)
        output_dir: Directory to save output files (default: output)
        
    Returns:
        Path to saved file
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Generate random hash
    random_hash = generate_random_hash(4)
    
    # Extract flat ID from URL
    flat_id = url.rstrip('/').split('/')[-1]
    
    # Create filename with hash
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"firecrawl_output_{random_hash}_{flat_id}_{timestamp}.txt"
    filepath = output_path / filename
    
    # Build formatted output with both terminal-style results and raw content
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
    
    confidence_str = f"{data.extraction_confidence:.1%}" if data.extraction_confidence else "N/A"
    
    output_content = f"""# FireCrawl Output Report
# Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

{'='*60}
🔥 FireCrawl AI Scraper for SAMOLET
{'='*60}

🔗 URL: {url}
📝 Hash: {random_hash}
📅 Scraped: {data.extraction_timestamp}

{'='*60}
📊 EXTRACTED APARTMENT INFORMATION
{'='*60}

🏢 Project: {data.project_name or 'Not found'}
🎯 Confidence: {confidence_str}

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

{'='*60}
🔍 RAW SCRAPER CONTENT
{'='*60}

{raw_content}

{'='*60}
End of Report
{'='*60}
"""
    
    # Save formatted content
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(output_content)
    
    print(f"\n📝 Complete output saved to: {filepath}")
    return filepath


def read_urls_from_file(filepath: str) -> List[str]:
    """Read URLs from text file (one per line)"""
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]


def main():
    parser = argparse.ArgumentParser(
        description='Scrape SAMOLET website using FireCrawl AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Setup:
  1. Create .env file in project directory:
     echo "FIRECRAWL_API_KEY=your-api-key-here" > .env
  
  2. Install dependencies:
     pip install firecrawl-py python-dotenv

Examples:
  # Basic scraping:
  python firecrawl_scraper.py --url "https://samolet.ru/..."
  
  # AI extraction with structured data:
  python firecrawl_scraper.py --url "..." --extract
  
  # Batch processing:
  python firecrawl_scraper.py --batch urls.txt
  
  # With custom extraction schema:
  python firecrawl_scraper.py --url "..." --extract --schema schema.json

Get API Key: https://www.firecrawl.dev/
Pricing: Free tier (500 credits), Hobby $16/mo, Standard $83/mo
        """
    )
    
    parser.add_argument('--url', type=str,
                        default="https://samolet.ru/project/oktyabrskaya-98/flats/308985/",
                        help='URL to scrape')
    parser.add_argument('--extract', action='store_true',
                        help='Use AI extraction for structured data')
    parser.add_argument('--schema', type=str,
                        help='JSON schema file for extraction')
    parser.add_argument('--batch', type=str,
                        help='File with URLs to process (one per line)')
    parser.add_argument('--save', action='store_true',
                        help='Save results to JSON file')
    parser.add_argument('--save-output', action='store_true',
                        help='Save raw scraper output to .txt file for review')
    parser.add_argument('--no-cache', action='store_true',
                        help='Skip cache, always fetch fresh data')
    parser.add_argument('--api-key', type=str,
                        help='FireCrawl API key (or set FIRECRAWL_API_KEY env var)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  🔥 FireCrawl AI Scraper for SAMOLET")
    print("  AI-powered web extraction")
    print("="*60)
    
    # Check API key (from args, env var, or .env file)
    api_key = args.api_key or os.getenv('FIRECRAWL_API_KEY')
    if not api_key:
        print("\n❌ FireCrawl API key not found!")
        print("\nGet your API key at: https://www.firecrawl.dev/")
        print("\nThen create a .env file in this directory:")
        print('  echo "FIRECRAWL_API_KEY=your-api-key-here" > .env')
        print("\nOr use alternative methods:")
        print("  1. Set environment variable: export FIRECRAWL_API_KEY='your-key'")
        print("  2. Pass as argument: --api-key 'your-key'")
        return
    
    # Initialize scraper
    try:
        scraper = FireCrawlScraper(api_key=api_key, save_output=args.save_output)
    except ValueError as e:
        print(f"\n❌ {e}")
        return
    
    # Load schema if provided
    schema = None
    if args.schema:
        with open(args.schema, 'r') as f:
            schema = json.load(f)
            print(f"📄 Loaded custom schema from {args.schema}")
    
    # Process batch or single URL
    if args.batch:
        # Batch mode
        urls = read_urls_from_file(args.batch)
        print(f"\n📋 Loaded {len(urls)} URLs from {args.batch}")
        
        results = asyncio.run(scraper.scrape_batch(
            urls, 
            use_extraction=args.extract,
            max_concurrent=3
        ))
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"📊 BATCH SUMMARY")
        print(f"{'='*60}")
        print(f"Total URLs: {len(urls)}")
        print(f"Successful: {len(results)}")
        print(f"Failed: {len(urls) - len(results)}")
        
        # Save all results
        if args.save and results:
            output_path = Path("scraped_data")
            output_path.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_file = output_path / f"firecrawl_batch_{timestamp}.json"
            
            with open(batch_file, 'w', encoding='utf-8') as f:
                json.dump([r.to_dict() for r in results], f, ensure_ascii=False, indent=2)
            
            print(f"\n💾 Batch results saved to: {batch_file}")
    
    else:
        # Single URL mode
        data = scraper.scrape_with_extraction(
            args.url,
            schema=schema,
            use_cache=not args.no_cache
        ) if args.extract else scraper.scrape_basic(
            args.url,
            use_cache=not args.no_cache
        )
        
        if data:
            print_apartment_info(data)
            
            if args.save:
                save_results(data)
            
            print("\n✅ Scraping completed!")
            print("\n📋 Next steps:")
            print("  1. Review extracted data above")
            print("  2. Use --extract flag for better AI extraction")
            print("  3. Integrate with main.py for price prediction")
        else:
            print("\n❌ Scraping failed")
            print("\nTroubleshooting:")
            print("  • Check if URL is accessible in browser")
            print("  • Try --no-cache to bypass cache")
            print("  • Check FireCrawl API key and credits")
            print("  • The site may have advanced anti-bot protection")


if __name__ == "__main__":
    main()
