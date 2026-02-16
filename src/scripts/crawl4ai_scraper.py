"""
Crawl4AI Integration Script for SAMOLET Website
Extracts apartment information using Crawl4AI open-source web scraping framework

Prerequisites:
    1. Install dependencies: pip install crawl4ai beautifulsoup4
    2. Run setup: crawl4ai-setup (installs browser dependencies)

Usage:
    python crawl4ai_scraper.py
    python crawl4ai_scraper.py --url "https://samolet.ru/project/oktyabrskaya-98/flats/308985/"
    python crawl4ai_scraper.py --url "..." --extract

Features:
    - Open-source and free (no API key required)
    - Handles JavaScript-heavy sites with headless browser
    - Returns clean markdown content
    - Optional LLM-based extraction for structured data
    - Fast and efficient async crawling

Note: SAMOLET website has strong anti-bot protection that can result in:

1. HTTP 403 Forbidden Error (ACCESS BLOCKED)
   - "Access to samolet.ru is forbidden"
   - "IP: xxx.xxx.xxx.xxx" - Your IP is being blocked
   - "Guru meditation" error from CDN/WAF protection
   - This means the site detected automated access and blocked your IP

2. Execution context destroyed errors (page navigation/redirects)
3. Empty markdown content but valid HTML (blocked page content)
4. Very slow loading times (2-3 minutes to fully render)
5. JavaScript-heavy content that requires extensive waiting
6. Browser fingerprint detection and blocking

This script handles these issues by:
1. Extended wait times (up to 3 minutes) for slow-loading pages
2. Multiple scroll events to trigger lazy loading
3. HTML parsing fallback when markdown extraction fails
4. BeautifulSoup-based structured data extraction from raw HTML
5. Anti-detection browser configuration

⚠️  IMPORTANT - If you get 403 Forbidden:
The site has detected and blocked your automated scraping attempt.
Your IP address is being blocked by their WAF/CDN protection.

Solutions:
1. Use --no-headless first to visually see what's happening
2. Try browser_scraper.py (Playwright with better anti-detection)
3. Use firecrawl_scraper.py (cloud-based bypass with API key) - RECOMMENDED
4. Wait several hours and try again (IP may be temporarily blocked)
5. Use a VPN or different network connection
6. Consider manual data entry from the property page
"""

import os
import json
import asyncio
import argparse
import re
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import time

# Try to import required libraries
try:
    from bs4 import BeautifulSoup
    print("✅ BeautifulSoup imported")
except ImportError:
    print("⚠️  BeautifulSoup not available, install with: pip install beautifulsoup4")
    BeautifulSoup = None

try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
    print("✅ Crawl4AI imported successfully")
except ImportError:
    print("❌ Crawl4AI not installed!")
    print("Please run:")
    print("  pip install crawl4ai")
    print("  crawl4ai-setup")
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


class Crawl4AIScraper:
    """Crawl4AI-based scraper for SAMOLET website"""
    
    def __init__(self, headless: bool = True):
        """
        Initialize Crawl4AI scraper
        
        Args:
            headless: Run browser in headless mode (default: True)
        """
        self.headless = headless
        
    async def scrape(self, url: str, extract: bool = False) -> Optional[ApartmentData]:
        """
        Scrape a URL using Crawl4AI
        
        Args:
            url: URL to scrape
            extract: Whether to attempt structured data extraction
            
        Returns:
            ApartmentData with extracted information
        """
        print(f"\n🔍 Scraping: {url}")
        print(f"  🌐 Starting Crawl4AI crawler...")
        
        start_time = time.time()
        
        # Browser configuration with anti-detection measures
        browser_config = BrowserConfig(
            headless=self.headless,
            browser_type="chromium",
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.0.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.0",
            extra_args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
                "--disable-site-isolation-trials",
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-accelerated-2d-canvas",
                "--disable-gpu",
                "--window-size=1920,1080",
                "--start-maximized",
            ]
        )
        
        # Crawler configuration with extended wait for slow-loading pages
        # SAMOLET takes 2-3 minutes to fully load, so we wait extensively
        crawler_config = CrawlerRunConfig(
            word_count_threshold=10,
            wait_for_images=True,
            page_timeout=180000,  # 3 minutes timeout
            js_code="""
                // SAMOLET loads very slowly (2-3 mins), so we wait extensively
                console.log('Starting extended wait for SAMOLET...');
                
                // Initial wait for page structure
                await new Promise(r => setTimeout(r, 5000));
                
                // Multiple scrolls to trigger lazy loading
                for (let i = 0; i < 5; i++) {
                    window.scrollTo(0, (document.body.scrollHeight / 5) * i);
                    await new Promise(r => setTimeout(r, 3000));
                    console.log('Scrolled to position', i);
                }
                
                // Final scroll to bottom
                window.scrollTo(0, document.body.scrollHeight);
                await new Promise(r => setTimeout(r, 5000));
                
                // Wait for any remaining async content
                await new Promise(r => setTimeout(r, 10000));
                
                console.log('Extended wait complete');
                return document.readyState;
            """,
            magic=True,  # Use magic mode for better content extraction
            verbose=True
        )
        
        try:
            # Run the crawler
            async with AsyncWebCrawler(config=browser_config) as crawler:
                result = await crawler.arun(
                    url=url,
                    config=crawler_config
                )
                
                elapsed = time.time() - start_time
                print(f"  ✅ Crawled in {elapsed:.2f}s")
                
                # Create data object
                data = ApartmentData(url=url)
                
                # Check what content we got
                markdown_len = len(result.markdown) if result.markdown else 0
                html_len = len(result.html) if result.html else 0
                print(f"  📝 Markdown: {markdown_len} chars, HTML: {html_len} chars")
                
                # Use whichever content is available and longer
                content_to_use = None
                content_type = None
                
                if markdown_len > 100:
                    content_to_use = result.markdown
                    content_type = "markdown"
                elif html_len > 100:
                    content_to_use = result.html
                    content_type = "html"
                elif markdown_len > 0:
                    content_to_use = result.markdown
                    content_type = "markdown"
                elif html_len > 0:
                    content_to_use = result.html
                    content_type = "html"
                
                if content_to_use:
                    # Check for 403 Forbidden error
                    is_blocked = False
                    blocked_indicators = [
                        "403 error", "forbidden", "access to samolet.ru is forbidden",
                        "guru meditation", "http 403", "access is forbidden"
                    ]
                    content_lower = content_to_use.lower()
                    for indicator in blocked_indicators:
                        if indicator in content_lower:
                            is_blocked = True
                            break
                    
                    if is_blocked:
                        print(f"\n{'='*60}")
                        print("🚫 ACCESS BLOCKED - HTTP 403 FORBIDDEN")
                        print(f"{'='*60}")
                        print("The SAMOLET website has detected and blocked your scraping attempt.")
                        print("Your IP address is being blocked by their anti-bot protection.")
                        print(f"\n{'='*60}")
                        print("SOLUTIONS:")
                        print(f"{'='*60}")
                        print("1. Use firecrawl_scraper.py (cloud-based bypass) - RECOMMENDED")
                        print("2. Try browser_scraper.py (Playwright with better anti-detection)")
                        print("3. Use --no-headless to visually debug")
                        print("4. Wait several hours and try again (temporary IP block)")
                        print("5. Use a VPN or different network connection")
                        print("6. Manual data entry from the property page")
                        print(f"{'='*60}\n")
                        data.raw_content = content_to_use[:5000]
                        return data
                    
                    data.raw_content = content_to_use[:10000]  # Limit storage
                    
                    # Print raw content to terminal
                    print(f"\n{'='*60}")
                    print(f"📄 RAW {content_type.upper()} CONTENT:")
                    print(f"{'='*60}")
                    print(content_to_use[:3000] if len(content_to_use) > 3000 else content_to_use)
                    if len(content_to_use) > 3000:
                        print(f"\n... ({len(content_to_use) - 3000} more characters)")
                    print(f"{'='*60}\n")
                    
                    # Extract structured data if requested
                    if extract:
                        if content_type == "html":
                            # Parse HTML with BeautifulSoup first
                            data = self._extract_from_html(data, content_to_use)
                            # Also try markdown extraction as fallback
                            data = self._extract_from_markdown(data, content_to_use)
                        else:
                            data = self._extract_from_markdown(data, content_to_use)
                else:
                    print(f"  ⚠️  No content available")
                    print(f"  🔍 Result keys: {list(result.__dict__.keys()) if hasattr(result, '__dict__') else 'N/A'}")
                
                # Get metadata if available
                if result.metadata:
                    try:
                        if hasattr(result.metadata, 'title'):
                            data.project_name = result.metadata.title
                            print(f"  ✓ Page title: {data.project_name}")
                        elif isinstance(result.metadata, dict):
                            if 'title' in result.metadata:
                                data.project_name = result.metadata['title']
                                print(f"  ✓ Page title: {data.project_name}")
                            if 'url' in result.metadata:
                                print(f"  ✓ Final URL: {result.metadata['url']}")
                    except Exception as e:
                        print(f"  ⚠️  Error reading metadata: {e}")
                
                # Check for empty content warning
                if not content_to_use or len(content_to_use) < 100:
                    print(f"\n⚠️  WARNING: Content appears to be empty or blocked")
                    print(f"   This site may have anti-bot protection.")
                    print(f"   Try: --no-headless to see what's happening")
                    print(f"   Or use firecrawl_scraper.py which uses cloud-based bypass")
                
                return data
                
        except Exception as e:
            print(f"  ❌ Crawl4AI error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _extract_from_markdown(self, data: ApartmentData, markdown: str) -> ApartmentData:
        """Extract apartment data from markdown content using regex patterns"""
        print(f"  🔍 Extracting structured data from markdown...")
        
        # Area patterns (e.g., "78.5 м²", "Площадь: 45 кв.м")
        area_patterns = [
            r'(\d+(?:\.\d+)?)\s*м²',
            r'(\d+(?:\.\d+)?)\s*м2',
            r'(\d+(?:\.\d+)?)\s*кв\.\s*м',
            r'площадь[:\s]*(\d+(?:\.\d+)?)',
            r'общая\s+площадь[:\s]*(\d+(?:\.\d+)?)',
        ]
        
        for pattern in area_patterns:
            match = re.search(pattern, markdown, re.IGNORECASE)
            if match:
                try:
                    data.total_area = float(match.group(1))
                    print(f"  ✓ Found area: {data.total_area} m²")
                    break
                except ValueError:
                    pass
        
        # Floor patterns (e.g., "5 этаж из 24", "Этаж: 3/17")
        floor_patterns = [
            r'(\d+)\s*этаж\s*из\s*(\d+)',
            r'этаж[:\s]*(\d+)\s*/\s*(\d+)',
            r'(\d+)\s*/\s*(\d+)\s*этаж',
        ]
        
        for pattern in floor_patterns:
            match = re.search(pattern, markdown, re.IGNORECASE)
            if match:
                try:
                    data.floor = int(match.group(1))
                    data.floors_total = int(match.group(2))
                    print(f"  ✓ Found floor info: {data.floor}/{data.floors_total}")
                    break
                except (ValueError, IndexError):
                    pass
        
        # Ceiling height patterns
        ceiling_patterns = [
            r'высота\s*потолков[:\s]*(\d+(?:\.\d+)?)',
            r'потолки[:\s]*(\d+(?:\.\d+)?)',
            r'потолок[:\s]*(\d+(?:\.\d+)?)\s*м',
        ]
        
        for pattern in ceiling_patterns:
            match = re.search(pattern, markdown, re.IGNORECASE)
            if match:
                try:
                    data.ceiling_height = float(match.group(1))
                    print(f"  ✓ Found ceiling height: {data.ceiling_height} m")
                    break
                except ValueError:
                    pass
        
        # Property type (rooms)
        room_patterns = [
            (r'(\d+)-комнатн', 'numbered'),
            (r'(\d+)\s*ккв', 'numbered'),
            (r'студия', 'studio'),
            (r'однокомнатн', '1'),
            (r'двухкомнатн', '2'),
            (r'трехкомнатн', '3'),
        ]
        
        for pattern, ptype in room_patterns:
            match = re.search(pattern, markdown, re.IGNORECASE)
            if match:
                if ptype == 'studio':
                    data.property_type = 'Студия'
                elif ptype == 'numbered':
                    rooms = int(match.group(1))
                    data.property_type = f'{rooms} ккв'
                else:
                    data.property_type = f'{ptype} ккв'
                print(f"  ✓ Found property type: {data.property_type}")
                break
        
        # Building type
        building_patterns = {
            'монолит': 'Монолит',
            'панель': 'Панель',
            'кирпич-монолит': 'Кирпич-монолит',
            'кирпич': 'Кирпич',
        }
        
        for pattern, btype in building_patterns.items():
            if re.search(pattern, markdown, re.IGNORECASE):
                data.building_type = btype
                print(f"  ✓ Found building type: {data.building_type}")
                break
        
        # Finishing
        finishing_patterns = {
            'чистовая': 'Чистовая',
            'подчистовая': 'Подчистовая',
            r'без\s*отделки': 'Без отделки',
        }
        
        for pattern, ftype in finishing_patterns.items():
            if re.search(pattern, markdown, re.IGNORECASE):
                data.finishing = ftype
                print(f"  ✓ Found finishing: {data.finishing}")
                break
        
        # Price patterns
        price_patterns = [
            r'(\d[\d\s\xa0]*)\s*₽',
            r'цена[:\s]*(\d[\d\s\xa0]*)',
            r'стоимость[:\s]*(\d[\d\s\xa0]*)',
            r'(\d[\d\s]*)\s*руб',
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, markdown, re.IGNORECASE)
            if match:
                try:
                    price_str = match.group(1).replace(' ', '').replace('\xa0', '').replace(',', '')
                    data.price = float(price_str)
                    print(f"  ✓ Found price: {data.price:,.0f} ₽")
                    break
                except ValueError:
                    pass
        
        # Class option
        class_patterns = {
            'эконом': 'Эконом',
            'комфорт': 'Комфорт',
            'бизнес': 'Бизнес',
            'элит': 'Элит',
        }
        
        for pattern, cname in class_patterns.items():
            if re.search(pattern, markdown, re.IGNORECASE):
                data.class_option = cname
                print(f"  ✓ Found class: {data.class_option}")
                break
        
        # Calculate confidence
        fields_filled = sum([
            data.total_area is not None,
            data.floor is not None,
            data.property_type is not None,
            data.building_type is not None,
            data.finishing is not None,
            data.price is not None,
            data.ceiling_height is not None,
        ])
        data.extraction_confidence = fields_filled / 7.0
        
        return data
    
    def _extract_from_html(self, data: ApartmentData, html: str) -> ApartmentData:
        """Extract apartment data from HTML content using BeautifulSoup"""
        if BeautifulSoup is None:
            print(f"  ⚠️  BeautifulSoup not available, skipping HTML parsing")
            return data
        
        print(f"  🔍 Parsing HTML content ({len(html)} chars)...")
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Try to extract from title tag
            title_tag = soup.find('title')
            if title_tag and title_tag.string:
                title_text = title_tag.string
                print(f"  🔍 Parsing title: {title_text[:100]}")
                
                # Extract property type from title (e.g., "3-комнатную")
                room_match = re.search(r'(\d+)-комнатную', title_text)
                if room_match and not data.property_type:
                    rooms = int(room_match.group(1))
                    data.property_type = f'{rooms} ккв'
                    print(f"  ✓ Found property type (title): {data.property_type}")
                
                # Extract price from title
                price_match = re.search(r'(\d[\d\s]*)\s*руб', title_text)
                if price_match and not data.price:
                    price_str = price_match.group(1).replace(' ', '').replace('\xa0', '')
                    try:
                        data.price = float(price_str)
                        print(f"  ✓ Found price (title): {data.price:,.0f} ₽")
                    except ValueError:
                        pass
                
                # Extract project name from title (ЖК «Name»)
                project_match = re.search(r'ЖК\s*«([^»]+)»', title_text)
                if project_match and not data.project_name:
                    data.project_name = project_match.group(1)
                    print(f"  ✓ Found project name (title): {data.project_name}")
            
            # Try to find JSON-LD structured data
            scripts = soup.find_all('script', type='application/ld+json')
            for script in scripts:
                try:
                    json_data = json.loads(script.string)
                    print(f"  ✓ Found JSON-LD data: {json_data.get('@type', 'unknown type')}")
                    
                    # Extract from JSON-LD
                    if 'floorSize' in json_data:
                        floor_size = json_data['floorSize']
                        if isinstance(floor_size, dict):
                            value = floor_size.get('value') or floor_size.get('@value')
                            if value and not data.total_area:
                                data.total_area = float(value)
                                print(f"  ✓ Found area (JSON-LD): {data.total_area} m²")
                    
                    if 'offers' in json_data:
                        offers = json_data['offers']
                        if isinstance(offers, dict) and 'price' in offers:
                            if not data.price:
                                data.price = float(offers['price'])
                                print(f"  ✓ Found price (JSON-LD): {data.price:,.0f} ₽")
                    
                    if 'address' in json_data:
                        address = json_data['address']
                        if isinstance(address, dict):
                            locality = address.get('addressLocality', '')
                            if locality and not data.district:
                                data.district = locality
                                print(f"  ✓ Found district (JSON-LD): {data.district}")
                                
                except (json.JSONDecodeError, AttributeError):
                    continue
            
            # Look for common data attributes and classes
            # Try to find elements with data attributes
            area_elem = soup.find(attrs={'data-area': True}) or soup.find(class_=re.compile(r'area|площадь', re.I))
            if area_elem and not data.total_area:
                text = area_elem.get_text(strip=True)
                match = re.search(r'(\d+(?:\.\d+)?)', text)
                if match:
                    data.total_area = float(match.group(1))
                    print(f"  ✓ Found area (element): {data.total_area} m²")
            
            # Look for price elements
            price_elem = soup.find(attrs={'data-price': True}) or soup.find(class_=re.compile(r'price|цена|стоимость', re.I))
            if price_elem and not data.price:
                text = price_elem.get_text(strip=True)
                match = re.search(r'(\d[\d\s\xa0]*)', text)
                if match:
                    try:
                        price_str = match.group(1).replace(' ', '').replace('\xa0', '').replace(',', '')
                        data.price = float(price_str)
                        print(f"  ✓ Found price (element): {data.price:,.0f} ₽")
                    except ValueError:
                        pass
            
            # Look for floor information
            floor_elem = soup.find(attrs={'data-floor': True}) or soup.find(class_=re.compile(r'floor|этаж', re.I))
            if floor_elem and not data.floor:
                text = floor_elem.get_text(strip=True)
                match = re.search(r'(\d+)\s*из\s*(\d+)', text)
                if match:
                    data.floor = int(match.group(1))
                    data.floors_total = int(match.group(2))
                    print(f"  ✓ Found floor info (element): {data.floor}/{data.floors_total}")
            
            # Extract from meta tags
            meta_desc = soup.find('meta', {'name': 'description'})
            if meta_desc and meta_desc.get('content'):
                desc = meta_desc['content']
                print(f"  🔍 Parsing meta description...")
                
                # Try to extract area from description
                if not data.total_area:
                    match = re.search(r'(\d+(?:\.\d+)?)\s*м²', desc)
                    if match:
                        data.total_area = float(match.group(1))
                        print(f"  ✓ Found area (meta): {data.total_area} m²")
                
                # Try to extract price from description
                if not data.price:
                    match = re.search(r'(\d[\d\s\xa0]*)\s*₽', desc)
                    if match:
                        try:
                            price_str = match.group(1).replace(' ', '').replace('\xa0', '').replace(',', '')
                            data.price = float(price_str)
                            print(f"  ✓ Found price (meta): {data.price:,.0f} ₽")
                        except ValueError:
                            pass
            
            print(f"  ✅ HTML parsing complete")
            
        except Exception as e:
            print(f"  ⚠️  Error parsing HTML: {e}")
        
        return data


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


def main():
    parser = argparse.ArgumentParser(
        description='Scrape SAMOLET website using Crawl4AI (Open-source)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Setup:
    pip install crawl4ai beautifulsoup4
    crawl4ai-setup

Examples:
    # Basic scraping:
    python crawl4ai_scraper.py --url "https://samolet.ru/..."
    
    # With data extraction:
    python crawl4ai_scraper.py --url "..." --extract
    
    # Visible browser (for debugging):
    python crawl4ai_scraper.py --url "..." --no-headless

Note:
    - Crawl4AI is open-source and free (no API key needed)
    - Uses headless Chromium browser
    - Great for JavaScript-heavy sites
    
⚠️  IMPORTANT - SAMOLET is VERY SLOW:
    The SAMOLET website takes 2-3 minutes to fully load!
    The scraper now waits extensively for content, but be patient.
    
    If you still get empty results:
    1. --no-headless flag to debug visually
    2. firecrawl_scraper.py (cloud-based, requires API key)
    3. browser_scraper.py (Playwright-based alternative)
        """
    )
    
    parser.add_argument('--url', type=str,
                        default="https://samolet.ru/project/oktyabrskaya-98/flats/308985/",
                        help='URL to scrape')
    parser.add_argument('--extract', action='store_true',
                        help='Attempt to extract structured apartment data')
    parser.add_argument('--no-headless', action='store_true',
                        help='Show browser window (useful for debugging)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  🕷️  Crawl4AI Scraper for SAMOLET")
    print("  Open-source web scraping")
    print("="*60)
    
    # Initialize scraper
    scraper = Crawl4AIScraper(headless=not args.no_headless)
    
    # Run scraper
    data = asyncio.run(scraper.scrape(args.url, extract=args.extract))
    
    if data:
        if args.extract:
            print_apartment_info(data)
        
        print("\n✅ Scraping completed!")
        print("\n📋 Raw content is displayed above.")
        print("Use --extract to see structured data extraction.")
    else:
        print("\n❌ Scraping failed")
        print("\nTroubleshooting:")
        print("  • Check if URL is accessible in browser")
        print("  • Try --no-headless to see what's happening")
        print("  • Ensure crawl4ai-setup completed successfully")


if __name__ == "__main__":
    main()
