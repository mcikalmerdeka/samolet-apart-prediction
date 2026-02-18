# Web Scraping Solutions for SAMOLET Apartment Data

This directory contains multiple approaches to extract apartment information from SAMOLET website URLs for price prediction.

## 🎯 Problem

The SAMOLET website (`https://samolet.ru/`) has advanced anti-bot protection that blocks automated scraping attempts with HTTP 401/403 errors:

- **401 Unauthorized** - Returned by request-based and browser-based scrapers
- **403 Forbidden** - Returned by more advanced scraping frameworks (Crawl4AI, Cloudflare blocking)

Example URL:
```
https://samolet.ru/project/oktyabrskaya-98/flats/308985/
```

## 📁 Project Structure

```
src/scripts/
├── basic_check/              # Simple verification scripts
│   ├── crawl4ai_check.py     # Verify Crawl4AI installation
│   ├── browser_check.py      # Verify Playwright browser automation
│   └── web_check.py          # Verify requests/BeautifulSoup
│
├── web_scraper.py            # HTTP requests-based scraper
├── browser_scraper.py        # Playwright browser automation
├── crawl4ai_scraper.py       # Open-source AI-powered crawling
├── firecrawl_scraper.py      # Cloud-based FireCrawl API scraper ⭐ RECOMMENDED
├── firecrawl_predict.py      # Complete pipeline: Scraping → Prediction
└── WEB_SCRAPING_README.md    # This file
```

## 🧪 Quick Verification

Before running full scrapers, verify your setup with simple check scripts:

```bash
# Verify each scraping method works on a safe test URL
python src/scripts/basic_check/web_check.py         # HTTP requests
python src/scripts/basic_check/browser_check.py     # Playwright browser
python src/scripts/basic_check/crawl4ai_check.py    # Crawl4AI framework
```

These scripts test the basic functionality using `https://karpathy.github.io/neuralnets/` (a safe, unprotected site).

---

## 📊 Scraping Approaches

### 1. **firecrawl_scraper.py** ⭐ RECOMMENDED
**AI-powered web scraping using FireCrawl API**

**Features:**
- JavaScript rendering for dynamic sites
- LLM-based structured data extraction
- Caching to reduce API costs
- Batch processing support
- Handles most anti-bot protection

**Installation:**
```bash
pip install firecrawl-py python-dotenv
# Get API key: https://www.firecrawl.dev/
```

**Configuration (.env file):**
Create a `.env` file in the project directory:
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your API key
FIRECRAWL_API_KEY=your-api-key-here
```

**Usage:**
```bash
# Basic scraping (API key loaded from .env)
python firecrawl_scraper.py --url "https://samolet.ru/project/oktyabrskaya-98/flats/308985/"

# AI extraction with structured data
python firecrawl_scraper.py --url "..." --extract

# Save JSON results
python firecrawl_scraper.py --url "..." --extract --save

# Save raw output to text file for review
python firecrawl_scraper.py --url "..." --extract --save-output

# Batch processing
python firecrawl_scraper.py --batch urls.txt
```

**Output Files:**
All outputs are saved to `output/` folder:
- `--save`: Saves structured JSON results (`firecrawl_{flat_id}_{timestamp}.json`)
- `--save-output`: Saves formatted report with extracted data + raw content (`firecrawl_output_{hash}_{flat_id}_{timestamp}.txt`)

**Pricing:**
- Free tier: 500 credits (one-time)
- Hobby: $16/month (3,000 credits)
- Standard: $83/month (100,000 credits)

---

### 2. **firecrawl_predict.py** ⭐ FULL PIPELINE
**Complete integration: Scraping → Extraction → Prediction**

**Features:**
- One-command pipeline from URL to price prediction
- Maps scraped data to model inputs
- Compares predicted vs listed price
- Saves structured results

**Usage:**
```bash
# Full pipeline
python firecrawl_predict.py --url "https://samolet.ru/project/oktyabrskaya-98/flats/308985/"

# Save JSON results with predictions
python firecrawl_predict.py --url "..." --save

# Save raw scraper output for review
python firecrawl_predict.py --url "..." --save-output

# Force fresh scrape (no cache)
python firecrawl_predict.py --url "..." --no-cache
```

**Output:**
All outputs are saved to `output/` folder:
- Scraped apartment details
- Predicted price vs listed price
- Price difference analysis
- `--save`: JSON export (`prediction_{flat_id}_{timestamp}.json`)
- `--save-output`: Formatted report with extracted data + raw content (`firecrawl_output_{hash}_{flat_id}_{timestamp}.txt`)

---

### 3. **crawl4ai_scraper.py**
**Open-source AI-powered web crawling**

**Features:**
- Open source (free, no API key required)
- JavaScript rendering with headless browser
- Markdown extraction
- Extended wait times for slow-loading pages
- HTML parsing fallback

**Error Code:** HTTP 403 Forbidden (blocked by SAMOLET's anti-bot protection)

**Installation:**
```bash
pip install crawl4ai beautifulsoup4
crawl4ai-setup  # Installs browser dependencies
```

**Usage:**
```bash
# Basic scraping (slow sites need time to load)
python crawl4ai_scraper.py --url "..."

# With data extraction
python crawl4ai_scraper.py --url "..." --extract

# Visible browser for debugging
python crawl4ai_scraper.py --url "..." --no-headless
```

**Note:** Despite extended wait times (3 minutes) and anti-detection measures, SAMOLET's protection blocks this method with HTTP 403.

---

### 4. **browser_scraper.py**
**Browser automation using Playwright**

**Features:**
- Real Chromium browser execution
- Human-like behavior simulation
- Anti-detection measures (browser fingerprint evasion)
- Scroll simulation
- JavaScript data extraction

**Error Code:** HTTP 401 Unauthorized (blocked by SAMOLET's anti-bot protection)

**Installation:**
```bash
pip install playwright beautifulsoup4
playwright install chromium
```

**Usage:**
```bash
# Headless mode (faster)
python browser_scraper.py --url "..." --headless

# Visible browser (for debugging)
python browser_scraper.py --url "..."

# Save JSON results
python browser_scraper.py --url "..." --headless --save

# Save formatted output to text file
python browser_scraper.py --url "..." --save-output
```

**Output Files:**
All outputs are saved to `output/` folder:
- `--save`: Saves structured JSON results (`apartment_browser_{flat_id}_{timestamp}.json`)
- `--save-output`: Saves formatted report with extracted data + raw HTML (`browser_output_{hash}_{flat_id}_{timestamp}.txt`)

**Note:** Despite browser automation and anti-detection, SAMOLET blocks with HTTP 401.

---

### 5. **web_scraper.py**
**Basic HTTP-based scraper (requests + BeautifulSoup)**

**Features:**
- Simple HTTP requests with user-agent rotation
- HTML parsing and JSON-LD extraction
- Multiple retry attempts with exponential backoff
- Table-based data extraction

**Error Code:** HTTP 401 Unauthorized (blocked by SAMOLET's anti-bot protection)

**Usage:**
```bash
# Basic scraping
python web_scraper.py --url "..."

# Test connectivity
python web_scraper.py --test-connection

# Save JSON results
python web_scraper.py --url "..." --save

# Save formatted output to text file
python web_scraper.py --url "..." --save-output
```

**Output Files:**
All outputs are saved to `output/` folder:
- `--save`: Saves structured JSON results (`apartment_{flat_id}_{timestamp}.json`)
- `--save-output`: Saves formatted report with extracted data + raw HTML (`webscraper_output_{hash}_{flat_id}_{timestamp}.txt`)

**Note:** Simple requests-based scraping returns HTTP 401 on SAMOLET.

---

## 📊 Approach Comparison

| Approach | Cost | Speed | Error Code | SAMOLET Success | Best For |
|----------|------|-------|------------|-----------------|----------|
| **firecrawl_scraper.py** | $16-83/mo | Fast | N/A | ✅ Works | ⭐ Production use |
| **firecrawl_predict.py** | $16-83/mo | Fast | N/A | ✅ Works | ⭐ Full pipeline |
| **crawl4ai_scraper.py** | Free | Medium | HTTP 403 | ❌ Blocked | Testing, other sites |
| **browser_scraper.py** | Free | Slow | HTTP 401 | ❌ Blocked | Other protected sites |
| **web_scraper.py** | Free | Fast | HTTP 401 | ❌ Blocked | Testing, other sites |
| **Manual Input** | Free | N/A | N/A | ✅ Always works | Fallback |

**Key Findings:**
- SAMOLET uses **Cloudflare/CDN protection** that detects and blocks automated access
- **Browser fingerprinting** detects headless/automated browsers (returns 401)
- **IP-based blocking** occurs after detection (returns 403)
- **FireCrawl** cloud-based service successfully bypasses protection

---

## 🏆 Recommended Setup

### Production Architecture:

```
User Input URL
     ↓
[Try FireCrawl Scraping]
     ↓
Success? → [Extract Data] → [Show to User for Verification]
     ↓ Yes                                        ↓
     ↓                              [Allow Corrections]
     ↓                                        ↓
     ↓                              [Run Prediction]
     ↓                                        ↓
     ↓                              [Display Results]
     ↓
     ↓ No
     ↓
[Manual Input Mode] → [User Enters Data] → [Run Prediction]
```

**Why this works:**
1. **Automation first**: FireCrawl handles the protection
2. **User verification**: Show extracted data for confirmation
3. **Always works**: Manual input fallback ensures availability
4. **Best accuracy**: User can correct extraction errors

---

## 🔑 Quick Start

### Step 1: Verify Setup
```bash
# Test basic scraping methods on a safe URL
python src/scripts/basic_check/web_check.py
python src/scripts/basic_check/browser_check.py
python src/scripts/basic_check/crawl4ai_check.py
```

### Step 2: Install FireCrawl (Recommended)
```bash
pip install firecrawl-py python-dotenv
```

### Step 3: Configure API Key
```bash
# Copy example and add your key
cp .env.example .env
# Edit .env: FIRECRAWL_API_KEY=your-api-key-here
```

### Step 4: Test with SAMOLET
```bash
# Full prediction pipeline
python firecrawl_predict.py --url "https://samolet.ru/project/oktyabrskaya-98/flats/308985/"

# With all outputs
python firecrawl_predict.py --url "..." --save --save-output
```

---

## 📁 Unified Output System

All scraping scripts use a **single unified output folder** (`output/`) for consistency:

### Output Structure:
```
output/
├── firecrawl_{flat_id}_{timestamp}.json              # FireCrawl JSON results
├── firecrawl_output_{hash}_{flat_id}_{timestamp}.txt # FireCrawl formatted report
├── prediction_{flat_id}_{timestamp}.json             # Prediction results
├── apartment_{flat_id}_{timestamp}.json             # Web scraper JSON
├── webscraper_output_{hash}_{flat_id}_{timestamp}.txt # Web scraper formatted report
├── apartment_browser_{flat_id}_{timestamp}.json      # Browser scraper JSON
└── browser_output_{hash}_{flat_id}_{timestamp}.txt   # Browser scraper formatted report
```

### File Types:
- **JSON files** (`--save`): Structured data for programmatic use
- **Text files** (`--save-output`): Human-readable reports with:
  - Terminal-formatted extracted data
  - Raw HTML/content (first 5000 chars)
  - Hash ID for tracking
  - Timestamp

---

## 🐛 Error Reference

### HTTP 401 Unauthorized
**Caused by:** `web_scraper.py`, `browser_scraper.py`
**Reason:** SAMOLET requires authentication or detected automated access
**Solution:** Use FireCrawl (cloud-based bypass) or manual input

### HTTP 403 Forbidden
**Caused by:** `crawl4ai_scraper.py` (after detection)
**Reason:** IP address blocked by anti-bot protection (WAF/CDN)
**Solution:** Use FireCrawl, wait several hours, or use VPN

### Empty Content
**Caused by:** Slow-loading JavaScript sites
**Reason:** Page content loads dynamically after initial request
**Solution:** Use `--no-headless` to debug, or use FireCrawl with extended timeouts

---

## 📝 Data Fields Extracted

All scrapers attempt to extract:

**Required for Prediction:**
- ✅ Total Area (m²)
- ✅ Floor / Total Floors
- ✅ Property Type (rooms)
- ✅ Class (Эконом/Комфорт/Бизнес/Элит)
- ✅ Building Type
- ✅ District / Location
- ✅ Ceiling Height
- ✅ Finishing

**Optional:**
- 📊 Living Area, Kitchen Area, Hallway Area
- 📊 Construction Phase
- 📊 Mortgage / Subsidies availability
- 💰 Listed Price (for comparison)

---

## 🎯 Integration with main.py

The Gradio interface includes documentation of scraping attempts in the **"Link-Based Input"** tab, explaining why automated extraction is unavailable and directing users to **"Manual Input"** instead.

---

## ⚖️ Legal Note

Always respect website Terms of Service:
- Check `robots.txt` before scraping
- Use reasonable request rates (caching helps)
- Consider official API if available
- This is for educational/demonstration purposes

---

## ✅ Summary

You have **5 different approaches** to extract SAMOLET apartment data:

1. **FireCrawl Scraper** - ✅ Production ready (cloud-based bypass)
2. **FireCrawl Pipeline** - ✅ Complete solution (scrape → predict)
3. **Crawl4AI** - ❌ HTTP 403 (blocked)
4. **Browser Automation** - ❌ HTTP 401 (blocked)
5. **HTTP Requests** - ❌ HTTP 401 (blocked)
6. **Manual Input** - ✅ Always works (fallback)

**Recommended workflow:**
```bash
# 1. Verify setup
python src/scripts/basic_check/web_check.py

# 2. Use FireCrawl for SAMOLET
python firecrawl_predict.py --url "https://samolet.ru/project/oktyabrskaya-98/flats/308985/"

# 3. Review output
ls output/
```

All outputs saved to `output/` folder for easy review!

Happy scraping! 🏠🔥
