# Web Scraping Solutions for SAMOLET Apartment Data

This directory contains multiple approaches to extract apartment information from SAMOLET website URLs for price prediction.

## 🎯 Problem

The SAMOLET website (`https://samolet.ru/`) has anti-bot protection that blocks standard HTTP requests (401 error). We need to extract apartment data from URLs like:
```
https://samolet.ru/project/oktyabrskaya-98/flats/308985/
```

## 📁 Created Scripts

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
All outputs are saved to `firecrawl_outputs/` folder:
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
All outputs are saved to `firecrawl_outputs/` folder:
- Scraped apartment details
- Predicted price vs listed price
- Price difference analysis
- `--save`: JSON export (`prediction_{flat_id}_{timestamp}.json`)
- `--save-output`: Formatted report with extracted data + raw content (`firecrawl_output_{hash}_{flat_id}_{timestamp}.txt`)

---

### 3. **browser_scraper.py**
**Browser automation using Playwright**

**Features:**
- Real Chromium browser execution
- Human-like behavior simulation
- Best for heavily protected sites
- Open source (free)

**Installation:**
```bash
pip install playwright beautifulsoup4
playwright install chromium
```

**Usage:**
```bash
# Headless mode (faster)
python scripts/browser_scraper.py --url "..." --headless

# Visible browser (for debugging)
python scripts/browser_scraper.py --url "..."

# Save JSON results
python scripts/browser_scraper.py --url "..." --headless --save

# Save formatted output to text file
python scripts/browser_scraper.py --url "..." --save-output
```

**Output Files:**
All outputs are saved to `firecrawl_outputs/` folder:
- `--save`: Saves structured JSON results (`apartment_browser_{flat_id}_{timestamp}.json`)
- `--save-output`: Saves formatted report with extracted data + raw HTML (`browser_output_{hash}_{flat_id}_{timestamp}.txt`)

**Pros:**
- Bypasses most anti-bot protection
- Free to use
- Full JavaScript support

**Cons:**
- Slower than API-based approaches
- Requires browser installation
- More resource-intensive

---

### 4. **web_scraper.py**
**Basic HTTP-based scraper (requests + BeautifulSoup)**

**Features:**
- Simple HTTP requests
- User agent rotation
- HTML parsing
- JSON-LD extraction

**Usage:**
```bash
# Basic scraping
python scripts/web_scraper.py --url "..."

# Test connectivity
python scripts/web_scraper.py --test-connection

# Save JSON results
python scripts/web_scraper.py --url "..." --save

# Save formatted output to text file
python scripts/web_scraper.py --url "..." --save-output
```

**Output Files:**
All outputs are saved to `firecrawl_outputs/` folder:
- `--save`: Saves structured JSON results (`apartment_{flat_id}_{timestamp}.json`)
- `--save-output`: Saves formatted report with extracted data + raw HTML (`webscraper_output_{hash}_{flat_id}_{timestamp}.txt`)

**Limitations:**
- ❌ Cannot bypass SAMOLET's anti-bot protection (401 error)
- ✅ Good for testing and fallback documentation

---

### 5. **demo_scraping.py**
**Testing and comparison script**

**Usage:**
```bash
# Compare all approaches
python scripts/demo_scraping.py --compare

# Test specific scraper
python scripts/demo_scraping.py --firecrawl "https://samolet.ru/..."
python scripts/demo_scraping.py --browser "https://samolet.ru/..."

# Run all tests
python scripts/demo_scraping.py --test-all
```

---

## 🏆 Recommended Approach

### Production Setup (Best Reliability)

**Hybrid Architecture:**

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
1. **Automation first**: Try FireCrawl for speed
2. **User verification**: Show extracted data for confirmation
3. **Always works**: Manual input fallback ensures system availability
4. **Best accuracy**: User can correct any extraction errors

---

## 🔑 Quick Start

### Step 1: Install Dependencies
```bash
pip install firecrawl-py python-dotenv
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

# Save raw output to text file for review
python firecrawl_scraper.py --url "..." --save-output

# Batch processing
python firecrawl_scraper.py --batch urls.txt

# Save JSON results
python firecrawl_scraper.py --url "..." --extract --save
```

**Note on API Version:**
This script uses FireCrawl SDK v1.x API:
- Class: `Firecrawl` (not `FirecrawlApp`)
- Method: `.scrape()` (not `.scrape_url()`)
- Parameters passed as keyword arguments (not nested in `params` dict)

### Step 2: Get FireCrawl API Key
1. Visit https://www.firecrawl.dev/
2. Sign up (free tier available)
3. Copy your API key

### Step 3: Create .env File
```bash
# Copy the example file
cp .env.example .env

# Edit .env with your API key
echo "FIRECRAWL_API_KEY=your-api-key-here" > .env

# Or manually edit .env file:
# FIRECRAWL_API_KEY=your-api-key-here
```

**Why .env file?**
- ✅ Cleaner than environment variables
- ✅ Easy to manage multiple configurations
- ✅ Won't accidentally leak in terminal history
- ✅ Standard practice for Python projects
```

### Step 4: Run Full Pipeline
```bash
# Basic prediction
python firecrawl_predict.py --url "https://samolet.ru/project/oktyabrskaya-98/flats/308985/"

# Save JSON results
python firecrawl_predict.py --url "..." --save

# Save formatted report for review
python firecrawl_predict.py --url "..." --save-output

# Both JSON and formatted report
python firecrawl_predict.py --url "..." --save --save-output
```

**Output Location:** All results saved to `firecrawl_outputs/` folder

---

## 📊 Approach Comparison

| Approach | Cost | Speed | Anti-Bot | Setup | Best For |
|----------|------|-------|----------|-------|----------|
| **FireCrawl** | $16-83/mo | Fast | Good | Easy | ⭐ Production use |
| **Browser** | Free | Slow | Best | Complex | Protected sites |
| **HTTP Basic** | Free | Fast | None | None | Testing only |
| **Manual** | Free | N/A | N/A | None | Fallback |

---

## 🎯 Integration with main.py

To add URL scraping to your Gradio interface:

```python
# In main.py, add to create_gradio_interface()

with gr.Tab("🔗 URL Input"):
    url_input = gr.Textbox(
        label="Apartment Listing URL",
        placeholder="https://samolet.ru/project/.../flats/.../",
        info="Paste SAMOLET apartment URL"
    )
    
    scrape_btn = gr.Button("🌐 Scrape & Predict")
    scrape_output = gr.Markdown()
    
    def scrape_and_predict_wrapper(url):
        result = scrape_and_predict(url)  # From firecrawl_predict.py
        return format_result(result)
    
    scrape_btn.click(
        fn=scrape_and_predict_wrapper,
        inputs=[url_input],
        outputs=scrape_output
    )
```

---

## 📁 Unified Output System

All scraping scripts now use a **single unified output folder** (`firecrawl_outputs/`) for consistency:

### Output Structure:
```
firecrawl_outputs/
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

This unified approach makes it easy to:
- ✅ Find all outputs in one location
- ✅ Compare results from different scrapers
- ✅ Review raw content for debugging
- ✅ Track scraping history

---

## 🐛 Troubleshooting

### FireCrawl returns limited data
- The site may have advanced anti-bot protection
- Try the browser scraper as fallback
- Use manual input mode with verification

### Browser scraper fails
- Ensure Playwright is installed: `playwright install chromium`
- Try running with visible browser (remove --headless)
- Check network connectivity

### API key errors
- Verify `FIRECRAWL_API_KEY` environment variable is set
- Check if API key is valid and has credits remaining

### Model prediction fails
- Ensure model artifacts exist in `model_artifacts/`
- Run training pipeline if needed
- Check feature names match between scraper and model

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

## 🚀 Future Enhancements

1. **Caching Layer**: Redis/Memcached for scraped data
2. **Rate Limiting**: Respect site terms of service
3. **Proxy Rotation**: For higher success rates
4. **API Endpoint**: Direct SAMOLET API integration
5. **Mobile App**: React Native with scraping backend

---

## 📞 Support

- **FireCrawl Docs**: https://docs.firecrawl.dev/
- **Playwright Docs**: https://playwright.dev/
- **SAMOLET Website**: https://samolet.ru/

---

## ⚖️ Legal Note

Always respect website Terms of Service:
- Check `robots.txt` before scraping
- Use reasonable request rates (caching helps)
- Consider official API if available
- This is for educational/demonstration purposes

---

## ✅ Summary

You now have **4 different approaches** to extract SAMOLET apartment data:

1. **FireCrawl** - Best for production (AI-powered)
2. **Browser** - Best for protection bypass (real browser)
3. **HTTP** - Simple but blocked (testing only)
4. **Manual** - Always works (user input)

**Next Step:** Get a FireCrawl API key and test with:
```bash
# FireCrawl (Recommended)
python firecrawl_predict.py --url "https://samolet.ru/project/oktyabrskaya-98/flats/308985/" --save-output

# Browser scraper (fallback for protected sites)
python scripts/browser_scraper.py --url "..." --headless --save-output

# Web scraper (for testing)
python scripts/web_scraper.py --url "..." --save-output
```

All outputs saved to `firecrawl_outputs/` folder for easy review!

Happy scraping! 🏠🔥
