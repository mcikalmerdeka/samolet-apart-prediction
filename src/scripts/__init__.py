"""
SAMOLET Apartment Price Prediction System - Scripts Module

This module contains web scraping scripts for extracting apartment data
from SAMOLET property listings. All scrapers are documented with their
limitations due to anti-bot protection.

Available Scrapers:
    - web_scraper.py: Basic requests-based scraper (HTTP 403 blocked)
    - browser_scraper.py: Playwright browser automation (HTTP 403 blocked)
    - crawl4ai_scraper.py: Open-source AI crawler (HTTP 403 blocked)
    - firecrawl_scraper.py: Cloud-based service (requires API key)
    - firecrawl_predict.py: Prediction interface using FireCrawl
    - crawl4ai_check.py: Crawl4AI installation checker

Note:
    All scrapers are currently blocked by SAMOLET's anti-bot protection.
    See WEB_SCRAPING_README.md for detailed documentation.
"""

__all__ = []
