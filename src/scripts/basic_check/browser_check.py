"""
Simple script to check if the Playwright browser scraper is working.
"""

from playwright.sync_api import sync_playwright

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        # Testing for random url
        page.goto("https://karpathy.github.io/neuralnets/")
        
        # # The url is the url of the apartment that we want to scrape.
        # page.goto("https://samolet.ru/project/novograd-pavlino/flats/455363/")
        
        print(f"✅ Page loaded successfully")
        print(f"📄 Title: {page.title()}")
        print(f"📝 Content preview (first 500 chars):")
        print("-" * 60)
        content = page.content()
        print(content[:500])
        print("-" * 60)
        
        browser.close()

if __name__ == "__main__":
    main()
