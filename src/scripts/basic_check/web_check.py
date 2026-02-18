"""
Simple script to check if the requests-based web scraper is working.
"""

import requests
from bs4 import BeautifulSoup

def main():
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    # Testing for random url
    url = "https://karpathy.github.io/neuralnets/"
    
    # # The url is the url of the apartment that we want to scrape.
    # url = "https://samolet.ru/project/novograd-pavlino/flats/455363/"
    
    response = requests.get(url, headers=headers, timeout=30)
    
    print(f"✅ Request successful")
    print(f"📊 Status Code: {response.status_code}")
    print(f"📄 Content Length: {len(response.text)} chars")
    print(f"📝 Content preview (first 500 chars):")
    print("-" * 60)
    
    # Parse and extract text
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text(separator='\n', strip=True)
    print(text[:500])
    print("-" * 60)

if __name__ == "__main__":
    main()
