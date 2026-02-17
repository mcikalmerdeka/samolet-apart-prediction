"""
Simple script to check if the Crawl4AI is working.
"""

import asyncio
from crawl4ai import *

async def main():
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(

            # Testing for random url
            url="https://karpathy.github.io/neuralnets/"
            
            # # The url is the url of the apartment that we want to scrape.
            # url="https://samolet.ru/project/novograd-pavlino/flats/455363/"
        )
        print(result.markdown)

if __name__ == "__main__":
    asyncio.run(main())