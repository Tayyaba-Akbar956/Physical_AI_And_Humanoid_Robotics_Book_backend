import asyncio
import aiohttp
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse
import logging
from dataclasses import dataclass


@dataclass
class ScrapedContent:
    """
    Data class to represent scraped content
    """
    url: str
    title: str
    content: str
    module_id: Optional[str] = None
    chapter_id: Optional[str] = None
    section_id: Optional[str] = None
    hierarchy_path: Optional[str] = None
    content_type: str = "text"  # Default content type


class ContentScraper:
    """
    Class to scrape content from the textbook website
    """
    
    def __init__(self, base_url: str, headers: Optional[Dict[str, str]] = None):
        self.base_url = base_url
        self.headers = headers or {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.session = None
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def __aenter__(self):
        """
        Async context manager entry
        """
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Async context manager exit
        """
        if self.session:
            await self.session.close()
    
    async def scrape_single_page(self, url: str) -> Optional[ScrapedContent]:
        """
        Scrape content from a single page
        """
        try:
            if not self.session:
                raise RuntimeError("Scraper session not initialized. Use 'async with' syntax.")
            
            async with self.session.get(url) as response:
                if response.status != 200:
                    self.logger.error(f"Failed to fetch {url}, status: {response.status}")
                    return None
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract title
                title_tag = soup.find('title')
                title = title_tag.get_text().strip() if title_tag else "No Title"
                
                # Extract content (excluding navigation, headers, footers, etc.)
                # This is a basic approach - in practice, you'd need more sophisticated logic
                # to identify the main content area of the textbook
                main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content') or soup.find('div', id='content') or soup.find('body')
                
                if main_content:
                    # Remove script and style elements
                    for script in main_content(["script", "style"]):
                        script.decompose()
                    
                    # Get text and clean it
                    content = main_content.get_text(separator='\\n')
                    # Clean up whitespace
                    content = '\\n'.join([line.strip() for line in content.split('\\n') if line.strip()])
                else:
                    content = ""
                
                # Determine content type and hierarchy (simplified)
                # In a real implementation, you would parse the URL structure or page structure
                # to determine module, chapter, and section
                parsed_url = urlparse(url)
                path_parts = parsed_url.path.strip('/').split('/')
                
                module_id = path_parts[0] if len(path_parts) > 0 else None
                chapter_id = path_parts[1] if len(path_parts) > 1 else None
                section_id = path_parts[2] if len(path_parts) > 2 else None
                
                hierarchy_path = f"{module_id}/{chapter_id}/{section_id}" if module_id else ""
                
                return ScrapedContent(
                    url=url,
                    title=title,
                    content=content,
                    module_id=module_id,
                    chapter_id=chapter_id,
                    section_id=section_id,
                    hierarchy_path=hierarchy_path
                )
        except Exception as e:
            self.logger.error(f"Error scraping {url}: {e}")
            return None
    
    async def scrape_multiple_pages(self, urls: List[str]) -> List[ScrapedContent]:
        """
        Scrape content from multiple pages concurrently
        """
        tasks = [self.scrape_single_page(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out None results and exceptions
        valid_results = []
        for result in results:
            if isinstance(result, ScrapedContent):
                valid_results.append(result)
            elif isinstance(result, Exception):
                self.logger.error(f"Scraping task failed: {result}")
        
        return valid_results
    
    async def discover_content_urls(self, start_url: str, max_pages: int = 100) -> List[str]:
        """
        Discover content URLs starting from a given URL
        This is a simplified implementation - in real usage, you'd need to adapt this
        based on the actual structure of the textbook website
        """
        urls_to_visit = {start_url}
        visited_urls = set()
        discovered_urls = []
        
        while urls_to_visit and len(discovered_urls) < max_pages:
            current_url = urls_to_visit.pop()
            
            if current_url in visited_urls:
                continue
            
            visited_urls.add(current_url)
            
            try:
                async with self.session.get(current_url) as response:
                    if response.status != 200:
                        continue
                    
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Find all links on the page
                    links = soup.find_all('a', href=True)
                    
                    for link in links:
                        href = link['href']
                        absolute_url = urljoin(current_url, href)
                        
                        # Only add URLs that are within the textbook domain
                        if self._is_valid_textbook_url(absolute_url):
                            if absolute_url not in visited_urls and absolute_url not in urls_to_visit:
                                urls_to_visit.add(absolute_url)
                                discovered_urls.append(absolute_url)
                                
                                if len(discovered_urls) >= max_pages:
                                    break
            except Exception as e:
                self.logger.error(f"Error discovering URLs from {current_url}: {e}")
        
        return discovered_urls[:max_pages]
    
    def _is_valid_textbook_url(self, url: str) -> bool:
        """
        Check if a URL is within the textbook domain
        """
        try:
            parsed_url = urlparse(url)
            base_parsed = urlparse(self.base_url)
            return parsed_url.netloc == base_parsed.netloc
        except:
            return False


# Example usage function
async def example_usage():
    """
    Example of how to use the ContentScraper
    """
    async with ContentScraper("https://example-textbook-site.com") as scraper:
        # Scrape specific URLs
        urls = [
            "https://example-textbook-site.com/module1/chapter1",
            "https://example-textbook-site.com/module1/chapter2",
            "https://example-textbook-site.com/module2/chapter1"
        ]
        
        contents = await scraper.scrape_multiple_pages(urls)
        
        for content in contents:
            print(f"Title: {content.title}")
            print(f"URL: {content.url}")
            print(f"Content preview: {content.content[:200]}...")
            print("---")
        
        # Discover and scrape more pages
        discovered_urls = await scraper.discover_content_urls("https://example-textbook-site.com", max_pages=10)
        print(f"Discovered {len(discovered_urls)} URLs")


if __name__ == "__main__":
    # Run example (this would be adapted based on your actual textbook URL)
    # asyncio.run(example_usage())
    print("ContentScraper module ready. Use within an async context manager.")