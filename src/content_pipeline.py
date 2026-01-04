import asyncio
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from .content_scraper import ContentScraper, ScrapedContent
from .content_parser import ContentParser, ParsedContent
from .content_chunker import ContentChunker, ChunkedContent
from .embedding_generator import GeminiEmbeddingGenerator, generate_embeddings_for_content
from .qdrant_uploader import ContentUploadManager, upload_embeddings_to_qdrant


# Load environment variables
load_dotenv()


class ContentProcessingPipeline:
    """
    Comprehensive pipeline for processing textbook content:
    1. Scrape content from textbook website
    2. Parse HTML to extract meaningful content
    3. Chunk content appropriately for embeddings
    4. Generate embeddings using GEMINI
    5. Upload embeddings to Qdrant
    """
    
    def __init__(self, 
                 base_url: str,
                 max_tokens_per_chunk: int = 500,
                 collection_name: str = "textbook_content_embeddings"):
        """
        Initialize the content processing pipeline
        
        Args:
            base_url: The base URL of the textbook website
            max_tokens_per_chunk: Maximum tokens per content chunk
            collection_name: Name of the Qdrant collection
        """
        self.base_url = base_url
        self.content_scraper = None
        self.content_parser = ContentParser()
        self.content_chunker = ContentChunker(max_tokens=max_tokens_per_chunk)
        self.embedding_generator = GeminiEmbeddingGenerator()
        self.content_uploader = ContentUploadManager(collection_name)
    
    async def run_pipeline(self, urls: List[str]) -> Dict[str, Any]:
        """
        Run the complete content processing pipeline
        
        Args:
            urls: List of URLs to process
            
        Returns:
            Dictionary with processing statistics and results
        """
        print("Starting content processing pipeline...")
        
        # Initialize the scraper within async context
        async with ContentScraper(self.base_url) as scraper:
            self.content_scraper = scraper
            
            results = {
                "total_urls": len(urls),
                "scraped_pages": 0,
                "parsed_chunks": 0,
                "embedded_chunks": 0,
                "uploaded_chunks": 0,
                "errors": []
            }
            
            try:
                # 1. Scrape content from URLs
                print("Step 1: Scraping content...")
                scraped_contents = await scraper.scrape_multiple_pages(urls)
                results["scraped_pages"] = len([c for c in scraped_contents if c is not None])
                
                # Filter out failed scrapes
                scraped_contents = [c for c in scraped_contents if c is not None]
                
                # 2. Parse HTML content
                print("Step 2: Parsing content...")
                all_parsed_chunks = []
                for scraped in scraped_contents:
                    parsed_chunks = self.content_parser.parse_textbook_content(
                        scraped.content, 
                        scraped.hierarchy_path or f"{scraped.module_id}/{scraped.chapter_id}/{scraped.section_id}"
                    )
                    all_parsed_chunks.extend(parsed_chunks)
                
                results["parsed_chunks"] = len(all_parsed_chunks)
                print(f"Parsed {results['parsed_chunks']} content chunks")
                
                # 3. Chunk content for embeddings
                print("Step 3: Chunking content...")
                chunked_contents = []
                for parsed in all_parsed_chunks:
                    chunks = self.content_chunker.chunk_by_semantic_boundaries([parsed])
                    chunked_contents.extend(chunks)
                
                print(f"Created {len(chunked_contents)} chunked content pieces")
                
                # 4. Generate embeddings
                print("Step 4: Generating embeddings...")
                embedding_results = await generate_embeddings_for_content(chunked_contents)
                
                # Filter out failed embeddings
                successful_embeddings = [r for r in embedding_results if r['embedding'] is not None]
                results["embedded_chunks"] = len(successful_embeddings)
                
                print(f"Generated embeddings for {results['embedded_chunks']} chunks")
                
                # 5. Upload to Qdrant
                print("Step 5: Uploading to Qdrant...")
                upload_success = await upload_embeddings_to_qdrant(successful_embeddings)
                
                if upload_success:
                    results["uploaded_chunks"] = results["embedded_chunks"]
                    print("Successfully uploaded embeddings to Qdrant!")
                else:
                    results["errors"].append("Failed to upload embeddings to Qdrant")
                
                print("Content processing pipeline completed!")
                return results
                
            except Exception as e:
                error_msg = f"Error in content processing pipeline: {str(e)}"
                print(error_msg)
                results["errors"].append(error_msg)
                return results
    
    async def process_single_module(self, module_url: str) -> Dict[str, Any]:
        """
        Process a single module (chapter or section) and all its content
        
        Args:
            module_url: URL of the module to process
            
        Returns:
            Dictionary with processing results
        """
        # Discover all URLs under this module
        async with ContentScraper(self.base_url) as scraper:
            urls = await scraper.discover_content_urls(module_url)
            print(f"Discovered {len(urls)} URLs under module {module_url}")
        
        # Process all discovered URLs
        return await self.run_pipeline(urls)
    
    async def update_content(self, urls: List[str]) -> Dict[str, Any]:
        """
        Update existing content by reprocessing and replacing embeddings
        
        Args:
            urls: List of URLs to update
            
        Returns:
            Dictionary with update statistics
        """
        # For now, this is the same as run_pipeline, but in the future 
        # it could include more sophisticated update logic
        return await self.run_pipeline(urls)


class PipelineManager:
    """
    Higher-level manager to coordinate multiple pipeline runs
    """
    
    def __init__(self):
        self.pipelines = {}
    
    def register_pipeline(self, name: str, pipeline: ContentProcessingPipeline):
        """
        Register a pipeline with a name
        """
        self.pipelines[name] = pipeline
    
    async def run_pipeline_by_name(self, name: str, urls: List[str]) -> Dict[str, Any]:
        """
        Run a registered pipeline by name
        """
        if name not in self.pipelines:
            raise ValueError(f"Pipeline '{name}' not registered")
        
        return await self.pipelines[name].run_pipeline(urls)
    
    async def run_all_pipelines(self, pipeline_urls: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
        """
        Run multiple pipelines with their respective URLs
        
        Args:
            pipeline_urls: Dictionary mapping pipeline names to lists of URLs
            
        Returns:
            Dictionary mapping pipeline names to their results
        """
        results = {}
        
        for name, urls in pipeline_urls.items():
            print(f"Running pipeline: {name}")
            results[name] = await self.run_pipeline_by_name(name, urls)
        
        return results


async def run_content_pipeline(base_url: str, urls: List[str]) -> Dict[str, Any]:
    """
    Convenience function to run the content pipeline without creating a full instance
    """
    pipeline = ContentProcessingPipeline(base_url)
    return await pipeline.run_pipeline(urls)


# Example usage and testing
async def example_usage():
    """
    Example of how to use the content processing pipeline
    """
    # Example URLs (these would be actual textbook URLs in real usage)
    example_urls = [
        "https://example-textbook.com/module1/chapter1",
        "https://example-textbook.com/module1/chapter2",
        "https://example-textbook.com/module2/chapter1"
    ]
    
    # Create and run pipeline
    pipeline = ContentProcessingPipeline("https://example-textbook.com")
    results = await pipeline.run_pipeline(example_urls)
    
    print("\\nPipeline Results:")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    # Run example
    # asyncio.run(example_usage())
    print("ContentProcessingPipeline module ready. Use with actual textbook URLs and valid API keys.")