import os
import asyncio
from typing import List, Dict, Any, Optional
import httpx
from dotenv import load_dotenv
from .content_chunker import ChunkedContent


# Load environment variables
load_dotenv()

# Get GEMINI API key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Gemini REST Configuration
GEMINI_REST_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent?key={key}"

def check_gemini_config():
    """
    Check if GEMINI_API_KEY is available
    """
    if not GEMINI_API_KEY:
        print("Warning: GEMINI_API_KEY is not set.")
        return False
    return True


class GeminiEmbeddingGenerator:
    """
    Class to generate embeddings using GEMINI models
    """
    
    def __init__(self, model_name: str = "embedding-001"):
        """
        Initialize the embedding generator
        
        Args:
            model_name: The name of the GEMINI embedding model to use
        """
        self.model_name = model_name
        self._embedding_model = None
    
    @property
    def embedding_model_ready(self):
        """
        Check if Gemini is configured
        """
        return check_gemini_config()
    
    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text using REST API
        """
        if not self.embedding_model_ready:
            print("Error: Gemini not configured. Check GEMINI_API_KEY.")
            return None
            
        try:
            model_name = self.model_name if not self.model_name.startswith("models/") else self.model_name.replace("models/", "")
            url = GEMINI_REST_URL.format(model=model_name, key=GEMINI_API_KEY)
            
            payload = {
                "content": {
                    "parts": [{"text": text}]
                },
                "taskType": "RETRIEVAL_QUERY"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, timeout=30.0)
                response.raise_for_status()
                data = response.json()
                
            return data.get('embedding', {}).get('values')
        except Exception as e:
            print(f"Error generating embedding via REST: {e}")
            return None
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Generate embeddings for a batch of texts
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            A list of embedding vectors (one for each text)
        """
        embeddings = []
        
        for text in texts:
            embedding = await self.generate_embedding(text)
            embeddings.append(embedding)
        
        return embeddings
    
    async def generate_embeddings_from_chunks(self, chunks: List[ChunkedContent]) -> List[Dict[str, Any]]:
        """
        Generate embeddings from chunked content
        
        Args:
            chunks: List of ChunkedContent objects
            
        Returns:
            List of dictionaries containing chunk info and embeddings
        """
        results = []
        
        for chunk in chunks:
            embedding = await self.generate_embedding(chunk.content)
            
            if embedding is not None:
                result = {
                    "id": f"emb_{chunk.id}",
                    "chunk_id": chunk.id,
                    "module_id": chunk.module_id,
                    "chapter_id": chunk.chapter_id,
                    "section_id": chunk.section_id,
                    "hierarchy_path": chunk.hierarchy_path,
                    "content_type": chunk.content_type,
                    "content": chunk.content,
                    "embedding": embedding,
                    "metadata": chunk.metadata or {}
                }
                results.append(result)
            else:
                print(f"Failed to generate embedding for chunk {chunk.id}")
        
        return results
    
    def get_embedding_dimensions(self) -> int:
        """
        Get the expected dimensions of the embeddings from this model
        """
        # This is a placeholder - in a real implementation, you would determine this
        # based on the actual embedding model being used
        # GEMINI embedding-001 model typically produces 768-dimensional vectors
        return 768


class EmbeddingCache:
    """
    Simple cache for embeddings to avoid regenerating them
    """
    
    def __init__(self):
        self.cache = {}
    
    def get(self, text_hash: str) -> Optional[List[float]]:
        """
        Get embedding from cache by text hash
        """
        return self.cache.get(text_hash)
    
    def set(self, text_hash: str, embedding: List[float]):
        """
        Store embedding in cache with text hash
        """
        self.cache[text_hash] = embedding
    
    def has(self, text_hash: str) -> bool:
        """
        Check if embedding exists in cache
        """
        return text_hash in self.cache


def hash_text(text: str) -> str:
    """
    Create a hash of the text for caching purposes
    """
    import hashlib
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


async def generate_embeddings_for_content(chunks: List[ChunkedContent], 
                                        model_name: str = "embedding-001") -> List[Dict[str, Any]]:
    """
    Convenience function to generate embeddings for content chunks without creating a full generator instance
    """
    generator = GeminiEmbeddingGenerator(model_name)
    return await generator.generate_embeddings_from_chunks(chunks)


# Example usage
async def example_usage():
    """
    Example of how to use the embedding generator
    """
    # Create sample chunked content
    from .content_chunker import ChunkedContent
    
    sample_chunks = [
        ChunkedContent(
            id="sample_chunk_1",
            module_id="module1",
            chapter_id="chapter1", 
            section_id="section1",
            hierarchy_path="module1/chapter1/section1",
            content_type="text",
            content="This is a sample sentence for embedding generation."
        ),
        ChunkedContent(
            id="sample_chunk_2",
            module_id="module1",
            chapter_id="chapter1",
            section_id="section1", 
            hierarchy_path="module1/chapter1/section1",
            content_type="text",
            content="This is another sample sentence for embedding generation."
        )
    ]
    
    # Generate embeddings
    generator = GeminiEmbeddingGenerator()
    results = await generator.generate_embeddings_from_chunks(sample_chunks)
    
    for result in results:
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Embedding length: {len(result['embedding'])}")
        print(f"Sample embedding values: {result['embedding'][:5]}...")  # First 5 values
        print()


if __name__ == "__main__":
    # Run example (this requires a valid GEMINI API key)
    # asyncio.run(example_usage())
    print("GeminiEmbeddingGenerator module ready. Use with a valid GEMINI API key.")