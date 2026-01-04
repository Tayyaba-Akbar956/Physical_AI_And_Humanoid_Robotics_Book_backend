import asyncio
from typing import List, Dict, Any, Optional
from uuid import uuid4
from .db.qdrant_client import QdrantManager
from .embedding_generator import hash_text


class QdrantUploader:
    """
    Class to upload embeddings to Qdrant vector database
    """
    
    def __init__(self, collection_name: str = "textbook_content_embeddings"):
        """
        Initialize the Qdrant uploader
        
        Args:
            collection_name: The name of the Qdrant collection to upload to
        """
        self.qdrant_manager = QdrantManager(collection_name)
        self.collection_name = collection_name
    
    async def upload_embeddings(self, embedding_results: List[Dict[str, Any]]) -> bool:
        """
        Upload embedding results to Qdrant
        
        Args:
            embedding_results: List of embedding results from the embedding generator
                              Each should have: id, chunk_id, content, embedding, metadata, etc.
            
        Returns:
            True if upload was successful, False otherwise
        """
        try:
            # Prepare points for Qdrant
            points = []
            for result in embedding_results:
                # Create a unique ID for this point in Qdrant
                point_id = str(uuid4())
                
                # Prepare the payload with all relevant information
                payload = {
                    "chunk_id": result.get("chunk_id"),
                    "module_id": result.get("module_id"),
                    "chapter_id": result.get("chapter_id"),
                    "section_id": result.get("section_id"),
                    "hierarchy_path": result.get("hierarchy_path"),
                    "content_type": result.get("content_type"),
                    "content": result.get("content"),
                    "metadata": result.get("metadata", {})
                }
                
                # Create the point structure for Qdrant
                point = {
                    "id": point_id,
                    "vector": result.get("embedding"),
                    "payload": payload
                }
                
                points.append(point)
            
            # Upload points to Qdrant
            success = self.qdrant_manager.add_embeddings(points)
            
            if success:
                print(f"Successfully uploaded {len(points)} embeddings to Qdrant")
                return True
            else:
                print("Failed to upload embeddings to Qdrant")
                return False
                
        except Exception as e:
            print(f"Error uploading embeddings to Qdrant: {e}")
            return False
    
    async def search_similar_content(self, query_embedding: List[float], top_k: int = 5, 
                                   filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Search for similar content in Qdrant
        
        Args:
            query_embedding: The embedding vector to search for similar content
            top_k: Number of similar items to return
            filters: Optional filters for module, chapter, section, etc.
            
        Returns:
            List of similar content results
        """
        try:
            results = self.qdrant_manager.search_similar(
                query_vector=query_embedding,
                top_k=top_k,
                filters=filters
            )
            
            return results
        except Exception as e:
            print(f"Error searching for similar content in Qdrant: {e}")
            return []
    
    async def delete_embedding(self, embedding_id: str) -> bool:
        """
        Delete a specific embedding from Qdrant
        
        Args:
            embedding_id: The ID of the embedding to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            success = self.qdrant_manager.delete_embedding(embedding_id)
            return success
        except Exception as e:
            print(f"Error deleting embedding from Qdrant: {e}")
            return False
    
    async def update_embedding(self, embedding_id: str, new_embedding: List[float], 
                             new_payload: Dict[str, Any]) -> bool:
        """
        Update a specific embedding in Qdrant
        
        Args:
            embedding_id: The ID of the embedding to update
            new_embedding: The new embedding vector
            new_payload: The new payload data
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            # For updating in Qdrant, we need to delete and re-add since Qdrant doesn't support direct updates
            delete_success = await self.delete_embedding(embedding_id)
            if not delete_success:
                return False
            
            # Prepare the new point
            point = {
                "id": embedding_id,
                "vector": new_embedding,
                "payload": new_payload
            }
            
            # Add the updated point
            success = self.qdrant_manager.add_embeddings([point])
            return success
        except Exception as e:
            print(f"Error updating embedding in Qdrant: {e}")
            return False
    
    async def get_collection_stats(self):
        """
        Get statistics about the Qdrant collection
        """
        try:
            info = self.qdrant_manager.get_collection_info()
            return info
        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return None


class ContentUploadManager:
    """
    Higher-level manager for handling the complete content upload process
    from embedding generation to Qdrant upload
    """
    
    def __init__(self, collection_name: str = "textbook_content_embeddings"):
        self.qdrant_uploader = QdrantUploader(collection_name)
    
    async def upload_content_chunks(self, embedding_results: List[Dict[str, Any]]) -> bool:
        """
        Upload embedding results for content chunks to Qdrant
        
        Args:
            embedding_results: List of embedding results from the embedding generator
            
        Returns:
            True if upload was successful, False otherwise
        """
        return await self.qdrant_uploader.upload_embeddings(embedding_results)
    
    async def upload_single_chunk(self, chunk_id: str, embedding: List[float], 
                                content_data: Dict[str, Any]) -> bool:
        """
        Upload a single content chunk with its embedding
        
        Args:
            chunk_id: The ID of the chunk to upload
            embedding: The embedding vector
            content_data: Dictionary with module_id, chapter_id, section_id, hierarchy_path, content, etc.
            
        Returns:
            True if upload was successful, False otherwise
        """
        try:
            # Prepare the payload
            payload = {
                "chunk_id": chunk_id,
                "module_id": content_data.get("module_id"),
                "chapter_id": content_data.get("chapter_id"),
                "section_id": content_data.get("section_id"),
                "hierarchy_path": content_data.get("hierarchy_path"),
                "content_type": content_data.get("content_type"),
                "content": content_data.get("content"),
                "metadata": content_data.get("metadata", {})
            }
            
            # Create the point structure for Qdrant
            point = {
                "id": str(uuid4()),  # Generate a unique ID
                "vector": embedding,
                "payload": payload
            }
            
            # Upload the point to Qdrant
            success = self.qdrant_uploader.qdrant_manager.add_embeddings([point])
            
            if success:
                print(f"Successfully uploaded chunk {chunk_id} to Qdrant")
                return True
            else:
                print(f"Failed to upload chunk {chunk_id} to Qdrant")
                return False
        except Exception as e:
            print(f"Error uploading single chunk to Qdrant: {e}")
            return False


async def upload_embeddings_to_qdrant(embedding_results: List[Dict[str, Any]], 
                                    collection_name: str = "textbook_content_embeddings") -> bool:
    """
    Convenience function to upload embeddings to Qdrant without creating a full uploader instance
    """
    uploader = ContentUploadManager(collection_name)
    return await uploader.upload_content_chunks(embedding_results)


# Example usage
async def example_usage():
    """
    Example of how to use the Qdrant uploader
    """
    # Create sample embedding results
    sample_results = [
        {
            "chunk_id": "sample_chunk_1",
            "module_id": "module1",
            "chapter_id": "chapter1",
            "section_id": "section1",
            "hierarchy_path": "module1/chapter1/section1",
            "content_type": "text",
            "content": "Sample content for embedding",
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5] * 150,  # Simulated embedding
            "metadata": {"word_count": 5, "source": "example"}
        }
    ]
    
    # Upload to Qdrant
    uploader = ContentUploadManager()
    success = await uploader.upload_content_chunks(sample_results)
    
    if success:
        print("Embeddings uploaded successfully!")
        
        # Test search functionality
        sample_query = [0.1, 0.2, 0.3, 0.4, 0.5] * 150  # Same dimension as embeddings
        search_results = await uploader.qdrant_uploader.search_similar_content(
            query_embedding=sample_query,
            top_k=5
        )
        
        print(f"Search returned {len(search_results)} results")
    else:
        print("Embedding upload failed!")


if __name__ == "__main__":
    # Run example
    # asyncio.run(example_usage())
    print("QdrantUploader module ready. Use with a valid Qdrant configuration.")