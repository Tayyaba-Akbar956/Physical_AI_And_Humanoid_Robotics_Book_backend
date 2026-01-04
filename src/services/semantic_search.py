import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from ..db.qdrant_client import QdrantManager
from ..embedding_generator import GeminiEmbeddingGenerator, hash_text
from .rag_agent import RAGAgentService


# Load environment variables
load_dotenv()


class SemanticSearchService:
    """
    Service for performing semantic search on textbook content using GEMINI embeddings
    Integrates with Qdrant for efficient vector search
    """
    
    def __init__(self, collection_name: str = "textbook_content_embeddings"):
        """
        Initialize the semantic search service
        
        Args:
            collection_name: The name of the Qdrant collection to search
        """
        self.qdrant_manager = QdrantManager(collection_name)
        self.collection_name = collection_name
        # We'll use the same embedding dimensions as in the embedding generator
        self.embedding_generator = None  # Will initialize when needed
        self._rag_agent_service = None
    
    @property
    def rag_agent_service(self):
        """
        Lazy initialization of RAG agent service to avoid circular imports
        """
        if self._rag_agent_service is None:
            from .rag_agent import RAGAgentService
            self._rag_agent_service = RAGAgentService(self.collection_name)
        return self._rag_agent_service
    
    def _get_embedding_generator(self):
        """
        Lazy initialization of embedding generator to avoid circular imports
        """
        if self.embedding_generator is None:
            self.embedding_generator = GeminiEmbeddingGenerator()
        return self.embedding_generator

    async def search(self, query: str, top_k: int = 5,
                    filters: Optional[Dict[str, Any]] = None,
                    include_module_content: bool = True,
                    min_score: float = 0.3,
                    return_embeddings: bool = False) -> List[Dict[str, Any]]:
        """
        Perform semantic search on the textbook content with enhanced functionality

        Args:
            query: The search query
            top_k: Number of results to return
            filters: Optional filters (e.g., by module, chapter)
            include_module_content: Whether to include the full content in results
            min_score: Minimum similarity score for results to be included
            return_embeddings: Whether to include the embedding vectors in results

        Returns:
            List of search results with metadata
        """
        try:
            # Generate embedding for the query using GEMINI
            embedding_gen = self._get_embedding_generator()
            query_embedding = await embedding_gen.generate_embedding(query)

            if not query_embedding:
                print("Failed to generate embedding for query")
                return []

            # Perform semantic search using Qdrant
            search_results = await self.qdrant_manager.search_similar(
                query_vector=query_embedding,
                top_k=top_k,
                filters=filters
            )

            # Filter results by minimum score
            filtered_results = [r for r in search_results if r["score"] >= min_score]

            # Format and enrich results
            formatted_results = []
            for result in filtered_results:
                payload = result["payload"]

                formatted_result = {
                    "id": result["id"],
                    "score": result["score"],
                    "module_id": payload.get("module_id", ""),
                    "chapter_id": payload.get("chapter_id", ""),
                    "section_id": payload.get("section_id", ""),
                    "hierarchy_path": payload.get("hierarchy_path", ""),
                    "content_type": payload.get("content_type", ""),
                    "metadata": payload.get("metadata", {}),
                    "similarity_score": result["score"],  # Same as score for clarity
                    "relative_relevance": result["score"]  # For ranking purposes
                }

                # Conditionally include content based on parameter
                if include_module_content:
                    formatted_result["content"] = payload.get("content", "")

                # Conditionally include embeddings based on parameter
                if return_embeddings:
                    formatted_result["embedding"] = payload.get("vector", [])

                formatted_results.append(formatted_result)

            # Sort by score (descending) in case filtering changed the order
            formatted_results.sort(key=lambda x: x["score"], reverse=True)

            return formatted_results

        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []

    async def search_with_query_expansion(self, query: str, top_k: int = 5,
                                        filters: Optional[Dict[str, Any]] = None,
                                        include_module_content: bool = True) -> List[Dict[str, Any]]:
        """
        Perform semantic search with automatic query expansion to improve results

        Args:
            query: The search query
            top_k: Number of results to return
            filters: Optional filters (e.g., by module, chapter)
            include_module_content: Whether to include the full content in results

        Returns:
            List of search results with metadata
        """
        try:
            # Generate original embedding for the query
            embedding_gen = self._get_embedding_generator()
            original_embedding = await embedding_gen.generate_embedding(query)

            if not original_embedding:
                print("Failed to generate embedding for original query")
                return []

            # Try to create expanded queries (this is a simple approach)
            # In a more sophisticated system, you might use the LLM to generate related concepts
            expanded_queries = [query]

            # Simple expansion by adding related terms for robotics/ai concepts
            if "robot" in query.lower():
                expanded_queries.append(query + " robotic system control")
            elif "ai" in query.lower() or "artificial intelligence" in query.lower():
                expanded_queries.append(query + " machine learning algorithm")
            elif "ros" in query.lower():
                expanded_queries.append(query + " ros2 framework communication")

            # Get embeddings for all queries
            all_embeddings = []
            for q in expanded_queries:
                emb = await embedding_gen.generate_embedding(q)
                if emb:
                    all_embeddings.append(emb)

            # Average the embeddings (pure Python implementation to avoid numpy)
            if all_embeddings:
                # Get the number of vectors and the dimension of each vector
                num_vectors = len(all_embeddings)
                vector_dim = len(all_embeddings[0])
                
                # Create a zero-initialized vector for the average
                avg_query_embedding = [0.0] * vector_dim
                
                # Sum all vectors element-wise
                for emb in all_embeddings:
                    for i in range(vector_dim):
                        avg_query_embedding[i] += emb[i]
                
                # Divide by the number of vectors
                for i in range(vector_dim):
                    avg_query_embedding[i] /= num_vectors
            else:
                avg_query_embedding = original_embedding

            # Perform semantic search using the averaged embedding
            search_results = await self.qdrant_manager.search_similar(
                query_vector=avg_query_embedding,
                top_k=top_k,
                filters=filters
            )

            # Apply minimum score filtering and format results
            formatted_results = []
            for result in search_results:
                if result["score"] >= 0.3:  # Minimum relevance threshold
                    payload = result["payload"]

                    formatted_result = {
                        "id": result["id"],
                        "score": result["score"],
                        "module_id": payload.get("module_id", ""),
                        "chapter_id": payload.get("chapter_id", ""),
                        "section_id": payload.get("section_id", ""),
                        "hierarchy_path": payload.get("hierarchy_path", ""),
                        "content_type": payload.get("content_type", ""),
                        "metadata": payload.get("metadata", {}),
                        "similarity_score": result["score"],
                        "query_expansion_used": len(expanded_queries) > 1,
                        "original_query": query
                    }

                    if include_module_content:
                        formatted_result["content"] = payload.get("content", "")

                    formatted_results.append(formatted_result)

            # Sort by score
            formatted_results.sort(key=lambda x: x["score"], reverse=True)

            return formatted_results

        except Exception as e:
            print(f"Error in semantic search with query expansion: {e}")
            # Fallback to regular search
            return await self.search(query, top_k, filters, include_module_content)

    async def multi_vector_search(self, query: str, top_k: int = 5,
                                filters: Optional[Dict[str, Any]] = None,
                                include_module_content: bool = True,
                                weight_content: float = 0.7,
                                weight_metadata: float = 0.3) -> List[Dict[str, Any]]:
        """
        Perform search using both content and metadata vectors for better results

        Args:
            query: The search query
            top_k: Number of results to return
            filters: Optional filters (e.g., by module, chapter)
            include_module_content: Whether to include the full content in results
            weight_content: Weight for content vector matching
            weight_metadata: Weight for metadata vector matching

        Returns:
            List of search results with metadata
        """
        try:
            embedding_gen = self._get_embedding_generator()
            query_embedding = await embedding_gen.generate_embedding(query)

            if not query_embedding:
                print("Failed to generate embedding for query")
                return []

            # Perform semantic search using Qdrant
            search_results = await self.qdrant_manager.search_similar(
                query_vector=query_embedding,
                top_k=top_k*2,  # Get more results to have options for hybrid ranking
                filters=filters
            )

            # For this implementation, we'll use the basic content search results
            # but in a more advanced system, we would implement true multi-vector search
            formatted_results = []
            for result in search_results:
                if result["score"] >= 0.3:  # Minimum relevance threshold
                    payload = result["payload"]

                    formatted_result = {
                        "id": result["id"],
                        "score": result["score"],
                        "module_id": payload.get("module_id", ""),
                        "chapter_id": payload.get("chapter_id", ""),
                        "section_id": payload.get("section_id", ""),
                        "hierarchy_path": payload.get("hierarchy_path", ""),
                        "content_type": payload.get("content_type", ""),
                        "metadata": payload.get("metadata", {}),
                        "similarity_score": result["score"],
                        "content_relevance": result["score"],  # Using same score as proxy
                        "metadata_relevance": result["score"] * 0.8,  # Using a derived score as proxy
                        "combined_score": result["score"]  # Placeholder for combined score
                    }

                    if include_module_content:
                        formatted_result["content"] = payload.get("content", "")

                    formatted_results.append(formatted_result)

            # Sort by score
            formatted_results.sort(key=lambda x: x["score"], reverse=True)

            return formatted_results[:top_k]

        except Exception as e:
            print(f"Error in multi-vector search: {e}")
            # Fallback to regular search
            return await self.search(query, top_k, filters, include_module_content)
    
    async def search_in_module(self, query: str, module_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search specifically within a given module
        
        Args:
            query: The search query
            module_id: The module to search within
            top_k: Number of results to return
            
        Returns:
            List of search results from the specified module
        """
        filters = {"module_id": module_id}
        return await self.search(query, top_k=top_k, filters=filters)
    
    async def search_in_chapter(self, query: str, module_id: str, chapter_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search specifically within a given chapter
        
        Args:
            query: The search query
            module_id: The module containing the chapter
            chapter_id: The chapter to search within
            top_k: Number of results to return
            
        Returns:
            List of search results from the specified chapter
        """
        filters = {
            "module_id": module_id,
            "chapter_id": chapter_id
        }
        return await self.search(query, top_k=top_k, filters=filters)
    
    async def get_content_by_ids(self, content_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve specific content by its IDs
        
        Args:
            content_ids: List of content IDs to retrieve
            
        Returns:
            List of content items with specified IDs
        """
        results = []
        
        for content_id in content_ids:
            embedding_data = await self.qdrant_manager.get_embedding_by_id(content_id)
            if embedding_data:
                payload = embedding_data["payload"]
                results.append({
                    "id": embedding_data["id"],
                    "module_id": payload.get("module_id", ""),
                    "chapter_id": payload.get("chapter_id", ""),
                    "section_id": payload.get("section_id", ""),
                    "hierarchy_path": payload.get("hierarchy_path", ""),
                    "content": payload.get("content", ""),
                    "content_type": payload.get("content_type", ""),
                    "metadata": payload.get("metadata", {}),
                    "vector": embedding_data["vector"]
                })
        
        return results
    
    async def find_related_content(self, content_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find content related to a specific piece of content
        
        Args:
            content_id: ID of the content to find related items for
            top_k: Number of related items to return
            
        Returns:
            List of related content items
        """
        try:
            # Get the content and its embedding
            content_data = await self.qdrant_manager.get_embedding_by_id(content_id)
            if not content_data:
                return []
            
            # Perform semantic search using the content's own embedding
            search_results = await self.qdrant_manager.search_similar(
                query_vector=content_data["vector"],
                top_k=top_k + 1  # +1 to exclude the original content
            )
            
            # Filter out the original content and format results
            related_results = []
            for result in search_results:
                if result["id"] != content_id:  # Exclude the original content
                    payload = result["payload"]
                    related_results.append({
                        "id": result["id"],
                        "score": result["score"],
                        "module_id": payload.get("module_id", ""),
                        "chapter_id": payload.get("chapter_id", ""),
                        "section_id": payload.get("section_id", ""),
                        "hierarchy_path": payload.get("hierarchy_path", ""),
                        "content_type": payload.get("content_type", ""),
                        "content": payload.get("content", ""),
                        "metadata": payload.get("metadata", {})
                    })
            
            return related_results[:top_k]  # Ensure we return only top_k results
            
        except Exception as e:
            print(f"Error finding related content: {e}")
            return []
    
    async def search_with_module_prioritization(self, query: str, current_module: Optional[str] = None,
                                              top_k: int = 5, prioritize_current_module: bool = True) -> List[Dict[str, Any]]:
        """
        Search with prioritization of results from the current module

        Args:
            query: The search query
            current_module: The module the user is currently viewing
            top_k: Number of results to return
            prioritize_current_module: Whether to prioritize content from the current module

        Returns:
            List of search results, potentially with current module content prioritized
        """
        if not prioritize_current_module or not current_module:
            # If not prioritizing current module, do a regular search
            return await self.search(query, top_k)

        try:
            # First, search within the current module
            current_module_results = await self.search_in_module(query, current_module, top_k)

            # If we have enough high-quality results from the current module, return them
            if len(current_module_results) >= top_k:
                return current_module_results[:top_k]

            # Otherwise, supplement with results from other modules
            additional_needed = top_k - len(current_module_results)

            # Search across all modules (excluding the current one to avoid duplicates)
            all_results = await self.search(query, top_k * 2)  # Get more to filter

            # Filter out results that are already in current module results
            filtered_results = [r for r in all_results if r["module_id"] != current_module]

            # Combine current module results with other module results
            combined_results = current_module_results + filtered_results[:additional_needed]

            # Sort by score to maintain relevance while prioritizing current module
            combined_results.sort(key=lambda x: x["score"], reverse=True)

            return combined_results[:top_k]

        except Exception as e:
            print(f"Error in search with module prioritization: {e}")
            # Fallback to regular search
            return await self.search(query, top_k)

    async def get_cross_module_references(self, query: str, current_module: str,
                                       top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Find related content in other modules that connect to the current query

        Args:
            query: The search query
            current_module: The current module to exclude from results
            top_k: Number of cross-reference results to return

        Returns:
            List of content from other modules that relates to the query
        """
        try:
            # Search across all modules except the current one
            all_results = await self.search(
                query,
                top_k=top_k * 2,  # Get more to ensure we have options after filtering
                filters={"module_id": {"$ne": current_module}}
            )

            # Return only top_k results that are from different modules
            cross_refs = [r for r in all_results if r["module_id"] != current_module]

            return cross_refs[:top_k]

        except Exception as e:
            print(f"Error finding cross module references: {e}")
            return []

    async def get_module_content_relevance(self, query: str, modules: List[str]) -> Dict[str, float]:
        """
        Get relevance scores for a query across different modules

        Args:
            query: The search query
            modules: List of module IDs to check

        Returns:
            Dictionary mapping module IDs to relevance scores
        """
        try:
            # Generate embedding for the query
            embedding_gen = self._get_embedding_generator()
            query_embedding = await embedding_gen.generate_embedding(query)

            if not query_embedding:
                print("Failed to generate embedding for query")
                return {}

            relevance_scores = {}

            for module_id in modules:
                # Search within this specific module
                results = await self.search_in_module(query, module_id, top_k=1)

                if results:
                    relevance_scores[module_id] = results[0]["score"]
                else:
                    relevance_scores[module_id] = 0.0  # No relevant content found

            return relevance_scores

        except Exception as e:
            print(f"Error calculating module content relevance: {e}")
            return {}

    async def get_content_navigation_suggestions(self, query: str, current_module: str,
                                               top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Get navigation suggestions to other modules that might have relevant content

        Args:
            query: The search query
            current_module: The current module
            top_k: Number of navigation suggestions to return

        Returns:
            List of navigation suggestions with module info
        """
        try:
            # Find related content in other modules
            cross_refs = await self.get_cross_module_references(query, current_module, top_k)

            suggestions = []
            for ref in cross_refs:
                suggestion = {
                    "target_module": ref["module_id"],
                    "target_chapter": ref["chapter_id"],
                    "target_section": ref["section_id"],
                    "content_preview": ref["content"][:150] + "..." if len(ref["content"]) > 150 else ref["content"],
                    "relevance_score": ref["score"],
                    "navigation_hint": f"See {ref['module_id']}, Chapter {ref['chapter_id']} for more details"
                }
                suggestions.append(suggestion)

            return suggestions

        except Exception as e:
            print(f"Error getting content navigation suggestions: {e}")
            return []
    
    async def get_collection_stats(self):
        """
        Get statistics about the searchable content collection
        """
        try:
            return await self.qdrant_manager.get_collection_info()
        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return None


class ModuleAwareSearchService(SemanticSearchService):
    """
    Enhanced search service with additional module-aware capabilities
    """
    
    async def search_with_module_context(self, query: str, current_module: str, 
                                       include_related_modules: bool = True,
                                       top_k: int = 5) -> Dict[str, Any]:
        """
        Search with awareness of the current module context
        
        Args:
            query: The search query
            current_module: The module the user is currently viewing
            include_related_modules: Whether to suggest content from other modules
            top_k: Number of results to return
            
        Returns:
            Dictionary with results from current module and potentially related modules
        """
        # Search within current module
        current_results = await self.search_in_module(query, current_module, top_k)
        
        response = {
            "current_module_results": current_results,
            "from_current_module": True,
            "query": query,
            "current_module": current_module
        }
        
        if include_related_modules and len(current_results) < top_k:
            # Search across all modules to find additional relevant content
            all_results = await self.search(query, top_k * 2)
            # Filter out already included results
            other_module_results = [
                r for r in all_results 
                if r["module_id"] != current_module and 
                not any(r["id"] == cr["id"] for cr in current_results)
            ]
            response["other_module_results"] = other_module_results[:top_k - len(current_results)]
        else:
            response["other_module_results"] = []
        
        return response


async def create_semantic_search_service(collection_name: str = "textbook_content_embeddings") -> SemanticSearchService:
    """
    Convenience function to create a semantic search service instance
    """
    return SemanticSearchService(collection_name)


if __name__ == "__main__":
    import asyncio
    
    async def example_usage():
        # Create search service
        search_service = SemanticSearchService()
        
        # Sample search
        query = "communication between ROS 2 nodes"
        results = await search_service.search(query, top_k=3)
        
        print(f"Search results for: '{query}'")
        for i, result in enumerate(results, 1):
            print(f"Result {i}:")
            print(f"  Module: {result['module_id']}")
            print(f"  Chapter: {result['chapter_id']}")
            print(f"  Section: {result['section_id']}")
            print(f"  Score: {result['score']:.3f}")
            print(f"  Content preview: {result['content'][:100]}...")
            print()
    
    # Run example
    # asyncio.run(example_usage())
    print("SemanticSearchService module ready. Use with a valid Qdrant configuration.")