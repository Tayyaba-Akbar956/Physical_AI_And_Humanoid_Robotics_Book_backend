import os
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from openai import AsyncOpenAI
from ..db.qdrant_client import QdrantManager
from ..db.connection import get_db
from ..models.textbook_content import TextbookContentDB
from sqlalchemy.orm import Session
import logging
import asyncio
from ..embedding_generator import GeminiEmbeddingGenerator


# Load environment variables
load_dotenv()

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class RAGAgentService:
    """
    RAG (Retrieval-Augmented Generation) Agent Service
    This service integrates textbook content retrieval with GEMINI-based response generation
    """
    
    def __init__(self, collection_name: str = "textbook_content_embeddings"):
        """
        Initialize the RAG Agent Service
        """
        from ..db.qdrant_client import get_qdrant_manager
        self.qdrant_manager = get_qdrant_manager()
        self.model = "gemini-2.0-flash"
        self._client = None

    @property
    def client(self):
        """Lazy initialization of the OpenAI client"""
        if self._client is None:
            if not GEMINI_API_KEY:
                # Log warning but don't crash startup
                import logging
                logger = logging.getLogger("rag_chatbot")
                logger.warning("GEMINI_API_KEY is not set in environment variables. Client will be None.")
                return None
            
            try:
                self._client = AsyncOpenAI(
                    api_key=GEMINI_API_KEY,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                )
            except Exception as e:
                import logging
                logger = logging.getLogger("rag_chatbot")
                logger.error(f"Failed to initialize OpenAI client: {e}")
                return None
        return self._client
        
    async def get_relevant_content(self, query: str, top_k: int = 5, module_filter: Optional[str] = None,
                            prioritize_current_module: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve relevant content from the textbook using semantic search

        Args:
            query: The user's query
            top_k: Number of relevant chunks to retrieve
            module_filter: Optional filter to restrict search to a specific module
            prioritize_current_module: Whether to prioritize content from the current module

        Returns:
            List of relevant content chunks with metadata
        """
        logger = logging.getLogger("rag_chatbot")
        try:
            # Import the semantic search service for enhanced search capabilities
            from .semantic_search import SemanticSearchService
            search_service = SemanticSearchService(self.qdrant_manager.collection_name)

            logger.debug(f"Retrieving relevant content for query: '{query[:50]}...', top_k: {top_k}, module_filter: {module_filter}")

            # Use the search service with module prioritization if specified
            search_results = await self._search_with_module_awareness(
                search_service,
                query,
                top_k,
                module_filter,
                prioritize_current_module
            )

            logger.info(f"Retrieved {len(search_results)} relevant content chunks for query: '{query[:50]}...'")

            return search_results

        except Exception as e:
            logger.error(f"Error retrieving relevant content for query '{query[:50]}...': {e}")
            return []

    async def _search_with_module_awareness(self, search_service: 'SemanticSearchService',
                                    query: str, top_k: int, module_filter: Optional[str],
                                    prioritize_current_module: bool) -> List[Dict[str, Any]]:
        """
        Helper method to perform search with module awareness

        Args:
            search_service: The semantic search service to use
            query: The search query
            top_k: Number of results to return
            module_filter: Module to filter by
            prioritize_current_module: Whether to prioritize the current module

        Returns:
            List of relevant content chunks with metadata
        """
        try:
            # If we want to prioritize the current module, use the specialized search method
            if module_filter and prioritize_current_module:
                search_results = await search_service.search_with_module_prioritization(
                    query=query,
                    current_module=module_filter,
                    top_k=top_k,
                    prioritize_current_module=prioritize_current_module
                )
            else:
                # For regular search or when not prioritizing current module
                filters = {}
                if module_filter:
                    filters["module_id"] = module_filter

                search_results = await search_service.search(
                    query=query,
                    top_k=top_k,
                    filters=filters
                )

            # Format results to match the expected structure
            formatted_results = []
            for result in search_results:
                formatted_results.append({
                    "id": result["id"],
                    "score": result["score"],
                    "module_id": result.get("module_id"),
                    "chapter_id": result.get("chapter_id"),
                    "section_id": result.get("section_id"),
                    "content": result.get("content", ""),
                    "hierarchy_path": result.get("hierarchy_path"),
                    "content_type": result.get("content_type"),
                    "metadata": result.get("metadata", {}),
                    "is_from_current_module": result.get("module_id") == module_filter if module_filter else False
                })

            return formatted_results
        except Exception as e:
            print(f"Error in module-aware search: {e}")
            # Fallback to basic search
            return await self._basic_search_fallback(query, top_k, module_filter)

    async def _basic_search_fallback(self, query: str, top_k: int, module_filter: Optional[str]) -> List[Dict[str, Any]]:
        """
        Basic fallback search method if the enhanced search fails

        Args:
            query: The user's query
            top_k: Number of relevant chunks to retrieve
            module_filter: Optional filter to restrict search to a specific module

        Returns:
            List of relevant content chunks with metadata
        """
        try:
            # Search for relevant content in Qdrant
            filters = {}
            if module_filter:
                filters["module_id"] = module_filter

            query_embedding = await self._generate_query_embedding(query)
            search_results = await self.qdrant_manager.search_similar(
                query_vector=query_embedding,
                top_k=top_k,
                filters=filters
            )

            relevant_content = []
            for result in search_results:
                payload = result["payload"]
                relevant_content.append({
                    "id": result["id"],
                    "score": result["score"],
                    "module_id": payload.get("module_id"),
                    "chapter_id": payload.get("chapter_id"),
                    "section_id": payload.get("section_id"),
                    "content": payload.get("content"),
                    "hierarchy_path": payload.get("hierarchy_path"),
                    "content_type": payload.get("content_type"),
                    "metadata": payload.get("metadata", {}),
                    "is_from_current_module": payload.get("module_id") == module_filter if module_filter else False
                })

            return relevant_content
        except Exception as e:
            print(f"Error in basic search fallback: {e}")
            return []

    async def _generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for the query to use in vector search
        """
        try:
            generator = GeminiEmbeddingGenerator()
            embedding = await generator.generate_embedding(query)
            if embedding:
                return embedding
            
            # Fallback to random if failed, but log it
            print(f"Warning: Failed to generate real embedding for query '{query[:30]}...', using random fallback")
            import random
            return [random.random() for _ in range(768)]
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            import random
            return [random.random() for _ in range(768)]

    async def integrate_with_semantic_search(self, query: str, top_k: int = 5, module_filter: Optional[str] = None,
                                     min_similarity_score: float = 0.3) -> List[Dict[str, Any]]:
        """
        Enhanced method to integrate with the semantic search service using GEMINI embeddings

        Args:
            query: The user's query
            top_k: Number of relevant chunks to retrieve
            module_filter: Optional filter to restrict search to a specific module
            min_similarity_score: Minimum similarity score for content to be considered relevant

        Returns:
            List of relevant content chunks with metadata from integrated semantic search
        """
        try:
            # Import the semantic search service to ensure integration
            from .semantic_search import SemanticSearchService

            # Create a semantic search instance
            search_service = SemanticSearchService(self.qdrant_manager.collection_name)

            # Prepare filters
            filters = {}
            if module_filter:
                filters["module_id"] = module_filter

            # Perform semantic search using the dedicated service
            search_results = await search_service.search(
                query=query,
                top_k=top_k,
                filters=filters,
                min_score=min_similarity_score
            )

            # Format results to match the expected structure
            integrated_content = []
            for result in search_results:
                # Make sure the result meets our minimum similarity requirement
                if result["score"] >= min_similarity_score:
                    content_item = {
                        "id": result["id"],
                        "score": result["score"],
                        "module_id": result.get("module_id"),
                        "chapter_id": result.get("chapter_id"),
                        "section_id": result.get("section_id"),
                        "content": result.get("content", ""),
                        "hierarchy_path": result.get("hierarchy_path"),
                        "content_type": result.get("content_type"),
                        "metadata": result.get("metadata", {}),
                        "similarity_score": result["score"],
                        "relative_relevance": result.get("relative_relevance", result["score"])
                    }
                    integrated_content.append(content_item)

            # Sort by similarity score in descending order
            integrated_content.sort(key=lambda x: x["score"], reverse=True)

            return integrated_content
        except Exception as e:
            print(f"Error integrating with semantic search: {e}")
            # Fallback to the original method
            return self.get_relevant_content(query, top_k, module_filter)
    
    async def generate_response(self, query: str, context: List[Dict[str, Any]],
                         selected_text: Optional[str] = None,
                         module_context: Optional[str] = None,
                         conversation_context: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Generate a context-aware response using GEMINI based on the query, context, and conversation history

        Args:
            query: The user's question
            context: List of relevant content chunks retrieved from textbook
            selected_text: Optional text that was selected by the user (for text selection queries)
            module_context: Current module the user is viewing (for prioritization)
            conversation_context: Previous conversation history for context awareness

        Returns:
            Dictionary containing the response and citations
        """
        logger = logging.getLogger("rag_chatbot")
        try:
            # Log the incoming request
            logger.info(f"Generating response for query: '{query[:50]}...', with {len(context)} context chunks")

            # Build the context for the LLM
            context_text = ""
            citations = []

            for i, chunk in enumerate(context):
                context_text += f"\\n\\nRelevant Content {i+1} (Module: {chunk.get('module_id', 'N/A')}, Chapter: {chunk.get('chapter_id', 'N/A')}):\\n{chunk['content']}\\n"
                citations.append({
                    "module": chunk.get("module_id", ""),
                    "chapter": chunk.get("chapter_id", ""),
                    "section": chunk.get("section_id", ""),
                    "content_type": chunk.get("content_type", ""),
                    "hierarchy_path": chunk.get("hierarchy_path", "")
                })

            # Build conversation context if available
            conversation_history = ""
            if conversation_context:
                conversation_history = "\\nPrevious conversation:\\n"
                # Include more detailed context from the conversation
                for msg in conversation_context[-10:]:  # Use last 10 messages as context
                    # Handle different data structures for conversation contexts
                    if isinstance(msg, dict):
                        sender_type = msg.get("sender_type", msg.get("role", "User"))
                        content = msg.get("content", "")
                        timestamp = msg.get("timestamp", "")

                        # Handle citations depending on format
                        citations_list = msg.get("citations", [])
                        if citations_list:
                            cited_modules = []
                            for cite in citations_list:
                                if isinstance(cite, dict):
                                    module = cite.get("module", cite.get("module_id", "N/A"))
                                    chapter = cite.get("chapter", cite.get("chapter_id", "N/A"))
                                    cited_modules.append(f"Module {module}, Chapter {chapter}")
                            cited_modules_str = f" [Citations: {', '.join(cited_modules)}]" if cited_modules else ""
                        else:
                            cited_modules_str = ""

                        conversation_history += f"{sender_type}: {content}{cited_modules_str}\\n"
                    else:
                        # Handle if msg is a string
                        conversation_history += f"User: {str(msg)}\\n"

            # Analyze conversation context to determine the type of response needed
            response_intent = self._analyze_response_intent(query, conversation_context)

            # Check if this is a follow-up question requiring contextual awareness
            is_follow_up = self._is_follow_up_question(query, conversation_context)

            # Analyze module context prioritization
            current_module_content = [chunk for chunk in context if chunk.get("is_from_current_module", False)]
            other_module_content = [chunk for chunk in context if not chunk.get("is_from_current_module", False)]

            # If there's selected text, prioritize it in the prompt
            if selected_text:
                system_prompt = f"""
                You are an educational assistant helping students with the Physical AI & Humanoid Robotics textbook.
                The student has selected the following text: "{selected_text}"
                They have asked: "{query}"

                Please provide an explanation based on the following textbook content.
                Prioritize explaining the selected text but enrich with related content if helpful.

                Conversation Context ({'Follow-up' if is_follow_up else 'New question'}):
                {conversation_history if conversation_history != '\\nPrevious conversation:\\n' else 'No previous conversation.'}

                Textbook Content:
                {context_text}

                Contextual Intent: {response_intent}

                IMPORTANT:
                - Only use information from the provided textbook content
                - Cite which modules/chapters the information comes from using the format: "According to Module X, Chapter Y..."
                - Prioritize information from the current module ({module_context}) when possible, but cite other modules when relevant
                - If the answer isn't in the textbook, clearly state this
                - Use textbook terminology
                - Keep response between 150-300 words
                - If the question is about something not in the text, politely decline and explain the scope
                - Format your response with clear paragraphs and use bullet points if appropriate
                - If asked about external topics (like exam dates, instructor contact info), politely decline and explain your scope
                - For follow-up questions, maintain continuity with previous answers
                - Acknowledge context when appropriate (e.g., "Building on what we discussed earlier...")
                """
            elif module_context and is_follow_up:
                # For follow-ups in a specific module context
                system_prompt = f"""
                You are an educational assistant helping students with the Physical AI & Humanoid Robotics textbook.
                The student is currently studying Module: {module_context}
                They have asked: "{query}"

                Please provide an answer based on the following textbook content.
                Since this appears to be a follow-up question, maintain continuity with previous responses.

                Conversation Context ({'Follow-up' if is_follow_up else 'New question'}):
                {conversation_history if conversation_history != '\\nPrevious conversation:\\n' else 'No previous conversation.'}

                Textbook Content:
                {context_text}

                Contextual Intent: {response_intent}

                Module Prioritization Info:
                - Content from current module ({module_context}): {len(current_module_content)} chunks
                - Content from other modules: {len(other_module_content)} chunks
                - Prioritize responses using content from current module when possible

                IMPORTANT:
                - Only use information from the provided textbook content
                - Cite which modules/chapters the information comes from using the format: "According to Module X, Chapter Y..."
                - Prioritize information from the current module ({module_context}) when possible, but cite other modules when relevant
                - If the answer isn't in the textbook, clearly state this
                - Use textbook terminology
                - Keep response between 150-300 words
                - If the question is about something not in the text, politely decline and explain the scope
                - Format your response with clear paragraphs and use bullet points if appropriate
                - If asked about external topics (like exam dates, instructor contact info), politely decline and explain your scope
                - For follow-up questions, reference previous topic if relevant: "Regarding the previous discussion about..."
                """
            elif module_context:
                # General question in a specific module context
                system_prompt = f"""
                You are an educational assistant helping students with the Physical AI & Humanoid Robotics textbook.
                The student is currently studying Module: {module_context}
                They have asked: "{query}"

                Please provide an answer based on the following textbook content.
                Prioritize content from the current module ({module_context}) but mention if information comes from other modules.

                Conversation Context ({'Follow-up' if is_follow_up else 'New question'}):
                {conversation_history if conversation_history != '\\nPrevious conversation:\\n' else 'No previous conversation.'}

                Textbook Content:
                {context_text}

                Contextual Intent: {response_intent}

                Module Prioritization Info:
                - Content from current module ({module_context}): {len(current_module_content)} chunks
                - Content from other modules: {len(other_module_content)} chunks
                - Prioritize responses using content from current module when possible

                IMPORTANT:
                - Only use information from the provided textbook content
                - Cite which modules/chapters the information comes from using the format: "According to Module X, Chapter Y..."
                - Prioritize information from the current module ({module_context}) when possible, but cite other modules when relevant
                - If the answer isn't in the textbook, clearly state this
                - Use textbook terminology
                - Keep response between 150-300 words
                - If the question is about something not in the text, politely decline and explain the scope
                - Format your response with clear paragraphs and use bullet points if appropriate
                - If asked about external topics (like exam dates, instructor contact info), politely decline and explain your scope
                - For follow-up questions, maintain continuity with previous exchanges
                - When providing content from other modules, indicate: "This concept is covered in more detail in Module Z..."
                """
            else:
                # General context without specific module
                system_prompt = f"""
                You are an educational assistant helping students with the Physical AI & Humanoid Robotics textbook.
                Student Question: "{query}"

                Please provide an answer based on the following textbook content.

                Conversation Context ({'Follow-up' if is_follow_up else 'New question'}):
                {conversation_history if conversation_history != '\\nPrevious conversation:\\n' else 'No previous conversation.'}

                Textbook Content:
                {context_text}

                Contextual Intent: {response_intent}

                IMPORTANT:
                - Only use information from the provided textbook content
                - Cite which modules/chapters the information comes from using the format: "According to Module X, Chapter Y..."
                - If the answer isn't in the textbook, clearly state this
                - Use textbook terminology
                - Keep response between 150-300 words
                - If the question is about something not in the text, politely decline and explain the scope
                - Format your response with clear paragraphs and use bullet points if appropriate
                - If asked about external topics (like exam dates, instructor contact info), politely decline and explain your scope
                - For follow-up questions, maintain continuity with the conversation
                """

            # Generate response using GEMINI via OpenAI-compatible API
            client = self.client
            if not client:
                return {
                    "response": "I'm sorry, but I'm unable to connect to the AI service (GEMINI_API_KEY may be missing or invalid). Please check the backend configuration.",
                    "citations": [],
                    "error": "OpenAI/Gemini client not initialized"
                }

            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                max_tokens=1000,  # Adjust based on desired response length (approx 300 words)
                temperature=0.2,  # Lower temperature for more factual and consistent responses
                stop=["\\n\\n", "Important:"]
            )

            # Extract the response text
            response_text = response.choices[0].message.content.strip()

            # Apply textbook terminology to the response
            response_text = self._apply_textbook_terminology(response_text, context)

            # Ensure the response meets length requirements (150-300 words)
            response_text = await self._control_response_length(
                response_text,
                system_prompt,
                query,
                min_words=150,
                max_words=300
            )

            # Validate the response quality before returning
            validation_result = self.validate_response_quality(response_text, query, context)

            # Ensure the response is properly cited and factual
            result = {
                "response": response_text,
                "citations": citations,
                "context_used": len(context),
                "model_used": self.model,
                "word_count": len(response_text.split()),
                "quality_validation": validation_result,
                "response_intent": response_intent,
                "is_follow_up": is_follow_up
            }

            return result

        except Exception as e:
            print(f"Error generating response: {e}")
            return {
                "response": "I'm sorry, but I encountered an error processing your request. Please try again.",
                "citations": [],
                "error": str(e),
                "context_used": 0,
                "response_intent": "error",
                "is_follow_up": False
            }

    def _analyze_response_intent(self, query: str, conversation_context: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Analyze the intent behind the user's query considering conversation context

        Args:
            query: The user's current query
            conversation_context: Previous conversation history

        Returns:
            String describing the likely intent
        """
        query_lower = query.lower()

        # Intent classification based on keywords and patterns
        if any(pattern in query_lower for pattern in ["explain", "describe", "what is", "how does", "why is"]):
            return "explanation"
        elif any(pattern in query_lower for pattern in ["summarize", "summary", "overview", "briefly"]):
            return "summary"
        elif any(pattern in query_lower for pattern in ["example", "example of", "show me", "like"]):
            return "example"
        elif any(pattern in query_lower for pattern in ["compare", "difference", "vs", "versus"]):
            return "comparison"
        elif any(pattern in query_lower for pattern in ["definition", "define", "meaning"]):
            return "definition"
        elif any(pattern in query_lower for pattern in ["step", "process", "procedure", "how to"]):
            return "instruction"
        elif any(pattern in query_lower for pattern in ["different", "another way", "alternatively", "else"]):
            return "alternative"
        elif any(pattern in query_lower for pattern in ["repeat", "again", "sorry didn't get", "repeat that"]):
            return "repetition"
        else:
            return "general"

    def detect_module_related_concepts(self, query: str, current_module: str, retrieved_content: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detect module-related concepts in the query and content to enhance module awareness

        Args:
            query: The user's query
            current_module: The module the user is currently viewing
            retrieved_content: Content retrieved from the semantic search

        Returns:
            Dictionary with detected module-related concepts and cross-references
        """
        import re

        # Look for potential cross-module references in the query
        cross_module_indicators = [
            "another module", "other module", "different chapter",
            r"module\s+\w+", r"chapter\s+\w+", "elsewhere in the book"
        ]

        cross_reference_detected = False
        for indicator in cross_module_indicators:
            if re.search(indicator, query, re.IGNORECASE):
                cross_reference_detected = True
                break

        # Analyze retrieved content for multi-module references
        modules_mentioned = set()
        current_module_content = []
        other_module_content = []

        for item in retrieved_content:
            module_id = item.get("module_id", "")

            if module_id:
                modules_mentioned.add(module_id)

                if module_id.lower() == current_module.lower():
                    current_module_content.append(item)
                else:
                    other_module_content.append(item)

        # Check if query specifically asks about a different module
        specific_module_request = None
        module_patterns = [r"module\s+(\w+)", r"(\w+)\s+module"]
        for pattern in module_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                potential_module = match.group(1).lower()
                # In a real implementation, we would validate this against actual modules
                specific_module_request = potential_module
                break

        return {
            "cross_reference_requested": cross_reference_detected,
            "modules_mentioned": list(modules_mentioned),
            "current_module_content_count": len(current_module_content),
            "other_module_content_count": len(other_module_content),
            "specific_module_request": specific_module_request,
            "content_distribution": {
                "current_module": len(current_module_content),
                "other_modules": len(other_module_content)
            }
        }

    def _is_follow_up_question(self, query: str, conversation_context: Optional[List[Dict[str, Any]]] = None) -> bool:
        """
        Determine if the current query is a follow-up to the conversation

        Args:
            query: The user's current query
            conversation_context: Previous conversation history

        Returns:
            Boolean indicating if this is a follow-up question
        """
        if not conversation_context:
            return False

        # Check for follow-up indicators in the query
        query_lower = query.lower()
        follow_up_indicators = [
            "that", "this", "it", "same", "similar", "different", "explain", "describe",
            "what was", "how does", "can you", "more about", "tell me", "earlier",
            "previously", "above", "mentioned", "about that", "about this", "like that"
        ]

        for indicator in follow_up_indicators:
            if indicator in query_lower:
                return True

        # If no clear indicators but conversation exists, it might still be related
        return len(conversation_context) > 0

    def _apply_textbook_terminology(self, response: str, context: List[Dict[str, Any]]) -> str:
        """
        Enhance the response to use appropriate textbook terminology

        Args:
            response: The initial response from the LLM
            context: The context used to generate the response (to extract proper terminology)

        Returns:
            Updated response that uses appropriate textbook terminology
        """
        # Extract terminology from the context
        terminology_set = set()
        for chunk in context:
            content = chunk.get('content', '')
            # Extract noun phrases or technical terms from the content
            # For simplicity, we'll look for capitalized terms and common technical terms
            import re

            # Find potential technical terms (capitalized words, compound terms)
            technical_terms = re.findall(r'\b(?:[A-Z][a-z]*\s*){1,3}[A-Z][a-z]*\b', content)
            terminology_set.update(term.lower() for term in technical_terms)

            # Also extract specific robotics/AI terms
            common_terms = re.findall(r'\b(?:robot|ros|node|service|topic|publisher|subscriber|tf|transform|gazebo|simulation|kinematics|dynamics|inverse|forward|jacobian|trajectory|control|motion|actuator|sensor|lidar|camera|imu|slam|navigation|mapping|localization|path\s+planning|motion\s+planning|computer\s+vision|machine\s+learning|neural\s+network|ai|artificial\s+intelligence|algorithm|protocol|architecture|framework|system|module|chapter|section)\b', content, re.IGNORECASE)
            terminology_set.update(term.lower() for term in common_terms)

        # Normalize the response for terminology matching
        updated_response = response

        # Ensure the response uses proper citations format as mentioned in requirements
        if "according to module" not in response.lower() and "chapter" not in response.lower():
            # Add proper citation format if not already present
            citation_examples = [c for c in context if 'module_id' in c and c['module_id']]
            if citation_examples:
                module_id = citation_examples[0].get('module_id', 'X')
                chapter_id = citation_examples[0].get('chapter_id', 'Y')
                if module_id and chapter_id:
                    # Add a citation reminder to the LLM in subsequent interactions
                    pass  # The LLM should handle this via the system prompt

        # Ensure the response uses technical terminology where appropriate
        for term in terminology_set:
            # Replace generic terms with specific textbook terminology if needed
            # This is a simplified approach - in practice you'd have more sophisticated matching
            pass  # The LLM should already be primed with the terminology via context

        return updated_response

    async def _control_response_length(self, response: str, system_prompt: str, query: str,
                                min_words: int = 150, max_words: int = 300) -> str:
        """
        Control the response length to meet the requirement of 150-300 words

        Args:
            response: The initial response from the LLM
            system_prompt: The original system prompt used to generate the response
            query: The original query
            min_words: Minimum number of words required (default 150)
            max_words: Maximum number of words allowed (default 300)

        Returns:
            Response with appropriate length (between min_words and max_words)
        """
        # Count current words
        current_words = len(response.split())

        # If response is too short, expand it
        if current_words < min_words:
            # Calculate how many more words are needed
            words_needed = min_words - current_words

            # Request additional content using the LLM
            expansion_prompt = (
                f"{system_prompt}\n\n"
                f"The previous response was too brief. The user asked: '{query}'\n"
                f"I need to expand on this to meet the requirement of approximately {min_words} words.\n"
                f"Provide additional relevant information based on the textbook content.\n"
                f"Previous response: {response}\n"
                f"Expanded response with additional detail:"
            )

            try:
                # Generate expansion using GEMINI
                expansion_response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": expansion_prompt}
                    ],
                    max_tokens=500,  # Approximately 150-200 additional words
                    temperature=0.3,
                )

                expansion_text = expansion_response.choices[0].message.content.strip()

                # Combine original response with expansion
                combined_response = f"{response} {expansion_text}"

                # Verify the final length
                final_word_count = len(combined_response.split())
                if final_word_count > max_words:
                    # Trim to max length if exceeded
                    words = combined_response.split()
                    trimmed_words = words[:max_words]
                    combined_response = " ".join(trimmed_words)

                return combined_response

            except Exception as e:
                print(f"Error expanding response: {e}")
                # If expansion fails, return original response
                return response

        # If response is too long, trim it while preserving meaning
        elif current_words > max_words:
            try:
                # Summarize the response to fit within limits
                summarization_prompt = (
                    f"{system_prompt}\n\n"
                    f"The following response is too long ({current_words} words) and needs to be "
                    f"condensed to approximately {max_words} words while preserving key information:\n\n"
                    f"Original response: {response}\n\n"
                    f"Make the response more concise but keep all the essential information "
                    f"and citations. Target: {max_words} words."
                )

                summarization_response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": summarization_prompt}
                    ],
                    max_tokens=max_words * 2,  # Allow enough tokens for restructuring
                    temperature=0.2,
                )

                condensed_response = summarization_response.choices[0].message.content.strip()

                # Verify length after condensation
                condensed_word_count = len(condensed_response.split())
                if condensed_word_count > max_words:
                    # Final fallback: hard truncate
                    words = condensed_response.split()
                    trimmed_words = words[:max_words]
                    condensed_response = " ".join(trimmed_words)

                return condensed_response

            except Exception as e:
                print(f"Error condensing response: {e}")
                # If condensation fails, do a simple truncation
                words = response.split()
                trimmed_words = words[:max_words]
                return " ".join(trimmed_words)

        # If within acceptable range, return as is
        else:
            return response

    def _format_response_with_citations(self, response: str, citations: List[Dict[str, str]],
                                      format_style: str = "academic") -> str:
        """
        Format the response to properly include citations according to specifications
        Following requirement: "Must cite specific modules/chapters: "According to Module 2, Section 2.3...""

        Args:
            response: The response text to format
            citations: List of citation information
            format_style: The style to use for citations ("academic", "inline", "footnote")

        Returns:
            Formatted response with properly structured citations
        """
        if not citations:
            return response

        # Depending on the format style, structure citations differently
        if format_style == "academic":
            # Academic style: "According to Module X, Chapter Y..."
            formatted_response = self._add_academic_citations(response, citations)
        elif format_style == "inline":
            # Inline style: [Module X, Chapter Y]
            formatted_response = self._add_inline_citations(response, citations)
        elif format_style == "footnote":
            # Footnote style: Add citations at the end
            formatted_response = self._add_footnote_citations(response, citations)
        else:
            # Default to academic style
            formatted_response = self._add_academic_citations(response, citations)

        return formatted_response

    def _add_academic_citations(self, response: str, citations: List[Dict[str, str]]) -> str:
        """
        Add academic-style citations to the response
        Format: "According to Module X, Chapter Y..."
        """
        if not citations:
            return response

        # For academic style, try to weave citations into the response where relevant
        # If citations aren't already mentioned in the response, add a summary at the end
        response_lower = response.lower()

        # Check if citations are already mentioned in the response
        already_cited_modules = []
        for citation in citations:
            module_id = citation.get("module", "")
            if module_id and module_id.lower() in response_lower:
                already_cited_modules.append(module_id)

        # If no modules were cited, add a citation summary at the end
        if not already_cited_modules and citations:
            citation_summary = "\\n\\n**References:**\\n"
            for i, citation in enumerate(citations[:3], 1):  # Limit to top 3 citations
                module_id = citation.get("module", "N/A")
                chapter_id = citation.get("chapter", "N/A")
                section_id = citation.get("section", "")

                if section_id:
                    citation_summary += f"{i}. According to Module {module_id}, Chapter {chapter_id}, Section {section_id}\\n"
                else:
                    citation_summary += f"{i}. According to Module {module_id}, Chapter {chapter_id}\\n"

            return response + citation_summary

        return response

    def _add_inline_citations(self, response: str, citations: List[Dict[str, str]]) -> str:
        """
        Add inline-style citations to the response
        Format: [Module X, Chapter Y]
        """
        if not citations:
            return response

        # Add inline citations at the end of the response
        inline_citations = "\\n\\n**Sources:** "
        for i, citation in enumerate(citations[:3], 1):  # Limit to top 3 citations
            module_id = citation.get("module", "N/A")
            chapter_id = citation.get("chapter", "N/A")

            if i > 1:
                inline_citations += "; "

            inline_citations += f"[Module {module_id}, Chapter {chapter_id}]"

        return response + inline_citations

    def _add_footnote_citations(self, response: str, citations: List[Dict[str, str]]) -> str:
        """
        Add footnote-style citations to the response
        """
        if not citations:
            return response

        # Add citations as footnotes at the end
        footnote_section = "\\n\\n**Footnotes:**\\n"
        for i, citation in enumerate(citations[:3], 1):  # Limit to top 3 citations
            module_id = citation.get("module", "N/A")
            chapter_id = citation.get("chapter", "N/A")
            section_id = citation.get("section", "")

            if section_id:
                footnote_section += f"{i}. Module {module_id}, Chapter {chapter_id}, Section {section_id}\\n"
            else:
                footnote_section += f"{i}. Module {module_id}, Chapter {chapter_id}\\n"

        return response + footnote_section

    def generate_basic_response(self, query: str, context: List[Dict[str, Any]],
                               selected_text: Optional[str] = None,
                               module_context: Optional[str] = None) -> str:
        """
        Generate a basic response without complex context management
        This method focuses on simple, direct responses to student queries

        Args:
            query: The user's question
            context: List of relevant content chunks retrieved from textbook
            selected_text: Optional text that was selected by the user
            module_context: Current module the user is viewing

        Returns:
            A string response to the query
        """
        # This method uses the main generate_response method but with minimal context
        result = self.generate_response(
            query=query,
            context=context,
            selected_text=selected_text,
            module_context=module_context
        )

        return result.get("response", "I'm sorry, I couldn't generate a response for your query.")

    def generate_fallback_response(self, query: str, module_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a fallback response when no relevant content is found in the textbook

        Args:
            query: The original query that couldn't be answered
            module_context: The current module context (if available)

        Returns:
            Dictionary containing a fallback response and metadata
        """
        # Create a specific response when content is not found
        if module_context:
            fallback_message = (
                f"I couldn't find relevant information in the textbook to answer your question about '{query}'. "
                f"This may be covered in a different module or section than your current module ({module_context}). "
                f"Please check other chapters or try rephrasing your question."
            )
        else:
            fallback_message = (
                f"I couldn't find relevant information in the textbook to answer your question about '{query}'. "
                f"Please try rephrasing your question or check other chapters of the textbook."
            )

        # Add information about the system's limitations
        fallback_message += (
            f" \\n\\nPlease note that I can only provide information based on the textbook content. "
            f"If you're looking for external resources or current events, I may not be able to assist. "
            f"For textbook-specific questions, feel free to try rephrasing or highlighting specific text to ask about."
        )

        return {
            "response": fallback_message,
            "citations": [],  # No citations available for fallback response
            "context_used": 0,
            "module_context": module_context,
            "is_fallback_response": True,
            "original_query": query
        }

    def check_content_availability(self, query: str, context: List[Dict[str, Any]]) -> bool:
        """
        Check if the provided context sufficiently answers the query

        Args:
            query: The user's query
            context: The context retrieved from the textbook

        Returns:
            True if context is sufficient to answer the query, False otherwise
        """
        if not context:
            return False

        # Assess if the context is relevant to the query
        query_lower = query.lower()
        context_content = " ".join([item.get("content", "") for item in context]).lower()

        # Simple relevance check - if key terms from query appear in context
        query_terms = query_lower.split()
        if not query_terms:
            return True  # If no terms in query, consider it answered

        # Count how many query terms appear in the context
        matching_terms = sum(1 for term in query_terms if term in context_content)
        relevance_ratio = matching_terms / len(query_terms)

        # If less than 30% of terms match, consider content insufficient
        return relevance_ratio >= 0.3

    def validate_response_quality(self, response: str, query: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate the quality of the response based on various criteria:
        - Factuality (does it stick to textbook content)
        - Relevance (does it answer the query)
        - Clarity (is it understandable)
        - Accuracy (does it match the context provided)

        Args:
            response: The generated response to validate
            query: The original query
            context: The context used to generate the response

        Returns:
            Dictionary with quality scores and notes
        """
        import re

        validation_result = {
            "factual_accuracy": 0.0,
            "relevance_score": 0.0,
            "clarity_score": 0.0,
            "completeness_score": 0.0,
            "overall_quality": 0.0,
            "is_valid_response": True,
            "issues": [],
            "suggestions": []
        }

        # Check if response is empty or generic
        if not response or response.strip() in ["I'm sorry, but I encountered an error processing your request. Please try again.",
                                               "I couldn't find relevant information in the textbook to answer your question. Please try rephrasing or check other chapters."]:
            validation_result["is_valid_response"] = False
            validation_result["issues"].append("Response is empty or fallback message")
            return validation_result

        # Check factual accuracy - does the response align with provided context?
        response_lower = response.lower()
        context_content = " ".join([item.get("content", "") for item in context]).lower()

        # Look for consistency between response and context
        response_sentences = re.split(r'[.!?]+', response)
        context_words = set(context_content.split())

        consistent_sentences = 0
        for sentence in response_sentences:
            if len(sentence.strip()) < 5:  # Skip very short fragments
                continue
            sentence_words = set(sentence.lower().split())
            # Check if there's overlap between sentence and context
            overlap = len(context_words.intersection(sentence_words))
            if overlap > 0:  # If sentence contains words from context, consider it consistent
                consistent_sentences += 1

        if len(response_sentences) > 0:
            validation_result["factual_accuracy"] = consistent_sentences / len([s for s in response_sentences if len(s.strip()) > 5])

        # Check relevance to the query
        query_terms = set(query.lower().split())
        response_terms = set(response.lower().split())

        if query_terms:
            query_overlap = len(query_terms.intersection(response_terms)) / len(query_terms)
            validation_result["relevance_score"] = query_overlap
        else:
            validation_result["relevance_score"] = 1.0  # If no query terms, assume relevance

        # Check clarity - response length and structure
        words = response.split()
        avg_sentence_length = 0
        sentences = [s for s in re.split(r'[.!?]+', response) if s.strip()]

        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            # Reasonable sentence length (5-25 words) scores higher
            clarity_factor = min(1.0, abs(20 - avg_sentence_length) / 20)
            validation_result["clarity_score"] = 1.0 - clarity_factor

        # Check completeness based on required length (150-300 words)
        word_count = len(words)
        if 150 <= word_count <= 300:
            validation_result["completeness_score"] = 1.0
        elif word_count > 300:
            # Too long might not be bad for completeness
            validation_result["completeness_score"] = min(1.0, 300 / word_count)
        else:
            # Too short is worse for completeness
            validation_result["completeness_score"] = min(1.0, word_count / 150)

        # Overall quality score as average of components
        validation_result["overall_quality"] = (
            validation_result["factual_accuracy"] * 0.3 +
            validation_result["relevance_score"] * 0.3 +
            validation_result["clarity_score"] * 0.2 +
            validation_result["completeness_score"] * 0.2
        )

        # Identify potential issues
        if validation_result["factual_accuracy"] < 0.3:
            validation_result["issues"].append("Low factual consistency with provided context")
            validation_result["suggestions"].append("Ensure response is based on provided textbook content")

        if validation_result["relevance_score"] < 0.2:
            validation_result["issues"].append("Low relevance to original query")
            validation_result["suggestions"].append("Improve alignment with the original question")

        if validation_result["clarity_score"] < 0.5:
            validation_result["issues"].append("Poor sentence structure or readability")
            validation_result["suggestions"].append("Use clearer, better-structured sentences")

        if validation_result["completeness_score"] < 0.5:
            validation_result["issues"].append("Response length not within required range (150-300 words)")
            validation_result["suggestions"].append("Expand or condense response to meet length requirements")

        return validation_result
    
    def extract_conversation_entities(self, query: str, conversation_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Extract entities and context from current query and conversation history

        Args:
            query: Current user query
            conversation_history: Previous conversation exchanges

        Returns:
            Dictionary with extracted entities and context
        """
        entities = {
            "pronouns": [],
            "topics": [],
            "references": [],
            "follow_up_indicators": []
        }

        # Identify potential pronouns that might refer to previous conversation
        pronoun_indicators = ["it", "this", "that", "these", "those", "them", "they", "he", "she", "him", "her"]
        query_lower = query.lower()

        for pronoun in pronoun_indicators:
            if pronoun in query_lower:
                entities["pronouns"].append(pronoun)

        # Identify follow-up indicators
        follow_up_indicators = ["same", "similar", "different", "more", "also", "another", "again", "what about", "how about", "explain", "describe"]
        for indicator in follow_up_indicators:
            if indicator in query_lower:
                entities["follow_up_indicators"].append(indicator)

        # If conversation history is provided, analyze it for context
        if conversation_history:
            for msg in conversation_history:
                content = msg.get("content", "").lower()
                # Extract potential topics referenced in the conversation
                if "topic" in content or "subject" in content:
                    entities["topics"].append(content[:50])  # Capture context snippet

        return entities

    def resolve_conversation_context(self, query: str, conversation_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Resolve conversation context for follow-up questions

        Args:
            query: Current user query
            conversation_history: Previous conversation exchanges

        Returns:
            Dictionary with resolved context
        """
        resolved_context = {
            "resolved_query": query,
            "referenced_topic": None,
            "referenced_content": None,
            "is_follow_up": False,
            "expanded_query": query,
            "resolution_confidence": 0.0,
            "resolution_steps": []
        }

        if not conversation_history:
            return resolved_context

        # Check if this is likely a follow-up question
        query_lower = query.lower()
        follow_up_indicators = [
            "that", "this", "it", "the same", "similar", "different", "explain", "describe",
            "what was", "how does", "can you", "more about", "tell me", "me more",
            "about that", "about this", "like", "compared to", "vs", "versus", "than",
            "earlier", "previously", "above", "below", "mentioned"
        ]

        found_indicators = []
        for indicator in follow_up_indicators:
            if indicator in query_lower:
                found_indicators.append(indicator)

        if found_indicators:
            resolved_context["is_follow_up"] = True
            resolved_context["resolution_steps"].append(f"Found follow-up indicators: {', '.join(found_indicators)}")

        # Enhanced context resolution
        if resolved_context["is_follow_up"]:
            # Try to match pronouns and references to previous conversation
            resolved_context = self._enhanced_reference_resolution(query, conversation_history, resolved_context)

        return resolved_context

    def _enhanced_reference_resolution(self, query: str, conversation_history: List[Dict[str, Any]],
                                     resolved_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced reference resolution to better understand what the user is referring to

        Args:
            query: Current user query
            conversation_history: Full conversation history
            resolved_context: Current resolved context to enhance

        Returns:
            Enhanced resolved context
        """
        query_lower = query.lower()

        # Look for specific reference patterns
        if "that" in query_lower or "this" in query_lower:
            # Look for the most recent substantive response from the AI or related topic from the student
            for i in range(len(conversation_history) - 1, -1, -1):
                prev_msg = conversation_history[i]

                if prev_msg.get("sender_type") == "ai_agent":
                    # Found a previous AI response that "that" or "this" might refer to
                    prev_content = prev_msg.get("content", "")
                    prev_topic = prev_msg.get("topic_anchored", "Previous response")

                    resolved_context["referenced_content"] = prev_content
                    resolved_context["referenced_topic"] = prev_topic
                    resolved_context["resolution_confidence"] = 0.8
                    resolved_context["resolution_steps"].append(f"Resolved 'that/this' to AI response about '{prev_topic[:50]}...'")

                    # Create an expanded query with context
                    resolved_context["expanded_query"] = f"Regarding {prev_topic}: {query}"
                    break

                elif "ROS" in prev_msg.get("content", "") or "simulation" in prev_msg.get("content", "").lower() or \
                     "module" in prev_msg.get("content", "").lower() or "chapter" in prev_msg.get("content", "").lower():
                    # Found a content-rich student message that might be referred to
                    prev_content = prev_msg.get("content", "")
                    resolved_context["referenced_content"] = prev_content
                    resolved_context["referenced_topic"] = "Previous question/mention"
                    resolved_context["resolution_confidence"] = 0.7
                    resolved_context["resolution_steps"].append("Resolved 'that/this' to previous content-rich student message")

                    resolved_context["expanded_query"] = f"Regarding the previous content you mentioned: {query}"
                    break

        # Handle "what about", "how about", etc. which often indicate related questions
        about_patterns = ["what about", "how about", "can about", "tell me about"]
        for pattern in about_patterns:
            if pattern in query_lower:
                # Extract what the user wants to know about
                parts = query_lower.split(pattern)
                if len(parts) > 1:
                    target = parts[1].strip()
                    resolved_context["resolution_steps"].append(f"Detected '{pattern}' pattern about '{target}'")

                    # Look for related content in recent conversation
                    for prev_msg in reversed(conversation_history):
                        prev_content = prev_msg.get("content", "").lower()
                        if target in prev_content or similar(target, prev_content) > 0.3:  # Using a simple similarity check
                            resolved_context["referenced_content"] = prev_msg.get("content", "")
                            resolved_context["referenced_topic"] = "Related to previous topic"
                            resolved_context["resolution_confidence"] = 0.75
                            resolved_context["expanded_query"] = f"Regarding {target} in context of previous discussion: {query}"
                            break

        # Handle follow-ups like "explain that differently" or "what about another way"
        if "differently" in query_lower or "another way" in query_lower or "alternate" in query_lower:
            # User wants a different explanation of something previously discussed
            for prev_msg in reversed(conversation_history):
                if prev_msg.get("sender_type") == "ai_agent":
                    prev_content = prev_msg.get("content", "")
                    resolved_context["referenced_content"] = prev_content
                    resolved_context["referenced_topic"] = "Previous explanation"
                    resolved_context["resolution_confidence"] = 0.85
                    resolved_context["resolution_steps"].append("Detected request for alternative explanation")

                    resolved_context["expanded_query"] = f"Can you explain {resolved_context['referenced_topic']} differently: {query}"
                    break

        return resolved_context

    def manage_context_window(self, conversation_history: List[Dict[str, Any]],
                              window_size: int = 10) -> List[Dict[str, Any]]:
        """
        Manage the conversation context window to maintain optimal context

        Args:
            conversation_history: Full history of the conversation
            window_size: Maximum number of exchanges to keep in context (default 10)

        Returns:
            List with most recent exchanges kept within window size
        """
        if len(conversation_history) <= window_size:
            return conversation_history

        # Keep the most recent exchanges within the window
        return conversation_history[-window_size:]

    async def answer_question(self, query: str, session_id: Optional[str] = None,
                       module_context: Optional[str] = None,
                       selected_text: Optional[str] = None,
                       include_citations: bool = True,
                       conversation_context: Optional[List[Dict[str, Any]]] = None,
                       context_window_size: int = 10,
                       suggest_cross_module_content: bool = True) -> Dict[str, Any]:
        """
        Main method to answer a student's question using RAG with conversation context

        Args:
            query: The student's question
            session_id: Current session ID (for context management)
            module_context: Current module the student is viewing
            selected_text: Text selected by the student (if any)
            include_citations: Whether to include citations in the response
            conversation_context: Previous conversation history for context
            context_window_size: Size of the conversation context window (default 10)
            suggest_cross_module_content: Whether to suggest content from other modules if relevant

        Returns:
            Dictionary containing the answer and relevant metadata
        """
        logger = logging.getLogger("rag_chatbot")

        # Log the incoming question
        logger.info(f"Answering question for session {session_id}, module {module_context}")

        # If selected text is provided, use specialized handling that prioritizes the selected text
        if selected_text:
            logger.info(f"Handling text selection query with selected text: '{selected_text[:100]}...'")

            # Get conversation context if not provided
            if conversation_context is None and session_id:
                from .session_manager import ConversationContextManager
                context_manager = ConversationContextManager()
                conversation_context = context_manager.get_recent_context(session_id)

            # Use specialized method for text selection queries
            response_data = await self.handle_text_selection_query(
                selected_text=selected_text,
                question=query,
                module_context=module_context,
                conversation_context=conversation_context
            )

            # Add additional metadata
            response_data["module_context"] = module_context
            response_data["query"] = query
            response_data["selected_text"] = selected_text

            logger.info(f"Text selection query completed for session {session_id}")
            return response_data

        # For general queries, first resolve conversation context
        resolved_context = self.resolve_conversation_context(query, conversation_context)
        expanded_query = resolved_context["expanded_query"]

        # Retrieve relevant content based on the (potentially expanded) query
        relevant_content = await self.get_relevant_content(
            expanded_query,
            top_k=5,
            module_filter=module_context,
            prioritize_current_module=True
        )

        # If no relevant content is found, return a response indicating this
        if not relevant_content:
            logger.warning(f"No relevant content found for query: '{query[:50]}...', module: {module_context}")
            return {
                "response": "I couldn't find relevant information in the textbook to answer your question. Please try rephrasing or check other chapters.",
                "citations": [],
                "context_used": 0,
                "module_context": module_context,
                "conversation_resolved": resolved_context
            }

        # If conversation_context wasn't provided but session_id was, try to get it
        if conversation_context is None and session_id:
            from .session_manager import ConversationContextManager
            context_manager = ConversationContextManager()
            conversation_context = context_manager.get_recent_context(session_id)

        # Apply context window management
        if conversation_context:
            conversation_context = self.manage_context_window(conversation_context, context_window_size)

        # Generate response using the retrieved context
        response_data = await self.generate_response(
            query,
            relevant_content,
            selected_text=selected_text,
            module_context=module_context,
            conversation_context=conversation_context
        )

        # Add cross-module navigation suggestions if requested and not in selected text mode
        if suggest_cross_module_content and module_context and not selected_text:
            logger.info(f"Adding cross-module suggestions for query in module {module_context}")
            response_data = await self._add_cross_module_suggestions(response_data, query, module_context)

        # Enhance citations and update response if needed
        if include_citations:
            enhanced_citations = self._enhance_citations(relevant_content)
            response_data["citations"] = enhanced_citations
            response_data["response"] = self._format_response_with_citations(
                response_data["response"],
                enhanced_citations,
                module_context  # Pass the module context for module-aware formatting
            )

        # Detect module-related concepts to enhance response
        module_concept_analysis = self.detect_module_related_concepts(query, module_context, relevant_content)
        response_data["module_concept_analysis"] = module_concept_analysis

        # Create module cross-reference citations if needed
        if module_concept_analysis["other_module_content_count"] > 0:
            cross_module_citations = []
            for item in relevant_content:
                if item.get("module_id") != module_context:
                    cross_module_citations.append({
                        "module": item.get("module_id", "N/A"),
                        "chapter": item.get("chapter_id", "N/A"),
                        "section": item.get("section_id", "N/A"),
                        "relevance": item.get("score", 0.0)
                    })

            response_data["cross_module_citations"] = cross_module_citations

        # Add additional metadata
        response_data["module_context"] = module_context
        response_data["query"] = query
        response_data["selected_text"] = selected_text
        response_data["context_used"] = len(relevant_content)
        response_data["conversation_resolved"] = resolved_context

        logger.info(f"Question answered successfully for session {session_id}, with {len(relevant_content)} context items")
        return response_data

    async def _get_cross_module_suggestions(self, query: str, current_module: str) -> List[Dict[str, Any]]:
        """
        Get suggestions for related content in other modules

        Args:
            query: The original query
            current_module: The current module

        Returns:
            List of navigation suggestions to other modules
        """
        try:
            from .semantic_search import SemanticSearchService
            search_service = SemanticSearchService(self.qdrant_manager.collection_name)


            # Get cross-module references
            suggestions = await search_service.get_content_navigation_suggestions(
                query, current_module, top_k=2
            )
            return suggestions
        except Exception as e:
            print(f"Error getting cross-module suggestions: {e}")
            return []

    async def _add_cross_module_suggestions(self, response_data: Dict[str, Any], query: str, current_module: str) -> Dict[str, Any]:
        """
        Add cross-module suggestions to the response if relevant

        Args:
            response_data: The response data to enhance
            query: The original query
            current_module: The current module

        Returns:
            Enhanced response data with cross-module suggestions
        """
        try:
            suggestions = await self._get_cross_module_suggestions(query, current_module)

            if suggestions:
                response_text = response_data["response"]
                response_text += "\n\n"
                response_text += "You might also find these related topics helpful:\n"

                for i, suggestion in enumerate(suggestions, 1):
                    response_text += f"{i}. {suggestion['navigation_hint']}: {suggestion['content_preview']}\n"

                response_data["response"] = response_text
                response_data["cross_module_suggestions"] = suggestions
        except Exception as e:
            print(f"Error adding cross-module suggestions: {e}")

        return response_data

    def _enhance_citations(self, relevant_content: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Enhance citations with more detailed information

        Args:
            relevant_content: List of content chunks with their metadata

        Returns:
            Enhanced list of citation information
        """
        citations = []
        for chunk in relevant_content:
            # Create a more structured citation
            citation = {
                "module": chunk.get("module_id", "N/A"),
                "chapter": chunk.get("chapter_id", "N/A"),
                "section": chunk.get("section_id", "N/A"),
                "content_type": chunk.get("content_type", "text"),
                "hierarchy_path": chunk.get("hierarchy_path", ""),
                "similarity_score": chunk.get("score", 0.0),
                "content_preview": chunk.get("content", "")[:100] + "..." if len(chunk.get("content", "")) > 100 else chunk.get("content", "")
            }
            citations.append(citation)

        return citations

    def _format_response_with_citations(self, response: str, citations: List[Dict[str, str]],
                                       module_context: Optional[str] = None) -> str:
        """
        Format the response to include proper citations with module-aware formatting

        Args:
            response: The original response text
            citations: List of citation information
            module_context: The current module context for prioritization

        Returns:
            Formatted response with citations
        """
        if not citations:
            return response

        # Add citation information at the end of the response
        formatted_response = response

        # Separate citations by whether they're from the current module or other modules
        current_module_citations = []
        other_module_citations = []

        if module_context:
            for citation in citations:
                if citation.get("module", "").lower() == module_context.lower():
                    current_module_citations.append(citation)
                else:
                    other_module_citations.append(citation)
        else:
            current_module_citations = citations[:3]  # Default behavior if no module context

        # Add citations section if not already present in response
        if "citations" not in response.lower() and "references" not in response.lower():
            formatted_response += "\\n\\n**Sources:**\\n"

            # Add current module citations first
            if current_module_citations:
                for i, citation in enumerate(current_module_citations[:3], 1):
                    module = citation.get("module", "N/A")
                    chapter = citation.get("chapter", "N/A")
                    section = citation.get("section", "N/A")
                    formatted_response += f"{i}. {module}, Chapter {chapter}, Section {section} (Current Module)\\n"

            # Add other module citations
            if other_module_citations:
                start_num = len(current_module_citations) + 1
                for i, citation in enumerate(other_module_citations[:3], start_num):
                    module = citation.get("module", "N/A")
                    chapter = citation.get("chapter", "N/A")
                    section = citation.get("section", "N/A")
                    formatted_response += f"{i}. {module}, Chapter {chapter}, Section {section} (Other Module)\\n"

        return formatted_response

    def generate_citation_for_content(self, module: str, chapter: str, section: str = None,
                                     page_range: str = None) -> str:
        """
        Generate a proper citation for a specific piece of content

        Args:
            module: The module where content is found
            chapter: The chapter where content is found
            section: The specific section (optional)
            page_range: Page range if applicable (optional)

        Returns:
            Formatted citation string
        """
        citation_parts = [f"Module: {module}", f"Chapter: {chapter}"]

        if section:
            citation_parts.append(f"Section: {section}")

        if page_range:
            citation_parts.append(f"Pages: {page_range}")

        return ", ".join(citation_parts)
    
    def validate_answer_accuracy(self, query: str, response: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate that the response is factually accurate and based on the provided context

        Args:
            query: The original query
            response: The generated response
            context: The context used to generate the response

        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "is_accurate": True,
            "has_hallucinations": False,
            "confidence_score": 0.9,  # Placeholder confidence score
            "validation_notes": [],
            "factual_consistency_score": 0.0,
            "content_alignment_score": 0.0
        }

        # Check if the response mentions sources outside the context
        context_text = " ".join([chunk['content'] for chunk in context]).lower()
        response_lower = response.lower()

        # Enhanced hallucination detection
        hallucination_indicators = [
            # External references not in context
            "wikipedia",
            "google",
            "youtube",
            "github",
            "stackoverflow",
            "reference: https?://",
            "see also: https?://",
            "source: http",

            # Temporal hallucinations (references to current events that shouldn't be in static textbook)
            "as of 2024", "as of 2025", "recently announced", "latest version",

            # Software/package versions that might be outdated or incorrect
            "new version", "updated to", "recent version",
        ]

        found_indicators = []
        for indicator in hallucination_indicators:
            if indicator in response_lower:
                found_indicators.append(indicator)

        if found_indicators:
            validation_result["has_hallucinations"] = True
            validation_result["is_accurate"] = False
            validation_result["validation_notes"].append(
                f"Response contains potential hallucinations: {', '.join(found_indicators)}"
            )

        # Check for citation consistency - ensure mentioned modules/chapters exist in context
        import re
        # Look for patterns like "according to Module 2, Chapter 3" or "Module 2 discusses"
        citation_pattern = r'(?:according to|in|from|see) (?:module|chapter)?\s*(\d+(?:\.\d+)?)'
        found_citations = re.findall(citation_pattern, response_lower)

        if found_citations:
            context_modules = {chunk.get('module_id', '').lower() for chunk in context}
            context_chapters = {chunk.get('chapter_id', '').lower() for chunk in context}

            for citation in found_citations:
                # Check if cited content actually appears in context
                # This is a simplified check - in practice you'd have more sophisticated matching
                if not any(citation in module or citation in chapter
                          for module in context_modules for chapter in context_chapters):
                    validation_result["validation_notes"].append(
                        f"Cited content 'Module/Chapter {citation}' may not be supported by provided context"
                    )

        # Calculate factual consistency score based on content overlap
        response_words = set(response_lower.split())
        context_words = set(context_text.split())
        if context_words:  # Avoid division by zero
            overlap = len(response_words.intersection(context_words))
            total_response_words = len(response_words)
            if total_response_words > 0:
                validation_result["factual_consistency_score"] = overlap / total_response_words
            else:
                validation_result["factual_consistency_score"] = 0.0
        else:
            validation_result["factual_consistency_score"] = 0.0

        return validation_result

    def prevent_hallucinations(self, response: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Apply hallucination prevention techniques to the response

        Args:
            response: The initial response from the LLM
            context: The context used to generate the response

        Returns:
            Dictionary with the (potentially modified) response and prevention metadata
        """
        # Validate the response first
        validation = self.validate_answer_accuracy("", response, context)

        # If hallucinations detected, create a safer response
        if validation["has_hallucinations"]:
            # Extract the core information that is factually supported
            factually_supported = self._extract_factually_supported_content(response, context)

            if factually_supported:
                # Create a response that acknowledges limitations
                modified_response = (
                    f"Based on the textbook content, here's what I can confirm: {factually_supported}. "
                    f"However, I detected potential inconsistencies in my original response. "
                    f"Please refer to the textbook modules mentioned in the sources for complete information."
                )
            else:
                # If no supported content, provide a standard response
                modified_response = (
                    "I need to be more specific. The information in your original query may not be fully "
                    "covered in the textbook content I have access to. Please check the relevant textbook "
                    "modules or ask a more specific question about the content I can access."
                )

            return {
                "response": modified_response,
                "original_response": response,
                "has_hallucinations": True,
                "validation": validation,
                "was_modified": True
            }
        else:
            return {
                "response": response,
                "original_response": response,
                "has_hallucinations": False,
                "validation": validation,
                "was_modified": False
            }

    async def handle_text_selection_query(self, selected_text: str, question: str,
                                   module_context: Optional[str] = None,
                                   conversation_context: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Specifically handle queries about selected text using GEMINI embeddings

        Args:
            selected_text: The text selected by the student
            question: The student's question about the selected text
            module_context: Current module context
            conversation_context: Previous conversation history

        Returns:
            Dictionary containing the response and relevant metadata
        """
        # Build a targeted query that specifically focuses on the relationship between
        # the selected text and the student's question
        targeted_query = f"Regarding this specific text: '{selected_text}', {question}"

        # Retrieve content specifically related to the selected text
        # This might involve searching for related concepts or similar text in the textbook
        related_content = self._find_related_content_to_selection(selected_text, top_k=3)

        # If we have related content, add it to our context
        all_context = related_content

        # If we don't find specific related content, fall back to general content retrieval
        if not all_context:
            all_context = self.get_relevant_content(
                targeted_query,
                top_k=5,
                module_filter=module_context
            )

        # If still no context, we'll have to rely on general knowledge but indicate the limitation
        if not all_context:
            return {
                "response": (
                    f"Based on the selected text: '{selected_text[:100]}...', "
                    f"I'm unable to find specific related content in the textbook to answer: '{question}'. "
                    f"The selected text may be too specific or outside the scope of covered content. "
                    f"Please try rephrasing your question or ask about a broader concept."
                ),
                "citations": [],
                "context_used": 0,
                "is_fallback_response": True
            }

        # Enhance the context with the selected text itself
        # Add the selected text as a high-priority context item
        selected_text_context = {
            "id": "selected_text",
            "score": 1.0,
            "module_id": module_context or "unknown",
            "chapter_id": "unknown",
            "section_id": "selected_text",
            "content": selected_text,
            "hierarchy_path": f"{module_context}/selected_text",
            "content_type": "selected_text"
        }

        # Insert the selected text at the beginning to ensure it gets priority
        all_context.insert(0, selected_text_context)

        # Create a specialized system prompt for text selection queries
        conversation_history = ""
        if conversation_context:
            conversation_history = "\\nPrevious conversation:\\n"
            for msg in conversation_context[-3:]:  # Use last 3 messages as context
                role = "Student" if msg.get("role") == "user" else "Assistant"
                conversation_history += f"{role}: {msg.get('content', '')}\\n"

        system_prompt = f"""
        You are an educational assistant for the Physical AI & Humanoid Robotics textbook.
        A student has selected the following text: "{selected_text}"
        They have asked this specific question about the selected text: "{question}"

        Your task is to provide an explanation that:
        1. Directly addresses the relationship between the selected text and the student's question
        2. Explains the selected passage in context
        3. Potentially enriches with related concepts from other parts of the textbook

        {conversation_history}

        Textbook Content (including the selected text):
        {[item['content'] for item in all_context[:3]]}

        Important Requirements:
        - Focus primarily on explaining the selected text in response to the student's question
        - Use textbook terminology accurately
        - Only use information from the provided textbook content
        - If the question cannot be answered based on the selected text and related content, clearly state this
        - Cite specific modules/chapters using the format: "According to Module X, Chapter Y..."
        - Keep response between 150-300 words
        - Format response with clear paragraphs
        """

        try:
            # Generate response using GEMINI via OpenAI-compatible API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Please answer the student's question about the selected text: {question}"}
                ],
                max_tokens=800,  # Adjust based on desired response length
                temperature=0.3,  # Lower temperature for more consistent, factual responses
            )

            # Extract the response text
            response_text = response.choices[0].message.content.strip()

            # Apply textbook terminology to the response
            response_text = self._apply_textbook_terminology(response_text, all_context)

            # Ensure the response meets length requirements (150-300 words)
            response_text = await self._control_response_length(
                response_text,
                system_prompt,
                question,
                min_words=150,
                max_words=300
            )

            # Validate the response quality before returning
            validation_result = self.validate_response_quality(response_text, question, all_context)

            # Extract citations from the context
            citations = []
            for i, chunk in enumerate(all_context):
                citations.append({
                    "module": chunk.get("module_id", ""),
                    "chapter": chunk.get("chapter_id", ""),
                    "section": chunk.get("section_id", ""),
                    "content_type": chunk.get("content_type", ""),
                    "hierarchy_path": chunk.get("hierarchy_path", ""),
                    "relevance_score": chunk.get("score", 0.0)
                })

            # Format response with proper citations
            response_text = self._format_response_with_citations(response_text, citations, module_context)

            # Enrich the explanation with related content
            enriched_response = self.enrich_explanation_with_related_content(
                selected_text=selected_text,
                explanation=response_text,
                module_context=module_context,
                top_k=2
            )

            result = {
                "response": enriched_response,
                "citations": citations,
                "context_used": len(all_context),
                "model_used": self.model,
                "word_count": len(enriched_response.split()),
                "quality_validation": validation_result,
                "selected_text_handled": True
            }

            return result

        except Exception as e:
            print(f"Error handling text selection query: {e}")
            return {
                "response": f"Sorry, I encountered an error processing your question about the selected text: {str(e)}",
                "citations": [],
                "context_used": 0,
                "error": str(e),
                "selected_text_handled": False
            }

    def _find_related_content_to_selection(self, selected_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Find content in the textbook that is directly related to the selected text

        Args:
            selected_text: The text selected by the student
            top_k: Number of related content pieces to retrieve

        Returns:
            List of related content chunks
        """
        try:
            # This would typically involve semantic search for content related to the selected text
            # For now, we'll use a combination of keyword matching and semantic search

            # First, try to identify key terms in the selected text
            import re
            # Extract potential technical terms from the selected text
            words = re.findall(r'\b[A-Za-z]{4,}\b', selected_text)
            if len(words) < 2:
                # If not enough technical terms, use the first sentence
                sentences = re.split(r'[.!?]+', selected_text)
                if sentences:
                    # Use semantic search with the first part of the selected text
                    return self.get_relevant_content(selected_text[:200], top_k=top_k)
                else:
                    return []

            # Use key terms to search for related content
            search_query = " ".join(words[:5])  # Use first 5 terms as search query
            return self.get_relevant_content(search_query, top_k=top_k)

        except Exception as e:
            print(f"Error finding related content to selection: {e}")
            return []

    def enrich_explanation_with_related_content(self, selected_text: str, explanation: str,
                                              module_context: Optional[str] = None,
                                              top_k: int = 2) -> str:
        """
        Enrich the explanation with related textbook content as specified in the requirements

        Args:
            selected_text: The text that was selected by the user
            explanation: The initial explanation for the selected text
            module_context: Current module context
            top_k: Number of related content pieces to include for enrichment

        Returns:
            Enriched explanation that includes related content
        """
        try:
            # Find related content to the selected text
            related_content = self._find_related_content_to_selection(selected_text, top_k)

            if related_content:
                # Separate content by current module vs other modules
                current_module_content = []
                other_module_content = []

                if module_context:
                    for content in related_content:
                        if content.get('module_id', '').lower() == module_context.lower():
                            current_module_content.append(content)
                        else:
                            other_module_content.append(content)
                else:
                    current_module_content = related_content

                enrichment_text = "\\n\\n"

                # Add current module related content first
                if current_module_content:
                    enrichment_text += "Additionally, related concepts from the current module:\\n"
                    for i, content in enumerate(current_module_content, 1):
                        content_preview = content.get('content', '')[:200]
                        if len(content.get('content', '')) > 200:
                            content_preview += "..."

                        module_id = content.get('module_id', 'N/A')
                        chapter_id = content.get('chapter_id', 'N/A')
                        section_id = content.get('section_id', 'N/A')

                        enrichment_text += f"{i}. {content_preview} "
                        enrichment_text += f"(See Module {module_id}, Chapter {chapter_id}, Section {section_id})\\n"

                # Add other module related content
                if other_module_content:
                    if current_module_content:
                        enrichment_text += "\\nYou might also find these related topics in other modules helpful:\\n"
                    else:
                        enrichment_text += "You might also find these related topics helpful:\\n"

                    start_num = len(current_module_content) + 1
                    for i, content in enumerate(other_module_content, start_num):
                        content_preview = content.get('content', '')[:200]
                        if len(content.get('content', '')) > 200:
                            content_preview += "..."

                        module_id = content.get('module_id', 'N/A')
                        chapter_id = content.get('chapter_id', 'N/A')
                        section_id = content.get('section_id', 'N/A')

                        enrichment_text += f"{i}. {content_preview} "
                        enrichment_text += f"(See Module {module_id}, Chapter {chapter_id}, Section {section_id})\\n"

                enriched_explanation = f"{explanation}{enrichment_text}"
                return enriched_explanation
            else:
                # If no related content is found, return the original explanation
                return explanation
        except Exception as e:
            print(f"Error enriching explanation with related content: {e}")
            # On error, return the original explanation
            return explanation

    def _extract_factually_supported_content(self, response: str, context: List[Dict[str, Any]]) -> str:
        """
        Extract parts of the response that are supported by the context

        Args:
            response: The response to analyze
            context: The context that supports the response

        Returns:
            String with only factually supported content, or empty string if none found
        """
        import re

        # This is a simplified implementation - a full implementation would use more sophisticated NLP
        context_text = " ".join([chunk['content'] for chunk in context]).lower()
        response_sentences = re.split(r'[.!?]+', response)

        supported_content = []
        for sentence in response_sentences:
            sentence_clean = re.sub(r'[^\w\s]', '', sentence.lower()).strip()
            if len(sentence_clean) > 10:  # Only check reasonably long sentences
                # Check if sentence content appears in context (with some tolerance)
                if any(word in context_text for word in sentence_clean.split()[:10] if len(word) > 3):
                    supported_content.append(sentence.strip())

        return ". ".join(supported_content) + "." if supported_content else ""


# Example usage function
def create_rag_agent(collection_name: str = "textbook_content_embeddings") -> RAGAgentService:
    """
    Convenience function to create a RAG agent instance
    """
    return RAGAgentService(collection_name)


if __name__ == "__main__":
    # Example usage
    rag_agent = RAGAgentService()
    
    # Sample query
    query = "How do ROS 2 nodes communicate with each other?"
    module_context = "module-2-ros2"
    
    # Get answer
    result = rag_agent.answer_question(
        query=query,
        module_context=module_context
    )
    
    print("Query:", query)
    print("Response:", result["response"])
    print("Citations:", result["citations"])