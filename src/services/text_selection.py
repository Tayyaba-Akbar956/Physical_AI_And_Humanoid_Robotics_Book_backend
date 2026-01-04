from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4
from datetime import datetime
from sqlalchemy.orm import Session
from ..db.connection import get_db, SessionLocal
from ..models.selected_text import SelectedTextDB, SelectedTextCreate
from ..models.message import MessageDB, MessageCreate
from ..models.chat_session import ChatSessionDB
from ..models.textbook_content import TextbookContentDB
from ..services.session_manager import SessionManagementService
from ..services.rag_agent import RAGAgentService
import re


class TextSelectionService:
    """
    Service for handling text selection and related functionality
    Manages selected text storage, validation, and integration with chat sessions
    """
    
    def __init__(self):
        """
        Initialize the text selection service
        """
        self.session_manager = SessionManagementService()
        self.rag_agent = RAGAgentService()
        self.enrichment_service = None  # Will be initialized when needed to avoid circular imports
    
    def validate_selected_text(self, selected_text: str) -> Dict[str, Any]:
        """
        Validate the selected text according to requirements

        Args:
            selected_text: The text that was selected by the user

        Returns:
            Dictionary with validation results
        """
        result = {
            "is_valid": True,
            "character_count": len(selected_text),
            "error_message": None,
            "can_ask_query": False,
            "suggestions": []
        }

        # Check minimum character requirement (20 characters)
        if len(selected_text) < 20:
            result["is_valid"] = False
            result["error_message"] = f"Selected text must be at least 20 characters. Current length: {len(selected_text)}"
            result["suggestions"].append("Please select more text (minimum 20 characters)")
        elif len(selected_text) > 5000:  # Maximum limit to prevent abuse
            result["is_valid"] = False
            result["error_message"] = f"Selected text is too long: {len(selected_text)}. Max allowed: 5000 characters"
            result["suggestions"].append("Please select a shorter portion of text")
        else:
            result["can_ask_query"] = True

        # Check for inappropriate content (basic filtering)
        # This is a simplified check - a real implementation would have more sophisticated content validation
        inappropriate_indicators = [
            "http://", "https://",  # External links
            "@",  # Email addresses
            "<script", "javascript:",  # Potential XSS attempts
        ]

        selected_lower = selected_text.lower()
        for indicator in inappropriate_indicators:
            if indicator in selected_lower:
                result["is_valid"] = False
                result["error_message"] = f"Selected text contains potentially inappropriate content: {indicator}"
                result["suggestions"].append("Please select text that doesn't contain links, emails, or scripts")
                break

        return result
    
    def store_selected_text(self, content: str, module_id: str, chapter_id: str, 
                           section_id: str, hierarchy_path: str) -> Optional[Dict[str, Any]]:
        """
        Store the selected text in the database
        
        Args:
            content: The selected text content
            module_id: Module where text was selected
            chapter_id: Chapter where text was selected
            section_id: Section where text was selected
            hierarchy_path: Full path in textbook hierarchy
            
        Returns:
            Dictionary with stored text information or None if failed
        """
        db = SessionLocal()
        try:
            validation = self.validate_selected_text(content)
            if not validation["is_valid"]:
                print(f"Validation failed: {validation['error_message']}")
                return None
            
            # Create selected text record
            selected_text_data = SelectedTextCreate(
                content=content,
                module_id=module_id,
                chapter_id=chapter_id,
                section_id=section_id,
                hierarchy_path=hierarchy_path
            )
            
            db_selected_text = SelectedTextDB(
                content=selected_text_data.content,
                module_id=selected_text_data.module_id,
                chapter_id=selected_text_data.chapter_id,
                section_id=selected_text_data.section_id,
                hierarchy_path=selected_text_data.hierarchy_path
            )
            
            db.add(db_selected_text)
            db.commit()
            db.refresh(db_selected_text)
            
            return {
                "id": db_selected_text.id,
                "content": db_selected_text.content,
                "module_id": db_selected_text.module_id,
                "chapter_id": db_selected_text.chapter_id,
                "section_id": db_selected_text.section_id,
                "hierarchy_path": db_selected_text.hierarchy_path,
                "created_at": db_selected_text.created_at
            }
        except Exception as e:
            print(f"Error storing selected text: {e}")
            db.rollback()
            return None
        finally:
            db.close()
    
    async def process_text_selection_query(self, session_id: UUID, selected_text_id: UUID,
                                   question: str, module_context: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Process a query about selected text

        Args:
            session_id: The session in which the query is made
            selected_text_id: The ID of the selected text
            question: The question about the selected text
            module_context: Current module context (optional)

        Returns:
            Dictionary with response and citations
        """
        db = SessionLocal()
        try:
            # Get the selected text
            selected_text = db.query(SelectedTextDB).filter(SelectedTextDB.id == selected_text_id).first()
            if not selected_text:
                print(f"Selected text with ID {selected_text_id} not found")
                return None

            # Retrieve the selected text content
            selected_text_content = selected_text.content

            # Enhance the query to specifically reference the selected text
            enhanced_query = f"Regarding the selected text: '{selected_text_content}', {question}"

            # Use the RAG agent to generate a response based on the selected text
            # We'll pass the selected text as additional context
            result = await self.rag_agent.answer_question(
                query=enhanced_query,
                session_id=str(session_id),
                module_context=module_context,
                selected_text=selected_text_content
            )

            # Add the selected text reference to the result
            result["selected_text_id"] = selected_text_id
            result["selected_text_content"] = selected_text_content

            # Format the response to clearly reference the selected text
            formatted_response = self.format_selected_text_response(
                result.get("response", "I couldn't generate a response for your query about the selected text."),
                selected_text_content
            )

            # Update the result with the formatted response
            result["response"] = formatted_response

            # Add the response to the session as a message
            if "response" in result:
                message_result = self.session_manager.add_message_to_session(
                    session_id=session_id,
                    sender_type="ai_agent",
                    content=result["response"],
                    citations=result.get("citations", []),
                    selected_text_ref=selected_text_id
                )
                result["message_added"] = message_result is not None

            return result
        except Exception as e:
            print(f"Error processing text selection query: {e}")
            return {
                "response": "I'm sorry, but I encountered an error processing your query about the selected text. Please try again.",
                "citations": [],
                "selected_text_id": selected_text_id,
                "selected_text_content": selected_text_content if 'selected_text_content' in locals() else "",
                "message_added": False
            }
        finally:
            db.close()

    def format_selected_text_response(self, response: str, selected_text: str) -> str:
        """
        Format the response to clearly indicate it's about the selected text

        Args:
            response: The AI-generated response
            selected_text: The text that was selected

        Returns:
            Formatted response string
        """
        # If the selected text is too long to include directly, we'll summarize
        if len(selected_text) > 100:
            selected_preview = selected_text[:100] + "..."
        else:
            selected_preview = selected_text

        formatted_response = (
            f"Based on the passage you selected: \"{selected_preview}\", "
            f"here is the explanation: {response}"
        )
        return formatted_response

    def enrich_explanation_with_related_content(self, selected_text: str, explanation: str,
                                              top_k: int = 2) -> str:
        """
        Enrich the explanation with related textbook content as specified in the requirements

        Args:
            selected_text: The text that was selected by the user
            explanation: The initial explanation for the selected text
            top_k: Number of related content pieces to include for enrichment

        Returns:
            Enriched explanation that includes related content
        """
        try:
            # Import semantic search service to find related content
            from .semantic_search import SemanticSearchService

            # Create semantic search instance
            search_service = SemanticSearchService()

            # Search for content related to the selected text
            related_content = self.get_relevant_content_around_selection(selected_text, top_k)

            if related_content:
                enrichment_text = "\\n\\nAdditionally, related concepts from the textbook:\\n"
                for i, content in enumerate(related_content, 1):
                    enrichment_text += f"{i}. {content.get('content', '')[:200]}... "
                    if content.get('module_id') and content.get('chapter_id'):
                        enrichment_text += f"(See Module {content.get('module_id')}, Chapter {content.get('chapter_id')})\\n"

                enriched_explanation = f"{explanation}{enrichment_text}"
                return enriched_explanation
            else:
                # If no related content is found, return the original explanation
                return explanation
        except Exception as e:
            print(f"Error enriching explanation with related content: {e}")
            # On error, return the original explanation
            return explanation

    async def get_relevant_content_around_selection(self, selected_text: str,
                                            top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Find content that's contextually related to the selected text

        Args:
            selected_text: The text that was selected
            top_k: Number of related content pieces to return

        Returns:
            List of related content
        """
        try:
            # Import semantic search service to find related content
            from .semantic_search import SemanticSearchService

            # Create semantic search instance
            search_service = SemanticSearchService()

            # Search for content related to the selected text in the textbook
            results = await search_service.search(
                query=selected_text,
                top_k=top_k,
                min_score=0.5  # Minimum relevance threshold
            )

            return results
        except Exception as e:
            print(f"Error finding related content: {e}")
            return []
    
    def create_text_selection_session(self, student_id: UUID, initial_selected_text: str,
                                    module_context: str) -> Optional[Dict[str, Any]]:
        """
        Create a special session for text selection queries
        
        Args:
            student_id: The ID of the student
            initial_selected_text: The initially selected text
            module_context: The module the student is viewing
            
        Returns:
            Dictionary with session information
        """
        # First, create a standard session
        session_result = self.session_manager.create_session(student_id, module_context)
        if not session_result:
            return None
        
        session_id = session_result["id"]
        
        # Store the selected text
        stored_text = self.store_selected_text(
            content=initial_selected_text,
            module_id=module_context.split('/')[0] if module_context else "unknown",
            chapter_id=module_context.split('/')[1] if '/' in module_context else "unknown",
            section_id=module_context.split('/')[2] if len(module_context.split('/')) > 2 else "unknown",
            hierarchy_path=module_context
        )
        
        if not stored_text:
            # If we couldn't store the text, end the session and return None
            self.session_manager.end_session(session_id)
            return None
        
        # Add the selected text as the first "message" in the session
        message_result = self.session_manager.add_message_to_session(
            session_id=session_id,
            sender_type="student",
            content=f"Selected text for question: {initial_selected_text}",
            selected_text_ref=stored_text["id"]
        )
        
        return {
            "session_id": session_id,
            "selected_text_id": stored_text["id"],
            "selected_text": stored_text["content"],
            "module_context": module_context,
            "session_created": session_result,
            "message_added": message_result is not None
        }
    
    def search_for_similar_text_in_module(self, selected_text: str, module_id: str, 
                                        top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for text in the same module that is similar to the selected text
        
        Args:
            selected_text: The text that was selected
            module_id: The module to search within
            top_k: Number of results to return
            
        Returns:
            List of similar text content
        """
        # This would typically involve semantic search in the specific module
        # For now, we'll return an empty list but in a real implementation, 
        # this would search the textbook content for similar text in the specified module
        try:
            # In a real implementation, this would:
            # 1. Generate an embedding for the selected text
            # 2. Search for similar content within the specified module
            # 3. Return the top_k most similar content pieces
            
            # Placeholder implementation
            return []
        except Exception as e:
            print(f"Error searching for similar text in module: {e}")
            return []
    
    def get_selection_context(self, selected_text_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Get the full context of a selected text (module, chapter, section, etc.)
        
        Args:
            selected_text_id: The ID of the selected text
            
        Returns:
            Dictionary with context information
        """
        db = SessionLocal()
        try:
            selected_text = db.query(SelectedTextDB).filter(SelectedTextDB.id == selected_text_id).first()
            
            if not selected_text:
                return None
            
            return {
                "id": selected_text.id,
                "content": selected_text.content,
                "module_id": selected_text.module_id,
                "chapter_id": selected_text.chapter_id,
                "section_id": selected_text.section_id,
                "hierarchy_path": selected_text.hierarchy_path,
                "created_at": selected_text.created_at
            }
        except Exception as e:
            print(f"Error getting selection context: {e}")
            return None
        finally:
            db.close()
    
    def format_selected_text_response(self, response: str, selected_text: str) -> str:
        """
        Format the response to clearly indicate it's about the selected text
        
        Args:
            response: The AI-generated response
            selected_text: The text that was selected
            
        Returns:
            Formatted response string
        """
        formatted_response = (
            f"Based on the passage you selected: \"{selected_text[:100]}{'...' if len(selected_text) > 100 else ''}\", "
            f"from {selected_text.module_id if hasattr(selected_text, 'module_id') else 'the textbook'}: "
            f"{response}"
        )
        return formatted_response


class TextSelectionUIHelper:
    """
    Helper class for frontend text selection UI functionality
    """
    
    def calculate_selection_position(self, element_path: str, text_position: int) -> Dict[str, Any]:
        """
        Calculate the position of the text selection for UI display
        
        Args:
            element_path: CSS selector path to the selected element
            text_position: Position within the element where selection starts
            
        Returns:
            Dictionary with position information for UI
        """
        # This would calculate where to display the "Ask about this" button
        return {
            "element_path": element_path,
            "text_position": text_position,
            "display_position": "top-right"  # Default position for the UI element
        }
    
    def validate_selection_for_module(self, selected_text: str, module_id: str) -> Dict[str, Any]:
        """
        Validate selection specifically in the context of a module
        
        Args:
            selected_text: The selected text
            module_id: The module where text was selected
            
        Returns:
            Validation result
        """
        # Perform basic validation
        validation = {
            "is_valid": True,
            "module_appropriate": True,
            "character_count": len(selected_text),
            "suggestions": []
        }
        
        # Check if text is long enough
        if len(selected_text) < 20:
            validation["is_valid"] = False
            validation["suggestions"].append("Please select at least 20 characters")
        
        # In a real implementation, we might check if the selected text is relevant
        # to the current module's topic
        
        return validation


def create_text_selection_service() -> TextSelectionService:
    """
    Convenience function to create a text selection service instance
    """
    return TextSelectionService()


if __name__ == "__main__":
    # Example usage
    text_selection_service = TextSelectionService()
    
    # Note: This example would require an active database
    # The following is illustrative only
    sample_text = "The robot's movement is controlled by inverse kinematics which determines joint angles."
    validation = text_selection_service.validate_selected_text(sample_text)
    
    print(f"Text selection validation: {validation}")
    print("TextSelectionService created. Use with an active database connection.")