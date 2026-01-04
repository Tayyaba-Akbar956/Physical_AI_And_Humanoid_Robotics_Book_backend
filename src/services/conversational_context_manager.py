"""
Conversational Context Manager for RAG Chatbot
Handles multi-turn conversations, context window management, and follow-up question resolution
"""

from typing import Dict, List, Optional, Any
from uuid import UUID
from datetime import datetime
import re


class ConversationalContextManager:
    """
    Manages conversational context for multi-turn interactions
    Handles context window management (5-10 exchanges), follow-up question resolution,
    and topic tracking across conversations
    """

    def __init__(self, context_window_size: int = 10):
        """
        Initialize the conversational context manager
        
        Args:
            context_window_size: Maximum number of exchanges to keep in context (default 10)
        """
        self.context_window_size = context_window_size

    def get_conversation_context(self, session_id: UUID, num_exchanges: int = 5) -> List[Dict[str, Any]]:
        """
        Get a formatted conversation context for the RAG agent
        
        Args:
            session_id: The session ID to retrieve context for
            num_exchanges: Number of exchanges to retrieve
            
        Returns:
            List of formatted exchanges for context
        """
        from .session_manager import SessionManagementService
        session_manager = SessionManagementService()

        # Get the recent message history
        messages = session_manager.get_conversation_context(
            session_id=session_id, 
            num_messages=num_exchanges * 2  # 2 messages per exchange (student + AI)
        )

        # Format for context with roles and metadata
        formatted_context = []
        for msg in messages:
            formatted_context.append({
                "role": "user" if msg["sender_type"] == "student" else "assistant",
                "content": msg["content"],
                "timestamp": msg["timestamp"],
                "citations": msg.get("citations", []),
                "id": msg["id"]
            })

        return formatted_context

    def resolve_follow_up_question(self, query: str, conversation_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Resolve a follow-up question by identifying what it refers to in the conversation
        
        Args:
            query: The follow-up question
            conversation_context: Recent conversation context
            
        Returns:
            Dictionary with resolved context and metadata
        """
        # Identify follow-up indicators in the query
        follow_up_indicators = [
            "that", "this", "it", "the same", "similar", "different", "explain", "describe",
            "what was", "how does", "can you", "more about", "tell me", "me more",
            "about that", "about this", "like", "compared to", "vs", "versus", "than",
            "earlier", "previously", "above", "below", "mentioned", "again"
        ]

        resolution_result = {
            "is_follow_up": False,
            "resolved_query": query,
            "references": [],
            "referred_content": None,
            "resolution_confidence": 0.0,
            "resolution_steps": []
        }

        # Check for follow-up indicators in the query
        query_lower = query.lower()
        found_indicators = [indicator for indicator in follow_up_indicators if indicator in query_lower]
        
        if found_indicators:
            resolution_result["is_follow_up"] = True
            resolution_result["resolution_steps"].append(f"Found follow-up indicators: {', '.join(found_indicators)}")

        # If it's a follow-up, try to resolve what it refers to
        if resolution_result["is_follow_up"] and conversation_context:
            # Look for pronoun resolution (that, this, it)
            if any(indicator in query_lower for indicator in ["that", "this", "it"]):
                # Find the most recent substantial response from the AI
                for msg in reversed(conversation_context):
                    if msg["role"] == "assistant" and len(msg["content"]) > 50:
                        resolution_result["referred_content"] = msg["content"]
                        resolution_result["references"].append({
                            "id": msg["id"],
                            "type": "ai_response",
                            "timestamp": msg["timestamp"]
                        })
                        resolution_result["resolution_confidence"] = 0.8
                        resolution_result["resolution_steps"].append(
                            f"Resolved 'that/this/it' to AI response: {msg['content'][:50]}..."
                        )
                        resolution_result["resolved_query"] = f"Regarding the previous response: {query}"
                        break

            # Handle "explain that differently" or similar requests
            if any(phrase in query_lower for phrase in ["explain that differently", "explain differently", "different way", "another way"]):
                for msg in reversed(conversation_context):
                    if msg["role"] == "assistant" and len(msg["content"]) > 50:
                        resolution_result["referred_content"] = msg["content"]
                        resolution_result["references"].append({
                            "id": msg["id"],
                            "type": "ai_response",
                            "timestamp": msg["timestamp"]
                        })
                        resolution_result["resolution_confidence"] = 0.9
                        resolution_result["resolution_steps"].append(
                            f"Detected request for alternative explanation of: {msg['content'][:50]}..."
                        )
                        resolution_result["resolved_query"] = f"Can you explain the previous response differently: {query}"
                        break

        return resolution_result

    def manage_context_window(self, conversation_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Manage the context window to maintain optimal conversation context
        
        Args:
            conversation_history: Full conversation history
            
        Returns:
            Trimmed conversation history within context window
        """
        if len(conversation_history) <= self.context_window_size:
            return conversation_history

        # Keep the most recent exchanges within the window
        return conversation_history[-self.context_window_size:]

    def track_conversation_topic(self, conversation_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Track and identify the main topic of the conversation
        
        Args:
            conversation_context: The conversation context to analyze
            
        Returns:
            Dictionary with topic information
        """
        if not conversation_context:
            return {"current_topic": None, "topic_confidence": 0.0, "topic_keywords": []}

        # Combine all content to identify topics
        all_content = " ".join([msg["content"] for msg in conversation_context])
        all_content_lower = all_content.lower()

        # Define topic keywords for Physical AI & Humanoid Robotics
        topic_keywords = {
            "ros": ["ros", "ros2", "node", "topic", "service", "action", "publisher", "subscriber", "tf", "transform"],
            "simulation": ["gazebo", "unity", "simulation", "physics", "dynamics", "modeling"],
            "isaac": ["nvidia", "isaac", "navigation", "perception", "sensing"],
            "humanoid": ["humanoid", "walking", "balance", "bipedal", "motion", "actuator", "sensor"],
            "control": ["control", "controller", "pid", "trajectory", "motion planning", "ik", "fk"],
            "learning": ["machine learning", "neural", "algorithm", "training", "ai", "artificial intelligence"]
        }

        topic_scores = {}
        for topic, keywords in topic_keywords.items():
            score = 0
            for keyword in keywords:
                # Count occurrences of each keyword
                count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', all_content_lower))
                score += count
            topic_scores[topic] = score

        # Find the topic with the highest score
        if topic_scores:
            current_topic = max(topic_scores, key=topic_scores.get)
            max_score = topic_scores[current_topic]
            total_score = sum(topic_scores.values())

            if total_score > 0:
                topic_confidence = max_score / total_score
            else:
                topic_confidence = 0.0

            return {
                "current_topic": current_topic if max_score > 0 else None,
                "topic_confidence": topic_confidence,
                "topic_keywords": [topic for topic, score in topic_scores.items() if score > 0],
                "topic_scores": topic_scores
            }
        else:
            return {"current_topic": None, "topic_confidence": 0.0, "topic_keywords": []}

    def serialize_conversation_state(self, session_id: UUID) -> Dict[str, Any]:
        """
        Serialize the conversation state for storage or transmission
        
        Args:
            session_id: The session to serialize
            
        Returns:
            Dictionary with serialized conversation state
        """
        from .session_manager import SessionManagementService
        session_manager = SessionManagementService()

        # Get session info and recent context
        session_info = session_manager.get_session(session_id)
        conversation_context = self.get_conversation_context(session_id, num_exchanges=10)

        topic_info = self.track_conversation_topic(conversation_context)

        # Serialize to return as a compact representation
        serialized_state = {
            "session_id": str(session_id),
            "module_context": session_info.get("current_module_context") if session_info else None,
            "last_interaction": session_info.get("updated_at") if session_info else None,
            "conversation_depth": len(conversation_context) // 2,  # Assuming 2 messages per exchange
            "current_topic": topic_info["current_topic"],
            "topic_confidence": topic_info["topic_confidence"],
            "recent_exchanges_count": len(conversation_context) // 2,
            "serialized_context": [
                {
                    "role": msg["role"],
                    "timestamp": msg["timestamp"].isoformat() if isinstance(msg["timestamp"], datetime) else str(msg["timestamp"]),
                    "content_preview": msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                }
                for msg in conversation_context[-4:]  # Last 4 messages as preview
            ]
        }

        return serialized_state

    def update_conversation_context_in_db(self, session_id: UUID, query: str, response: str) -> bool:
        """
        Update the conversation context in the database after each exchange
        
        Args:
            session_id: The session ID
            query: The user's query
            response: The AI's response
            
        Returns:
            True if update was successful, False otherwise
        """
        from .session_manager import SessionManagementService
        session_manager = SessionManagementService()

        # Get current conversation context
        conversation_context = self.get_conversation_context(session_id, num_exchanges=5)

        # Identify the current topic
        topic_info = self.track_conversation_topic(conversation_context)
        current_topic = topic_info["current_topic"]

        # Update topic in session
        from ..models.chat_session import ChatSessionUpdate
        update_result = session_manager.update_conversation_context(session_id, current_topic)

        return update_result

    def validate_conversation_state(self, session_id: UUID) -> Dict[str, Any]:
        """
        Validate the current conversation state for consistency
        
        Args:
            session_id: The session ID to validate
            
        Returns:
            Dictionary with validation results
        """
        from .session_manager import SessionManagementService
        session_manager = SessionManagementService()

        validation_result = {
            "session_exists": False,
            "session_active": False,
            "has_conversation_context": False,
            "context_length_valid": False,
            "topic_consistency": True,
            "validation_details": []
        }

        # Check if session exists and is active
        session_info = session_manager.get_session(session_id)
        if session_info:
            validation_result["session_exists"] = True
            validation_result["session_active"] = session_info["is_active"]
            
            if session_info["is_active"]:
                # Check if there's conversation context
                conversation_context = self.get_conversation_context(session_id)
                validation_result["has_conversation_context"] = len(conversation_context) > 0
                validation_result["context_length_valid"] = len(conversation_context) <= self.context_window_size

                # Check topic consistency
                topic_info = self.track_conversation_topic(conversation_context)
                if topic_info["topic_confidence"] < 0.3:
                    validation_result["topic_consistency"] = False
                    validation_result["validation_details"].append(
                        "Low topic consistency detected - conversation may be drifting"
                    )

        return validation_result


# Singleton instance
conversational_context_manager = ConversationalContextManager()


def get_conversational_context_manager() -> ConversationalContextManager:
    """
    Get the singleton instance of the conversational context manager
    """
    return conversational_context_manager