from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4
from sqlalchemy.orm import Session
from ..db.connection import get_db, SessionLocal
from ..models.chat_session import ChatSessionDB, ChatSessionCreate, ChatSessionUpdate
from ..models.message import MessageDB, MessageCreate
from ..models.student import StudentDB
import json
import logging


class SessionManagementService:
    """
    Service for managing chat sessions and conversation history
    Handles session creation, retrieval, updates, and message management
    """
    
    def __init__(self):
        """
        Initialize the session management service
        """
        pass
    
    def create_session(self, student_id: UUID, current_module_context: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Create a new chat session for a student

        Args:
            student_id: The ID of the student starting the session
            current_module_context: The module the student is currently viewing

        Returns:
            Dictionary with session information or None if failed
        """
        logger = logging.getLogger("rag_chatbot")
        db = SessionLocal()
        try:
            # Verify student exists
            student = db.query(StudentDB).filter(StudentDB.id == student_id).first()
            if not student:
                logger.warning(f"Student with ID {student_id} not found")
                return None

            # Create new session
            session_data = ChatSessionCreate(
                student_id=student_id,
                current_module_context=current_module_context
            )

            db_session = ChatSessionDB(
                student_id=session_data.student_id,
                current_module_context=session_data.current_module_context,
                session_metadata={}
            )

            db.add(db_session)
            db.commit()
            db.refresh(db_session)

            logger.info(f"Created new session {db_session.id} for student {student_id}, module {current_module_context}")

            return {
                "id": db_session.id,
                "student_id": db_session.student_id,
                "created_at": db_session.created_at,
                "current_module_context": db_session.current_module_context,
                "is_active": db_session.is_active,
                "session_metadata": db_session.session_metadata
            }
        except Exception as e:
            logger.error(f"Error creating session for student {student_id}: {e}")
            db.rollback()
            return None
        finally:
            db.close()
    
    def get_session(self, session_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific chat session
        
        Args:
            session_id: The ID of the session to retrieve
            
        Returns:
            Dictionary with session information or None if not found
        """
        db = SessionLocal()
        try:
            db_session = db.query(ChatSessionDB).filter(ChatSessionDB.id == session_id).first()
            
            if not db_session:
                return None
            
            return {
                "id": db_session.id,
                "student_id": db_session.student_id,
                "created_at": db_session.created_at,
                "updated_at": db_session.updated_at,
                "current_module_context": db_session.current_module_context,
                "is_active": db_session.is_active,
                "session_metadata": db_session.session_metadata
            }
        except Exception as e:
            print(f"Error retrieving session: {e}")
            return None
        finally:
            db.close()
    
    def update_session(self, session_id: UUID, update_data: ChatSessionUpdate) -> bool:
        """
        Update a chat session with new information
        
        Args:
            session_id: The ID of the session to update
            update_data: The data to update in the session
            
        Returns:
            True if update was successful, False otherwise
        """
        db = SessionLocal()
        try:
            db_session = db.query(ChatSessionDB).filter(ChatSessionDB.id == session_id).first()
            
            if not db_session:
                return False
            
            # Update fields if they are provided in update_data
            if update_data.current_module_context is not None:
                db_session.current_module_context = update_data.current_module_context
            if update_data.is_active is not None:
                db_session.is_active = update_data.is_active
            if update_data.session_metadata is not None:
                db_session.session_metadata = update_data.session_metadata
            
            # Update the updated_at timestamp
            db_session.updated_at = datetime.utcnow()
            
            db.commit()
            return True
        except Exception as e:
            print(f"Error updating session: {e}")
            db.rollback()
            return False
        finally:
            db.close()
    
    def end_session(self, session_id: UUID) -> bool:
        """
        End a chat session by setting its active status to False
        
        Args:
            session_id: The ID of the session to end
            
        Returns:
            True if session was ended successfully, False otherwise
        """
        update_data = ChatSessionUpdate(is_active=False)
        return self.update_session(session_id, update_data)
    
    def get_student_sessions(self, student_id: UUID, active_only: bool = False) -> List[Dict[str, Any]]:
        """
        Get all sessions for a particular student
        
        Args:
            student_id: The ID of the student
            active_only: Whether to return only active sessions
            
        Returns:
            List of session dictionaries
        """
        db = SessionLocal()
        try:
            query = db.query(ChatSessionDB).filter(ChatSessionDB.student_id == student_id)
            
            if active_only:
                query = query.filter(ChatSessionDB.is_active == True)
            
            db_sessions = query.all()
            
            sessions = []
            for db_session in db_sessions:
                sessions.append({
                    "id": db_session.id,
                    "student_id": db_session.student_id,
                    "created_at": db_session.created_at,
                    "updated_at": db_session.updated_at,
                    "current_module_context": db_session.current_module_context,
                    "is_active": db_session.is_active,
                    "session_metadata": db_session.session_metadata
                })
            
            return sessions
        except Exception as e:
            print(f"Error retrieving student sessions: {e}")
            return []
        finally:
            db.close()
    
    def add_message_to_session(self, session_id: UUID, sender_type: str, content: str,
                              citations: Optional[List[Dict[str, str]]] = None,
                              selected_text_ref: Optional[UUID] = None,
                              parent_message_id: Optional[UUID] = None,
                              topic_anchored: Optional[str] = None,
                              follow_up_to: Optional[UUID] = None) -> Optional[Dict[str, Any]]:
        """
        Add a message to a chat session with conversation context

        Args:
            session_id: The ID of the session to add the message to
            sender_type: The type of sender ('student' or 'ai_agent')
            content: The content of the message
            citations: Any citations to textbook content
            selected_text_ref: Reference to selected text if this is a text selection query
            parent_message_id: ID of parent message this is responding to
            topic_anchored: Topic this message is anchored to
            follow_up_to: ID of message this is a follow-up to

        Returns:
            Dictionary with message information or None if failed
        """
        db = SessionLocal()
        try:
            # Verify session exists and is active
            db_session = db.query(ChatSessionDB).filter(
                ChatSessionDB.id == session_id,
                ChatSessionDB.is_active == True
            ).first()

            if not db_session:
                print(f"Session with ID {session_id} not found or not active")
                return None

            # Calculate the conversation turn (position in conversation)
            conversation_turn = db.query(MessageDB).filter(MessageDB.session_id == session_id).count()

            # Create message
            message_data = MessageCreate(
                session_id=session_id,
                sender_type=sender_type,
                content=content,
                citations=citations,
                selected_text_ref=selected_text_ref,
                conversation_turn=conversation_turn,
                parent_message_id=parent_message_id,
                topic_anchored=topic_anchored,
                follow_up_to=follow_up_to
            )

            db_message = MessageDB(
                session_id=message_data.session_id,
                sender_type=message_data.sender_type,
                content=message_data.content,
                citations=message_data.citations,
                selected_text_ref=message_data.selected_text_ref,
                conversation_turn=message_data.conversation_turn,
                parent_message_id=message_data.parent_message_id,
                topic_anchored=message_data.topic_anchored,
                follow_up_to=message_data.follow_up_to
            )

            db.add(db_message)
            db.commit()
            db.refresh(db_message)

            # Update the session's conversation context
            self.update_conversation_context(session_id, topic_anchored)

            return {
                "id": db_message.id,
                "session_id": db_message.session_id,
                "sender_type": db_message.sender_type,
                "content": db_message.content,
                "timestamp": db_message.timestamp,
                "citations": db_message.citations,
                "selected_text_ref": db_message.selected_text_ref,
                "conversation_turn": db_message.conversation_turn,
                "parent_message_id": db_message.parent_message_id,
                "topic_anchored": db_message.topic_anchored,
                "follow_up_to": db_message.follow_up_to
            }
        except Exception as e:
            print(f"Error adding message to session: {e}")
            db.rollback()
            return None
        finally:
            db.close()
    
    def get_session_messages(self, session_id: UUID, limit: Optional[int] = 50) -> List[Dict[str, Any]]:
        """
        Retrieve all messages for a chat session
        
        Args:
            session_id: The ID of the session
            limit: Maximum number of messages to return (None for all)
            
        Returns:
            List of message dictionaries
        """
        db = SessionLocal()
        try:
            query = db.query(MessageDB).filter(MessageDB.session_id == session_id).order_by(MessageDB.timestamp.asc())
            
            if limit:
                db_messages = query.limit(limit).all()
            else:
                db_messages = query.all()
            
            messages = []
            for db_message in db_messages:
                messages.append({
                    "id": db_message.id,
                    "session_id": db_message.session_id,
                    "sender_type": db_message.sender_type,
                    "content": db_message.content,
                    "timestamp": db_message.timestamp,
                    "citations": db_message.citations,
                    "selected_text_ref": db_message.selected_text_ref
                })
            
            return messages
        except Exception as e:
            print(f"Error retrieving session messages: {e}")
            return []
        finally:
            db.close()
    
    def clear_session_history(self, session_id: UUID) -> bool:
        """
        Clear all messages from a session while keeping the session itself
        
        Args:
            session_id: The ID of the session to clear
            
        Returns:
            True if history was cleared successfully, False otherwise
        """
        db = SessionLocal()
        try:
            # Delete all messages associated with the session
            db.query(MessageDB).filter(MessageDB.session_id == session_id).delete()
            db.commit()
            
            # Update the session as if it just started
            db_session = db.query(ChatSessionDB).filter(ChatSessionDB.id == session_id).first()
            if db_session:
                db_session.updated_at = datetime.utcnow()
                db.commit()
                return True
            else:
                return False
        except Exception as e:
            print(f"Error clearing session history: {e}")
            db.rollback()
            return False
        finally:
            db.close()
    
    def get_conversation_context(self, session_id: UUID, num_messages: int = 10) -> List[Dict[str, Any]]:
        """
        Get the recent conversation context for a session

        Args:
            session_id: The ID of the session
            num_messages: Number of recent messages to include

        Returns:
            List of the most recent messages in the session
        """
        db = SessionLocal()
        try:
            # Get the most recent messages from the session
            db_messages = db.query(MessageDB)\
                           .filter(MessageDB.session_id == session_id)\
                           .order_by(MessageDB.timestamp.desc())\
                           .limit(num_messages)\
                           .all()

            # Reverse the list to get chronological order (oldest first)
            context = []
            for db_message in reversed(db_messages):
                context.append({
                    "id": db_message.id,
                    "sender_type": db_message.sender_type,
                    "content": db_message.content,
                    "timestamp": db_message.timestamp,
                    "citations": db_message.citations,
                    "selected_text_ref": db_message.selected_text_ref,
                    "conversation_turn": db_message.conversation_turn,
                    "parent_message_id": db_message.parent_message_id,
                    "topic_anchored": db_message.topic_anchored,
                    "follow_up_to": db_message.follow_up_to
                })

            return context
        except Exception as e:
            print(f"Error retrieving conversation context: {e}")
            return []
        finally:
            db.close()


    def get_conversation_history(self, session_id: UUID, include_context: bool = True) -> List[Dict[str, Any]]:
        """
        Get the full or partial conversation history with additional context information

        Args:
            session_id: The ID of the session
            include_context: Whether to include extended context information

        Returns:
            List of all messages in the session with or without extended context
        """
        db = SessionLocal()
        try:
            # Get all messages from the session ordered chronologically
            db_messages = db.query(MessageDB)\
                           .filter(MessageDB.session_id == session_id)\
                           .order_by(MessageDB.timestamp.asc())\
                           .all()

            history = []
            for db_message in db_messages:
                message_dict = {
                    "id": db_message.id,
                    "session_id": db_message.session_id,
                    "sender_type": db_message.sender_type,
                    "content": db_message.content,
                    "timestamp": db_message.timestamp,
                    "citations": db_message.citations,
                    "selected_text_ref": db_message.selected_text_ref
                }

                if include_context:
                    message_dict.update({
                        "conversation_turn": db_message.conversation_turn,
                        "parent_message_id": db_message.parent_message_id,
                        "topic_anchored": db_message.topic_anchored,
                        "follow_up_to": db_message.follow_up_to
                    })

                history.append(message_dict)

            return history
        except Exception as e:
            print(f"Error retrieving conversation history: {e}")
            return []
        finally:
            db.close()


    def update_conversation_topic(self, session_id: UUID, topic: str) -> bool:
        """
        Update the active topic in a conversation session

        Args:
            session_id: The ID of the session
            topic: The new topic to set

        Returns:
            True if the topic was updated successfully, False otherwise
        """
        db = SessionLocal()
        try:
            db_session = db.query(ChatSessionDB).filter(ChatSessionDB.id == session_id).first()

            if not db_session:
                return False

            db_session.active_topic = topic
            db_session.updated_at = datetime.utcnow()

            db.commit()
            return True
        except Exception as e:
            print(f"Error updating conversation topic: {e}")
            db.rollback()
            return False
        finally:
            db.close()


    def get_active_conversation_topic(self, session_id: UUID) -> Optional[str]:
        """
        Get the currently active topic in a conversation session

        Args:
            session_id: The ID of the session

        Returns:
            The active topic as a string, or None if not set
        """
        db = SessionLocal()
        try:
            db_session = db.query(ChatSessionDB).filter(ChatSessionDB.id == session_id).first()

            if not db_session:
                return None

            return db_session.active_topic
        except Exception as e:
            print(f"Error retrieving conversation topic: {e}")
            return None
        finally:
            db.close()


    def get_conversation_depth(self, session_id: UUID) -> int:
        """
        Get the depth (number of exchanges) in a conversation

        Args:
            session_id: The ID of the session

        Returns:
            The number of exchanges in the conversation
        """
        db = SessionLocal()
        try:
            # Count total messages and divide by 2 to get number of exchanges (assuming each exchange has a student and AI message)
            total_messages = db.query(MessageDB)\
                              .filter(MessageDB.session_id == session_id)\
                              .count()

            return total_messages // 2 if total_messages > 0 else 0
        except Exception as e:
            print(f"Error retrieving conversation depth: {e}")
            return 0
        finally:
            db.close()


    def validate_conversation_state(self, session_id: UUID) -> Dict[str, Any]:
        """
        Validate the current conversation state for consistency

        Args:
            session_id: The session ID to validate

        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "session_exists": False,
            "session_active": False,
            "has_conversation_context": False,
            "context_length_valid": False,
            "validation_details": []
        }

        # Check if session exists and is active
        session_info = self.get_session(session_id)
        if session_info:
            validation_result["session_exists"] = True
            validation_result["session_active"] = session_info["is_active"]

            if session_info["is_active"]:
                # Check if there's conversation context
                conversation_context = self.get_conversation_context(session_id)
                validation_result["has_conversation_context"] = len(conversation_context) > 0
                validation_result["context_length_valid"] = len(conversation_context) <= 20  # Arbitrary limit

        return validation_result

    def get_full_conversation_history(self, session_id: UUID) -> List[Dict[str, Any]]:
        """
        Get the complete conversation history for a session

        Args:
            session_id: The ID of the session

        Returns:
            List of all messages in the session in chronological order
        """
        db = SessionLocal()
        try:
            # Get all messages from the session ordered chronologically
            db_messages = db.query(MessageDB)\
                           .filter(MessageDB.session_id == session_id)\
                           .order_by(MessageDB.timestamp.asc())\
                           .all()

            history = []
            for db_message in db_messages:
                history.append({
                    "id": db_message.id,
                    "sender_type": db_message.sender_type,
                    "content": db_message.content,
                    "timestamp": db_message.timestamp,
                    "citations": db_message.citations,
                    "selected_text_ref": db_message.selected_text_ref
                })

            return history
        except Exception as e:
            print(f"Error retrieving full conversation history: {e}")
            return []
        finally:
            db.close()

    def get_conversation_summary(self, session_id: UUID) -> Dict[str, Any]:
        """
        Get a summary of the conversation for context tracking

        Args:
            session_id: The ID of the session

        Returns:
            Dictionary with conversation summary information
        """
        db = SessionLocal()
        try:
            # Get session info to include in summary
            db_session = db.query(ChatSessionDB).filter(ChatSessionDB.id == session_id).first()
            if not db_session:
                return {}

            # Count total messages
            total_messages = db.query(MessageDB)\
                              .filter(MessageDB.session_id == session_id)\
                              .count()

            # Get first and last message timestamps
            first_message = db.query(MessageDB)\
                             .filter(MessageDB.session_id == session_id)\
                             .order_by(MessageDB.timestamp.asc())\
                             .first()

            last_message = db.query(MessageDB)\
                            .filter(MessageDB.session_id == session_id)\
                            .order_by(MessageDB.timestamp.desc())\
                            .first()

            # Identify the main topic if available
            main_topic = db_session.active_topic or self._infer_topic_from_messages(session_id)

            return {
                "session_id": session_id,
                "total_exchanges": total_messages // 2 if total_messages > 0 else 0,  # Assuming each exchange has 2 messages (student + AI)
                "total_messages": total_messages,
                "start_time": first_message.timestamp if first_message else None,
                "last_interaction": last_message.timestamp if last_message else None,
                "current_topic": main_topic,
                "module_context": db_session.current_module_context,
                "active": db_session.is_active
            }
        except Exception as e:
            print(f"Error retrieving conversation summary: {e}")
            return {}
        finally:
            db.close()

    def get_module_context_history(self, student_id: UUID, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the module context history for a student across sessions

        Args:
            student_id: The ID of the student
            limit: Number of recent sessions to include

        Returns:
            List of session info with module context
        """
        db = SessionLocal()
        try:
            # Get recent sessions for the student
            db_sessions = db.query(ChatSessionDB)\
                           .filter(ChatSessionDB.student_id == student_id)\
                           .order_by(ChatSessionDB.updated_at.desc())\
                           .limit(limit)\
                           .all()

            history = []
            for db_session in db_sessions:
                # Count messages in the session to determine engagement
                message_count = db.query(MessageDB)\
                                 .filter(MessageDB.session_id == db_session.id)\
                                 .count()

                history.append({
                    "session_id": str(db_session.id),
                    "module_context": db_session.current_module_context,
                    "created_at": db_session.created_at,
                    "updated_at": db_session.updated_at,
                    "is_active": db_session.is_active,
                    "message_count": message_count,
                    "conversation_depth": db_session.conversation_depth,
                    "active_topic": db_session.active_topic
                })

            return history
        except Exception as e:
            print(f"Error retrieving module context history: {e}")
            return []
        finally:
            db.close()

    def update_session_module_context(self, session_id: UUID, module_id: str) -> bool:
        """
        Update the module context for a session with validation

        Args:
            session_id: The ID of the session
            module_id: The new module ID

        Returns:
            True if the update was successful, False otherwise
        """
        try:
            # Get the current module context to log the change
            current_session = self.get_session(session_id)
            old_module = current_session.get("current_module_context") if current_session else None

            from ..models.chat_session import ChatSessionUpdate
            update_result = self.update_session(
                session_id,
                ChatSessionUpdate(current_module_context=module_id)
            )

            # Log the module context switch if it changed
            if update_result and old_module != module_id:
                self.log_module_context_switch(session_id, old_module, module_id)

            return update_result
        except Exception as e:
            print(f"Error updating session module context: {e}")
            return False

    def log_module_context_switch(self, session_id: UUID, old_module: str, new_module: str):
        """
        Log when a session's module context is switched

        Args:
            session_id: The session ID
            old_module: The previous module
            new_module: The new module
        """
        import logging
        logger = logging.getLogger("rag_chatbot")
        logger.info(f"MODULE_CONTEXT_SWITCH: Session {session_id} switched from '{old_module}' to '{new_module}'")

    def _infer_topic_from_messages(self, session_id: UUID, num_messages: int = 5) -> Optional[str]:
        """
        Infer the current topic from recent messages in the conversation

        Args:
            session_id: The ID of the session
            num_messages: Number of recent messages to analyze

        Returns:
            Inferred topic or None
        """
        context = self.get_conversation_context(session_id, num_messages)
        if not context:
            return None

        # Simple keyword-based topic detection
        # In a more advanced implementation, this could use NLP techniques
        all_content = " ".join([msg["content"] for msg in context])

        # Look for common robotics/AI terms that might indicate the topic
        topic_indicators = {
            "ros": ["ros", "ros2", "node", "topic", "service", "action"],
            "simulation": ["gazebo", "unity", "simulation", "physics", "dynamics"],
            "isaac": ["nvidia", "isaac", "navigation", "perception"],
            "humanoid": ["humanoid", "walking", "balance", "bipedal"],
            "control": ["control", "controller", "pid", "trajectory", "motion planning"]
        }

        all_content_lower = all_content.lower()
        detected_topics = []

        for topic, keywords in topic_indicators.items():
            for keyword in keywords:
                if keyword in all_content_lower:
                    detected_topics.append(topic)
                    break  # Don't count the same topic multiple times

        return ", ".join(detected_topics) if detected_topics else None

    def update_conversation_context(self, session_id: UUID, topic: Optional[str] = None) -> bool:
        """
        Update the conversation context for a session

        Args:
            session_id: The ID of the session
            topic: The current topic being discussed

        Returns:
            True if update was successful, False otherwise
        """
        db = SessionLocal()
        try:
            db_session = db.query(ChatSessionDB).filter(ChatSessionDB.id == session_id).first()

            if not db_session:
                return False

            # Update topic if provided
            if topic is not None:
                db_session.active_topic = topic

            # Update the last interaction time
            db_session.last_interaction_at = datetime.utcnow()

            # Update conversation depth based on message count
            message_count = db.query(MessageDB)\
                             .filter(MessageDB.session_id == session_id)\
                             .count()
            db_session.conversation_depth = message_count // 2  # Assuming each exchange is 2 messages

            db.commit()
            return True
        except Exception as e:
            print(f"Error updating conversation context: {e}")
            db.rollback()
            return False
        finally:
            db.close()
    
    def is_session_active(self, session_id: UUID) -> bool:
        """
        Check if a session is currently active
        
        Args:
            session_id: The ID of the session to check
            
        Returns:
            True if session is active, False otherwise
        """
        session_info = self.get_session(session_id)
        return session_info is not None and session_info["is_active"] if session_info else False
    
    def get_session_module_context(self, session_id: UUID) -> Optional[str]:
        """
        Get the current module context for a session
        
        Args:
            session_id: The ID of the session
            
        Returns:
            The current module context or None if not found
        """
        session_info = self.get_session(session_id)
        return session_info["current_module_context"] if session_info else None


class ConversationContextManager:
    """
    Higher-level manager for handling conversation context and history
    """
    
    def __init__(self):
        self.session_service = SessionManagementService()
    
    def get_recent_context(self, session_id: UUID) -> List[Dict[str, str]]:
        """
        Get the recent conversation context formatted for the LLM
        """
        messages = self.session_service.get_conversation_context(session_id, num_messages=10)
        
        # Format for LLM context
        context = []
        for msg in messages:
            context.append({
                "role": "user" if msg["sender_type"] == "student" else "assistant",
                "content": msg["content"]
            })
        
        return context
    
    def update_module_context(self, session_id: UUID, module_id: str) -> bool:
        """
        Update the module context for a session
        """
        update_data = ChatSessionUpdate(current_module_context=module_id)
        return self.session_service.update_session(session_id, update_data)


def create_session_manager() -> SessionManagementService:
    """
    Convenience function to create a session management service instance
    """
    return SessionManagementService()


if __name__ == "__main__":
    # Example usage
    session_manager = SessionManagementService()
    
    # Note: This example would require an active database with a student record
    # The following is illustrative only
    print("SessionManagementService created. Use with an active database connection.")