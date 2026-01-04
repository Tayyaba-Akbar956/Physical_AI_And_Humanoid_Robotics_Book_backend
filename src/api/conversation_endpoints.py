from datetime import datetime
from fastapi import APIRouter, HTTPException
from typing import Optional, Dict, Any, List
from uuid import UUID, uuid4
from pydantic import BaseModel
import json
from ..services.session_manager import SessionManagementService
from ..services.conversational_context_manager import get_conversational_context_manager


# Create API router
router = APIRouter(prefix="/api/conversation", tags=["conversation"])


class ConversationHistoryResponse(BaseModel):
    """
    Response model for conversation history
    """
    session_id: str
    messages: List[Dict[str, Any]]
    total_messages: int
    total_exchanges: int
    start_time: Optional[str] = None
    last_interaction: Optional[str] = None


class ConversationContextResponse(BaseModel):
    """
    Response model for conversation context
    """
    session_id: str
    context_messages: List[Dict[str, Any]]
    current_topic: Optional[str] = None
    topic_confidence: Optional[float] = None
    conversation_depth: int = 0


class ConversationStateValidationResponse(BaseModel):
    """
    Response model for conversation state validation
    """
    session_id: str
    is_valid: bool
    validation_results: Dict[str, Any]


@router.get("/history/{session_id}", response_model=ConversationHistoryResponse)
async def get_conversation_history(session_id: str):
    """
    Retrieve the full conversation history for a session
    """
    try:
        # Convert session_id to UUID
        try:
            uuid_session_id = UUID(session_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid session ID format")

        session_manager = SessionManagementService()

        # Get session info to confirm session exists and is active
        session_info = session_manager.get_session(uuid_session_id)
        if not session_info:
            raise HTTPException(status_code=404, detail="Session not found")

        # Get the conversation history
        history = session_manager.get_conversation_history(uuid_session_id, include_context=True)

        # Get summary for additional metadata
        summary = session_manager.get_conversation_summary(uuid_session_id)

        response = ConversationHistoryResponse(
            session_id=session_id,
            messages=history,
            total_messages=summary.get("total_messages", 0),
            total_exchanges=summary.get("total_exchanges", 0),
            start_time=summary.get("start_time").isoformat() if summary.get("start_time") else None,
            last_interaction=summary.get("last_interaction").isoformat() if summary.get("last_interaction") else None
        )

        return response

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"Error getting conversation history: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/context/{session_id}", response_model=ConversationContextResponse)
async def get_conversation_context(session_id: str):
    """
    Retrieve the recent conversation context for a session
    """
    try:
        # Convert session_id to UUID
        try:
            uuid_session_id = UUID(session_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid session ID format")

        session_manager = SessionManagementService()
        context_manager = get_conversational_context_manager()

        # Get session info to confirm session exists and is active
        session_info = session_manager.get_session(uuid_session_id)
        if not session_info:
            raise HTTPException(status_code=404, detail="Session not found")

        # Get the recent conversation context
        context = session_manager.get_conversation_context(uuid_session_id, num_messages=10)

        # Get topic information using the conversational context manager
        topic_info = context_manager.track_conversation_topic(context)
        conversation_depth = session_manager.get_conversation_depth(uuid_session_id)

        response = ConversationContextResponse(
            session_id=session_id,
            context_messages=context,
            current_topic=topic_info.get("current_topic"),
            topic_confidence=topic_info.get("topic_confidence"),
            conversation_depth=conversation_depth
        )

        return response

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"Error getting conversation context: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/state-validation/{session_id}", response_model=ConversationStateValidationResponse)
async def validate_conversation_state(session_id: str):
    """
    Validate the current conversation state for consistency
    """
    try:
        # Convert session_id to UUID
        try:
            uuid_session_id = UUID(session_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid session ID format")

        session_manager = SessionManagementService()

        # Validate conversation state
        validation_results = session_manager.validate_conversation_state(uuid_session_id)

        response = ConversationStateValidationResponse(
            session_id=session_id,
            is_valid=all([
                validation_results["session_exists"],
                validation_results["session_active"],
                validation_results["has_conversation_context"]
            ]),
            validation_results=validation_results
        )

        return response

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"Error validating conversation state: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/summary/{session_id}")
async def get_conversation_summary(session_id: str):
    """
    Get a summary of the conversation for context tracking
    """
    try:
        # Convert session_id to UUID
        try:
            uuid_session_id = UUID(session_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid session ID format")

        session_manager = SessionManagementService()

        # Get session info to confirm session exists and is active
        session_info = session_manager.get_session(uuid_session_id)
        if not session_info:
            raise HTTPException(status_code=404, detail="Session not found")

        # Get the conversation summary
        summary = session_manager.get_conversation_summary(uuid_session_id)

        return summary

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"Error getting conversation summary: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/reset-context/{session_id}")
async def reset_conversation_context(session_id: str):
    """
    Reset the conversation context, starting fresh while preserving session
    """
    try:
        # Convert session_id to UUID
        try:
            uuid_session_id = UUID(session_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid session ID format")

        session_manager = SessionManagementService()

        # Get session info to confirm session exists and is active
        session_info = session_manager.get_session(uuid_session_id)
        if not session_info:
            raise HTTPException(status_code=404, detail="Session not found")

        # Reset the conversation context while preserving the session
        # This involves clearing the active topic and resetting related fields
        from ..models.chat_session import ChatSessionUpdate
        update_result = session_manager.update_session(
            uuid_session_id,
            ChatSessionUpdate(
                active_topic=None,
                conversation_context=None,
                conversation_depth=0
            )
        )

        if not update_result:
            raise HTTPException(status_code=500, detail="Failed to reset conversation context")

        # Also clear the conversation history properly but keep the session
        clear_result = session_manager.clear_session_history(uuid_session_id)
        if not clear_result:
            raise HTTPException(status_code=500, detail="Failed to clear conversation history")

        return {
            "message": "Conversation context reset successfully",
            "session_id": session_id
        }

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"Error resetting conversation context: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Health check endpoint
@router.get("/health")
async def health_check():
    """
    Health check endpoint for the conversation management service
    """
    return {
        "status": "healthy",
        "service": "conversation",
        "timestamp": str(datetime.now())
    }