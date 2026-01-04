"""
API endpoints for text selection-based queries

This module implements the text selection detection and processing functionality
for the RAG Chatbot as specified in User Story 2.
"""
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from typing import Optional, Dict, Any, List
import json
from uuid import UUID, uuid4
from pydantic import BaseModel, Field
import asyncio
import logging

from ..services.rag_agent import RAGAgentService
from ..services.session_manager import SessionManagementService
from ..services.text_selection import TextSelectionService
from ..db.connection import get_db


# Global service instances (initially None)
_rag_agent_instance = None
_session_manager_instance = None
_text_selection_instance = None

def get_rag_agent_service() -> RAGAgentService:
    global _rag_agent_instance
    if _rag_agent_instance is None:
        _rag_agent_instance = RAGAgentService()
    return _rag_agent_instance

def get_session_manager_service() -> SessionManagementService:
    global _session_manager_instance
    if _session_manager_instance is None:
        _session_manager_instance = SessionManagementService()
    return _session_manager_instance

def get_text_selection_service() -> TextSelectionService:
    global _text_selection_instance
    if _text_selection_instance is None:
        _text_selection_instance = TextSelectionService()
    return _text_selection_instance

# Create API router
router = APIRouter(prefix="/api/text-selection", tags=["text-selection"])


class TextSelectionRequest(BaseModel):
    """
    Request model for text selection queries
    """
    session_id: str  # Will be converted to UUID in endpoint
    selected_text: str = Field(..., min_length=20, max_length=5000)
    question: str = Field(..., min_length=1, max_length=2000)
    module_id: str
    chapter_id: str
    element_path: Optional[str] = None  # CSS selector path to the selected element


class TextSelectionResponse(BaseModel):
    """
    Response model for text selection queries
    """
    session_id: str
    response_id: str
    message: str
    citations: List[Dict[str, str]]
    timestamp: str
    validation_result: Dict[str, Any]


class TextSelectionValidationResponse(BaseModel):
    """
    Response model for text selection validation
    """
    validation_result: str
    character_count: int
    can_ask_query: bool
    suggestions: Optional[List[str]] = None


@router.post("/detect", response_model=TextSelectionValidationResponse)
async def detect_text_selection(request: TextSelectionRequest):
    """
    Detect and validate text selection on the textbook page for the "Ask about this" interface
    """
    try:
        # Validate the API contract - ensure all required fields are present
        validation_errors = []

        if not request.selected_text or len(request.selected_text.strip()) < 20:
            validation_errors.append("Selected text must be at least 20 characters")

        if not request.question or len(request.question.strip()) < 1:
            validation_errors.append("Question cannot be empty")

        if not request.module_id:
            validation_errors.append("Module ID is required")

        if not request.chapter_id:
            validation_errors.append("Chapter ID is required")

        if validation_errors:
            return TextSelectionValidationResponse(
                validation_result="invalid",
                character_count=len(request.selected_text) if request.selected_text else 0,
                can_ask_query=False,
                suggestions=validation_errors
            )

        # Validate the selected text using the service
        validation = get_text_selection_service().validate_selected_text(request.selected_text)

        return TextSelectionValidationResponse(
            validation_result="valid" if validation["is_valid"] else "invalid",
            character_count=validation["character_count"],
            can_ask_query=validation["can_ask_query"],
            suggestions=validation.get("suggestions")
        )
    except Exception as e:
        print(f"Error in text selection detection: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/query", response_model=TextSelectionResponse)
async def query_selected_text(request: TextSelectionRequest):
    """
    Handle queries based on selected text from the textbook
    """
    try:
        # First, validate the session exists and is active
        try:
            session_id = UUID(request.session_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid session ID format")
        
        session_info = get_session_manager_service().get_session(session_id)
        if not session_info:
            raise HTTPException(status_code=404, detail="Session not found")
        if not session_info["is_active"]:
            raise HTTPException(status_code=400, detail="Session is not active")
        
        # Store the selected text
        stored_text = get_text_selection_service().store_selected_text(
            content=request.selected_text,
            module_id=request.module_id,
            chapter_id=request.chapter_id,
            section_id=request.element_path or "unknown",
            hierarchy_path=f"{request.module_id}/{request.chapter_id}/{request.element_path or 'section'}"
        )
        
        if not stored_text:
            raise HTTPException(status_code=500, detail="Failed to store selected text")
        
        # Process the query about the selected text
        result = await get_text_selection_service().process_text_selection_query(
            session_id=session_id,
            selected_text_id=UUID(stored_text["id"]),
            question=request.question,
            module_context=request.module_id
        )
        
        if not result:
            raise HTTPException(status_code=500, detail="Failed to process text selection query")
        
        # Add the selected text query to the session history
        user_message = get_session_manager_service().add_message_to_session(
            session_id=session_id,
            sender_type="student",
            content=f"Regarding selected text: '{request.selected_text[:100]}...', I ask: {request.question}",
            selected_text_ref=UUID(stored_text["id"])
        )
        
        if not user_message:
            raise HTTPException(status_code=500, detail="Failed to store user message")
        
        # Add AI response to session
        if "response" in result:
            ai_message = get_session_manager_service().add_message_to_session(
                session_id=session_id,
                sender_type="ai_agent",
                content=result["response"],
                citations=result.get("citations", []),
                selected_text_ref=UUID(stored_text["id"])
            )
            
            if not ai_message:
                raise HTTPException(status_code=500, detail="Failed to store AI response")
        
        # Prepare response
        response = TextSelectionResponse(
            session_id=request.session_id,
            response_id=str(uuid4()),
            message=result.get("response", "I couldn't process your query about the selected text. Please try again."),
            citations=result.get("citations", []),
            timestamp=str(datetime.now().timestamp()),
            validation_result={"processed": True}
        )
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"Error in text selection query: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# Health check endpoint for text selection service
@router.get("/health")
async def text_selection_health_check():
    """
    Health check endpoint for the text selection service
    """
    return {
        "status": "healthy",
        "service": "text-selection",
        "timestamp": str(datetime.now().timestamp())
    }


# WebSocket endpoint for real-time text selection
class TextSelectionConnectionManager:
    def __init__(self):
        self.active_connections: Dict[WebSocket, str] = {}  # Maps WebSocket to session_id

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[websocket] = session_id

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            del self.active_connections[websocket]

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections.keys():
            await connection.send_text(message)


selection_manager = TextSelectionConnectionManager()


@router.websocket("/ws/{session_id}")
async def text_selection_websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time text selection processing
    """
    # Validate session_id format
    try:
        UUID(session_id)
    except ValueError:
        await websocket.close(code=1003)  # Invalid session ID format
        return
    
    await selection_manager.connect(websocket, session_id)
    try:
        while True:
            data = await websocket.receive_text()
            
            # Parse the received message
            try:
                message_data = json.loads(data)
                message_type = message_data.get("type", "text_selection")
                selected_text = message_data.get("selected_text", "")
                question = message_data.get("question", "")
                module_id = message_data.get("module_id", "")
                chapter_id = message_data.get("chapter_id", "")
            except json.JSONDecodeError:
                await selection_manager.send_personal_message(
                    json.dumps({
                        "type": "error",
                        "message": "Invalid JSON format"
                    }), 
                    websocket
                )
                continue
            
            if message_type == "text_selection":
                # Validate selected text
                validation = get_text_selection_service().validate_selected_text(selected_text)
                
                if not validation["can_ask_query"]:
                    await selection_manager.send_personal_message(
                        json.dumps({
                            "type": "validation_error",
                            "message": validation["error_message"]
                        }),
                        websocket
                    )
                    continue
                
                # Process the text selection query
                result = await get_text_selection_service().process_text_selection_query(
                    session_id=UUID(session_id),
                    selected_text_id=None,  # We'll let the service handle storing
                    question=question,
                    module_context=module_id
                )
                
                # Create response message
                response_message = {
                    "type": "text_selection_response",
                    "response_id": str(uuid4()),
                    "session_id": session_id,
                    "message": result.get("response", "I couldn't process your text selection query"),
                    "citations": result.get("citations", []),
                    "timestamp": str(datetime.now().timestamp()),
                    "validation_result": validation
                }
                
                await selection_manager.send_personal_message(json.dumps(response_message), websocket)
            
    except WebSocketDisconnect:
        selection_manager.disconnect(websocket)


if __name__ == "__main__":
    # This would be run if the module is executed directly
    print("Text selection endpoints module loaded")