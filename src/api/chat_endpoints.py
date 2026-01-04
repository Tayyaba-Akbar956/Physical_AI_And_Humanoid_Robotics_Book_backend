import os
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect
from typing import Optional, Dict, Any, List
from uuid import UUID, uuid4
from pydantic import BaseModel, Field
import json
import asyncio
import time
from functools import wraps
from typing import Callable, Awaitable

from ..services.rag_agent import RAGAgentService
from ..services.session_manager import SessionManagementService
from ..services.conversational_context_manager import get_conversational_context_manager
from ..db.connection import get_db
from ..models.chat_session import ChatSessionCreate
from .validation_utils import ChatQueryRequest, validate_request_and_respond

# Serverless optimizations
TIMEOUT_LIMIT = int(os.getenv("TIMEOUT_WARNING_SECONDS", "8"))
query_cache = {}  # Simple in-memory cache for common queries

def get_cache_key(request: ChatQueryRequest):
    return f"{request.message[:100]}_{request.module_context}_{request.selected_text}"


# Performance monitoring utilities
class PerformanceMonitor:
    """
    Utility class for monitoring API performance
    """
    def __init__(self):
        self.request_times = []
        self.api_call_counts = {}

    def log_api_call(self, endpoint: str, execution_time: float):
        """
        Log an API call with its execution time
        """
        if endpoint not in self.api_call_counts:
            self.api_call_counts[endpoint] = 0
        self.api_call_counts[endpoint] += 1
        self.request_times.append(execution_time)

        # Keep only recent requests (last 1000) to avoid memory issues
        if len(self.request_times) > 1000:
            self.request_times = self.request_times[-1000:]

    def get_avg_response_time(self) -> float:
        """
        Get the average response time
        """
        if not self.request_times:
            return 0.0
        return sum(self.request_times) / len(self.request_times)

    def get_p95_response_time(self) -> float:
        """
        Get the 95th percentile response time
        """
        if not self.request_times:
            return 0.0
        sorted_times = sorted(self.request_times)
        index = int(0.95 * len(sorted_times))
        return sorted_times[index] if index < len(sorted_times) else 0.0

    def get_requests_per_minute(self) -> float:
        """
        Get estimated requests per minute
        """
        # This is a simplified implementation - in a real system you'd track timestamps
        # to calculate actual requests per minute
        if not self.request_times:
            return 0.0
        # Assume the recorded times represent a certain period
        return len(self.request_times) / 10  # Simplified calculation


performance_monitor = PerformanceMonitor()


def monitor_performance(func: Callable) -> Callable:
    """
    Decorator to monitor performance of API endpoints
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            execution_time = time.time() - start_time
            # Log the performance metric using the function name as the endpoint identifier
            endpoint_name = f"{func.__module__}.{func.__name__}"
            performance_monitor.log_api_call(endpoint_name, execution_time)
    return wrapper


# Global service instances (initially None)
_rag_agent_instance = None
_session_manager_instance = None

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

# Create API router
router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatQueryRequest(BaseModel):
    """
    Request model for chat queries
    """
    session_id: Optional[str] = None  # Will be converted to UUID in endpoint
    message: str = Field(..., min_length=1, max_length=2000)
    module_context: Optional[str] = None
    selected_text: Optional[str] = None  # For text selection queries


class ChatQueryResponse(BaseModel):
    """
    Response model for chat queries
    """
    session_id: str
    response_id: str
    message: str
    citations: List[Dict[str, str]]
    timestamp: str
    validation_result: Optional[Dict[str, Any]] = None  # Include validation details


class SessionCreateRequest(BaseModel):
    """
    Request model for creating a new session
    """
    student_id: str  # Will be converted to UUID in endpoint
    module_context: Optional[str] = None


class SessionResponse(BaseModel):
    """
    Response model for session operations
    """
    session_id: str
    created_at: str
    module_context: Optional[str] = None


@router.post("/query", response_model=ChatQueryResponse)
async def chat_query(request: ChatQueryRequest):
    """
    Handle a chat query and return a response with validation
    Enhanced with conversational context management
    """
    try:
        # Validate the request using our validation utility
        validation_result = validate_request_and_respond(request)

        if not validation_result["valid"]:
            raise HTTPException(
                status_code=400,
                detail=f"Request validation failed: {validation_result['message']}, Errors: {validation_result.get('errors', [])}"
            )

        # Convert session_id to UUID if provided, otherwise create a new session
        if request.session_id:
            try:
                session_id = UUID(request.session_id)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid session ID format")

            # Verify session exists and is active
            session_info = get_session_manager_service().get_session(session_id)
            if not session_info:
                raise HTTPException(status_code=404, detail="Session not found")
            if not session_info["is_active"]:
                raise HTTPException(status_code=400, detail="Session is not active")
        else:
            # Create a new session (for this example, using a temporary student ID)
            # In a real application, you would have proper authentication
            temp_student_id = uuid4()  # This would come from authentication
            session_info = get_session_manager_service().create_session(
                student_id=temp_student_id,
                current_module_context=request.module_context
            )

            if not session_info:
                raise HTTPException(status_code=500, detail="Failed to create session")

            session_id = session_info["id"]

        # Update session with current module context if provided
        if request.module_context:
            from ..models.chat_session import ChatSessionUpdate
            update_result = get_session_manager_service().update_session(
                session_id,
                ChatSessionUpdate(current_module_context=request.module_context)
            )
            if not update_result:
                raise HTTPException(status_code=500, detail="Failed to update session module context")

        # Get conversation context using the conversational context manager
        try:
            context_manager = get_conversational_context_manager()

            # Get conversation context for the session
            conversation_context = context_manager.get_conversation_context(session_id, num_exchanges=5)

            # Validate conversation state
            validation = context_manager.validate_conversation_state(session_id)
            if not validation["session_exists"]:
                raise HTTPException(status_code=404, detail="Session not found")
            if not validation["session_active"]:
                raise HTTPException(status_code=400, detail="Session is not active")

            # Resolve follow-up question if applicable
            resolved_context = context_manager.resolve_follow_up_question(request.message, conversation_context)

            # Get the actual query to use (may be modified for follow-ups)
            actual_query = resolved_context["resolved_query"]

            # Track the current topic
            topic_info = context_manager.track_conversation_topic(conversation_context)
            current_topic = topic_info["current_topic"]
        except Exception as e:
            print(f"Error in conversation context management: {e}")
            import traceback
            traceback.print_exc()
            # Continue with empty context if context management fails
            conversation_context = []
            current_topic = None
            actual_query = request.message

        # Add user message to session with conversation context
        user_message = get_session_manager_service().add_message_to_session(
            session_id=session_id,
            sender_type="student",
            content=request.message,
            topic_anchored=current_topic,  # Use the current topic as the anchor
            follow_up_to=resolved_context.get("references", [{}])[0].get("id") if resolved_context.get("references") else None
        )
        if not user_message:
            raise HTTPException(status_code=500, detail="Failed to store user message")

        # Simple query preprocessing for greetings
        greetings = ["hi", "hello", "hey", "how are you", "good morning", "good afternoon", "good evening"]
        if request.message.lower().strip() in greetings:
            return ChatQueryResponse(
                session_id=str(session_id),
                response_id=str(uuid4()),
                message="Hello! I'm your Physical AI & Humanoid Robotics assistant. How can I help you with your textbook today?",
                citations=[],
                timestamp=datetime.utcnow().isoformat(),
                validation_result={"valid": True, "preprocessed": True}
            )

        # Check cache first
        cache_key = get_cache_key(request)
        if cache_key in query_cache:
            cache_entry = query_cache[cache_key]
            # Verify if it's still fresh (within 1 hour)
            if time.time() - cache_entry["timestamp"] < 3600:
                return ChatQueryResponse(
                    session_id=str(session_id),
                    response_id=str(uuid4()),
                    message=cache_entry["response"],
                    citations=cache_entry["citations"],
                    timestamp=datetime.utcnow().isoformat(),
                    validation_result={"valid": True, "cached": True}
                )

        # Get response from RAG agent with enhanced conversation context and timeout
        try:
            response_data = await asyncio.wait_for(
                get_rag_agent_service().answer_question(
                    query=actual_query,
                    session_id=str(session_id),
                    module_context=request.module_context,
                    selected_text=request.selected_text,
                    conversation_context=conversation_context,
                    context_window_size=10
                ),
                timeout=TIMEOUT_LIMIT
            )
        except asyncio.TimeoutError:
            return ChatQueryResponse(
                session_id=str(session_id),
                response_id=str(uuid4()),
                message="I'm sorry, the retrieval process is taking longer than expected due to serverless limitations. Please try a simpler question or try again in a moment.",
                citations=[],
                timestamp=datetime.utcnow().isoformat(),
                validation_result={"valid": True, "error": "timeout"}
            )
        except Exception as e:
            print(f"Error calling RAG agent: {e}")
            import traceback
            traceback.print_exc()
            return ChatQueryResponse(
                session_id=str(session_id),
                response_id=str(uuid4()),
                message="I'm sorry, but I encountered an error processing your request. Please try again.",
                citations=[],
                timestamp=datetime.utcnow().isoformat(),
                validation_result={"valid": True, "error": str(e)}
            )

        # If response indicates no relevant content found, provide appropriate message
        if not response_data.get("response") or "couldn't find relevant information" in response_data.get("response", "").lower():
            response_text = "I couldn't find relevant information in the textbook to answer your question. Please try rephrasing or check other chapters."
            citations = []
        else:
            response_text = response_data["response"]
            citations = response_data.get("citations", [])
            
            # Cache the successful response
            query_cache[cache_key] = {
                "response": response_text,
                "citations": citations,
                "timestamp": time.time()
            }

        # Add AI response to session with conversation context
        ai_message = get_session_manager_service().add_message_to_session(
            session_id=session_id,
            sender_type="ai_agent",
            content=response_text,
            citations=citations,
            parent_message_id=user_message["id"] if user_message else None,  # Link to user message
            topic_anchored=current_topic,  # Maintain topic context
            follow_up_to=user_message["id"]  # Mark this as a follow-up to the user's message
        )

        if not ai_message:
            raise HTTPException(status_code=500, detail="Failed to store AI response")

        # Update conversation context in the database after the exchange
        update_success = context_manager.update_conversation_context_in_db(
            session_id,
            request.message,
            response_text
        )
        if not update_success:
            print(f"Warning: Failed to update conversation context in DB for session {session_id}")

        # Create response
        response = ChatQueryResponse(
            session_id=str(session_id),
            response_id=str(uuid4()),
            message=response_text,
            citations=citations,
            timestamp=ai_message["timestamp"].isoformat(),
            validation_result=validation_result  # Include validation result in response
        )

        return response

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"Error in chat query: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/session/create", response_model=SessionResponse)
async def create_session(request: SessionCreateRequest):
    """
    Create a new chat session
    """
    try:
        # Convert student_id to UUID
        try:
            student_id = UUID(request.student_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid student ID format")
        
        # Create session
        session_info = get_session_manager_service().create_session(
            student_id=student_id,
            current_module_context=request.module_context
        )
        
        if not session_info:
            raise HTTPException(status_code=500, detail="Failed to create session")
        
        return SessionResponse(
            session_id=str(session_info["id"]),
            created_at=session_info["created_at"].isoformat(),
            module_context=session_info["current_module_context"]
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/session/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """
    Get information about a specific session
    """
    try:
        # Convert session_id to UUID
        try:
            uuid_session_id = UUID(session_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid session ID format")
        
        session_info = get_session_manager_service().get_session(uuid_session_id)

        if not session_info:
            raise HTTPException(status_code=404, detail="Session not found")

        return SessionResponse(
            session_id=str(session_info["id"]),
            created_at=session_info["created_at"].isoformat(),
            module_context=session_info["current_module_context"]
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"Error getting session: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/session/{session_id}/clear")
async def clear_session_history(session_id: str):
    """
    Clear the conversation history for a session
    """
    try:
        # Convert session_id to UUID
        try:
            uuid_session_id = UUID(session_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid session ID format")
        
        success = get_session_manager_service().clear_session_history(uuid_session_id)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to clear session history")

        return {"message": "Session history cleared successfully", "session_id": session_id}
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"Error clearing session history: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Streaming response helper function
async def send_streaming_response(websocket: WebSocket, response_generator, session_id: str):
    """
    Send response in streaming fashion to the WebSocket client
    """
    response_id = str(uuid4())

    # Send response in chunks to simulate streaming
    full_response = ""
    for i, chunk in enumerate(response_generator):
        is_final_chunk = i == len(response_generator) - 1  # Mark final chunk

        chunk_message = {
            "type": "streaming_response",
            "response_id": response_id,
            "session_id": session_id,
            "chunk": chunk,
            "is_final_chunk": is_final_chunk,
            "timestamp": str(datetime.now())
        }

        try:
            await websocket.send_text(json.dumps(chunk_message))
        except WebSocketDisconnect:
            break
        except Exception as e:
            print(f"Error sending WebSocket message: {e}")
            break


# WebSocket endpoint for real-time chat
class ConnectionManager:
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


manager = ConnectionManager()


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time chat with response streaming
    """
    # Validate session_id format
    try:
        UUID(session_id)
    except ValueError:
        await websocket.close(code=1003)  # Invalid session ID format
        return

    await manager.connect(websocket, session_id)
    try:
        while True:
            data = await websocket.receive_text()

            # Parse the received message
            try:
                message_data = json.loads(data)
                message_type = message_data.get("type", "chat_message")
                message_content = message_data.get("message", "")
                module_context = message_data.get("module_context")
            except json.JSONDecodeError:
                await websocket.send_text(
                    json.dumps({
                        "type": "error",
                        "message": "Invalid JSON format"
                    })
                )
                continue

            if message_type == "chat_message":
                try:
                    # Process the chat query using our existing logic
                    response_data = await get_rag_agent_service().answer_question(
                        query=message_content,
                        session_id=session_id,
                        module_context=module_context
                    )

                    # Stream the response in chunks
                    full_response = response_data.get("response", "Sorry, I couldn't process your request.")

                    # Break the response into chunks for streaming
                    words = full_response.split()
                    chunk_size = 10  # Words per chunk

                    for i in range(0, len(words), chunk_size):
                        chunk_words = words[i:i + chunk_size]
                        chunk_text = " ".join(chunk_words)
                        is_final_chunk = (i + chunk_size >= len(words))

                        response_message = {
                            "type": "streaming_response",
                            "response_id": str(uuid4()),
                            "session_id": session_id,
                            "chunk": chunk_text,
                            "is_final_chunk": is_final_chunk,
                            "citations": response_data.get("citations", []) if is_final_chunk else [],
                            "timestamp": str(datetime.now())
                        }

                        await websocket.send_text(json.dumps(response_message))
                except Exception as e:
                    print(f"Error in WebSocket chat processing: {e}")
                    import traceback
                    traceback.print_exc()
                    error_message = {
                        "type": "error",
                        "message": "Sorry, I encountered an error processing your request."
                    }
                    await websocket.send_text(json.dumps(error_message))

    except WebSocketDisconnect:
        manager.disconnect(websocket)


@router.get("/stream/{session_id}")
async def stream_response_endpoint(session_id: str, query: str, module_context: Optional[str] = None):
    """
    HTTP endpoint for response streaming (alternative to WebSocket)
    Uses Server-Sent Events (SSE) to stream response chunks
    """
    from fastapi.responses import StreamingResponse

    async def generate_stream():
        try:
            # Process the query
            response_data = await get_rag_agent_service().answer_question(
                query=query,
                session_id=session_id,
                module_context=module_context
            )

            full_response = response_data.get("response", "Sorry, I couldn't process your request.")

            # Break the response into chunks for streaming
            sentences = full_response.split('. ')
            for i, sentence in enumerate(sentences):
                if sentence.strip():  # Skip empty sentences
                    # Add a period back if it's not the last sentence
                    if i < len(sentences) - 1:
                        sentence = sentence + '.'

                    chunk_data = {
                        "chunk": sentence,
                        "is_final_chunk": i == len(sentences) - 1,
                        "citations": response_data.get("citations", []) if i == len(sentences) - 1 else [],
                        "timestamp": str(datetime.now())
                    }

                    yield f"data: {json.dumps(chunk_data)}\\n\\n"

                    # Small delay between chunks to simulate streaming
                    await asyncio.sleep(0.1)

        except Exception as e:
            print(f"Error in HTTP streaming: {e}")
            import traceback
            traceback.print_exc()
            error_data = {
                "type": "error",
                "message": f"Error generating response: {str(e)}"
            }
            yield f"data: {json.dumps(error_data)}\\n\\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")


# Health check endpoint
@router.get("/health")
async def health_check():
    """
    Health check endpoint for the chat service
    """
    return {
        "status": "healthy",
        "service": "chat",
        "timestamp": str(datetime.now().timestamp())
    }


# Performance monitoring endpoints
@router.get("/performance/metrics")
async def performance_metrics():
    """
    Get performance metrics for the chat service
    """
    avg_response_time = performance_monitor.get_avg_response_time()
    p95_response_time = performance_monitor.get_p95_response_time()
    req_per_min = performance_monitor.get_requests_per_minute()

    return {
        "average_response_time_seconds": round(avg_response_time, 3),
        "p95_response_time_seconds": round(p95_response_time, 3),
        "estimated_requests_per_minute": round(req_per_min, 2),
        "total_recorded_calls": len(performance_monitor.request_times),
        "endpoint_breakdown": performance_monitor.api_call_counts,
        "timestamp": str(time.time())
    }


@router.get("/performance/clear")
async def clear_performance_metrics():
    """
    Clear performance monitoring metrics
    """
    performance_monitor.request_times = []
    performance_monitor.api_call_counts = {}

    return {
        "status": "Metrics cleared",
        "timestamp": str(time.time())
    }


if __name__ == "__main__":
    # This would be run if the module is executed directly
    print("Chat endpoints module loaded")