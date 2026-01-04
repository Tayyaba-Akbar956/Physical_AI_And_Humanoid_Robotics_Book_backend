from fastapi import APIRouter, HTTPException
from typing import Optional, Dict, Any, List
from uuid import UUID
from pydantic import BaseModel
from ..services.session_manager import SessionManagementService
from ..services.semantic_search import SemanticSearchService


# Create API router
router = APIRouter(prefix="/api/module-context", tags=["module-context"])


class ModuleContextRequest(BaseModel):
    """
    Request model for module context operations
    """
    session_id: str
    module_id: str


class ModuleContextResponse(BaseModel):
    """
    Response model for module context operations
    """
    session_id: str
    current_module: str
    message: str


class ModuleRelevanceRequest(BaseModel):
    """
    Request model for checking module relevance
    """
    query: str
    modules: List[str]


class ModuleRelevanceResponse(BaseModel):
    """
    Response model for module relevance results
    """
    query: str
    relevance_scores: Dict[str, float]


@router.post("/set", response_model=ModuleContextResponse)
async def set_module_context(request: ModuleContextRequest):
    """
    Set the current module context for a session
    """
    try:
        # Convert session_id to UUID
        try:
            session_id = UUID(request.session_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid session ID format")

        session_manager = SessionManagementService()

        # Verify session exists
        session_info = session_manager.get_session(session_id)
        if not session_info:
            raise HTTPException(status_code=404, detail="Session not found")

        # Update the session with the new module context
        from ..models.chat_session import ChatSessionUpdate
        update_result = session_manager.update_session(
            session_id,
            ChatSessionUpdate(current_module_context=request.module_id)
        )

        if not update_result:
            raise HTTPException(status_code=500, detail="Failed to update module context")

        response = ModuleContextResponse(
            session_id=str(session_id),
            current_module=request.module_id,
            message="Module context updated successfully"
        )

        return response

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"Error setting module context: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/current/{session_id}", response_model=ModuleContextResponse)
async def get_current_module_context(session_id: str):
    """
    Get the current module context for a session
    """
    try:
        # Convert session_id to UUID
        try:
            uuid_session_id = UUID(session_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid session ID format")

        session_manager = SessionManagementService()

        # Get session info
        session_info = session_manager.get_session(uuid_session_id)
        if not session_info:
            raise HTTPException(status_code=404, detail="Session not found")

        response = ModuleContextResponse(
            session_id=session_id,
            current_module=session_info["current_module_context"] or "unknown",
            message="Current module context retrieved successfully"
        )

        return response

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"Error getting current module context: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/relevance", response_model=ModuleRelevanceResponse)
async def get_module_relevance(request: ModuleRelevanceRequest):
    """
    Get relevance scores for a query across different modules
    """
    try:
        # Use semantic search service to calculate relevance
        search_service = SemanticSearchService()
        
        
        relevance_scores = await search_service.get_module_content_relevance(
            query=request.query,
            modules=request.modules
        )

        response = ModuleRelevanceResponse(
            query=request.query,
            relevance_scores=relevance_scores
        )

        return response

    except Exception as e:
        print(f"Error calculating module relevance: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/suggestions/{module_id}/{query}")
async def get_module_content_suggestions(module_id: str, query: str):
    """
    Get content suggestions within a module based on a query
    """
    try:
        search_service = SemanticSearchService()
        
        
        results = await search_service.search_in_module(
            query=query,
            module_id=module_id,
            top_k=5
        )
        
        return {
            "module_id": module_id,
            "query": query,
            "suggestions": results
        }

    except Exception as e:
        print(f"Error getting module content suggestions: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Health check endpoint
@router.get("/health")
async def health_check():
    """
    Health check endpoint for the module context service
    """
    return {
        "status": "healthy",
        "service": "module-context",
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }