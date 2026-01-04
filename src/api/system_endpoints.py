from fastapi import APIRouter
from typing import Dict, Any
from datetime import datetime
from ..services.rag_agent import RAGAgentService
from ..services.semantic_search import SemanticSearchService
from ..db.qdrant_client import QdrantManager
from ..db.connection import get_db


# Create API router
router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/health")
async def system_health_check() -> Dict[str, Any]:
    """
    Comprehensive health check for the entire system
    Checks the status of all major components
    """
    # Test RAG agent service
    rag_agent_ok = True
    try:
        rag_agent = RAGAgentService()
    except Exception as e:
        rag_agent_ok = False

    # Test semantic search service
    semantic_search_ok = True
    try:
        search_service = SemanticSearchService()
    except Exception as e:
        semantic_search_ok = False

    # Test Qdrant connection
    qdrant_ok = True
    try:
        qdrant_client = QdrantManager()
        # Test basic connection
        collection_info = await qdrant_client.get_collection_info()
    except Exception as e:
        qdrant_ok = False

    # Test database connection
    db_ok = True
    try:
        db = next(get_db())
        # Perform a simple query to test connection
        db.execute("SELECT 1")
    except Exception as e:
        db_ok = False
    finally:
        if 'db' in locals():
            db.close()

    overall_status = all([rag_agent_ok, semantic_search_ok, qdrant_ok, db_ok])

    return {
        "status": "healthy" if overall_status else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "rag_agent": "healthy" if rag_agent_ok else "unhealthy",
            "semantic_search": "healthy" if semantic_search_ok else "unhealthy",
            "qdrant": "healthy" if qdrant_ok else "unhealthy",
            "database": "healthy" if db_ok else "unhealthy"
        },
        "overall_status": overall_status
    }


@router.get("/health/dependencies")
async def dependency_health_check() -> Dict[str, Any]:
    """
    Check the health of external dependencies
    """
    try:
        # Import and check GEMINI API connectivity
        import os
        from dotenv import load_dotenv

        load_dotenv()
        gemini_api_key = os.getenv("GEMINI_API_KEY")

        if not gemini_api_key:
            return {
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "dependencies": {
                    "gemini_api": {
                        "status": "unhealthy",
                        "reason": "GEMINI_API_KEY not set in environment"
                    }
                }
            }

        # Try to create a client and make a simple request
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            api_key=gemini_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",  # GEMINI OpenAI-compatible endpoint
        )

        # Test API connectivity with a minimal request
        test_response = await client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )

        gemini_ok = test_response is not None

        return {
            "status": "healthy" if gemini_ok else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "dependencies": {
                "gemini_api": {
                    "status": "healthy" if gemini_ok else "unhealthy",
                    "message": "API connectivity test passed" if gemini_ok else "API connectivity test failed"
                }
            }
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "dependencies": {
                "gemini_api": {
                    "status": "unhealthy",
                    "error": str(e)
                }
            }
        }


@router.get("/metrics")
async def system_metrics() -> Dict[str, Any]:
    """
    Get system metrics and performance indicators
    """
    # Get metrics from various services
    try:
        from ..api.chat_endpoints import performance_monitor
        
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "average_response_time": performance_monitor.get_avg_response_time(),
                "p95_response_time": performance_monitor.get_p95_response_time(),
                "requests_per_minute": performance_monitor.get_requests_per_minute(),
                "total_recorded_calls": len(performance_monitor.request_times),
            }
        }
    except Exception as e:
        return {
            "timestamp": datetime.now().isoformat(),
            "error": f"Could not retrieve metrics: {str(e)}"
        }


@router.get("/status")
async def system_status() -> Dict[str, Any]:
    """
    Get overall system status including version and configuration
    """
    import sys
    import platform
    
    return {
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "architecture": platform.architecture()[0]
        },
        "features": {
            "conversational_context": True,
            "module_aware": True,
            "text_selection": True,
            "semantic_search": True
        }
    }