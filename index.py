# Vercel entry point for the backend API
import os
import sys
from pathlib import Path

# Add the backend directory to the Python path
backend_path = Path(__file__).parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

try:
    from src.main import app as fastapi_app
    from mangum import Mangum

    # Wrap the FastAPI app with Mangum for Vercel serverless
    # Lifespan="off" avoids initialization issues in serverless environments
    handler = Mangum(fastapi_app, lifespan="off")
    
    # We export the mangum handler as 'app' because Vercel looks for 'app' by default
    app = handler

except Exception as e:
    print(f"CRITICAL: Backend failed to initialize: {e}")
    # Fallback minimal app for debugging
    from fastapi import FastAPI
    err_app = FastAPI(title="Physical AI RAG Backend - Error Fallback")
    
    @err_app.get("/")
    @err_app.get("/health")
    @err_app.get("/api/health")
    async def health():
        return {
            "status": "error",
            "message": "Backend initialization failed",
            "error_detail": str(e)
        }
    
    from mangum import Mangum
    app = Mangum(err_app, lifespan="off")
