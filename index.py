"""
Vercel serverless entry point for Physical AI RAG Backend
This is the ONLY entry point file you need
"""
import os
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

# 1. Path manipulation to ensure src is found correctly
# Add the current directory to sys.path
backend_path = Path(__file__).parent.absolute()
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

# 2. Initialize the FastAPI app and Mangum handler
try:
    # Importing from our main application
    from src.main import app as fastapi_app
    
    # Wrap with Mangum for Vercel Serverless (AWS Lambda)
    # lifespan="off" is often safer for serverless to avoid startup delays/errors
    handler = Mangum(fastapi_app, lifespan="off")
    
    # The variable MUST be named 'app' for Vercel's default detection
    app = handler

except Exception as e:
    # 3. Fallback error app if main import fails
    print(f"CRITICAL ERROR during initialization: {e}")
    
    error_app = FastAPI(title="Physical AI RAG - Error Handler")
    
    @error_app.get("/{full_path:path}")
    async def error_fallback(full_path: str):
        return {
            "status": "initialization_error",
            "message": "The backend failed to start. Check Vercel logs.",
            "error": str(e),
            "requested_path": full_path
        }
    
    app = Mangum(error_app, lifespan="off")

