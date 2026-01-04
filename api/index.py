from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
import os
import sys
from datetime import datetime
from pathlib import Path

# Add the parent directory to sys.path to allow importing from src
# In Vercel, the backend folder is usually the root of the deployment
backend_root = Path(__file__).parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from src.api.chat_endpoints import router as chat_router
from src.api.text_selection_endpoints import router as text_selection_router
from src.api.conversation_endpoints import router as conversation_router
from src.api.module_context_endpoints import router as module_context_router
from src.api.system_endpoints import router as system_router

app = FastAPI(title="Physical AI RAG Backend")

# CORS Configuration
allowed_origins_raw = os.getenv("ALLOWED_ORIGINS", "*")
origins = [o.strip() for o in allowed_origins_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
@app.get("/api/")
async def root():
    return {
        "message": "Physical AI RAG Backend",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "service": "rag-backend",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/quick-test")
async def quick_test():
    """Fast connectivity testing without RAG processing"""
    return {
        "status": "connected",
        "timestamp": datetime.utcnow().isoformat(),
        "message": "Backend connectivity is working correctly"
    }

# Include existing routers
app.include_router(chat_router)
app.include_router(text_selection_router)
app.include_router(conversation_router)
app.include_router(module_context_router)
app.include_router(system_router)

# CRITICAL: Vercel serverless handler
handler = Mangum(app, lifespan="off")
