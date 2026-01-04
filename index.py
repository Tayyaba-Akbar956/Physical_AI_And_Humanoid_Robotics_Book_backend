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
from datetime import datetime

# Add current directory to Python path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Create FastAPI app
app = FastAPI(
    title="Physical AI RAG Backend",
    version="1.0.0",
    description="RAG Chatbot for Physical AI & Humanoid Robotics"
)

# CORS Configuration
allowed_origins_raw = os.getenv("ALLOWED_ORIGINS", "*")
if allowed_origins_raw == "*":
    origins = ["*"]
else:
    origins = [o.strip() for o in allowed_origins_raw.split(",") if o.strip()]

# Add Vercel-specific origins
if "*" not in origins:
    vercel_url = os.getenv("VERCEL_URL")
    if vercel_url:
        origins.append(f"https://{vercel_url}")
    # Add your production domain
    origins.extend([
        "https://physical-ai-book.vercel.app",
        "https://physical-ai-book-five-ivory.vercel.app"
    ])

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Basic endpoints
@app.get("/")
@app.get("/api")
async def root():
    return {
        "message": "Physical AI RAG Backend",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/health")
@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "service": "rag-backend",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/quick-test")
async def quick_test():
    return {
        "status": "connected",
        "message": "Backend is operational",
        "timestamp": datetime.utcnow().isoformat()
    }

# Import and include routers
try:
    from src.api.chat_endpoints import router as chat_router
    from src.api.text_selection_endpoints import router as text_selection_router
    from src.api.conversation_endpoints import router as conversation_router
    from src.api.module_context_endpoints import router as module_context_router
    from src.api.system_endpoints import router as system_router
    
    app.include_router(chat_router, prefix="/api")
    app.include_router(text_selection_router, prefix="/api")
    app.include_router(conversation_router, prefix="/api")
    app.include_router(module_context_router, prefix="/api")
    app.include_router(system_router, prefix="/api")
    
except ImportError as e:
    print(f"Warning: Could not import routers: {e}")
    # Continue with basic endpoints only

# Vercel serverless handler
handler = Mangum(app, lifespan="off")