# API entry point for deployment
import os
import sys
from pathlib import Path

# Add the backend directory to the Python path to handle relative imports
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

# Change to the backend directory to handle relative imports properly
original_cwd = os.getcwd()
os.chdir(str(backend_path))

# Import the main app after adjusting the path
from src.main import app

# Restore the original working directory
os.chdir(original_cwd)

# Export the app for uvicorn
# Usage: uvicorn api:app --host 0.0.0.0 --port $PORT
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)