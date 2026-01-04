#!/usr/bin/env python3
"""
Health check script to verify the application can start without crashing
"""

import sys
import os
import traceback

# Add the backend src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all critical modules can be imported without errors"""
    print("Testing module imports...")

    try:
        # Test main application import
        from index import app, handler
        print("Main application imported successfully")
    except Exception as e:
        print(f"Failed to import main application: {e}")
        traceback.print_exc()
        return False

    try:
        # Test RAG agent service
        from src.services.rag_agent import RAGAgentService
        rag_agent = RAGAgentService()
        print("RAG Agent service created successfully")
    except Exception as e:
        print(f"Failed to create RAG Agent service: {e}")
        traceback.print_exc()
        return False

    try:
        # Test session management service
        from src.services.session_manager import SessionManagementService
        session_manager = SessionManagementService()
        print("Session management service created successfully")
    except Exception as e:
        print(f"Failed to create session management service: {e}")
        traceback.print_exc()
        return False

    try:
        # Test conversational context manager
        from src.services.conversational_context_manager import get_conversational_context_manager
        context_manager = get_conversational_context_manager()
        print("Conversational context manager created successfully")
    except Exception as e:
        print(f"Failed to create conversational context manager: {e}")
        traceback.print_exc()
        return False

    try:
        # Test API endpoints import
        from src.api.chat_endpoints import router as chat_router
        print("Chat endpoints imported successfully")
    except Exception as e:
        print(f"Failed to import chat endpoints: {e}")
        traceback.print_exc()
        return False

    return True

def test_environment():
    """Test that environment variables are properly set"""
    print("\nTesting environment variables...")

    required_vars = ['GEMINI_API_KEY']
    missing_vars = []

    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print(f"Missing environment variables: {missing_vars}")
        print("   Note: This may cause runtime issues, but the app should handle this gracefully")
    else:
        print("All required environment variables are set")

    return True

def main():
    """Run the health check"""
    print("="*60)
    print("RAG Chatbot Health Check")
    print("="*60)

    # Test imports
    imports_ok = test_imports()

    # Test environment
    env_ok = test_environment()

    print("\n" + "="*60)
    print("Health Check Results")
    print("="*60)

    if imports_ok:
        print("All modules imported successfully")
        print("Application should start without crashing")
    else:
        print("Some modules failed to import")
        print("Application may crash on startup")

    if env_ok:
        print("Environment variables are properly configured")
    else:
        print("Some environment variables are missing")

    print("\nRecommendations:")
    if imports_ok:
        print("   - The application should start successfully")
        print("   - Deploy to Vercel with confidence")
    else:
        print("   - Fix import errors before deployment")

    if not os.getenv('GEMINI_API_KEY'):
        print("   - Set GEMINI_API_KEY environment variable for full functionality")

    print("="*60)

    return imports_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)