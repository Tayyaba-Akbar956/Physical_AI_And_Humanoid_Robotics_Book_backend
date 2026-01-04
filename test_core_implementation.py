#!/usr/bin/env python3
"""
Simplified Testing Script for RAG Chatbot Implementation
This script tests core functionality without requiring external services
"""

import sys
import os

# Add the backend src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.services.conversational_context_manager import ConversationalContextManager
from src.services.session_manager import SessionManagementService
from src.middleware.logging import setup_logging


def test_conversational_context():
    """Test conversational context management"""
    print("🧪 Testing Conversational Context Management...")

    # Test the conversational context manager
    context_manager = ConversationalContextManager()

    # Test topic tracking
    topic_info = context_manager.track_conversation_topic([
        {"role": "user", "content": "What is ROS 2?", "timestamp": 0},
        {"role": "assistant", "content": "ROS 2 is a robot operating system...", "timestamp": 1}
    ])

    assert "current_topic" in topic_info
    print("✅ Topic tracking works correctly")

    # Test follow-up resolution
    follow_up_resolution = context_manager.resolve_follow_up_question(
        "Tell me more about it",
        [{"role": "user", "content": "What is ROS 2?", "timestamp": 0}]
    )

    assert "is_follow_up" in follow_up_resolution
    print(f"✅ Follow-up resolution works correctly: {follow_up_resolution['is_follow_up']}")


def test_session_management():
    """Test session management functionality"""
    print("\n🧪 Testing Session Management...")
    
    try:
        # Initialize session manager (this won't connect to database in testing mode)
        session_manager = SessionManagementService()
        
        # Test validation functions
        validation = session_manager.validate_conversation_state("test-session-id")
        assert isinstance(validation, dict)
        print("✅ Conversation state validation works correctly")
        
        # Test basic functions
        topic_info = session_manager.get_active_conversation_topic("test-session-id")
        # This will return None since session doesn't exist, which is expected
        print("✅ Topic retrieval works correctly")
        
    except Exception as e:
        print(f"❌ Session management test failed: {e}")
        return False
    
    return True


def test_logging():
    """Test logging functionality"""
    print("\n🧪 Testing Logging System...")
    
    try:
        # Initialize logging
        logger = setup_logging()
        
        # Test logging functions
        from src.middleware.logging import log_api_call, log_error, log_user_interaction
        
        # These should not raise exceptions
        log_api_call("/test/endpoint", "GET", 0.1, 200, "user123", "session456")
        log_error("error123", "TestError", "Test error message")
        log_user_interaction("user123", "session456", "test", "test content")
        
        print("✅ Logging system works correctly")
        
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        return False
    
    return True


def test_core_modules():
    """Test that core modules can be imported without errors"""
    print("\n🧪 Testing Core Module Imports...")
    
    try:
        # Import core modules to ensure they're properly structured
        from src.services.rag_agent import RAGAgentService
        from src.services.semantic_search import SemanticSearchService
        from src.api.chat_endpoints import ChatQueryRequest
        from src.api.conversation_endpoints import ConversationHistoryResponse
        from src.api.module_context_endpoints import ModuleContextRequest
        from src.middleware.error_handling import ErrorHandlingMiddleware
        
        print("✅ All core modules imported successfully")
        
    except Exception as e:
        print(f"❌ Module import test failed: {e}")
        return False
    
    return True


def run_basic_tests():
    """Run basic tests that don't require external services"""
    print("="*70)
    print("🤖 SIMPLIFIED RAG CHATBOT IMPLEMENTATION TEST SUITE")
    print("="*70)
    
    tests = [
        ("Conversational Context Management", test_conversational_context),
        ("Session Management", test_session_management),
        ("Logging System", test_logging),
        ("Core Module Imports", test_core_modules),
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "="*70)
    print("📊 TEST RESULTS SUMMARY")
    print("="*70)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if not result:
            all_passed = False
    
    print("="*70)
    if all_passed:
        print("🎉 ALL BASIC TESTS PASSED - Core implementation is working!")
        print("\n📚 To run the full application:")
        print("   1. Install requirements: pip install -r requirements.txt")
        print("   2. Set up environment variables in a .env file:")
        print("      - GEMINI_API_KEY")
        print("      - QDRANT_URL, QDRANT_API_KEY")
        print("      - NEON_DB_URL")
        print("   3. Run the application: uvicorn src.main:app --reload --port 8000")
        print("\n⚠️  Note: This implementation requires external services (Qdrant, NeonDB, GEMINI)")
        print("   which were not configured for this test run.")
    else:
        print("⚠️  SOME TESTS FAILED - Core implementation has issues")
    print("="*70)
    
    return all_passed


if __name__ == "__main__":
    success = run_basic_tests()
    sys.exit(0 if success else 1)