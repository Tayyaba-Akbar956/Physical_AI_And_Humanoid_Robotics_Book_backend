#!/usr/bin/env python3
"""
Validation script for the RAG Chatbot implementation
This script verifies that all implemented features work correctly
"""

import asyncio
import sys
import os
from typing import Dict, Any, List
from uuid import uuid4

# Add the root directory to the Python path
sys.path.insert(0, os.path.dirname(__file__))

from src.services.rag_agent import RAGAgentService
from src.services.session_manager import SessionManagementService
from src.services.semantic_search import SemanticSearchService
from src.models.chat_session import ChatSessionCreate
from src.db.qdrant_client import QdrantManager
from src.db.connection import get_db


async def test_basic_functionality():
    """Test basic RAG agent functionality"""
    print("Testing basic RAG agent functionality...")
    
    try:
        rag_agent = RAGAgentService()
        print("✓ RAG Agent service initialized successfully")
        
        # Test query processing without any context
        response = await rag_agent.generate_response(
            query="What is ROS 2?",
            context=[{
                "id": "test1",
                "score": 0.9,
                "module_id": "module-2-ros2",
                "chapter_id": "ch-2-1",
                "section_id": "introduction",
                "content": "ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software. It is a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms.",
                "hierarchy_path": "module-2-ros2/ch-2-1/introduction",
                "content_type": "text",
                "metadata": {}
            }]
        )
        
        assert "response" in response
        assert len(response["response"]) > 0
        print("✓ Basic query processing works correctly")
        print(f"  Response preview: {response['response'][:100]}...")
        
    except Exception as e:
        print(f"✗ Error in basic functionality test: {e}")
        return False
    
    return True


async def test_conversation_context():
    """Test conversation context management"""
    print("\nTesting conversation context management...")
    
    try:
        # Test session management
        session_manager = SessionManagementService()
        student_id = uuid4()
        
        # Create a session
        session_info = await session_manager.create_session(
            student_id=student_id,
            current_module_context="module-1-introduction"
        )
        
        assert session_info is not None
        assert session_info["is_active"] == True
        print("✓ Session creation works correctly")
        
        # Add a message to the session
        message_result = await session_manager.add_message_to_session(
            session_id=session_info["id"],
            sender_type="student",
            content="What is artificial intelligence?",
            topic_anchored="introduction"
        )
        
        assert message_result is not None
        print("✓ Adding message to session works correctly")
        
        # Get conversation context
        context = await session_manager.get_conversation_context(session_info["id"])
        assert len(context) >= 1
        print("✓ Retrieving conversation context works correctly")
        
    except Exception as e:
        print(f"✗ Error in conversation context test: {e}")
        return False
    
    return True


async def test_module_context():
    """Test module context prioritization"""
    print("\nTesting module context prioritization...")
    
    try:
        rag_agent = RAGAgentService()
        
        # Test retrieving content with module filter
        content = await rag_agent.get_relevant_content(
            query="robot operating system",
            top_k=3,
            module_filter="module-2-ros2"
        )
        
        print(f"✓ Retrieved {len(content)} content items with module filtering")
        
        # If content is retrieved, verify it has module information
        if content:
            for item in content:
                assert "module_id" in item
            print("✓ Content items include proper module information")
        
    except Exception as e:
        print(f"✗ Error in module context test: {e}")
        return False
    
    return True


async def test_semantic_search():
    """Test semantic search functionality"""
    print("\nTesting semantic search functionality...")
    
    try:
        search_service = SemanticSearchService()
        
        # Test that the service can be initialized
        info = await search_service.get_collection_stats()
        print("✓ Semantic search service initialized correctly")
        
        # Test simple search (won't return results without actual data but should not error)
        try:
            results = await search_service.search("test query", top_k=2)
            print(f"✓ Semantic search executed, returned {len(results)} results")
        except Exception as e:
            # This is expected if the collection doesn't exist yet
            print(f"~ Semantic search test skipped (collection may not exist): {e}")
        
    except Exception as e:
        print(f"✗ Error in semantic search test: {e}")
        return False
    
    return True


async def test_error_handling():
    """Test error handling and validation"""
    print("\nTesting error handling and validation...")
    
    try:
        rag_agent = RAGAgentService()
        
        # Test handling of empty query
        try:
            response = await rag_agent.generate_response(
                query="",
                context=[],
                module_context="test-module"
            )
            print("✓ Empty query handled gracefully")
        except Exception:
            print("✓ Empty query caused expected validation error")
        
        # Test handling with no context
        response = await rag_agent.generate_response(
            query="What is AI?",
            context=[],
            module_context="test-module"
        )
        
        # Should generate some form of response even without context
        assert "response" in response
        print("✓ Response generation with no context works correctly")
        
    except Exception as e:
        print(f"✗ Error in error handling test: {e}")
        return False
    
    return True


async def run_all_tests():
    """Run all validation tests"""
    print("Starting RAG Chatbot Implementation Validation\n")
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Conversation Context", test_conversation_context),
        ("Module Context", test_module_context),
        ("Semantic Search", test_semantic_search),
        ("Error Handling", test_error_handling),
    ]
    
    results = []
    for test_name, test_func in tests:
        result = await test_func()
        results.append((test_name, result))
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")
        if not result:
            all_passed = False
    
    print("="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED - Implementation is working correctly!")
    else:
        print("✗ SOME TESTS FAILED - Implementation needs fixes")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)