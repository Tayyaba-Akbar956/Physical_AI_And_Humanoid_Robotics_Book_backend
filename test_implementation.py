#!/usr/bin/env python3
"""
Comprehensive Testing Script for RAG Chatbot Implementation
This script demonstrates and tests all implemented features
"""

import asyncio
import sys
import os
from typing import Dict, Any, List
from uuid import UUID, uuid4
import json
import time

# Add the backend src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.services.rag_agent import RAGAgentService
from src.services.session_manager import SessionManagementService
from src.services.semantic_search import SemanticSearchService
from src.services.conversational_context_manager import ConversationalContextManager
from src.api.chat_endpoints import ChatQueryRequest, chat_query
from src.main import app
from fastapi.testclient import TestClient


def test_basic_rag_functionality():
    """Test basic RAG functionality"""
    print("🧪 Testing Basic RAG Functionality...")
    
    try:
        rag_agent = RAGAgentService()
        
        # Test basic query processing
        response = rag_agent.generate_response(
            query="What is artificial intelligence?",
            context=[{
                "id": "test1",
                "score": 0.9,
                "module_id": "module-1-introduction",
                "chapter_id": "ch-1-1",
                "section_id": "definition",
                "content": "Artificial Intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of 'intelligent agents'.",
                "hierarchy_path": "module-1-introduction/ch-1-1/definition",
                "content_type": "text",
                "metadata": {}
            }]
        )
        
        assert "response" in response
        assert len(response["response"]) > 0
        print("✅ Basic RAG functionality test passed")
        print(f"   Response preview: {response['response'][:100]}...")
        
    except Exception as e:
        print(f"❌ Basic RAG functionality test failed: {e}")
        return False
    
    return True


def test_conversation_context():
    """Test conversation context management"""
    print("\n🧪 Testing Conversation Context Management...")
    
    try:
        # Test session management
        session_manager = SessionManagementService()
        student_id = uuid4()
        
        # Create a session
        session_info = session_manager.create_session(
            student_id=student_id,
            current_module_context="module-2-ros2"
        )
        
        assert session_info is not None
        assert session_info["is_active"] == True
        print("✅ Session creation works correctly")
        
        # Add messages to the session
        user_msg = session_manager.add_message_to_session(
            session_id=session_info["id"],
            sender_type="student",
            content="What is ROS 2?",
            topic_anchored="introduction"
        )
        
        assert user_msg is not None
        print("✅ Adding user message to session works correctly")
        
        ai_msg = session_manager.add_message_to_session(
            session_id=session_info["id"],
            sender_type="ai_agent",
            content="ROS 2 is a flexible framework for writing robot software...",
            parent_message_id=user_msg["id"]
        )
        
        assert ai_msg is not None
        print("✅ Adding AI message to session works correctly")
        
        # Test conversation context retrieval
        context = session_manager.get_conversation_context(session_info["id"])
        assert len(context) >= 2
        print("✅ Retrieving conversation context works correctly")
        
        # Test conversation summary
        summary = session_manager.get_conversation_summary(session_info["id"])
        assert "total_exchanges" in summary
        print("✅ Conversation summary generation works correctly")
        
    except Exception as e:
        print(f"❌ Conversation context test failed: {e}")
        return False
    
    return True


def test_module_awareness():
    """Test module-aware context features"""
    print("\n🧪 Testing Module-Aware Context Features...")
    
    try:
        rag_agent = RAGAgentService()
        
        # Test content retrieval with module filtering
        content = rag_agent.get_relevant_content(
            query="robotics concepts",
            top_k=3,
            module_filter="module-2-ros2"
        )
        
        print(f"✅ Retrieved {len(content)} content items with module filtering")
        
        # Test module concept detection
        analysis = rag_agent.detect_module_related_concepts(
            "How does ROS 2 differ from ROS 1?",
            "module-2-ros2",
            content
        )
        
        assert "modules_mentioned" in analysis
        print("✅ Module concept detection works correctly")
        print(f"   Analysis: {json.dumps(analysis, indent=2)[:200]}...")
        
    except Exception as e:
        print(f"❌ Module-aware context test failed: {e}")
        return False
    
    return True


def test_api_endpoints():
    """Test API endpoints using FastAPI TestClient"""
    print("\n🧪 Testing API Endpoints...")
    
    try:
        client = TestClient(app)
        
        # Test health endpoint
        response = client.get("/health")
        assert response.status_code == 200
        print("✅ Health endpoint works correctly")
        
        # Test system health endpoint
        response = client.get("/api/system/health")
        assert response.status_code == 200
        print("✅ System health endpoint works correctly")
        
        # Test conversation endpoints
        response = client.get("/api/system/status")
        assert response.status_code == 200
        print("✅ System status endpoint works correctly")
        
    except Exception as e:
        print(f"❌ API endpoint test failed: {e}")
        return False
    
    return True


def test_conversational_context_management():
    """Test advanced conversational context management"""
    print("\n🧪 Testing Advanced Conversational Context Management...")
    
    try:
        # Test conversational context manager
        context_manager = ConversationalContextManager()
        
        # This would require an actual session to test properly
        # For now, let's test the methods that don't require sessions
        topic_info = context_manager.track_conversation_topic([
            {"role": "user", "content": "What is ROS 2?", "timestamp": time.time()},
            {"role": "assistant", "content": "ROS 2 is a robot operating system...", "timestamp": time.time()}
        ])
        
        assert "current_topic" in topic_info
        print("✅ Topic tracking works correctly")
        
        # Test follow-up resolution
        follow_up_resolution = context_manager.resolve_follow_up_question(
            "Tell me more about it",
            [{"role": "user", "content": "What is ROS 2?", "timestamp": time.time()}]
        )
        
        assert "is_follow_up" in follow_up_resolution
        print(f"✅ Follow-up resolution works correctly: {follow_up_resolution['is_follow_up']}")
        
    except Exception as e:
        print(f"❌ Conversational context management test failed: {e}")
        return False
    
    return True


def test_full_conversation_flow():
    """Test a full conversation flow"""
    print("\n🧪 Testing Full Conversation Flow...")
    
    try:
        # Initialize services
        session_manager = SessionManagementService()
        rag_agent = RAGAgentService()
        
        # Create a session
        student_id = uuid4()
        session_info = session_manager.create_session(
            student_id=student_id,
            current_module_context="module-3-simulation"
        )
        
        assert session_info is not None
        session_id = session_info["id"]
        print("✅ Session created for conversation flow test")
        
        # Simulate a conversation
        queries = [
            "What is robot simulation?",
            "How does Gazebo work?",
            "Can you explain that differently?",
            "What about physics engines?"
        ]
        
        for i, query in enumerate(queries):
            print(f"   Query {i+1}: {query}")
            
            # Get response from RAG agent with conversation context
            response_data = rag_agent.answer_question(
                query=query,
                session_id=str(session_id),
                module_context="module-3-simulation",
                conversation_context=session_manager.get_conversation_context(session_id)
            )
            
            assert "response" in response_data
            assert len(response_data["response"]) > 0
            
            # Add messages to session
            session_manager.add_message_to_session(
                session_id=session_id,
                sender_type="student",
                content=query
            )
            
            session_manager.add_message_to_session(
                session_id=session_id,
                sender_type="ai_agent",
                content=response_data["response"][:100] + "..."  # Truncate for storage
            )
        
        print("✅ Full conversation flow completed successfully")
        
        # Verify conversation history
        history = session_manager.get_conversation_history(session_id)
        assert len(history) == len(queries) * 2  # 2 messages per query (student + AI)
        print(f"✅ Conversation history preserved: {len(history)} messages")
        
    except Exception as e:
        print(f"❌ Full conversation flow test failed: {e}")
        return False
    
    return True


def run_comprehensive_tests():
    """Run all tests and provide a summary"""
    print("="*70)
    print("🤖 COMPREHENSIVE RAG CHATBOT IMPLEMENTATION TEST SUITE")
    print("="*70)
    
    tests = [
        ("Basic RAG Functionality", test_basic_rag_functionality),
        ("Conversation Context", test_conversation_context),
        ("Module-Aware Context", test_module_awareness),
        ("API Endpoints", test_api_endpoints),
        ("Conversational Context Management", test_conversational_context_management),
        ("Full Conversation Flow", test_full_conversation_flow),
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
        print("🎉 ALL TESTS PASSED - Implementation is working correctly!")
        print("\n💡 To run the application:")
        print("   1. Install requirements: pip install -r backend/requirements.txt")
        print("   2. Set up environment variables in a .env file")
        print("   3. Run the application: uvicorn src.main:app --reload --port 8000")
        print("   4. Access the API at http://localhost:8000")
    else:
        print("⚠️  SOME TESTS FAILED - Implementation needs fixes")
    print("="*70)
    
    return all_passed


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)