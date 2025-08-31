#!/usr/bin/env python3
"""Test script for the chat platform"""

import asyncio
import json
from app.chat_models import ChatRequest, ChatMessage, MessageRole
from app.core.chat_engine import ChatEngine
from app.core.memory_manager import MemoryManager

async def test_chat_platform():
    """Test the chat platform components"""
    
    print("🧪 Testing Chat Platform Components")
    print("=" * 50)
    
    # Test 1: Memory Manager
    print("\n1. Testing Memory Manager...")
    try:
        memory_manager = MemoryManager()
        
        # Add a memory
        result = await memory_manager.add_to_memory(
            conversation_id="test_conv_1",
            content="User prefers technical explanations",
            metadata={"type": "preference"}
        )
        print("   ✅ Memory added successfully")
        
        # Search memory
        memories = await memory_manager.search_memory(
            query="technical",
            conversation_id="test_conv_1"
        )
        print(f"   ✅ Found {len(memories)} relevant memories")
        
    except Exception as e:
        print(f"   ❌ Memory Manager test failed: {e}")
    
    # Test 2: Chat Engine
    print("\n2. Testing Chat Engine...")
    try:
        chat_engine = ChatEngine()
        
        # Create a chat request
        request = ChatRequest(
            message="Hello, can you help me understand how memory works in this system?",
            use_long_term_memory=True,
            use_web_search=False
        )
        
        # Process message
        response = await chat_engine.process_message(request)
        print(f"   ✅ Chat response generated")
        print(f"   - Conversation ID: {response.conversation_id}")
        print(f"   - Response length: {len(response.message.content)} chars")
        print(f"   - Suggestions: {len(response.suggestions)} suggestions")
        
    except Exception as e:
        print(f"   ❌ Chat Engine test failed: {e}")
    
    # Test 3: Research Integration
    print("\n3. Testing Research Integration...")
    try:
        request = ChatRequest(
            message="What are the latest trends in artificial intelligence?",
            use_long_term_memory=True,
            use_web_search=True
        )
        
        # This would trigger research in a real scenario
        needs_research = await chat_engine._needs_research(
            request.message,
            {}
        )
        print(f"   ✅ Research detection: {needs_research}")
        
    except Exception as e:
        print(f"   ❌ Research integration test failed: {e}")
    
    # Test 4: Conversation Management
    print("\n4. Testing Conversation Management...")
    try:
        # List conversations
        conversations = await chat_engine.list_conversations()
        print(f"   ✅ Found {len(conversations)} conversations")
        
        # Get specific conversation if exists
        if response.conversation_id:
            conv = await chat_engine.get_conversation(response.conversation_id)
            if conv:
                print(f"   ✅ Retrieved conversation with {conv.get_message_count()} messages")
        
    except Exception as e:
        print(f"   ❌ Conversation management test failed: {e}")
    
    print("\n" + "=" * 50)
    print("✨ Chat Platform Tests Complete!")
    print("\nTo start the full application, run:")
    print("  ./run_chat.sh")
    print("\nThen open: http://localhost:8000/chat")

if __name__ == "__main__":
    asyncio.run(test_chat_platform())