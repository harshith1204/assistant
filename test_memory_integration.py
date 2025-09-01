"""
Integration test for cross-conversation memory without running the full server
"""

import asyncio
import uuid
from app.core.memory_manager import MemoryManager
from app.core.chat_engine import ChatEngine
from app.chat_models import ChatRequest, MessageRole
import json

async def test_cross_conversation_memory():
    """Test that memories persist across conversations for the same user"""
    
    print("="*60)
    print("TESTING CROSS-CONVERSATION MEMORY")
    print("="*60)
    
    # Initialize components
    chat_engine = ChatEngine()
    memory_manager = chat_engine.memory_manager
    
    # Create a consistent user_id
    user_id = f"test_user_{uuid.uuid4().hex[:8]}"
    print(f"\nUser ID: {user_id}")
    
    # Create different conversation IDs
    conv1 = f"conv_{uuid.uuid4().hex[:8]}"
    conv2 = f"conv_{uuid.uuid4().hex[:8]}"
    conv3 = f"conv_{uuid.uuid4().hex[:8]}"
    
    print(f"Conversation 1: {conv1}")
    print(f"Conversation 2: {conv2}")
    print(f"Conversation 3: {conv3}")
    
    # ========== STEP 1: Set profile facts ==========
    print("\n" + "-"*60)
    print("STEP 1: Setting Profile Facts")
    print("-"*60)
    
    await memory_manager.set_profile_fact(user_id, "name", "Alice Test", 90)
    await memory_manager.set_profile_fact(user_id, "diet", "vegetarian", 85)
    await memory_manager.set_profile_fact(user_id, "location", "San Francisco", 70)
    await memory_manager.set_profile_fact(user_id, "allergies", "peanuts, shellfish", 95)
    
    profile = await memory_manager.get_profile(user_id)
    print(f"Profile facts stored: {len(profile)}")
    for fact in profile:
        if isinstance(fact, dict):
            print(f"  - {fact.get('memory', fact)[:100]}")
    
    # ========== STEP 2: Conversation 1 - Add memories ==========
    print("\n" + "-"*60)
    print("STEP 2: Conversation 1 - Adding User Memories")
    print("-"*60)
    
    # Simulate chat in conversation 1
    request1 = ChatRequest(
        conversation_id=conv1,
        message="I love hiking and outdoor activities. Remember I have a bad knee.",
        user_id=user_id,
        use_long_term_memory=True
    )
    
    events = []
    async for event in chat_engine.stream_message(request1, user_id):
        events.append(event)
        if event.get("type") == "memory.used":
            print(f"Memory used: {len(event.get('items', []))} items")
        elif event.get("type") == "memory.written":
            print(f"Memory written: {event.get('level')}")
    
    # Add another memory
    request2 = ChatRequest(
        conversation_id=conv1,
        message="Also remember I prefer boutique hotels over chain hotels when traveling.",
        user_id=user_id,
        use_long_term_memory=True
    )
    
    async for event in chat_engine.stream_message(request2, user_id):
        pass  # Process but don't print
    
    # ========== STEP 3: Conversation 2 - Check memory retrieval ==========
    print("\n" + "-"*60)
    print("STEP 3: Conversation 2 - Testing Memory Retrieval")
    print("-"*60)
    
    # Different conversation, same user
    request3 = ChatRequest(
        conversation_id=conv2,  # DIFFERENT conversation
        message="Can you recommend restaurants for dinner? Keep my preferences in mind.",
        user_id=user_id,
        use_long_term_memory=True
    )
    
    memory_used_conv2 = []
    async for event in chat_engine.stream_message(request3, user_id):
        if event.get("type") == "memory.used":
            memory_used_conv2 = event.get("items", [])
            print(f"Memories retrieved in Conv2: {len(memory_used_conv2)}")
            
            # Check what memories were used
            profile_memories = [m for m in memory_used_conv2 if m.get("level") == "profile"]
            user_memories = [m for m in memory_used_conv2 if m.get("level") == "user"]
            
            print(f"  - Profile memories: {len(profile_memories)}")
            print(f"  - User memories: {len(user_memories)}")
            
            # Print some examples
            if profile_memories:
                print("\n  Profile facts used:")
                for m in profile_memories[:3]:
                    print(f"    • {m.get('line', '')[:80]}")
            
            if user_memories:
                print("\n  User memories used:")
                for m in user_memories[:3]:
                    print(f"    • {m.get('line', '')[:80]}")
    
    # ========== STEP 4: Conversation 3 - Comprehensive check ==========
    print("\n" + "-"*60)
    print("STEP 4: Conversation 3 - Comprehensive Memory Check")
    print("-"*60)
    
    # Third conversation, should have access to all user facts
    request4 = ChatRequest(
        conversation_id=conv3,  # YET ANOTHER conversation
        message="Based on everything you know about me, suggest a weekend trip.",
        user_id=user_id,
        use_long_term_memory=True
    )
    
    memory_used_conv3 = []
    async for event in chat_engine.stream_message(request4, user_id):
        if event.get("type") == "memory.used":
            memory_used_conv3 = event.get("items", [])
            print(f"Memories retrieved in Conv3: {len(memory_used_conv3)}")
            
            profile_memories = [m for m in memory_used_conv3 if m.get("level") == "profile"]
            user_memories = [m for m in memory_used_conv3 if m.get("level") == "user"]
            
            print(f"  - Profile memories: {len(profile_memories)}")
            print(f"  - User memories: {len(user_memories)}")
    
    # ========== STEP 5: Direct memory search test ==========
    print("\n" + "-"*60)
    print("STEP 5: Direct Memory Search Across Conversations")
    print("-"*60)
    
    # Search for user memories across all conversations
    all_user_memories = await memory_manager.search_memory(
        query="preferences hotels hiking diet",
        conversation_id=None,  # Don't limit to a conversation
        user_id=user_id,
        limit=20,
        search_scope="user"  # Only user-level memories
    )
    
    print(f"Total user memories found: {len(all_user_memories)}")
    for mem in all_user_memories[:5]:
        memory_text = mem.get("memory", mem.get("content", str(mem)))
        print(f"  - {memory_text[:100]}...")
    
    # ========== VALIDATION ==========
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    
    # Success criteria
    success_criteria = {
        "profile_facts_stored": len(profile) >= 4,
        "memories_in_conv2": len(memory_used_conv2) > 0,
        "memories_in_conv3": len(memory_used_conv3) > 0,
        "cross_conv_memories": len(all_user_memories) >= 3
    }
    
    print("\nSuccess Criteria:")
    for criterion, passed in success_criteria.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {criterion}: {passed}")
    
    all_passed = all(success_criteria.values())
    
    if all_passed:
        print("\n🎉 SUCCESS: Cross-conversation memory is working!")
        print("User memories are accessible across different conversations.")
    else:
        print("\n❌ FAILURE: Cross-conversation memory not working properly")
        print("Check that:")
        print("  1. User memories are stored with memory_type='user' or 'both'")
        print("  2. Profile facts are being retrieved with user_id")
        print("  3. search_memory is searching with scope='both' or 'user'")
        print("  4. The user_id is consistent across all operations")
    
    return all_passed


if __name__ == "__main__":
    print("Running Integration Test for Cross-Conversation Memory")
    print("-" * 60)
    
    # Run the test
    result = asyncio.run(test_cross_conversation_memory())
    
    if result:
        print("\n✅ Integration test PASSED")
    else:
        print("\n❌ Integration test FAILED")
        print("Review the implementation and check logs for issues.")