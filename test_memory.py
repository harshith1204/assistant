#!/usr/bin/env python3
"""Test script to verify memory persistence and retrieval"""

import asyncio
import httpx
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"

async def test_memory_system():
    """Test the memory system with a specific user"""
    
    # Test user ID (simulating a persistent user)
    user_id = "test_user_001"
    conversation_id_1 = f"conv_{datetime.now().timestamp()}_1"
    conversation_id_2 = f"conv_{datetime.now().timestamp()}_2"
    
    async with httpx.AsyncClient() as client:
        print("\n=== Memory System Test ===\n")
        
        # 1. Send messages in first conversation
        print("1. Sending messages in first conversation...")
        messages = [
            "My name is John and I work at TechCorp",
            "I prefer Python over Java for backend development",
            "Remember that I'm allergic to peanuts",
            "I live in San Francisco and love hiking"
        ]
        
        for msg in messages:
            response = await client.post(
                f"{BASE_URL}/chat/message",
                json={
                    "message": msg,
                    "conversation_id": conversation_id_1,
                    "user_id": user_id,
                    "use_long_term_memory": True
                }
            )
            if response.status_code == 200:
                print(f"  ✓ Message sent: {msg[:50]}...")
            else:
                print(f"  ✗ Failed to send message: {response.text}")
            await asyncio.sleep(1)  # Small delay between messages
        
        # 2. Check memory stats
        print("\n2. Checking memory stats...")
        response = await client.get(f"{BASE_URL}/chat/memory/stats?user_id={user_id}")
        if response.status_code == 200:
            stats = response.json()
            print(f"  Total memories: {stats.get('total_memories', 0)}")
            print(f"  Short-term conversations: {stats.get('short_term_conversations', 0)}")
            print(f"  Cache size: {stats.get('cache_size', 0)}")
        
        # 3. Debug memory for user
        print(f"\n3. Debugging memory for user {user_id}...")
        response = await client.get(
            f"{BASE_URL}/chat/memory/debug/{user_id}?conversation_id={conversation_id_1}"
        )
        if response.status_code == 200:
            debug_data = response.json()
            print(f"  Total memories: {debug_data.get('total_memories', 0)}")
            print(f"  Profile facts: {debug_data.get('profile_facts', 0)}")
            print(f"  Active conversations: {len(debug_data.get('active_conversations', []))}")
            
            if debug_data.get('profile'):
                print("\n  Profile facts found:")
                for fact in debug_data['profile'][:5]:
                    if isinstance(fact, dict):
                        memory = fact.get('memory', fact.get('content', str(fact)))
                    else:
                        memory = str(fact)
                    print(f"    - {memory[:100]}")
            
            if debug_data.get('recent_memories'):
                print("\n  Recent memories:")
                for mem in debug_data['recent_memories'][:5]:
                    if isinstance(mem, dict):
                        memory = mem.get('memory', mem.get('content', str(mem)))
                    else:
                        memory = str(mem)
                    print(f"    - {memory[:100]}")
        
        # 4. Start NEW conversation with same user
        print(f"\n4. Starting NEW conversation with same user...")
        response = await client.post(
            f"{BASE_URL}/chat/message",
            json={
                "message": "What do you remember about me?",
                "conversation_id": conversation_id_2,  # NEW conversation
                "user_id": user_id,  # SAME user
                "use_long_term_memory": True
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n  Assistant response (should include memories):")
            print(f"  {result['response'][:500]}...")
            
            # Check if response contains our facts
            response_lower = result['response'].lower()
            facts_found = []
            if 'john' in response_lower:
                facts_found.append("Name: John")
            if 'techcorp' in response_lower:
                facts_found.append("Company: TechCorp")
            if 'python' in response_lower:
                facts_found.append("Preference: Python")
            if 'peanut' in response_lower:
                facts_found.append("Allergy: Peanuts")
            if 'san francisco' in response_lower:
                facts_found.append("Location: San Francisco")
            
            print("\n  Facts retrieved from memory:")
            if facts_found:
                for fact in facts_found:
                    print(f"    ✓ {fact}")
            else:
                print("    ✗ No facts found in response!")
        
        # 5. Final memory debug
        print(f"\n5. Final memory check...")
        response = await client.get(
            f"{BASE_URL}/chat/memory/debug/{user_id}?conversation_id={conversation_id_2}"
        )
        if response.status_code == 200:
            debug_data = response.json()
            print(f"  Total memories after test: {debug_data.get('total_memories', 0)}")
            print(f"  Active conversations: {debug_data.get('active_conversations', [])}")
        
        print("\n=== Test Complete ===\n")

if __name__ == "__main__":
    asyncio.run(test_memory_system())