"""
Test script to verify long-term context is maintained across different conversations
for the same user
"""

import asyncio
import json
import websockets
from typing import Dict, Any, Optional
import uuid

class CrossConversationTestClient:
    """Test client to verify cross-conversation memory"""
    
    def __init__(self, url: str = "ws://localhost:8000/ws"):
        self.url = url
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.user_id: Optional[str] = None
        self.messages_received = []
        
    async def connect(self):
        """Connect to WebSocket server"""
        self.websocket = await websockets.connect(self.url)
        print("✅ Connected to WebSocket")
        
        # Start receiving messages
        self.receive_task = asyncio.create_task(self.receive_messages())
        await asyncio.sleep(0.5)  # Wait for session info
        
    async def receive_messages(self):
        """Receive and store messages"""
        try:
            async for message in self.websocket:
                msg = json.loads(message)
                self.messages_received.append(msg)
                await self.handle_message(msg)
        except websockets.exceptions.ConnectionClosed:
            pass
            
    async def handle_message(self, msg: Dict[str, Any]):
        """Handle received message"""
        msg_type = msg.get("type")
        data = msg.get("data", {})
        
        if msg_type == "session_info":
            if not self.user_id:
                self.user_id = data.get("user_id")
                print(f"📌 User ID: {self.user_id}")
                
        elif msg_type == "memory.used":
            items = data.get("items", [])
            if items:
                print(f"🧠 Memory Used ({len(items)} items):")
                for item in items[:5]:
                    print(f"   [{item['level']}] {item['line'][:80]}...")
                    
        elif msg_type == "memory.written":
            level = data.get("level", "unknown")
            key = data.get("key", "")
            print(f"💾 Memory Written [{level}]: {key}")
            
        elif msg_type == "chat.token":
            print(data.get("delta", ""), end="", flush=True)
            
        elif msg_type == "chat.final":
            print("\n" + "-"*50)
            
        elif msg_type == "memory.snapshot":
            items = data.get("data", {}).get("items", [])
            print(f"📚 Memory Snapshot ({len(items)} items):")
            for item in items:
                memory = item.get("memory", item)
                level = item.get("memory_level", "unknown")
                print(f"   [{level}] {memory[:100]}...")
    
    async def send_chat(self, message: str, conversation_id: str, use_research: bool = False):
        """Send a chat message"""
        print(f"\n👤 USER ({conversation_id}): {message}")
        await self.websocket.send(json.dumps({
            "type": "chat.send",
            "data": {
                "conversation_id": conversation_id,
                "message": message,
                "use_web_search": use_research,
                "context_window": 4096,
                "user_id": self.user_id  # Ensure same user across conversations
            }
        }))
        # Wait for response
        await asyncio.sleep(3)
    
    async def set_profile(self, key: str, value: str, priority: int = 80):
        """Set a profile fact"""
        print(f"\n⚙️ Setting profile: {key} = {value}")
        await self.websocket.send(json.dumps({
            "type": "memory.set",
            "data": {
                "user_id": self.user_id,
                "key": key,
                "value": value,
                "priority": priority
            }
        }))
        await asyncio.sleep(1)
    
    async def check_memories(self, conversation_id: str):
        """Check what memories are available"""
        print(f"\n🔍 Checking memories for conversation {conversation_id}...")
        await self.websocket.send(json.dumps({
            "type": "memory.get",
            "data": {
                "conversation_id": conversation_id,
                "user_id": self.user_id,
                "scope": "both",
                "limit": 10,
                "query": "*"
            }
        }))
        await asyncio.sleep(2)
    
    async def close(self):
        """Close connection"""
        if self.receive_task:
            self.receive_task.cancel()
        if self.websocket:
            await self.websocket.close()


async def test_cross_conversation_memory():
    """Test that user memories persist across different conversations"""
    
    print("="*70)
    print("CROSS-CONVERSATION MEMORY TEST")
    print("="*70)
    
    client = CrossConversationTestClient()
    await client.connect()
    
    # Generate unique conversation IDs
    conv1 = f"test-conv-{uuid.uuid4().hex[:8]}"
    conv2 = f"test-conv-{uuid.uuid4().hex[:8]}"
    conv3 = f"test-conv-{uuid.uuid4().hex[:8]}"
    
    print(f"\n📝 Testing with 3 conversations:")
    print(f"  - Conversation 1: {conv1}")
    print(f"  - Conversation 2: {conv2}")
    print(f"  - Conversation 3: {conv3}")
    print(f"  - User: {client.user_id}")
    
    # ========== CONVERSATION 1: Establish user facts ==========
    print("\n" + "="*70)
    print("CONVERSATION 1: Establishing User Facts")
    print("="*70)
    
    # Set profile facts
    await client.set_profile("name", "Alice Chen", priority=90)
    await client.set_profile("diet", "strict vegetarian", priority=85)
    await client.set_profile("location", "San Francisco", priority=70)
    
    # Natural conversation with personal info
    await client.send_chat(
        "Hi! I'm planning a trip. Remember that I'm allergic to peanuts and shellfish. "
        "I also prefer boutique hotels over chains.",
        conv1
    )
    
    await client.send_chat(
        "Also, I love hiking and outdoor activities, but I have a bad knee so nothing too strenuous.",
        conv1
    )
    
    # Check what was stored
    await client.check_memories(conv1)
    
    # ========== CONVERSATION 2: Different topic, should remember user ==========
    print("\n" + "="*70)
    print("CONVERSATION 2: New Topic (Should Remember User)")
    print("="*70)
    
    await client.send_chat(
        "Can you recommend some restaurants for dinner tonight?",
        conv2
    )
    
    # The agent should remember dietary restrictions even in new conversation
    await client.send_chat(
        "What about something with good ambiance for a date?",
        conv2
    )
    
    # Check memories in this conversation
    await client.check_memories(conv2)
    
    # ========== CONVERSATION 3: Verify all facts are accessible ==========
    print("\n" + "="*70)
    print("CONVERSATION 3: Comprehensive Check")
    print("="*70)
    
    await client.send_chat(
        "I need help planning a weekend getaway. What do you know about my preferences?",
        conv3
    )
    
    # This should trigger retrieval of all user facts
    await client.send_chat(
        "Based on everything you know about me, what would be the perfect destination?",
        conv3
    )
    
    # Final memory check
    await client.check_memories(conv3)
    
    # ========== VALIDATION ==========
    print("\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)
    
    # Check if user memories were consistently used across conversations
    memory_used_events = [
        msg for msg in client.messages_received 
        if msg.get("type") == "memory.used"
    ]
    
    print(f"\n✅ Memory retrieval events: {len(memory_used_events)}")
    
    # Check if profile facts were used
    profile_memories_used = 0
    user_memories_used = 0
    
    for event in memory_used_events:
        items = event.get("data", {}).get("items", [])
        for item in items:
            if item.get("level") == "profile":
                profile_memories_used += 1
            elif item.get("level") == "user":
                user_memories_used += 1
    
    print(f"✅ Profile memories used: {profile_memories_used}")
    print(f"✅ User memories used: {user_memories_used}")
    
    # Success criteria
    success = profile_memories_used > 0 and (
        user_memories_used > 0 or profile_memories_used >= 6  # At least facts used multiple times
    )
    
    if success:
        print("\n🎉 SUCCESS: User context maintained across conversations!")
    else:
        print("\n❌ FAILURE: User context not properly maintained")
        print("   Check that:")
        print("   - User memories are stored with memory_type='user' or 'both'")
        print("   - Profile facts are being retrieved")
        print("   - search_memory uses search_scope='both'")
    
    await client.close()
    
    return success


async def test_profile_persistence():
    """Test that profile facts persist and are always injected"""
    
    print("\n" + "="*70)
    print("PROFILE PERSISTENCE TEST")
    print("="*70)
    
    client = CrossConversationTestClient()
    await client.connect()
    
    # Set comprehensive profile
    print("\n📝 Setting comprehensive profile...")
    await client.set_profile("name", "Bob Smith", 95)
    await client.set_profile("occupation", "Software Engineer", 80)
    await client.set_profile("hobby", "Photography", 70)
    await client.set_profile("dietary_restriction", "Gluten-free", 90)
    await client.set_profile("preferred_tone", "Professional but friendly", 75)
    
    # Test in multiple conversations
    for i in range(3):
        conv_id = f"profile-test-{i}"
        print(f"\n🔄 Testing conversation {i+1}: {conv_id}")
        
        await client.send_chat(
            "Hello, can you tell me what you know about me?",
            conv_id
        )
        
        # Check if profile was used
        recent_memory_events = [
            msg for msg in client.messages_received[-5:]
            if msg.get("type") == "memory.used"
        ]
        
        if recent_memory_events:
            last_event = recent_memory_events[-1]
            items = last_event.get("data", {}).get("items", [])
            profile_items = [
                item for item in items 
                if item.get("level") == "profile"
            ]
            print(f"   Profile facts used: {len(profile_items)}")
        
    await client.close()
    
    print("\n✅ Profile persistence test complete")


if __name__ == "__main__":
    print("Testing Cross-Conversation Memory")
    print("Make sure the server is running on localhost:8000")
    print("-" * 40)
    
    # Run the main test
    asyncio.run(test_cross_conversation_memory())
    
    # Run profile persistence test
    asyncio.run(test_profile_persistence())