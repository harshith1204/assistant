"""
Example client demonstrating the single WebSocket architecture

This shows how to:
1. Connect to the WebSocket
2. Send typed messages (chat.send, memory.set, etc.)
3. Handle streamed events (memory.used, research.*, chat.token, etc.)
4. Maintain personalization across turns
"""

import asyncio
import json
import websockets
from typing import Optional

class AgentClient:
    """Simple client for the unified WebSocket agent"""
    
    def __init__(self, url: str = "ws://localhost:8000/ws"):
        self.url = url
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.user_id: Optional[str] = None
        self.conversation_id: Optional[str] = None
        
    async def connect(self):
        """Connect to the agent WebSocket"""
        self.websocket = await websockets.connect(self.url)
        
        # Listen for initial session info
        message = await self.websocket.recv()
        msg = json.loads(message)
        if msg["type"] == "session_info":
            self.user_id = msg["data"]["user_id"]
            self.conversation_id = msg["data"]["connection_id"]
            print(f"Connected as user: {self.user_id}")
    
    async def chat(self, message: str, use_research: bool = False):
        """Send a chat message and stream the response"""
        # Send chat.send message
        await self.websocket.send(json.dumps({
            "type": "chat.send",
            "data": {
                "conversation_id": self.conversation_id,
                "message": message,
                "use_web_search": use_research,
                "context_window": 4096
            }
        }))
        
        # Stream response events
        full_response = ""
        while True:
            msg = json.loads(await self.websocket.recv())
            event_type = msg["type"]
            data = msg.get("data", {})
            
            if event_type == "memory.used":
                # Shows what memories were injected
                print(f"[Memory] Using {len(data['items'])} memories")
                
            elif event_type == "research.started":
                print("[Research] Starting research...")
                
            elif event_type == "research.done":
                print(f"[Research] Complete: {data['sources']} sources analyzed")
                
            elif event_type == "chat.token":
                # Stream tokens as they arrive
                token = data.get("delta", "")
                full_response += token
                print(token, end="", flush=True)
                
            elif event_type == "memory.written":
                print(f"\n[Memory] Saved: {data.get('key', 'fact')}")
                
            elif event_type == "chat.final":
                # Final message with suggestions
                print("\n")
                suggestions = data.get("suggestions", [])
                if suggestions:
                    print("\nSuggested follow-ups:")
                    for s in suggestions:
                        print(f"  • {s}")
                return full_response
                
            elif event_type == "clarification_needed":
                print(f"\n[Clarification] {data['question']}")
                return None
                
            elif event_type == "error":
                print(f"\n[Error] {data['message']}")
                return None
    
    async def set_profile(self, key: str, value: str, priority: int = 50):
        """Set a profile fact"""
        await self.websocket.send(json.dumps({
            "type": "memory.set",
            "data": {
                "key": key,
                "value": value,
                "priority": priority
            }
        }))
        
        # Wait for confirmation
        msg = json.loads(await self.websocket.recv())
        if msg["type"] == "memory.written":
            print(f"[Profile] Set {key} = {value}")
    
    async def get_memories(self, limit: int = 5):
        """Get current memories"""
        await self.websocket.send(json.dumps({
            "type": "memory.get",
            "data": {
                "conversation_id": self.conversation_id,
                "scope": "both",
                "limit": limit
            }
        }))
        
        msg = json.loads(await self.websocket.recv())
        if msg["type"] == "memory.snapshot":
            items = msg["data"]["items"]
            print(f"[Memory] {len(items)} memories:")
            for item in items:
                print(f"  • {item.get('memory', item)[:100]}...")
            return items
        return []
    
    async def close(self):
        """Close the connection"""
        if self.websocket:
            await self.websocket.close()


async def example_conversation():
    """Example conversation showing the flow"""
    
    client = AgentClient()
    await client.connect()
    
    print("\n" + "="*60)
    print("EXAMPLE: Personal Travel Assistant")
    print("="*60)
    
    # Set profile preferences
    print("\n1. Setting profile preferences...")
    await client.set_profile("name", "Alice", priority=90)
    await client.set_profile("diet", "vegetarian", priority=80)
    await client.set_profile("tone", "casual and friendly", priority=70)
    
    # First query with research
    print("\n2. Asking about Goa travel (with research)...")
    response = await client.chat(
        "I'm planning a 3-day trip to Goa in November. What should I know?",
        use_research=True
    )
    
    # Follow-up using context
    print("\n\n3. Follow-up question (using context)...")
    response = await client.chat(
        "What about vegetarian restaurants there? Keep it budget-friendly.",
        use_research=False
    )
    
    # Check what memories were created
    print("\n\n4. Checking memories...")
    await client.get_memories(limit=8)
    
    # Another query to test personalization
    print("\n\n5. Testing personalization...")
    response = await client.chat(
        "Can you suggest a day-by-day itinerary?",
        use_research=False
    )
    
    await client.close()
    print("\n✅ Example complete!")


async def example_profile_update():
    """Example of profile updates and memory management"""
    
    client = AgentClient()
    await client.connect()
    
    print("\n" + "="*60)
    print("EXAMPLE: Profile Updates and Memory")
    print("="*60)
    
    # Natural profile update
    print("\n1. Natural profile update...")
    await client.chat(
        "Remember that I'm vegetarian and I prefer budget travel under $50/day"
    )
    
    # Explicit memory commands
    print("\n2. Asking to remember something...")
    await client.chat(
        "Also remember that I'm allergic to peanuts and I love beach activities"
    )
    
    # Test memory retrieval
    print("\n3. Testing memory retrieval...")
    await client.chat(
        "What do you know about my preferences?"
    )
    
    await client.close()
    print("\n✅ Example complete!")


if __name__ == "__main__":
    print("Single WebSocket Agent - Example Client")
    print("Make sure the server is running on localhost:8000")
    print("-" * 40)
    
    # Run example
    asyncio.run(example_conversation())
    # Uncomment to run profile example:
    # asyncio.run(example_profile_update())