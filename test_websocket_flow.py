"""
Test script for the complete WebSocket flow with single socket architecture
"""

import asyncio
import json
import websockets
from typing import Dict, Any

class WebSocketTestClient:
    """Test client for the WebSocket flow"""
    
    def __init__(self, url: str = "ws://localhost:8000/ws"):
        self.url = url
        self.user_id = None
        self.conversation_id = None
        
    async def connect(self):
        """Connect to WebSocket server"""
        self.websocket = await websockets.connect(self.url)
        print("✅ Connected to WebSocket")
        
    async def send_message(self, msg_type: str, data: Dict[str, Any]):
        """Send a typed message"""
        message = {"type": msg_type, "data": data}
        await self.websocket.send(json.dumps(message))
        print(f"📤 Sent: {msg_type}")
        
    async def receive_messages(self):
        """Receive and print messages"""
        try:
            async for message in self.websocket:
                msg = json.loads(message)
                await self.handle_message(msg)
        except websockets.exceptions.ConnectionClosed:
            print("❌ Connection closed")
            
    async def handle_message(self, msg: Dict[str, Any]):
        """Handle received message"""
        msg_type = msg.get("type")
        data = msg.get("data", {})
        
        if msg_type == "session_info":
            self.user_id = data.get("user_id")
            self.conversation_id = data.get("connection_id")
            print(f"📥 Session Info: user_id={self.user_id}")
            
        elif msg_type == "memory.used":
            items = data.get("items", [])
            print(f"🧠 Memory Used ({len(items)} items):")
            for item in items[:3]:
                print(f"   [{item['level']}] {item['line'][:50]}...")
                
        elif msg_type == "research.started":
            print("🔍 Research started...")
            
        elif msg_type == "research.chunk":
            chunk_data = data.get("data", {})
            print(f"   Research progress: {chunk_data}")
            
        elif msg_type == "research.done":
            print(f"✅ Research complete: {data.get('sources', 0)} sources, {data.get('findings', 0)} findings")
            
        elif msg_type == "chat.token":
            # Stream tokens (print inline)
            print(data.get("delta", ""), end="", flush=True)
            
        elif msg_type == "memory.written":
            print(f"\n💾 Memory written: [{data.get('level')}] {data.get('key', 'fact')}")
            
        elif msg_type == "memory.summary_created":
            print("📝 Rolling summary created")
            
        elif msg_type == "chat.final":
            print("\n" + "="*50)
            message = data.get("message", {})
            print(f"🤖 Final Response: {message.get('content', '')[:200]}...")
            suggestions = data.get("suggestions", [])
            if suggestions:
                print("💡 Suggestions:")
                for s in suggestions:
                    print(f"   - {s}")
            print("="*50)
            
        elif msg_type == "clarification_needed":
            print(f"❓ Clarification needed: {data.get('question')}")
            print(f"   Confidence: {data.get('confidence', 0):.2f}")
            
        elif msg_type == "error":
            print(f"❌ Error: {data.get('message')}")
            
        else:
            print(f"📥 {msg_type}: {data}")
            
    async def test_flow(self):
        """Test the complete flow"""
        await self.connect()
        
        # Start receiving messages in background
        receive_task = asyncio.create_task(self.receive_messages())
        
        # Wait a bit for connection
        await asyncio.sleep(1)
        
        print("\n" + "="*50)
        print("Testing Complete WebSocket Flow")
        print("="*50 + "\n")
        
        # Test 1: Profile update
        print("📝 Test 1: Profile Update")
        await self.send_message("chat.send", {
            "conversation_id": "test-conv-1",
            "message": "Remember that my name is Alice and I prefer vegetarian food",
            "use_web_search": False,
            "context_window": 4096
        })
        await asyncio.sleep(5)
        
        # Test 2: Memory retrieval
        print("\n📝 Test 2: Memory Check")
        await self.send_message("memory.get", {
            "conversation_id": "test-conv-1",
            "scope": "both",
            "limit": 5
        })
        await asyncio.sleep(2)
        
        # Test 3: Research request
        print("\n📝 Test 3: Research Request")
        await self.send_message("chat.send", {
            "conversation_id": "test-conv-1",
            "message": "Research the best vegetarian restaurants in Goa for November travel",
            "use_web_search": True,
            "context_window": 4096
        })
        await asyncio.sleep(10)
        
        # Test 4: Follow-up with context
        print("\n📝 Test 4: Follow-up Question")
        await self.send_message("chat.send", {
            "conversation_id": "test-conv-1",
            "message": "What about budget-friendly options under 500 rupees?",
            "use_web_search": False,
            "context_window": 4096
        })
        await asyncio.sleep(5)
        
        # Test 5: Explicit memory set
        print("\n📝 Test 5: Explicit Memory Set")
        await self.send_message("memory.set", {
            "key": "budget_preference",
            "value": "Under 500 rupees per meal",
            "priority": 70
        })
        await asyncio.sleep(2)
        
        # Test 6: Cancel test (start long task and cancel)
        print("\n📝 Test 6: Cancellation Test")
        cancel_task = asyncio.create_task(self.send_message("chat.send", {
            "conversation_id": "test-conv-2",
            "message": "Research everything about starting a tech company in India",
            "use_web_search": True,
            "context_window": 4096
        }))
        await asyncio.sleep(2)
        await self.send_message("agent.cancel", {
            "conversation_id": "test-conv-2"
        })
        await asyncio.sleep(2)
        
        print("\n✅ All tests completed!")
        
        # Close connection
        receive_task.cancel()
        await self.websocket.close()

async def main():
    """Main test function"""
    client = WebSocketTestClient()
    try:
        await client.test_flow()
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    print("Starting WebSocket Flow Test...")
    print("Make sure the server is running on localhost:8000")
    print("-" * 50)
    asyncio.run(main())