"""Simplified WebSocket handler for keyword-based conversational chat"""

import json
import asyncio
from typing import Dict, Set, Optional
from datetime import datetime, timezone
from fastapi import WebSocket, WebSocketDisconnect
import structlog
import uuid

from app.chat_models import (
    ChatRequest, WebSocketMessage,
    MessageRole, MessageType
)
from app.core.chat_engine import ChatEngine

logger = structlog.get_logger()


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class ConnectionManager:
    """Simple connection manager for WebSocket chat"""

    def __init__(self):
        print("🔧 Initializing ConnectionManager...")
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_user: Dict[str, str] = {}  # connection_id -> user_id
        try:
            self.chat_engine = ChatEngine()
            print("✅ Chat engine initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize chat engine: {e}")
            self.chat_engine = None

    async def connect(
        self,
        websocket: WebSocket,
        connection_id: str,
        user_id: Optional[str] = None
    ):
        """Accept and register a new connection"""
        await websocket.accept()
        self.active_connections[connection_id] = websocket

        # Generate user_id if not provided
        if not user_id:
            user_id = f"anon_{uuid.uuid4().hex[:8]}"

        self.connection_user[connection_id] = user_id

        # Send connection confirmation
        await self.send_message(
            connection_id,
            WebSocketMessage(
                type="connected",
                data={
                    "connection_id": connection_id,
                    "user_id": user_id,
                    "message": "Connected to conversational chat",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        )

        logger.info("WebSocket connected", connection_id=connection_id, user_id=user_id)

    def disconnect(self, connection_id: str):
        """Remove a connection"""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
            user_id = self.connection_user.pop(connection_id, None)
            logger.info("WebSocket disconnected", connection_id=connection_id, user_id=user_id)

    async def send_message(
        self,
        connection_id: str,
        message: WebSocketMessage
    ):
        """Send a message to a specific connection"""
        print(f"📤 Sending message to {connection_id}: type={message.type}")
        websocket = self.active_connections.get(connection_id)
        if websocket:
            try:
                message_data = message.model_dump()
                # Serialize with custom encoder to handle datetime objects
                json_string = json.dumps(message_data, cls=DateTimeEncoder)
                print(f"📦 Message data: {json_string[:100]}...")
                await websocket.send_text(json_string)
                print(f"✅ Message sent successfully to {connection_id}")
            except Exception as e:
                print(f"❌ Failed to send message to {connection_id}: {e}")
                logger.error("Failed to send message", connection_id=connection_id, error=str(e))
                self.disconnect(connection_id)
        else:
            print(f"⚠️ No active connection found for {connection_id}")
    
    async def handle_chat_message(
        self,
        connection_id: str,
        data: Dict,
        user_id: Optional[str] = None
    ):
        """Handle incoming chat message with simple keyword detection"""
        try:
            print(f"🤖 Processing chat message: {data}")
            logger.info("Handling chat message", connection_id=connection_id, data=data)
            user_id = user_id or self.connection_user.get(connection_id)

            # Handle nested message structure
            if "data" in data and isinstance(data["data"], dict):
                message_data = data["data"]
                message_text = message_data.get("message", "")
                conversation_id = message_data.get("conversation_id", f"{uuid.uuid4().hex[:8]}")
                stream = message_data.get("stream", True)
            else:
                message_text = data.get("message", "")
                conversation_id = data.get("conversation_id", f"conv_{uuid.uuid4().hex[:8]}")
                stream = data.get("stream", True)

            print(f"📝 Message text: '{message_text}', User: {user_id}, Conversation: {conversation_id}")
            logger.info("Extracted message text", connection_id=connection_id, message_text=message_text, user_id=user_id)

            # Simple keyword detection for research vs general chat
            is_research = self._is_research_query(message_text)

            # Create chat request
            request = ChatRequest(
                message=message_text,
                conversation_id=conversation_id,
                use_web_search=is_research,
                stream=stream
            )

            # Send typing indicator
            await self.send_message(
                connection_id,
                WebSocketMessage(type="typing", data={"status": "thinking"})
            )

            # Check if chat engine is available
            if not self.chat_engine:
                print(f"❌ Chat engine not available for connection {connection_id}")
                await self.send_message(
                    connection_id,
                    WebSocketMessage(type="error", data={"error": "Chat engine not initialized. Check API key configuration."})
                )
                return

            # Stream the response
            await self.stream_chat_response(connection_id, request, user_id)

        except Exception as e:
            logger.error("Failed to handle chat message", error=str(e))
            await self.send_message(
                connection_id,
                WebSocketMessage(type="error", data={"error": str(e)})
            )

    def _is_research_query(self, message: str) -> bool:
        """Simple keyword-based research detection"""
        research_keywords = [
            "research", "find out", "look up", "search", "analyze", "investigate",
            "what is", "tell me about", "explain", "how does", "market", "competitor",
            "pricing", "trends", "statistics", "data", "information"
        ]
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in research_keywords)
    
    async def stream_chat_response(
        self,
        connection_id: str,
        request: ChatRequest,
        user_id: Optional[str] = None
    ):
        """Stream chat response using the chat engine"""
        try:
            logger.info("Starting to stream chat response", connection_id=connection_id, request_message=request.message, user_id=user_id)
            # Stream response from chat engine
            full_response = ""
            event_count = 0
            has_received_events = False

            async for event in self.chat_engine.stream_message(request, user_id):
                has_received_events = True
                event_count += 1
                logger.info("Received chat engine event", connection_id=connection_id, event_type=event.get("type"), event_count=event_count)
                event_type = event.get("type", "")

                if event_type == "chat.token":
                    # Send token chunks
                    delta = event.get("delta", "")
                    full_response += delta
                    await self.send_message(
                        connection_id,
                        WebSocketMessage(
                            type="token",
                            data={"delta": delta, "conversation_id": request.conversation_id}
                        )
                    )
                    print(f"🔤 Sent token: '{delta}'")
                elif event_type == "chat.final":
                    # Send final message
                    message_data = event.get("message", {})
                    final_message = {
                        "conversation_id": request.conversation_id,
                        "content": full_response,
                        "role": message_data.get("role", "assistant"),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                    print(f"🏁 Sending final message: {final_message}")
                    await self.send_message(
                        connection_id,
                        WebSocketMessage(
                            type="message_complete",
                            data=final_message
                        )
                    )
                elif event_type in ["research.started", "research.done", "memory.written", "memory.used"]:
                    # Forward research/memory events
                    await self.send_message(
                        connection_id,
                        WebSocketMessage(type=event_type, data=event)
                    )
                elif event_type == "research.chunk":
                    # Forward research progress chunks
                    await self.send_message(
                        connection_id,
                        WebSocketMessage(type="research_progress", data={
                            "content": event.get("content", ""),
                            "conversation_id": request.conversation_id,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        })
                    )
                elif event_type == "research_progress":
                    # Enhanced research progress event
                    progress_data = event.get("data", {})
                    await self.send_message(
                        connection_id,
                        WebSocketMessage(type="research_progress", data={
                            "content": event.get("content", "Research in progress..."),
                            "sources_count": progress_data.get("sources_count", 0),
                            "conversation_id": request.conversation_id,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        })
                    )
                elif event_type == "error" and event.get("stage") == "research":
                    # Handle research-specific errors
                    await self.send_message(
                        connection_id,
                        WebSocketMessage(type="research_error", data={
                            "error": event.get("message", "Research failed"),
                            "error_type": event.get("error_type", "unknown"),
                            "conversation_id": request.conversation_id,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        })
                    )

            # If no events were received, send a fallback response
            if not has_received_events:
                logger.warning("No events received from chat engine, sending fallback response", connection_id=connection_id)
                fallback_message = "I received your message but couldn't process it properly. Please try again."
                await self.send_message(
                    connection_id,
                    WebSocketMessage(
                        type="message_complete",
                        data={
                            "conversation_id": request.conversation_id,
                            "content": fallback_message,
                            "role": "assistant",
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                    )
                )

        except Exception as e:
            logger.error("Failed to stream chat response", error=str(e))
            # Send error message to client
            await self.send_message(
                connection_id,
                WebSocketMessage(type="error", data={"error": str(e)})
            )


# Global connection manager
manager = ConnectionManager()


async def websocket_endpoint(
    websocket: WebSocket,
    connection_id: str,
    user_id: Optional[str] = None
):
    """Simplified WebSocket endpoint for conversational chat"""
    print(f"🔌 WebSocket endpoint called: connection_id={connection_id}, user_id={user_id}")
    await manager.connect(websocket, connection_id, user_id)
    print(f"✅ WebSocket connected: {connection_id}")

    try:
        while True:
            # Receive message
            print(f"⏳ Waiting for message on connection: {connection_id}")
            data = await websocket.receive_json()
            print(f"📨 Received message on {connection_id}: {data}")

            # Debug: Log received message
            logger.info("Received WebSocket message", connection_id=connection_id, data=data)

            # Handle different message types
            message_type = data.get("type", "chat")

            # Handle chat messages (explicit type or fallback for messages with message field)
            if message_type == "test":
                # Simple test message
                print(f"🧪 Received test message: {data}")
                await manager.send_message(
                    connection_id,
                    WebSocketMessage(type="test_response", data={"message": "WebSocket is working!", "timestamp": datetime.now(timezone.utc).isoformat()})
                )
            elif message_type == "chat" or ("message" in data and message_type != "ping"):
                # Handle chat message with keyword detection
                await manager.handle_chat_message(
                    connection_id,
                    data,
                    user_id or manager.connection_user.get(connection_id)
                )
            elif message_type == "ping":
                # Handle ping/pong for connection health
                await manager.send_message(
                    connection_id,
                    WebSocketMessage(type="pong", data={})
                )
            else:
                # Unknown message type
                await manager.send_message(
                    connection_id,
                    WebSocketMessage(
                        type="error",
                        data={"error": f"Unknown message type: {message_type}"}
                    )
                )

    except WebSocketDisconnect:
        manager.disconnect(connection_id)
    except Exception as e:
        logger.error("WebSocket error", connection_id=connection_id, error=str(e))
        manager.disconnect(connection_id)
