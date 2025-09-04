"""WebSocket handler with MCP integration for CRM/PM/HRMS data"""

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
from app.integrations.mcp_client import mongodb_mcp_client

logger = structlog.get_logger()


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class ConnectionManager:
    """Connection manager with MCP integration for WebSocket chat"""

    def __init__(self):
        print("🔧 Initializing ConnectionManager...")
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_user: Dict[str, str] = {}  # connection_id -> user_id
        # Chat engine has been removed
        self.chat_engine = None

        # Initialize MCP client
        self.mcp_client = mongodb_mcp_client

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

            # Check for database queries first
            is_database_query = self._is_database_query(message_text)

            if is_database_query:
                # Handle database query
                await self._handle_database_query(message_text, connection_id, user_id)
                return

            # Simple keyword detection for research vs general chat
            is_research = self._is_research_query(message_text)

            # Create chat request
            request = ChatRequest(
                message=message_text,
                conversation_id=conversation_id,
                user_id=user_id,  # Add user_id to request for profile loading
                use_web_search=is_research,
                stream=stream
            )

            # Send typing indicator
            await self.send_message(
                connection_id,
                WebSocketMessage(type="typing", data={"status": "thinking"})
            )

            # Chat engine has been removed - send simple response
            print(f"ℹ️ Chat engine has been removed, sending basic response")
            await self.send_message(
                connection_id,
                WebSocketMessage(
                    type="message_complete",
                    data={
                        "conversation_id": request.conversation_id,
                        "content": "Chat engine functionality has been removed. Database queries and MCP client operations are still available.",
                        "role": "assistant",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )
            )

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

    def _is_database_query(self, message: str) -> bool:
        """Detect if message should query CRM/PM/HRMS databases"""
        database_keywords = [
            # CRM
            "crm", "lead", "customer", "contact", "deal", "sales", "account", "opportunity",
            # PM
            "project", "task", "pm", "work item", "milestone", "sprint", "backlog",
            # Staff/HRMS
            "staff", "employee", "team", "personnel", "hr", "hrms", "leave", "vacation",
            "who", "what", "show me", "list", "find", "get", "query", "database"
        ]
        message_lower = message.lower()

        # Must have at least one database keyword
        has_db_keyword = any(keyword in message_lower for keyword in database_keywords)

        # Should not be a research query (avoid overlap)
        is_research = self._is_research_query(message)

        return has_db_keyword and not is_research

    async def _handle_database_query(self, message: str, connection_id: str, user_id: Optional[str] = None):
        """Handle CRM/PM/HRMS database queries"""
        message_lower = message.lower()

        try:
            # CRM queries
            if any(word in message_lower for word in ["crm", "lead", "customer", "contact", "deal", "sales"]):
                await self.send_message(
                    connection_id,
                    WebSocketMessage(type="data_query", data={"source": "CRM", "query": message})
                )

                # Get CRM leads
                async for result in self.mcp_client.find_crm_leads(
                    filter_query={"status": {"$in": ["open", "active", "qualified"]}},
                    limit=20,
                    user_id=user_id
                ):
                    if result.get("type") == "tool.output.data" and result.get("data"):
                        await self.send_message(
                            connection_id,
                            WebSocketMessage(type="data", data={
                                "source": "CRM",
                                "type": "lead",
                                **result["data"]
                            })
                        )

                # Get CRM stats
                async for result in self.mcp_client.aggregate_crm_stats(user_id=user_id):
                    if result.get("type") == "tool.output.data" and result.get("data"):
                        await self.send_message(
                            connection_id,
                            WebSocketMessage(type="data", data={
                                "source": "CRM",
                                "type": "stats",
                                **result["data"]
                            })
                        )

            # PM queries
            elif any(word in message_lower for word in ["project", "task", "pm", "work item", "milestone", "sprint"]):
                await self.send_message(
                    connection_id,
                    WebSocketMessage(type="data_query", data={"source": "PM", "query": message})
                )

                # Get PM tasks
                async for result in self.mcp_client.find_pm_tasks(
                    filter_query={"status": {"$ne": "done"}},
                    limit=30,
                    user_id=user_id
                ):
                    if result.get("type") == "tool.output.data" and result.get("data"):
                        await self.send_message(
                            connection_id,
                            WebSocketMessage(type="data", data={
                                "source": "PM",
                                "type": "task",
                                **result["data"]
                            })
                        )

                # Get PM stats
                async for result in self.mcp_client.aggregate_pm_stats(user_id=user_id):
                    if result.get("type") == "tool.output.data" and result.get("data"):
                        await self.send_message(
                            connection_id,
                            WebSocketMessage(type="data", data={
                                "source": "PM",
                                "type": "stats",
                                **result["data"]
                            })
                        )

            # Staff/HRMS queries
            elif any(word in message_lower for word in ["staff", "employee", "team", "personnel", "hr", "hrms", "leave", "vacation"]):
                await self.send_message(
                    connection_id,
                    WebSocketMessage(type="data_query", data={"source": "HRMS", "query": message})
                )

                # Get staff directory
                async for result in self.mcp_client.find_staff_directory(
                    filter_query={},
                    limit=50,
                    user_id=user_id
                ):
                    if result.get("type") == "tool.output.data" and result.get("data"):
                        await self.send_message(
                            connection_id,
                            WebSocketMessage(type="data", data={
                                "source": "HRMS",
                                "type": "staff",
                                **result["data"]
                            })
                        )

                # Get leave records if asking about availability
                if any(word in message_lower for word in ["leave", "vacation", "off", "away", "available"]):
                    async for result in self.mcp_client.find_hrms_leaves(
                        filter_query={},
                        limit=20,
                        user_id=user_id
                    ):
                        if result.get("type") == "tool.output.data" and result.get("data"):
                            await self.send_message(
                                connection_id,
                                WebSocketMessage(type="data", data={
                                    "source": "HRMS",
                                    "type": "leave",
                                    **result["data"]
                                })
                            )

        except Exception as e:
            logger.error("Failed to handle database query", error=str(e))
            await self.send_message(
                connection_id,
                WebSocketMessage(type="error", data={"error": f"Database query failed: {str(e)}"})
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
