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
from app.core.chat_engine import ChatEngine
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
        try:
            self.chat_engine = ChatEngine()
            print("✅ Chat engine initialized successfully")
        except Exception as e:
            error_message = str(e)
            print(f"❌ Failed to initialize chat engine: {error_message}")
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
                error_message = str(e)
                print(f"❌ Failed to send message to {connection_id}: {error_message}")
                logger.error("Failed to send message", connection_id=connection_id, error=error_message)
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
            error_message = str(e)
            logger.error("Failed to handle chat message", error=error_message)
            await self.send_message(
                connection_id,
                WebSocketMessage(type="error", data={"error": error_message})
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
            error_message = str(e)
            logger.error("Failed to handle database query", error=error_message)
            await self.send_message(
                connection_id,
                WebSocketMessage(type="error", data={"error": f"Database query failed: {error_message}"})
            )
    
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

            # Use agentic streaming for more intelligent responses
            async for event in self.chat_engine.stream_message_agentic(request, user_id):
                has_received_events = True
                event_count += 1

                # Handle both dict and string events
                if isinstance(event, dict):
                    event_type = event.get("type", "")
                elif isinstance(event, str):
                    # If event is a string, treat it as a token
                    event_type = "chat.token"
                else:
                    # Skip unknown event types
                    logger.warning("Unknown event type received", connection_id=connection_id, event_type=type(event).__name__)
                    continue

                if event_type == "chat.token":
                    # Send token chunks
                    if isinstance(event, dict):
                        delta = event.get("data", {}).get("delta", "")
                    else:  # string event
                        delta = event
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
                    if isinstance(event, dict):
                        message_data = event.get("data", {}).get("message", {})
                        role = message_data.get("role", "assistant")
                        content = message_data.get("content", full_response)  # Use event content, fallback to accumulated
                    else:
                        role = "assistant"
                        content = full_response
                    final_message = {
                        "conversation_id": request.conversation_id,
                        "content": content,
                        "role": role,
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
                    if isinstance(event, dict):
                        await self.send_message(
                            connection_id,
                            WebSocketMessage(type=event_type, data=event)
                        )
                elif event_type == "research.chunk":
                    # Forward research progress chunks
                    if isinstance(event, dict):
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
                    if isinstance(event, dict):
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
                elif event_type == "error" and isinstance(event, dict) and event.get("stage") == "research":
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
                elif event_type == "report_generation_started":
                    # Handle report generation start
                    if isinstance(event, dict):
                        await self.send_message(
                            connection_id,
                            WebSocketMessage(type="report_progress", data={
                                "content": event.get("content", "Generating report..."),
                                "conversation_id": request.conversation_id,
                                "timestamp": datetime.now(timezone.utc).isoformat()
                            })
                        )
                elif event_type == "report_progress":
                    # Handle report generation progress
                    if isinstance(event, dict):
                        await self.send_message(
                            connection_id,
                            WebSocketMessage(type="report_progress", data={
                                "content": event.get("content", "Report generation in progress..."),
                                "conversation_id": request.conversation_id,
                                "timestamp": datetime.now(timezone.utc).isoformat()
                            })
                        )
                elif event_type == "report_ready":
                    # Handle completed report
                    if isinstance(event, dict):
                        report_data = event.get("report", {})
                        await self.send_message(
                            connection_id,
                            WebSocketMessage(type="report_complete", data={
                                "content": event.get("content", "Report generated successfully!"),
                                "report": report_data,
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
            error_message = str(e)
            logger.error("Failed to stream chat response", error=error_message)
            # Send error message to client
            try:
                await self.send_message(
                    connection_id,
                    WebSocketMessage(type="error", data={"error": error_message})
                )
            except Exception as send_error:
                logger.error("Failed to send error message to client", error=str(send_error))
                # Don't try to send another error message to avoid infinite loop


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
        error_message = str(e)
        logger.error("WebSocket error", connection_id=connection_id, error=error_message)
        manager.disconnect(connection_id)
