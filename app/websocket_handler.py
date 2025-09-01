"""WebSocket handler for real-time chat"""

import json
import asyncio
from typing import Dict, Set, Optional
from datetime import datetime, timezone
from fastapi import WebSocket, WebSocketDisconnect
import structlog

from app.chat_models import (
    ChatRequest, ChatResponse, WebSocketMessage,
    StreamChunk, MessageRole, MessageType
)
from app.core.chat_engine import ChatEngine

logger = structlog.get_logger()


class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_connections: Dict[str, Set[str]] = {}  # user_id -> connection_ids
        self.connection_user: Dict[str, str] = {}  # connection_id -> user_id
        self.chat_engine = ChatEngine()
    
    async def connect(
        self,
        websocket: WebSocket,
        connection_id: str,
        user_id: Optional[str] = None
    ):
        """Accept and register a new connection with dual context"""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        
        # Ensure a stable user_id for this connection
        if not user_id:
            import uuid
            user_id = f"anon_{uuid.uuid4().hex[:12]}"
            logger.info("Generated anonymous user_id for websocket", user_id=user_id)

        # Track mappings
        self.connection_user[connection_id] = user_id
        if user_id not in self.user_connections:
            self.user_connections[user_id] = set()
        self.user_connections[user_id].add(connection_id)
        
        # Send user_id back to client for session persistence
        await self.send_message(
            connection_id,
            WebSocketMessage(
                type="session_info",
                data={"user_id": user_id, "connection_id": connection_id}
            )
        )
        
        logger.info(
            "WebSocket connected",
            connection_id=connection_id,
            user_id=user_id
        )
        
        # Send welcome message
        await self.send_message(
            connection_id,
            WebSocketMessage(
                type="connected",
                data={
                    "connection_id": connection_id,
                    "message": "Connected to chat service",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        )
    
    def disconnect(self, connection_id: str):
        """Remove a connection"""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
            
            # Remove from user connections
            user_id = self.connection_user.pop(connection_id, None)
            if user_id and user_id in self.user_connections:
                connections = self.user_connections[user_id]
                if connection_id in connections:
                    connections.remove(connection_id)
                if not connections:
                    del self.user_connections[user_id]
            
            logger.info("WebSocket disconnected", connection_id=connection_id)
    
    def _serialize_datetime(self, obj):
        """Recursively convert datetime objects to ISO format strings"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {key: self._serialize_datetime(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_datetime(item) for item in obj]
        else:
            return obj
    
    async def send_message(
        self,
        connection_id: str,
        message: WebSocketMessage
    ):
        """Send a message to a specific connection"""
        websocket = self.active_connections.get(connection_id)
        if websocket:
            try:
                # Convert to dict and ensure all datetime objects are serialized
                message_data = message.model_dump()
                # Recursively convert all datetime objects to ISO format strings
                message_data = self._serialize_datetime(message_data)
                await websocket.send_json(message_data)
            except Exception as e:
                logger.error(
                    "Failed to send message",
                    connection_id=connection_id,
                    error=str(e)
                )
                self.disconnect(connection_id)
    
    async def broadcast_to_user(
        self,
        user_id: str,
        message: WebSocketMessage
    ):
        """Broadcast a message to all connections of a user"""
        connections = self.user_connections.get(user_id, set())
        for connection_id in connections:
            await self.send_message(connection_id, message)
    
    async def handle_chat_message(
        self,
        connection_id: str,
        data: Dict,
        user_id: Optional[str] = None
    ):
        """Handle incoming chat message with dual context"""
        try:
            # Ensure user_id for dual context
            user_id = user_id or self.connection_user.get(connection_id)
            if not user_id:
                import uuid
                user_id = f"anon_{uuid.uuid4().hex[:12]}"
                self.connection_user[connection_id] = user_id
                if user_id not in self.user_connections:
                    self.user_connections[user_id] = set()
                self.user_connections[user_id].add(connection_id)
                logger.info("Generated anonymous user_id for chat", user_id=user_id)
                # Inform client of assigned user_id
                await self.send_message(
                    connection_id,
                    WebSocketMessage(
                        type="session_info",
                        data={"user_id": user_id, "connection_id": connection_id}
                    )
                )
            
            # Parse request
            request = ChatRequest(**data)
            
            # Send typing indicator
            await self.send_message(
                connection_id,
                WebSocketMessage(
                    type="typing",
                    data={"conversation_id": request.conversation_id}
                )
            )
            
            # Use conversational flow when requested
            if data.get("conversational", True):
                await self.handle_conversational_message(
                    connection_id,
                    request,
                    user_id
                )
            elif request.stream:
                # Stream response
                await self.stream_response(
                    connection_id,
                    request,
                    user_id
                )
            else:
                # Send complete response
                response = await self.chat_engine.process_message(
                    request,
                    user_id
                )
                
                await self.send_message(
                    connection_id,
                    WebSocketMessage(
                        type="message",
                        data=response.model_dump()
                    )
                )
                
        except Exception as e:
            logger.error(
                "Failed to handle chat message",
                connection_id=connection_id,
                error=str(e)
            )
            await self.send_message(
                connection_id,
                WebSocketMessage(
                    type="error",
                    data={"error": str(e)}
                )
            )
    
    async def stream_response(
        self,
        connection_id: str,
        request: ChatRequest,
        user_id: Optional[str] = None
    ):
        """Stream chat response"""
        try:
            conversation_id = request.conversation_id or str(asyncio.create_task(asyncio.sleep(0)).get_coro().cr_frame.f_locals.get('conversation_id', 'new'))
            
            # Start streaming
            await self.send_message(
                connection_id,
                WebSocketMessage(
                    type="stream_start",
                    data={"conversation_id": conversation_id}
                )
            )
            
            # Stream chunks
            full_response = ""
            async for chunk in self.chat_engine.stream_message(request, user_id):
                full_response += chunk
                
                stream_chunk = StreamChunk(
                    conversation_id=conversation_id,
                    content=chunk,
                    is_final=False
                )
                
                await self.send_message(
                    connection_id,
                    WebSocketMessage(
                        type="stream_chunk",
                        data=stream_chunk.model_dump()
                    )
                )
            
            # Send final chunk
            await self.send_message(
                connection_id,
                WebSocketMessage(
                    type="stream_end",
                    data={
                        "conversation_id": conversation_id,
                        "full_response": full_response
                    }
                )
            )
            
        except Exception as e:
            logger.error(
                "Failed to stream response",
                connection_id=connection_id,
                error=str(e)
            )
            await self.send_message(
                connection_id,
                WebSocketMessage(
                    type="stream_error",
                    data={"error": str(e)}
                )
            )
    
    async def handle_conversational_message(
        self,
        connection_id: str,
        request: ChatRequest,
        user_id: Optional[str] = None
    ):
        """Handle conversational message with enhanced flow"""
        try:
            # Process through merged chat engine conversational flow
            async for update in self.chat_engine.process_conversational_message(
                request,
                user_id
            ):
                # Send each update to the client
                await self.send_message(
                    connection_id,
                    WebSocketMessage(
                        type="conversational_update",
                        data=update
                    )
                )
                
                # Handle special update types
                if update.get("type") == "clarification_needed":
                    # Store that we're waiting for clarification
                    self.active_connections[connection_id].waiting_for = "clarification"
                    
                elif update.get("type") == "confirmation_needed":
                    # Store that we're waiting for confirmation
                    self.active_connections[connection_id].waiting_for = "confirmation"
                    
        except Exception as e:
            logger.error(
                "Failed to handle conversational message",
                connection_id=connection_id,
                error=str(e)
            )
            await self.send_message(
                connection_id,
                WebSocketMessage(
                    type="conversational_error",
                    data={"error": str(e)}
                )
            )
    
    async def handle_research_request(
        self,
        connection_id: str,
        data: Dict,
        user_id: Optional[str] = None
    ):
        """Handle research request"""
        try:
            # Send status update
            await self.send_message(
                connection_id,
                WebSocketMessage(
                    type="research_status",
                    data={
                        "status": "started",
                        "message": "Research initiated..."
                    }
                )
            )
            
            # Create chat request with research flag
            request = ChatRequest(
                message=data.get("query", ""),
                use_web_search=True,
                conversation_id=data.get("conversation_id")
            )
            
            # Process with research
            response = await self.chat_engine.process_message(request, user_id)
            
            # Send research results
            await self.send_message(
                connection_id,
                WebSocketMessage(
                    type="research_complete",
                    data=response.model_dump()
                )
            )
            
        except Exception as e:
            logger.error(
                "Failed to handle research request",
                connection_id=connection_id,
                error=str(e)
            )
            await self.send_message(
                connection_id,
                WebSocketMessage(
                    type="research_error",
                    data={"error": str(e)}
                )
            )
    
    async def handle_conversation_list(
        self,
        connection_id: str,
        user_id: Optional[str] = None
    ):
        """Send list of conversations"""
        try:
            conversations = await self.chat_engine.list_conversations(user_id)
            
            conversation_list = [
                {
                    "conversation_id": conv.conversation_id,
                    "title": conv.title or f"Conversation {conv.conversation_id[:8]}",
                    "created_at": conv.created_at.isoformat(),
                    "updated_at": conv.updated_at.isoformat(),
                    "message_count": conv.get_message_count(),
                    "status": conv.status
                }
                for conv in conversations
            ]
            
            await self.send_message(
                connection_id,
                WebSocketMessage(
                    type="conversation_list",
                    data={"conversations": conversation_list}
                )
            )
            
        except Exception as e:
            logger.error(
                "Failed to get conversation list",
                connection_id=connection_id,
                error=str(e)
            )
    
    async def handle_conversation_history(
        self,
        connection_id: str,
        data: Dict,
        user_id: Optional[str] = None
    ):
        """Send conversation history"""
        try:
            conversation_id = data.get("conversation_id")
            if not conversation_id:
                raise ValueError("conversation_id is required")
            
            conversation = await self.chat_engine.get_conversation(conversation_id)
            
            if conversation:
                messages = [
                    {
                        "message_id": msg.message_id,
                        "role": msg.role.value,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat(),
                        "type": msg.message_type.value
                    }
                    for msg in conversation.messages
                ]
                
                await self.send_message(
                    connection_id,
                    WebSocketMessage(
                        type="conversation_history",
                        data={
                            "conversation_id": conversation_id,
                            "messages": messages,
                            "context": conversation.context.model_dump()
                        }
                    )
                )
            else:
                await self.send_message(
                    connection_id,
                    WebSocketMessage(
                        type="error",
                        data={"error": "Conversation not found"}
                    )
                )
                
        except Exception as e:
            logger.error(
                "Failed to get conversation history",
                connection_id=connection_id,
                error=str(e)
            )


# Global connection manager
manager = ConnectionManager()


async def websocket_endpoint(
    websocket: WebSocket,
    connection_id: str,
    user_id: Optional[str] = None
):
    """WebSocket endpoint handler"""
    await manager.connect(websocket, connection_id, user_id)
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            # Route based on message type
            message_type = data.get("type")
            
            if message_type == "chat":
                await manager.handle_chat_message(
                    connection_id,
                    data.get("data", {}),
                    user_id
                )
            elif message_type == "research":
                await manager.handle_research_request(
                    connection_id,
                    data.get("data", {}),
                    user_id
                )
            elif message_type == "list_conversations":
                await manager.handle_conversation_list(
                    connection_id,
                    user_id
                )
            elif message_type == "get_history":
                await manager.handle_conversation_history(
                    connection_id,
                    data.get("data", {}),
                    user_id
                )
            elif message_type == "ping":
                await manager.send_message(
                    connection_id,
                    WebSocketMessage(type="pong", data={})
                )
            else:
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
        logger.error(
            "WebSocket error",
            connection_id=connection_id,
            error=str(e)
        )
        manager.disconnect(connection_id)
