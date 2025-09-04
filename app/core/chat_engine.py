"""Clean, agentic chat engine with integrated AI assistant capabilities"""

import json
import asyncio
import uuid
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime, timezone
from groq import AsyncGroq
import structlog

from app.config import settings
from app.chat_models import (
    ChatMessage, ChatRequest, ChatResponse, Conversation,
    MessageRole, MessageType
)
from app.core.memory_manager import MemoryManager
from app.core.agent import AgenticAssistant
from app.integrations.mcp_client import get_mongodb_mcp_client

logger = structlog.get_logger()


class ChatEngine:
    """Clean, agentic chat engine with integrated AI assistant capabilities"""
    
    def __init__(self):
        """Initialize chat engine with agentic assistant"""
        print("🤖 Initializing Agentic ChatEngine")
        self.memory_manager = MemoryManager()
        self.agentic_assistant = AgenticAssistant()
        self.mcp_client = get_mongodb_mcp_client()
        self.conversations: Dict[str, Conversation] = {}
        self._cancel_signals: Dict[str, asyncio.Event] = {}

        # Initialize MCP client connection
        try:
            loop = asyncio.get_running_loop()
            asyncio.create_task(self._init_mcp_client())
        except RuntimeError:
            logger.info("No running event loop, MCP client will be initialized on first use")
            self._mcp_initialized = False

        print("✅ Agentic ChatEngine initialized successfully")

    # ============================================================================
    # CORE AGENTIC METHODS
    # ============================================================================

    async def process_message(
        self,
        request: ChatRequest,
        user_id: Optional[str] = None
    ) -> ChatResponse:
        """Process a chat message using the agentic assistant"""
        return await self.process_message_agentic(request, user_id)

    async def process_message_agentic(
        self,
        request: ChatRequest,
        user_id: Optional[str] = None
    ) -> ChatResponse:
        """Process a chat message using the new agentic system"""
        try:
            # Ensure dual context
            conversation_id = request.conversation_id or str(uuid.uuid4())
            conversation_id, user_id = self.memory_manager.ensure_dual_context(
                conversation_id,
                request.user_id or user_id
            )
            
            # Get or create conversation
            conversation = await self._get_or_create_conversation(conversation_id, user_id)
            
            # Create user message
            user_message = ChatMessage(
                conversation_id=conversation.conversation_id,
                role=MessageRole.USER,
                content=request.message,
                message_type=MessageType.TEXT
            )
            
            # Add to conversation and memory
            conversation.add_message(user_message)
            await self.memory_manager.update_from_message(user_message, user_id)
            
            # Get conversation history for agent context
            conversation_history = [
                {
                    "user_message": msg.content if msg.role == MessageRole.USER else "",
                    "assistant_response": msg.content if msg.role == MessageRole.ASSISTANT else "",
                    "timestamp": msg.timestamp.isoformat() if msg.timestamp else None
                }
                for msg in conversation.messages[-10:]
            ]

            # Get user profile
            user_profile = await self.memory_manager.get_profile(user_id)

            # Process with agentic assistant
            agentic_response = ""
            async for chunk in self.agentic_assistant.process_request(
                user_message=request.message,
                user_id=user_id,
                conversation_id=conversation_id
            ):
                if isinstance(chunk, str):
                    agentic_response += chunk

            # Create assistant message
            assistant_message = ChatMessage(
                                conversation_id=conversation.conversation_id,
                                role=MessageRole.ASSISTANT,
                content=agentic_response,
                                message_type=MessageType.TEXT
            )

            # Add to conversation and memory
            conversation.add_message(assistant_message)
            await self.memory_manager.update_from_message(assistant_message, user_id)
            
            # Generate suggestions
            suggestions = await self._generate_suggestions(conversation, agentic_response)

            # Store conversation and rolling summary
            self.conversations[conversation.conversation_id] = conversation
            await self._maybe_create_rolling_summary(conversation)
            
            return ChatResponse(
                conversation_id=conversation.conversation_id,
                message=assistant_message,
                suggestions=suggestions,
                metadata={"agentic_processing": True, "processing_mode": "agentic_assistant"}
            )
            
        except Exception as e:
            logger.error("Failed to process message agentically", error=str(e))
            raise
    
    async def stream_message(
        self,
        request: ChatRequest,
        user_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream a chat message using the agentic system"""
        async for event in self.stream_message_agentic(request, user_id):
            yield event

    async def stream_message_agentic(
        self,
        request: ChatRequest,
        user_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream a chat message using the agentic system with progress updates"""
        try:
            # Ensure dual context
            conversation_id = request.conversation_id or str(uuid.uuid4())
            conversation_id, user_id = self.memory_manager.ensure_dual_context(
                conversation_id,
                request.user_id or user_id
            )

            # Get or create conversation
            conversation = await self._get_or_create_conversation(conversation_id, user_id)

            # Create user message
            user_message = ChatMessage(
                conversation_id=conversation.conversation_id,
                role=MessageRole.USER,
                content=request.message,
                message_type=MessageType.TEXT
            )

            # Add to conversation and memory
            conversation.add_message(user_message)
            await self.memory_manager.update_from_message(user_message, user_id)

            # Get conversation history for agent context
            conversation_history = [
                {
                    "user_message": msg.content if msg.role == MessageRole.USER else "",
                    "assistant_response": msg.content if msg.role == MessageRole.ASSISTANT else "",
                    "timestamp": msg.timestamp.isoformat() if msg.timestamp else None
                }
                for msg in conversation.messages[-10:]
            ]

            # Get user profile
            user_profile = await self.memory_manager.get_profile(user_id)

            # Progress tracking
            full_response = ""

            # Stream agentic processing with progress updates
            def streaming_callback(message: str):
                """Callback for streaming agentic responses with status updates"""
                nonlocal full_response
                if message:
                    full_response += message

            # Process with agentic assistant
            async for event in self.agentic_assistant.process_request(
                user_message=request.message,
                user_id=user_id,
                conversation_id=conversation_id,
                streaming_callback=streaming_callback
            ):

                if event:
                    # Format the event properly for the websocket handler
                    if isinstance(event, str):
                        yield {
                            "type": "chat.token",
                            "data": {"delta": event, "conversation_id": conversation_id}
                        }
                    else:
                        # If it's already a dict, yield as-is
                        yield event

            # Create assistant message
            assistant_message = ChatMessage(
                conversation_id=conversation.conversation_id,
                role=MessageRole.ASSISTANT,
                content=full_response,
                message_type=MessageType.TEXT
            )
            
            # Add to conversation and memory
            conversation.add_message(assistant_message)
            await self.memory_manager.update_from_message(assistant_message, user_id)

            # Generate suggestions
            suggestions = await self._generate_suggestions(conversation, full_response)
            
            # Store conversation and rolling summary
            self.conversations[conversation.conversation_id] = conversation
            await self._maybe_create_rolling_summary(conversation)

            # Send final message
            yield {
                "type": "chat.final",
                "data": {
                    "message": {
                "conversation_id": conversation.conversation_id,
                        "content": full_response,
                        "role": "assistant",
                "suggestions": suggestions,
                "metadata": {
                            "agentic_processing": True,
                            "processing_mode": "agentic_assistant"
                        }
                    }
                }
                }

        except Exception as e:
            error_message = str(e)
            logger.error("Failed to stream message agentically", error=error_message)
            yield {
                "type": "chat.error",
                "data": {
                    "error": error_message,
                    "conversation_id": request.conversation_id
                }
            }

    # ============================================================================
    # UTILITY METHODS
    # ============================================================================

    async def _init_mcp_client(self):
        """Initialize MCP client connection"""
        try:
            logger.info("Initializing MCP client connection...")
            connected = await self.mcp_client.connect()
            if connected:
                logger.info("✅ MCP client connected successfully")
                self._mcp_status = "connected"
            else:
                logger.warning("❌ Failed to connect MCP client")
                self._mcp_status = "disconnected"
        except Exception as e:
            logger.error("Failed to initialize MCP client", error=str(e))
            self._mcp_status = "error"

    async def _ensure_mcp_initialized(self):
        """Ensure MCP client is initialized"""
        if getattr(self, '_mcp_initialized', True):
            return  # Already initialized or skipped

        try:
            await self._init_mcp_client()
            self._mcp_initialized = True
        except Exception as e:
            logger.error("Failed to initialize MCP client on demand", error=str(e))
            self._mcp_initialized = True  # Don't try again

    async def check_mcp_health(self) -> Dict[str, Any]:
        """Check MCP client health status"""
        try:
            await self._ensure_mcp_initialized()
            health = {
                "status": getattr(self, '_mcp_status', 'unknown'),
                "connected": getattr(self.mcp_client, 'connected', False),
                "tools_count": 0
            }
            return health
        except Exception as e:
            logger.error("MCP health check failed", error=str(e))
            return {"status": "error", "error": str(e), "connected": False}

    async def _get_or_create_conversation(self, conversation_id: str, user_id: str) -> Conversation:
        """Get existing or create new conversation"""
        if conversation_id in self.conversations:
            return self.conversations[conversation_id]

        conversation = Conversation(conversation_id=conversation_id, user_id=user_id)

        # Load context if existing conversation
        context = await self.memory_manager.get_conversation_context(conversation_id, user_id)
        conversation.context = context

        return conversation

    async def _generate_suggestions(self, conversation: Conversation, response: str) -> List[str]:
        """Generate follow-up suggestions based on conversation"""
        try:
            # Simple suggestion generation based on response content
            suggestions = []

            if "research" in response.lower():
                suggestions.append("Would you like me to research this topic further?")
            if "data" in response.lower() or "database" in response.lower():
                suggestions.append("Should I query the database for more information?")
            if "analyze" in response.lower():
                suggestions.append("Would you like me to perform additional analysis?")
            if "calculate" in response.lower():
                suggestions.append("Need me to perform any calculations?")

            # Default suggestions if none generated
            if not suggestions:
                suggestions = [
                    "What would you like to explore next?",
                    "Can I help you with anything else?",
                    "Would you like me to clarify anything?"
                ]

            return suggestions[:3]  # Limit to 3 suggestions
            
        except Exception as e:
            logger.error("Failed to generate suggestions", error=str(e))
            return ["How can I help you further?"]

    async def _maybe_create_rolling_summary(self, conversation: Conversation):
        """Create rolling summary if needed"""
        try:
            # Simple rolling summary - could be enhanced
            if len(conversation.messages) % 10 == 0:  # Every 10 messages
                await self.memory_manager.create_conversation_summary(
                    conversation.conversation_id,
                    conversation.user_id
                )
        except Exception as e:
            logger.error("Failed to create rolling summary", error=str(e))

    # ============================================================================
    # LEGACY COMPATIBILITY METHODS (redirect to agentic methods)
    # ============================================================================

    async def process_conversational_message(self, request: ChatRequest, user_id: Optional[str] = None):
        """Legacy method - redirects to agentic processing"""
        return await self.process_message_agentic(request, user_id)
