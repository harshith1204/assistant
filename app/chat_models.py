"""Chat models for conversation management"""

from typing import List, Optional, Dict, Any, Literal
from datetime import datetime, timezone
from pydantic import BaseModel, Field
import uuid
from enum import Enum


class MessageRole(str, Enum):
    """Message roles in conversation"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    RESEARCH = "research"


class MessageType(str, Enum):
    """Types of messages"""
    TEXT = "text"
    RESEARCH_REQUEST = "research_request"
    RESEARCH_RESULT = "research_result"
    PLAN_REQUEST = "plan_request"
    PLAN_RESULT = "plan_result"
    ERROR = "error"
    STATUS = "status"


class ChatMessage(BaseModel):
    """Individual chat message"""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str = Field(..., description="Conversation ID")
    role: MessageRole = Field(..., description="Message role")
    content: str = Field(..., description="Message content")
    message_type: MessageType = Field(MessageType.TEXT, description="Type of message")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field({}, description="Additional metadata")
    attachments: List[Dict[str, str]] = Field([], description="Attachments")
    
    # Context information
    context_used: Optional[Dict[str, Any]] = Field(None, description="Context used for this message")
    memory_refs: List[str] = Field([], description="Memory references used")
    
    # Research-specific fields
    research_brief_id: Optional[str] = Field(None, description="Associated research brief ID")
    sources_count: Optional[int] = Field(None, description="Number of sources referenced")


class ConversationContext(BaseModel):
    """Context for a conversation"""
    short_term: Dict[str, Any] = Field({}, description="Short-term context (current session)")
    long_term: Dict[str, Any] = Field({}, description="Long-term context (persistent)")
    entities: List[str] = Field([], description="Identified entities")
    topics: List[str] = Field([], description="Identified topics")
    preferences: Dict[str, Any] = Field({}, description="User preferences")
    research_history: List[str] = Field([], description="Previous research brief IDs")


class Conversation(BaseModel):
    """Complete conversation"""
    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = Field(None, description="User ID")
    title: Optional[str] = Field(None, description="Conversation title")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    messages: List[ChatMessage] = Field([], description="Conversation messages")
    context: ConversationContext = Field(default_factory=ConversationContext)
    status: str = Field("active", description="Conversation status")
    metadata: Dict[str, Any] = Field({}, description="Additional metadata")
    
    def add_message(self, message: ChatMessage):
        """Add a message to the conversation"""
        message.conversation_id = self.conversation_id
        self.messages.append(message)
        self.updated_at = datetime.now(timezone.utc)
    
    def get_recent_messages(self, limit: int = 10) -> List[ChatMessage]:
        """Get recent messages"""
        return self.messages[-limit:] if self.messages else []
    
    def get_message_count(self) -> int:
        """Get total message count"""
        return len(self.messages)


class ChatRequest(BaseModel):
    """Request for chat interaction"""
    conversation_id: Optional[str] = Field(None, description="Existing conversation ID")
    message: str = Field(..., description="User message")
    context_window: int = Field(10, description="Number of previous messages to include")
    use_long_term_memory: bool = Field(True, description="Use long-term memory")
    use_web_search: bool = Field(True, description="Allow web search for research")
    stream: bool = Field(False, description="Stream response")


class ChatResponse(BaseModel):
    """Response from chat interaction"""
    conversation_id: str = Field(..., description="Conversation ID")
    message: ChatMessage = Field(..., description="Assistant response")
    suggestions: List[str] = Field([], description="Suggested follow-up questions")
    context_summary: Optional[Dict[str, Any]] = Field(None, description="Context summary")
    research_triggered: bool = Field(False, description="Whether research was triggered")


class ConversationSummary(BaseModel):
    """Summary of a conversation"""
    conversation_id: str
    title: Optional[str]
    created_at: datetime
    updated_at: datetime
    message_count: int
    last_message: Optional[str]
    topics: List[str]
    status: str


class MemoryUpdate(BaseModel):
    """Update to conversation memory"""
    conversation_id: str = Field(..., description="Conversation ID")
    memory_type: Literal["short_term", "long_term"] = Field(..., description="Memory type")
    key: str = Field(..., description="Memory key")
    value: Any = Field(..., description="Memory value")
    operation: Literal["add", "update", "delete"] = Field("add", description="Operation type")


class WebSocketMessage(BaseModel):
    """WebSocket message format"""
    type: str = Field(..., description="Message type")
    data: Dict[str, Any] = Field(..., description="Message data")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    
class StreamChunk(BaseModel):
    """Streaming response chunk"""
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str
    content: str
    is_final: bool = Field(False)
    metadata: Dict[str, Any] = Field({})
