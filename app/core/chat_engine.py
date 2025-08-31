"""Chat engine with research integration and context management"""

import json
import asyncio
import uuid
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime
from groq import AsyncGroq
import structlog

from app.config import settings
from app.chat_models import (
    ChatMessage, ChatRequest, ChatResponse, Conversation,
    MessageRole, MessageType, ConversationContext
)
from app.models import ResearchRequest, ResearchBrief
from app.core.memory_manager import MemoryManager
from app.core.research_engine import ResearchEngine

logger = structlog.get_logger()


class ChatEngine:
    """Main chat engine with research capabilities"""
    
    def __init__(self):
        """Initialize chat engine"""
        self.groq_client = AsyncGroq(api_key=settings.groq_api_key)
        self.memory_manager = MemoryManager()
        self.research_engine = ResearchEngine()
        self.conversations: Dict[str, Conversation] = {}
        
    async def process_message(
        self,
        request: ChatRequest,
        user_id: Optional[str] = None
    ) -> ChatResponse:
        """Process a chat message with dual context"""
        try:
            # Ensure we have both IDs for dual context
            conversation_id = request.conversation_id or str(uuid.uuid4())
            conversation_id, user_id = self.memory_manager.ensure_dual_context(
                conversation_id,
                user_id
            )
            
            # Get or create conversation with both IDs
            conversation = await self._get_or_create_conversation(
                conversation_id,
                user_id
            )
            
            # Create user message
            user_message = ChatMessage(
                conversation_id=conversation.conversation_id,
                role=MessageRole.USER,
                content=request.message,
                message_type=MessageType.TEXT
            )
            
            # Add to conversation
            conversation.add_message(user_message)
            
            # Update memory
            await self.memory_manager.update_from_message(user_message, user_id)
            
            # Get context
            context = await self._prepare_context(
                conversation,
                request,
                user_id
            )
            
            # Determine if research is needed
            needs_research = await self._needs_research(request.message, context)
            
            if needs_research and request.use_web_search:
                # Trigger research
                response = await self._handle_research_request(
                    conversation,
                    request.message,
                    context,
                    user_id
                )
            else:
                # Regular chat response
                response = await self._generate_chat_response(
                    conversation,
                    request.message,
                    context,
                    user_id
                )
            
            # Generate suggestions
            suggestions = await self._generate_suggestions(
                conversation,
                response.message.content
            )
            response.suggestions = suggestions
            
            # Store conversation
            self.conversations[conversation.conversation_id] = conversation
            
            return response
            
        except Exception as e:
            logger.error("Failed to process message", error=str(e))
            raise
    
    async def stream_message(
        self,
        request: ChatRequest,
        user_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Stream chat response"""
        try:
            # Get or create conversation
            conversation = await self._get_or_create_conversation(
                request.conversation_id,
                user_id
            )
            
            # Create user message
            user_message = ChatMessage(
                conversation_id=conversation.conversation_id,
                role=MessageRole.USER,
                content=request.message,
                message_type=MessageType.TEXT
            )
            
            # Add to conversation
            conversation.add_message(user_message)
            
            # Update memory
            await self.memory_manager.update_from_message(user_message, user_id)
            
            # Get context
            context = await self._prepare_context(
                conversation,
                request,
                user_id
            )
            
            # Prepare messages for LLM
            messages = await self._prepare_llm_messages(
                conversation,
                request.message,
                context
            )
            
            # Stream from LLM
            stream = await self.groq_client.chat.completions.create(
                model=settings.groq_model,
                messages=messages,
                temperature=settings.chat_temperature,
                max_tokens=settings.chat_max_tokens,
                stream=True
            )
            
            full_response = ""
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content
            
            # Create assistant message
            assistant_message = ChatMessage(
                conversation_id=conversation.conversation_id,
                role=MessageRole.ASSISTANT,
                content=full_response,
                message_type=MessageType.TEXT,
                context_used={"context_window": request.context_window}
            )
            
            # Add to conversation and memory
            conversation.add_message(assistant_message)
            await self.memory_manager.update_from_message(assistant_message, user_id)
            
            # Store conversation
            self.conversations[conversation.conversation_id] = conversation
            
        except Exception as e:
            logger.error("Failed to stream message", error=str(e))
            yield f"Error: {str(e)}"
    
    async def _get_or_create_conversation(
        self,
        conversation_id: Optional[str],
        user_id: Optional[str]
    ) -> Conversation:
        """Get existing or create new conversation"""
        if conversation_id and conversation_id in self.conversations:
            return self.conversations[conversation_id]
        
        # Create new conversation
        if conversation_id:
            conversation = Conversation(
                conversation_id=conversation_id,
                user_id=user_id
            )
        else:
            # Let the default factory generate a UUID
            conversation = Conversation(
                user_id=user_id
            )
        
        # Load context if existing conversation
        if conversation_id:
            context = await self.memory_manager.get_conversation_context(
                conversation_id,
                user_id
            )
            conversation.context = context
        
        return conversation
    
    async def _prepare_context(
        self,
        conversation: Conversation,
        request: ChatRequest,
        user_id: Optional[str]
    ) -> Dict[str, Any]:
        """Prepare context for message processing"""
        context = {}
        
        # Get conversation context
        conv_context = await self.memory_manager.get_conversation_context(
            conversation.conversation_id,
            user_id
        )
        
        # Add recent messages
        recent_messages = conversation.get_recent_messages(request.context_window)
        context["recent_messages"] = [
            {
                "role": msg.role.value,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat()
            }
            for msg in recent_messages
        ]
        
        # Search both user and conversation memories if enabled
        if request.use_long_term_memory:
            # Search both user-level and conversation-level memories
            memories = await self.memory_manager.search_memory(
                request.message,
                conversation.conversation_id,
                user_id,
                limit=5,
                search_scope="both"  # Search both contexts
            )
            context["relevant_memories"] = memories
            
            # Separate memories by level for better context
            context["user_memories"] = [m for m in memories if m.get("memory_level") == "user"]
            context["conversation_memories"] = [m for m in memories if m.get("memory_level") == "conversation"]
        
        # Add conversation metadata
        context["conversation_metadata"] = {
            "message_count": conversation.get_message_count(),
            "created_at": conversation.created_at.isoformat(),
            "entities": conv_context.entities,
            "topics": conv_context.topics
        }
        
        return context
    
    async def _needs_research(
        self,
        message: str,
        context: Dict[str, Any]
    ) -> bool:
        """Determine if the message needs research"""
        research_indicators = [
            "research", "find out", "look up", "search for",
            "what is", "how does", "analyze", "investigate",
            "tell me about", "explain", "market analysis",
            "competitor", "pricing", "trends", "statistics"
        ]
        
        message_lower = message.lower()
        
        # Check for research indicators
        if any(indicator in message_lower for indicator in research_indicators):
            return True
        
        # Check if it's a question that might need current information
        if "?" in message and any(
            word in message_lower 
            for word in ["latest", "current", "recent", "today", "now"]
        ):
            return True
        
        return False
    
    async def _handle_research_request(
        self,
        conversation: Conversation,
        message: str,
        context: Dict[str, Any],
        user_id: Optional[str]
    ) -> ChatResponse:
        """Handle a research request"""
        try:
            # Create status message
            status_message = ChatMessage(
                conversation_id=conversation.conversation_id,
                role=MessageRole.SYSTEM,
                content="🔍 Initiating research... This may take a moment.",
                message_type=MessageType.STATUS
            )
            conversation.add_message(status_message)
            
            # Create research request
            research_request = ResearchRequest(
                query=message,
                max_sources=15,
                deep_dive=True
            )
            
            # Run research
            logger.info("Running research from chat", query=message)
            research_brief = await self.research_engine.run_research(research_request)
            
            # Format research results
            formatted_response = await self._format_research_results(research_brief)
            
            # Create research message
            research_message = ChatMessage(
                conversation_id=conversation.conversation_id,
                role=MessageRole.ASSISTANT,
                content=formatted_response,
                message_type=MessageType.RESEARCH_RESULT,
                research_brief_id=research_brief.brief_id,
                sources_count=research_brief.total_sources,
                metadata={"research_brief": research_brief.brief_id}
            )
            
            # Add to conversation and memory
            conversation.add_message(research_message)
            await self.memory_manager.update_from_message(research_message, user_id)
            
            # Update conversation context
            conversation.context.research_history.append(research_brief.brief_id)
            
            return ChatResponse(
                conversation_id=conversation.conversation_id,
                message=research_message,
                research_triggered=True,
                context_summary={
                    "sources_analyzed": research_brief.total_sources,
                    "findings": len(research_brief.findings),
                    "ideas_generated": len(research_brief.ideas)
                }
            )
            
        except Exception as e:
            logger.error("Research request failed", error=str(e))
            error_message = ChatMessage(
                conversation_id=conversation.conversation_id,
                role=MessageRole.ASSISTANT,
                content=f"I encountered an error while researching: {str(e)}. Let me try to help you with the information I have.",
                message_type=MessageType.ERROR
            )
            conversation.add_message(error_message)
            
            # Fall back to regular response
            return await self._generate_chat_response(
                conversation,
                message,
                context,
                user_id
            )
    
    async def _generate_chat_response(
        self,
        conversation: Conversation,
        message: str,
        context: Dict[str, Any],
        user_id: Optional[str]
    ) -> ChatResponse:
        """Generate a regular chat response"""
        try:
            # Prepare messages for LLM
            messages = await self._prepare_llm_messages(
                conversation,
                message,
                context
            )
            
            # Generate response
            response = await self.groq_client.chat.completions.create(
                model=settings.groq_model,
                messages=messages,
                temperature=settings.chat_temperature,
                max_tokens=settings.chat_max_tokens
            )
            
            # Create assistant message
            assistant_message = ChatMessage(
                conversation_id=conversation.conversation_id,
                role=MessageRole.ASSISTANT,
                content=response.choices[0].message.content,
                message_type=MessageType.TEXT,
                context_used=context
            )
            
            # Add to conversation and memory
            conversation.add_message(assistant_message)
            await self.memory_manager.update_from_message(assistant_message, user_id)
            
            return ChatResponse(
                conversation_id=conversation.conversation_id,
                message=assistant_message,
                research_triggered=False,
                context_summary={
                    "memories_used": len(context.get("relevant_memories", [])),
                    "context_window": len(context.get("recent_messages", []))
                }
            )
            
        except Exception as e:
            logger.error("Failed to generate response", error=str(e))
            raise
    
    async def _prepare_llm_messages(
        self,
        conversation: Conversation,
        current_message: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Prepare messages for LLM"""
        messages = []
        
        # System prompt
        system_prompt = """You are an intelligent assistant with research capabilities and long-term memory.
You help users with various tasks including research, analysis, planning, and general conversation.
You have access to conversation history and can remember important information across sessions.

When responding:
1. Be helpful, accurate, and concise
2. Use the context and memories provided to give personalized responses
3. If you reference previous conversations or memories, mention it naturally
4. For research requests, provide comprehensive, well-structured information
5. Suggest follow-up questions when appropriate
"""
        
        # Add context to system prompt
        if context.get("relevant_memories"):
            system_prompt += "\n\nRelevant memories from previous conversations:\n"
            for memory in context["relevant_memories"][:3]:
                system_prompt += f"- {memory.get('memory', '')}\n"
        
        if context.get("conversation_metadata", {}).get("topics"):
            topics = context["conversation_metadata"]["topics"]
            system_prompt += f"\n\nTopics discussed: {', '.join(topics)}"
        
        messages.append({"role": "system", "content": system_prompt})
        
        # Add recent messages
        for msg in context.get("recent_messages", [])[-5:]:
            if msg["role"] != "system":
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Add current message
        messages.append({"role": "user", "content": current_message})
        
        return messages
    
    async def _format_research_results(self, brief: ResearchBrief) -> str:
        """Format research results for chat"""
        formatted = f"## Research Results: {brief.query}\n\n"
        
        if brief.executive_summary:
            formatted += f"### Executive Summary\n{brief.executive_summary}\n\n"
        
        formatted += f"### Key Findings ({len(brief.findings)} total)\n"
        for i, finding in enumerate(brief.findings[:5], 1):
            formatted += f"\n**{i}. {finding.title}**\n"
            formatted += f"{finding.summary}\n"
            if finding.key_insights:
                formatted += "- " + "\n- ".join(finding.key_insights[:3]) + "\n"
        
        if brief.ideas:
            formatted += f"\n### Actionable Ideas (Top {min(5, len(brief.ideas))})\n"
            sorted_ideas = sorted(
                brief.ideas,
                key=lambda x: x.rice.score if x.rice.score else 0,
                reverse=True
            )
            for i, idea in enumerate(sorted_ideas[:5], 1):
                formatted += f"\n**{i}. {idea.idea}**\n"
                formatted += f"   *Rationale:* {idea.rationale}\n"
                if idea.rice.score:
                    formatted += f"   *RICE Score:* {idea.rice.score:.1f}\n"
        
        formatted += f"\n📊 *Analyzed {brief.total_sources} sources with {brief.average_confidence:.0%} average confidence*"
        formatted += f"\n🔗 *Research Brief ID: {brief.brief_id}*"
        
        return formatted
    
    async def _generate_suggestions(
        self,
        conversation: Conversation,
        last_response: str
    ) -> List[str]:
        """Generate follow-up suggestions"""
        try:
            prompt = f"""Based on this conversation response, suggest 3 relevant follow-up questions the user might ask:

Response: {last_response[:500]}

Provide 3 short, specific questions that would naturally continue the conversation."""
            
            response = await self.groq_client.chat.completions.create(
                model=settings.groq_model,
                messages=[
                    {"role": "system", "content": "Generate follow-up questions"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=settings.summary_max_tokens
            )
            
            # Parse suggestions
            suggestions_text = response.choices[0].message.content
            suggestions = [
                s.strip().strip("-•").strip()
                for s in suggestions_text.split("\n")
                if s.strip() and not s.strip().startswith("#")
            ][:3]
            
            return suggestions
            
        except Exception as e:
            logger.error("Failed to generate suggestions", error=str(e))
            return [
                "Can you tell me more about this?",
                "What are the next steps?",
                "Do you have any examples?"
            ]
    
    async def get_conversation(
        self,
        conversation_id: str
    ) -> Optional[Conversation]:
        """Get a conversation by ID"""
        return self.conversations.get(conversation_id)
    
    async def list_conversations(
        self,
        user_id: Optional[str] = None
    ) -> List[Conversation]:
        """List all conversations for a user"""
        if user_id:
            return [
                conv for conv in self.conversations.values()
                if conv.user_id == user_id
            ]
        return list(self.conversations.values())
    
    async def delete_conversation(
        self,
        conversation_id: str
    ) -> bool:
        """Delete a conversation"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            await self.memory_manager.clear_short_term_memory(conversation_id)
            return True
        return False
