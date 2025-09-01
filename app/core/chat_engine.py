"""Chat engine with research integration and context management"""

import json
import asyncio
import uuid
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime
import math
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
from app.core.intent import (
    ConversationalFlowManager
)
from app.integrations.crm_client import CRMClient
from app.integrations.pms_client import PMSClient

logger = structlog.get_logger()


class ChatEngine:
    """Main chat engine with research capabilities"""
    
    def __init__(self):
        """Initialize chat engine"""
        self.groq_client = AsyncGroq(api_key=settings.groq_api_key)
        self.memory_manager = MemoryManager()
        self.research_engine = ResearchEngine()
        # Enhanced conversational capabilities
        self.flow_manager = ConversationalFlowManager()
        self.crm_client = CRMClient()
        self.pms_client = PMSClient()
        self.conversations: Dict[str, Conversation] = {}
        self._turn_counters: Dict[str, int] = {}
        self._cancel_signals: Dict[str, asyncio.Event] = {}
        
        # Action handlers mapping (for conversational routing)
        self.action_handlers = {
            "research_engine": self._handle_research_action,
            "crm_client": self._handle_crm_action,
            "pms_client": self._handle_pms_action,
            "report_generator": self._handle_report_action,
            "calendar_manager": self._handle_calendar_action,
            "chat_engine": self._handle_chat_action
        }
        
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
                request.user_id or user_id
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
            
            # Rolling summary after completed assistant turn
            await self._maybe_create_rolling_summary(conversation)
            
            return response
            
        except Exception as e:
            logger.error("Failed to process message", error=str(e))
            raise
    
    async def stream_message(
        self,
        request: ChatRequest,
        user_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream typed events over a single WebSocket loop."""
        try:
            # Ensure dual context
            conversation_id = request.conversation_id or str(uuid.uuid4())
            conversation_id, user_id = self.memory_manager.ensure_dual_context(
                conversation_id,
                request.user_id or user_id
            )

            if conversation_id not in self._cancel_signals:
                self._cancel_signals[conversation_id] = asyncio.Event()
            cancel_signal = self._cancel_signals[conversation_id]

            # Get or create conversation
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

            # Add to conversation and memory
            conversation.add_message(user_message)
            await self.memory_manager.update_from_message(user_message, user_id)

            # Prepare context
            context = await self._prepare_context(
                conversation,
                request,
                user_id
            )

            # Emit memory.used
            def _extract_line(m: Dict[str, Any]) -> str:
                if isinstance(m, dict):
                    return (m.get("memory") or m.get("content") or str(m)).strip()
                return str(m).strip()

            memory_used: List[Dict[str, Any]] = []
            for m in context.get("profile", [])[:8]:
                memory_used.append({"level": "profile", "line": _extract_line(m)})
            for m in context.get("ranked_memories", [])[:7]:
                memory_used.append({
                    "level": m.get("memory_level", "user"),
                    "line": _extract_line(m)
                })
            yield {"type": "memory.used", "conversation_id": conversation.conversation_id, "items": memory_used}

            # Research decision
            research_notes: Optional[str] = None
            needs_research = await self._needs_research(request.message, context)
            if needs_research and request.use_web_search and not cancel_signal.is_set():
                yield {"type": "research.started", "conversation_id": conversation.conversation_id}
                try:
                    research_request = ResearchRequest(query=request.message, max_sources=15, deep_dive=True)
                    brief = await self.research_engine.run_research(research_request)
                    research_notes = await self._format_research_conversationally(brief)
                    yield {
                        "type": "research.done",
                        "conversation_id": conversation.conversation_id,
                        "brief_id": brief.brief_id,
                        "sources": brief.total_sources,
                        "findings": len(brief.findings),
                        "ideas": len(brief.ideas)
                    }
                    research_message = ChatMessage(
                        conversation_id=conversation.conversation_id,
                        role=MessageRole.ASSISTANT,
                        content=research_notes,
                        message_type=MessageType.RESEARCH_RESULT,
                        research_brief_id=brief.brief_id
                    )
                    conversation.add_message(research_message)
                    await self.memory_manager.update_from_message(research_message, user_id)
                    context["research_notes"] = research_notes
                except Exception as e:
                    logger.error("Research stage failed", error=str(e))
                    yield {"type": "error", "stage": "research", "message": str(e)}

            if cancel_signal.is_set():
                yield {"type": "chat.final", "conversation_id": conversation.conversation_id, "message": {"content": "Cancelled.", "role": MessageRole.ASSISTANT.value}}
                return

            # LLM messages
            messages = await self._prepare_llm_messages(conversation, request.message, context)

            # Stream generation
            stream = await self.groq_client.chat.completions.create(
                model=settings.groq_model,
                messages=messages,
                temperature=settings.chat_temperature,
                max_tokens=settings.chat_max_tokens,
                stream=True
            )

            full_response = ""
            async for chunk in stream:
                if cancel_signal.is_set():
                    break
                if chunk.choices[0].delta.content:
                    delta = chunk.choices[0].delta.content
                    full_response += delta
                    yield {"type": "chat.token", "conversation_id": conversation.conversation_id, "delta": delta}

            if cancel_signal.is_set():
                yield {"type": "chat.final", "conversation_id": conversation.conversation_id, "message": {"content": "Cancelled.", "role": MessageRole.ASSISTANT.value}}
                cancel_signal.clear()
                return

            # Persist assistant turn
            assistant_message = ChatMessage(
                conversation_id=conversation.conversation_id,
                role=MessageRole.ASSISTANT,
                content=full_response,
                message_type=MessageType.TEXT,
                context_used={"context_window": request.context_window}
            )
            conversation.add_message(assistant_message)
            await self.memory_manager.update_from_message(assistant_message, user_id)

            # Rolling summary
            created = await self._maybe_create_rolling_summary(conversation)
            if created:
                yield {"type": "memory.summary_created", "conversation_id": conversation.conversation_id}

            # Store
            self.conversations[conversation.conversation_id] = conversation

            # Suggestions and final
            suggestions = await self._generate_suggestions(conversation, full_response)
            yield {
                "type": "chat.final",
                "conversation_id": conversation.conversation_id,
                "message": {"content": full_response, "role": MessageRole.ASSISTANT.value},
                "suggestions": suggestions
            }

        except Exception as e:
            logger.error("Failed to stream message", error=str(e))
            yield {"type": "error", "message": str(e)}
    
    async def process_conversational_message(
        self,
        request: ChatRequest,
        user_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Process message conversationally with streaming updates (merged from EnhancedChatEngine)"""
        try:
            # Ensure dual context (conversation and user)
            conversation_id = request.conversation_id or str(uuid.uuid4())
            conversation_id, user_id = self.memory_manager.ensure_dual_context(
                conversation_id,
                user_id
            )
            
            # Get or create conversation
            conversation = await self._get_or_create_conversation(
                conversation_id,
                user_id
            )
            
            # Create and persist user message
            user_message = ChatMessage(
                conversation_id=conversation.conversation_id,
                role=MessageRole.USER,
                content=request.message,
                message_type=MessageType.TEXT
            )
            conversation.add_message(user_message)
            await self.memory_manager.update_from_message(user_message, user_id)
            
            # Run conversational flow manager
            flow_result = await self.flow_manager.process_conversation_turn(
                request.message,
                conversation.conversation_id,
                user_id
            )
            
            # Stream conversational updates based on strategy/routing
            async for update in self._handle_conversational_flow(
                flow_result,
                conversation,
                user_id
            ):
                yield update
            
            # Store conversation
            self.conversations[conversation.conversation_id] = conversation
        except Exception as e:
            logger.error("Failed to process conversational message", error=str(e))
            yield {
                "type": "error",
                "content": f"I encountered an error: {str(e)}. Let me try to help you differently.",
                "error": str(e)
            }
    
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
    
    async def _handle_conversational_flow(
        self,
        flow_result: Dict[str, Any],
        conversation: Conversation,
        user_id: Optional[str]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle conversational flow with appropriate responses"""
        routing = flow_result["routing"]
        strategy = flow_result["strategy"]
        
        # Initial acknowledgment
        yield {
            "type": "acknowledgment",
            "content": self._get_acknowledgment_message(routing),
            "intent": routing["intent"],
            "confidence": routing["confidence"]
        }
        
        if strategy["type"] == "clarification":
            async for update in self._handle_clarification_flow(routing, conversation):
                yield update
        elif strategy["type"] == "confirmation":
            async for update in self._handle_confirmation_flow(routing, conversation):
                yield update
        elif strategy["type"] == "multi-step":
            async for update in self._handle_multi_step_flow(routing, conversation, user_id):
                yield update
        else:
            async for update in self._handle_direct_action(routing, conversation, user_id):
                yield update
    
    def _get_acknowledgment_message(self, routing: Dict[str, Any]) -> str:
        intent = routing["intent"]
        confidence = routing["confidence"]
        if confidence > 0.8:
            messages = {
                "research": "I'll help you research that. Let me gather the information...",
                "crm_action": "I'll update your CRM with that information...",
                "pms_action": "I'll handle that in your project management system...",
                "report_generation": "I'll generate that report for you...",
                "meeting_scheduling": "I'll help you schedule that meeting...",
                "general_chat": "Let me help you with that..."
            }
        else:
            messages = {
                "research": "I think you want me to research something. Let me confirm...",
                "crm_action": "It seems you want to update the CRM. Let me verify...",
                "pms_action": "You'd like me to update the project system, correct?",
                "report_generation": "You need a report generated, right?",
                "meeting_scheduling": "You want to schedule a meeting, is that correct?",
                "general_chat": "Let me make sure I understand correctly..."
            }
        return messages.get(intent, routing.get("suggested_response", "I understand. Let me help you with that..."))
    
    async def _handle_clarification_flow(
        self,
        routing: Dict[str, Any],
        conversation: Conversation
    ) -> AsyncGenerator[Dict[str, Any], None]:
        questions = routing.get("clarification_questions", [
            "Could you provide more details?",
            "What specific information are you looking for?",
            "What would you like me to focus on?"
        ])
        yield {
            "type": "clarification_needed",
            "content": "I need a bit more information to help you effectively.",
            "questions": questions,
            "waiting_for_response": True
        }
    
    async def _handle_confirmation_flow(
        self,
        routing: Dict[str, Any],
        conversation: Conversation
    ) -> AsyncGenerator[Dict[str, Any], None]:
        action_preview = self._generate_action_preview(routing)
        yield {
            "type": "confirmation_needed",
            "content": "Here's what I'm about to do:",
            "preview": action_preview,
            "actions": [
                {"label": "Confirm", "value": "confirm"},
                {"label": "Modify", "value": "modify"},
                {"label": "Cancel", "value": "cancel"}
            ],
            "waiting_for_response": True
        }
    
    async def _handle_multi_step_flow(
        self,
        routing: Dict[str, Any],
        conversation: Conversation,
        user_id: Optional[str]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        handler_name = routing["handler"]
        handler = self.action_handlers.get(handler_name, self._handle_chat_action)
        steps = routing["strategy"]["steps"]
        for i, step in enumerate(steps):
            if step == "acknowledge":
                continue
            elif step == "show_progress":
                yield {
                    "type": "progress",
                    "content": "Working on it...",
                    "step": i + 1,
                    "total_steps": len(steps),
                    "current_action": step
                }
            elif step == "execute":
                async for result in handler(routing, conversation, user_id):
                    yield result
            elif step == "present_results":
                pass
            await asyncio.sleep(0.5)
    
    async def _handle_direct_action(
        self,
        routing: Dict[str, Any],
        conversation: Conversation,
        user_id: Optional[str]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        handler_name = routing["handler"]
        handler = self.action_handlers.get(handler_name, self._handle_chat_action)
        async for result in handler(routing, conversation, user_id):
            yield result
    
    async def _handle_research_action(
        self,
        routing: Dict[str, Any],
        conversation: Conversation,
        user_id: Optional[str]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        try:
            params = routing.get("parameters", {})
            entities = routing.get("entities", {})
            topics = entities.get("topics", [])
            original_message = conversation.messages[-1].content
            research_request = ResearchRequest(
                query=original_message,
                max_sources=params.get("max_sources", 15),
                deep_dive=True
            )
            yield {
                "type": "research_started",
                "content": "🔍 Starting research...",
                "topics": topics
            }
            research_brief = await self.research_engine.run_research(research_request)
            yield {
                "type": "research_progress",
                "content": f"Found {research_brief.total_sources} relevant sources",
                "sources_count": research_brief.total_sources
            }
            formatted_response = await self._format_research_conversationally(research_brief)
            yield {
                "type": "research_complete",
                "content": formatted_response,
                "brief_id": research_brief.brief_id,
                "findings_count": len(research_brief.findings),
                "ideas_count": len(research_brief.ideas),
                "metadata": {
                    "sources": research_brief.total_sources,
                    "confidence": research_brief.average_confidence
                }
            }
            research_message = ChatMessage(
                conversation_id=conversation.conversation_id,
                role=MessageRole.ASSISTANT,
                content=formatted_response,
                message_type=MessageType.RESEARCH_RESULT,
                research_brief_id=research_brief.brief_id
            )
            conversation.add_message(research_message)
            await self.memory_manager.update_from_message(research_message, user_id)
        except Exception as e:
            logger.error("Research action failed", error=str(e))
            yield {"type": "error", "content": f"I encountered an issue with the research: {str(e)}"}
    
    async def _handle_crm_action(
        self,
        routing: Dict[str, Any],
        conversation: Conversation,
        user_id: Optional[str]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        action = routing.get("action")
        params = routing.get("parameters", {})
        try:
            if action == "create_note":
                yield {"type": "crm_action_started", "content": "Creating note in CRM...", "action": action}
                note_id = await self.crm_client.create_note(
                    lead_id=params.get("lead_id", "default"),
                    subject=params.get("subject", "Note from conversation"),
                    description=params.get("content", conversation.messages[-1].content)
                )
                yield {"type": "crm_action_complete", "content": f"✅ Note created successfully in CRM (ID: {note_id})", "result": {"note_id": note_id}}
            elif action == "create_task":
                yield {"type": "crm_action_started", "content": "Creating task in CRM...", "action": action}
                task_id = await self.crm_client.create_task(
                    business_id=params.get("business_id", "default"),
                    name=params.get("title", "Task from conversation"),
                    description=params.get("description", ""),
                    priority=params.get("priority", "MEDIUM")
                )
                yield {"type": "crm_action_complete", "content": f"✅ Task created successfully (ID: {task_id})", "result": {"task_id": task_id}}
            else:
                yield {"type": "crm_action_info", "content": "I can help you with CRM actions like creating notes, tasks, or scheduling meetings. What would you like to do?"}
        except Exception as e:
            logger.error("CRM action failed", error=str(e))
            yield {"type": "error", "content": f"CRM action failed: {str(e)}"}
    
    async def _handle_pms_action(
        self,
        routing: Dict[str, Any],
        conversation: Conversation,
        user_id: Optional[str]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        action = routing.get("action")
        params = routing.get("parameters", {})
        try:
            if action == "create_work_item":
                yield {"type": "pms_action_started", "content": "Creating work item in project system...", "action": action}
                work_item_id = await self.pms_client.create_work_item(
                    project_id=params.get("project_id", "default"),
                    title=params.get("title", "Work item from conversation"),
                    description=params.get("description", ""),
                    item_type=params.get("type", "TASK"),
                    priority=params.get("priority", "MEDIUM")
                )
                yield {"type": "pms_action_complete", "content": f"✅ Work item created successfully (ID: {work_item_id})", "result": {"work_item_id": work_item_id}}
            elif action == "create_page":
                yield {"type": "pms_action_started", "content": "Creating documentation page...", "action": action}
                page_id = await self.pms_client.create_page(
                    project_id=params.get("project_id", "default"),
                    title=params.get("title", "Documentation"),
                    content=params.get("content", "")
                )
                yield {"type": "pms_action_complete", "content": f"✅ Documentation page created (ID: {page_id})", "result": {"page_id": page_id}}
            else:
                yield {"type": "pms_action_info", "content": "I can help you manage projects, create work items, or update documentation. What would you like to do?"}
        except Exception as e:
            logger.error("PMS action failed", error=str(e))
            yield {"type": "error", "content": f"Project action failed: {str(e)}"}
    
    async def _handle_report_action(
        self,
        routing: Dict[str, Any],
        conversation: Conversation,
        user_id: Optional[str]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        yield {"type": "report_generation_started", "content": "Generating report based on our conversation..."}
        summary = await self._generate_conversation_summary(conversation)
        yield {"type": "report_ready", "content": "Report generated successfully!", "report": summary, "format_options": ["PDF", "Word", "Markdown"]}
    
    async def _handle_calendar_action(
        self,
        routing: Dict[str, Any],
        conversation: Conversation,
        user_id: Optional[str]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        yield {
            "type": "meeting_scheduling",
            "content": "Checking calendar availability...",
            "proposed_times": [
                "Tomorrow at 2:00 PM",
                "Thursday at 10:00 AM",
                "Friday at 3:00 PM"
            ]
        }
    
    async def _handle_chat_action(
        self,
        routing: Dict[str, Any],
        conversation: Conversation,
        user_id: Optional[str]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        # Prepare dual context for conversational chat
        context = await self._prepare_dual_context(conversation, user_id)
        messages = await self._prepare_conversational_messages(
            conversation,
            routing,
            context
        )
        # Enrich system prompt with profile and ranked long-term memories
        try:
            memory_context = await self._prepare_context(
                conversation,
                ChatRequest(message=conversation.messages[-1].content if conversation.messages else "", conversation_id=conversation.conversation_id, use_long_term_memory=True, use_web_search=False),
                user_id
            )
            system_injection_lines: List[str] = []
            def _extract_line(m: Dict[str, Any]) -> str:
                if isinstance(m, dict):
                    return (m.get("memory") or m.get("content") or str(m)).strip()
                return str(m).strip()
            profile = memory_context.get("profile", [])
            ranked = memory_context.get("ranked_memories", [])
            if profile:
                system_injection_lines.append("\nPROFILE:")
                for m in profile[:8]:
                    system_injection_lines.append(f"- {_extract_line(m)[:200]}")
            if ranked:
                system_injection_lines.append("\nLONG-TERM FACTS:")
                for m in ranked[:5]:
                    system_injection_lines.append(f"- {_extract_line(m)[:200]}")
            if system_injection_lines and messages and messages[0].get("role") == "system":
                messages[0]["content"] = messages[0]["content"] + "\n" + "\n".join(system_injection_lines)
        except Exception:
            # Non-fatal enrichment failure
            pass
        stream = await self.groq_client.chat.completions.create(
            model=settings.groq_model,
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
            stream=True
        )
        full_response = ""
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                yield {"type": "chat_chunk", "content": content}
        assistant_message = ChatMessage(
            conversation_id=conversation.conversation_id,
            role=MessageRole.ASSISTANT,
            content=full_response,
            message_type=MessageType.TEXT
        )
        conversation.add_message(assistant_message)
        await self.memory_manager.update_from_message(assistant_message, user_id)
        yield {"type": "chat_complete", "full_response": full_response}
    
    async def _prepare_context(
        self,
        conversation: Conversation,
        request: ChatRequest,
        user_id: Optional[str]
    ) -> Dict[str, Any]:
        """Prepare context for message processing"""
        context: Dict[str, Any] = {}
        
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
        
        # Profile facts (always retrieve if user_id)
        profile: List[Dict[str, Any]] = []
        if user_id:
            profile = await self.memory_manager.get_profile(user_id)
        context["profile"] = profile

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

            # Composite ranking of memories
            now = datetime.utcnow()
            def _score(m: Dict[str, Any]) -> float:
                meta = m.get("metadata", {}) if isinstance(m, dict) else {}
                semantic = float(m.get("score", 0.0))
                ts = meta.get("timestamp")
                try:
                    # Support both with and without Z suffix
                    ts_norm = ts.replace("Z", "+00:00") if isinstance(ts, str) else None
                    age_days = (now - datetime.fromisoformat(ts_norm)).total_seconds()/86400 if ts_norm else 365.0
                except Exception:
                    age_days = 365.0
                rec = math.exp(-age_days/30.0)
                boost = 0.0
                if meta.get("conversation_id") == conversation.conversation_id:
                    boost += 0.1
                if meta.get("memory_level") == "profile" or meta.get("pinned") is True:
                    boost += 0.2
                return 0.6*semantic + 0.3*rec + boost
            ranked = sorted(memories, key=_score, reverse=True)
            context["ranked_memories"] = ranked
        
        # Add conversation metadata
        context["conversation_metadata"] = {
            "message_count": conversation.get_message_count(),
            "created_at": conversation.created_at.isoformat(),
            "entities": conv_context.entities,
            "topics": conv_context.topics
        }
        
        # Include any existing conversation summary from short-term cache if present
        st = conv_context.short_term or {}
        summaries = [m for m in st.get("recent_messages", []) if isinstance(m, dict) and m.get("metadata", {}).get("type") == "summary"]
        if summaries:
            last = summaries[-1]
            context["conversation_summary"] = last.get("content")

        return context

    async def _maybe_create_rolling_summary(self, conversation: Conversation) -> bool:
        """Create a rolling summary after every ~10 turns and store it as conversation-level memory.
        Returns True if a summary was created this turn.
        """
        conv_id = conversation.conversation_id
        self._turn_counters[conv_id] = self._turn_counters.get(conv_id, 0) + 1
        if self._turn_counters[conv_id] < 10:
            return False
        # Reset counter
        self._turn_counters[conv_id] = 0
        try:
            messages_text = "\n".join([
                f"{m.role.value}: {m.content}" for m in conversation.messages[-40:]
            ])
            prompt = f"Summarize this recent conversation window concisely in 6-8 bullet points:\n\n{messages_text}"
            response = await self.groq_client.chat.completions.create(
                model=settings.groq_model,
                messages=[
                    {"role": "system", "content": "Create a concise rolling summary"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=settings.summary_max_tokens
            )
            summary_text = response.choices[0].message.content
            await self.memory_manager.add_to_memory(
                conversation_id=conv_id,
                content=summary_text,
                metadata={"memory_level": "conversation", "type": "summary", "pinned": True},
                user_id=conversation.user_id,
                memory_type="conversation"
            )
            return True
        except Exception as e:
            logger.warning("Failed to create rolling summary", error=str(e))
        return False
    
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

        # Structured system prompt scaffold
        def _extract_line(m: Dict[str, Any]) -> str:
            if isinstance(m, dict):
                return m.get("memory") or m.get("content") or str(m)
            return str(m)

        system_lines: List[str] = []
        system_lines.append(
            "You are an intelligent assistant with research capabilities and long-term memory."
        )
        system_lines.append(
            "Be helpful, accurate, concise, and personalize using the provided profile and context."
        )

        # PROFILE
        profile = context.get("profile", [])
        if profile:
            system_lines.append("\nPROFILE:")
            for m in profile[:8]:
                system_lines.append(f"- {_extract_line(m)[:200]}")

        # RECENT SUMMARY
        summary_text = context.get("conversation_summary")
        if summary_text:
            system_lines.append("\nRECENT SUMMARY:")
            system_lines.append(summary_text[:800])

        # LONG-TERM FACTS (ranked)
        ranked = context.get("ranked_memories", [])
        if ranked:
            system_lines.append("\nLONG-TERM FACTS:")
            for m in ranked[:5]:
                system_lines.append(f"- {_extract_line(m)[:200]}")

        # Topics/Entities
        topics = context.get("conversation_metadata", {}).get("topics")
        if topics:
            system_lines.append("\nTOPICS: " + ", ".join(topics[:10]))

        # RESEARCH NOTES (if present)
        research_notes = context.get("research_notes")
        if research_notes:
            system_lines.append("\nRESEARCH NOTES:")
            system_lines.append(research_notes[:800])

        system_prompt = "\n".join(system_lines)
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
    
    async def _prepare_conversational_messages(
        self,
        conversation: Conversation,
        routing: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Prepare messages for conversational LLM response"""
        system_prompt = """You are a helpful AI assistant with access to various tools and systems.
You can help with research, manage CRM and project systems, generate reports, and schedule meetings.
Always be conversational, natural, and helpful. Acknowledge what the user is asking for and explain what you're doing.

Current context:
- Intent detected: {intent}
- Confidence: {confidence}
- Entities: {entities}

Respond naturally and conversationally.""".format(
            intent=routing.get("intent", "unknown"),
            confidence=routing.get("confidence", 0),
            entities=json.dumps(routing.get("entities", {}))
        )
        messages = [{"role": "system", "content": system_prompt}]
        for msg in conversation.messages[-5:]:
            if msg.role != MessageRole.SYSTEM:
                messages.append({
                    "role": msg.role.value,
                    "content": msg.content
                })
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
    
    async def _format_research_conversationally(self, brief: ResearchBrief) -> str:
        """Format research results conversationally"""
        response = f"I've completed the research on **{brief.query}**. Here's what I found:\n\n"
        if brief.executive_summary:
            response += f"**Summary:** {brief.executive_summary}\n\n"
        response += f"**Key Findings:**\n"
        for i, finding in enumerate(brief.findings[:3], 1):
            response += f"{i}. **{finding.title}** - {finding.summary}\n"
        if brief.ideas:
            response += f"\n**Recommendations:**\n"
            for i, idea in enumerate(brief.ideas[:3], 1):
                response += f"{i}. {idea.idea}\n"
        response += f"\n*Based on {brief.total_sources} sources with {brief.average_confidence:.0%} confidence*"
        return response
    
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
    
    async def _generate_conversation_summary(
        self,
        conversation: Conversation
    ) -> Dict[str, Any]:
        """Generate summary of conversation"""
        messages_text = "\n".join([
            f"{msg.role.value}: {msg.content}"
            for msg in conversation.messages[-10:]
        ])
        prompt = f"Summarize this conversation concisely:\n\n{messages_text}"
        response = await self.groq_client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": "Generate a concise summary"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=500
        )
        return {
            "summary": response.choices[0].message.content,
            "message_count": conversation.get_message_count(),
            "duration": (datetime.utcnow() - conversation.created_at).total_seconds(),
            "topics": conversation.context.topics
        }
    
    async def _prepare_dual_context(
        self,
        conversation: Conversation,
        user_id: Optional[str]
    ) -> Dict[str, Any]:
        """Prepare dual (user + conversation) context for conversational flows"""
        context = await self.memory_manager.get_conversation_context(
            conversation.conversation_id,
            user_id
        )
        return {
            "conversation_id": conversation.conversation_id,
            "user_id": user_id,
            "message_count": conversation.get_message_count(),
            "entities": context.entities,
            "topics": context.topics,
            "conversation_context": context.short_term,
            "user_context": context.long_term
        }
    
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

    async def cancel(self, conversation_id: str) -> None:
        """Cancel in-flight generation for the given conversation."""
        signal = self._cancel_signals.get(conversation_id)
        if signal:
            signal.set()
