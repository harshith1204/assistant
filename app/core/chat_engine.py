"""Chat engine with research integration and context management"""

import json
import asyncio
import uuid
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime, timezone
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
from app.research.service import ResearchService
from app.core.research_engine import ResearchEngine
from app.core.intent import (
    IntentDetector
)
from app.integrations.mcp_client import mongodb_mcp_client

logger = structlog.get_logger()


class ChatEngine:
    """Main chat engine with research capabilities"""
    
    def __init__(self):
        """Initialize chat engine"""
        print(f"🔧 Initializing ChatEngine with model: {settings.llm_model}")
        self.groq_client = AsyncGroq(api_key=settings.groq_api_key)
        self.memory_manager = MemoryManager()
        self.research_service = ResearchService()
        self.research_engine = ResearchEngine()
        # Simple intent detection
        self.intent_detector = IntentDetector()
        self.mcp_client = mongodb_mcp_client
        self.conversations: Dict[str, Conversation] = {}
        self._turn_counters: Dict[str, int] = {}
        self._cancel_signals: Dict[str, asyncio.Event] = {}

        # Initialize MCP client connection (will be done asynchronously when needed)
        # asyncio.create_task(self._init_mcp_client())  # Commented out to avoid event loop issues
        print("✅ ChatEngine initialized successfully")

        # Action handlers mapping (for conversational routing)
        self.action_handlers = {
            "research_engine": self._handle_research_action,
            "mongodb_client": self._handle_unified_action,  # Unified handler for MCP client
            "report_generator": self._handle_report_action,
            "calendar_manager": self._handle_calendar_action,
            "chat_engine": self._handle_chat_action
        }

    async def _init_mcp_client(self):
        """Initialize MCP client connection"""
        try:
            logger.info("Initializing MCP client connection...")
            connected = await self.mcp_client.connect()
            if connected:
                logger.info("✅ MCP client connected successfully")
                # Test the connection by listing tools
                tools = await self.mcp_client.list_tools()
                logger.info(f"MCP client found {len(tools)} tools", tools=[tool.get('name') for tool in tools])
                # Run smoke test for MongoDB connectivity
                await self._smoke_test_mongo()
            else:
                logger.warning("❌ Failed to connect MCP client")
        except Exception as e:
            logger.error("Failed to initialize MCP client", error=str(e))

    async def _smoke_test_mongo(self):
        """Smoke test MongoDB MCP connectivity"""
        try:
            logger.info("Running MCP smoke test...")
            row_count = 0
            async for r in self.mcp_client.find_documents("Lead", {"_id": {"$exists": True}}, {"_id": 1}, 1, user_id="smoke"):
                if r.get("type") == "tool.output.data":
                    logger.info("MCP smoke test OK", sample=r.get("data"))
                    row_count += 1
                    break
                elif r.get("type") == "error":
                    logger.error("MCP smoke test error", error=r)
                    break
                else:
                    logger.debug("MCP smoke test received non-data event", event=r)

            if row_count == 0:
                logger.warning("MCP smoke test returned no rows (check collection names and server connectivity)")
        except Exception as e:
            logger.error("MCP smoke test failed", error=str(e))

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
        """Stream typed events over a single WebSocket loop - 5-stage FSM pipeline."""
        try:
            # ===== STAGE 1: RECEIVE & LOG =====
            # Ensure dual context - prioritize passed user_id over request
            conversation_id = request.conversation_id or str(uuid.uuid4())
            # Use the user_id passed to the method (from websocket) as primary source
            effective_user_id = user_id or request.user_id
            conversation_id, effective_user_id = self.memory_manager.ensure_dual_context(
                conversation_id,
                effective_user_id
            )
            # Use effective_user_id throughout this method
            user_id = effective_user_id

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

            # Add to conversation and memory (with write gates)
            conversation.add_message(user_message)
            await self.memory_manager.update_from_message(user_message, user_id)

            # Load context (profile + rolling summary + ranked memories)
            # CRITICAL: This must load ALL user memories across conversations
            context = await self._prepare_context(
                conversation,
                request,
                user_id  # This user_id is consistent for the user across all conversations
            )

            # Emit memory.used event
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

            # ===== STAGE 2: UNDERSTAND INTENT =====
            intent_result = await self.intent_detector.detect_intent(request.message, context)

            # Normalize intent payload from intent.py (handle both "label" and "intent" fields)
            raw_intent = intent_result.get("label") or intent_result.get("intent") or "general"
            intent = str(raw_intent).lower()

            confidence = intent_result.get("confidence", 0.5)
            entities = intent_result.get("entities", {}) or {}

            # Heuristic fallback: if user is clearly asking about DB-y stuff, force a db intent
            msg_lower = (request.message or "").lower()
            if intent == "general" and any(k in msg_lower for k in ["crm", "project", "task", "staff", "hrms", "mongo", "mongodb", "db", "collection"]):
                intent = "db.find"

            # Debug logging for intent detection
            logger.info("Intent detection result", intent=intent, confidence=confidence, entities=entities)

            # Low confidence - request clarification
            if confidence < 0.5 and not request.skip_clarification:
                yield {
                    "type": "clarification_needed",
                    "conversation_id": conversation.conversation_id,
                    "question": intent_result.get("clarification_question", "Could you please clarify what you'd like me to help with?"),
                    "confidence": confidence,
                    "detected_intent": intent
                }
                return

            # ===== STAGE 3: PLAN (routing policy) =====
            routing_strategy = self._determine_routing_strategy(
                intent, confidence, entities, request, context
            )

            # Debug logging for routing strategy
            logger.info("Routing strategy determined", intent=intent, strategy=routing_strategy)

            # ===== STAGE 4: EXECUTE =====
            research_notes: Optional[str] = None

            # Handle profile update intent
            if intent == "profile_update" and entities.get("profile_facts"):
                for fact in entities.get("profile_facts", []):
                    await self.memory_manager.set_profile_fact(
                        user_id, fact["key"], fact["value"], fact.get("priority", 50)
                    )
                    yield {
                        "type": "memory.written",
                        "level": "profile",
                        "key": fact["key"],
                        "value": fact["value"]
                    }
                # Confirm and return
                yield {
                    "type": "chat.final",
                    "conversation_id": conversation.conversation_id,
                    "message": {"content": "I've updated your profile with the information you provided.", "role": MessageRole.ASSISTANT.value}
                }
                return

            # MongoDB-first strategy
            if routing_strategy == "mongodb-first" and not cancel_signal.is_set():
                yield {"type": "mongodb.started", "conversation_id": conversation.conversation_id}
                try:
                    # Call the MongoDB handler directly for streaming
                    routing_info = {
                        "intent": intent,
                        "entities": entities,
                        "parameters": {},
                        "handler": "mongodb_client"
                    }
                    async for chunk in self._handle_mongodb_action(routing_info, conversation, user_id):
                        yield {
                            "type": "mongodb.chunk",
                            "conversation_id": conversation.conversation_id,
                            "data": chunk
                        }
                    yield {
                        "type": "mongodb.done",
                        "conversation_id": conversation.conversation_id,
                        "intent": intent,
                        "entities": entities
                    }
                except Exception as e:
                    logger.error("MongoDB stage failed", error=str(e))
                    yield {"type": "error", "stage": "mongodb", "message": str(e)}

            # Research-first strategy
            if routing_strategy == "research-first" and not cancel_signal.is_set():
                yield {"type": "research.started", "conversation_id": conversation.conversation_id}
                try:
                    research_request = ResearchRequest(query=request.message, max_sources=15, deep_dive=True)
                    # Stream research progress
                    async for chunk in self._stream_research(research_request):
                        yield {
                            "type": "research.chunk",
                            "conversation_id": conversation.conversation_id,
                            "data": chunk
                        }
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
                    context["research_notes"] = research_notes
                except Exception as e:
                    logger.error("Research stage failed", error=str(e))
                    yield {"type": "error", "stage": "research", "message": str(e)}

            if cancel_signal.is_set():
                yield {"type": "chat.final", "conversation_id": conversation.conversation_id, "message": {"content": "Cancelled.", "role": MessageRole.ASSISTANT.value}}
                return

            # Generate response with enhanced context
            messages = await self._prepare_llm_messages(conversation, request.message, context)

            # Stream generation
            print(f"🤖 Making API call with model: '{settings.groq_model}', temperature: {settings.chat_temperature}")
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

            # ===== STAGE 5: COMPLETE =====
            # Persist assistant turn (with write gates)
            assistant_message = ChatMessage(
                conversation_id=conversation.conversation_id,
                role=MessageRole.ASSISTANT,
                content=full_response,
                message_type=MessageType.TEXT,
                context_used={"context_window": request.context_window, "intent": intent}
            )
            conversation.add_message(assistant_message)
            
            # Check for high-signal facts to write
            high_signal_facts = await self._extract_high_signal_facts(full_response, entities)
            if high_signal_facts:
                await self.memory_manager.update_from_message(assistant_message, user_id)
                yield {"type": "memory.written", "level": "user", "facts": len(high_signal_facts)}

            # Rolling summary every N turns
            created = await self._maybe_create_rolling_summary(conversation)
            if created:
                yield {"type": "memory.summary_created", "conversation_id": conversation.conversation_id}

            # Store conversation
            self.conversations[conversation.conversation_id] = conversation

            # Generate suggestions
            suggestions = await self._generate_suggestions(conversation, full_response)
            
            # Final event with complete message
            yield {
                "type": "chat.final",
                "conversation_id": conversation.conversation_id,
                "message": {"content": full_response, "role": MessageRole.ASSISTANT.value},
                "suggestions": suggestions,
                "intent": intent,
                "confidence": confidence
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

            # Prepare base context for research
            base_context = await self._prepare_context(
                conversation,
                ChatRequest(message=original_message),  # Create a minimal request for context
                user_id
            )

            # Prepare specialized research context
            research_context = await self._prepare_research_context(
                conversation, original_message, user_id, base_context
            )

            # Extract inferred parameters and merge with routing parameters
            inferred_params = research_context.get("inferred_parameters", {})
            merged_params = {**inferred_params, **params}  # routing params override inferred

            yield {
                "type": "research_started",
                "content": "🔍 Starting comprehensive research...",
                "topics": topics,
                "inferred_params": inferred_params
            }

            # Prepare conversation context for research service
            conversation_context = {
                "entities": research_context.get("research_entities", {}),
                "topics": research_context.get("conversation_metadata", {}).get("topics", []),
                "preferences": {
                    "industry": merged_params.get("industry"),
                    "geography": merged_params.get("geography")
                },
                "research_history": research_context.get("research_history", [])
            }

            research_brief = await self.research_service.research(
                query=original_message,
                scope=merged_params.get("scope", []),
                industry=merged_params.get("industry"),
                geography=merged_params.get("geography"),
                max_sources=merged_params.get("max_sources", 15),
                user_id=user_id,
                conversation_context=conversation_context
            )
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
                    "confidence": getattr(research_brief, 'research_quality_score', 0.8),
                    "business_type": getattr(research_brief, 'business_type', None),
                    "strategic_priorities": getattr(research_brief, 'strategic_priorities', [])
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
            # Create intelligent error response
            error_type = self._categorize_research_error(e)
            error_message = await self._create_generic_error_response(original_message, e)
            yield {"type": "error", "content": error_message, "error_type": error_type}
    
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
                note_id = await self.mcp_client.create_note(
                    collection=settings.mongodb_crm_collection or "crm_notes",
                    lead_id=params.get("lead_id", "default"),
                    subject=params.get("subject", "Note from conversation"),
                    description=params.get("content", conversation.messages[-1].content),
                    user_id=user_id
                )
                yield {"type": "crm_action_complete", "content": f"✅ Note created successfully in CRM (ID: {note_id})", "result": {"note_id": note_id}}
            elif action == "create_task":
                yield {"type": "crm_action_started", "content": "Creating task in CRM...", "action": action}
                task_id = await self.mcp_client.create_task(
                    collection=settings.mongodb_crm_collection or "crm_notes",
                    business_id=params.get("business_id", "default"),
                    name=params.get("title", "Task from conversation"),
                    description=params.get("description", ""),
                    priority=params.get("priority", "MEDIUM"),
                    user_id=user_id
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
                work_item_id = await self.mcp_client.create_work_item(
                    collection=settings.mongodb_pms_collection or "pms_pages",
                    project_id=params.get("project_id", "default"),
                    title=params.get("title", "Work item from conversation"),
                    description=params.get("description", ""),
                    item_type=params.get("type", "TASK"),
                    priority=params.get("priority", "MEDIUM"),
                    user_id=user_id
                )
                yield {"type": "pms_action_complete", "content": f"✅ Work item created successfully (ID: {work_item_id})", "result": {"work_item_id": work_item_id}}
            elif action == "create_page":
                yield {"type": "pms_action_started", "content": "Creating documentation page...", "action": action}
                page_id = await self.mcp_client.create_page(
                    collection=settings.mongodb_pms_collection or "pms_pages",
                    project_id=params.get("project_id", "default"),
                    title=params.get("title", "Documentation"),
                    content=params.get("content", ""),
                    user_id=user_id
                )
                yield {"type": "pms_action_complete", "content": f"✅ Documentation page created (ID: {page_id})", "result": {"page_id": page_id}}
            else:
                yield {"type": "pms_action_info", "content": "I can help you manage projects, create work items, or update documentation. What would you like to do?"}
        except Exception as e:
            logger.error("PMS action failed", error=str(e))
            yield {"type": "error", "content": f"Project action failed: {str(e)}"}

    async def _handle_unified_action(
        self,
        routing: Dict[str, Any],
        conversation: Conversation,
        user_id: Optional[str]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Unified handler for MCP client actions (CRM, PMS, and MongoDB operations)"""
        intent = routing.get("intent", "")
        action = routing.get("action")
        params = routing.get("parameters", {})

        try:
            # Route based on intent type
            if intent == "crm_action":
                async for result in self._handle_crm_action(routing, conversation, user_id):
                    yield result
            elif intent == "pms_action":
                async for result in self._handle_pms_action(routing, conversation, user_id):
                    yield result
            else:
                # Handle MongoDB database operations
                async for result in self._handle_mongodb_action(routing, conversation, user_id):
                    yield result

        except Exception as e:
            logger.error("Unified action failed", error=str(e), intent=intent)
            yield {"type": "error", "content": f"Action failed: {str(e)}", "intent": intent}

    async def _handle_mongodb_action(
        self,
        routing: Dict[str, Any],
        conversation: Conversation,
        user_id: Optional[str]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle MongoDB database operations via MCP"""
        intent = routing.get("intent", "")
        entities = routing.get("entities", {})
        params = routing.get("parameters", {})

        # Log that MCP is being called
        logger.info("MCP handler called", intent=intent, entities=entities, params=params)

        try:
            # Extract collection and query details from entities
            collections = entities.get("collections", [])
            collection = collections[0] if collections else params.get("collection", "test_db")  # Default to test_db

            # If collection is missing, infer from message/entities
            if not collections and collection == "test_db":
                msg = (conversation.messages[-1].content if conversation.messages else "").lower()
                inferred = None

                # Priority-based inference (most specific first)
                if "lead status" in msg or ("status" in msg and ("lead" in msg or "customer" in msg)):
                    inferred = "leadStatus"
                elif any(x in msg for x in ["activity", "activities"]) and not "status" in msg:
                    inferred = "activity"
                elif any(x in msg for x in ["note", "notes", "comment", "comments"]) and not "lead" in msg:
                    inferred = "notes"
                elif any(x in msg for x in ["task", "tasks", "todo", "work item"]):
                    inferred = "task"
                elif any(x in msg for x in ["meeting", "meetings", "appointment", "call"]):
                    inferred = "meeting"
                elif any(x in msg for x in ["crm", "lead", "leads", "deal", "customer", "client"]):
                    inferred = "Lead"
                elif any(x in msg for x in ["project", "projects", "sprint", "milestone"]):
                    inferred = "ProjectManagement.project"
                elif any(x in msg for x in ["staff", "employee", "hr", "personnel"]):
                    inferred = "Staff.staff"

                if inferred:
                    collection = inferred
                    entities["collections"] = [inferred]

            # Determine operation type from intent
            if intent == "db.find":
                yield {"type": "mongodb_started", "content": f"Searching documents in collection '{collection}'...", "operation": "find"}

                # Build filter from entities
                filter_query = params.get("filter", {})
                if entities.get("queries"):
                    # Simple query parsing - could be enhanced
                    filter_query.update({"$text": {"$search": " ".join(entities["queries"])}})

                async for result in self.mcp_client.find_documents(
                    collection=collection,
                    filter_query=filter_query,
                    limit=params.get("limit", 10),
                    user_id=user_id
                ):
                    yield result

            elif intent == "db.aggregate":
                yield {"type": "mongodb_started", "content": f"Running aggregation on collection '{collection}'...", "operation": "aggregate"}

                # Build aggregation pipeline
                pipeline = params.get("pipeline", [])

                # Add basic aggregation if none specified
                if not pipeline and entities.get("queries"):
                    pipeline = [
                        {"$match": {"$text": {"$search": " ".join(entities["queries"])}}},
                        {"$limit": 10}
                    ]

                async for result in self.mcp_client.aggregate_documents(
                    collection=collection,
                    pipeline=pipeline,
                    limit=params.get("limit", 50),
                    user_id=user_id
                ):
                    yield result

            elif intent == "db.vectorSearch":
                if not settings.mongodb_vector_search_enabled:
                    yield {"type": "error", "content": "Vector search is not enabled in configuration"}
                    return

                yield {"type": "mongodb_started", "content": f"Performing vector search in collection '{collection}'...", "operation": "vectorSearch"}

                # For vector search, we'd need to generate embeddings from the query
                # This is a simplified example - in practice, you'd use an embedding service
                query_text = " ".join(entities.get("queries", []))
                if not query_text:
                    query_text = conversation.messages[-1].content if conversation.messages else ""

                # Placeholder for embedding generation
                # In practice, you'd call your embedding service here
                query_vector = [0.1] * settings.mongodb_vector_dimensions  # Placeholder

                # For vector search, we'd need to implement this properly in MCP
                # For now, use aggregate with vector search pipeline
                vector_pipeline = [
                    {
                        "$vectorSearch": {
                            "index": settings.mongodb_vector_index_name,
                            "path": settings.mongodb_vector_path,
                            "queryVector": query_vector,
                            "numCandidates": 200,
                            "limit": params.get("limit", 5)
                        }
                    },
                    {
                        "$project": {
                            "_id": 1,
                            "title": 1,
                            "content": 1,
                            "score": {"$meta": "vectorSearchScore"}
                        }
                    }
                ]

                async for result in self.mcp_client.call_tool(
                    tool_name="mongodb.aggregate",
                    arguments={
                        "collection": collection,
                        "pipeline": vector_pipeline
                    },
                    user_id=user_id
                ):
                    yield result

            elif intent == "db.runCommand":
                yield {"type": "mongodb_started", "content": "Running MongoDB command...", "operation": "runCommand"}

                command = params.get("command", {})
                async for result in self.mcp_client.call_tool(
                    tool_name="mongodb.runCommand",
                    arguments={
                        "command": command
                    },
                    user_id=user_id
                ):
                    yield result

            else:
                yield {
                    "type": "mongodb_info",
                    "content": f"I can help you query the database. I detected intent: {intent}. Available operations: find, aggregate, vector search, run command. What would you like to do?"
                }

            # Add sources used information
            if collection:
                yield {
                    "type": "sources.used",
                    "sources": [{"type": "mongodb", "collection": collection, "database": settings.mongodb_database}]
                }

        except Exception as e:
            logger.error("MongoDB action failed", error=str(e), intent=intent)
            yield {"type": "error", "content": f"Database operation failed: {str(e)}", "operation": intent}

        # Always provide feedback that MCP was attempted
        yield {
            "type": "mongodb_attempted",
            "content": f"I attempted to access the database collection '{collection}' via MCP. If you see 'no access to data' messages, please check that the MCP server is running and properly configured.",
            "collection": collection,
            "intent": intent,
            "available_collections": settings.mongodb_allowed_collections
        }
    
    async def _handle_report_action(
        self,
        routing: Dict[str, Any],
        conversation: Conversation,
        user_id: Optional[str]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle report generation requests, including CRM reports"""

        # Check if this is a CRM report request
        message_text = ""
        if conversation.messages:
            message_text = conversation.messages[-1].content.lower()

        is_crm_report = any(word in message_text for word in ["crm", "customer", "lead", "sales", "account", "deal"])

        if is_crm_report:
            # Generate CRM report
            yield {"type": "report_generation_started", "content": "Generating CRM report with real data..."}
            crm_report = await self._generate_crm_report(conversation, user_id)
            yield {"type": "report_ready", "content": "CRM report generated successfully!", "report": crm_report, "format_options": ["PDF", "Word", "Markdown", "CSV"]}
        else:
            # Generate conversation summary report
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
        
        # Profile facts (always retrieve if user_id) - these persist across ALL conversations
        profile: List[Dict[str, Any]] = []
        if user_id:
            profile = await self.memory_manager.get_profile(user_id)
        context["profile"] = profile

        # Search both user and conversation memories if enabled
        if request.use_long_term_memory and user_id:
            # CRITICAL: Search ALL user memories (cross-conversation) + current conversation memories
            memories = await self.memory_manager.search_memory(
                query=request.message,
                conversation_id=conversation.conversation_id,
                user_id=user_id,  # This ensures we get ALL memories for this user
                limit=20,  # Increased from 10 to get more cross-conversation facts
                search_scope="both"  # Search BOTH user (cross-conversation) and conversation-specific
            )
            context["relevant_memories"] = memories
            
            # Separate memories by level for better context
            context["user_memories"] = [m for m in memories if m.get("memory_level") == "user"]
            context["conversation_memories"] = [m for m in memories if m.get("memory_level") == "conversation"]

            # Composite ranking of memories
            now = datetime.now(timezone.utc)
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

    async def _prepare_research_context(
        self,
        conversation: Conversation,
        message: str,
        user_id: Optional[str],
        base_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare specialized context for research operations"""
        research_context = base_context.copy()

        # Extract research-relevant information from user profile
        if base_context.get("profile"):
            profile_facts = base_context["profile"]

            # Look for business/research preferences
            research_preferences = {
                "industries": [],
                "geographies": [],
                "research_topics": [],
                "business_focus": []
            }

            for fact in profile_facts:
                fact_text = str(fact).lower()

                # Extract industry preferences
                if any(word in fact_text for word in ["industry", "sector", "business"]):
                    research_preferences["industries"].append(str(fact))

                # Extract geography preferences
                if any(word in fact_text for word in ["location", "country", "region", "market"]):
                    research_preferences["geographies"].append(str(fact))

                # Extract research topics
                if any(word in fact_text for word in ["research", "study", "analysis", "investigation"]):
                    research_preferences["research_topics"].append(str(fact))

                # Extract business focus
                if any(word in fact_text for word in ["startup", "enterprise", "consulting", "strategy"]):
                    research_preferences["business_focus"].append(str(fact))

            research_context["research_preferences"] = research_preferences

        # Extract conversation topics and entities relevant to research
        conversation_metadata = base_context.get("conversation_metadata", {})
        research_context["research_entities"] = {
            "mentioned_companies": [],
            "mentioned_industries": [],
            "research_queries": [],
            "key_topics": conversation_metadata.get("topics", [])
        }

        # Analyze recent messages for research patterns
        recent_messages = base_context.get("recent_messages", [])
        for msg in recent_messages[-10:]:  # Last 10 messages
            content = msg.get("content", "").lower()

            # Look for company mentions
            company_indicators = ["company", "corporation", "startup", "business", "organization"]
            if any(indicator in content for indicator in company_indicators):
                research_context["research_entities"]["mentioned_companies"].append(msg["content"])

            # Look for industry mentions
            industry_keywords = ["industry", "sector", "market", "technology", "finance", "healthcare"]
            if any(keyword in content for keyword in industry_keywords):
                research_context["research_entities"]["mentioned_industries"].append(msg["content"])

            # Look for previous research queries
            research_keywords = ["research", "find", "analyze", "investigate", "study"]
            if any(keyword in content for keyword in research_keywords):
                research_context["research_entities"]["research_queries"].append(msg["content"])

        # Add research history from memory
        if user_id:
            research_history = await self.memory_manager.search_memory(
                query="research OR analysis OR investigation OR study",
                user_id=user_id,
                limit=10,
                search_scope="user"
            )
            research_context["research_history"] = research_history

        # Determine research scope and parameters based on message analysis
        research_context["inferred_parameters"] = self._infer_research_parameters(message, research_context)

        return research_context

    def _infer_research_parameters(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Infer research parameters from message and context"""
        message_lower = message.lower()
        params = {
            "max_sources": 15,
            "scope": [],
            "industry": None,
            "geography": None
        }

        # Determine research depth based on source count
        if any(word in message_lower for word in ["comprehensive", "detailed", "thorough", "extensive", "in-depth"]):
            params["max_sources"] = 25
        elif any(word in message_lower for word in ["quick", "brief", "overview", "summary"]):
            params["max_sources"] = 8

        # Infer industry from message or context
        industries = ["technology", "finance", "healthcare", "retail", "manufacturing",
                     "energy", "education", "real estate", "automotive", "food"]

        for industry in industries:
            if industry in message_lower:
                params["industry"] = industry
                break

        # Use industry from research preferences if not found in message
        if not params["industry"] and context.get("research_preferences", {}).get("industries"):
            # Extract industry from preferences (simplified)
            pref_text = " ".join(context["research_preferences"]["industries"]).lower()
            for industry in industries:
                if industry in pref_text:
                    params["industry"] = industry
                    break

        # Infer geography
        geographies = ["united states", "europe", "asia", "china", "india", "germany",
                      "japan", "uk", "canada", "australia"]

        for geo in geographies:
            if geo in message_lower:
                params["geography"] = geo
                break

        # Determine research scope
        scope_keywords = {
            "market": ["market", "demand", "competition", "pricing"],
            "competitors": ["competitor", "competition", "rival", "versus"],
            "technology": ["technology", "tech", "innovation", "software"],
            "customer": ["customer", "user", "consumer", "audience"],
            "pricing": ["pricing", "cost", "revenue", "profit"],
            "regulatory": ["regulation", "legal", "compliance", "policy"]
        }

        for scope_name, keywords in scope_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                params["scope"].append(scope_name)

        # Default scope if none detected
        if not params["scope"]:
            params["scope"] = ["market"]

        return params

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
        """Determine if the message needs research with enhanced detection"""
        message_lower = message.lower()

        # High-confidence research indicators
        high_confidence_indicators = [
            "research", "investigate", "analyze", "market analysis",
            "competitor analysis", "industry report", "trends",
            "statistics", "data on", "find information about",
            "what's the latest on", "current status of"
        ]

        # Medium-confidence research indicators
        medium_confidence_indicators = [
            "find out", "look up", "search for", "tell me about",
            "explain how", "how does", "what are the",
            "who are the", "where can I find"
        ]

        # Check for high-confidence indicators
        if any(indicator in message_lower for indicator in high_confidence_indicators):
            return True

        # Check for medium-confidence indicators
        if any(indicator in message_lower for indicator in medium_confidence_indicators):
            return True

        # Check for questions that need current/fresh information
        if "?" in message:
            temporal_indicators = [
                "latest", "current", "recent", "today", "now",
                "this year", "this month", "recently", "new",
                "updated", "fresh", "breaking"
            ]

            if any(word in message_lower for word in temporal_indicators):
                return True

            # Check for specific business/research questions
            business_questions = [
                "what is", "how much", "how many", "where is",
                "who is", "when was", "why is", "which is"
            ]

            question_words = message_lower.split()
            if any(q in " ".join(question_words[:3]) for q in business_questions):
                return True

        # Check for specific research domains
        research_domains = [
            "market", "industry", "competition", "pricing",
            "strategy", "business", "company", "product",
            "technology", "innovation", "growth", "revenue"
        ]

        domain_count = sum(1 for domain in research_domains if domain in message_lower)
        if domain_count >= 2:  # Multiple research domains suggest research needed
            return True

        # Check for database/data queries that should use MCP
        database_keywords = [
            "database", "data", "collection", "records", "documents",
            "find", "search", "query", "lookup", "retrieve", "fetch",
            "show me", "tell me", "what's in", "list", "display"
        ]

        database_count = sum(1 for keyword in database_keywords if keyword in message_lower)
        if database_count >= 1:  # Any database keyword suggests MCP usage
            return False  # Don't research, use MCP instead

        # Check conversation context for research patterns
        if context.get("profile"):
            # If user has previously asked research questions, be more likely to research
            profile_text = " ".join([str(fact) for fact in context["profile"]])
            if any(indicator in profile_text.lower() for indicator in high_confidence_indicators):
                # Lower threshold for research if user has research history
                if len(message.split()) > 6:  # Longer messages more likely to need research
                    return True

        return False
    
    async def _handle_research_request(
        self,
        conversation: Conversation,
        message: str,
        context: Dict[str, Any],
        user_id: Optional[str]
    ) -> ChatResponse:
        """Handle a research request with enhanced context"""
        try:
            # Prepare specialized research context
            research_context = await self._prepare_research_context(
                conversation, message, user_id, context
            )

            # Create status message
            status_message = ChatMessage(
                conversation_id=conversation.conversation_id,
                role=MessageRole.SYSTEM,
                content="🔍 Initiating research... This may take a moment.",
                message_type=MessageType.STATUS
            )
            conversation.add_message(status_message)

            # Extract inferred parameters
            inferred_params = research_context.get("inferred_parameters", {})

            # Prepare conversation context for research service
            conversation_context = {
                "entities": research_context.get("research_entities", {}),
                "topics": research_context.get("conversation_metadata", {}).get("topics", []),
                "preferences": {
                    "industry": inferred_params.get("industry"),
                    "geography": inferred_params.get("geography")
                },
                "research_history": research_context.get("research_history", [])
            }

            # Execute research using the unified research service
            logger.info("Starting research via unified service", query=message)
            research_brief = await self.research_service.research(
                query=message,
                scope=inferred_params.get("scope", []),
                industry=inferred_params.get("industry"),
                geography=inferred_params.get("geography"),
                max_sources=inferred_params.get("max_sources", 15),
                user_id=user_id,
                conversation_context=conversation_context
            )
            
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
            # Use enhanced error handling with intelligent fallback
            return await self._handle_research_error(
                error=e,
                query=message,
                conversation=conversation,
                user_id=user_id,
                research_context=research_context
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
            print(f"🤖 Making non-stream API call with model: '{settings.groq_model}'")
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
        """Prepare messages for LLM using token-tight scaffold"""
        messages = []
        
        # Use the synthesis module to build the structured prompt
        from app.core.synthesis import ResearchSynthesizer
        synthesizer = ResearchSynthesizer()
        
        # Get intent and entities from context if available
        intent = context.get("intent", "general")
        entities = context.get("entities", {})
        research_notes = context.get("research_notes")
        
        # Build the token-tight prompt scaffold
        system_prompt = synthesizer.build_prompt(
            context=context,
            intent=intent,
            entities=entities,
            research_notes=research_notes
        )
        
        messages.append({"role": "system", "content": system_prompt})
        
        # Add recent messages (last 5)
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
        """Format research results in a conversational, engaging way"""
        # Start with an engaging introduction
        query_keywords = brief.query.lower().split()[:3]
        response = f"I've completed comprehensive research on **{brief.query}**. Here's what I discovered:\n\n"

        # Executive summary with personality
        if brief.executive_summary:
            response += f"**In summary:** {brief.executive_summary}\n\n"
        else:
            response += f"**Here's what stands out:**\n\n"

        # Key findings with better formatting
        if brief.findings:
            response += "**Key Findings:**\n"
            for i, finding in enumerate(brief.findings[:4], 1):
                # Make titles more conversational
                title = finding.title
                if not title.endswith('?') and not title.endswith('.'):
                    title = title.rstrip() + '.'

                response += f"• **{title}** {finding.summary}"

                # Add key insights if available
                if finding.key_insights:
                    insights = finding.key_insights[:2]  # Limit to 2 most important
                    response += f"\n  Key points: {' • '.join(insights)}"

                response += "\n\n"

        # Actionable ideas with RICE scoring
        if brief.ideas:
            response += "**Actionable Recommendations:**\n"
            # Sort by RICE score if available
            sorted_ideas = sorted(
                brief.ideas,
                key=lambda x: x.rice.score if x.rice.score else 0,
                reverse=True
            )

            for i, idea in enumerate(sorted_ideas[:3], 1):
                response += f"• **{idea.idea}**\n"
                response += f"  _{idea.rationale}_\n"

                # Add RICE score if meaningful
                if idea.rice.score and idea.rice.score > 0:
                    response += f"  *Priority score: {idea.rice.score:.1f}*\n"

                response += "\n"

        # Strategic priorities (if available from business research)
        if hasattr(brief, 'strategic_priorities') and brief.strategic_priorities:
            response += "**Strategic Focus Areas:**\n"
            for priority in brief.strategic_priorities[:3]:
                response += f"• {priority}\n"
            response += "\n"

        # Risk assessment (if available)
        if hasattr(brief, 'risk_assessment') and brief.risk_assessment:
            risk_level = brief.risk_assessment.get('overall_risk_level', 'unknown')
            if risk_level != 'unknown':
                risk_emoji = {'low': '🟢', 'medium': '🟡', 'high': '🔴'}.get(risk_level, '⚪')
                response += f"**Risk Assessment:** {risk_emoji} {risk_level.title()} overall risk level\n\n"

        # Sources and confidence with context
        confidence_desc = {
            0.8: "highly confident",
            0.6: "moderately confident",
            0.4: "somewhat confident"
        }.get(round(brief.average_confidence, 1), "preliminary")

        response += f"---\n*This analysis is based on {brief.total_sources} diverse sources. "
        response += f"I'm {confidence_desc} in these findings "
        response += f"(average confidence: {brief.average_confidence:.0%}).*\n\n"

        # Add research brief ID for tracking
        response += f"*Research Brief ID: {brief.brief_id[:8]}...*"

        return response

    async def _handle_research_error(
        self,
        error: Exception,
        query: str,
        conversation: Conversation,
        user_id: Optional[str],
        research_context: Optional[Dict[str, Any]] = None
    ) -> ChatResponse:
        """Handle research errors with intelligent fallback and recovery"""
        logger.error("Research failed", error=str(error), query=query)

        # Categorize the error type
        error_type = self._categorize_research_error(error)

        # Create appropriate fallback response based on error type
        if error_type == "network":
            fallback_response = await self._create_network_error_response(query, research_context)
        elif error_type == "parsing":
            fallback_response = await self._create_parsing_error_response(query)
        elif error_type == "rate_limit":
            fallback_response = await self._create_rate_limit_response(query)
        elif error_type == "service_unavailable":
            fallback_response = await self._create_service_unavailable_response(query)
        else:
            fallback_response = await self._create_generic_error_response(query, error)

        # Create error message
        error_message = ChatMessage(
            conversation_id=conversation.conversation_id,
            role=MessageRole.ASSISTANT,
            content=fallback_response,
            message_type=MessageType.ERROR,
            metadata={"error_type": error_type, "original_error": str(error)}
        )

        conversation.add_message(error_message)
        await self.memory_manager.update_from_message(error_message, user_id)

        return ChatResponse(
            conversation_id=conversation.conversation_id,
            message=error_message,
            research_triggered=True,
            error_occurred=True,
            error_details={"type": error_type, "message": str(error)}
        )

    def _categorize_research_error(self, error: Exception) -> str:
        """Categorize research errors for better handling"""
        error_str = str(error).lower()

        if any(keyword in error_str for keyword in ["timeout", "connection", "network", "unreachable"]):
            return "network"
        elif any(keyword in error_str for keyword in ["parse", "json", "format", "structure"]):
            return "parsing"
        elif any(keyword in error_str for keyword in ["rate limit", "quota", "429"]):
            return "rate_limit"
        elif any(keyword in error_str for keyword in ["service", "unavailable", "503", "502"]):
            return "service_unavailable"
        else:
            return "generic"

    async def _create_network_error_response(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Create response for network-related errors"""
        response = "I encountered a network issue while researching. Let me provide some general insights based on my knowledge:\n\n"

        # Try to provide basic insights from context if available
        if context and context.get("research_entities", {}).get("mentioned_industries"):
            industries = context["research_entities"]["mentioned_industries"][:2]
            response += f"**Quick insights for {', '.join(industries)}:**\n\n"

        response += "🔄 **Try again in a moment** - Network issues are usually temporary.\n"
        response += "💡 **Pro tip:** You can also ask me to research specific aspects or try rephrasing your question."

        return response

    async def _create_parsing_error_response(self, query: str) -> str:
        """Create response for parsing-related errors"""
        return f"I had trouble processing the research results for '{query}'. This sometimes happens with complex queries.\n\n" \
               "**Suggestions:**\n" \
               "• Try breaking down your question into smaller, more specific parts\n" \
               "• Focus on one aspect at a time\n" \
               "• Use simpler language in your query\n\n" \
               "Would you like me to try researching a specific part of your question?"

    async def _create_rate_limit_response(self, query: str) -> str:
        """Create response for rate limit errors"""
        return "The research service is currently busy (rate limit reached). This is common during peak usage.\n\n" \
               "**What you can do:**\n" \
               "• Wait 1-2 minutes and try again\n" \
               "• Try a simpler or more specific query\n" \
               "• Ask about a different topic in the meantime\n\n" \
               "I can still help with general questions while you wait!"

    async def _create_service_unavailable_response(self, query: str) -> str:
        """Create response for service unavailable errors"""
        return "The research service is temporarily unavailable. This doesn't happen often, but it's usually resolved quickly.\n\n" \
               "**In the meantime:**\n" \
               "• I can answer based on my general knowledge\n" \
               "• Try asking about a different topic\n" \
               "• Come back in a few minutes to try your research again\n\n" \
               "Would you like to explore another aspect of your question?"

    async def _create_generic_error_response(self, query: str, error: Exception) -> str:
        """Create response for generic errors"""
        return f"I encountered an unexpected issue while researching '{query}'. Don't worry - this happens sometimes!\n\n" \
               "**What you can do:**\n" \
               "• Try rephrasing your question\n" \
               "• Ask about a specific aspect of the topic\n" \
               "• Check back in a moment and try again\n\n" \
               "I'm here to help however I can in the meantime."

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
            "duration": (datetime.now(timezone.utc) - conversation.created_at).total_seconds(),
            "topics": conversation.context.topics
        }

    async def _generate_crm_report(
        self,
        conversation: Conversation,
        user_id: Optional[str]
    ) -> Dict[str, Any]:
        """Generate a comprehensive CRM report with real data"""

        report_data = {
            "title": "CRM Data Report",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sections": [],
            "summary": "",
            "recommendations": []
        }

        try:
            # Section 1: CRM Leads Overview
            leads_data = []
            leads_count = 0

            async for result in self.mcp_client.find_crm_leads(limit=100, user_id=user_id):
                if result.get("type") == "tool.output.data" and result.get("data"):
                    leads_data.append(result["data"])
                    leads_count += 1

            # Section 2: CRM Statistics
            stats_data = []
            async for result in self.mcp_client.aggregate_crm_stats(user_id=user_id):
                if result.get("type") == "tool.output.data" and result.get("data"):
                    stats_data.append(result["data"])

            # Section 3: CRM Accounts (if available)
            accounts_data = []
            accounts_count = 0

            async for result in self.mcp_client.find_crm_accounts(limit=50, user_id=user_id):
                if result.get("type") == "tool.output.data" and result.get("data"):
                    accounts_data.append(result["data"])
                    accounts_count += 1

            # Build report sections
            report_data["sections"] = [
                {
                    "title": "CRM Leads Overview",
                    "type": "leads",
                    "count": leads_count,
                    "data": leads_data[:10],  # Show first 10 leads
                    "description": f"Found {leads_count} CRM leads in total"
                },
                {
                    "title": "CRM Statistics",
                    "type": "statistics",
                    "data": stats_data,
                    "description": "Lead distribution by stage and status"
                },
                {
                    "title": "CRM Accounts",
                    "type": "accounts",
                    "count": accounts_count,
                    "data": accounts_data[:10],  # Show first 10 accounts
                    "description": f"Found {accounts_count} CRM accounts"
                }
            ]

            # Generate AI summary
            stats_summary = ""
            if stats_data:
                stats_text = "\n".join([f"- {stat.get('_id', 'Unknown')}: {stat.get('count', 0)} leads" for stat in stats_data])
                stats_summary = f"Lead distribution:\n{stats_text}"

            summary_prompt = f"""
            Generate a concise executive summary for this CRM report:

            Total Leads: {leads_count}
            Total Accounts: {accounts_count}
            Statistics: {stats_summary}

            Focus on key insights and actionable recommendations.
            """

            summary_response = await self.groq_client.chat.completions.create(
                model=settings.groq_model,
                messages=[
                    {"role": "system", "content": "Generate a concise CRM report summary"},
                    {"role": "user", "content": summary_prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )

            report_data["summary"] = summary_response.choices[0].message.content

            # Generate recommendations
            rec_prompt = f"""
            Based on this CRM data, provide 3-5 actionable recommendations:

            Leads: {leads_count}
            Accounts: {accounts_count}
            Statistics: {stats_summary}
            """

            rec_response = await self.groq_client.chat.completions.create(
                model=settings.groq_model,
                messages=[
                    {"role": "system", "content": "Generate CRM recommendations"},
                    {"role": "user", "content": rec_prompt}
                ],
                temperature=0.4,
                max_tokens=200
            )

            recommendations_text = rec_response.choices[0].message.content
            report_data["recommendations"] = [rec.strip() for rec in recommendations_text.split('\n') if rec.strip()]

        except Exception as e:
            logger.error("Failed to generate CRM report", error=str(e))
            report_data["error"] = f"Failed to generate CRM report: {str(e)}"
            report_data["summary"] = "Unable to generate CRM report due to data access issues."

        return report_data
    
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
    
    def _determine_routing_strategy(
        self,
        intent: str,
        confidence: float,
        entities: Dict[str, Any],
        request: ChatRequest,
        context: Dict[str, Any]
    ) -> str:
        """Determine routing strategy based on intent and context"""
        # MongoDB database intents
        if intent.startswith("db."):
            return "mongodb-first"

        # CRM and PMS actions
        if intent in ["crm_action", "pms_action"]:
            return "mongodb-first"

        # Check for database/data keywords in the message
        message_lower = request.message.lower()
        database_keywords = [
            "database", "data", "collection", "records", "documents",
            "find", "search", "query", "lookup", "retrieve", "fetch",
            "show me", "tell me", "what's in", "list", "display",
            "get", "check", "see", "view", "access", "connect to"
        ]

        if any(keyword in message_lower for keyword in database_keywords):
            # If it looks like a database query but wasn't detected as such,
            # route it to MCP with a general db.query intent
            if not intent.startswith("db."):
                return "mongodb-first"
            return "mongodb-first"

        # Explicit research request
        if request.use_web_search or intent == "research":
            return "research-first"

        # Profile update intent
        if intent == "profile_update":
            return "profile-update"

        # Check for research indicators in message
        research_keywords = [
            "research", "find out", "look up", "search for",
            "what is", "how does", "analyze", "investigate",
            "tell me about", "explain", "market analysis",
            "competitor", "pricing", "trends", "statistics"
        ]
        if any(keyword in message_lower for keyword in research_keywords):
            return "research-first"

        # Default to direct answer
        return "direct-answer"
    
    async def _stream_research(self, research_request: ResearchRequest) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream research progress updates"""
        # This is a placeholder - actual implementation would stream from research_engine
        yield {"status": "searching", "sources_found": 0}
        await asyncio.sleep(0.5)
        yield {"status": "analyzing", "sources_found": 5}
        await asyncio.sleep(0.5)
        yield {"status": "synthesizing", "sources_found": 10}
    
    async def _extract_high_signal_facts(
        self,
        content: str,
        entities: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Extract high-signal facts worth persisting to long-term memory"""
        facts = []
        
        # Look for explicit facts patterns
        fact_patterns = [
            r"I am (.+)",
            r"My (.+) is (.+)",
            r"I prefer (.+)",
            r"I work at (.+)",
            r"I live in (.+)",
            r"My goal is (.+)"
        ]
        
        content_lower = content.lower()
        for pattern in fact_patterns:
            import re
            matches = re.findall(pattern, content_lower)
            for match in matches:
                facts.append({"type": "extracted", "content": match})
        
        # Add entity-based facts
        if entities.get("profile_facts"):
            for fact in entities["profile_facts"]:
                facts.append({"type": "profile", "key": fact["key"], "value": fact["value"]})
        
        return facts
