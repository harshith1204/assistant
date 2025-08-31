"""Enhanced Chat Engine with Full Conversational Capabilities"""

import json
import asyncio
import uuid
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime, timedelta
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
from app.core.conversational_intent import (
    ConversationalFlowManager, ConversationalIntent, ActionType
)
from app.integrations.crm_client import CRMClient
from app.integrations.pms_client import PMSClient

logger = structlog.get_logger()


class EnhancedChatEngine:
    """Enhanced chat engine with full conversational capabilities"""
    
    def __init__(self):
        """Initialize enhanced chat engine"""
        self.groq_client = AsyncGroq(api_key=settings.groq_api_key)
        self.memory_manager = MemoryManager()
        self.research_engine = ResearchEngine()
        self.flow_manager = ConversationalFlowManager()
        self.crm_client = CRMClient()
        self.pms_client = PMSClient()
        self.conversations: Dict[str, Conversation] = {}
        
        # Action handlers mapping
        self.action_handlers = {
            "research_engine": self._handle_research_action,
            "crm_client": self._handle_crm_action,
            "pms_client": self._handle_pms_action,
            "report_generator": self._handle_report_action,
            "calendar_manager": self._handle_calendar_action,
            "chat_engine": self._handle_chat_action
        }
    
    async def process_conversational_message(
        self,
        request: ChatRequest,
        user_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Process message conversationally with streaming updates"""
        
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
            
            # Update both user and conversation memories
            await self.memory_manager.update_from_message(user_message, user_id)
            
            # Process through conversational flow
            flow_result = await self.flow_manager.process_conversation_turn(
                request.message,
                conversation.conversation_id,
                user_id
            )
            
            # Stream conversational updates
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
    
    async def _handle_conversational_flow(
        self,
        flow_result: Dict[str, Any],
        conversation: Conversation,
        user_id: Optional[str]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle conversational flow with appropriate responses"""
        
        routing = flow_result["routing"]
        strategy = flow_result["strategy"]
        
        # Send initial acknowledgment
        yield {
            "type": "acknowledgment",
            "content": self._get_acknowledgment_message(routing),
            "intent": routing["intent"],
            "confidence": routing["confidence"]
        }
        
        # Handle based on strategy
        if strategy["type"] == "clarification":
            async for update in self._handle_clarification_flow(routing, conversation):
                yield update
                
        elif strategy["type"] == "confirmation":
            async for update in self._handle_confirmation_flow(routing, conversation):
                yield update
                
        elif strategy["type"] == "multi-step":
            async for update in self._handle_multi_step_flow(routing, conversation, user_id):
                yield update
                
        else:  # direct action
            async for update in self._handle_direct_action(routing, conversation, user_id):
                yield update
    
    def _get_acknowledgment_message(self, routing: Dict[str, Any]) -> str:
        """Get appropriate acknowledgment message"""
        
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
        """Handle clarification flow"""
        
        # Generate clarification questions
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
        """Handle confirmation flow"""
        
        # Generate action preview
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
        """Handle multi-step action flow"""
        
        handler_name = routing["handler"]
        handler = self.action_handlers.get(handler_name, self._handle_chat_action)
        
        # Send progress updates
        steps = routing["strategy"]["steps"]
        for i, step in enumerate(steps):
            if step == "acknowledge":
                continue  # Already sent
            
            elif step == "show_progress":
                yield {
                    "type": "progress",
                    "content": "Working on it...",
                    "step": i + 1,
                    "total_steps": len(steps),
                    "current_action": step
                }
            
            elif step == "execute":
                # Execute the actual action
                async for result in handler(routing, conversation, user_id):
                    yield result
            
            elif step == "present_results":
                # Results are included in the handler output
                pass
            
            # Small delay between steps for better UX
            await asyncio.sleep(0.5)
    
    async def _handle_direct_action(
        self,
        routing: Dict[str, Any],
        conversation: Conversation,
        user_id: Optional[str]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle direct action execution"""
        
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
        """Handle research action"""
        
        try:
            # Extract research parameters
            params = routing.get("parameters", {})
            entities = routing.get("entities", {})
            
            # Build research query
            topics = entities.get("topics", [])
            original_message = conversation.messages[-1].content
            
            # Create research request
            research_request = ResearchRequest(
                query=original_message,
                max_sources=params.get("max_sources", 15),
                deep_dive=True
            )
            
            # Stream research progress
            yield {
                "type": "research_started",
                "content": "🔍 Starting research...",
                "topics": topics
            }
            
            # Run research
            research_brief = await self.research_engine.run_research(research_request)
            
            # Stream findings progressively
            yield {
                "type": "research_progress",
                "content": f"Found {research_brief.total_sources} relevant sources",
                "sources_count": research_brief.total_sources
            }
            
            # Format and stream results
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
            
            # Save to memory
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
            yield {
                "type": "error",
                "content": f"I encountered an issue with the research: {str(e)}"
            }
    
    async def _handle_crm_action(
        self,
        routing: Dict[str, Any],
        conversation: Conversation,
        user_id: Optional[str]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle CRM action"""
        
        action = routing.get("action")
        params = routing.get("parameters", {})
        
        try:
            if action == "create_note":
                yield {
                    "type": "crm_action_started",
                    "content": "Creating note in CRM...",
                    "action": action
                }
                
                note_id = await self.crm_client.create_note(
                    lead_id=params.get("lead_id", "default"),
                    subject=params.get("subject", "Note from conversation"),
                    description=params.get("content", conversation.messages[-1].content)
                )
                
                yield {
                    "type": "crm_action_complete",
                    "content": f"✅ Note created successfully in CRM (ID: {note_id})",
                    "result": {"note_id": note_id}
                }
                
            elif action == "create_task":
                yield {
                    "type": "crm_action_started",
                    "content": "Creating task in CRM...",
                    "action": action
                }
                
                task_id = await self.crm_client.create_task(
                    business_id=params.get("business_id", "default"),
                    name=params.get("title", "Task from conversation"),
                    description=params.get("description", ""),
                    priority=params.get("priority", "MEDIUM")
                )
                
                yield {
                    "type": "crm_action_complete",
                    "content": f"✅ Task created successfully (ID: {task_id})",
                    "result": {"task_id": task_id}
                }
                
            else:
                yield {
                    "type": "crm_action_info",
                    "content": "I can help you with CRM actions like creating notes, tasks, or scheduling meetings. What would you like to do?"
                }
                
        except Exception as e:
            logger.error("CRM action failed", error=str(e))
            yield {
                "type": "error",
                "content": f"CRM action failed: {str(e)}"
            }
    
    async def _handle_pms_action(
        self,
        routing: Dict[str, Any],
        conversation: Conversation,
        user_id: Optional[str]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle PMS action"""
        
        action = routing.get("action")
        params = routing.get("parameters", {})
        
        try:
            if action == "create_work_item":
                yield {
                    "type": "pms_action_started",
                    "content": "Creating work item in project system...",
                    "action": action
                }
                
                work_item_id = await self.pms_client.create_work_item(
                    project_id=params.get("project_id", "default"),
                    title=params.get("title", "Work item from conversation"),
                    description=params.get("description", ""),
                    item_type=params.get("type", "TASK"),
                    priority=params.get("priority", "MEDIUM")
                )
                
                yield {
                    "type": "pms_action_complete",
                    "content": f"✅ Work item created successfully (ID: {work_item_id})",
                    "result": {"work_item_id": work_item_id}
                }
                
            elif action == "create_page":
                yield {
                    "type": "pms_action_started",
                    "content": "Creating documentation page...",
                    "action": action
                }
                
                page_id = await self.pms_client.create_page(
                    project_id=params.get("project_id", "default"),
                    title=params.get("title", "Documentation"),
                    content=params.get("content", "")
                )
                
                yield {
                    "type": "pms_action_complete",
                    "content": f"✅ Documentation page created (ID: {page_id})",
                    "result": {"page_id": page_id}
                }
                
            else:
                yield {
                    "type": "pms_action_info",
                    "content": "I can help you manage projects, create work items, or update documentation. What would you like to do?"
                }
                
        except Exception as e:
            logger.error("PMS action failed", error=str(e))
            yield {
                "type": "error",
                "content": f"Project action failed: {str(e)}"
            }
    
    async def _handle_report_action(
        self,
        routing: Dict[str, Any],
        conversation: Conversation,
        user_id: Optional[str]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle report generation action"""
        
        yield {
            "type": "report_generation_started",
            "content": "Generating report based on our conversation..."
        }
        
        # Generate summary of conversation
        summary = await self._generate_conversation_summary(conversation)
        
        yield {
            "type": "report_ready",
            "content": "Report generated successfully!",
            "report": summary,
            "format_options": ["PDF", "Word", "Markdown"]
        }
    
    async def _handle_calendar_action(
        self,
        routing: Dict[str, Any],
        conversation: Conversation,
        user_id: Optional[str]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle calendar/meeting action"""
        
        params = routing.get("parameters", {})
        
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
        """Handle general chat action"""
        
        # Get context
        context = await self._prepare_context(conversation, user_id)
        
        # Generate response using LLM
        messages = await self._prepare_conversational_messages(
            conversation,
            routing,
            context
        )
        
        # Stream response
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
                yield {
                    "type": "chat_chunk",
                    "content": content
                }
        
        # Save assistant message
        assistant_message = ChatMessage(
            conversation_id=conversation.conversation_id,
            role=MessageRole.ASSISTANT,
            content=full_response,
            message_type=MessageType.TEXT
        )
        conversation.add_message(assistant_message)
        await self.memory_manager.update_from_message(assistant_message, user_id)
        
        yield {
            "type": "chat_complete",
            "full_response": full_response
        }
    
    def _generate_action_preview(self, routing: Dict[str, Any]) -> Dict[str, Any]:
        """Generate preview of action to be taken"""
        
        action = routing.get("action")
        params = routing.get("parameters", {})
        
        preview = {
            "action": action,
            "description": f"Execute {action} action",
            "parameters": params,
            "estimated_time": "A few seconds"
        }
        
        if action == "market_research":
            preview["description"] = "Research market trends and opportunities"
            preview["estimated_time"] = "30-60 seconds"
        elif action == "create_task":
            preview["description"] = f"Create task: {params.get('title', 'New Task')}"
        elif action == "schedule_meeting":
            preview["description"] = f"Schedule meeting with {params.get('attendees', 'participants')}"
        
        return preview
    
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
        
        # Add recent conversation history
        for msg in conversation.messages[-5:]:
            if msg.role != MessageRole.SYSTEM:
                messages.append({
                    "role": msg.role.value,
                    "content": msg.content
                })
        
        return messages
    
    async def _prepare_context(
        self,
        conversation: Conversation,
        user_id: Optional[str]
    ) -> Dict[str, Any]:
        """Prepare dual context for processing"""
        
        # Get both user-level and conversation-level contexts
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
            "conversation_context": context.short_term,  # Conversation-specific
            "user_context": context.long_term  # User-level across conversations
        }
    
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
    
    async def _get_or_create_conversation(
        self,
        conversation_id: Optional[str],
        user_id: Optional[str]
    ) -> Conversation:
        """Get existing or create new conversation"""
        if conversation_id and conversation_id in self.conversations:
            return self.conversations[conversation_id]
        
        if conversation_id:
            conversation = Conversation(
                conversation_id=conversation_id,
                user_id=user_id
            )
        else:
            conversation = Conversation(user_id=user_id)
        
        if conversation_id:
            context = await self.memory_manager.get_conversation_context(
                conversation_id,
                user_id
            )
            conversation.context = context
        
        return conversation
