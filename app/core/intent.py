"""Conversational Intent Detection and Routing System"""

import re
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from groq import AsyncGroq
import json
import structlog
from datetime import datetime, timezone

from app.config import settings

logger = structlog.get_logger()


class ConversationalIntent(Enum):
    """Types of conversational intents"""
    GREETING = "greeting"
    RESEARCH = "research"
    CRM_ACTION = "crm_action"
    PMS_ACTION = "pms_action"
    REPORT_GENERATION = "report_generation"
    DATA_QUERY = "data_query"
    TASK_MANAGEMENT = "task_management"
    MEETING_SCHEDULING = "meeting_scheduling"
    GENERAL_CHAT = "general_chat"
    CLARIFICATION = "clarification"
    FOLLOW_UP = "follow_up"
    CONFIRMATION = "confirmation"
    CANCELLATION = "cancellation"


class ActionType(Enum):
    """Specific action types within intents"""
    # Research actions
    MARKET_RESEARCH = "market_research"
    COMPETITOR_ANALYSIS = "competitor_analysis"
    PRICING_RESEARCH = "pricing_research"
    CUSTOMER_RESEARCH = "customer_research"
    
    # CRM actions
    CREATE_NOTE = "create_note"
    UPDATE_LEAD = "update_lead"
    CREATE_TASK = "create_task"
    SCHEDULE_MEETING = "schedule_meeting"
    VIEW_LEAD = "view_lead"
    
    # PMS actions
    CREATE_PAGE = "create_page"
    CREATE_WORK_ITEM = "create_work_item"
    UPDATE_WORK_ITEM = "update_work_item"
    ADD_COMMENT = "add_comment"
    
    # Report actions
    GENERATE_SUMMARY = "generate_summary"
    CREATE_PRESENTATION = "create_presentation"
    EXPORT_DATA = "export_data"
    
    # General actions
    CHAT = "chat"
    HELP = "help"
    STATUS = "status"


class ConversationalIntentDetector:
    """Detect intent and extract entities from conversational messages"""
    
    def __init__(self):
        self.client = AsyncGroq(api_key=settings.groq_api_key)
        self.model = settings.llm_model
        
        # Intent patterns for quick detection
        self.intent_patterns = {
            ConversationalIntent.GREETING: [
                r'\b(hi|hello|hey|good morning|good afternoon|good evening)\b',
                r'\b(how are you|what\'s up|sup)\b'
            ],
            ConversationalIntent.RESEARCH: [
                r'\b(research|find out|look up|search|analyze|investigate)\b',
                r'\b(what is|tell me about|explain|how does)\b',
                r'\b(market|competitor|pricing|customer|trend)\b'
            ],
            ConversationalIntent.CRM_ACTION: [
                r'\b(crm|lead|contact|customer|client|deal)\b',
                r'\b(create note|update lead|add task|schedule meeting)\b',
                r'\b(follow up|reminder|call|email)\b'
            ],
            ConversationalIntent.PMS_ACTION: [
                r'\b(project|task|work item|sprint|milestone)\b',
                r'\b(create page|documentation|wiki)\b',
                r'\b(assign|update status|comment)\b'
            ],
            ConversationalIntent.REPORT_GENERATION: [
                r'\b(report|summary|presentation|document)\b',
                r'\b(generate|create|prepare|compile)\b',
                r'\b(export|download|share)\b'
            ],
            ConversationalIntent.MEETING_SCHEDULING: [
                r'\b(meeting|appointment|call|demo|discussion)\b',
                r'\b(schedule|book|arrange|set up)\b',
                r'\b(calendar|availability|time slot)\b'
            ]
        }
    
    async def detect_intent(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Detect intent from conversational message"""
        
        # Quick pattern-based detection
        quick_intent = self._quick_intent_detection(message)
        
        # LLM-based deep understanding
        deep_analysis = await self._deep_intent_analysis(message, context)
        
        # Combine results
        return self._combine_analyses(quick_intent, deep_analysis, message)
    
    def _quick_intent_detection(self, message: str) -> Optional[ConversationalIntent]:
        """Quick pattern-based intent detection"""
        message_lower = message.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower, re.IGNORECASE):
                    return intent
        
        return None
    
    async def _deep_intent_analysis(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Deep intent analysis using LLM"""
        
        system_prompt = """You are a conversational intent analyzer. Analyze the user's message and determine:
1. Primary intent (research, crm_action, pms_action, report_generation, meeting_scheduling, general_chat)
2. Specific action needed
3. Entities mentioned (names, dates, topics, etc.)
4. Parameters for the action
5. Confidence level (0-1)
6. Whether clarification is needed

Consider the conversation context if provided.

Return ONLY valid JSON:
{
    "intent": "primary_intent",
    "action": "specific_action",
    "entities": {
        "topics": [],
        "people": [],
        "organizations": [],
        "dates": [],
        "locations": [],
        "other": {}
    },
    "parameters": {
        // action-specific parameters
    },
    "confidence": 0.0-1.0,
    "needs_clarification": false,
    "clarification_questions": [],
    "suggested_response": "natural response text"
}"""
        
        user_prompt = f"Message: {message}"
        if context:
            user_prompt += f"\n\nContext: {json.dumps(context, indent=2)}"
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            # Check if response has content
            if not response.choices or not response.choices[0].message.content:
                logger.warning("Empty response from LLM for intent analysis")
                return {
                    "intent": "general_chat",
                    "confidence": 0.5,
                    "needs_clarification": True
                }
            
            # Parse JSON response
            content = response.choices[0].message.content.strip()
            
            # Try to extract JSON if there's extra text
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group()
            
            result = json.loads(content)
            return result
            
        except json.JSONDecodeError as e:
            logger.error("Failed to parse intent JSON", error=str(e), response_content=content if 'content' in locals() else None)
            return {
                "intent": "general_chat",
                "confidence": 0.5,
                "needs_clarification": True
            }
        except Exception as e:
            logger.error("Failed to analyze intent", error=str(e))
            return {
                "intent": "general_chat",
                "confidence": 0.5,
                "needs_clarification": True
            }
    
    def _combine_analyses(
        self,
        quick_intent: Optional[ConversationalIntent],
        deep_analysis: Dict[str, Any],
        message: str
    ) -> Dict[str, Any]:
        """Combine quick and deep analyses"""
        
        # Start with deep analysis
        result = deep_analysis.copy()
        
        # Validate and adjust with quick detection
        if quick_intent:
            quick_intent_str = quick_intent.value
            if result.get("confidence", 0) < 0.7:
                # Use quick detection if LLM confidence is low
                result["intent"] = quick_intent_str
                result["confidence"] = max(0.6, result.get("confidence", 0))
        
        # Add message metadata
        result["original_message"] = message
        result["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        return result
    
    async def extract_action_parameters(
        self,
        message: str,
        intent: str,
        action: str
    ) -> Dict[str, Any]:
        """Extract specific parameters for an action"""
        
        parameter_prompts = {
            "market_research": """Extract: industry, geography, timeframe, specific_questions""",
            "create_note": """Extract: lead_id, subject, content, attachments""",
            "create_task": """Extract: title, description, assignee, due_date, priority""",
            "schedule_meeting": """Extract: attendees, date, time, duration, agenda, type (virtual/in-person)""",
            "create_work_item": """Extract: title, description, type, priority, assignee, tags"""
        }
        
        prompt = parameter_prompts.get(action, "Extract all relevant parameters")
        
        system_prompt = f"""Extract action parameters from the message.
{prompt}

Return ONLY valid JSON with the extracted parameters."""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                temperature=0.2,
                max_tokens=500
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error("Failed to extract parameters", error=str(e))
            return {}


class ConversationalRouter:
    """Route conversations to appropriate handlers based on intent"""
    
    def __init__(self):
        self.intent_detector = ConversationalIntentDetector()
        self.active_contexts: Dict[str, Dict[str, Any]] = {}
    
    async def route_message(
        self,
        message: str,
        conversation_id: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Route message to appropriate handler"""
        
        # Get conversation context
        context = self.active_contexts.get(conversation_id, {})
        
        # Detect intent
        intent_result = await self.intent_detector.detect_intent(message, context)
        
        # Update context
        self._update_context(conversation_id, intent_result)
        
        # Determine routing
        routing = {
            "intent": intent_result["intent"],
            "action": intent_result.get("action"),
            "confidence": intent_result.get("confidence", 0),
            "needs_clarification": intent_result.get("needs_clarification", False),
            "parameters": intent_result.get("parameters", {}),
            "entities": intent_result.get("entities", {}),
            "suggested_response": intent_result.get("suggested_response"),
            "handler": self._get_handler(intent_result["intent"]),
            "requires_confirmation": self._requires_confirmation(intent_result)
        }
        
        return routing
    
    def _update_context(self, conversation_id: str, intent_result: Dict[str, Any]):
        """Update conversation context"""
        if conversation_id not in self.active_contexts:
            self.active_contexts[conversation_id] = {
                "history": [],
                "entities": {},
                "current_task": None
            }
        
        context = self.active_contexts[conversation_id]
        context["history"].append(intent_result)
        
        # Update entities
        if "entities" in intent_result:
            for key, value in intent_result["entities"].items():
                if key not in context["entities"]:
                    context["entities"][key] = []
                if isinstance(value, list):
                    context["entities"][key].extend(value)
                else:
                    context["entities"][key].append(value)
        
        # Track current task
        if intent_result.get("action"):
            context["current_task"] = {
                "action": intent_result["action"],
                "started_at": datetime.now(timezone.utc).isoformat(),
                "parameters": intent_result.get("parameters", {})
            }
    
    def _get_handler(self, intent: str) -> str:
        """Get handler for intent"""
        handlers = {
            "research": "research_engine",
            "crm_action": "crm_client",
            "pms_action": "pms_client",
            "report_generation": "report_generator",
            "meeting_scheduling": "calendar_manager",
            "general_chat": "chat_engine"
        }
        return handlers.get(intent, "chat_engine")
    
    def _requires_confirmation(self, intent_result: Dict[str, Any]) -> bool:
        """Check if action requires confirmation"""
        critical_actions = [
            "create_task", "schedule_meeting", "update_lead",
            "create_work_item", "export_data"
        ]
        
        action = intent_result.get("action")
        confidence = intent_result.get("confidence", 0)
        
        # Require confirmation for critical actions or low confidence
        return action in critical_actions or confidence < 0.7
    
    def get_context(self, conversation_id: str) -> Dict[str, Any]:
        """Get conversation context"""
        return self.active_contexts.get(conversation_id, {})
    
    def clear_context(self, conversation_id: str):
        """Clear conversation context"""
        if conversation_id in self.active_contexts:
            del self.active_contexts[conversation_id]


class IntentDetector:
    """Simple intent detector for the 5-stage pipeline"""
    
    def __init__(self):
        self.client = AsyncGroq(api_key=settings.groq_api_key)
        self.model = settings.llm_model
    
    async def detect_intent(
        self,
        message: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect intent and return label, confidence, and entities"""
        
        # Quick pattern matching for common intents
        message_lower = message.lower()
        
        # Profile update patterns
        if any(phrase in message_lower for phrase in ["remember that", "save that", "note that", "i am", "i prefer", "my name is"]):
            return {
                "label": "profile_update",
                "confidence": 0.8,
                "entities": self._extract_profile_facts(message),
                "clarification_question": None
            }
        
        # Research patterns
        if any(phrase in message_lower for phrase in ["research", "find out", "look up", "what is", "how does"]):
            return {
                "label": "research",
                "confidence": 0.85,
                "entities": {"topics": self._extract_topics(message)},
                "clarification_question": None
            }
        
        # General chat with lower confidence
        return {
            "label": "general",
            "confidence": 0.6,
            "entities": {},
            "clarification_question": None
        }
    
    def _extract_profile_facts(self, message: str) -> Dict[str, Any]:
        """Extract profile facts from message"""
        facts = []
        
        # Common profile patterns
        import re
        patterns = [
            (r"my name is (\w+)", "name"),
            (r"i prefer (.*?)(?:\.|$)", "preference"),
            (r"i am (.*?)(?:\.|$)", "description"),
            (r"i work at (.*?)(?:\.|$)", "workplace"),
            (r"i live in (.*?)(?:\.|$)", "location")
        ]
        
        for pattern, key in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                facts.append({"key": key, "value": match.group(1).strip(), "priority": 60})
        
        return {"profile_facts": facts} if facts else {}
    
    def _extract_topics(self, message: str) -> List[str]:
        """Extract research topics from message"""
        # Simple topic extraction
        topics = []
        topic_keywords = {
            "market": ["market", "industry", "sector"],
            "competitors": ["competitor", "competition", "rival"],
            "pricing": ["price", "cost", "pricing"],
            "technology": ["tech", "software", "platform"]
        }
        
        message_lower = message.lower()
        for topic, keywords in topic_keywords.items():
            if any(kw in message_lower for kw in keywords):
                topics.append(topic)
        
        return topics

class ConversationalFlowManager:
    """Manage conversational flow and state transitions"""
    
    def __init__(self):
        self.router = ConversationalRouter()
        self.flows: Dict[str, Dict[str, Any]] = {}
    
    async def process_conversation_turn(
        self,
        message: str,
        conversation_id: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a conversation turn"""
        
        # Get or create flow
        flow = self._get_or_create_flow(conversation_id)
        
        # Route message
        routing = await self.router.route_message(message, conversation_id, user_id)
        
        # Update flow state
        flow["current_intent"] = routing["intent"]
        flow["last_routing"] = routing
        flow["turn_count"] += 1
        
        # Determine response strategy
        response_strategy = self._determine_response_strategy(flow, routing)
        
        return {
            "routing": routing,
            "flow": flow,
            "strategy": response_strategy,
            "conversation_id": conversation_id
        }
    
    def _get_or_create_flow(self, conversation_id: str) -> Dict[str, Any]:
        """Get or create conversation flow"""
        if conversation_id not in self.flows:
            self.flows[conversation_id] = {
                "conversation_id": conversation_id,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "turn_count": 0,
                "current_intent": None,
                "pending_actions": [],
                "completed_actions": [],
                "state": "active"
            }
        return self.flows[conversation_id]
    
    def _determine_response_strategy(
        self,
        flow: Dict[str, Any],
        routing: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine response strategy"""
        
        strategy = {
            "type": "direct",  # direct, clarification, confirmation, multi-step
            "steps": [],
            "immediate_action": True
        }
        
        # Need clarification?
        if routing["needs_clarification"]:
            strategy["type"] = "clarification"
            strategy["immediate_action"] = False
            strategy["steps"] = ["ask_clarification", "wait_response", "process_action"]
        
        # Need confirmation?
        elif routing["requires_confirmation"]:
            strategy["type"] = "confirmation"
            strategy["immediate_action"] = False
            strategy["steps"] = ["show_preview", "ask_confirmation", "execute_if_confirmed"]
        
        # Multi-step action?
        elif routing["action"] in ["market_research", "competitor_analysis"]:
            strategy["type"] = "multi-step"
            strategy["steps"] = ["acknowledge", "show_progress", "execute", "present_results"]
        
        return strategy
    
    def add_pending_action(
        self,
        conversation_id: str,
        action: Dict[str, Any]
    ):
        """Add pending action to flow"""
        flow = self._get_or_create_flow(conversation_id)
        flow["pending_actions"].append({
            "action": action,
            "added_at": datetime.now(timezone.utc).isoformat()
        })
    
    def complete_action(
        self,
        conversation_id: str,
        action_id: str,
        result: Any
    ):
        """Mark action as completed"""
        flow = self._get_or_create_flow(conversation_id)
        
        # Move from pending to completed
        pending = flow["pending_actions"]
        for i, item in enumerate(pending):
            if item.get("id") == action_id:
                completed = pending.pop(i)
                completed["completed_at"] = datetime.now(timezone.utc).isoformat()
                completed["result"] = result
                flow["completed_actions"].append(completed)
                break
    
    def get_flow_state(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get current flow state"""
        return self.flows.get(conversation_id)
