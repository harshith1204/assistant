"""Clean Agentic Intent Understanding and Semantic Routing System"""

import json
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass, field
from groq import AsyncGroq
import structlog
from datetime import datetime, timezone
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings

logger = structlog.get_logger()


class ConversationalIntent(Enum):
    """Types of conversational intents with semantic understanding"""
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

    # Database Intents
    DB_FIND = "db.find"
    DB_AGGREGATE = "db.aggregate"
    DB_VECTOR_SEARCH = "db.vectorSearch"
    DB_RUN_COMMAND = "db.runCommand"
    DB_QUERY = "db.query"


@dataclass
class IntentUnderstanding:
    """Comprehensive intent understanding with reasoning"""
    primary_intent: ConversationalIntent
    confidence: float
    reasoning: str
    entities: Dict[str, Any] = field(default_factory=dict)
    context_clues: List[str] = field(default_factory=list)
    follow_up_questions: List[str] = field(default_factory=list)
    complexity_level: str = "simple"  # simple, moderate, complex
    urgency_level: str = "normal"  # low, normal, high, urgent
    domain_expertise: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SemanticContext:
    """Semantic context for understanding user messages"""
    user_message: str
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    user_profile: Dict[str, Any] = field(default_factory=dict)
    domain_knowledge: Dict[str, Any] = field(default_factory=dict)
    current_context: Dict[str, Any] = field(default_factory=dict)
    reasoning_steps: List[str] = field(default_factory=list)


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


class AgenticIntentDetector:
    """Agentic intent detector using semantic understanding and reasoning"""
    
    def __init__(self):
        self.client = AsyncGroq(api_key=settings.groq_api_key)
        self.model = settings.llm_model
        
        # Semantic understanding templates for different intent categories
        self.intent_templates = {
            "greeting": "Simple social interaction, greeting, or checking in",
            "research": "Seeking information, analysis, or investigation about topics, markets, competitors, etc.",
            "crm_action": "Actions related to customer relationship management, leads, contacts, deals",
            "pms_action": "Project management actions, tasks, work items, documentation",
            "report_generation": "Creating reports, summaries, presentations, or documents",
            "data_query": "Querying or retrieving data from databases or systems",
            "task_management": "Managing tasks, todos, schedules, or workflows",
            "meeting_scheduling": "Scheduling meetings, appointments, or calendar events",
            "general_chat": "Casual conversation, general questions, or open-ended discussion",
            "clarification": "Seeking clarification or asking for more details",
            "follow_up": "Following up on previous conversations or actions",
            "confirmation": "Confirming actions, decisions, or information",
            "cancellation": "Cancelling or stopping actions or processes"
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def understand_intent_semantic(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        user_context: Optional[Dict[str, Any]] = None
    ) -> IntentUnderstanding:
        """Understand user intent using semantic analysis and reasoning"""

        # Create semantic context
        context = SemanticContext(
            user_message=user_message,
            conversation_history=conversation_history or [],
            user_profile=user_context or {},
            current_context=self._extract_current_context(conversation_history)
        )

        # Phase 1: Initial semantic analysis
        await self._analyze_semantic_context(context)

        # Phase 2: Intent classification with reasoning
        intent_understanding = await self._classify_intent_with_reasoning(context)

        # Phase 3: Entity extraction and enhancement
        intent_understanding = await self._extract_entities_with_reasoning(intent_understanding, context)

        # Phase 4: Quality assessment and refinement
        intent_understanding = await self._assess_and_refine_understanding(intent_understanding, context)

        return intent_understanding

    async def _analyze_semantic_context(self, context: SemanticContext):
        """Analyze the semantic context of the user message"""

        system_prompt = """You are an expert at analyzing conversational context and user intent.

Analyze the user's message and conversation history to understand:
1. What the user is trying to accomplish
2. The context and background of their request
3. Any implicit requirements or constraints
4. The complexity and urgency of their request

Return ONLY valid JSON:
{
    "semantic_analysis": "Brief analysis of the message meaning and context",
    "key_concepts": ["main concepts or topics mentioned"],
    "communication_style": "formal|casual|technical|business",
    "information_need": "factual|analysis|action|clarification",
    "domain_indicators": ["business", "technical", "personal"],
    "urgency_indicators": ["deadline", "ASAP", "whenever"],
    "relationship_to_previous": "new_topic|follow_up|clarification|correction"
}"""

        # Prepare conversation context
        recent_history = context.conversation_history[-5:] if context.conversation_history else []
        history_summary = "\n".join([
            f"User: {msg.get('user_message', '')}"
            f"Assistant: {msg.get('assistant_response', '')}"[:200]
            for msg in recent_history
        ])

        user_prompt = f"""Analyze this user message in context:

Current Message: {context.user_message}

Recent Conversation History:
{history_summary}

User Profile Context: {json.dumps(context.user_profile)}

Provide semantic analysis:"""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=600
            )

            result = json.loads(response.choices[0].message.content)
            context.reasoning_steps.append(f"Semantic analysis: {result.get('semantic_analysis', '')}")
            context.current_context.update(result)

        except Exception as e:
            logger.error("Failed to analyze semantic context", error=str(e))
            context.reasoning_steps.append("Semantic analysis failed, using basic understanding")

    async def _classify_intent_with_reasoning(self, context: SemanticContext) -> IntentUnderstanding:
        """Classify intent using reasoning rather than pattern matching"""

        system_prompt = """You are an expert intent classifier that uses reasoning and semantic understanding.

Based on the message analysis, classify the user's intent by:
1. Understanding what they're trying to accomplish
2. Considering the context and conversation history
3. Evaluating the complexity and specificity of their request
4. Determining the most appropriate intent category

Intent Categories:
- greeting: Social interaction, checking in
- research: Seeking information or analysis
- crm_action: Customer relationship management tasks
- pms_action: Project management tasks
- report_generation: Creating reports or summaries
- data_query: Database or data retrieval requests
- task_management: Managing tasks or workflows
- meeting_scheduling: Calendar and meeting related
- general_chat: Casual conversation
- clarification: Seeking more details
- follow_up: Following up on previous topics
- confirmation: Confirming actions or information
- cancellation: Stopping or cancelling actions

Return ONLY valid JSON:
{
    "primary_intent": "intent_category",
    "confidence": 0.85,
    "reasoning": "Detailed reasoning for this classification",
    "intent_characteristics": {
        "clarity": "clear|ambiguous|needs_clarification",
        "complexity": "simple|moderate|complex",
        "urgency": "low|normal|high|urgent",
        "action_required": true|false
    },
    "alternative_intents": [
        {"intent": "alternative_intent", "confidence": 0.3, "reason": "why this could be"}
    ]
}"""

        semantic_analysis = context.current_context.get('semantic_analysis', 'No analysis available')
        key_concepts = context.current_context.get('key_concepts', [])
        communication_style = context.current_context.get('communication_style', 'unknown')

        user_prompt = f"""Classify the intent of this user message:

Message: {context.user_message}
Semantic Analysis: {semantic_analysis}
Key Concepts: {', '.join(key_concepts)}
Communication Style: {communication_style}
Information Need: {context.current_context.get('information_need', 'unknown')}

Classify the intent:"""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=800
            )

            result = json.loads(response.choices[0].message.content)

            # Map string intent to enum
            intent_str = result.get('primary_intent', 'general_chat')
            primary_intent = self._map_string_to_intent(intent_str)

            understanding = IntentUnderstanding(
                primary_intent=primary_intent,
                confidence=result.get('confidence', 0.5),
                reasoning=result.get('reasoning', ''),
                complexity_level=result.get('intent_characteristics', {}).get('complexity', 'simple'),
                urgency_level=result.get('intent_characteristics', {}).get('urgency', 'normal')
            )

            # Store additional metadata
            understanding.entities['intent_characteristics'] = result.get('intent_characteristics', {})
            understanding.entities['alternative_intents'] = result.get('alternative_intents', [])

            return understanding
            
        except Exception as e:
            logger.error("Failed to classify intent", error=str(e))
            return IntentUnderstanding(
                primary_intent=ConversationalIntent.GENERAL_CHAT,
                confidence=0.3,
                reasoning="Fallback due to classification error",
                complexity_level="simple",
                urgency_level="normal"
            )

    def _map_string_to_intent(self, intent_str: str) -> ConversationalIntent:
        """Map string intent to ConversationalIntent enum"""
        intent_mapping = {
            "greeting": ConversationalIntent.GREETING,
            "research": ConversationalIntent.RESEARCH,
            "crm_action": ConversationalIntent.CRM_ACTION,
            "pms_action": ConversationalIntent.PMS_ACTION,
            "report_generation": ConversationalIntent.REPORT_GENERATION,
            "data_query": ConversationalIntent.DATA_QUERY,
            "task_management": ConversationalIntent.TASK_MANAGEMENT,
            "meeting_scheduling": ConversationalIntent.MEETING_SCHEDULING,
            "general_chat": ConversationalIntent.GENERAL_CHAT,
            "clarification": ConversationalIntent.CLARIFICATION,
            "follow_up": ConversationalIntent.FOLLOW_UP,
            "confirmation": ConversationalIntent.CONFIRMATION,
            "cancellation": ConversationalIntent.CANCELLATION,
            "db.find": ConversationalIntent.DB_FIND,
            "db.aggregate": ConversationalIntent.DB_AGGREGATE,
            "db.vectorSearch": ConversationalIntent.DB_VECTOR_SEARCH,
            "db.runCommand": ConversationalIntent.DB_RUN_COMMAND,
            "db.query": ConversationalIntent.DB_QUERY
        }

        return intent_mapping.get(intent_str, ConversationalIntent.GENERAL_CHAT)

    def _extract_current_context(self, conversation_history: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Extract current conversation context"""
        if not conversation_history:
            return {"conversation_state": "new", "previous_topics": []}

        # Analyze recent conversation
        recent_messages = conversation_history[-3:]
        topics = []
        last_intent = None

        for msg in recent_messages:
            if msg.get('intent'):
                last_intent = msg['intent']
            if msg.get('topics'):
                topics.extend(msg['topics'])

        return {
            "conversation_state": "ongoing",
            "previous_topics": list(set(topics)),
            "last_intent": last_intent,
            "message_count": len(conversation_history)
        }

    async def _extract_entities_with_reasoning(
        self,
        understanding: IntentUnderstanding,
        context: SemanticContext
    ) -> IntentUnderstanding:
        """Extract entities using reasoning rather than regex"""

        system_prompt = """You are an expert entity extractor that identifies key information from user messages.

Extract relevant entities based on the intent and context. Focus on:
1. Named entities (people, organizations, products)
2. Temporal information (dates, times, durations)
3. Quantitative information (numbers, amounts, percentages)
4. Action parameters (what, when, how, where)
5. Domain-specific terms and concepts

Return ONLY valid JSON:
{
    "entities": {
        "people": ["person names"],
        "organizations": ["company names"],
        "dates": ["date references"],
        "times": ["time references"],
        "quantities": ["numbers and amounts"],
        "actions": ["verbs indicating actions"],
        "objects": ["nouns indicating targets"],
        "conditions": ["conditional statements"],
        "constraints": ["limitations or requirements"]
    },
    "relationships": [
        {"entity1": "entity", "relationship": "type", "entity2": "entity"}
    ],
    "context_clues": ["important contextual information"],
    "follow_up_questions": ["questions to clarify ambiguous parts"]
}"""

        user_prompt = f"""Extract entities from this message:

Message: {context.user_message}
Intent: {understanding.primary_intent.value}
Confidence: {understanding.confidence}
Reasoning: {understanding.reasoning}

Extract relevant entities:"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=600
            )

            result = json.loads(response.choices[0].message.content)

            # Update understanding with extracted entities
            understanding.entities.update(result.get('entities', {}))
            understanding.context_clues = result.get('context_clues', [])
            understanding.follow_up_questions = result.get('follow_up_questions', [])

            # Add relationships and domain expertise if available
            if result.get('relationships'):
                understanding.entities['relationships'] = result['relationships']

        except Exception as e:
            logger.error("Failed to extract entities", error=str(e))

        return understanding

    async def _assess_and_refine_understanding(
        self,
        understanding: IntentUnderstanding,
        context: SemanticContext
    ) -> IntentUnderstanding:
        """Assess and refine the intent understanding"""

        # Check for clarification needs
        if understanding.confidence < 0.6 or understanding.follow_up_questions:
            understanding.entities['needs_clarification'] = True
        else:
            understanding.entities['needs_clarification'] = False

        # Assess domain expertise requirements
        domain_indicators = context.current_context.get('domain_indicators', [])
        if domain_indicators:
            understanding.domain_expertise = domain_indicators

        # Final confidence adjustment based on analysis quality
        if context.reasoning_steps:
            # Boost confidence if we have good reasoning
            understanding.confidence = min(0.95, understanding.confidence + 0.1)

        return understanding
