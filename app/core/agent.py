"""Strict agent loop that enforces tool usage for grounded responses"""

from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from enum import Enum
import asyncio
import structlog

from app.core.intent import classify_intent, ConversationalIntent
from app.core.research_engine import ResearchEngine
from app.integrations.mcp_client import mongodb_mcp_client
from app.core.synthesis import synthesize_final_answer
from app.core.memory_manager import MemoryManager
from app.chat_models import ChatMessage, MessageRole
from groq import AsyncGroq
from app.config import settings


logger = structlog.get_logger()


class Agent:
    """Strict agent that forces tool usage when required - no hallucinations"""
    
    def __init__(
        self,
        llm: AsyncGroq,
        mcp_client,
        research_engine: ResearchEngine,
        memory_manager: MemoryManager
    ):
        self.llm = llm
        self.mcp = mcp_client
        self.research = research_engine
        self.memory = memory_manager
        self.logger = logger.bind(component="agent")
    
    async def handle_user_turn(
        self,
        user_id: str,
        conversation_id: str,
        message: str,
        geo: Optional[str] = None
    ) -> Dict[str, Any]:
        """Main agent loop - routes to tools and prevents hallucinations"""
        
        # 0) Remember the message
        await self.memory.add_to_memory(
            conversation_id=conversation_id,
            content=message,
            metadata={"role": "user", "timestamp": datetime.now(timezone.utc).isoformat()},
            user_id=user_id,
            memory_type="conversation"
        )
        
        # 1) Classify intent with structured output
        intent = await classify_intent(self.llm, message)
        self.logger.info("router.intent", intent=intent.value, message=message[:100])
        
        # 2) Check MCP health
        mcp_ok = await self._check_mcp_health()
        self.logger.info("mcp.health", ok=mcp_ok)
        
        # 3) Route based on intent - STRICT routing, no fallbacks to LLM
        if intent in {
            ConversationalIntent.DB_LOOKUP,
            ConversationalIntent.CRM_ACTION,
            ConversationalIntent.PMS_ACTION,
        }:
            if not mcp_ok:
                return self._fail_safe(
                    "Database tools are currently unavailable. Please try again in a moment or contact support.",
                    intent=intent.value
                )
            
            tool_result = await self._run_mcp_flow(intent, message, user_id)
            return await self._finalize(
                tool_result.get("content", ""),
                sources=tool_result.get("sources", []),
                intent=intent.value
            )
        
        if intent in {
            ConversationalIntent.RESEARCH,
            ConversationalIntent.NEWS,
            ConversationalIntent.PRICING_RESEARCH,
            ConversationalIntent.COMPETITOR_ANALYSIS,
        }:
            # MUST use research engine - no fallback
            brief = await self.research.run(
                query=message,
                user_id=user_id,
                conversation_id=conversation_id,
                geography=geo,
                use_mcp=mcp_ok,
                max_sources=15
            )
            
            if not brief or not brief.sources:
                return self._fail_safe(
                    "I couldn't find reliable sources for that query. Please try rephrasing or asking about a different topic.",
                    intent=intent.value
                )
            
            return await self._finalize(
                brief.formatted_answer,
                sources=brief.sources,
                intent=intent.value
            )
        
        # 4) Check if general question requires fresh info
        if self._requires_fresh_info(message):
            # Force research path
            self.logger.info("forcing.research", reason="fresh_info_required")
            brief = await self.research.run(
                query=message,
                user_id=user_id,
                conversation_id=conversation_id,
                geography=geo,
                use_mcp=mcp_ok,
                max_sources=10
            )
            
            if not brief or not brief.sources:
                return self._fail_safe(
                    "I need current sources to answer that accurately. Please try again or ask about something else.",
                    intent="research_forced"
                )
            
            return await self._finalize(
                brief.formatted_answer,
                sources=brief.sources,
                intent="research_forced"
            )
        
        # 5) Only for true general chat with no data requirements
        if intent == ConversationalIntent.GENERAL:
            # Get conversation context
            context = await self.memory.get_conversation_context(
                conversation_id,
                user_id
            )
            
            # Prepare messages with strict system prompt
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. IMPORTANT: Only answer questions that don't require "
                        "current data, research, or database lookups. If the user asks about prices, dates, "
                        "statistics, or anything requiring fresh information, say you need to research that. "
                        "Be honest about your limitations."
                    )
                },
                {"role": "user", "content": message}
            ]
            
            # Add recent context
            recent = context.short_term.get("recent_messages", [])[-3:]
            for msg in recent:
                if msg.get("role") and msg.get("content"):
                    messages.insert(-1, {"role": msg["role"], "content": msg["content"]})
            
            response = await self.llm.chat.completions.create(
                model=settings.groq_model,
                messages=messages,
                temperature=0.3,  # Low temperature for consistency
                max_tokens=500
            )
            
            answer = response.choices[0].message.content
            
            # Double-check the answer doesn't contain data claims
            if self._contains_data_claims(answer):
                return self._fail_safe(
                    "I need to research that to give you accurate information.",
                    intent=intent.value
                )
            
            return await self._finalize(answer, sources=[], intent=intent.value)
        
        # Default: unclear intent, suggest clarification
        return self._fail_safe(
            "I'm not sure how to help with that. Could you please clarify what you're looking for?",
            intent=intent.value
        )
    
    async def _check_mcp_health(self) -> bool:
        """Check if MCP client is healthy"""
        try:
            if hasattr(self.mcp, 'connected'):
                return self.mcp.connected
            
            # Try a lightweight operation
            tools = await self.mcp.list_tools()
            return len(tools) > 0
        except Exception as e:
            self.logger.error("mcp.health_check_failed", error=str(e))
            return False
    
    async def _run_mcp_flow(
        self,
        intent: ConversationalIntent,
        message: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Execute MCP tool calls based on intent"""
        try:
            # Map intent to MCP operations
            if intent == ConversationalIntent.DB_LOOKUP:
                # Extract collection and query from natural language
                collection = self._extract_collection(message)
                query = self._extract_query(message)
                
                results = []
                async for chunk in self.mcp.find_documents(
                    collection=collection,
                    filter_query=query,
                    limit=10,
                    user_id=user_id
                ):
                    if chunk.get("type") == "tool.output.data":
                        results.append(chunk["data"])
                
                if not results:
                    return {
                        "content": f"No documents found in {collection} matching your query.",
                        "sources": [{"type": "mongodb", "collection": collection}]
                    }
                
                # Format results
                formatted = self._format_db_results(results, collection)
                return {
                    "content": formatted,
                    "sources": [{"type": "mongodb", "collection": collection, "count": len(results)}]
                }
            
            elif intent == ConversationalIntent.CRM_ACTION:
                # Route to CRM operations
                return await self._handle_crm_action(message, user_id)
            
            elif intent == ConversationalIntent.PMS_ACTION:
                # Route to PMS operations
                return await self._handle_pms_action(message, user_id)
            
            else:
                return {"content": "Unsupported MCP operation", "sources": []}
                
        except Exception as e:
            self.logger.error("mcp.execution_failed", error=str(e), intent=intent.value)
            return {
                "content": f"Database operation failed: {str(e)}",
                "sources": []
            }
    
    def _extract_collection(self, message: str) -> str:
        """Extract collection name from natural language"""
        msg_lower = message.lower()
        
        # Priority-based collection mapping
        if any(x in msg_lower for x in ["lead status", "leadstatus"]):
            return "leadStatus"
        elif "activit" in msg_lower and "status" not in msg_lower:
            return "activity"
        elif any(x in msg_lower for x in ["note", "comment"]) and "lead" not in msg_lower:
            return "notes"
        elif any(x in msg_lower for x in ["task", "todo", "work item"]):
            return "task"
        elif any(x in msg_lower for x in ["meeting", "appointment", "call"]):
            return "meeting"
        elif any(x in msg_lower for x in ["lead", "customer", "client", "deal"]):
            return "Lead"
        elif any(x in msg_lower for x in ["project", "sprint", "milestone"]):
            return "ProjectManagement.project"
        elif any(x in msg_lower for x in ["staff", "employee", "hr", "personnel"]):
            return "Staff.staff"
        
        # Default
        return "Lead"
    
    def _extract_query(self, message: str) -> Dict[str, Any]:
        """Extract query parameters from natural language"""
        # Simple extraction - could be enhanced with NLP
        query = {}
        
        msg_lower = message.lower()
        
        # Look for status filters
        if "status" in msg_lower:
            if "new" in msg_lower:
                query["status"] = "NEW"
            elif "qualified" in msg_lower:
                query["status"] = "QUALIFIED"
            elif "closed" in msg_lower:
                query["status"] = "CLOSED"
        
        # Look for time filters
        if "today" in msg_lower:
            query["createdAt"] = {"$gte": datetime.now(timezone.utc).replace(hour=0, minute=0).isoformat()}
        elif "yesterday" in msg_lower:
            yesterday = datetime.now(timezone.utc).replace(hour=0, minute=0)
            yesterday = yesterday.replace(day=yesterday.day - 1)
            query["createdAt"] = {
                "$gte": yesterday.isoformat(),
                "$lt": datetime.now(timezone.utc).replace(hour=0, minute=0).isoformat()
            }
        
        # Look for count/limit
        if "last" in msg_lower:
            # Will be handled by limit parameter
            pass
        
        return query
    
    def _format_db_results(self, results: List[Dict], collection: str) -> str:
        """Format database results for display"""
        if not results:
            return f"No documents found in {collection}."
        
        formatted = f"Found {len(results)} documents in {collection}:\n\n"
        
        for i, doc in enumerate(results[:5], 1):  # Show first 5
            # Extract key fields based on collection type
            if collection in ["Lead", "leadStatus"]:
                name = doc.get("name", doc.get("company", "Unknown"))
                status = doc.get("status", "N/A")
                created = doc.get("createdAt", "")[:10]  # Date only
                formatted += f"{i}. **{name}** - Status: {status} (Created: {created})\n"
            
            elif collection == "task":
                title = doc.get("title", doc.get("name", "Untitled"))
                priority = doc.get("priority", "NORMAL")
                status = doc.get("status", "PENDING")
                formatted += f"{i}. **{title}** - Priority: {priority}, Status: {status}\n"
            
            elif collection == "notes":
                subject = doc.get("subject", "Note")
                created = doc.get("createdAt", "")[:10]
                formatted += f"{i}. **{subject}** (Created: {created})\n"
            
            else:
                # Generic formatting
                id_field = doc.get("_id", doc.get("id", ""))
                name_field = doc.get("name", doc.get("title", doc.get("subject", "")))
                formatted += f"{i}. {name_field or id_field or 'Document'}\n"
        
        if len(results) > 5:
            formatted += f"\n... and {len(results) - 5} more documents."
        
        return formatted.strip()
    
    async def _handle_crm_action(self, message: str, user_id: str) -> Dict[str, Any]:
        """Handle CRM-specific actions"""
        # This would map natural language to specific CRM operations
        # For now, return a placeholder
        return {
            "content": "CRM action detected. This would execute the appropriate CRM operation.",
            "sources": [{"type": "crm_action", "system": "mongodb"}]
        }
    
    async def _handle_pms_action(self, message: str, user_id: str) -> Dict[str, Any]:
        """Handle PMS-specific actions"""
        # This would map natural language to specific PMS operations
        # For now, return a placeholder
        return {
            "content": "PMS action detected. This would execute the appropriate project management operation.",
            "sources": [{"type": "pms_action", "system": "mongodb"}]
        }
    
    async def _finalize(
        self,
        answer: str,
        sources: List[Dict[str, Any]],
        intent: str
    ) -> Dict[str, Any]:
        """Finalize the response with proper formatting and source attribution"""
        final = synthesize_final_answer(answer, sources, intent)
        
        # Log the final decision
        self.logger.info(
            "agent.final",
            intent=intent,
            sources_count=len(sources),
            has_content=bool(final.get("content"))
        )
        
        return final
    
    def _requires_fresh_info(self, text: str) -> bool:
        """Check if the query requires current/fresh information"""
        needles = [
            "latest", "current", "today", "now", "recent",
            "price", "cost", "pricing", "rate",
            "who is", "what is the status",
            "breaking", "news", "update",
            "schedule", "release", "launch",
            "version", "available",
            "how many", "how much",
            "statistics", "data", "numbers"
        ]
        text_lower = text.lower()
        return any(needle in text_lower for needle in needles)
    
    def _contains_data_claims(self, text: str) -> bool:
        """Check if the response contains specific data claims that should be sourced"""
        data_indicators = [
            "%", "$", "€", "£", "¥",  # Financial data
            "million", "billion", "thousand",
            "2024", "2023", "2022",  # Recent years
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
            "Q1", "Q2", "Q3", "Q4",  # Quarters
            "study shows", "research indicates", "data reveals",
            "according to", "report states", "survey found"
        ]
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in data_indicators)
    
    def _fail_safe(self, msg: str, intent: str = "unknown") -> Dict[str, Any]:
        """Return a safe failure message"""
        self.logger.warning("agent.fail_safe", message=msg, intent=intent)
        return {
            "content": msg,
            "sources": [],
            "intent": intent,
            "status": "failed"
        }