"""Synthesis module to format final answers with source attribution and prevent fabrication"""

from typing import Dict, Any, List, Optional
import structlog

logger = structlog.get_logger()


def synthesize_final_answer(
    draft: str | Dict[str, Any],
    sources: List[Dict[str, Any]],
    intent: str = "unknown"
) -> Dict[str, Any]:
    """
    Synthesize final answer with proper source attribution.
    Prevents fabrication by refusing to answer without sources when needed.
    """
    
    # Intents that REQUIRE sources
    source_required_intents = {
        "research", "news", "pricing_research", "competitor_analysis",
        "db_lookup", "db.find", "db.aggregate", "db.vectorSearch"
    }
    
    # Check if sources are required but missing
    if intent in source_required_intents and not sources:
        logger.warning("No sources for source-required intent", intent=intent)
        return {
            "content": _get_no_source_message(intent),
            "sources": [],
            "intent": intent,
            "status": "no_sources"
        }
    
    # Normalize the draft content
    if isinstance(draft, dict) and "content" in draft:
        content = draft["content"]
    else:
        content = str(draft)
    
    # If no content generated
    if not content or content.strip() == "":
        return {
            "content": "I couldn't generate a response. Please try rephrasing your question.",
            "sources": [],
            "intent": intent,
            "status": "no_content"
        }
    
    # Normalize and enrich sources
    normalized_sources = []
    for source in sources:
        normalized = _normalize_source(source)
        if normalized:
            normalized_sources.append(normalized)
    
    # Add source citations to content if research/news
    if intent in ["research", "news", "pricing_research", "competitor_analysis"] and normalized_sources:
        content = _add_source_citations(content, normalized_sources)
    
    return {
        "content": content,
        "sources": normalized_sources,
        "intent": intent,
        "status": "success",
        "metadata": {
            "sources_count": len(normalized_sources),
            "has_web_sources": any(s.get("type") == "web" for s in normalized_sources),
            "has_db_sources": any(s.get("type") in ["mongodb", "database"] for s in normalized_sources)
        }
    }


def _get_no_source_message(intent: str) -> str:
    """Get appropriate message when sources are missing"""
    messages = {
        "research": "I need to research that topic to provide accurate information. Please ensure web search is enabled or try rephrasing your question.",
        "news": "I need access to current news sources to answer that. Please try again with web search enabled.",
        "pricing_research": "I need to look up current pricing information from reliable sources. Please try again.",
        "competitor_analysis": "I need to research competitor information to provide an accurate analysis. Please try again.",
        "db_lookup": "I couldn't access the database. Please ensure the MCP client is connected and try again.",
        "db.find": "Database query failed. Please check the connection and try again.",
        "db.aggregate": "Database aggregation failed. Please verify the query and try again.",
        "db.vectorSearch": "Vector search failed. Please ensure vector search is enabled and try again."
    }
    
    return messages.get(
        intent,
        "I need additional data sources to answer that accurately. Please try again or rephrase your question."
    )


def _normalize_source(source: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize source information for consistent display"""
    if not source:
        return None
    
    normalized = {
        "type": source.get("type", "unknown"),
        "title": source.get("title", source.get("name", "Untitled")),
        "url": source.get("url", source.get("link")),
        "snippet": source.get("snippet", source.get("description", "")),
        "metadata": {}
    }
    
    # Handle database sources
    if source.get("type") in ["mongodb", "database"] or source.get("collection"):
        normalized["type"] = "mongodb"
        normalized["collection"] = source.get("collection", "unknown")
        normalized["database"] = source.get("database", "default")
        normalized["count"] = source.get("count", 0)
        normalized["metadata"]["query_type"] = source.get("operation", "find")
    
    # Handle web sources
    elif source.get("type") == "web" or source.get("url"):
        normalized["type"] = "web"
        normalized["domain"] = _extract_domain(source.get("url", ""))
        normalized["metadata"]["relevance_score"] = source.get("relevance_score", 0.5)
    
    # Handle research brief sources
    elif source.get("type") == "research_brief":
        normalized["type"] = "research"
        normalized["brief_id"] = source.get("brief_id")
        normalized["metadata"]["findings_count"] = source.get("findings_count", 0)
    
    # Skip invalid sources
    if normalized["type"] == "unknown" and not normalized["title"]:
        return None
    
    return normalized


def _extract_domain(url: str) -> str:
    """Extract domain from URL"""
    if not url:
        return ""
    
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc or parsed.path.split('/')[0]
    except:
        return ""


def _add_source_citations(content: str, sources: List[Dict[str, Any]]) -> str:
    """Add source citations to content for research responses"""
    if not sources:
        return content
    
    # Don't add citations if they're already present
    if "Sources:" in content or "References:" in content:
        return content
    
    # Build citation section
    citations = ["\n\n**Sources:**"]
    
    for i, source in enumerate(sources[:5], 1):  # Limit to 5 sources
        if source["type"] == "web":
            domain = source.get("domain", "Unknown")
            title = source.get("title", "Web Source")
            citations.append(f"{i}. [{title}]({source.get('url', '#')}) - {domain}")
        
        elif source["type"] == "mongodb":
            collection = source.get("collection", "Unknown")
            count = source.get("count", 0)
            citations.append(f"{i}. Database: {collection} ({count} records)")
        
        elif source["type"] == "research":
            brief_id = source.get("brief_id", "Unknown")[:8]
            findings = source.get("metadata", {}).get("findings_count", 0)
            citations.append(f"{i}. Research Brief {brief_id} ({findings} findings)")
        
        else:
            title = source.get("title", "Source")
            citations.append(f"{i}. {title}")
    
    if len(sources) > 5:
        citations.append(f"... and {len(sources) - 5} more sources")
    
    return content + "\n".join(citations)


class ResearchSynthesizer:
    """Synthesize research results with citations"""
    
    def __init__(self):
        self.logger = logger.bind(component="synthesizer")
    
    def build_prompt(
        self,
        context: Dict[str, Any],
        intent: str,
        entities: Dict[str, Any],
        research_notes: Optional[str] = None
    ) -> str:
        """Build token-tight prompt scaffold for synthesis"""
        
        # Base instruction based on intent
        base_instructions = {
            "research": "Provide comprehensive research findings with specific details and examples.",
            "news": "Summarize the latest news and developments with dates and sources.",
            "pricing_research": "Present pricing information with specific numbers and comparisons.",
            "competitor_analysis": "Analyze competitors with specific strengths, weaknesses, and market positioning.",
            "db_lookup": "Present the database query results clearly and concisely.",
            "general": "Be helpful and conversational. Acknowledge when you don't have specific information."
        }
        
        instruction = base_instructions.get(intent, base_instructions["general"])
        
        # Build context section
        context_parts = []
        
        # User profile (if available)
        if context.get("profile"):
            profile_facts = [str(fact) for fact in context["profile"][:5]]
            if profile_facts:
                context_parts.append(f"User Context: {'; '.join(profile_facts)}")
        
        # Recent conversation summary
        if context.get("conversation_summary"):
            context_parts.append(f"Recent Discussion: {context['conversation_summary'][:200]}...")
        
        # Research notes (if provided)
        if research_notes:
            context_parts.append(f"Research Findings:\n{research_notes}")
        
        # Relevant memories
        if context.get("ranked_memories"):
            memories = [m.get("content", "") for m in context["ranked_memories"][:3]]
            if memories:
                context_parts.append(f"Relevant Context: {'; '.join(memories[:100] for m in memories)}")
        
        # Build the prompt
        prompt = f"""You are an AI assistant. {instruction}

CRITICAL RULES:
- Only make factual claims when you have sources
- Say "I don't have current information on that" if unsure
- Be specific with numbers, dates, and facts
- Acknowledge limitations honestly

{chr(10).join(context_parts) if context_parts else "No additional context."}

Remember: Quality over speculation. If you don't have solid information, say so."""
        
        return prompt
    
    def format_with_citations(
        self,
        content: str,
        sources: List[Dict[str, Any]]
    ) -> str:
        """Format content with inline citations"""
        
        if not sources:
            return content
        
        # Add footnote-style citations for key facts
        # This is a simplified version - could be enhanced with NLP
        
        return content  # For now, return as-is (citations added by _add_source_citations)