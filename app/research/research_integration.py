"""Integration layer between Open Deep Research and existing application"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

import structlog
from ddgs import DDGS
from langchain_core.tools import BaseTool, tool
from langchain_core.messages import HumanMessage

from app.config import settings
from app.models import ResearchRequest, ResearchBrief
from app.research.configuration import Configuration, SearchAPI
from app.research.deep_researcher import deep_researcher
from app.research.state import AgentInputState

logger = structlog.get_logger()


class DDGSSearchTool(BaseTool):
    """Custom DuckDuckGo search tool for Open Deep Research"""

    name: str = "ddgs_search"
    description: str = "Search the web using DuckDuckGo for comprehensive, accurate results"

    def __init__(self):
        super().__init__()
        self.ddgs = DDGS()

    async def _arun(self, queries: List[str], max_results: int = 5) -> str:
        """Execute DDGS search queries"""
        all_results = []

        for query in queries:
            try:
                # Use synchronous search since DDGS doesn't have async methods
                results = self.ddgs.text(query, max_results=max_results)

                formatted_results = []
                for result in results:
                    formatted_results.append({
                        'title': result.get('title', ''),
                        'url': result.get('href', ''),
                        'snippet': result.get('body', ''),
                        'source': 'duckduckgo'
                    })

                all_results.extend(formatted_results)

            except Exception as e:
                logger.warning("DDGS search failed", query=query, error=str(e))
                continue

        # Format results for Open Deep Research
        if not all_results:
            return "No search results found."

        formatted_output = "Search Results:\n\n"
        for i, result in enumerate(all_results[:max_results], 1):
            formatted_output += f"{i}. **{result['title']}**\n"
            formatted_output += f"   URL: {result['url']}\n"
            formatted_output += f"   Summary: {result['snippet']}\n\n"

        return formatted_output

    def _run(self, queries: List[str], max_results: int = 5) -> str:
        """Synchronous wrapper for compatibility"""
        return asyncio.run(self._arun(queries, max_results))


class BusinessDeepResearcher:
    """Enhanced deep research integration for business use cases"""

    def __init__(self):
        self.ddgs_tool = DDGSSearchTool()
        self.config = self._create_business_config()

    def _create_business_config(self) -> Configuration:
        """Create configuration optimized for business research"""

        return Configuration(
            # Use Groq models (compatible with Open Deep Research)
            research_model="groq:llama3-8b-8192",
            summarization_model="groq:llama3-8b-8192",
            compression_model="groq:llama3-8b-8192",
            final_report_model="groq:llama3-8b-8192",

            # Use custom DDGS integration
            search_api=SearchAPI.NONE,  # We'll override with custom tools

            # Business-optimized settings
            max_concurrent_research_units=3,  # Conservative for business use
            max_researcher_iterations=4,     # Focused research depth
            max_react_tool_calls=8,          # Balanced tool usage
            max_structured_output_retries=2, # Quick retries

            # Content processing
            max_content_length=40000,  # Reasonable for business content
            allow_clarification=True,  # Enable for complex business queries
        )

    async def run_business_research(
        self,
        request: ResearchRequest,
        conversation_context: Optional[Dict[str, Any]] = None
    ) -> ResearchBrief:
        """Execute deep research for business use case"""

        logger.info("Starting business deep research", query=request.query)

        try:
            # Prepare research input with business context
            research_input = self._prepare_research_input(request, conversation_context)

            # Configure the researcher with business-optimized settings
            config_dict = {
                "configurable": {
                    "research_model": self.config.research_model,
                    "summarization_model": self.config.summarization_model,
                    "compression_model": self.config.compression_model,
                    "final_report_model": self.config.final_report_model,
                    "max_concurrent_research_units": self.config.max_concurrent_research_units,
                    "max_researcher_iterations": self.config.max_researcher_iterations,
                    "search_api": "none",  # Use custom tools
                    "allow_clarification": self.config.allow_clarification,
                    "max_structured_output_retries": self.config.max_structured_output_retries,
                }
            }

            # Execute deep research
            result = await deep_researcher.ainvoke(
                {"messages": [HumanMessage(content=research_input)]},
                config=config_dict
            )

            # Convert result to ResearchBrief format
            brief = self._convert_to_research_brief(result, request)

            logger.info("Business deep research completed", brief_id=brief.brief_id)
            return brief

        except Exception as e:
            logger.error("Business deep research failed", error=str(e))
            raise

    def _prepare_research_input(
        self,
        request: ResearchRequest,
        conversation_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Prepare research input with business context"""

        # Base query
        research_prompt = f"Conduct comprehensive business research on: {request.query}\n\n"

        # Add business context
        if request.industry:
            research_prompt += f"Industry Focus: {request.industry}\n"

        if request.geo:
            research_prompt += f"Geographic Focus: {request.geo}\n"

        if request.scope:
            scope_list = [s.value for s in request.scope]
            research_prompt += f"Research Scope: {', '.join(scope_list)}\n"

        if request.timeframe:
            research_prompt += f"Timeframe: {request.timeframe}\n"

        # Add conversation context if available
        if conversation_context:
            context_str = self._format_conversation_context(conversation_context)
            if context_str:
                research_prompt += f"\nConversation Context:\n{context_str}\n"

        # Business-specific instructions
        research_prompt += """
Business Research Requirements:
1. Focus on market analysis, competitive landscape, and business opportunities
2. Include financial metrics, market size, and growth projections where relevant
3. Identify actionable insights for business strategy and decision-making
4. Consider regulatory, economic, and industry trends
5. Provide concrete recommendations with rationale

Please provide a comprehensive business research report with findings, analysis, and strategic recommendations.
"""

        return research_prompt

    def _format_conversation_context(self, context: Dict[str, Any]) -> str:
        """Format conversation context for research input"""

        context_parts = []

        if context.get("entities"):
            context_parts.append(f"Key Entities: {', '.join(context['entities'])}")

        if context.get("topics"):
            context_parts.append(f"Topics Discussed: {', '.join(context['topics'])}")

        if context.get("preferences"):
            prefs = context["preferences"]
            if prefs.get("industry"):
                context_parts.append(f"Preferred Industry: {prefs['industry']}")
            if prefs.get("geography"):
                context_parts.append(f"Preferred Geography: {prefs['geography']}")

        if context.get("research_history"):
            context_parts.append(f"Previous Research: {len(context['research_history'])} briefs")

        return "\n".join(context_parts) if context_parts else ""

    def _convert_to_research_brief(self, result: Dict[str, Any], request: ResearchRequest) -> ResearchBrief:
        """Convert Open Deep Research result to ResearchBrief format"""

        # Extract final report
        final_report = result.get("final_report", "")
        messages = result.get("messages", [])

        # Parse research findings from the final report
        findings = self._extract_findings_from_report(final_report)
        ideas = self._extract_ideas_from_report(final_report)

        # Create research brief
        brief = ResearchBrief(
            query=request.query,
            date=datetime.now(timezone.utc),
            entities=self._extract_entities_from_messages(messages),
            key_questions=self._extract_key_questions(final_report),
            findings=findings,
            ideas=ideas,
            executive_summary=self._extract_executive_summary(final_report),
            metadata={
                "research_method": "deep_research_agent",
                "model_used": self.config.research_model,
                "search_api": "ddgs",
                "max_sources": request.max_sources or 20,
                "original_request": request.model_dump(),
            }
        )

        return brief

    def _extract_findings_from_report(self, report: str) -> List[Dict[str, Any]]:
        """Extract findings from the final report"""
        # This is a simplified extraction - could be enhanced with better parsing
        findings = []

        # Look for sections that contain findings
        sections = report.split("##")

        for section in sections:
            if any(keyword in section.lower() for keyword in ["finding", "analysis", "result", "data"]):
                # Create a finding from this section
                lines = section.strip().split("\n")
                title = lines[0].strip() if lines else "Research Finding"

                findings.append({
                    "title": title,
                    "summary": "\n".join(lines[1:])[:500],  # Truncate for summary
                    "evidence": [],  # Would need to extract from raw research
                    "confidence": 0.8,  # Default confidence
                    "recency": "current",
                    "scope": "market",  # Default scope
                    "key_insights": self._extract_key_insights(section)
                })

        return findings

    def _extract_ideas_from_report(self, report: str) -> List[Dict[str, Any]]:
        """Extract actionable ideas from the report"""
        ideas = []

        # Look for recommendation sections
        if "recommendation" in report.lower() or "strategy" in report.lower():
            ideas.append({
                "idea": "Implement strategic recommendations from research",
                "rationale": "Research indicates specific opportunities and actions",
                "rice": {
                    "reach": 1000,
                    "impact": 3,
                    "confidence": 0.8,
                    "effort": 20,
                    "score": None
                },
                "prerequisites": ["Review full research report", "Assess resource availability"],
                "risks": ["Market conditions may change", "Implementation challenges"],
                "related_findings": []
            })

        return ideas

    def _extract_entities_from_messages(self, messages: List) -> List[str]:
        """Extract entities from research messages"""
        entities = []
        # Simple entity extraction - could be enhanced with NER
        common_entities = ["market", "industry", "company", "product", "service"]
        for message in messages:
            content = str(message.get("content", ""))
            for entity in common_entities:
                if entity in content.lower():
                    entities.append(entity.capitalize())

        return list(set(entities))

    def _extract_key_questions(self, report: str) -> List[str]:
        """Extract key questions addressed in the report"""
        # Simple extraction based on common question patterns
        questions = []
        lines = report.split("\n")

        for line in lines:
            line = line.strip()
            if line.endswith("?") and len(line) > 20:
                questions.append(line)

        return questions[:5]  # Limit to 5 key questions

    def _extract_executive_summary(self, report: str) -> Optional[str]:
        """Extract or generate executive summary"""
        # Look for summary section
        if "## Executive Summary" in report:
            start = report.find("## Executive Summary")
            end = report.find("##", start + 1)
            if end == -1:
                end = len(report)
            return report[start:end].strip()
        else:
            # Generate summary from first few paragraphs
            paragraphs = report.split("\n\n")[:3]
            return "\n\n".join(paragraphs)

    def _extract_key_insights(self, section: str) -> List[str]:
        """Extract key insights from a section"""
        insights = []
        lines = section.split("\n")

        for line in lines:
            line = line.strip()
            # Look for bullet points or numbered lists
            if line.startswith(("-", "*", "•")) or (line[0].isdigit() and ". " in line):
                insights.append(line.lstrip("-*•0123456789. "))

        return insights[:5]  # Limit insights


# Convenience functions for easy integration
async def run_business_deep_research(
    request: ResearchRequest,
    conversation_context: Optional[Dict[str, Any]] = None
) -> ResearchBrief:
    """Convenience function to run business deep research"""
    researcher = BusinessDeepResearcher()
    return await researcher.run_business_research(request, conversation_context)
