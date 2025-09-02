"""Simplified Open Deep Research Integration with Business Enhancements"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

import structlog
from langchain_core.messages import HumanMessage

from app.models import ResearchRequest,ResearchBrief
from .configuration import BusinessDeepResearchConfig
from .state import BusinessResearchBrief, BusinessResearchType

logger = structlog.get_logger()

# Import local deep researcher
from .deep_researcher import deep_researcher


class BusinessDeepResearcher:
    """Direct Open Deep Research with Business Enhancements

    This class integrates Open Deep Research directly with business-focused
    enhancements, DDGS search, and Groq models for optimal business research.
    """

    def __init__(self):
        # Create business-optimized configuration
        self.config = BusinessDeepResearchConfig.create_business_config()



    async def run_business_research(
        self,
        request: ResearchRequest,
        conversation_context: Optional[Dict[str, Any]] = None
    ) -> ResearchBrief:
        """Execute research using Open Deep Research with business enhancements.

        This method uses Open Deep Research as the core research engine with
        business-focused customizations, DDGS search, and Groq models.

        Args:
            request: Research request containing query, industry, geography, and parameters
            conversation_context: Optional context from ongoing conversation

        Returns:
            BusinessResearchBrief with comprehensive research findings
        """
        logger.info("Starting Open Deep Research with business enhancements", query=request.query)

        try:
            # Prepare research input with business context
            research_input = self._prepare_research_input(request, conversation_context)

            # Get business-optimized configuration
            config_dict = BusinessDeepResearchConfig.get_business_research_config(
                query_type="general",
                industry=request.industry,
                geography=request.geo
            )

            # Execute research using Open Deep Research directly
            result = await deep_researcher.ainvoke(
                {"messages": [HumanMessage(content=research_input)]},
                config=config_dict
            )

            # Convert result to BusinessResearchBrief format
            brief = self._convert_to_business_research_brief(result, request)

            logger.info("Open Deep Research completed successfully", brief_id=brief.brief_id)
            return brief

        except Exception as e:
            logger.error("Open Deep Research failed", error=str(e))
            raise

    def _prepare_research_input(
        self,
        request: ResearchRequest,
        conversation_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Prepare comprehensive research input optimized for business intelligence gathering.

        This method constructs a detailed research prompt that incorporates business context,
        industry focus, geographic scope, and conversation history to provide the deep research
        agent with comprehensive context for generating high-quality business insights.

        Args:
            request: Research request containing query, industry, geography, and scope parameters
            conversation_context: Optional context from ongoing conversation for personalization

        Returns:
            Formatted research prompt string optimized for business research execution
        """

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

    def _convert_to_business_research_brief(self, result: Dict[str, Any], request: ResearchRequest) -> BusinessResearchBrief:
        """Convert Open Deep Research result to BusinessResearchBrief format

        This method transforms the raw Open Deep Research output into a structured
        BusinessResearchBrief with business intelligence enhancements.
        """

        # Extract final report
        final_report = result.get("final_report", "")
        messages = result.get("messages", [])

        # Parse research findings from the final report
        findings = self._extract_findings_from_report(final_report)
        ideas = self._extract_ideas_from_report(final_report)

        # Create business research brief
        brief = BusinessResearchBrief(
            query=request.query,
            date=datetime.now(timezone.utc),
            entities=self._extract_entities_from_messages(messages),
            key_questions=self._extract_key_questions(final_report),
            findings=findings,
            ideas=ideas,
            executive_summary=self._extract_executive_summary(final_report),
            business_type=self._infer_business_type(request.query),
            metadata={
                "research_method": "open_deep_research_business",
                "model_used": self.config.research_model,
                "search_api": "ddgs",
                "max_sources": request.max_sources or 20,
                "original_request": request.model_dump(),
            }
        )

        # Calculate quality score
        brief.calculate_research_quality_score()

        return brief

    def _infer_business_type(self, query: str):
        """Infer business research type from query"""
        from .state import BusinessResearchType

        query_lower = query.lower()

        if any(word in query_lower for word in ["market analysis", "market opportunity"]):
            return BusinessResearchType.MARKET_ANALYSIS
        elif any(word in query_lower for word in ["competitor", "competition"]):
            return BusinessResearchType.COMPETITOR_ANALYSIS
        elif any(word in query_lower for word in ["industry", "sector"]):
            return BusinessResearchType.INDUSTRY_REPORT

        return BusinessResearchType.MARKET_ANALYSIS

    def _extract_findings_from_report(self, report: str) -> List[Dict[str, Any]]:
        """Extract findings from the final report"""
        findings = []
        sections = report.split("##")

        for section in sections:
            if any(keyword in section.lower() for keyword in ["finding", "analysis", "result", "data"]):
                lines = section.strip().split("\n")
                title = lines[0].strip() if lines else "Research Finding"

                findings.append({
                    "title": title,
                    "summary": "\n".join(lines[1:])[:500],
                    "evidence": [],
                    "confidence": 0.8,
                    "recency": "current",
                    "scope": "market",
                    "key_insights": self._extract_key_insights(section)
                })

        return findings[:3]  # Limit to top 3 findings

    def _extract_ideas_from_report(self, report: str) -> List[Dict[str, Any]]:
        """Extract actionable ideas from the report"""
        ideas = []

        if "recommendation" in report.lower() or "strategy" in report.lower():
            ideas.append({
                "idea": "Implement strategic recommendations from research",
                "rationale": "Research indicates specific opportunities and actions",
                "rice": {"reach": 1000, "impact": 3, "confidence": 0.8, "effort": 20, "score": None},
                "prerequisites": ["Review full research report"],
                "risks": ["Market conditions may change"],
                "related_findings": []
            })

        return ideas[:2]  # Limit to top 2 ideas

    def _extract_entities_from_messages(self, messages: List) -> List[str]:
        """Extract entities from research messages"""
        entities = set()
        common_entities = ["market", "industry", "company", "product", "service"]

        for message in messages:
            content = str(message.get("content", ""))
            for entity in common_entities:
                if entity in content.lower():
                    entities.add(entity.capitalize())

        return list(entities)[:5]  # Limit to 5 entities

    def _extract_key_questions(self, report: str) -> List[str]:
        """Extract key questions addressed in the report"""
        questions = []
        for line in report.split("\n"):
            line = line.strip()
            if line.endswith("?") and len(line) > 20:
                questions.append(line)

        return questions[:3]  # Limit to 3 key questions

    def _extract_executive_summary(self, report: str) -> Optional[str]:
        """Extract or generate executive summary"""
        if "## Executive Summary" in report:
            start = report.find("## Executive Summary")
            end = report.find("##", start + 1)
            if end == -1:
                end = len(report)
            return report[start:end].strip()
        else:
            # Generate summary from first few paragraphs
            paragraphs = report.split("\n\n")[:2]
            return "\n\n".join(paragraphs) if paragraphs else None

    def _extract_key_insights(self, section: str) -> List[str]:
        """Extract key insights from a section"""
        insights = []
        for line in section.split("\n"):
            line = line.strip()
            if line.startswith(("-", "*", "•")) or (line[0].isdigit() and ". " in line):
                insights.append(line.lstrip("-*•0123456789. "))

        return insights[:3]  # Limit to 3 insights


# Convenience functions for easy integration
async def run_business_deep_research(
    request: ResearchRequest,
    conversation_context: Optional[Dict[str, Any]] = None
) -> BusinessResearchBrief:
    """Execute comprehensive business-focused deep research with contextual intelligence.

    This is the main entry point for running business research that leverages
    the full deep research pipeline with business-specific enhancements.
    It automatically applies business intelligence, risk assessment, and
    strategic prioritization to research findings.

    Args:
        request: Research request containing query, scope, industry, and other parameters
        conversation_context: Optional context from ongoing conversation for personalized research

    Returns:
        BusinessResearchBrief containing comprehensive research findings, analysis, and recommendations
    """
    researcher = BusinessDeepResearcher()
    return await researcher.run_business_research(request, conversation_context)
