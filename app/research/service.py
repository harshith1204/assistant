"""Simplified Research Service - Open Deep Research with Business Enhancements"""

from typing import Dict, Any, List, Optional

import structlog

from app.models import ResearchRequest, ResearchScope
from app.core.memory_manager import MemoryManager

from .state import BusinessResearchBrief, BusinessResearchType
from .research_integration import BusinessDeepResearcher

logger = structlog.get_logger()


class ResearchService:
    """Simplified research service using Open Deep Research with business enhancements"""

    def __init__(self):
        # Lazy load heavy dependencies for faster startup
        self._memory_manager = None
        self.researcher = BusinessDeepResearcher()

    @property
    def memory_manager(self):
        if self._memory_manager is None:
            self._memory_manager = MemoryManager()
        return self._memory_manager

    async def research(
        self,
        query: str,
        scope: Optional[List[ResearchScope]] = None,
        industry: Optional[str] = None,
        geography: Optional[str] = None,
        max_sources: int = 15,
        user_id: Optional[str] = None,
        conversation_context: Optional[Dict[str, Any]] = None
    ) -> BusinessResearchBrief:
        """Execute research using Open Deep Research with business enhancements

        This method uses Open Deep Research as the core research engine, enhanced with
        business intelligence, DDGS search, and Groq models for optimal performance.

        Args:
            query: The research query string
            scope: Optional list of research scope areas
            industry: Optional industry focus
            geography: Optional geographic focus
            max_sources: Maximum number of sources to analyze
            user_id: Optional user identifier for memory updates
            conversation_context: Optional context from ongoing conversation

        Returns:
            BusinessResearchBrief containing comprehensive research findings and analysis
        """
        logger.info("Starting Open Deep Research with business enhancements", query=query, user_id=user_id)

        try:
            # Create research request
            request = ResearchRequest(
                query=query,
                scope=scope or [],
                industry=industry,
                geo=geography,
                max_sources=max_sources
            )

            # Execute research using Open Deep Research with business enhancements
            brief = await self.researcher.run_business_research(request, conversation_context)

            # Determine research type for metadata
            research_type = self._infer_research_type(query, scope)

            # Enhance brief with business intelligence
            brief.business_type = research_type
            brief.calculate_research_quality_score()

            # Update memory if user_id provided
            if user_id:
                await self.memory_manager.update_from_research_brief(brief, user_id)

            logger.info("Open Deep Research completed successfully",
                       brief_id=brief.brief_id,
                       quality_score=brief.research_quality_score,
                       sources=brief.total_sources)

            return brief

        except Exception as e:
            logger.error("Open Deep Research failed", error=str(e), query=query)
            raise


    def _infer_research_type(self, query: str, scope: Optional[List[ResearchScope]]) -> BusinessResearchType:
        """Analyze query content to determine the most appropriate business research type.

        This method examines the user's query and research scope to automatically classify
        the type of business research needed, enabling the system to apply appropriate
        research strategies, templates, and analysis frameworks.

        Args:
            query: The user's research query string
            scope: Optional list of research scope areas to consider

        Returns:
            BusinessResearchType enum value representing the inferred research category
        """

        query_lower = query.lower()

        # Check for specific research patterns
        if any(word in query_lower for word in ["market analysis", "market opportunity", "market research", "market size"]):
            return BusinessResearchType.MARKET_ANALYSIS

        elif any(word in query_lower for word in ["competitor", "competition", "competitive", "rival", "versus"]):
            return BusinessResearchType.COMPETITOR_ANALYSIS

        elif any(word in query_lower for word in ["industry", "sector", "industry report", "sector analysis"]):
            return BusinessResearchType.INDUSTRY_REPORT

        elif any(word in query_lower for word in ["feasibility", "business plan", "startup", "new venture", "roi"]):
            return BusinessResearchType.FEASIBILITY_STUDY

        elif any(word in query_lower for word in ["strategic", "strategy", "strategic planning", "business strategy"]):
            return BusinessResearchType.STRATEGIC_PLANNING

        elif any(word in query_lower for word in ["product", "development", "innovation", "new product"]):
            return BusinessResearchType.PRODUCT_DEVELOPMENT

        elif any(word in query_lower for word in ["market entry", "expansion", "new market", "entry strategy"]):
            return BusinessResearchType.MARKET_ENTRY

        elif any(word in query_lower for word in ["investment", "funding", "valuation", "investor"]):
            return BusinessResearchType.INVESTMENT_ANALYSIS

        else:
            return BusinessResearchType.MARKET_ANALYSIS  # Default

