"""Main Research Service - Clean integration for application flow"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

import structlog

from app.config import settings
from app.models import ResearchRequest, ResearchBrief, ResearchScope
from app.core.memory_manager import MemoryManager

from .config import BusinessDeepResearchConfig
from .research_state import (
    BusinessResearchBrief, BusinessResearchState,
    BusinessResearchType, DeepResearchMetadata
)
from .research_integration import run_business_deep_research

logger = structlog.get_logger()


class ResearchService:
    """High-performance research service optimized for Groq + DDGS stack"""

    def __init__(self):
        # Lazy load heavy dependencies for faster startup
        self._memory_manager = None
        self.active_sessions: Dict[str, BusinessResearchState] = {}

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
        mode: str = "standard",
        user_id: Optional[str] = None,
        conversation_context: Optional[Dict[str, Any]] = None
    ) -> BusinessResearchBrief:
        """Main research method - unified interface for all research needs"""

        logger.info("Starting research service", query=query, mode=mode, user_id=user_id)

        # Create research request
        request = ResearchRequest(
            query=query,
            scope=scope or [],
            industry=industry,
            geo=geography,
            max_sources=max_sources
        )

        # Determine research type
        research_type = self._infer_research_type(query, scope)

        # Create business research state
        research_state = BusinessResearchState(
            business_type=research_type,
            industry_context=industry,
            geography_context=geography,
            research_status="processing"
        )

        try:
            # Execute research based on mode
            if mode == "light":
                brief = await self._light_research(request, research_state)
            elif mode == "standard":
                brief = await self._standard_research(request, research_state, conversation_context)
            elif mode == "deep":
                brief = await self._deep_research(request, research_state, conversation_context)
            else:
                # Auto mode - determine best approach
                brief = await self._auto_research(request, research_state, conversation_context)

            # Enhance with business intelligence
            brief = await self._enhance_business_intelligence(brief, research_state)

            # Add metadata
            brief.deep_research_metadata = DeepResearchMetadata(
                research_method="business_research_service",
                model_used=settings.llm_model,
                search_api="ddgs",
                total_iterations=1,
                sources_discovered=brief.total_sources,
                research_subtopics=self._extract_topics(brief)
            )

            # Calculate quality score
            brief.calculate_research_quality_score()

            # Update memory if user_id provided
            if user_id:
                await self.memory_manager.update_from_research_brief(brief, user_id)

            logger.info("Research completed successfully",
                       brief_id=brief.brief_id,
                       quality_score=brief.research_quality_score,
                       sources=brief.total_sources)

            return brief

        except Exception as e:
            logger.error("Research failed", error=str(e), query=query)
            research_state.add_error("research_error", str(e))
            raise


    def _infer_research_type(self, query: str, scope: Optional[List[ResearchScope]]) -> BusinessResearchType:
        """Infer the type of business research needed"""

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

    async def _light_research(self, request: ResearchRequest, research_state: BusinessResearchState) -> BusinessResearchBrief:
        """Light research - optimized quick analysis with limited depth"""
        # Use traditional research engine for light research (faster than deep research)
        from app.core.research_engine import ResearchEngine
        engine = ResearchEngine()

        # Reduce sources for faster execution
        light_request = ResearchRequest(
            query=request.query,
            scope=request.scope,
            industry=request.industry,
            geo=request.geo,
            max_sources=min(request.max_sources, 8)  # Limit for speed
        )

        traditional_brief = await engine.run_research(light_request)

        # Convert to business research brief with optimized processing
        brief = BusinessResearchBrief(
            query=request.query,
            business_type=research_state.business_type,
            executive_summary=traditional_brief.executive_summary,
            entities=traditional_brief.entities,
            key_questions=traditional_brief.key_questions,
            findings=traditional_brief.findings,
            ideas=traditional_brief.ideas,
            metadata={
                **traditional_brief.metadata,
                "research_method": "light_traditional",
                "performance_optimized": True
            }
        )

        return brief

    async def _standard_research(
        self,
        request: ResearchRequest,
        research_state: BusinessResearchState,
        conversation_context: Optional[Dict[str, Any]] = None
    ) -> BusinessResearchBrief:
        """Standard research - balanced deep research approach"""
        return await run_business_deep_research(request, conversation_context)

    async def _deep_research(
        self,
        request: ResearchRequest,
        research_state: BusinessResearchState,
        conversation_context: Optional[Dict[str, Any]] = None
    ) -> BusinessResearchBrief:
        """Deep research - comprehensive analysis"""
        # Use direct deep research integration
        return await run_business_deep_research(request, conversation_context)

    async def _auto_research(
        self,
        request: ResearchRequest,
        research_state: BusinessResearchState,
        conversation_context: Optional[Dict[str, Any]] = None
    ) -> BusinessResearchBrief:
        """Auto research - automatically determine best approach"""

        query_complexity = self._assess_query_complexity(request.query)
        scope_breadth = len(request.scope) if request.scope else 0
        has_specific_requirements = bool(request.industry or request.geo)

        # Decision logic
        if query_complexity >= 8 or (scope_breadth >= 3 and has_specific_requirements):
            return await self._deep_research(request, research_state, conversation_context)
        elif query_complexity >= 5 or scope_breadth >= 2:
            return await self._standard_research(request, research_state, conversation_context)
        else:
            return await self._light_research(request, research_state)

    def _assess_query_complexity(self, query: str) -> int:
        """Assess query complexity on a scale of 1-10"""
        complexity = 1

        # Length-based complexity
        if len(query.split()) > 25:
            complexity += 3
        elif len(query.split()) > 15:
            complexity += 2
        elif len(query.split()) > 8:
            complexity += 1

        # Keyword-based complexity
        complex_keywords = [
            "compare", "analyze", "evaluate", "assess", "strategic",
            "market", "industry", "competition", "feasibility", "roi",
            "comprehensive", "detailed", "thorough", "extensive"
        ]

        for keyword in complex_keywords:
            if keyword in query.lower():
                complexity += 1

        # Question count and depth indicators
        question_count = query.count("?")
        complexity += question_count * 2

        return min(10, complexity)

    async def _enhance_business_intelligence(
        self,
        brief: BusinessResearchBrief,
        research_state: BusinessResearchState
    ) -> BusinessResearchBrief:
        """Enhance research brief with business intelligence"""

        # Add strategic priorities
        brief.strategic_priorities = self._generate_strategic_priorities(brief)

        # Add risk assessment
        brief.risk_assessment = self._assess_risks(brief)

        return brief

    def _generate_strategic_priorities(self, brief: BusinessResearchBrief) -> List[str]:
        """Generate strategic priorities from research"""

        priorities = []

        # Extract from key findings
        for finding in brief.findings[:5]:
            if any(word in finding.title.lower() for word in ["opportunity", "growth", "potential", "advantage"]):
                priorities.append(f"Capitalize on: {finding.title[:50]}")
            elif any(word in finding.title.lower() for word in ["risk", "threat", "challenge", "competition"]):
                priorities.append(f"Address: {finding.title[:50]}")

        # Add business type specific priorities
        if brief.business_type == BusinessResearchType.MARKET_ANALYSIS:
            priorities.extend(["Market opportunity assessment", "Customer segmentation", "Competitive positioning"])
        elif brief.business_type == BusinessResearchType.FEASIBILITY_STUDY:
            priorities.extend(["Business model validation", "Financial feasibility", "Go-to-market strategy"])

        return list(set(priorities))[:5]

    def _assess_risks(self, brief: BusinessResearchBrief) -> Dict[str, Any]:
        """Assess risks based on research findings"""

        risk_factors = []
        mitigation_strategies = []

        # Analyze findings for risk indicators
        risk_keywords = ["risk", "threat", "challenge", "competition", "regulation", "uncertainty", "volatility"]

        for finding in brief.findings:
            content = (finding.title + finding.summary).lower()

            if any(keyword in content for keyword in risk_keywords):
                risk_factors.append({
                    "factor": finding.title,
                    "severity": "medium",
                    "description": finding.summary[:100]
                })
                mitigation_strategies.append(f"Develop strategy for: {finding.title[:50]}")

        # Determine overall risk level
        risk_level = "low" if len(risk_factors) == 0 else "medium" if len(risk_factors) <= 3 else "high"

        return {
            "overall_risk_level": risk_level,
            "risk_factors": risk_factors[:5],
            "mitigation_strategies": mitigation_strategies[:5],
            "confidence_level": 0.75
        }

    def _extract_topics(self, brief: BusinessResearchBrief) -> List[str]:
        """Extract research topics from the brief"""

        topics = []

        for finding in brief.findings:
            words = finding.title.split()[:3]
            topics.append(" ".join(words))

        for idea in brief.ideas:
            words = idea.idea.split()[:3]
            topics.append(" ".join(words))

        return list(set(topics))[:10]

