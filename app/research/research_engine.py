"""Simplified Research Engine - Agentic and Modular"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import structlog

from app.config import settings
from app.models import ResearchRequest, ResearchBrief, ResearchStatus
from app.research.service import AgenticResearchService

logger = structlog.get_logger()


class ResearchEngine:
    """Simplified research engine using agentic approach"""

    def __init__(self):
        self.research_service = AgenticResearchService()
        self.status = ResearchStatus(status="idle", progress=0)
    
    async def run_research(self, request: ResearchRequest) -> ResearchBrief:
        """Execute research using agentic approach"""

        logger.info("Starting agentic research", query=request.query)
        self.status = ResearchStatus(status="running", progress=0, current_step="Initializing")

        try:
            # Step 1: Prepare research parameters (20%)
            self.status.current_step = "Preparing research parameters"
            self.status.progress = 20

            research_params = {
                'query': request.query,
                'industry': request.industry,
                'geography': request.geo,
                'max_sources': request.max_sources or settings.max_sources_per_query,
                'scope': [s.value for s in request.scope] if request.scope else None
            }

            # Step 2: Execute agentic research (60%)
            self.status.current_step = "Conducting agentic research"
            self.status.progress = 40

            research_results = await self.research_service.perform_research(**research_params)

            # Step 3: Structure results into ResearchBrief (80%)
            self.status.current_step = "Structuring results"
            self.status.progress = 80

            brief = await self._create_research_brief_from_results(request, research_results)

            # Step 4: Complete research (100%)
            self.status = ResearchStatus(
                status="completed",
                progress=100,
                current_step="Research completed"
            )

            logger.info(
                "Agentic research completed",
                brief_id=brief.brief_id,
                findings_count=len(brief.findings),
                sources_count=len(research_results.get("sources", [])),
                confidence=research_results.get("confidence", 0)
            )

            return brief

        except Exception as e:
            logger.error("Agentic research failed", error=str(e))
            self.status = ResearchStatus(
                status="failed",
                progress=self.status.progress,
                current_step="Error",
                errors=[str(e)]
            )
            raise

    async def _create_research_brief_from_results(self, request: ResearchRequest, results: Dict[str, Any]) -> ResearchBrief:
        """Create ResearchBrief from agentic research results"""

        from app.models import Finding

        # Convert findings to proper format
        findings = []
        for finding_data in results.get("findings", []):
            finding = Finding(
                title=finding_data.get("title", "Research Finding"),
                summary=finding_data.get("content", ""),
                key_insights=[finding_data.get("title", "")],
                sources=["agent_research"],
                confidence=finding_data.get("confidence", 0.8)
            )
            findings.append(finding)

        # Create research brief
        brief = ResearchBrief(
            query=request.query,
            date=datetime.now(timezone.utc),
            entities=[],  # Could be extracted from results
            key_questions=[],  # Could be generated
            findings=findings,
            ideas=[],  # Could be generated from findings
            executive_summary=results.get("raw_response", "")[:500] + "...",
            metadata={
                "method": "agentic_research",
                "confidence": results.get("confidence", 0.5),
                "sources_count": len(results.get("sources", [])),
                "success": results.get("success", True)
            }
        )

        return brief

    async def quick_research(self, query: str, **kwargs) -> Dict[str, Any]:
        """Perform quick research using the agentic service"""
        return await self.research_service.quick_research(query, **kwargs)

    def get_status(self) -> ResearchStatus:
        """Get current research status"""
        return self.status