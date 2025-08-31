"""Main research engine orchestrator"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import structlog
import json

from app.config import settings
from app.models import (
    ResearchRequest, ResearchBrief, ResearchStatus,
    Finding, Idea, Evidence
)
from app.core.query_understanding import QueryUnderstanding
from app.core.web_research import WebResearchPipeline
from app.core.synthesis import ResearchSynthesizer
from app.integrations.crm_client import CRMClient
from app.integrations.pms_client import PMSClient

logger = structlog.get_logger()


class ResearchEngine:
    """Main orchestrator for research operations"""
    
    def __init__(self):
        self.query_understanding = QueryUnderstanding()
        self.synthesizer = ResearchSynthesizer()
        self.crm_client = CRMClient()
        self.pms_client = PMSClient()
        self.status = ResearchStatus(status="idle", progress=0)
    
    async def run_research(self, request: ResearchRequest) -> ResearchBrief:
        """Execute complete research pipeline"""
        
        logger.info("Starting research", query=request.query)
        self.status = ResearchStatus(status="running", progress=0, current_step="Initializing")
        
        try:
            # Step 1: Parse and understand query (10%)
            self.status.current_step = "Understanding query"
            self.status.progress = 10
            
            sanitized_query = self.query_understanding.sanitize_query(request.query)
            parsed = await self.query_understanding.parse_query(sanitized_query)
            
            # Override with request parameters if provided
            if request.geo:
                parsed['geography'] = request.geo
            if request.industry:
                parsed['industry'] = request.industry
            if request.scope:
                parsed['scope'] = [s.value for s in request.scope]
            
            # Step 2: Expand to subqueries (20%)
            self.status.current_step = "Expanding queries"
            self.status.progress = 20
            
            subqueries = await self.query_understanding.expand_to_subqueries(sanitized_query, parsed)
            key_questions = await self.query_understanding.generate_key_questions(sanitized_query, parsed)
            
            # Step 3: Search and fetch sources (50%)
            self.status.current_step = "Searching sources"
            self.status.progress = 30
            
            async with WebResearchPipeline() as pipeline:
                # Search for sources
                search_results = await pipeline.search_sources(
                    subqueries,
                    max_results=request.max_sources or settings.max_sources_per_query
                )
                
                self.status.current_step = "Fetching content"
                self.status.progress = 40
                
                # Fetch content from sources
                fetch_tasks = [
                    pipeline.fetch_content(result['url'])
                    for result in search_results
                ]
                sources = await asyncio.gather(*fetch_tasks)
                
                # Filter out None results
                sources = [s for s in sources if s]
                
                # Deduplicate and cluster
                self.status.current_step = "Processing sources"
                self.status.progress = 50
                
                sources = pipeline.deduplicate_sources(sources)
                clustered_sources = pipeline.cluster_by_topic(sources)
                
                # Extract evidence
                all_evidence = []
                for source in sources:
                    evidence = await pipeline.extract_facts(source, sanitized_query)
                    all_evidence.extend(evidence)
            
            # Step 4: Synthesize findings (70%)
            self.status.current_step = "Synthesizing findings"
            self.status.progress = 60
            
            findings = await self.synthesizer.synthesize_findings(
                clustered_sources,
                sanitized_query,
                key_questions
            )
            
            # Add evidence to findings
            for finding in findings:
                # Match evidence to findings by topic/scope
                relevant_evidence = [
                    e for e in all_evidence
                    if self._is_evidence_relevant(e, finding)
                ][:5]  # Limit to 5 evidence per finding
                finding.evidence = relevant_evidence
            
            # Step 5: Generate ideas (80%)
            self.status.current_step = "Generating ideas"
            self.status.progress = 70
            
            business_context = {
                "geography": parsed.get('geography'),
                "industry": parsed.get('industry'),
                "timeframe": request.timeframe
            }
            
            ideas = await self.synthesizer.generate_ideas(
                findings,
                sanitized_query,
                business_context
            )
            
            # Step 6: Generate executive summary (90%)
            self.status.current_step = "Creating summary"
            self.status.progress = 80
            
            executive_summary = await self.synthesizer.generate_executive_summary(
                findings,
                ideas,
                sanitized_query
            )
            
            # Step 7: Create research brief (100%)
            self.status.current_step = "Finalizing brief"
            self.status.progress = 90
            
            brief = ResearchBrief(
                query=request.query,
                date=datetime.utcnow(),
                entities=parsed.get('entities', []),
                key_questions=key_questions,
                findings=findings,
                ideas=ideas,
                executive_summary=executive_summary,
                metadata={
                    "parsed_query": parsed,
                    "subqueries": subqueries,
                    "sources_fetched": len(sources),
                    "clusters": list(clustered_sources.keys())
                }
            )
            
            self.status = ResearchStatus(
                status="completed",
                progress=100,
                current_step="Research completed"
            )
            
            logger.info(
                "Research completed",
                brief_id=brief.brief_id,
                findings_count=len(findings),
                ideas_count=len(ideas),
                sources_count=brief.total_sources
            )
            
            return brief
            
        except Exception as e:
            logger.error("Research failed", error=str(e))
            self.status = ResearchStatus(
                status="failed",
                progress=self.status.progress,
                current_step="Error",
                errors=[str(e)]
            )
            raise
    
    def _is_evidence_relevant(self, evidence: Evidence, finding: Finding) -> bool:
        """Check if evidence is relevant to a finding"""
        # Simple relevance check - can be improved with embeddings
        finding_text = (finding.title + finding.summary).lower()
        evidence_text = evidence.quote.lower()
        
        # Check for keyword overlap
        finding_words = set(finding_text.split())
        evidence_words = set(evidence_text.split())
        
        overlap = len(finding_words & evidence_words)
        return overlap > 5  # Arbitrary threshold
    
    async def save_to_crm(
        self,
        brief: ResearchBrief,
        lead_id: Optional[str] = None,
        business_id: Optional[str] = None,
        create_tasks: bool = False
    ) -> Dict[str, Any]:
        """Save research brief to CRM"""
        
        logger.info("Saving to CRM", brief_id=brief.brief_id, lead_id=lead_id)
        
        brief_dict = brief.model_dump()
        result = await self.crm_client.save_research_brief(
            brief_dict,
            lead_id=lead_id,
            business_id=business_id,
            create_tasks=create_tasks
        )
        
        return result
    
    async def save_to_pms(
        self,
        brief: ResearchBrief,
        project_id: str,
        create_work_items: bool = False
    ) -> Dict[str, Any]:
        """Save research brief to PMS"""
        
        logger.info("Saving to PMS", brief_id=brief.brief_id, project_id=project_id)
        
        brief_dict = brief.model_dump()
        result = await self.pms_client.save_research_brief(
            brief_dict,
            project_id=project_id,
            create_work_items=create_work_items
        )
        
        return result
    
    async def ideas_to_plan(
        self,
        brief: ResearchBrief,
        selected_idea_ids: List[str],
        timeline_weeks: int = 12
    ) -> Dict[str, Any]:
        """Convert selected ideas to execution plan"""
        
        selected_ideas = [
            idea for idea in brief.ideas
            if idea.id in selected_idea_ids
        ]
        
        if not selected_ideas:
            raise ValueError("No valid ideas selected")
        
        # Sort by RICE score
        selected_ideas.sort(key=lambda i: i.rice.score or 0, reverse=True)
        
        # Create plan structure
        plan = {
            "title": f"Execution Plan: {brief.query}",
            "timeline_weeks": timeline_weeks,
            "start_date": datetime.utcnow().isoformat(),
            "objectives": [],
            "initiatives": [],
            "milestones": []
        }
        
        # Group ideas into initiatives
        weeks_per_idea = timeline_weeks // len(selected_ideas)
        current_week = 0
        
        for i, idea in enumerate(selected_ideas, 1):
            initiative = {
                "id": f"init_{i}",
                "title": idea.idea,
                "description": idea.rationale,
                "start_week": current_week,
                "duration_weeks": min(weeks_per_idea, idea.rice.effort // 5),  # Assume 5 days per week
                "effort_days": idea.rice.effort,
                "prerequisites": idea.prerequisites,
                "risks": idea.risks,
                "expected_impact": {
                    "reach": idea.rice.reach,
                    "impact_level": idea.rice.impact,
                    "confidence": idea.rice.confidence
                },
                "tasks": self._generate_tasks_for_idea(idea)
            }
            
            plan["initiatives"].append(initiative)
            
            # Add milestone
            milestone = {
                "week": current_week + initiative["duration_weeks"],
                "title": f"Complete: {idea.idea[:50]}",
                "success_criteria": [
                    f"Reach {idea.rice.reach} entities",
                    f"Achieve impact level {idea.rice.impact}"
                ]
            }
            plan["milestones"].append(milestone)
            
            current_week += initiative["duration_weeks"]
        
        # Add objectives based on findings
        for finding in brief.findings[:3]:
            objective = {
                "title": f"Address: {finding.title}",
                "key_results": finding.key_insights[:3],
                "related_initiatives": [init["id"] for init in plan["initiatives"][:2]]
            }
            plan["objectives"].append(objective)
        
        return plan
    
    def _generate_tasks_for_idea(self, idea: Idea) -> List[Dict[str, Any]]:
        """Generate tasks breakdown for an idea"""
        
        tasks = []
        
        # Prerequisites as tasks
        for prereq in idea.prerequisites:
            tasks.append({
                "title": f"Prerequisite: {prereq}",
                "type": "prerequisite",
                "estimated_hours": 8
            })
        
        # Main implementation tasks
        tasks.extend([
            {
                "title": "Research and planning",
                "type": "planning",
                "estimated_hours": idea.rice.effort * 2  # 20% of effort
            },
            {
                "title": "Implementation",
                "type": "implementation",
                "estimated_hours": idea.rice.effort * 5  # 50% of effort
            },
            {
                "title": "Testing and validation",
                "type": "testing",
                "estimated_hours": idea.rice.effort * 2  # 20% of effort
            },
            {
                "title": "Documentation and handover",
                "type": "documentation",
                "estimated_hours": idea.rice.effort * 1  # 10% of effort
            }
        ])
        
        return tasks
    
    def get_status(self) -> ResearchStatus:
        """Get current research status"""
        return self.status