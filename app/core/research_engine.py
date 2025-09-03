"""Main research engine orchestrator"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import structlog
import json

from app.config import settings
from app.models import (
    ResearchRequest, ResearchBrief, ResearchStatus,
    Finding, Idea, Evidence
)
from app.core.query_understanding import QueryUnderstanding
from app.core.synthesis import ResearchSynthesizer
from app.integrations.mcp_client import mongodb_mcp_client
from app.research.service import ResearchService

logger = structlog.get_logger()


class ResearchEngine:
    """Main orchestrator for research operations with strict source requirements"""
    
    def __init__(self):
        self.query_understanding = QueryUnderstanding()
        self.synthesizer = ResearchSynthesizer()
        self.mcp_client = mongodb_mcp_client
        self.research_service = ResearchService()
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
            
            # Step 3: Use ResearchService for research (50%)
            self.status.current_step = "Conducting research"
            self.status.progress = 30

            # Check MCP availability for enhanced research
            mcp_available = False
            try:
                mcp_health = await self.mcp_client.health_check()
                mcp_available = mcp_health.get("status") == "connected"
                if mcp_available:
                    logger.info("MCP client available for research enhancement", tools=mcp_health.get("available_tools", 0))
            except Exception as e:
                logger.warning("MCP health check failed, proceeding without MCP", error=str(e))

            # Use the working ResearchService instead of broken WebResearchPipeline
            research_brief = await self.research_service.research(
                query=sanitized_query,
                scope=request.scope,
                industry=request.industry,
                geography=request.geo,
                max_sources=request.max_sources or settings.max_sources_per_query,
                mcp_available=mcp_available
            )

            # Extract findings and ideas from the research brief
            findings = research_brief.findings
            ideas = research_brief.ideas

            # Enhance with MCP data if available
            if mcp_available:
                self.status.current_step = "Enhancing with database data"
                self.status.progress = 60
                research_brief = await self.enhance_research_with_mcp_data(research_brief, sanitized_query)
                findings = research_brief.findings  # Update with enhanced findings

            self.status.progress = 70

            # Step 4: Use executive summary from research brief (80%)
            self.status.current_step = "Finalizing results"
            self.status.progress = 80

            executive_summary = research_brief.executive_summary
            
            # Step 5: Create research brief (100%)
            self.status.current_step = "Finalizing brief"
            self.status.progress = 90

            brief = ResearchBrief(
                query=request.query,
                date=datetime.now(timezone.utc),
                entities=parsed.get('entities', []),
                key_questions=key_questions,
                findings=findings,
                ideas=ideas,
                executive_summary=executive_summary,
                metadata={
                    "parsed_query": parsed,
                    "subqueries": subqueries,
                    "sources_fetched": research_brief.total_sources,
                    "research_method": "deep_research_integration"
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
                sources_count=research_brief.total_sources
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

    async def query_mcp_for_research_data(self, query: str, collection: str = None) -> List[Dict[str, Any]]:
        """Query MCP for research-relevant data"""
        try:
            # Check MCP availability
            mcp_health = await self.mcp_client.health_check()
            if mcp_health.get("status") != "connected":
                logger.warning("MCP not available for research data query")
                return []

            # Determine collection if not specified
            if not collection:
                # Try to infer collection from query
                query_lower = query.lower()
                if any(word in query_lower for word in ["lead", "customer", "crm"]):
                    collection = "Lead"
                elif any(word in query_lower for word in ["project", "task"]):
                    collection = "ProjectManagement.project"
                elif any(word in query_lower for word in ["staff", "employee", "hr"]):
                    collection = "Staff.staff"
                else:
                    collection = "Lead"  # Default

            # Perform search
            search_results = []
            async for result in self.mcp_client.find_documents(
                collection=collection,
                filter_query={"$text": {"$search": query}},
                limit=20
            ):
                if result.get("type") == "tool.output.data":
                    search_results.append(result.get("data", {}))

            logger.info(f"MCP research query found {len(search_results)} results from {collection}")
            return search_results

        except Exception as e:
            logger.error("Failed to query MCP for research data", error=str(e))
            return []

    async def enhance_research_with_mcp_data(self, research_brief: ResearchBrief, query: str) -> ResearchBrief:
        """Enhance research brief with MCP data"""
        try:
            # Get relevant data from MCP
            mcp_data = await self.query_mcp_for_research_data(query)

            if not mcp_data:
                return research_brief

            # Create additional findings from MCP data
            mcp_findings = []
            for i, data_item in enumerate(mcp_data[:5]):  # Limit to 5 additional findings
                finding = Finding(
                    title=f"MCP Data Insight {i+1}",
                    summary=f"Relevant data found in database: {str(data_item)[:200]}...",
                    key_insights=[
                        f"Database record contains: {list(data_item.keys())[:5]}",
                        "This data may be relevant to your research query"
                    ],
                    sources=["Internal Database"],
                    confidence=0.7
                )
                mcp_findings.append(finding)

            # Add MCP findings to the research brief
            research_brief.findings.extend(mcp_findings)

            # Update metadata
            research_brief.metadata["mcp_enhanced"] = True
            research_brief.metadata["mcp_data_points"] = len(mcp_data)

            logger.info(f"Enhanced research brief with {len(mcp_findings)} MCP findings")
            return research_brief

        except Exception as e:
            logger.error("Failed to enhance research with MCP data", error=str(e))
            return research_brief
    
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
            "start_date": datetime.now(timezone.utc).isoformat(),
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
    
    async def run(
        self,
        query: str,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        geography: Optional[str] = None,
        use_mcp: bool = True,
        max_sources: int = 15
    ) -> Any:
        """Simplified interface for agent - returns object with sources and formatted_answer"""
        from types import SimpleNamespace
        
        try:
            # Create research request
            request = ResearchRequest(
                query=query,
                user_id=user_id,
                conversation_id=conversation_id,
                geo=geography,
                max_sources=max_sources
            )
            
            # Run research through service (which does the actual web fetching)
            research_brief = await self.research_service.research(
                query=query,
                geography=geography,
                max_sources=max_sources,
                user_id=user_id
            )
            
            # Extract sources from findings
            sources = []
            for finding in research_brief.findings:
                if hasattr(finding, 'sources'):
                    for source_url in finding.sources:
                        sources.append({
                            "type": "web",
                            "url": source_url,
                            "title": finding.title,
                            "snippet": finding.summary[:200]
                        })
            
            # Add metadata source
            sources.append({
                "type": "research_brief",
                "brief_id": research_brief.brief_id,
                "findings_count": len(research_brief.findings),
                "ideas_count": len(research_brief.ideas)
            })
            
            # Format the answer
            formatted_answer = await self._format_research_conversationally(research_brief)
            
            return SimpleNamespace(
                sources=sources,
                formatted_answer=formatted_answer,
                brief=research_brief
            )
            
        except Exception as e:
            logger.error("Research run failed", error=str(e))
            # Return empty result on failure
            return SimpleNamespace(
                sources=[],
                formatted_answer="",
                brief=None
            )
    
    async def _format_research_conversationally(self, brief: ResearchBrief) -> str:
        """Format research results in a conversational way"""
        response = f"Based on my research on **{brief.query}**, here's what I found:\n\n"
        
        if brief.executive_summary:
            response += f"{brief.executive_summary}\n\n"
        
        if brief.findings:
            response += "**Key Findings:**\n"
            for i, finding in enumerate(brief.findings[:4], 1):
                response += f"{i}. **{finding.title}** - {finding.summary}\n"
                if finding.key_insights:
                    response += f"   • {finding.key_insights[0]}\n"
                response += "\n"
        
        if brief.ideas:
            response += "**Recommendations:**\n"
            sorted_ideas = sorted(brief.ideas, key=lambda x: x.rice.score if x.rice.score else 0, reverse=True)
            for i, idea in enumerate(sorted_ideas[:3], 1):
                response += f"{i}. {idea.idea}\n"
        
        response += f"\n*Analyzed {brief.total_sources} sources*"
        
        return response