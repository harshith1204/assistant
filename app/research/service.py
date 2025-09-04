"""Agentic Research Service - Modular and Focused"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import structlog

from app.models import ResearchRequest, ResearchBrief, Finding, Idea, Evidence
from app.core.memory_manager import MemoryManager

logger = structlog.get_logger()


class AgenticResearchService:
    """Agentic research service using modular tools instead of monolithic research engine"""

    def __init__(self, agent=None):
        self._memory_manager = None
        self.agent = agent  # Can be AgenticAssistant or None for standalone use

    @property
    def memory_manager(self):
        if self._memory_manager is None:
            self._memory_manager = MemoryManager()
        return self._memory_manager

    async def perform_research(self, query: str, **kwargs) -> Dict[str, Any]:
        """Execute research using agentic approach with modular tools

        This method uses the agent's reasoning capabilities to plan and execute
        research using focused, specialized tools instead of a monolithic engine.

        Args:
            query: The research query string
            **kwargs: Additional research parameters (industry, geography, etc.)

        Returns:
            Dictionary containing research results and metadata
        """
        logger.info("Starting agentic research", query=query)

        try:
            if self.agent:
                # Use agent for full agentic research
                return await self._perform_agentic_research(query, **kwargs)
            else:
                # Use simplified research approach
                return await self._perform_simplified_research(query, **kwargs)

        except Exception as e:
            logger.error("Agentic research failed", error=str(e), query=query)
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "findings": [],
                "sources": [],
                "confidence": 0.0
            }

    async def _perform_agentic_research(self, query: str, **kwargs) -> Dict[str, Any]:
        """Perform full agentic research using the agent"""
        # Create a specialized research context
        research_context = f"""
        Conduct comprehensive research on: {query}

        Additional Context:
        - Industry: {kwargs.get('industry', 'General')}
        - Geography: {kwargs.get('geography', 'Global')}
        - Scope: {kwargs.get('scope', 'Comprehensive')}
        - Max Sources: {kwargs.get('max_sources', 10)}

        Use the available research tools to gather information, analyze it, and synthesize findings.
        """

        # Use agent to process research request
        async def research_callback(message: str):
            logger.info(f"Research progress: {message}")

        # Execute research through agent
        result = await self.agent.process_request(
            user_message=research_context,
            user_id=kwargs.get('user_id', 'research_user'),
            conversation_id=f"research_{query.replace(' ', '_')}_{datetime.now(timezone.utc).timestamp()}",
            streaming_callback=research_callback
        )

        # Parse and structure the research results
        research_results = self._parse_research_results(result, query)

        # Store in memory if user_id provided
        if kwargs.get('user_id'):
            await self.memory_manager.store_memory(
                content=f"Research completed: {query}",
                memory_type="research",
                user_id=kwargs['user_id'],
                metadata={
                    "query": query,
                    "results": research_results,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )

        logger.info("Agentic research completed successfully", query=query)
        return research_results

    async def _perform_simplified_research(self, query: str, **kwargs) -> Dict[str, Any]:
        """Perform simplified research without full agent"""
        # Basic web search and analysis
        try:
            # Import here to avoid circular imports
            from app.core.agent import WebSearchTool, ContentAnalysisTool

            search_tool = WebSearchTool()
            analysis_tool = ContentAnalysisTool(None)  # No LLM client for basic analysis

            # Perform web search
            search_result = await search_tool._execute_impl(
                query=query,
                num_results=kwargs.get('max_sources', 5)
            )

            if search_result.get('results'):
                # Analyze search results
                search_content = "\n".join([
                    f"{r['title']}: {r.get('snippet', '')}"
                    for r in search_result['results']
                ])

                # Basic analysis without LLM
                analysis_result = {
                    "content_length": len(search_content),
                    "analysis_type": "summary",
                    "analysis": f"Found {len(search_result['results'])} search results for '{query}'",
                    "confidence": 0.6
                }
            else:
                analysis_result = {
                    "content_length": 0,
                    "analysis_type": "summary",
                    "analysis": "No search results found",
                    "confidence": 0.0
                }

            return {
                "success": True,
                "query": query,
                "method": "simplified_research",
                "findings": [{
                    "title": f"Search Results for '{query}'",
                    "content": analysis_result["analysis"],
                    "confidence": analysis_result["confidence"],
                    "source": "web_search"
                }],
                "sources": [r.get('url', r.get('title', 'Unknown')) for r in search_result.get('results', [])],
                "confidence": analysis_result["confidence"]
            }

        except Exception as e:
            logger.error("Simplified research failed", error=str(e))
            return {
                "success": False,
                "error": f"Simplified research failed: {str(e)}",
                "query": query,
                "findings": [],
                "sources": [],
                "confidence": 0.0
            }

    async def _quick_research_with_agent(self, query: str, num_sources: int = 3) -> Dict[str, Any]:
        """Perform quick research using agent tools (moved here to avoid circular imports)"""
        # Import here to avoid circular import
        try:
            from app.core.agent import WebSearchTool, ContentAnalysisTool

            # Use web search tool directly
            search_tool = WebSearchTool()
            search_result = await search_tool._execute_impl(
                query=query,
                num_results=num_sources
            )

            if not search_result.get("success"):
                return {
                    "success": False,
                    "error": "Web search failed",
                    "query": query
                }

            # Analyze the search results
            search_content = "\n".join([
                f"{result['title']}: {result.get('snippet', '')}"
                for result in search_result.get("results", [])
            ])

            analysis_tool = ContentAnalysisTool(None)  # No LLM client for basic analysis
            analysis_result = await analysis_tool._execute_impl(
                content=search_content,
                analysis_type="insights"
            )

            return {
                "success": True,
                "query": query,
                "search_results": search_result,
                "analysis": analysis_result,
                "method": "quick_research"
            }

        except Exception as e:
            logger.error("Quick research with agent failed", error=str(e))
            return {
                "success": False,
                "error": f"Quick research failed: {str(e)}",
                "query": query
            }

    def _parse_research_results(self, agent_response: str, query: str) -> Dict[str, Any]:
        """Parse the agent's research response into structured format"""

        # Extract key components from agent response
        findings = self._extract_findings_from_response(agent_response)
        sources = self._extract_sources_from_response(agent_response)
        confidence = self._calculate_confidence_score(agent_response)

        return {
            "success": True,
            "query": query,
            "findings": findings,
            "sources": sources,
            "confidence": confidence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": "agentic_research",
            "raw_response": agent_response
        }

    def _extract_findings_from_response(self, response: str) -> List[Dict[str, Any]]:
        """Extract research findings from agent response"""
        findings = []

        # Look for structured findings in the response
        lines = response.split('\n')
        current_finding = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Look for finding indicators
            if any(indicator in line.lower() for indicator in [
                'finding:', 'key insight:', 'important:', 'analysis:', 'result:'
            ]):
                if current_finding:
                    findings.append(current_finding)

                current_finding = {
                    "title": line,
                    "content": "",
                    "confidence": 0.8,
                    "source": "agent_analysis"
                }
            elif current_finding and line.startswith(('-', '*', '•')):
                current_finding["content"] += line + "\n"

        if current_finding:
            findings.append(current_finding)

        return findings[:5]  # Limit to top 5 findings

    def _extract_sources_from_response(self, response: str) -> List[str]:
        """Extract sources mentioned in the response"""
        sources = []
        lines = response.split('\n')

        for line in lines:
            line = line.strip()
            # Look for URLs or source indicators
            if ('http' in line.lower() or 'www.' in line.lower()) and len(line) > 10:
                sources.append(line)

        return sources[:10]  # Limit sources

    def _calculate_confidence_score(self, response: str) -> float:
        """Calculate confidence score based on response quality indicators"""
        confidence = 0.5  # Base confidence

        # Increase confidence based on response characteristics
        if len(response) > 500:
            confidence += 0.2
        if 'sources' in response.lower() or 'research' in response.lower():
            confidence += 0.1
        if any(word in response.lower() for word in ['analysis', 'findings', 'insights']):
            confidence += 0.1
        if len(response.split('\n')) > 10:  # Substantial response
            confidence += 0.1

        return min(confidence, 0.95)  # Cap at 95%

    async def quick_research(self, query: str, num_sources: int = 3) -> Dict[str, Any]:
        """Perform quick research using just web search and basic analysis"""
        logger.info("Starting quick research", query=query)

        try:
            if self.agent:
                # Use agent tools for quick research (new implementation to avoid circular imports)
                return await self._quick_research_with_agent_new(query, num_sources)
            else:
                # Use simplified approach
                return await self._perform_simplified_research(query, max_sources=num_sources)

        except Exception as e:
            logger.error("Quick research failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "query": query
            }

    async def _quick_research_with_agent_old(self, query: str, num_sources: int = 3) -> Dict[str, Any]:
        """Perform quick research using agent tools (old implementation)"""
        if not self.agent:
            return await self._perform_simplified_research(query, max_sources=num_sources)

        # Use web search tool directly
        search_result = await self.agent.execute_tool_by_name(
            "web_search",
            query=query,
            num_results=num_sources
        )

        if not search_result.get("success"):
            return {
                "success": False,
                "error": "Web search failed",
                "query": query
            }

        # Analyze the search results
        search_content = "\n".join([
            f"{result['title']}: {result.get('snippet', '')}"
            for result in search_result.get("results", [])
        ])

        analysis_result = await self.agent.execute_tool_by_name(
            "content_analyzer",
            content=search_content,
            analysis_type="insights"
        )

        return {
            "success": True,
            "query": query,
            "search_results": search_result,
            "analysis": analysis_result,
            "method": "quick_research"
        }

    # Backward compatibility methods
    async def research(self, query: str, **kwargs) -> ResearchBrief:
        """Backward compatibility method for existing integrations"""
        result = await self.perform_research(query, **kwargs)

        # Convert to ResearchBrief format
        from app.models import ResearchBrief

        brief = ResearchBrief(
            query=query,
            date=datetime.now(timezone.utc),
            entities=[],  # Could be extracted from results
            key_questions=[],  # Could be generated
            findings=[
                Finding(
                    title=finding.get("title", "Research Finding"),
                    summary=finding.get("content", ""),
                    key_insights=[finding.get("title", "")],
                    sources=["agent_research"],
                    confidence=finding.get("confidence", 0.8)
                )
                for finding in result.get("findings", [])
            ],
            ideas=[],  # Could be generated from findings
            executive_summary=result.get("raw_response", "")[:500] + "...",
            metadata={
                "method": "agentic_research",
                "confidence": result.get("confidence", 0.5),
                "sources_count": len(result.get("sources", []))
            }
        )

        return brief


# Backward compatibility alias
ResearchService = AgenticResearchService

