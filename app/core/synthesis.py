"""Synthesis and summarization of research findings"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from groq import AsyncGroq
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.models import Finding, Evidence, ResearchScope, Idea, RICEScore, ResearchBrief

logger = structlog.get_logger()


class ResearchSynthesizer:
    """Synthesize research findings and generate insights"""
    
    def __init__(self):
        self.client = AsyncGroq(api_key=settings.groq_api_key)
        self.model = settings.llm_model
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def synthesize_findings(
        self,
        sources: Dict[str, List[Dict[str, Any]]],
        query: str,
        key_questions: List[str]
    ) -> List[Finding]:
        """Synthesize findings from clustered sources"""
        
        findings = []
        
        for topic, topic_sources in sources.items():
            if not topic_sources:
                continue
            
            # Prepare context from sources
            context = self._prepare_context(topic_sources, limit=5)
            
            # Generate finding for this topic
            finding = await self._generate_finding(
                topic=topic,
                context=context,
                query=query,
                questions=key_questions
            )
            
            if finding:
                findings.append(finding)
        
        # Sort by confidence
        findings.sort(key=lambda f: f.confidence, reverse=True)
        
        return findings
    
    def _prepare_context(self, sources: List[Dict[str, Any]], limit: int = 5) -> str:
        """Prepare context from sources"""
        context_parts = []
        
        for source in sources[:limit]:
            if source.get('content'):
                # Take first 1000 chars
                content_snippet = source['content'][:1000]
                context_parts.append(f"""
Source: {source.get('title', 'Unknown')}
URL: {source.get('url', '')}
Date: {source.get('date', 'Unknown')}
Content: {content_snippet}...
---""")
        
        return "\n".join(context_parts)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _generate_finding(
        self,
        topic: str,
        context: str,
        query: str,
        questions: List[str]
    ) -> Optional[Finding]:
        """Generate a finding for a topic"""
        
        system_prompt = """You are a research analyst creating findings from sources.
        
        For the given topic and sources, create a finding that:
        1. Has a clear, descriptive title
        2. Provides a concise summary (2-3 sentences)
        3. Lists 3-5 key insights
        4. Assesses confidence level (0-1)
        5. Notes recency of information
        
        IMPORTANT: Every claim must be supported by the provided sources.
        
        Return ONLY valid JSON (no additional text):
        {
            "title": "...",
            "summary": "...",
            "key_insights": ["insight1", "insight2", ...],
            "confidence": 0.7,
            "recency": "last-6-months"
        }"""
        
        user_prompt = f"""
Topic: {topic}
Original Query: {query}
Key Questions: {', '.join(questions[:3])}

Sources:
{context}

Create a finding for this topic based on the sources."""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Map topic to scope
            scope_map = {
                'market': ResearchScope.MARKET,
                'competitors': ResearchScope.COMPETITORS,
                'pricing': ResearchScope.PRICING,
                'technology': ResearchScope.TECHNOLOGY,
                'regulation': ResearchScope.COMPLIANCE,
                'customers': ResearchScope.CUSTOMER
            }
            
            finding = Finding(
                title=result['title'],
                summary=result['summary'],
                evidence=[],  # Will be added separately
                confidence=result.get('confidence', 0.5),
                recency=result.get('recency', 'unknown'),
                scope=scope_map.get(topic, ResearchScope.MARKET),
                key_insights=result.get('key_insights', [])
            )
            
            return finding
            
        except Exception as e:
            logger.error("Failed to generate finding", topic=topic, error=str(e))
            return None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_ideas(
        self,
        findings: List[Finding],
        query: str,
        business_context: Optional[Dict[str, Any]] = None
    ) -> List[Idea]:
        """Generate actionable ideas from findings"""
        
        system_prompt = """You are a strategic advisor generating actionable ideas from research.
        
        Based on the findings, generate 3-5 actionable ideas that:
        1. Are specific and implementable
        2. Address the original query
        3. Have clear rationale based on findings
        4. Include RICE scoring estimates
        5. Note prerequisites and risks
        
        RICE Scoring:
        - Reach: Number of people/entities affected (integer)
        - Impact: 1 (minimal), 2 (moderate), 3 (massive)
        - Confidence: 0-1 (probability of success)
        - Effort: Person-days required (integer)
        
        Return ONLY valid JSON (no additional text):
        {
            "ideas": [
                {
                    "idea": "...",
                    "rationale": "...",
                    "reach": 1000,
                    "impact": 2,
                    "confidence": 0.7,
                    "effort": 20,
                    "prerequisites": ["..."],
                    "risks": ["..."]
                }
            ]
        }"""
        
        # Prepare findings summary
        findings_summary = "\n".join([
            f"- {f.title}: {f.summary} (confidence: {f.confidence})"
            for f in findings[:10]
        ])
        
        user_prompt = f"""
Original Query: {query}

Key Findings:
{findings_summary}

Business Context: {json.dumps(business_context) if business_context else 'General'}

Generate actionable ideas based on these findings."""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=settings.synthesis_temperature
            )
            
            result = json.loads(response.choices[0].message.content)
            
            ideas = []
            for idea_data in result.get('ideas', []):
                rice = RICEScore(
                    reach=idea_data.get('reach', 100),
                    impact=idea_data.get('impact', 2),
                    confidence=idea_data.get('confidence', 0.5),
                    effort=idea_data.get('effort', 10)
                )
                rice.calculate_score()
                
                idea = Idea(
                    idea=idea_data['idea'],
                    rationale=idea_data.get('rationale', ''),
                    rice=rice,
                    prerequisites=idea_data.get('prerequisites', []),
                    risks=idea_data.get('risks', []),
                    related_findings=[f.title for f in findings[:3]]
                )
                ideas.append(idea)
            
            # Sort by RICE score
            ideas.sort(key=lambda i: i.rice.score or 0, reverse=True)
            
            return ideas
            
        except Exception as e:
            logger.error("Failed to generate ideas", error=str(e))
            return []
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_executive_summary(
        self,
        findings: List[Finding],
        ideas: List[Idea],
        query: str
    ) -> str:
        """Generate executive summary of research"""
        
        system_prompt = """You are creating an executive summary of research findings.
        
        Write a concise 3-4 paragraph summary that:
        1. Answers the original query
        2. Highlights the most important findings
        3. Recommends top 2-3 actions
        4. Notes any significant risks or considerations
        
        Be direct, use data when available, and focus on actionable insights."""
        
        # Prepare context
        top_findings = "\n".join([
            f"- {f.title}: {f.summary}"
            for f in findings[:5]
        ])
        
        top_ideas = "\n".join([
            f"- {i.idea} (RICE score: {i.rice.score:.1f})"
            for i in ideas[:3]
        ])
        
        user_prompt = f"""
Query: {query}

Top Findings:
{top_findings}

Recommended Actions:
{top_ideas}

Write an executive summary."""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
                max_tokens=settings.synthesis_max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error("Failed to generate summary", error=str(e))
            return "Executive summary generation failed. Please review the findings and ideas above."
    
    def validate_citations(self, finding: Finding, sources: List[Dict[str, Any]]) -> bool:
        """Validate that findings have proper citations"""
        # Simple validation - check if evidence URLs exist in sources
        source_urls = {s.get('url') for s in sources if s.get('url')}
        
        for evidence in finding.evidence:
            if evidence.url not in source_urls:
                logger.warning("Invalid citation found", url=evidence.url)
                return False
        
        return True
    
    def check_hallucination(self, text: str, sources: List[Dict[str, Any]]) -> float:
        """Simple hallucination check - overlap between claims and sources"""
        # This is a simplified version - in production, use more sophisticated methods
        
        text_lower = text.lower()
        source_text = ' '.join([
            s.get('content', '')[:1000] for s in sources
        ]).lower()
        
        # Check for key terms overlap
        text_words = set(text_lower.split())
        source_words = set(source_text.split())
        
        if not text_words:
            return 0.0
        
        overlap = len(text_words & source_words) / len(text_words)
        return min(overlap, 1.0)
    
    def build_prompt(
        self,
        context: Dict[str, Any],
        intent: str,
        entities: Dict[str, Any],
        research_notes: Optional[str] = None
    ) -> str:
        """Build token-tight prompt scaffold for chat generation
        
        Structure:
        - SYSTEM: Role + safety
        - PROFILE: ≤ 8 short lines
        - RECENT SUMMARY: ≤ 120 tokens
        - LONG-TERM FACTS: 5-7 single-line bullets
        - CURRENT TASK: Intent + entities
        - RESEARCH NOTES: If available (≤ 150 tokens + citations)
        """
        sections = []
        
        # SYSTEM section
        sections.append("SYSTEM:")
        sections.append("- You are an intelligent assistant with research capabilities and long-term memory.")
        sections.append("- Be helpful, accurate, concise, and personalize using the provided profile and context.")
        sections.append("")
        
        # PROFILE section (≤ 8 lines)
        profile = context.get("profile", [])
        if profile:
            sections.append("PROFILE:")
            for item in profile[:8]:
                # Extract clean line from profile item
                if isinstance(item, dict):
                    line = item.get("memory") or item.get("content") or str(item)
                else:
                    line = str(item)
                # Truncate to keep it short
                if len(line) > 80:
                    line = line[:77] + "..."
                sections.append(f"- {line}")
            sections.append("")
        
        # RECENT SUMMARY section (≤ 120 tokens ~480 chars)
        summary = context.get("conversation_summary")
        if summary:
            sections.append("RECENT SUMMARY:")
            if len(summary) > 480:
                summary = summary[:477] + "..."
            sections.append(summary)
            sections.append("")
        
        # LONG-TERM FACTS section (5-7 bullets)
        memories = context.get("ranked_memories", [])
        if memories:
            sections.append("LONG-TERM FACTS:")
            for mem in memories[:7]:
                if isinstance(mem, dict):
                    fact = mem.get("memory") or mem.get("content") or str(mem)
                else:
                    fact = str(mem)
                # Single line bullet
                if len(fact) > 100:
                    fact = fact[:97] + "..."
                sections.append(f"- {fact}")
            sections.append("")
        
        # CURRENT TASK section
        sections.append("CURRENT TASK:")
        sections.append(f"- Intent: {intent}")
        if entities:
            # Format entities compactly
            entity_parts = []
            for key, value in entities.items():
                if value:
                    if isinstance(value, list):
                        entity_parts.append(f"{key}: {', '.join(str(v) for v in value[:3])}")
                    else:
                        entity_parts.append(f"{key}: {str(value)[:50]}")
            if entity_parts:
                sections.append(f"- Entities: [{', '.join(entity_parts)}]")
        sections.append("")
        
        # RESEARCH NOTES section (if available)
        if research_notes:
            sections.append("RESEARCH NOTES:")
            # Limit to ~150 tokens (600 chars) plus citations
            if len(research_notes) > 600:
                # Find a good break point
                cutoff = research_notes[:600].rfind(". ")
                if cutoff > 400:
                    research_notes = research_notes[:cutoff + 1]
                else:
                    research_notes = research_notes[:597] + "..."
            sections.append(research_notes)
            sections.append("")
        
        return "\n".join(sections)