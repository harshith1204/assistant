"""Agentic synthesis and summarization of research findings"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from groq import AsyncGroq
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential
from dataclasses import dataclass, field
from enum import Enum

from app.config import settings
from app.models import Finding, Evidence, ResearchScope, Idea, RICEScore, ResearchBrief

logger = structlog.get_logger()


class SynthesisStrategy(Enum):
    """Different strategies for synthesizing information"""
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    TREND_ANALYSIS = "trend_analysis"
    ACTION_ORIENTED = "action_oriented"
    EXPLORATORY = "exploratory"


class SynthesisMode(Enum):
    """How to approach synthesis"""
    COMPREHENSIVE = "comprehensive"
    FOCUSED = "focused"
    CRITICAL = "critical"
    CREATIVE = "creative"


@dataclass
class SynthesisContext:
    """Context for synthesis operations"""
    user_query: str
    research_scope: ResearchScope
    available_findings: List[Finding]
    business_context: Optional[Dict[str, Any]] = None
    synthesis_strategy: SynthesisStrategy = SynthesisStrategy.ANALYTICAL
    synthesis_mode: SynthesisMode = SynthesisMode.COMPREHENSIVE
    confidence_threshold: float = 0.6
    max_findings_to_synthesize: int = 10
    reasoning_chain: List[str] = field(default_factory=list)


class AgenticResearchSynthesizer:
    """Agentic synthesizer that reasons about and dynamically synthesizes research findings"""

    def __init__(self):
        self.client = AsyncGroq(api_key=settings.groq_api_key)
        self.model = settings.llm_model

    async def synthesize_findings_agentic(
        self,
        sources: Dict[str, List[Dict[str, Any]]],
        query: str,
        key_questions: List[str],
        business_context: Optional[Dict[str, Any]] = None
    ) -> List[Finding]:
        """Agentically synthesize findings with reasoning and dynamic adaptation"""

        # Create synthesis context
        synthesis_context = SynthesisContext(
            user_query=query,
            research_scope=self._determine_research_scope(query),
            available_findings=[],  # Will be populated
            business_context=business_context,
            synthesis_strategy=self._choose_synthesis_strategy(query, key_questions),
            synthesis_mode=self._choose_synthesis_mode(query, len(sources))
        )

        # Phase 1: Analyze and reason about the synthesis approach
        await self._reason_about_synthesis_approach(synthesis_context, sources, key_questions)

        # Phase 2: Dynamic finding generation based on reasoning
        findings = await self._generate_findings_with_reasoning(synthesis_context, sources, key_questions)

        # Phase 3: Quality assessment and refinement
        findings = await self._assess_and_refine_findings(findings, synthesis_context)

        # Phase 4: Cross-validation and coherence check
        findings = await self._ensure_coherence(findings, synthesis_context)

        return findings

    async def _reason_about_synthesis_approach(
        self,
        context: SynthesisContext,
        sources: Dict[str, List[Dict[str, Any]]],
        key_questions: List[str]
    ):
        """Reason about the best approach to synthesis"""

        system_prompt = """You are an expert research synthesizer analyzing how to best approach synthesizing findings.

Consider:
1. The nature of the user's query
2. Available data sources and their quality
3. Key questions that need answers
4. Business context and objectives
5. Best synthesis strategy for the situation

Return ONLY valid JSON:
{
    "reasoning_steps": [
        "Step-by-step analysis of the synthesis approach"
    ],
    "recommended_strategy": "analytical|comparative|trend_analysis|action_oriented|exploratory",
    "recommended_mode": "comprehensive|focused|critical|creative",
    "confidence_threshold": 0.6,
    "key_focus_areas": ["area1", "area2"],
    "potential_challenges": ["challenge1", "challenge2"]
}"""

        source_summary = "\n".join([
            f"- {topic}: {len(topic_sources)} sources"
            for topic, topic_sources in sources.items()
        ])

        user_prompt = f"""Analyze this synthesis task:

Query: {context.user_query}
Key Questions: {', '.join(key_questions)}
Source Summary: {source_summary}
Business Context: {context.business_context or 'General research'}

Determine the optimal synthesis approach:"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )

            result = json.loads(response.choices[0].message.content)

            # Update context based on reasoning
            context.reasoning_chain.extend(result.get("reasoning_steps", []))
            context.synthesis_strategy = SynthesisStrategy(result.get("recommended_strategy", "analytical"))
            context.synthesis_mode = SynthesisMode(result.get("recommended_mode", "comprehensive"))
            context.confidence_threshold = result.get("confidence_threshold", 0.6)
            context.metadata = {
                "key_focus_areas": result.get("key_focus_areas", []),
                "potential_challenges": result.get("potential_challenges", [])
            }

        except Exception as e:
            logger.error("Failed to reason about synthesis approach", error=str(e))
            # Use defaults
            context.reasoning_chain.append("Using default analytical synthesis approach due to reasoning error")

    async def _generate_findings_with_reasoning(
        self,
        context: SynthesisContext,
        sources: Dict[str, List[Dict[str, Any]]],
        key_questions: List[str]
    ) -> List[Finding]:
        """Generate findings using dynamic, reasoning-based approach"""

        findings = []

        for topic, topic_sources in sources.items():
            if not topic_sources:
                continue

            # Dynamic context preparation based on synthesis strategy
            context_window = self._determine_context_window(context, len(topic_sources))
            prepared_context = self._prepare_context_adaptive(topic_sources, context_window, context)

            # Generate finding with strategy-specific prompting
            finding = await self._generate_finding_with_strategy(
                topic=topic,
                context=prepared_context,
                synthesis_context=context,
                questions=key_questions
            )

            if finding:
                findings.append(finding)

        return findings

    def _determine_context_window(self, context: SynthesisContext, source_count: int) -> int:
        """Determine how many sources to include in context based on strategy"""
        if context.synthesis_mode == SynthesisMode.FOCUSED:
            return min(3, source_count)
        elif context.synthesis_mode == SynthesisMode.COMPREHENSIVE:
            return min(8, source_count)
        else:
            return min(5, source_count)

    def _prepare_context_adaptive(
        self,
        sources: List[Dict[str, Any]],
        limit: int,
        context: SynthesisContext
    ) -> str:
        """Prepare context adaptively based on synthesis strategy"""

        # Sort sources by relevance if we have quality scores
        sorted_sources = sorted(
            sources,
            key=lambda s: s.get('relevance_score', 0.5),
            reverse=True
        )

        context_parts = []

        for source in sorted_sources[:limit]:
            if source.get('content'):
                # Adaptive content extraction based on strategy
                content_snippet = self._extract_content_for_strategy(
                    source['content'],
                    context.synthesis_strategy
                )

                context_parts.append(f"""
Source: {source.get('title', 'Unknown')}
URL: {source.get('url', '')}
Date: {source.get('date', 'Unknown')}
Relevance: {source.get('relevance_score', 'N/A')}
Content: {content_snippet}...
---""")

        return "\n".join(context_parts)

    def _extract_content_for_strategy(self, content: str, strategy: SynthesisStrategy) -> str:
        """Extract content based on synthesis strategy"""
        content_length = len(content)

        if strategy == SynthesisStrategy.FOCUSED:
            # Extract key facts and conclusions
            return content[:800] + "..." if content_length > 800 else content
        elif strategy == SynthesisStrategy.ANALYTICAL:
            # Extract detailed analysis sections
            return content[:1200] + "..." if content_length > 1200 else content
        elif strategy == SynthesisStrategy.COMPARATIVE:
            # Extract comparison-relevant content
            return content[:1000] + "..." if content_length > 1000 else content
        else:
            # Default extraction
            return content[:1000] + "..." if content_length > 1000 else content
    
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
    async def _generate_finding_with_strategy(
        self,
        topic: str,
        context: str,
        synthesis_context: SynthesisContext,
        questions: List[str]
    ) -> Optional[Finding]:
        """Generate a finding using strategy-specific reasoning"""

        # Dynamic system prompt based on synthesis strategy
        system_prompt = self._create_strategy_specific_prompt(synthesis_context)

        # Build user prompt with reasoning context
        user_prompt = self._build_reasoning_user_prompt(
            topic, context, synthesis_context, questions
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self._get_temperature_for_strategy(synthesis_context.synthesis_strategy)
            )

            result = json.loads(response.choices[0].message.content)

            # Enhanced finding creation with strategy-specific processing
            finding = await self._create_finding_with_enhanced_processing(
                result, topic, synthesis_context
            )

            return finding

        except Exception as e:
            logger.error("Failed to generate finding with strategy", topic=topic, error=str(e))
            return None

    def _create_strategy_specific_prompt(self, context: SynthesisContext) -> str:
        """Create strategy-specific system prompt"""

        base_prompt = """You are an expert research analyst creating findings from sources."""

        strategy_instructions = {
            SynthesisStrategy.ANALYTICAL: """
Focus on:
- Detailed analysis and breakdown of information
- Evidence-based reasoning with citations
- Identification of patterns and relationships
- Critical evaluation of source quality""",

            SynthesisStrategy.COMPARATIVE: """
Focus on:
- Comparing different perspectives and sources
- Identifying consensus vs. conflicting information
- Evaluating relative strengths of different approaches
- Highlighting trade-offs and alternatives""",

            SynthesisStrategy.TREND_ANALYSIS: """
Focus on:
- Temporal patterns and changes over time
- Emerging trends and their trajectories
- Historical context and evolution
- Future projections based on current data""",

            SynthesisStrategy.ACTION_ORIENTED: """
Focus on:
- Practical implications and applications
- Actionable recommendations
- Implementation considerations
- Risk assessment and mitigation""",

            SynthesisStrategy.EXPLORATORY: """
Focus on:
- Novel insights and unexpected findings
- Generating new hypotheses
- Identifying knowledge gaps
- Suggesting areas for further investigation"""
        }

        mode_instructions = {
            SynthesisMode.COMPREHENSIVE: "Provide thorough analysis covering all aspects",
            SynthesisMode.FOCUSED: "Focus on the most important and relevant information",
            SynthesisMode.CRITICAL: "Emphasize critical evaluation and potential limitations",
            SynthesisMode.CREATIVE: "Generate innovative insights and connections"
        }

        return f"""{base_prompt}

{strategy_instructions.get(context.synthesis_strategy, "")}

{mode_instructions.get(context.synthesis_mode, "")}

Requirements:
1. Title should be clear and descriptive
2. Summary should be {self._get_summary_length_for_mode(context.synthesis_mode)}
3. Key insights should be {self._get_insight_count_for_strategy(context.synthesis_strategy)}
4. Confidence assessment must consider source quality and consistency
5. All claims must be supported by evidence

Return ONLY valid JSON (no additional text):
{{
    "title": "...",
    "summary": "...",
    "key_insights": ["insight1", "insight2", ...],
    "confidence": 0.7,
    "recency": "last-6-months",
    "evidence_quality": "high|medium|low",
    "gaps_identified": ["gap1", "gap2"]
}}"""

    def _get_summary_length_for_mode(self, mode: SynthesisMode) -> str:
        """Get appropriate summary length for synthesis mode"""
        lengths = {
            SynthesisMode.COMPREHENSIVE: "3-4 sentences with detailed analysis",
            SynthesisMode.FOCUSED: "2-3 sentences focusing on key points",
            SynthesisMode.CRITICAL: "2-3 sentences with critical evaluation",
            SynthesisMode.CREATIVE: "2-3 sentences with innovative insights"
        }
        return lengths.get(mode, "2-3 sentences")

    def _get_insight_count_for_strategy(self, strategy: SynthesisStrategy) -> str:
        """Get appropriate insight count for synthesis strategy"""
        counts = {
            SynthesisStrategy.ANALYTICAL: "4-6 detailed insights",
            SynthesisStrategy.COMPARATIVE: "3-5 comparative insights",
            SynthesisStrategy.TREND_ANALYSIS: "3-4 trend-based insights",
            SynthesisStrategy.ACTION_ORIENTED: "3-5 actionable insights",
            SynthesisStrategy.EXPLORATORY: "4-6 exploratory insights"
        }
        return counts.get(strategy, "3-5 key insights")

    def _build_reasoning_user_prompt(
        self,
        topic: str,
        context: str,
        synthesis_context: SynthesisContext,
        questions: List[str]
    ) -> str:
        """Build user prompt with reasoning context"""

        reasoning_context = "\n".join([
            f"- {step}" for step in synthesis_context.reasoning_chain[-3:]
        ])

        return f"""
Topic: {topic}
Original Query: {synthesis_context.user_query}
Key Questions: {', '.join(questions[:3])}
Synthesis Strategy: {synthesis_context.synthesis_strategy.value}
Synthesis Mode: {synthesis_context.synthesis_mode.value}

Reasoning Context:
{reasoning_context}

Key Focus Areas: {', '.join(synthesis_context.metadata.get('key_focus_areas', []))}
Potential Challenges: {', '.join(synthesis_context.metadata.get('potential_challenges', []))}

Sources:
{context}

Create a finding for this topic using the specified synthesis approach."""

    def _get_temperature_for_strategy(self, strategy: SynthesisStrategy) -> float:
        """Get appropriate temperature for synthesis strategy"""
        temperatures = {
            SynthesisStrategy.ANALYTICAL: 0.2,
            SynthesisStrategy.COMPARATIVE: 0.3,
            SynthesisStrategy.TREND_ANALYSIS: 0.4,
            SynthesisStrategy.ACTION_ORIENTED: 0.3,
            SynthesisStrategy.EXPLORATORY: 0.6
        }
        return temperatures.get(strategy, 0.3)

    async def _create_finding_with_enhanced_processing(
        self,
        result: Dict[str, Any],
        topic: str,
        context: SynthesisContext
    ) -> Finding:
        """Create finding with enhanced processing based on strategy"""

        # Enhanced confidence calculation
        base_confidence = result.get('confidence', 0.5)
        enhanced_confidence = self._enhance_confidence_score(
            base_confidence,
            result.get('evidence_quality', 'medium'),
            context
        )

        # Strategy-specific scope mapping
        scope = self._map_topic_to_scope_with_strategy(topic, context)

        # Enhanced key insights processing
        key_insights = self._process_key_insights_for_strategy(
            result.get('key_insights', []),
            context.synthesis_strategy
        )

        finding = Finding(
            title=result['title'],
            summary=result['summary'],
            evidence=[],  # Will be populated separately
            confidence=enhanced_confidence,
            recency=result.get('recency', 'unknown'),
            scope=scope,
            key_insights=key_insights,
            metadata={
                "synthesis_strategy": context.synthesis_strategy.value,
                "evidence_quality": result.get('evidence_quality', 'medium'),
                "gaps_identified": result.get('gaps_identified', []),
                "reasoning_steps": context.reasoning_chain
            }
        )

        return finding

    def _enhance_confidence_score(
        self,
        base_confidence: float,
        evidence_quality: str,
        context: SynthesisContext
    ) -> float:
        """Enhance confidence score based on evidence quality and context"""

        quality_multipliers = {
            'high': 1.0,
            'medium': 0.9,
            'low': 0.7
        }

        # Apply evidence quality multiplier
        multiplier = quality_multipliers.get(evidence_quality, 0.8)

        # Apply strategy-specific adjustments
        if context.synthesis_strategy == SynthesisStrategy.EXPLORATORY:
            multiplier *= 0.9  # Slightly reduce for exploratory findings
        elif context.synthesis_strategy == SynthesisStrategy.ANALYTICAL:
            multiplier *= 1.1  # Increase for analytical depth

        enhanced = base_confidence * multiplier

        # Ensure reasonable bounds
        return max(0.1, min(0.95, enhanced))

    def _map_topic_to_scope_with_strategy(self, topic: str, context: SynthesisContext) -> ResearchScope:
        """Map topic to research scope considering synthesis strategy"""

        base_scope_map = {
            'market': ResearchScope.MARKET,
            'competitors': ResearchScope.COMPETITORS,
            'pricing': ResearchScope.PRICING,
            'technology': ResearchScope.TECHNOLOGY,
            'regulation': ResearchScope.COMPLIANCE,
            'customers': ResearchScope.CUSTOMER
        }

        # Strategy-specific scope adjustments
        if context.synthesis_strategy == SynthesisStrategy.TREND_ANALYSIS:
            # Trends might span multiple scopes
            if 'market' in topic.lower() or 'industry' in topic.lower():
                return ResearchScope.MARKET
        elif context.synthesis_strategy == SynthesisStrategy.ACTION_ORIENTED:
            # Action-oriented findings often relate to business operations
            if any(word in topic.lower() for word in ['pricing', 'strategy', 'implementation']):
                return ResearchScope.PRICING

        return base_scope_map.get(topic.lower(), ResearchScope.MARKET)

    def _process_key_insights_for_strategy(
        self,
        insights: List[str],
        strategy: SynthesisStrategy
    ) -> List[str]:
        """Process key insights based on synthesis strategy"""

        if not insights:
            return []

        # Strategy-specific insight enhancement
        if strategy == SynthesisStrategy.ACTION_ORIENTED:
            # Ensure insights are actionable
            processed_insights = []
            for insight in insights:
                if not any(word in insight.lower() for word in ['should', 'could', 'can', 'recommend']):
                    insight = f"Action: {insight}"
                processed_insights.append(insight)
            return processed_insights

        elif strategy == SynthesisStrategy.COMPARATIVE:
            # Ensure comparative language
            processed_insights = []
            for insight in insights:
                if not any(word in insight.lower() for word in ['versus', 'compared to', 'while', 'whereas']):
                    insight = f"Comparison: {insight}"
                processed_insights.append(insight)
            return processed_insights

        return insights

    def _determine_research_scope(self, query: str) -> ResearchScope:
        """Determine research scope from query using reasoning"""
        query_lower = query.lower()

        if any(word in query_lower for word in ['market', 'industry', 'trend']):
            return ResearchScope.MARKET
        elif any(word in query_lower for word in ['competitor', 'competition', 'rival']):
            return ResearchScope.COMPETITORS
        elif any(word in query_lower for word in ['price', 'pricing', 'cost']):
            return ResearchScope.PRICING
        elif any(word in query_lower for word in ['technology', 'tech', 'software', 'platform']):
            return ResearchScope.TECHNOLOGY
        elif any(word in query_lower for word in ['regulation', 'compliance', 'legal', 'law']):
            return ResearchScope.COMPLIANCE
        elif any(word in query_lower for word in ['customer', 'user', 'consumer']):
            return ResearchScope.CUSTOMER

        return ResearchScope.MARKET  # Default

    def _choose_synthesis_strategy(self, query: str, key_questions: List[str]) -> SynthesisStrategy:
        """Choose synthesis strategy based on query and questions"""
        query_lower = query.lower()
        all_text = query_lower + ' ' + ' '.join(key_questions).lower()

        if any(word in all_text for word in ['compare', 'versus', 'alternative', 'option']):
            return SynthesisStrategy.COMPARATIVE
        elif any(word in all_text for word in ['trend', 'evolution', 'change', 'future']):
            return SynthesisStrategy.TREND_ANALYSIS
        elif any(word in all_text for word in ['action', 'recommend', 'implement', 'strategy']):
            return SynthesisStrategy.ACTION_ORIENTED
        elif any(word in all_text for word in ['explore', 'discover', 'novel', 'new']):
            return SynthesisStrategy.EXPLORATORY

        return SynthesisStrategy.ANALYTICAL  # Default

    def _choose_synthesis_mode(self, query: str, source_count: int) -> SynthesisMode:
        """Choose synthesis mode based on query complexity and source availability"""
        query_words = len(query.split())

        if query_words < 10 and source_count < 5:
            return SynthesisMode.FOCUSED
        elif any(word in query.lower() for word in ['analyze', 'evaluate', 'assess']):
            return SynthesisMode.CRITICAL
        elif any(word in query.lower() for word in ['creative', 'innovative', 'new']):
            return SynthesisMode.CREATIVE

        return SynthesisMode.COMPREHENSIVE  # Default

    async def _assess_and_refine_findings(
        self,
        findings: List[Finding],
        context: SynthesisContext
    ) -> List[Finding]:
        """Assess and refine findings for quality and coherence"""

        if not findings:
            return findings

        # Filter by confidence threshold
        filtered_findings = [
            f for f in findings
            if f.confidence >= context.confidence_threshold
        ]

        # Sort by enhanced criteria
        filtered_findings.sort(key=lambda f: self._calculate_finding_score(f, context), reverse=True)

        # Limit to max findings
        return filtered_findings[:context.max_findings_to_synthesize]

    def _calculate_finding_score(self, finding: Finding, context: SynthesisContext) -> float:
        """Calculate overall finding score for ranking"""
        base_score = finding.confidence

        # Bonus for strategy alignment
        strategy_bonus = 0.1 if finding.metadata.get('synthesis_strategy') == context.synthesis_strategy.value else 0

        # Bonus for evidence quality
        evidence_bonus = 0.05 if finding.metadata.get('evidence_quality') == 'high' else 0

        return base_score + strategy_bonus + evidence_bonus

    async def _ensure_coherence(self, findings: List[Finding], context: SynthesisContext) -> List[Finding]:
        """Ensure coherence across findings"""

        if len(findings) <= 1:
            return findings

        # Check for redundant findings and merge if needed
        coherent_findings = []
        seen_titles = set()

        for finding in findings:
            # Simple deduplication based on title similarity
            title_key = finding.title.lower()[:50]  # First 50 chars as key

            if title_key not in seen_titles:
                seen_titles.add(title_key)
                coherent_findings.append(finding)
            else:
                # Merge with existing finding
                existing = next(f for f in coherent_findings if f.title.lower()[:50] == title_key)
                await self._merge_findings(existing, finding)

        return coherent_findings

    async def _merge_findings(self, existing: Finding, new_finding: Finding):
        """Merge two similar findings"""
        # Combine summaries
        existing.summary = f"{existing.summary}\n\nAdditionally: {new_finding.summary}"

        # Combine key insights (avoid duplicates)
        existing_insights = set(existing.key_insights)
        new_insights = [insight for insight in new_finding.key_insights if insight not in existing_insights]
        existing.key_insights.extend(new_insights)

        # Take higher confidence
        existing.confidence = max(existing.confidence, new_finding.confidence)

        # Update metadata
        existing.metadata['merged'] = True

    # Legacy method for backward compatibility
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def synthesize_findings(
        self,
        sources: Dict[str, List[Dict[str, Any]]],
        query: str,
        key_questions: List[str]
    ) -> List[Finding]:
        """Legacy method for backward compatibility - redirects to agentic version"""
        return await self.synthesize_findings_agentic(sources, query, key_questions)

  