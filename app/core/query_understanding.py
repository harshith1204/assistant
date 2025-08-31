"""Query understanding and expansion using LLM"""

import json
from typing import List, Dict, Any, Optional
from groq import AsyncGroq
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.models import ResearchScope

logger = structlog.get_logger()


class QueryUnderstanding:
    """Parse and expand research queries using LLM"""
    
    def __init__(self):
        self.client = AsyncGroq(api_key=settings.groq_api_key)
        self.model = settings.llm_model
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def parse_query(self, query: str) -> Dict[str, Any]:
        """Parse query to extract key components"""
        
        system_prompt = """You are a research query analyzer. Extract key components from the query:
        1. Industry/domain
        2. Geographic focus
        3. Target audience/ICP
        4. Timeframe
        5. Specific constraints or requirements
        6. Research scope (market, competitors, pricing, channels, compliance, technology, customer)
        
        Return ONLY valid JSON with these fields:
        {
            "industry": "...",
            "geography": "...",
            "target_audience": "...",
            "timeframe": "...",
            "constraints": [...],
            "scope": [...],
            "entities": [...],
            "key_terms": [...]
        }
        
        Important: Return ONLY the JSON object, no additional text."""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Parse this research query: {query}"}
                ],
                temperature=0.3
            )
            
            parsed = json.loads(response.choices[0].message.content)
            logger.info("Query parsed successfully", query=query, parsed=parsed)
            return parsed
            
        except Exception as e:
            logger.error("Failed to parse query", query=query, error=str(e))
            return self._fallback_parse(query)
    
    def _fallback_parse(self, query: str) -> Dict[str, Any]:
        """Fallback parsing using simple heuristics"""
        return {
            "industry": None,
            "geography": None,
            "target_audience": None,
            "timeframe": None,
            "constraints": [],
            "scope": [ResearchScope.MARKET.value],
            "entities": query.split()[:5],
            "key_terms": query.split()[:10]
        }
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def expand_to_subqueries(self, query: str, parsed: Dict[str, Any]) -> List[str]:
        """Expand query into focused sub-queries"""
        
        system_prompt = """You are a research strategist. Given a main query and its components,
        generate 5-10 focused sub-queries that will help answer the main question comprehensively.
        
        Consider:
        - Market size and growth
        - Competition analysis
        - Customer segments and needs
        - Pricing and business models
        - Distribution channels
        - Regulatory environment
        - Technology trends
        - Success factors and risks
        
        Return ONLY valid JSON: {"subqueries": ["query1", "query2", ...]}
        No additional text, just the JSON object."""
        
        context = f"""
        Main query: {query}
        Industry: {parsed.get('industry', 'N/A')}
        Geography: {parsed.get('geography', 'N/A')}
        Target: {parsed.get('target_audience', 'N/A')}
        Timeframe: {parsed.get('timeframe', 'N/A')}
        Scope: {', '.join(parsed.get('scope', []))}
        """
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context}
                ],
                temperature=0.7
            )
            
            result = json.loads(response.choices[0].message.content)
            subqueries = result.get("subqueries", [])
            
            logger.info("Generated subqueries", count=len(subqueries))
            return subqueries[:10]  # Limit to 10
            
        except Exception as e:
            logger.error("Failed to expand queries", error=str(e))
            return self._fallback_subqueries(query, parsed)
    
    def _fallback_subqueries(self, query: str, parsed: Dict[str, Any]) -> List[str]:
        """Generate fallback subqueries"""
        base = query.replace("?", "")
        geo = parsed.get('geography', '')
        industry = parsed.get('industry', '')
        
        subqueries = [
            f"{base} market size {geo}",
            f"{base} top competitors {industry}",
            f"{base} pricing models",
            f"{base} customer acquisition channels",
            f"{base} regulatory compliance {geo}",
            f"{base} growth trends forecast",
            f"{base} success factors challenges"
        ]
        
        return [q for q in subqueries if q.strip()]
    
    async def generate_key_questions(self, query: str, parsed: Dict[str, Any]) -> List[str]:
        """Generate key questions to be answered"""
        
        scope_questions = {
            ResearchScope.MARKET: [
                "What is the total addressable market size?",
                "What is the expected growth rate?",
                "Who are the key market segments?"
            ],
            ResearchScope.COMPETITORS: [
                "Who are the top 10 competitors?",
                "What is their market positioning?",
                "What are their strengths and weaknesses?"
            ],
            ResearchScope.PRICING: [
                "What are the common pricing models?",
                "What is the average price point?",
                "How price-sensitive are customers?"
            ],
            ResearchScope.CHANNELS: [
                "What are the primary distribution channels?",
                "What is the customer acquisition cost by channel?",
                "Which channels have the highest conversion rates?"
            ],
            ResearchScope.COMPLIANCE: [
                "What are the key regulatory requirements?",
                "What licenses or certifications are needed?",
                "What are the compliance costs and timelines?"
            ]
        }
        
        questions = []
        for scope_str in parsed.get('scope', [ResearchScope.MARKET.value]):
            try:
                scope = ResearchScope(scope_str)
                questions.extend(scope_questions.get(scope, []))
            except ValueError:
                continue
        
        # Add custom questions based on the specific query
        if parsed.get('geography'):
            questions.append(f"What are the specific considerations for {parsed['geography']}?")
        
        if parsed.get('target_audience'):
            questions.append(f"What are the needs and pain points of {parsed['target_audience']}?")
        
        return questions[:15]  # Limit to 15 questions
    
    def sanitize_query(self, query: str) -> str:
        """Sanitize query to remove PII and normalize"""
        # Remove potential PII patterns
        import re
        
        # Remove email addresses
        query = re.sub(r'\S+@\S+', '[EMAIL]', query)
        
        # Remove phone numbers
        query = re.sub(r'\b\d{10,}\b', '[PHONE]', query)
        
        # Remove credit card patterns
        query = re.sub(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[CARD]', query)
        
        # Normalize whitespace
        query = ' '.join(query.split())
        
        return query