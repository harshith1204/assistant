"""Web research pipeline for fetching and processing sources"""

import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import AsyncDDGS
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential
import trafilatura

from app.config import settings
from app.models import Evidence, SourceType

logger = structlog.get_logger()


class WebResearchPipeline:
    """Fetch, process, and extract information from web sources"""
    
    def __init__(self):
        self.session = None
        self.ddgs = AsyncDDGS()
        self.trusted_domains = {
            'wikipedia.org', 'reuters.com', 'bloomberg.com', 'wsj.com',
            'ft.com', 'economist.com', 'harvard.edu', 'mit.edu', 'stanford.edu',
            'mckinsey.com', 'bcg.com', 'gartner.com', 'forrester.com',
            'techcrunch.com', 'venturebeat.com', 'forbes.com', 'businessinsider.com'
        }
        self.blocked_domains = {
            'facebook.com', 'instagram.com', 'tiktok.com', 'pinterest.com'
        }
        
    async def __aenter__(self):
        self.session = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={'User-Agent': settings.user_agent}
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()
    
    async def search_sources(self, queries: List[str], max_results: int = 20) -> List[Dict[str, Any]]:
        """Search for sources using multiple queries"""
        all_results = []
        seen_urls = set()
        
        for query in queries:
            try:
                # Use DuckDuckGo search (no API key required)
                results = await self._search_ddg(query, max_results=10)
                
                # Deduplicate
                for result in results:
                    url = result.get('url', '')
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_results.append(result)
                        
                        if len(all_results) >= max_results:
                            break
                            
                if len(all_results) >= max_results:
                    break
                    
            except Exception as e:
                logger.warning("Search failed for query", query=query, error=str(e))
                continue
        
        logger.info("Search completed", total_results=len(all_results))
        return all_results[:max_results]
    
    async def _search_ddg(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo"""
        try:
            results = []
            async for result in self.ddgs.text(query, max_results=max_results):
                results.append({
                    'url': result.get('href', ''),
                    'title': result.get('title', ''),
                    'snippet': result.get('body', ''),
                    'source': 'duckduckgo'
                })
            return results
        except Exception as e:
            logger.error("DuckDuckGo search failed", query=query, error=str(e))
            return []
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def fetch_content(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetch and extract content from URL"""
        try:
            # Check domain
            domain = urlparse(url).netloc.lower()
            if any(blocked in domain for blocked in self.blocked_domains):
                logger.info("Skipping blocked domain", url=url)
                return None
            
            # Fetch page
            response = await self.session.get(url)
            response.raise_for_status()
            
            # Extract content using trafilatura
            content = trafilatura.extract(
                response.text,
                include_comments=False,
                include_tables=True,
                deduplicate=True
            )
            
            if not content:
                # Fallback to BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
                content = self._extract_with_bs4(soup)
            
            # Extract metadata
            metadata = trafilatura.extract_metadata(response.text)
            
            return {
                'url': url,
                'content': content,
                'title': metadata.title if metadata else None,
                'author': metadata.author if metadata else None,
                'date': metadata.date if metadata else None,
                'domain': domain,
                'fetched_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.warning("Failed to fetch content", url=url, error=str(e))
            return None
    
    def _extract_with_bs4(self, soup: BeautifulSoup) -> str:
        """Fallback content extraction with BeautifulSoup"""
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean up
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text[:10000]  # Limit length
    
    def calculate_credibility(self, source: Dict[str, Any]) -> float:
        """Calculate credibility score for a source"""
        score = 0.5  # Base score
        
        domain = source.get('domain', '')
        
        # Trusted domain bonus
        if any(trusted in domain for trusted in self.trusted_domains):
            score += 0.3
        
        # Has author
        if source.get('author'):
            score += 0.1
        
        # Has date
        if source.get('date'):
            score += 0.1
            
            # Recency bonus
            try:
                pub_date = datetime.fromisoformat(source['date'])
                age_days = (datetime.utcnow() - pub_date).days
                if age_days < 30:
                    score += 0.1
                elif age_days < 180:
                    score += 0.05
            except:
                pass
        
        # Content length (substantial content)
        content_len = len(source.get('content', ''))
        if content_len > 1000:
            score += 0.05
        if content_len > 5000:
            score += 0.05
        
        return min(score, 1.0)
    
    def deduplicate_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate or near-duplicate sources"""
        unique_sources = []
        seen_hashes = set()
        
        for source in sources:
            if not source or not source.get('content'):
                continue
            
            # Create content hash (first 1000 chars)
            content_sample = source['content'][:1000].lower()
            content_hash = hashlib.md5(content_sample.encode()).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_sources.append(source)
        
        return unique_sources
    
    def cluster_by_topic(self, sources: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Cluster sources by topic (simple keyword-based clustering)"""
        clusters = {
            'market': [],
            'competitors': [],
            'pricing': [],
            'technology': [],
            'regulation': [],
            'customers': [],
            'other': []
        }
        
        topic_keywords = {
            'market': ['market', 'TAM', 'growth', 'size', 'forecast', 'trend'],
            'competitors': ['competitor', 'rival', 'versus', 'comparison', 'alternative'],
            'pricing': ['price', 'pricing', 'cost', 'fee', 'subscription', 'revenue'],
            'technology': ['technology', 'tech', 'platform', 'software', 'AI', 'digital'],
            'regulation': ['regulation', 'compliance', 'law', 'legal', 'policy', 'government'],
            'customers': ['customer', 'client', 'user', 'buyer', 'segment', 'persona']
        }
        
        for source in sources:
            content_lower = (source.get('content', '') + source.get('title', '')).lower()
            
            assigned = False
            for topic, keywords in topic_keywords.items():
                if any(keyword in content_lower for keyword in keywords):
                    clusters[topic].append(source)
                    assigned = True
                    break
            
            if not assigned:
                clusters['other'].append(source)
        
        # Remove empty clusters
        return {k: v for k, v in clusters.items() if v}
    
    async def extract_facts(self, source: Dict[str, Any], query: str) -> List[Evidence]:
        """Extract facts and evidence from source"""
        if not source or not source.get('content'):
            return []
        
        evidence_list = []
        content = source['content']
        
        # Extract sentences with numbers, percentages, or key facts
        import re
        sentences = content.split('.')
        
        for sentence in sentences[:20]:  # Limit to first 20 sentences
            sentence = sentence.strip()
            
            # Look for sentences with data
            has_number = bool(re.search(r'\d+', sentence))
            has_percent = '%' in sentence
            has_currency = any(curr in sentence for curr in ['$', '€', '£', '₹'])
            
            if has_number or has_percent or has_currency:
                if len(sentence) > 30 and len(sentence) < 500:
                    evidence = Evidence(
                        quote=sentence,
                        url=source['url'],
                        title=source.get('title'),
                        author=source.get('author'),
                        published=datetime.fromisoformat(source['date']) if source.get('date') else None,
                        source_type=self._determine_source_type(source),
                        credibility_score=self.calculate_credibility(source)
                    )
                    evidence_list.append(evidence)
        
        return evidence_list[:5]  # Limit evidence per source
    
    def _determine_source_type(self, source: Dict[str, Any]) -> SourceType:
        """Determine the type of source"""
        domain = source.get('domain', '').lower()
        
        if any(news in domain for news in ['reuters', 'bloomberg', 'wsj', 'ft.com']):
            return SourceType.NEWS
        elif any(academic in domain for academic in ['.edu', 'scholar', 'journal']):
            return SourceType.ACADEMIC
        elif any(report in domain for report in ['mckinsey', 'bcg', 'gartner', 'forrester']):
            return SourceType.REPORT
        elif 'blog' in domain or 'medium.com' in domain:
            return SourceType.BLOG
        else:
            return SourceType.WEB