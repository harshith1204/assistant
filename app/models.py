"""Data models for the Research Engine"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, UUID4
import uuid


class SourceType(str, Enum):
    """Types of sources for research"""
    WEB = "web"
    NEWS = "news"
    ACADEMIC = "academic"
    SOCIAL = "social"
    REPORT = "report"
    BLOG = "blog"


class ConfidenceLevel(str, Enum):
    """Confidence levels for findings"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ResearchScope(str, Enum):
    """Scope of research"""
    MARKET = "market"
    COMPETITORS = "competitors"
    PRICING = "pricing"
    CHANNELS = "channels"
    COMPLIANCE = "compliance"
    TECHNOLOGY = "technology"
    CUSTOMER = "customer"


class Evidence(BaseModel):
    """Evidence supporting a finding"""
    quote: str = Field(..., description="Direct quote or key information")
    url: str = Field(..., description="Source URL")
    title: Optional[str] = Field(None, description="Source title")
    author: Optional[str] = Field(None, description="Author name")
    published: Optional[datetime] = Field(None, description="Publication date")
    source_type: SourceType = Field(SourceType.WEB, description="Type of source")
    credibility_score: float = Field(0.5, ge=0, le=1, json_schema_extra={"description": "Credibility score"})


class Finding(BaseModel):
    """A research finding with supporting evidence"""
    title: str = Field(..., description="Finding title")
    summary: str = Field(..., description="Finding summary")
    evidence: List[Evidence] = Field([], description="Supporting evidence")
    confidence: float = Field(0.5, ge=0, le=1, json_schema_extra={"description": "Confidence score"})
    recency: str = Field("unknown", description="Recency indicator")
    scope: ResearchScope = Field(ResearchScope.MARKET, description="Research scope")
    key_insights: List[str] = Field([], description="Key insights")


class RICEScore(BaseModel):
    """RICE scoring for ideas"""
    reach: int = Field(..., ge=0, json_schema_extra={"description": "Number of people/entities affected"})
    impact: int = Field(..., ge=1, le=3, json_schema_extra={"description": "Impact level (1-3)"})
    confidence: float = Field(..., ge=0, le=1, json_schema_extra={"description": "Confidence in estimates"})
    effort: int = Field(..., ge=1, json_schema_extra={"description": "Effort in person-days"})
    score: Optional[float] = Field(None, description="Calculated RICE score")
    
    def calculate_score(self) -> float:
        """Calculate RICE score"""
        self.score = (self.reach * self.impact * self.confidence) / self.effort
        return self.score


class Idea(BaseModel):
    """An actionable idea from research"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    idea: str = Field(..., description="Idea description")
    rationale: str = Field(..., description="Why this idea makes sense")
    rice: RICEScore = Field(..., description="RICE score")
    prerequisites: List[str] = Field([], description="Prerequisites")
    risks: List[str] = Field([], description="Potential risks")
    related_findings: List[str] = Field([], description="Related finding IDs")


class ResearchBrief(BaseModel):
    """Complete research brief"""
    brief_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str = Field(..., description="Original query")
    date: datetime = Field(default_factory=datetime.utcnow)
    entities: List[str] = Field([], description="Identified entities")
    key_questions: List[str] = Field([], description="Key questions addressed")
    findings: List[Finding] = Field([], description="Research findings")
    ideas: List[Idea] = Field([], description="Actionable ideas")
    attachments: List[Dict[str, str]] = Field([], description="File attachments")
    metadata: Dict[str, Any] = Field({}, description="Additional metadata")
    executive_summary: Optional[str] = Field(None, description="Executive summary")
    
    @property
    def total_sources(self) -> int:
        """Total number of unique sources"""
        sources = set()
        for finding in self.findings:
            for evidence in finding.evidence:
                sources.add(evidence.url)
        return len(sources)
    
    @property
    def average_confidence(self) -> float:
        """Average confidence across findings"""
        if not self.findings:
            return 0.0
        return sum(f.confidence for f in self.findings) / len(self.findings)


class ResearchRequest(BaseModel):
    """Request to run research"""
    query: str = Field(..., description="Research query")
    scope: Optional[List[ResearchScope]] = Field(None, description="Research scope")
    geo: Optional[str] = Field(None, description="Geographic focus")
    industry: Optional[str] = Field(None, description="Industry focus")
    timeframe: Optional[str] = Field(None, description="Timeframe")
    freshness: Optional[int] = Field(None, description="Max age in days")
    max_sources: Optional[int] = Field(20, description="Maximum sources to fetch")
    deep_dive: bool = Field(False, description="Enable deep dive mode")


class SaveRequest(BaseModel):
    """Request to save research to CRM/PMS"""
    brief_id: str = Field(..., description="Research brief ID")
    crm_ref: Optional[Dict[str, str]] = Field(None, description="CRM reference")
    pms_ref: Optional[Dict[str, str]] = Field(None, description="PMS reference")
    attachments: List[str] = Field([], description="Attachment paths")
    create_tasks: bool = Field(False, description="Create tasks from ideas")


class PlanRequest(BaseModel):
    """Request to convert ideas to plan"""
    brief_id: str = Field(..., description="Research brief ID")
    selected_ideas: List[str] = Field(..., description="Selected idea IDs")
    timeline_weeks: int = Field(12, description="Timeline in weeks")
    team_size: Optional[int] = Field(None, description="Team size")
    budget: Optional[float] = Field(None, description="Budget constraint")


class ResearchStatus(BaseModel):
    """Status of research operation"""
    status: str = Field(..., description="Status")
    progress: float = Field(0, ge=0, le=100, json_schema_extra={"description": "Progress percentage"})
    current_step: str = Field("", description="Current step")
    errors: List[str] = Field([], description="Errors encountered")
    warnings: List[str] = Field([], description="Warnings")


class SubscriptionRequest(BaseModel):
    """Request to subscribe to research updates"""
    query: str = Field(..., description="Research query")
    cadence: str = Field("weekly", description="Update cadence")
    scope: Optional[List[ResearchScope]] = Field(None, description="Research scope")
    notify_email: Optional[str] = Field(None, description="Notification email")
    notify_webhook: Optional[str] = Field(None, description="Webhook URL")