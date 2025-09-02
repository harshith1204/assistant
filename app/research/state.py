"""Unified state definitions and data structures for the Deep Research agent."""

import operator
from typing import Annotated, Optional, List, Dict, Any, Literal
from datetime import datetime, timezone
from enum import Enum

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# Import core models for business extensions
from app.models import ResearchBrief, Finding, Idea, RICEScore, Evidence


###################
# Structured Outputs
###################
class ConductResearch(BaseModel):
    """Call this tool to conduct research on a specific topic."""
    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
    )

class ResearchComplete(BaseModel):
    """Call this tool to indicate that the research is complete."""

class Summary(BaseModel):
    """Research summary with key findings."""

    summary: str
    key_excerpts: str



class ResearchQuestion(BaseModel):
    """Research question and brief for guiding research."""

    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )


###################
# Business Research Types and Enums
###################
class BusinessResearchType(str, Enum):
    """Types of business research supported"""
    MARKET_ANALYSIS = "market_analysis"
    COMPETITOR_ANALYSIS = "competitor_analysis"
    INDUSTRY_REPORT = "industry_report"
    FEASIBILITY_STUDY = "feasibility_study"
    STRATEGIC_PLANNING = "strategic_planning"
    PRODUCT_DEVELOPMENT = "product_development"
    MARKET_ENTRY = "market_entry"
    INVESTMENT_ANALYSIS = "investment_analysis"

class BusinessResearchScope(str, Enum):
    """Business research scope options"""
    LOCAL = "local"
    REGIONAL = "regional"
    NATIONAL = "national"
    GLOBAL = "global"
    INDUSTRY_SPECIFIC = "industry_specific"
    CROSS_INDUSTRY = "cross_industry"


###################
# Business-Specific Extensions
###################
class DeepResearchMetadata(BaseModel):
    """Metadata specific to deep research execution"""
    research_method: str = "deep_research_agent"
    model_used: str
    search_api: str
    total_iterations: int = 0
    research_duration_seconds: float = 0.0
    concurrent_researchers_used: int = 0
    total_tool_calls: int = 0
    sources_discovered: int = 0
    clarification_requests: int = 0
    supervisor_iterations: int = 0
    research_subtopics: List[str] = []
    quality_metrics: Dict[str, Any] = {}


###################
# Enhanced Business Models
###################
class BusinessResearchBrief(ResearchBrief):
    """Enhanced research brief with deep research capabilities"""

    # Deep research metadata
    deep_research_metadata: Optional[DeepResearchMetadata] = None

    # Business-specific enhancements
    business_type: Optional[BusinessResearchType] = None
    strategic_priorities: List[str] = []
    risk_assessment: Dict[str, Any] = {}
    financial_projections: Dict[str, Any] = {}
    implementation_roadmap: List[Dict[str, Any]] = []

    # Integration fields
    crm_integration_status: Optional[str] = None
    pms_integration_status: Optional[str] = None
    conversation_context_id: Optional[str] = None

    # Quality and validation
    research_quality_score: float = Field(0.0, ge=0.0, le=1.0)
    validation_status: str = "pending"  # pending, validated, needs_review
    reviewer_notes: Optional[str] = None

    def add_deep_research_metadata(self, metadata: DeepResearchMetadata):
        """Add deep research execution metadata"""
        self.deep_research_metadata = metadata

    def update_business_context(
        self,
        business_type: BusinessResearchType = None,
        strategic_priorities: List[str] = None,
        risk_assessment: Dict[str, Any] = None
    ):
        """Update business-specific context"""
        if business_type:
            self.business_type = business_type
        if strategic_priorities:
            self.strategic_priorities = strategic_priorities
        if risk_assessment:
            self.risk_assessment = risk_assessment

    def calculate_research_quality_score(self) -> float:
        """Calculate overall research quality score"""
        score_components = []

        # Findings quality (30%)
        if self.findings:
            avg_confidence = sum(f.confidence for f in self.findings) / len(self.findings)
            evidence_per_finding = sum(len(f.evidence) for f in self.findings) / len(self.findings)
            score_components.append(min(1.0, (avg_confidence + evidence_per_finding / 5) / 2) * 0.3)

        # Ideas quality (25%)
        if self.ideas:
            avg_rice_score = sum(i.rice.score or 0 for i in self.ideas if i.rice.score) / len(self.ideas)
            score_components.append(min(1.0, avg_rice_score / 100) * 0.25)

        # Source diversity (20%)
        if self.total_sources > 0:
            source_score = min(1.0, self.total_sources / 20)  # Max score at 20 sources
            score_components.append(source_score * 0.2)

        # Deep research bonus (15%)
        if self.deep_research_metadata:
            deep_score = min(1.0, self.deep_research_metadata.total_iterations / 10)
            score_components.append(deep_score * 0.15)

        # Completeness bonus (10%)
        completeness_factors = [
            bool(self.executive_summary),
            bool(self.key_questions),
            len(self.findings) > 0,
            len(self.ideas) > 0,
            bool(self.metadata)
        ]
        completeness_score = sum(completeness_factors) / len(completeness_factors)
        score_components.append(completeness_score * 0.1)

        self.research_quality_score = sum(score_components)
        return self.research_quality_score


class BusinessIdea(Idea):
    """Enhanced business idea with additional strategic context"""

    # Business-specific fields
    target_market_size: Optional[int] = None
    market_penetration_potential: float = Field(0.0, ge=0.0, le=1.0)
    competitive_advantage: Optional[str] = None
    regulatory_requirements: List[str] = []
    partnership_opportunities: List[str] = []

    # Strategic alignment
    strategic_alignment_score: float = Field(0.0, ge=0.0, le=5.0)
    risk_level: Literal["low", "medium", "high"] = "medium"
    time_to_market_months: Optional[int] = None

    # Financial enhancements
    projected_revenue_year1: Optional[float] = None
    projected_revenue_year3: Optional[float] = None
    customer_acquisition_cost: Optional[float] = None
    lifetime_value: Optional[float] = None

    def calculate_enhanced_rice_score(self) -> float:
        """Calculate enhanced RICE score with business-specific factors"""
        if not self.rice.score:
            return 0.0

        # Base RICE score
        base_score = self.rice.score

        # Business multipliers
        market_multiplier = min(2.0, 1.0 + (self.market_penetration_potential * 0.5))
        strategic_multiplier = min(1.5, 1.0 + (self.strategic_alignment_score / 10))

        # Risk adjustment
        risk_multiplier = {"low": 1.2, "medium": 1.0, "high": 0.8}[self.risk_level]

        enhanced_score = base_score * market_multiplier * strategic_multiplier * risk_multiplier
        self.rice.score = enhanced_score
        return enhanced_score


class BusinessFinding(Finding):
    """Enhanced finding with business-specific context"""

    # Business context
    business_impact: Literal["low", "medium", "high", "critical"] = "medium"
    strategic_importance: float = Field(0.0, ge=0.0, le=5.0)
    action_required: bool = False
    action_deadline: Optional[datetime] = None

    # Market context
    market_segment: Optional[str] = None
    competitive_implications: Optional[str] = None
    regulatory_implications: Optional[str] = None

    # Financial context
    revenue_impact: Optional[float] = None
    cost_impact: Optional[float] = None

    def assess_business_impact(self) -> str:
        """Assess the business impact of this finding"""
        impact_assessment = f"**Business Impact Assessment:** {self.business_impact.upper()}\n"

        if self.revenue_impact:
            impact_assessment += f"**Revenue Impact:** ${self.revenue_impact:,.0f}\n"

        if self.cost_impact:
            impact_assessment += f"**Cost Impact:** ${self.cost_impact:,.0f}\n"

        if self.competitive_implications:
            impact_assessment += f"**Competitive Implications:** {self.competitive_implications}\n"

        if self.regulatory_implications:
            impact_assessment += f"**Regulatory Implications:** {self.regulatory_implications}\n"

        if self.action_required:
            deadline_str = f" by {self.action_deadline.strftime('%Y-%m-%d')}" if self.action_deadline else ""
            impact_assessment += f"**Action Required:** Yes{deadline_str}\n"

        return impact_assessment


class BusinessResearchState(BaseModel):
    """Enhanced state management for business research workflows"""

    # Core research state
    current_brief: Optional[BusinessResearchBrief] = None
    research_status: str = "idle"  # idle, planning, researching, synthesizing, completed, failed

    # Business context
    business_type: Optional[BusinessResearchType] = None
    industry_context: Optional[str] = None
    geography_context: Optional[str] = None
    strategic_objectives: List[str] = []

    # Progress tracking
    progress_percentage: float = 0.0
    current_phase: str = ""
    completed_phases: List[str] = []
    estimated_completion_time: Optional[datetime] = None

    # Quality and validation
    quality_checks_passed: List[str] = []
    quality_checks_failed: List[str] = []
    validation_required: bool = False

    # Integration status
    crm_sync_status: str = "not_synced"  # not_synced, syncing, synced, failed
    pms_sync_status: str = "not_synced"  # not_synced, syncing, synced, failed
    memory_updated: bool = False

    # Error handling
    errors: List[Dict[str, Any]] = []
    warnings: List[str] = []

    def update_progress(self, phase: str, percentage: float):
        """Update research progress"""
        self.current_phase = phase
        self.progress_percentage = percentage

        if percentage >= 100:
            self.research_status = "completed"
            if phase not in self.completed_phases:
                self.completed_phases.append(phase)

    def add_error(self, error_type: str, message: str, details: Any = None):
        """Add error to the state"""
        error = {
            "type": error_type,
            "message": message,
            "timestamp": datetime.now(timezone.utc),
            "details": details
        }
        self.errors.append(error)

        if error_type in ["critical", "fatal"]:
            self.research_status = "failed"

    def add_quality_check(self, check_name: str, passed: bool, details: str = None):
        """Add quality check result"""
        if passed:
            if check_name not in self.quality_checks_passed:
                self.quality_checks_passed.append(check_name)
        else:
            if check_name not in self.quality_checks_failed:
                self.quality_checks_failed.append(check_name)
                if details:
                    self.warnings.append(f"Quality check failed: {check_name} - {details}")

    def get_completion_summary(self) -> Dict[str, Any]:
        """Get completion summary"""
        return {
            "status": self.research_status,
            "progress": self.progress_percentage,
            "completed_phases": self.completed_phases,
            "quality_score": len(self.quality_checks_passed) / max(1, len(self.quality_checks_passed) + len(self.quality_checks_failed)),
            "errors_count": len(self.errors),
            "warnings_count": len(self.warnings),
            "crm_synced": self.crm_sync_status == "synced",
            "pms_synced": self.pms_sync_status == "synced"
        }


###################
# State Definitions
###################

def override_reducer(current_value, new_value):
    """Reducer function that allows overriding values in state."""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)


class AgentInputState(MessagesState):
    """InputState is only 'messages'."""

class AgentState(MessagesState):
    """Main agent state containing messages and research data."""

    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: Optional[str]
    raw_notes: Annotated[list[str], override_reducer] = []
    notes: Annotated[list[str], override_reducer] = []
    final_report: str

class SupervisorState(TypedDict):
    """State for the supervisor that manages research tasks."""

    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    notes: Annotated[list[str], override_reducer] = []
    research_iterations: int = 0
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherState(TypedDict):
    """State for individual researchers conducting research."""

    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int = 0
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherOutputState(BaseModel):
    """Output state from individual researchers."""

    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []


###################
# Utility Functions
###################

def create_business_research_brief(
    query: str,
    business_type: BusinessResearchType = None,
    industry: str = None,
    geography: str = None
) -> BusinessResearchBrief:
    """Create a new business research brief"""

    brief = BusinessResearchBrief(
        query=query,
        business_type=business_type,
        metadata={
            "industry": industry,
            "geography": geography,
            "business_focus": True
        }
    )

    return brief


def enhance_idea_with_business_context(idea: Idea) -> BusinessIdea:
    """Convert a regular idea to a business idea with enhanced context"""

    business_idea = BusinessIdea(
        **idea.model_dump(),
        risk_level="medium",  # Default
        strategic_alignment_score=3.0  # Default neutral
    )

    # Calculate enhanced RICE score
    business_idea.calculate_enhanced_rice_score()

    return business_idea


def enhance_finding_with_business_context(finding: Finding) -> BusinessFinding:
    """Convert a regular finding to a business finding with enhanced context"""

    business_finding = BusinessFinding(**finding.model_dump())

    # Assess business impact based on content
    content_lower = (finding.title + finding.summary).lower()

    if any(word in content_lower for word in ["revenue", "profit", "market share", "growth"]):
        business_finding.business_impact = "high"
    elif any(word in content_lower for word in ["risk", "threat", "competition", "challenge"]):
        business_finding.business_impact = "medium"
    else:
        business_finding.business_impact = "low"

    return business_finding


###################
# Core LangGraph State Definitions
###################

class AgentInputState(MessagesState):
    """InputState is only 'messages'."""

class AgentState(MessagesState):
    """Main agent state containing messages and research data."""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: Optional[str]
    raw_notes: Annotated[list[str], override_reducer] = []
    notes: Annotated[list[str], override_reducer] = []
    final_report: str

class SupervisorState(TypedDict):
    """State for the supervisor that manages research tasks."""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    notes: Annotated[list[str], override_reducer] = []
    research_iterations: int = 0
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherState(TypedDict):
    """State for individual researchers conducting research."""
    
    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int = 0
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherOutputState(BaseModel):
    """Output state from individual researchers."""
    
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []