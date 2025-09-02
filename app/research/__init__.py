"""Research Module - Advanced Business Research Service"""

from .service import ResearchService
from .state import (
    BusinessResearchBrief, BusinessResearchType, BusinessResearchState,
    BusinessIdea, BusinessFinding, DeepResearchMetadata,
    AgentState, SupervisorState, ResearcherState, AgentInputState
)
from .configuration import BusinessDeepResearchConfig, Configuration, SearchAPI

__all__ = [
    "ResearchService",
    "BusinessResearchBrief",
    "BusinessResearchType",
    "BusinessResearchState",
    "BusinessIdea",
    "BusinessFinding",
    "DeepResearchMetadata",
    "BusinessDeepResearchConfig",
    "Configuration",
    "SearchAPI",
    "AgentState",
    "SupervisorState",
    "ResearcherState",
    "AgentInputState"
]

__version__ = "1.0.0"
