"""Research Module - Advanced Business Research Service"""

from .service import ResearchService
from .research_state import BusinessResearchBrief, BusinessResearchType
from .config import BusinessDeepResearchConfig

__all__ = [
    "ResearchService",
    "BusinessResearchBrief",
    "BusinessResearchType",
    "BusinessDeepResearchConfig"
]

__version__ = "1.0.0"
