"""Research Module - Agentic Research Service"""

from .service import AgenticResearchService, ResearchService  # Backward compatibility

__all__ = [
    "AgenticResearchService",
    "ResearchService"  # Backward compatibility alias
]

__version__ = "2.0.0"
