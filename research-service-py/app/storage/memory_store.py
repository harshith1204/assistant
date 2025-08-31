from __future__ import annotations
from typing import Dict, Optional
from ..models import ResearchBrief


class MemoryStore:
    def __init__(self) -> None:
        self._briefs: Dict[str, ResearchBrief] = {}

    def save_brief(self, brief: ResearchBrief) -> None:
        self._briefs[brief.briefId] = brief

    def get_brief(self, brief_id: str) -> Optional[ResearchBrief]:
        return self._briefs.get(brief_id)


store = MemoryStore()
