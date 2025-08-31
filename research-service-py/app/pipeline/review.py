from __future__ import annotations
from typing import List, Tuple
from ..models import Finding


def review(findings: List[Finding], summary: str) -> Tuple[List[Finding], str]:
	clean = [f for f in findings if f.evidence]
	return clean, summary