from __future__ import annotations
from typing import List
from ..models import Finding, Idea, RICE


def summarize_brief(findings: List[Finding], query: str) -> str:
	bullets = [f"- {f.title}: {f.summary}" for f in findings[:5]]
	return "\n".join([f"Brief for: {query}", *bullets])


def generate_ideas(findings: List[Finding], query: str) -> List[Idea]:
	base = [
		Idea(idea=f"Validate top 2 channels for: {query}", RICE=RICE(reach=1000, impact=2, confidence=0.6, effort=10)),
		Idea(idea="Pricing sensitivity interviews with 10 ICP prospects", RICE=RICE(reach=10, impact=2, confidence=0.7, effort=8)),
		Idea(idea="Competitor teardown: top 3", RICE=RICE(reach=0, impact=1, confidence=0.8, effort=6)),
	]
	return base