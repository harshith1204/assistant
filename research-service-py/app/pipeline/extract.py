from __future__ import annotations
from typing import List, Dict
from ..models import Finding, Evidence


def extract_findings(docs: List[Dict]) -> List[Finding]:
	findings: List[Finding] = []
	for d in docs[:8]:
		text = (d.get("contentText") or "")
		first_line = " ".join(text.split())[:280]
		findings.append(Finding(
			title=d.get("title") or d.get("url"),
			summary=first_line,
			evidence=[Evidence(quote=first_line[:200], url=d.get("url"))],
			confidence=max(0.4, min(0.95, d.get("score", 0.5))),
			recency="unknown",
		))
	return findings