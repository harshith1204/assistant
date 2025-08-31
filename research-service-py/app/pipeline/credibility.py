from __future__ import annotations
from typing import List, Dict
from urllib.parse import urlparse


def _domain_score(url: str) -> float:
	host = urlparse(url).hostname or ""
	if host.endswith(".gov") or host.endswith(".gov.in"):
		return 0.95
	if host.endswith(".edu"):
		return 0.9
	if any(k in host for k in ["mckinsey", "bcg", "gartner"]):
		return 0.85
	if any(k in host for k in ["economictimes", "thehindu", "medianama"]):
		return 0.75
	return 0.5


def score_docs(docs: List[Dict]):
	out = [{**d, "score": _domain_score(d.get("url", ""))} for d in docs]
	out.sort(key=lambda x: x["score"], reverse=True)
	return out