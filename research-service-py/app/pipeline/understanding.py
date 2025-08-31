from __future__ import annotations
from typing import List

DEFAULT_FACETS = [
	"Market size & growth",
	"Buyer roles & cycles",
	"Channels & CAC benchmarks",
	"Top competitors & positioning",
	"Pricing models",
	"Regulations & compliance",
]

def extract_entities(query: str) -> List[str]:
	parts = [p for p in query.replace("/", " ").split() if p]
	seen = []
	for p in parts:
		if p not in seen:
			seen.append(p)
			if len(seen) >= 10:
				break
	return seen


def key_questions() -> List[str]:
	return DEFAULT_FACETS


def sub_queries(query: str, geo: List[str], scope: List[str]) -> List[str]:
	candidates = [query]
	candidates += [f"{query} {f}" for f in DEFAULT_FACETS]
	for g in geo:
		candidates += [f"{query} {g} market size", f"{query} {g} competitors", f"{query} {g} pricing benchmarks"]
	candidates += [f"{query} {s}" for s in scope]
	# dedupe and cap
	out: List[str] = []
	for c in candidates:
		if c not in out:
			out.append(c)
			if len(out) >= 25:
				break
	return out