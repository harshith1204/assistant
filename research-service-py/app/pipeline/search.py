from __future__ import annotations
from typing import List, Dict, Any


async def search_web(queries: List[str], freshness: Any) -> List[Dict[str, str]]:
	seen = set()
	results = []
	for q in queries:
		url = f"https://example.com/search?q={q.replace(' ', '+')}"
		if url not in seen:
			seen.add(url)
			results.append({"url": url, "title": q})
			if len(results) >= 50:
				break
	return results