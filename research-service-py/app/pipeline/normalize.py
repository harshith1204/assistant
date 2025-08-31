from __future__ import annotations
from typing import List, Dict


def _simple_hash(s: str) -> str:
	h = 0
	for ch in s[:2000]:
		h = (h * 31 + ord(ch)) & 0xFFFFFFFF
	return str(h)


def normalize_and_cluster(docs: List[Dict]):
	dedup: dict[str, dict] = {}
	for d in docs:
		h = _simple_hash(d.get("contentText", ""))
		cluster = (d.get("title") or d.get("url", "")).lower().split()[:4]
		key = "-".join(cluster) or "misc"
		nd = {**d, "hash": h, "clusterKey": key}
		if h not in dedup:
			dedup[h] = nd
	return list(dedup.values())