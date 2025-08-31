from __future__ import annotations
from typing import List, Dict
import httpx
from readability import Document
from bs4 import BeautifulSoup


async def fetch_and_extract(hits: List[Dict[str, str]]):
	out = []
	async with httpx.AsyncClient(timeout=8.0) as client:
		for h in hits[:20]:
			try:
				resp = await client.get(h["url"]) 
				if resp.status_code != 200:
					continue
				doc = Document(resp.text)
				html = doc.summary(html_partial=True)
				soup = BeautifulSoup(html, "html5lib")
				text = soup.get_text("\n").strip()
				if not text:
					continue
				out.append({
					"url": h["url"],
					"title": doc.short_title() or h.get("title"),
					"contentText": text,
				})
			except Exception:
				continue
	return out