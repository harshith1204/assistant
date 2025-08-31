from __future__ import annotations
from datetime import datetime
from uuid import uuid4
from typing import List
from ..models import ResearchBrief, Finding, Idea, RICE, RunRequest, Evidence
from .understanding import extract_entities, key_questions, sub_queries
from .search import search_web
from .fetch import fetch_and_extract
from .normalize import normalize_and_cluster
from .credibility import score_docs
from .extract import extract_findings
from .summarize import summarize_brief, generate_ideas
from .review import review


async def run_research(req: RunRequest) -> ResearchBrief:
	entities = extract_entities(req.query)
	keys = key_questions()
	queries = sub_queries(req.query, req.geo or [], req.scope or [])
	search_results = await search_web(queries, req.freshness)
	docs = await fetch_and_extract(search_results)
	norm = normalize_and_cluster(docs)
	scored = score_docs(norm)
	findings = extract_findings(scored)
	summary = summarize_brief(findings, req.query)
	reviewed = review(findings, summary)
	ideas = generate_ideas(reviewed[0], req.query)
	brief = ResearchBrief(
		briefId=str(uuid4()),
		query=req.query,
		date=datetime.utcnow().isoformat(),
		entities=entities,
		keyQuestions=keys,
		findings=reviewed[0],
		ideas=ideas,
		attachments=[],
		summary=reviewed[1],
	)
	return brief