from __future__ import annotations
from fastapi import APIRouter, HTTPException
from uuid import uuid4
from ..models import (
	RunRequest,
	SaveRequest,
	IdeasToPlanRequest,
	SubscribeRequest,
	ResearchBrief,
	PlanJSON,
	OKR,
	KR,
	Initiative,
	PlanTask,
)
from ..storage.memory_store import store
from ..pipeline.run import run_research
from ..integrations.crm import CRMClient
from ..subscriptions.scheduler import schedule_subscription

router = APIRouter()


@router.post("/run", response_model=ResearchBrief)
async def run(req: RunRequest) -> ResearchBrief:
	brief = await run_research(req)
	store.save_brief(brief)
	return brief


@router.post("/save")
async def save(req: SaveRequest):
	brief = store.get_brief(req.briefId)
	if not brief:
		raise HTTPException(status_code=404, detail="brief_not_found")
	client = CRMClient()
	notes_id = None
	if req.crmRef is not None:
		notes_id = await client.create_note(
			subject=f"Research Brief: {brief.query} ({brief.date})",
			description=f"{{\"briefId\": \"{brief.briefId}\", \"summary\": \"{(brief.summary or '')[:200]}\"}}",
			lead_id=req.crmRef.get("leadId") if isinstance(req.crmRef, dict) else None,
		)
		if isinstance(req.crmRef, dict) and req.crmRef.get("taskId") and req.attachments:
			await client.create_task_attachment(
				task_id=req.crmRef["taskId"],
				title="Research artifacts",
				description=f"Artifacts for brief {brief.briefId}",
				attachment_urls=[{"url": a.url, "name": a.name or "artifact"} for a in req.attachments],
			)
	return {"saved": True, "crmNotesId": notes_id}


@router.post("/ideas-to-plan", response_model=PlanJSON)
async def ideas_to_plan(req: IdeasToPlanRequest) -> PlanJSON:
	brief = store.get_brief(req.briefId)
	if not brief:
		raise HTTPException(status_code=404, detail="brief_not_found")
	selected = [brief.ideas[i] for i in req.selections if 0 <= i < len(brief.ideas)]
	okrs = [OKR(objective=f"Execute research initiatives for: {brief.query}", keyResults=[KR(metric="validated-ideas", target=len(selected))])]
	initiatives = []
	for idx, idea in enumerate(selected, start=1):
		initiatives.append(
			Initiative(
				id=f"init-{idx}",
				title=idea.idea,
				tasks=[
					PlanTask(id=f"task-{idx}-1", title="Define success metrics", assignee="TBD"),
					PlanTask(id=f"task-{idx}-2", title="Run pilot/experiment", assignee="TBD"),
					PlanTask(id=f"task-{idx}-3", title="Analyze results", assignee="TBD"),
				],
			)
		)
	return PlanJSON(okrs=okrs, initiatives=initiatives)


@router.post("/subscribe")
async def subscribe(req: SubscribeRequest):
	sub = schedule_subscription(query=req.query, cadence=req.cadence)
	return sub