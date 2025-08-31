from __future__ import annotations
from typing import Literal, Dict
from apscheduler.schedulers.background import BackgroundScheduler

_scheduler: BackgroundScheduler | None = None


def _get_scheduler() -> BackgroundScheduler:
	global _scheduler
	if _scheduler is None:
		_scheduler = BackgroundScheduler()
		_scheduler.start()
	return _scheduler


def schedule_subscription(query: str, cadence: Literal["weekly", "monthly"]) -> Dict[str, str]:
	s = _get_scheduler()
	job_id = f"sub-{abs(hash((query, cadence)))}"
	if cadence == "weekly":
		trigger = {"trigger": "cron", "day_of_week": "mon", "hour": 8, "minute": 0}
	else:
		trigger = {"trigger": "cron", "day": 1, "hour": 8, "minute": 0}
	# noop job
	def job():
		return None
	try:
		s.add_job(job, id=job_id, replace_existing=True, **trigger)
	except Exception:
		pass
	return {"id": job_id, "query": query, "cadence": cadence, "nextRunAt": "soon"}