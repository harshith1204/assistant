from __future__ import annotations
import os
import httpx
from typing import Optional, List, Dict, Any


class CRMClient:
    def __init__(self) -> None:
        self.base_url = os.getenv("CRM_BASE_URL", "https://stage-api.simpo.ai/crm").rstrip("/")
        self.token = os.getenv("CRM_BEARER_TOKEN")
        self._client = httpx.AsyncClient(base_url=self.base_url, headers=self._headers(), timeout=10.0)

    def _headers(self) -> Dict[str, str]:
        h: Dict[str, str] = {}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    async def create_note(self, subject: str, description: str, lead_id: Optional[str] = None) -> Optional[str]:
        payload: Dict[str, Any] = {
            "subject": subject,
            "description": description,
        }
        if lead_id:
            payload["leadId"] = lead_id
        resp = await self._client.post("/notes/create", json=payload)
        try:
            data = resp.json()
        except Exception:
            data = None
        return data.get("notesId") if isinstance(data, dict) else None

    async def create_task_attachment(self, task_id: str, title: str, description: str, attachment_urls: List[Dict[str, str]]):
        payload: Dict[str, Any] = {
            "title": title,
            "description": description,
            "attachmentUrl": attachment_urls,
        }
        return (await self._client.post(f"/task/attachment/create/{task_id}", json=payload)).json()