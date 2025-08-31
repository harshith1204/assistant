"""CRM integration client"""

import json
from typing import Dict, Any, Optional, List
from datetime import datetime
import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings

logger = structlog.get_logger()


class CRMClient:
    """Client for CRM API integration"""
    
    def __init__(self):
        self.base_url = settings.crm_base_url
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        if settings.crm_auth_token:
            self.headers['Authorization'] = f'Bearer {settings.crm_auth_token}'
        if settings.crm_api_key:
            self.headers['X-API-Key'] = settings.crm_api_key
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def create_note(
        self,
        lead_id: str,
        subject: str,
        description: str,
        attachments: Optional[List[Dict[str, str]]] = None
    ) -> Optional[str]:
        """Create a note in CRM"""
        
        payload = {
            "leadId": lead_id,
            "subject": subject,
            "description": description,
            "notesAttachments": attachments or []
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/notes/create",
                    json=payload,
                    headers=self.headers,
                    timeout=30.0
                )
                response.raise_for_status()
                
                result = response.json()
                note_id = result.get('data', {}).get('notesId')
                
                logger.info("CRM note created", note_id=note_id, lead_id=lead_id)
                return note_id
                
            except Exception as e:
                logger.error("Failed to create CRM note", error=str(e), lead_id=lead_id)
                return None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def update_note(
        self,
        note_id: str,
        subject: Optional[str] = None,
        description: Optional[str] = None
    ) -> bool:
        """Update an existing note"""
        
        payload = {"notesId": note_id}
        if subject:
            payload["subject"] = subject
        if description:
            payload["description"] = description
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.put(
                    f"{self.base_url}/notes/update",
                    json=payload,
                    headers=self.headers,
                    timeout=30.0
                )
                response.raise_for_status()
                
                logger.info("CRM note updated", note_id=note_id)
                return True
                
            except Exception as e:
                logger.error("Failed to update CRM note", error=str(e), note_id=note_id)
                return False
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def create_task(
        self,
        business_id: str,
        name: str,
        description: str,
        due_date: Optional[datetime] = None,
        assigned_to: Optional[str] = None,
        parent_id: Optional[str] = None,
        priority: str = "MEDIUM"
    ) -> Optional[str]:
        """Create a task in CRM"""
        
        payload = {
            "businessId": business_id,
            "name": name,
            "description": description,
            "priority": priority,
            "taskStatus": "NEW"
        }
        
        if due_date:
            payload["dueDate"] = due_date.isoformat()
        if assigned_to:
            payload["assignedTo"] = assigned_to
        if parent_id:
            payload["parentId"] = parent_id
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/task/create",
                    json=payload,
                    headers=self.headers,
                    timeout=30.0
                )
                response.raise_for_status()
                
                result = response.json()
                task_id = result.get('data', {}).get('id')
                
                logger.info("CRM task created", task_id=task_id, business_id=business_id)
                return task_id
                
            except Exception as e:
                logger.error("Failed to create CRM task", error=str(e), business_id=business_id)
                return None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def update_task_status(
        self,
        task_id: str,
        status: str
    ) -> bool:
        """Update task status"""
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.put(
                    f"{self.base_url}/task/change/status",
                    params={"taskId": task_id, "taskStatus": status},
                    headers=self.headers,
                    timeout=30.0
                )
                response.raise_for_status()
                
                logger.info("CRM task status updated", task_id=task_id, status=status)
                return True
                
            except Exception as e:
                logger.error("Failed to update task status", error=str(e), task_id=task_id)
                return False
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def add_task_attachment(
        self,
        task_id: str,
        title: str,
        description: str,
        attachment_urls: List[str]
    ) -> Optional[str]:
        """Add attachment to task"""
        
        payload = {
            "title": title,
            "description": description,
            "attachmentUrl": [{"url": url} for url in attachment_urls],
            "taskId": task_id
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/task/attachment/create/{task_id}",
                    json=payload,
                    headers=self.headers,
                    timeout=30.0
                )
                response.raise_for_status()
                
                result = response.json()
                attachment_id = result.get('data', {}).get('attachmentId')
                
                logger.info("Task attachment added", task_id=task_id, attachment_id=attachment_id)
                return attachment_id
                
            except Exception as e:
                logger.error("Failed to add task attachment", error=str(e), task_id=task_id)
                return None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def create_meeting(
        self,
        business_id: str,
        lead_id: str,
        title: str,
        description: str,
        start_time: datetime,
        end_time: datetime,
        meeting_type: str = "VIRTUAL"
    ) -> Optional[str]:
        """Create a meeting in CRM"""
        
        payload = {
            "businessId": business_id,
            "leadId": lead_id,
            "title": title,
            "description": description,
            "startDateTime": start_time.isoformat(),
            "endDateTime": end_time.isoformat(),
            "meetingType": meeting_type,
            "meetingStatus": "SCHEDULED"
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/meeting/create",
                    json=payload,
                    headers=self.headers,
                    timeout=30.0
                )
                response.raise_for_status()
                
                result = response.json()
                meeting_id = result.get('data', {}).get('meetingId')
                
                logger.info("CRM meeting created", meeting_id=meeting_id, lead_id=lead_id)
                return meeting_id
                
            except Exception as e:
                logger.error("Failed to create meeting", error=str(e), lead_id=lead_id)
                return None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_lead(self, lead_id: str) -> Optional[Dict[str, Any]]:
        """Get lead details"""
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/leads/get-one",
                    params={"leadId": lead_id},
                    headers=self.headers,
                    timeout=30.0
                )
                response.raise_for_status()
                
                result = response.json()
                return result.get('data')
                
            except Exception as e:
                logger.error("Failed to get lead", error=str(e), lead_id=lead_id)
                return None
    
    async def save_research_brief(
        self,
        brief: Dict[str, Any],
        lead_id: Optional[str] = None,
        business_id: Optional[str] = None,
        create_tasks: bool = False
    ) -> Dict[str, Any]:
        """Save research brief to CRM"""
        
        results = {
            "note_id": None,
            "task_ids": [],
            "attachment_ids": [],
            "errors": []
        }
        
        # Create main note with research brief
        if lead_id:
            note_subject = f"Research Brief: {brief.get('query', 'Research')}"
            note_description = self._format_brief_for_note(brief)
            
            note_id = await self.create_note(
                lead_id=lead_id,
                subject=note_subject,
                description=note_description
            )
            
            if note_id:
                results["note_id"] = note_id
            else:
                results["errors"].append("Failed to create note")
        
        # Create tasks from ideas if requested
        if create_tasks and business_id:
            for idea in brief.get('ideas', []):
                task_name = f"Research Action: {idea['idea'][:100]}"
                task_description = f"""
Idea: {idea['idea']}
Rationale: {idea.get('rationale', '')}
RICE Score: {idea.get('rice', {}).get('score', 'N/A')}
Prerequisites: {', '.join(idea.get('prerequisites', []))}
Risks: {', '.join(idea.get('risks', []))}
"""
                
                task_id = await self.create_task(
                    business_id=business_id,
                    name=task_name,
                    description=task_description,
                    parent_id=lead_id
                )
                
                if task_id:
                    results["task_ids"].append(task_id)
        
        return results
    
    def _format_brief_for_note(self, brief: Dict[str, Any]) -> str:
        """Format research brief for CRM note"""
        
        sections = [
            f"**Research Query:** {brief.get('query', 'N/A')}",
            f"**Date:** {brief.get('date', datetime.utcnow().isoformat())}",
            f"**Executive Summary:**\n{brief.get('executive_summary', 'N/A')}",
            "\n**Key Findings:**"
        ]
        
        for finding in brief.get('findings', []):
            sections.append(f"\n• **{finding['title']}**")
            sections.append(f"  {finding['summary']}")
            sections.append(f"  Confidence: {finding.get('confidence', 0):.1%}")
        
        sections.append("\n**Recommended Actions:**")
        for idea in brief.get('ideas', [])[:5]:
            sections.append(f"\n• {idea['idea']}")
            if idea.get('rice', {}).get('score'):
                sections.append(f"  RICE Score: {idea['rice']['score']:.1f}")
        
        sections.append(f"\n**Total Sources:** {brief.get('total_sources', 0)}")
        
        return "\n".join(sections)