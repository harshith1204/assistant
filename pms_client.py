"""PMS integration client"""

import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings

logger = structlog.get_logger()


class PMSClient:
    """Client for PMS API integration"""
    
    def __init__(self):
        self.base_url = settings.pms_base_url
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        if settings.pms_auth_token:
            self.headers['Authorization'] = f'Bearer {settings.pms_auth_token}'
        if settings.pms_api_key:
            self.headers['X-API-Key'] = settings.pms_api_key
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def create_page(
        self,
        project_id: str,
        title: str,
        content: str,
        parent_id: Optional[str] = None
    ) -> Optional[str]:
        """Create a page in PMS for research documentation"""
        
        payload = {
            "projectId": project_id,
            "title": title,
            "content": content,
            "type": "RESEARCH"
        }
        
        if parent_id:
            payload["parentId"] = parent_id
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/page/create",
                    json=payload,
                    headers=self.headers,
                    timeout=30.0
                )
                response.raise_for_status()
                
                result = response.json()
                page_id = result.get('data', {}).get('pageId')
                
                logger.info("PMS page created", page_id=page_id, project_id=project_id)
                return page_id
                
            except Exception as e:
                logger.error("Failed to create PMS page", error=str(e), project_id=project_id)
                return None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def update_page(
        self,
        page_id: str,
        title: Optional[str] = None,
        content: Optional[str] = None
    ) -> bool:
        """Update an existing page"""
        
        payload = {"pageId": page_id}
        if title:
            payload["title"] = title
        if content:
            payload["content"] = content
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.put(
                    f"{self.base_url}/page/update",
                    json=payload,
                    headers=self.headers,
                    timeout=30.0
                )
                response.raise_for_status()
                
                logger.info("PMS page updated", page_id=page_id)
                return True
                
            except Exception as e:
                logger.error("Failed to update PMS page", error=str(e), page_id=page_id)
                return False
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def create_work_item(
        self,
        project_id: str,
        title: str,
        description: str,
        item_type: str = "TASK",
        priority: str = "MEDIUM",
        assigned_to: Optional[str] = None,
        due_date: Optional[datetime] = None,
        tags: Optional[List[str]] = None
    ) -> Optional[str]:
        """Create a work item in PMS"""
        
        payload = {
            "projectId": project_id,
            "title": title,
            "description": description,
            "type": item_type,
            "priority": priority,
            "status": "TODO",
            "tags": tags or ["research"]
        }
        
        if assigned_to:
            payload["assignedTo"] = assigned_to
        if due_date:
            payload["dueDate"] = due_date.isoformat()
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/project/work-item",
                    json=payload,
                    headers=self.headers,
                    timeout=30.0
                )
                response.raise_for_status()
                
                result = response.json()
                work_item_id = result.get('data', {}).get('workItemId')
                
                logger.info("PMS work item created", work_item_id=work_item_id, project_id=project_id)
                return work_item_id
                
            except Exception as e:
                logger.error("Failed to create work item", error=str(e), project_id=project_id)
                return None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def update_work_item(
        self,
        work_item_id: str,
        status: Optional[str] = None,
        description: Optional[str] = None,
        assigned_to: Optional[str] = None
    ) -> bool:
        """Update work item"""
        
        payload = {"workItemId": work_item_id}
        if status:
            payload["status"] = status
        if description:
            payload["description"] = description
        if assigned_to:
            payload["assignedTo"] = assigned_to
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.put(
                    f"{self.base_url}/project/work-item",
                    json=payload,
                    headers=self.headers,
                    timeout=30.0
                )
                response.raise_for_status()
                
                logger.info("Work item updated", work_item_id=work_item_id)
                return True
                
            except Exception as e:
                logger.error("Failed to update work item", error=str(e), work_item_id=work_item_id)
                return False
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def add_work_item_attachment(
        self,
        work_item_id: str,
        file_path: str,
        file_name: str,
        file_type: str = "document"
    ) -> Optional[str]:
        """Add attachment to work item"""
        
        async with httpx.AsyncClient() as client:
            try:
                with open(file_path, 'rb') as f:
                    files = {'file': (file_name, f, 'application/octet-stream')}
                    data = {'type': file_type}
                    
                    response = await client.post(
                        f"{self.base_url}/project/work-item/{work_item_id}/add-attachment",
                        files=files,
                        data=data,
                        headers={k: v for k, v in self.headers.items() if k != 'Content-Type'},
                        timeout=60.0
                    )
                
                response.raise_for_status()
                
                result = response.json()
                attachment_id = result.get('data', {}).get('attachmentId')
                
                logger.info("Attachment added", work_item_id=work_item_id, attachment_id=attachment_id)
                return attachment_id
                
            except Exception as e:
                logger.error("Failed to add attachment", error=str(e), work_item_id=work_item_id)
                return None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def add_work_item_comment(
        self,
        work_item_id: str,
        comment: str,
        author_id: Optional[str] = None
    ) -> Optional[str]:
        """Add comment to work item"""
        
        payload = {
            "workItemId": work_item_id,
            "comment": comment
        }
        
        if author_id:
            payload["authorId"] = author_id
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/project/work-items/comment",
                    json=payload,
                    headers=self.headers,
                    timeout=30.0
                )
                response.raise_for_status()
                
                result = response.json()
                comment_id = result.get('data', {}).get('commentId')
                
                logger.info("Comment added", work_item_id=work_item_id, comment_id=comment_id)
                return comment_id
                
            except Exception as e:
                logger.error("Failed to add comment", error=str(e), work_item_id=work_item_id)
                return None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def create_view(
        self,
        project_id: str,
        name: str,
        view_type: str = "BOARD",
        filters: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Create a view in PMS"""
        
        payload = {
            "projectId": project_id,
            "name": name,
            "type": view_type,
            "filters": filters or {"tags": ["research"]}
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/project/view",
                    json=payload,
                    headers=self.headers,
                    timeout=30.0
                )
                response.raise_for_status()
                
                result = response.json()
                view_id = result.get('data', {}).get('viewId')
                
                logger.info("PMS view created", view_id=view_id, project_id=project_id)
                return view_id
                
            except Exception as e:
                logger.error("Failed to create view", error=str(e), project_id=project_id)
                return None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def favourite_view(self, view_id: str) -> bool:
        """Mark view as favourite"""
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/project/view/{view_id}/favourite",
                    headers=self.headers,
                    timeout=30.0
                )
                response.raise_for_status()
                
                logger.info("View marked as favourite", view_id=view_id)
                return True
                
            except Exception as e:
                logger.error("Failed to favourite view", error=str(e), view_id=view_id)
                return False
    
    async def save_research_brief(
        self,
        brief: Dict[str, Any],
        project_id: str,
        create_work_items: bool = False
    ) -> Dict[str, Any]:
        """Save research brief to PMS"""
        
        results = {
            "page_id": None,
            "work_item_ids": [],
            "view_id": None,
            "errors": []
        }
        
        # Create research page
        page_title = f"Research: {brief.get('query', 'Research Brief')}"
        page_content = self._format_brief_for_page(brief)
        
        page_id = await self.create_page(
            project_id=project_id,
            title=page_title,
            content=page_content
        )
        
        if page_id:
            results["page_id"] = page_id
        else:
            results["errors"].append("Failed to create page")
        
        # Create work items from ideas if requested
        if create_work_items:
            for idea in brief.get('ideas', []):
                work_item_title = f"Research Task: {idea['idea'][:100]}"
                work_item_description = f"""
**Idea:** {idea['idea']}

**Rationale:** {idea.get('rationale', '')}

**RICE Score:** {idea.get('rice', {}).get('score', 'N/A')}
- Reach: {idea.get('rice', {}).get('reach', 'N/A')}
- Impact: {idea.get('rice', {}).get('impact', 'N/A')}
- Confidence: {idea.get('rice', {}).get('confidence', 'N/A')}
- Effort: {idea.get('rice', {}).get('effort', 'N/A')} days

**Prerequisites:**
{chr(10).join('- ' + p for p in idea.get('prerequisites', []))}

**Risks:**
{chr(10).join('- ' + r for r in idea.get('risks', []))}

**Related Research:** [View Research Page](page:{page_id})
"""
                
                # Determine priority based on RICE score
                rice_score = idea.get('rice', {}).get('score', 0)
                if rice_score > 100:
                    priority = "HIGH"
                elif rice_score > 50:
                    priority = "MEDIUM"
                else:
                    priority = "LOW"
                
                work_item_id = await self.create_work_item(
                    project_id=project_id,
                    title=work_item_title,
                    description=work_item_description,
                    priority=priority,
                    tags=["research", "idea", brief.get('query', 'research')[:20]]
                )
                
                if work_item_id:
                    results["work_item_ids"].append(work_item_id)
                    
                    # Add comment linking to research
                    await self.add_work_item_comment(
                        work_item_id=work_item_id,
                        comment=f"This task was generated from research brief: {brief.get('brief_id', 'unknown')}"
                    )
        
        # Create Research Board view
        view_id = await self.create_view(
            project_id=project_id,
            name="Research Board",
            view_type="BOARD",
            filters={"tags": ["research"]}
        )
        
        if view_id:
            results["view_id"] = view_id
            await self.favourite_view(view_id)
        
        return results
    
    def _format_brief_for_page(self, brief: Dict[str, Any]) -> str:
        """Format research brief for PMS page (Markdown)"""
        
        sections = [
            f"# Research Brief: {brief.get('query', 'Research')}",
            f"\n**Date:** {brief.get('date', datetime.now(timezone.utc).isoformat())}",
            f"**Brief ID:** `{brief.get('brief_id', 'N/A')}`",
            f"\n## Executive Summary\n\n{brief.get('executive_summary', 'No summary available.')}",
            "\n## Key Questions\n"
        ]
        
        for question in brief.get('key_questions', []):
            sections.append(f"- {question}")
        
        sections.append("\n## Findings\n")
        
        for i, finding in enumerate(brief.get('findings', []), 1):
            sections.append(f"\n### {i}. {finding['title']}")
            sections.append(f"\n{finding['summary']}")
            sections.append(f"\n**Confidence:** {finding.get('confidence', 0):.1%} | **Recency:** {finding.get('recency', 'unknown')}")
            
            if finding.get('key_insights'):
                sections.append("\n**Key Insights:**")
                for insight in finding['key_insights']:
                    sections.append(f"- {insight}")
            
            if finding.get('evidence'):
                sections.append("\n**Sources:**")
                for evidence in finding['evidence'][:3]:
                    sections.append(f"- [{evidence.get('title', 'Source')}]({evidence['url']})")
        
        sections.append("\n## Recommended Actions\n")
        
        for i, idea in enumerate(brief.get('ideas', [])[:10], 1):
            sections.append(f"\n### Action {i}: {idea['idea']}")
            sections.append(f"\n{idea.get('rationale', '')}")
            
            if idea.get('rice'):
                rice = idea['rice']
                sections.append(f"\n**RICE Score:** {rice.get('score', 0):.1f}")
                sections.append(f"- Reach: {rice.get('reach', 0):,} entities")
                sections.append(f"- Impact: {rice.get('impact', 0)}/3")
                sections.append(f"- Confidence: {rice.get('confidence', 0):.1%}")
                sections.append(f"- Effort: {rice.get('effort', 0)} person-days")
        
        sections.append(f"\n## Metadata")
        sections.append(f"\n- **Total Sources Analyzed:** {brief.get('total_sources', 0)}")
        sections.append(f"- **Average Confidence:** {brief.get('average_confidence', 0):.1%}")
        sections.append(f"- **Entities Identified:** {', '.join(brief.get('entities', []))}")
        
        return "\n".join(sections)