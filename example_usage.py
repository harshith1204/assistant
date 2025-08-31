"""Example usage of the Research & Brainstorming Engine API"""

import asyncio
import httpx
import json
from datetime import datetime
from typing import Dict, Any


# API Configuration
API_BASE_URL = "http://localhost:8000"


async def run_research_example():
    """Example: Run a complete research query"""
    
    async with httpx.AsyncClient() as client:
        # 1. Run research
        print("🔍 Starting research...")
        research_request = {
            "query": "How can we grow B2B edtech market share in India?",
            "scope": ["market", "competitors", "pricing", "channels"],
            "geo": "India",
            "industry": "EdTech",
            "timeframe": "next 12 months",
            "max_sources": 20,
            "deep_dive": True
        }
        
        response = await client.post(
            f"{API_BASE_URL}/research/run",
            json=research_request,
            timeout=120.0
        )
        
        if response.status_code == 200:
            brief = response.json()
            print(f"✅ Research completed: {brief['brief_id']}")
            print(f"   - Findings: {len(brief['findings'])}")
            print(f"   - Ideas: {len(brief['ideas'])}")
            print(f"   - Sources: {brief.get('total_sources', 0)}")
            
            return brief
        else:
            print(f"❌ Research failed: {response.text}")
            return None


async def save_to_systems_example(brief_id: str):
    """Example: Save research to CRM and PMS"""
    
    async with httpx.AsyncClient() as client:
        save_request = {
            "brief_id": brief_id,
            "crm_ref": {
                "lead_id": "550e8400-e29b-41d4-a716-446655440000",
                "business_id": "550e8400-e29b-41d4-a716-446655440001"
            },
            "pms_ref": {
                "project_id": "550e8400-e29b-41d4-a716-446655440002"
            },
            "create_tasks": True
        }
        
        print("\n💾 Saving to CRM and PMS...")
        response = await client.post(
            f"{API_BASE_URL}/research/save",
            json=save_request,
            timeout=30.0
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Saved successfully:")
            
            if result.get('crm'):
                print(f"   CRM Note: {result['crm'].get('note_id')}")
                print(f"   CRM Tasks: {len(result['crm'].get('task_ids', []))}")
            
            if result.get('pms'):
                print(f"   PMS Page: {result['pms'].get('page_id')}")
                print(f"   PMS Work Items: {len(result['pms'].get('work_item_ids', []))}")
            
            return result
        else:
            print(f"❌ Save failed: {response.text}")
            return None


async def create_plan_example(brief_id: str, idea_ids: list):
    """Example: Convert ideas to execution plan"""
    
    async with httpx.AsyncClient() as client:
        plan_request = {
            "brief_id": brief_id,
            "selected_ideas": idea_ids[:3],  # Select top 3 ideas
            "timeline_weeks": 12,
            "team_size": 5,
            "budget": 50000
        }
        
        print("\n📋 Creating execution plan...")
        response = await client.post(
            f"{API_BASE_URL}/research/ideas-to-plan",
            json=plan_request,
            timeout=30.0
        )
        
        if response.status_code == 200:
            plan = response.json()
            print("✅ Plan created:")
            print(f"   - Timeline: {plan['timeline_weeks']} weeks")
            print(f"   - Initiatives: {len(plan['initiatives'])}")
            print(f"   - Milestones: {len(plan['milestones'])}")
            
            for init in plan['initiatives']:
                print(f"\n   Initiative: {init['title'][:50]}...")
                print(f"     Duration: {init['duration_weeks']} weeks")
                print(f"     Effort: {init['effort_days']} days")
                print(f"     Tasks: {len(init['tasks'])}")
            
            return plan
        else:
            print(f"❌ Plan creation failed: {response.text}")
            return None


async def subscribe_example():
    """Example: Subscribe to research updates"""
    
    async with httpx.AsyncClient() as client:
        subscription_request = {
            "query": "AI trends in healthcare 2025",
            "cadence": "weekly",
            "scope": ["technology", "market"],
            "notify_email": "research@example.com"
        }
        
        print("\n🔔 Creating subscription...")
        response = await client.post(
            f"{API_BASE_URL}/research/subscribe",
            json=subscription_request,
            timeout=30.0
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Subscription created: {result['subscription_id']}")
            print(f"   Cadence: {result['cadence']}")
            return result
        else:
            print(f"❌ Subscription failed: {response.text}")
            return None


async def list_briefs_example():
    """Example: List all research briefs"""
    
    async with httpx.AsyncClient() as client:
        print("\n📚 Listing research briefs...")
        response = await client.get(
            f"{API_BASE_URL}/research/list",
            params={"limit": 10, "offset": 0}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Found {result['total']} briefs:")
            
            for brief in result['briefs']:
                print(f"\n   📄 {brief['query'][:50]}...")
                print(f"      ID: {brief['brief_id']}")
                print(f"      Date: {brief['date']}")
                print(f"      Findings: {brief['findings_count']}")
                print(f"      Ideas: {brief['ideas_count']}")
                print(f"      Confidence: {brief['average_confidence']:.1%}")
            
            return result
        else:
            print(f"❌ List failed: {response.text}")
            return None


async def check_health():
    """Check if the API is running"""
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{API_BASE_URL}/health", timeout=5.0)
            if response.status_code == 200:
                print("✅ API is healthy")
                return True
        except:
            pass
    
    print("❌ API is not responding. Please start the server first:")
    print("   uvicorn app.main:app --reload")
    return False


async def main():
    """Run all examples"""
    
    print("=" * 60)
    print("Research & Brainstorming Engine - Example Usage")
    print("=" * 60)
    
    # Check health
    if not await check_health():
        return
    
    # Run research
    brief = await run_research_example()
    
    if brief:
        # Extract idea IDs
        idea_ids = [idea['id'] for idea in brief.get('ideas', [])]
        
        # Save to systems (commented out to avoid errors with mock IDs)
        # await save_to_systems_example(brief['brief_id'])
        
        # Create plan
        if idea_ids:
            await create_plan_example(brief['brief_id'], idea_ids)
    
    # Subscribe to updates
    await subscribe_example()
    
    # List all briefs
    await list_briefs_example()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())