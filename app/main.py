"""Main FastAPI application for Research & Brainstorming Engine"""

from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
import asyncio
import uuid

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import structlog
import uvicorn

from app.config import settings
from app.models import (
    ResearchRequest, ResearchBrief, SaveRequest, PlanRequest,
    ResearchStatus, SubscriptionRequest
)
from app.chat_models import (
    ChatRequest, ChatResponse, ConversationSummary,
    MemoryUpdate
)
from app.core.research_engine import ResearchEngine
from app.core.unified_chat_engine import UnifiedChatEngine
from app.websocket_handler import websocket_endpoint

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Store research briefs in memory (use Redis/DB in production)
research_store: Dict[str, ResearchBrief] = {}
research_engines: Dict[str, ResearchEngine] = {}
chat_engine = UnifiedChatEngine()  # Unified chat engine instance


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the application"""
    logger.info("Starting Research & Brainstorming Engine", version=settings.app_version)
    yield
    logger.info("Shutting down Research & Brainstorming Engine")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered research and brainstorming engine with CRM/PMS integration",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "operational"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": settings.app_version,
        "debug": settings.debug
    }


@app.post("/research/run", response_model=ResearchBrief)
async def run_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks
) -> ResearchBrief:
    """
    Run research based on query
    
    This endpoint initiates a research process that:
    1. Parses and understands the query
    2. Searches multiple sources across the web
    3. Synthesizes findings with citations
    4. Generates actionable ideas with RICE scoring
    5. Creates an executive summary
    """
    try:
        # Create research engine instance
        engine = ResearchEngine()
        engine_id = str(uuid.uuid4())
        research_engines[engine_id] = engine
        
        # Run research
        logger.info("Starting research", query=request.query, engine_id=engine_id)
        brief = await engine.run_research(request)
        
        # Store brief
        research_store[brief.brief_id] = brief
        
        # Clean up engine after delay
        background_tasks.add_task(cleanup_engine, engine_id, delay=300)
        
        return brief
        
    except Exception as e:
        logger.error("Research failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Research failed: {str(e)}")


@app.get("/research/status/{engine_id}", response_model=ResearchStatus)
async def get_research_status(engine_id: str) -> ResearchStatus:
    """Get status of ongoing research"""
    
    engine = research_engines.get(engine_id)
    if not engine:
        raise HTTPException(status_code=404, detail="Research engine not found")
    
    return engine.get_status()


@app.get("/research/brief/{brief_id}", response_model=ResearchBrief)
async def get_research_brief(brief_id: str) -> ResearchBrief:
    """Get a specific research brief by ID"""
    
    brief = research_store.get(brief_id)
    if not brief:
        raise HTTPException(status_code=404, detail="Research brief not found")
    
    return brief


@app.post("/research/save")
async def save_research(request: SaveRequest) -> Dict[str, Any]:
    """
    Save research brief to CRM and/or PMS
    
    This endpoint:
    1. Retrieves the research brief
    2. Saves it to CRM as notes/tasks if crm_ref provided
    3. Saves it to PMS as pages/work items if pms_ref provided
    4. Returns the created IDs for tracking
    """
    
    brief = research_store.get(request.brief_id)
    if not brief:
        raise HTTPException(status_code=404, detail="Research brief not found")
    
    results = {
        "brief_id": request.brief_id,
        "crm": None,
        "pms": None
    }
    
    engine = ResearchEngine()
    
    # Save to CRM if requested
    if request.crm_ref:
        try:
            crm_result = await engine.save_to_crm(
                brief,
                lead_id=request.crm_ref.get("lead_id"),
                business_id=request.crm_ref.get("business_id"),
                create_tasks=request.create_tasks
            )
            results["crm"] = crm_result
            logger.info("Saved to CRM", brief_id=request.brief_id, result=crm_result)
        except Exception as e:
            logger.error("CRM save failed", error=str(e))
            results["crm"] = {"error": str(e)}
    
    # Save to PMS if requested
    if request.pms_ref:
        try:
            pms_result = await engine.save_to_pms(
                brief,
                project_id=request.pms_ref.get("project_id"),
                create_work_items=request.create_tasks
            )
            results["pms"] = pms_result
            logger.info("Saved to PMS", brief_id=request.brief_id, result=pms_result)
        except Exception as e:
            logger.error("PMS save failed", error=str(e))
            results["pms"] = {"error": str(e)}
    
    return results


@app.post("/research/ideas-to-plan")
async def convert_ideas_to_plan(request: PlanRequest) -> Dict[str, Any]:
    """
    Convert selected research ideas into an execution plan
    
    This endpoint:
    1. Takes selected ideas from a research brief
    2. Creates a structured plan with timeline
    3. Generates initiatives and milestones
    4. Can be used to create PMS work items
    """
    
    brief = research_store.get(request.brief_id)
    if not brief:
        raise HTTPException(status_code=404, detail="Research brief not found")
    
    engine = ResearchEngine()
    
    try:
        plan = await engine.ideas_to_plan(
            brief,
            request.selected_ideas,
            request.timeline_weeks
        )
        
        logger.info(
            "Plan generated",
            brief_id=request.brief_id,
            ideas_count=len(request.selected_ideas),
            initiatives_count=len(plan.get("initiatives", []))
        )
        
        return plan
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Plan generation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Plan generation failed: {str(e)}")


@app.post("/research/subscribe")
async def subscribe_to_research(request: SubscriptionRequest) -> Dict[str, Any]:
    """
    Subscribe to periodic research updates
    
    This endpoint creates a subscription that will:
    1. Re-run the research query periodically
    2. Compare with previous results
    3. Send notifications about new findings
    """
    
    # This is a placeholder - implement with a task queue in production
    subscription_id = str(uuid.uuid4())
    
    logger.info(
        "Research subscription created",
        subscription_id=subscription_id,
        query=request.query,
        cadence=request.cadence
    )
    
    return {
        "subscription_id": subscription_id,
        "query": request.query,
        "cadence": request.cadence,
        "status": "active",
        "next_run": "Based on cadence",
        "message": "Subscription created. Implementation requires task queue setup."
    }


@app.get("/research/list")
async def list_research_briefs(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0)
) -> Dict[str, Any]:
    """List all research briefs (paginated)"""
    
    brief_ids = list(research_store.keys())
    total = len(brief_ids)
    
    # Paginate
    paginated_ids = brief_ids[offset:offset + limit]
    
    briefs = []
    for brief_id in paginated_ids:
        brief = research_store[brief_id]
        briefs.append({
            "brief_id": brief.brief_id,
            "query": brief.query,
            "date": brief.date.isoformat(),
            "findings_count": len(brief.findings),
            "ideas_count": len(brief.ideas),
            "total_sources": brief.total_sources,
            "average_confidence": brief.average_confidence
        })
    
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "briefs": briefs
    }


@app.delete("/research/brief/{brief_id}")
async def delete_research_brief(brief_id: str) -> Dict[str, str]:
    """Delete a research brief"""
    
    if brief_id not in research_store:
        raise HTTPException(status_code=404, detail="Research brief not found")
    
    del research_store[brief_id]
    logger.info("Research brief deleted", brief_id=brief_id)
    
    return {"message": "Research brief deleted", "brief_id": brief_id}


async def cleanup_engine(engine_id: str, delay: int = 300):
    """Clean up research engine after delay"""
    await asyncio.sleep(delay)
    if engine_id in research_engines:
        del research_engines[engine_id]
        logger.info("Research engine cleaned up", engine_id=engine_id)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(
        "Unhandled exception",
        path=request.url.path,
        method=request.method,
        error=str(exc)
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )


# Chat API Endpoints
@app.post("/chat/message", response_model=ChatResponse)
async def send_chat_message(request: ChatRequest) -> ChatResponse:
    """Send a chat message and get response"""
    try:
        response = await chat_engine.process_message(request)
        return response
    except Exception as e:
        logger.error("Chat message failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.get("/chat/conversations")
async def list_conversations(
    user_id: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
) -> Dict[str, Any]:
    """List all conversations"""
    try:
        conversations = await chat_engine.list_conversations(user_id)
        
        # Paginate
        total = len(conversations)
        paginated = conversations[offset:offset + limit]
        
        summaries = []
        for conv in paginated:
            summaries.append(ConversationSummary(
                conversation_id=conv.conversation_id,
                title=conv.title,
                created_at=conv.created_at,
                updated_at=conv.updated_at,
                message_count=conv.get_message_count(),
                last_message=conv.messages[-1].content[:100] if conv.messages else None,
                topics=conv.context.topics,
                status=conv.status
            ))
        
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "conversations": [s.model_dump() for s in summaries]
        }
    except Exception as e:
        logger.error("Failed to list conversations", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/conversation/{conversation_id}")
async def get_conversation(conversation_id: str) -> Dict[str, Any]:
    """Get a specific conversation"""
    try:
        conversation = await chat_engine.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return {
            "conversation_id": conversation.conversation_id,
            "title": conversation.title,
            "created_at": conversation.created_at.isoformat(),
            "updated_at": conversation.updated_at.isoformat(),
            "messages": [
                {
                    "message_id": msg.message_id,
                    "role": msg.role.value,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "type": msg.message_type.value,
                    "metadata": msg.metadata
                }
                for msg in conversation.messages
            ],
            "context": conversation.context.model_dump(),
            "status": conversation.status
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get conversation", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/chat/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str) -> Dict[str, str]:
    """Delete a conversation"""
    try:
        success = await chat_engine.delete_conversation(conversation_id)
        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return {"message": "Conversation deleted", "conversation_id": conversation_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete conversation", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/memory/update")
async def update_memory(update: MemoryUpdate) -> Dict[str, Any]:
    """Update conversation memory"""
    try:
        memory_manager = chat_engine.memory_manager
        
        if update.operation == "add":
            result = await memory_manager.add_to_memory(
                conversation_id=update.conversation_id,
                content=f"{update.key}: {update.value}",
                metadata={"type": update.memory_type}
            )
        elif update.operation == "delete":
            await memory_manager.clear_short_term_memory(update.conversation_id)
            result = {"success": True}
        else:
            result = {"success": False, "error": "Unsupported operation"}
        
        return result
    except Exception as e:
        logger.error("Failed to update memory", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/memory/stats")
async def get_memory_stats(user_id: Optional[str] = Query(None)) -> Dict[str, Any]:
    """Get memory statistics"""
    try:
        stats = await chat_engine.memory_manager.get_memory_stats(user_id)
        return stats
    except Exception as e:
        logger.error("Failed to get memory stats", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket endpoint
@app.websocket("/ws/chat/{connection_id}")
async def websocket_chat(
    websocket: WebSocket,
    connection_id: str,
    user_id: Optional[str] = Query(None)
):
    """WebSocket endpoint for real-time chat"""
    await websocket_endpoint(websocket, connection_id, user_id)


# Serve chat UI
@app.get("/chat", response_class=HTMLResponse)
async def chat_ui():
    """Serve the chat UI"""
    try:
        with open("static/chat.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content=get_default_chat_html())


def get_default_chat_html():
    """Get default chat HTML if file not found"""
    return """<!DOCTYPE html>
<html>
<head>
    <title>Chat Interface</title>
    <style>body { font-family: Arial; } .container { max-width: 800px; margin: 0 auto; padding: 20px; }</style>
</head>
<body>
    <div class="container">
        <h1>Chat Interface</h1>
        <p>Chat interface file not found. Please create static/chat.html</p>
    </div>
</body>
</html>"""


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )