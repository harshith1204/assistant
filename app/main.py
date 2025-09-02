"""Main FastAPI application for Conversational Chat Engine"""

import warnings
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
import asyncio

from fastapi import FastAPI, WebSocket, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import structlog
import uvicorn

from app.config import settings
from app.core.chat_engine import ChatEngine
from app.websocket_handler import websocket_endpoint

# Suppress websockets library deprecation warning
warnings.filterwarnings("ignore", message="remove second argument of ws_handler", category=DeprecationWarning, module="websockets")

# Suppress Pydantic class-based config warning
warnings.filterwarnings("ignore", message="Support for class-based `config` is deprecated", category=DeprecationWarning, module="pydantic")

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)

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

# Global chat engine instance
print(f"🔧 Loading settings - Model: {settings.llm_model}, API Key set: {bool(settings.groq_api_key)}")
chat_engine = ChatEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the application"""
    logger.info("Starting Conversational Chat Engine", version=settings.app_version)
    yield
    logger.info("Shutting down Conversational Chat Engine")


# Create FastAPI app
app = FastAPI(
    title="Conversational Chat Engine",
    version=settings.app_version,
    description="AI-powered conversational chat with keyword-based processing",
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
        "name": "Conversational Chat Engine",
        "version": settings.app_version,
        "status": "operational",
        "websocket_endpoint": "/ws/chat/{connection_id}"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": settings.app_version,
        "debug": settings.debug
    }


# WebSocket endpoint for conversational chat
@app.websocket("/ws/chat/{connection_id}")
async def websocket_chat(
    websocket: WebSocket,
    connection_id: str,
    user_id: Optional[str] = None
):
    """WebSocket endpoint for real-time conversational chat"""
    await websocket_endpoint(websocket, connection_id, user_id)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(
        "Unhandled exception",
        path=request.url.path,
        method=request.method,
        error=str(exc)
    )
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )



@app.get("/chat/memory/debug/{user_id}")
async def debug_user_memory(
    user_id: str,
    conversation_id: Optional[str] = Query(None)
) -> Dict[str, Any]:
    """Debug endpoint to view all memories for a user"""
    try:
        memory_manager = chat_engine.memory_manager
        
        # Get all user memories
        all_memories = await asyncio.to_thread(
            memory_manager.memory.get_all,
            user_id=user_id
        )
        
        # Extract results
        if isinstance(all_memories, dict) and "results" in all_memories:
            memories = all_memories["results"]
        elif isinstance(all_memories, list):
            memories = all_memories
        else:
            memories = []
        
        # Get profile
        profile = await memory_manager.get_profile(user_id)
        
        # Search memories if query provided
        search_results = []
        if conversation_id:
            search_results = await memory_manager.search_memory(
                query="*",  # Get all
                conversation_id=conversation_id,
                user_id=user_id,
                limit=50,
                search_scope="both"
            )
        
        # Get short-term cache for conversation
        short_term = {}
        if conversation_id and conversation_id in memory_manager.short_term_cache:
            cache_data = memory_manager.short_term_cache[conversation_id]
            short_term = {
                "message_count": len(cache_data.get("messages", [])),
                "recent_messages": cache_data.get("messages", [])[-5:],
                "timestamp": cache_data.get("timestamp", "").isoformat() if cache_data.get("timestamp") else None
            }
       
        return {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "total_memories": len(memories),
            "profile_facts": len(profile),
            "profile": profile[:10],  # First 10 profile facts
            "recent_memories": memories[:20],  # Recent 20 memories
            "search_results": search_results[:10],  # Top 10 search results
            "short_term_cache": short_term,
            "active_conversations": list(memory_manager.short_term_cache.keys())
        }
    except Exception as e:
        logger.error("Failed to debug memory", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))




if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )