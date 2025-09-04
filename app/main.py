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

# Global settings info
print(f"🔧 Loading settings - Model: {settings.llm_model}, API Key set: {bool(settings.groq_api_key)}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the application"""
    logger.info("Starting Conversational Chat Engine", version=settings.app_version)

    # Initialize MCP client
    try:
        from app.integrations.mcp_client import initialize_mongodb_mcp_client
        await initialize_mongodb_mcp_client()
    except Exception as e:
        logger.warning("Failed to initialize MCP client", error=str(e))

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
        "debug": settings.debug,
        "features": {
            "chat": False,  # Chat engine removed
            "research": True,
            "memory": False,  # Memory manager was part of chat engine
            "database": True
        }
    }


@app.get("/mcp/status")
async def mcp_status():
    """Get MCP client status"""
    try:
        from app.integrations.mcp_client import mongodb_mcp_client
        # Check if MCP client is connected
        connected = hasattr(mongodb_mcp_client, '_client') and mongodb_mcp_client._client is not None
        return {
            "status": "connected" if connected else "disconnected",
            "connected": connected,
            "available_collections": settings.mongodb_allowed_collections_list,
        }
    except Exception as e:
        logger.error("Failed to get MCP status", error=str(e))
        return {
            "status": "error",
            "error": str(e),
            "connected": False
        }


@app.post("/mcp/reconnect")
async def reconnect_mcp():
    """Manually trigger MCP reconnection"""
    try:
        logger.info("Manually triggering MCP reconnection...")
        from app.integrations.mcp_client import mongodb_mcp_client
        # Try to reconnect
        connected = await mongodb_mcp_client.connect()
        
        return {
            "success": connected,
            "message": "MCP reconnection attempted",
            "status": "connected" if connected else "disconnected",
            "connected": connected
        }
    except Exception as e:
        logger.error("Failed to reconnect MCP", error=str(e))
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to reconnect MCP client"
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
    """Debug endpoint - memory manager has been removed with chat engine"""
    return {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "message": "Memory manager has been removed with the chat engine",
        "total_memories": 0,
        "profile_facts": 0,
        "profile_level_memories": 0,
        "profile": [],
        "profile_level": [],
        "recent_memories": [],
        "search_results": [],
        "short_term_cache": {},
        "active_conversations": [],
        "memory_stats": {
            "user_level": 0,
            "profile_level": 0,
            "conversation_level": 0,
            "pinned": 0
        }
    }




if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )