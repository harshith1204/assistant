#!/usr/bin/env python3
"""MongoDB MCP Server with HTTP/SSE transport"""

import os
import sys
import asyncio
import json
import uuid
from typing import Dict, Any, List, Optional
import structlog

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.config import settings

# Import FastAPI and MCP components
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn

# Import MCP server components
try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# Import MongoDB driver
try:
    import motor.motor_asyncio
    import pymongo
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False

logger = structlog.get_logger()

class MongoDBMCPServer:
    """MCP Server for MongoDB operations with HTTP/SSE transport"""

    def __init__(self):
        self.server = None
        self.mongodb_client = None
        self.database = None
        self.sessions: Dict[str, Dict[str, Any]] = {}

        if not MCP_AVAILABLE:
            logger.error("MCP SDK not installed")
            sys.exit(1)

        if not MONGODB_AVAILABLE:
            logger.error("MongoDB drivers not installed")
            sys.exit(1)

        # Initialize MongoDB connection
        self._init_mongodb()

        # Create MCP server
        self._create_server()

    def _init_mongodb(self):
        """Initialize MongoDB connection"""
        try:
            # Check if MongoDB connection string is configured
            if not settings.mongodb_connection_string or settings.mongodb_connection_string == "":
                logger.warning("MongoDB connection string not configured. Set MONGODB_CONNECTION_STRING in environment variables.")
                self.mongodb_client = None
                self.database = None
                return

            # Create async MongoDB client
            self.mongodb_client = motor.motor_asyncio.AsyncIOMotorClient(
                settings.mongodb_connection_string
            )

            # Get database
            self.database = self.mongodb_client[settings.mongodb_database]

            logger.info(
                "Connected to MongoDB",
                database=settings.mongodb_database,
                connection_string=settings.mongodb_connection_string.replace(settings.mongodb_readonly_password, "***") if settings.mongodb_readonly_password else settings.mongodb_connection_string
            )

        except Exception as e:
            logger.error("Failed to connect to MongoDB", error=str(e))
            logger.warning("MCP server will continue running but MongoDB operations will be unavailable")
            self.mongodb_client = None
            self.database = None

    def _create_server(self):
        """Create MCP server instance"""
        self.server = Server("mongodb-mcp-server")

        # Store available tools
        self.available_tools = [
            Tool(
                name="mongodb.find",
                description="Find documents in a MongoDB collection",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "collection": {
                            "type": "string",
                            "description": "Collection name"
                        },
                        "filter": {
                            "type": "object",
                            "description": "Query filter"
                        },
                        "projection": {
                            "type": "object",
                            "description": "Field projection"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of documents to return",
                            "default": 100
                        }
                    },
                    "required": ["collection"]
                }
            ),
            Tool(
                name="mongodb.aggregate",
                description="Perform MongoDB aggregation pipeline",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "collection": {
                            "type": "string",
                            "description": "Collection name"
                        },
                        "pipeline": {
                            "type": "array",
                            "description": "Aggregation pipeline stages"
                        }
                    },
                    "required": ["collection", "pipeline"]
                }
            ),
            Tool(
                name="mongodb.insertOne",
                description="Insert a single document into MongoDB collection",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "collection": {
                            "type": "string",
                            "description": "Collection name"
                        },
                        "document": {
                            "type": "object",
                            "description": "Document to insert"
                        }
                    },
                    "required": ["collection", "document"]
                }
            ),
            Tool(
                name="mongodb.runCommand",
                description="Run a MongoDB database command",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "object",
                            "description": "MongoDB command to run"
                        }
                    },
                    "required": ["command"]
                }
            )
        ]

    async def _handle_find(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle find operation"""
        # Check if MongoDB is connected
        if self.mongodb_client is None or self.database is None:
            return [TextContent(
                type="text",
                text="MongoDB not connected. Please configure MONGODB_CONNECTION_STRING environment variable."
            )]

        collection_param = arguments.get("collection")
        filter_query = arguments.get("filter", {})
        projection = arguments.get("projection", {})
        limit = min(arguments.get("limit", 100), settings.mongodb_max_rows_per_query)

        # Parse collection parameter - support database.collection format
        if "." in collection_param:
            db_name, collection_name = collection_param.split(".", 1)
            target_database = self.mongodb_client[db_name]
        else:
            collection_name = collection_param
            target_database = self.database

        # Check if collection is allowed (allow all collections for now, or check against allowed list)
        # For cross-database support, we'll be more permissive
        allowed_collections = settings.mongodb_allowed_collections_list
        if collection_name not in allowed_collections and collection_param not in allowed_collections:
            return [TextContent(
                type="text",
                text=f"Collection '{collection_param}' not in allowed list. Allowed: {allowed_collections}"
            )]

        try:
            collection = target_database[collection_name]

            # Execute find query
            cursor = collection.find(filter_query, projection).limit(limit)
            documents = await cursor.to_list(length=limit)

            result = {
                "operation": "find",
                "database": db_name if "." in collection_param else settings.mongodb_database,
                "collection": collection_name,
                "filter": filter_query,
                "count": len(documents),
                "documents": documents
            }

            return [TextContent(
                type="text",
                text=json.dumps(result, default=str, indent=2)
            )]

        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Find operation failed: {str(e)}"
            )]

    async def _handle_aggregate(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle aggregation operation"""
        # Check if MongoDB is connected
        if self.mongodb_client is None or self.database is None:
            return [TextContent(
                type="text",
                text="MongoDB not connected. Please configure MONGODB_CONNECTION_STRING environment variable."
            )]

        collection_param = arguments.get("collection")
        pipeline = arguments.get("pipeline", [])

        # Parse collection parameter - support database.collection format
        if "." in collection_param:
            db_name, collection_name = collection_param.split(".", 1)
            target_database = self.mongodb_client[db_name]
        else:
            collection_name = collection_param
            target_database = self.database

        # Check if collection is allowed
        allowed_collections = settings.mongodb_allowed_collections_list
        if collection_name not in allowed_collections and collection_param not in allowed_collections:
            return [TextContent(
                type="text",
                text=f"Collection '{collection_param}' not in allowed list. Allowed: {allowed_collections}"
            )]

        try:
            collection = target_database[collection_name]

            # Execute aggregation pipeline
            cursor = collection.aggregate(pipeline)
            results = await cursor.to_list(length=settings.mongodb_max_rows_per_query)

            result = {
                "operation": "aggregate",
                "database": db_name if "." in collection_param else settings.mongodb_database,
                "collection": collection_name,
                "pipeline": pipeline,
                "count": len(results),
                "results": results
            }

            return [TextContent(
                type="text",
                text=json.dumps(result, default=str, indent=2)
            )]

        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Aggregation operation failed: {str(e)}"
            )]

    async def _handle_insert_one(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle insert one operation"""
        # Check if MongoDB is connected
        if self.mongodb_client is None or self.database is None:
            return [TextContent(
                type="text",
                text="MongoDB not connected. Please configure MONGODB_CONNECTION_STRING environment variable."
            )]

        collection_param = arguments.get("collection")
        document = arguments.get("document", {})

        # Parse collection parameter - support database.collection format
        if "." in collection_param:
            db_name, collection_name = collection_param.split(".", 1)
            target_database = self.mongodb_client[db_name]
        else:
            collection_name = collection_param
            target_database = self.database

        # Check if collection is allowed
        allowed_collections = settings.mongodb_allowed_collections_list
        if collection_name not in allowed_collections and collection_param not in allowed_collections:
            return [TextContent(
                type="text",
                text=f"Collection '{collection_param}' not in allowed list. Allowed: {allowed_collections}"
            )]

        try:
            collection = target_database[collection_name]

            # Insert document
            result = await collection.insert_one(document)

            response = {
                "operation": "insertOne",
                "database": db_name if "." in collection_param else settings.mongodb_database,
                "collection": collection_name,
                "inserted_id": str(result.inserted_id),
                "acknowledged": result.acknowledged
            }

            return [TextContent(
                type="text",
                text=json.dumps(response, indent=2)
            )]

        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Insert operation failed: {str(e)}"
            )]

    async def _handle_run_command(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle run command operation"""
        # Check if MongoDB is connected
        if self.mongodb_client is None or self.database is None:
            return [TextContent(
                type="text",
                text="MongoDB not connected. Please configure MONGODB_CONNECTION_STRING environment variable."
            )]

        command = arguments.get("command", {})

        try:
            # Run database command
            result = await self.database.command(command)

            response = {
                "operation": "runCommand",
                "command": command,
                "result": result
            }

            return [TextContent(
                type="text",
                text=json.dumps(response, default=str, indent=2)
            )]

        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Command execution failed: {str(e)}"
            )]


# Global MCP server instance
mcp_server = MongoDBMCPServer()

# FastAPI app
app = FastAPI(
    title="MongoDB MCP Server",
    version="1.0.0",
    description="MCP Server for MongoDB operations with HTTP/SSE transport"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "MongoDB MCP Server",
        "version": "1.0.0",
        "status": "running",
        "sse_endpoint": "/sse",
        "messages_endpoint": "/messages"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "database": settings.mongodb_database,
        "allowed_collections": settings.mongodb_allowed_collections_list
    }


@app.get("/sse")
async def sse_endpoint():
    """Server-Sent Events endpoint for MCP clients"""
    async def event_generator():
        session_id = str(uuid.uuid4())

        # Send session ID
        yield f"data: /messages/?session_id={session_id}\n\n"

        # Keep connection alive
        while True:
            await asyncio.sleep(30)
            yield ": keepalive\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control",
        }
    )


@app.post("/messages/")
async def handle_message(request: Request, session_id: Optional[str] = None):
    """Handle MCP messages"""
    try:
        body = await request.json()
        logger.info("Received MCP message", session_id=session_id, method=body.get("method"))

        # Handle initialization
        if body.get("method") == "initialize":
            response = {
                "jsonrpc": "2.0",
                "id": body.get("id"),
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {
                            "listChanged": True
                        }
                    },
                    "serverInfo": {
                        "name": "mongodb-mcp-server",
                        "version": "1.0.0"
                    }
                }
            }
            return response

        # Handle tools/list
        elif body.get("method") == "tools/list":
            try:
                response = {
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "result": {
                        "tools": [
                            {
                                "name": tool.name,
                                "description": tool.description,
                                "inputSchema": tool.inputSchema
                            }
                            for tool in mcp_server.available_tools
                        ]
                    }
                }
                return response
            except Exception as tool_error:
                logger.error("Error listing tools", error=str(tool_error))
                return {
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "error": {
                        "code": -32603,
                        "message": f"Error listing tools: {str(tool_error)}"
                    }
                }

        # Handle tools/call
        elif body.get("method") == "tools/call":
            try:
                params = body.get("params", {})
                tool_name = params.get("name")
                arguments = params.get("arguments", {})

                # Call the appropriate tool handler
                result = []
                if tool_name == "mongodb.find":
                    result = await mcp_server._handle_find(arguments)
                elif tool_name == "mongodb.aggregate":
                    result = await mcp_server._handle_aggregate(arguments)
                elif tool_name == "mongodb.insertOne":
                    result = await mcp_server._handle_insert_one(arguments)
                elif tool_name == "mongodb.runCommand":
                    result = await mcp_server._handle_run_command(arguments)
                else:
                    result = [TextContent(
                        type="text",
                        text=f"Unknown tool: {tool_name}"
                    )]

                response = {
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "result": {
                        "content": [
                            {
                                "type": content.type,
                                "text": content.text
                            }
                            for content in result
                        ]
                    }
                }
                return response
            except Exception as call_error:
                logger.error("Error calling tool", error=str(call_error))
                return {
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "error": {
                        "code": -32603,
                        "message": f"Error calling tool: {str(call_error)}"
                    }
                }

        # Unknown method
        else:
            response = {
                "jsonrpc": "2.0",
                "id": body.get("id"),
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {body.get('method')}"
                }
            }
            return response

    except Exception as e:
        logger.error("Error handling MCP message", error=str(e))
        return {
            "jsonrpc": "2.0",
            "id": body.get("id") if 'body' in locals() else None,
            "error": {
                "code": -32603,
                "message": f"Internal error: {str(e)}"
            }
        }


if __name__ == "__main__":
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

    logger.info("Starting MongoDB MCP Server on port 8001")
    logger.info(f"Database: {settings.mongodb_database}")
    logger.info(f"Allowed collections: {settings.mongodb_allowed_collections_list}")

    # Run the server
    uvicorn.run(
        "mcp_server:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )
