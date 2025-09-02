"""MongoDB MCP Client for read-only database operations"""

import json
import asyncio
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime, timezone
import structlog

from app.config import settings

# Import MCP SDK components (these will be available after pip install mcp)
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

logger = structlog.get_logger()


class MongoDBMCPClient:
    """MCP client for MongoDB read-only operations"""

    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.server_params: Optional[StdioServerParameters] = None
        self.connected = False
        self.available_tools: List[Dict[str, Any]] = []

        # Validate configuration
        if not settings.mongodb_mcp_enabled:
            logger.info("MongoDB MCP integration disabled by configuration")
            return

        if not MCP_AVAILABLE:
            logger.warning("MCP SDK not installed. Install with: pip install mcp")
            return

        # Initialize MCP server parameters
        self._setup_server_params()

    def _setup_server_params(self):
        """Setup MCP server parameters for MongoDB connection"""
        try:
            # For the official mongodb-mcp-server, we need to set up the connection
            # This assumes the MCP server is running as a separate process
            self.server_params = StdioServerParameters(
                command="mongodb-mcp-server",  # This would be the command to run the MCP server
                args=[
                    "--connection-string", settings.mongodb_connection_string,
                    "--database", settings.mongodb_database,
                    "--readonly", "true"
                ],
                env={
                    "MONGODB_USER": settings.mongodb_readonly_user,
                    "MONGODB_PASSWORD": settings.mongodb_readonly_password,
                }
            )
            logger.info("MongoDB MCP server parameters configured")
        except Exception as e:
            logger.error("Failed to setup MCP server parameters", error=str(e))

    async def connect(self) -> bool:
        """Connect to MCP server and list available tools"""
        if not MCP_AVAILABLE or not self.server_params:
            logger.warning("MCP not available or not configured")
            return False

        try:
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    self.session = session
                    self.connected = True

                    # List available tools
                    tools_result = await session.list_tools()
                    self.available_tools = tools_result.tools

                    logger.info(
                        "Connected to MongoDB MCP server",
                        tool_count=len(self.available_tools),
                        tools=[tool.name for tool in self.available_tools]
                    )
                    return True

        except Exception as e:
            logger.error("Failed to connect to MongoDB MCP server", error=str(e))
            self.connected = False
            return False

    async def disconnect(self):
        """Disconnect from MCP server"""
        if self.session:
            # Close the session (this will be handled by the async context manager)
            pass
        self.connected = False
        self.session = None
        logger.info("Disconnected from MongoDB MCP server")

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available MongoDB tools"""
        if not self.connected or not self.session:
            logger.warning("Not connected to MCP server")
            return []

        try:
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    tools_result = await session.list_tools()
                    return [tool.model_dump() for tool in tools_result.tools]
        except Exception as e:
            logger.error("Failed to list tools", error=str(e))
            return []

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Call a MongoDB MCP tool with streaming support"""

        if not self.connected or not self.session:
            yield {"type": "error", "message": "Not connected to MCP server"}
            return

        # Validate collection access
        if "collection" in arguments:
            collection = arguments["collection"]
            if collection not in settings.mongodb_allowed_collections:
                yield {
                    "type": "error",
                    "message": f"Collection '{collection}' not in allowed list",
                    "allowed_collections": settings.mongodb_allowed_collections
                }
                return

        # Add tenant filtering for multi-tenant safety
        if user_id and settings.mongodb_tenant_field:
            arguments = self._add_tenant_filter(arguments, user_id)

        # Log the tool call for audit
        logger.info(
            "Calling MongoDB MCP tool",
            tool_name=tool_name,
            arguments=json.dumps(arguments, default=str),
            user_id=user_id
        )

        try:
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    # Emit tool will use event
                    yield {
                        "type": "tool.will_use",
                        "tool": tool_name,
                        "why": f"Executing {tool_name} on MongoDB collection",
                        "arguments": arguments
                    }

                    # Call the tool
                    result = await session.call_tool(tool_name, arguments)

                    # Process and stream results
                    for content in result.content:
                        if hasattr(content, 'type'):
                            if content.type == "text":
                                # Parse JSON results if possible
                                try:
                                    data = json.loads(content.text)
                                    if isinstance(data, list):
                                        # Stream array results in chunks
                                        for i, item in enumerate(data):
                                            if i >= settings.mongodb_max_rows_per_query:
                                                yield {
                                                    "type": "tool.output.summary",
                                                    "truncated": True,
                                                    "max_rows": settings.mongodb_max_rows_per_query
                                                }
                                                break
                                            yield {
                                                "type": "tool.output.summary",
                                                "data": item,
                                                "row_index": i
                                            }
                                    else:
                                        yield {
                                            "type": "tool.output.summary",
                                            "data": data
                                        }
                                except json.JSONDecodeError:
                                    # Not JSON, return as text
                                    yield {
                                        "type": "tool.output.summary",
                                        "text": content.text
                                    }
                            else:
                                # Handle other content types
                                yield {
                                    "type": "tool.output.summary",
                                    "content": str(content)
                                }

                    # Emit completion
                    yield {
                        "type": "tool.complete",
                        "tool": tool_name,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }

        except Exception as e:
            logger.error("Failed to call MongoDB MCP tool", error=str(e), tool_name=tool_name)
            yield {
                "type": "error",
                "message": f"Tool call failed: {str(e)}",
                "tool": tool_name
            }

    def _add_tenant_filter(self, arguments: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Add tenant filtering to arguments for multi-tenant safety"""
        modified_args = arguments.copy()

        # For find operations
        if "filter" in modified_args:
            if settings.mongodb_tenant_field not in modified_args["filter"]:
                modified_args["filter"][settings.mongodb_tenant_field] = user_id
        else:
            modified_args["filter"] = {settings.mongodb_tenant_field: user_id}

        # For aggregation pipelines
        if "pipeline" in modified_args:
            pipeline = modified_args["pipeline"]
            # Ensure first stage is a $match with tenant filter
            if pipeline and pipeline[0].get("$match"):
                if settings.mongodb_tenant_field not in pipeline[0]["$match"]:
                    pipeline[0]["$match"][settings.mongodb_tenant_field] = user_id
            else:
                # Insert tenant filter at the beginning
                tenant_filter = {"$match": {settings.mongodb_tenant_field: user_id}}
                modified_args["pipeline"].insert(0, tenant_filter)

        return modified_args

    async def find_documents(
        self,
        collection: str,
        filter_query: Optional[Dict[str, Any]] = None,
        projection: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        user_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Find documents in a collection"""

        # Enforce limits
        if limit is None:
            limit = min(100, settings.mongodb_max_rows_per_query)
        else:
            limit = min(limit, settings.mongodb_max_rows_per_query)

        arguments = {
            "collection": collection,
            "filter": filter_query or {},
            "projection": projection or {},
            "limit": limit
        }

        async for result in self.call_tool("mongodb.find", arguments, user_id):
            yield result

    async def aggregate_documents(
        self,
        collection: str,
        pipeline: List[Dict[str, Any]],
        user_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Aggregate documents using MongoDB aggregation pipeline"""

        arguments = {
            "collection": collection,
            "pipeline": pipeline
        }

        async for result in self.call_tool("mongodb.aggregate", arguments, user_id):
            yield result

    async def vector_search(
        self,
        collection: str,
        query_vector: List[float],
        filter_query: Optional[Dict[str, Any]] = None,
        limit: int = 5,
        user_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Perform vector search using Atlas Vector Search"""

        if not settings.mongodb_vector_search_enabled:
            yield {"type": "error", "message": "Vector search not enabled"}
            return

        # Build vector search pipeline
        vector_search_stage = {
            "$vectorSearch": {
                "index": settings.mongodb_vector_index_name,
                "path": settings.mongodb_vector_path,
                "queryVector": query_vector,
                "numCandidates": 200,
                "limit": min(limit, settings.mongodb_max_rows_per_query),
                "filter": filter_query or {}
            }
        }

        pipeline = [
            vector_search_stage,
            {
                "$project": {
                    "_id": 1,
                    "title": 1,
                    "content": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]

        arguments = {
            "collection": collection,
            "pipeline": pipeline
        }

        async for result in self.call_tool("mongodb.aggregate", arguments, user_id):
            yield result

    async def run_command(
        self,
        command: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run a MongoDB database command"""

        # Validate command for read-only operations
        if not self._is_read_only_command(command):
            yield {"type": "error", "message": "Command not allowed - must be read-only"}
            return

        arguments = {
            "command": command
        }

        async for result in self.call_tool("mongodb.runCommand", arguments, user_id):
            yield result

    def _is_read_only_command(self, command: Dict[str, Any]) -> bool:
        """Check if a MongoDB command is read-only"""
        read_only_commands = {
            "find", "aggregate", "count", "distinct", "listCollections",
            "listDatabases", "dbStats", "collStats", "ping", "buildInfo"
        }

        command_name = list(command.keys())[0] if command else ""
        return command_name in read_only_commands

    async def health_check(self) -> Dict[str, Any]:
        """Check MongoDB MCP server health"""
        if not self.connected:
            return {"status": "disconnected", "available_tools": 0}

        tools = await self.list_tools()
        return {
            "status": "connected",
            "available_tools": len(tools),
            "tools": [tool.get("name") for tool in tools]
        }


# Global MongoDB MCP client instance
mongodb_mcp_client = MongoDBMCPClient()
