"""MongoDB MCP Client for database operations"""

import json
import asyncio
import uuid
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime, timezone
import structlog

from app.config import settings

# Import aiohttp for direct HTTP requests
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

logger = structlog.get_logger()


class MongoMCP:
    """Robust MCP client for MongoDB operations"""

    def __init__(self, base_url: str, max_rows: int = 200):
        self.base_url = base_url.rstrip("/")
        self.session_id: Optional[str] = None
        self.max_rows = max_rows

    async def _ensure_session(self) -> None:
        if self.session_id:
            return

        try:
            async with aiohttp.ClientSession() as http:
                # Try to create a session (common pattern)
                async with http.post(f"{self.base_url}/sessions") as r:
                    if r.status == 404:
                        # Some servers combine session + initialize; fall back to initialize-only
                        self.session_id = str(uuid.uuid4())
                        return
                    r.raise_for_status()
                    data = await r.json()
                    self.session_id = data.get("id") or data.get("session_id") or str(uuid.uuid4())

                # Initialize the session
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"client": "chat-agent", "protocol": "1.0", "capabilities": ["tools"]},
                }
                async with aiohttp.ClientSession() as http2:
                    async with http2.post(f"{self.base_url}/messages?session_id={self.session_id}", json=payload) as r2:
                        r2.raise_for_status()
                        # Not all servers return a payload here; ignore content
        except Exception as e:
            logger.warning("Failed to establish MCP session, proceeding without session", error=str(e))
            self.session_id = None  # Explicitly set to None to indicate no session

    async def list_tools(self) -> List[str]:
        await self._ensure_session()
        payload = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        url = f"{self.base_url}/messages" + (f"?session_id={self.session_id}" if self.session_id else "")
        async with aiohttp.ClientSession() as http:
            async with http.post(url, json=payload) as r:
                r.raise_for_status()
                data = await r.json()
                tools = data.get("result", {}).get("tools", [])
                return [t.get("name") for t in tools if isinstance(t, dict)]

    async def call_tool(self, tool: str, args: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        await self._ensure_session()
        payload = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": tool, "arguments": args},
        }
        url = f"{self.base_url}/messages" + (f"?session_id={self.session_id}" if self.session_id else "")
        async with aiohttp.ClientSession() as http:
            async with http.post(url, json=payload) as r:
                r.raise_for_status()
                data = await r.json()

        # Robust handling for various MCP server response formats
        tool_result = data.get("result", {})

        # 0) If server returned a raw array or object at top-level result, yield it directly
        if isinstance(tool_result, list):
            for i, item in enumerate(tool_result):
                if i >= self.max_rows:
                    yield {"type": "tool.output.summary", "truncated": True, "max_rows": self.max_rows}
                    break
                yield {"type": "tool.output.data", "data": item}
            return

        if isinstance(tool_result, dict) and any(k in tool_result for k in ("docs", "items", "data")):
            seq = tool_result.get("docs") or tool_result.get("items") or tool_result.get("data")
            if isinstance(seq, list):
                for i, item in enumerate(seq):
                    if i >= self.max_rows:
                        yield {"type": "tool.output.summary", "truncated": True, "max_rows": self.max_rows}
                        break
                    yield {"type": "tool.output.data", "data": item}
                return

        # 1) Handle MCP format with content chunks
        if isinstance(tool_result, dict) and "content" in tool_result:
            for content_item in tool_result["content"]:
                if isinstance(content_item, dict) and "text" in content_item:
                    try:
                        # Parse the JSON response from MCP server
                        parsed_result = json.loads(content_item["text"])

                        # Extract documents array from the MCP server response format
                        if isinstance(parsed_result, dict) and "documents" in parsed_result:
                            documents = parsed_result["documents"]
                            if isinstance(documents, list):
                                logger.debug(f"Extracted {len(documents)} documents from MCP response")
                                for i, doc in enumerate(documents):
                                    if i >= self.max_rows:
                                        yield {"type": "tool.output.summary", "truncated": True, "max_rows": self.max_rows}
                                        return
                                    yield {"type": "tool.output.data", "data": doc}
                                return
                        elif isinstance(parsed_result, dict) and "result" in parsed_result:
                            # Handle nested result format
                            result_data = parsed_result["result"]
                            if isinstance(result_data, list):
                                for i, item in enumerate(result_data):
                                    if i >= self.max_rows:
                                        yield {"type": "tool.output.summary", "truncated": True, "max_rows": self.max_rows}
                                        return
                                    yield {"type": "tool.output.data", "data": item}
                                return
                            else:
                                yield {"type": "tool.output.data", "data": result_data}
                                return

                        # If no documents/result found, yield the whole parsed result
                        yield {"type": "tool.output.data", "data": parsed_result}

                    except (json.JSONDecodeError, TypeError) as e:
                        # If not JSON or parsing fails, yield the text directly
                        logger.warning("Failed to parse MCP response as JSON", error=str(e), text=content_item["text"][:200])
                        yield {"type": "tool.output.data", "data": content_item["text"]}
                else:
                    yield {"type": "tool.output.data", "data": content_item}
        else:
            # 2) Fallback: yield the entire result as-is
            yield {"type": "tool.output.data", "data": tool_result}

    # Convenience wrappers
    async def find(self, collection: str, filter: Dict[str, Any] = None, projection: Dict[str, int] = None, limit: int = 100):
        args = {
            "collection": collection,
            "filter": filter or {},
            "projection": projection or {},
            "limit": min(limit, self.max_rows),
        }
        async for row in self.call_tool("mongodb.find", args):
            yield row

    async def aggregate(self, collection: str, pipeline: List[Dict[str, Any]], limit: int = 200):
        args = {"collection": collection, "pipeline": pipeline, "limit": min(limit, self.max_rows)}
        async for row in self.call_tool("mongodb.aggregate", args):
            yield row

    async def insert_one(self, collection: str, document: Dict[str, Any]):
        args = {"collection": collection, "document": document}
        out = None
        async for row in self.call_tool("mongodb.insertOne", args):
            out = row
        return out


class MongoDBMCPClient:
    """Enhanced MCP client with convenience methods for CRM/PM/HRMS"""

    def __init__(self):
        self.mcp: Optional[MongoMCP] = None
        self.connected = False

        # Validate configuration
        if not settings.mongodb_mcp_enabled:
            logger.info("MongoDB MCP integration disabled by configuration")
            return

        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not installed. Install with: pip install aiohttp")
            return

        # Initialize MCP client
        self._setup_client()

    def _setup_client(self):
        """Setup MCP client"""
        try:
            base_url = settings.mongodb_mcp_server_url.rstrip('/')
            self.mcp = MongoMCP(base_url, settings.mongodb_max_rows_per_query)
            logger.info("MongoDB MCP client configured", url=base_url)
        except Exception as e:
            logger.error("Failed to setup MCP client", error=str(e))

    async def connect(self) -> bool:
        """Connect to MCP server"""
        if not self.mcp:
            logger.warning("MCP client not configured")
            return False

        if self.connected:
            logger.info("MCP client already connected")
            return True

        try:
            # Test connection by listing tools
            tools = await self.mcp.list_tools()
            logger.info(f"MCP client connected, found {len(tools)} tools", tools=tools)
            self.connected = True
            return True
        except Exception as e:
            logger.error("Failed to connect MCP client", error=str(e))
            self.connected = False
            return False

    async def disconnect(self):
        """Disconnect from MCP server"""
        self.connected = False
        logger.info("Disconnected from MongoDB MCP server")

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available MongoDB tools"""
        if not self.mcp:
            logger.warning("MCP client not initialized")
            return []

        try:
            # Try to list tools even if not marked as connected
            # The MCP client itself will handle session establishment
            tools = await self.mcp.list_tools()
            # If we get here, mark as connected for future calls
            self.connected = True
            return [{"name": tool} for tool in tools]
        except Exception as e:
            logger.error("Failed to list tools", error=str(e))
            self.connected = False
            return []

    async def ensure_connected(self) -> bool:
        """Ensure MCP client is connected, reconnect if needed"""
        if not self.connected:
            logger.info("MCP client not connected, attempting to reconnect...")
            return await self.connect()
        return True

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Call a MongoDB MCP tool with streaming support"""

        # Ensure we're connected before proceeding
        if not await self.ensure_connected():
            yield {"type": "error", "message": "Failed to connect to MCP server"}
            return

        if not self.mcp:
            yield {"type": "error", "message": "MCP client not initialized"}
            return

        # Validate collection access
        if "collection" in arguments:
            collection = arguments["collection"]
            if collection not in settings.mongodb_allowed_collections_list:
                yield {
                    "type": "error",
                    "message": f"Collection '{collection}' not in allowed list",
                    "allowed_collections": settings.mongodb_allowed_collections_list
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
            # Emit tool will use event
            yield {
                "type": "tool.will_use",
                "tool": tool_name,
                "why": f"Executing {tool_name} on MongoDB collection",
                "arguments": arguments
            }

            # Call the tool using our MCP client
            async for result in self.mcp.call_tool(tool_name, arguments):
                yield result

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

    # CRM Operations
    async def find_crm_leads(self, filter_query: Optional[Dict[str, Any]] = None, limit: int = 50, user_id: Optional[str] = None):
        """Find CRM leads"""
        projection = {"_id": 0, "name": 1, "email": 1, "stage": 1, "owner": 1, "value": 1, "created_at": 1}
        async for result in self.mcp.find("crm_leads", filter_query, projection, limit):
            yield result

    async def find_crm_accounts(self, filter_query: Optional[Dict[str, Any]] = None, limit: int = 50, user_id: Optional[str] = None):
        """Find CRM accounts"""
        projection = {"_id": 0, "name": 1, "domain": 1, "industry": 1, "owner": 1, "created_at": 1}
        async for result in self.mcp.find("crm_accounts", filter_query, projection, limit):
            yield result

    # PM Operations
    async def find_pm_tasks(self, filter_query: Optional[Dict[str, Any]] = None, limit: int = 100, user_id: Optional[str] = None):
        """Find PM tasks"""
        projection = {"_id": 0, "title": 1, "assignee": 1, "status": 1, "priority": 1, "due": 1, "project_id": 1}
        async for result in self.mcp.find("pm_tasks", filter_query, projection, limit):
            yield result

    async def find_pm_projects(self, filter_query: Optional[Dict[str, Any]] = None, limit: int = 50, user_id: Optional[str] = None):
        """Find PM projects"""
        projection = {"_id": 0, "name": 1, "status": 1, "owner": 1, "due": 1, "created_at": 1}
        async for result in self.mcp.find("pm_projects", filter_query, projection, limit):
            yield result

    # Staff/HRMS Operations
    async def find_staff_directory(self, filter_query: Optional[Dict[str, Any]] = None, limit: int = 200, user_id: Optional[str] = None):
        """Find staff directory entries"""
        projection = {"_id": 0, "name": 1, "email": 1, "role": 1, "manager": 1, "department": 1}
        async for result in self.mcp.find("staff_directory", filter_query, projection, limit):
            yield result

    async def find_hrms_leaves(self, filter_query: Optional[Dict[str, Any]] = None, limit: int = 100, user_id: Optional[str] = None):
        """Find HRMS leave records"""
        projection = {"_id": 0, "user_id": 1, "type": 1, "start_date": 1, "end_date": 1, "status": 1}
        async for result in self.mcp.find("hrms_leaves", filter_query, projection, limit):
            yield result

    # Aggregation helpers
    async def aggregate_crm_stats(self, user_id: Optional[str] = None):
        """Get CRM statistics"""
        pipeline = [
            {"$group": {"_id": "$stage", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        async for result in self.mcp.aggregate("crm_leads", pipeline, 50):
            yield result

    async def aggregate_pm_stats(self, user_id: Optional[str] = None):
        """Get PM statistics"""
        pipeline = [
            {"$group": {"_id": "$status", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        async for result in self.mcp.aggregate("pm_tasks", pipeline, 50):
            yield result

    # Generic methods for any collection
    async def find_documents(self, collection: str, filter_query: Optional[Dict[str, Any]] = None, projection: Optional[Dict[str, Any]] = None, limit: int = 50, user_id: Optional[str] = None):
        """Generic find method for any collection"""
        max_rows = self.mcp.max_rows if self.mcp else 1000
        args = {
            "collection": collection,
            "filter": filter_query or {},
            "projection": projection or {},
            "limit": min(limit, max_rows),
        }
        async for result in self.call_tool("mongodb.find", args):
            yield result

    async def aggregate_documents(self, collection: str, pipeline: List[Dict[str, Any]], limit: int = 100, user_id: Optional[str] = None):
        """Generic aggregate method for any collection"""
        max_rows = self.mcp.max_rows if self.mcp else 1000
        args = {
            "collection": collection,
            "pipeline": pipeline,
            "limit": min(limit, max_rows),
        }
        async for result in self.call_tool("mongodb.aggregate", args):
            yield result

    # Insert operations
    async def create_crm_note(self, lead_id: str, subject: str, description: str, user_id: Optional[str] = None):
        """Create a CRM note"""
        doc = {
            "_id": str(uuid.uuid4()),
            "lead_id": lead_id,
            "subject": subject,
            "description": description,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "user_id": user_id
        }
        return await self.mcp.insert_one("crm_notes", doc)

    async def create_pm_task(self, title: str, description: str, assignee: str, project_id: str, user_id: Optional[str] = None):
        """Create a PM task"""
        doc = {
            "_id": str(uuid.uuid4()),
            "title": title,
            "description": description,
            "assignee": assignee,
            "project_id": project_id,
            "status": "todo",
            "priority": "medium",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "user_id": user_id
        }
        return await self.mcp.insert_one("pm_tasks", doc)

    async def health_check(self) -> Dict[str, Any]:
        """Check MongoDB MCP server health"""
        if not self.connected:
            return {"status": "disconnected", "available_tools": 0}

        # Get tools and update cache
        tools = await self.list_tools()

        return {
            "status": "connected",
            "available_tools": len(tools),
            "tools": [tool.get("name") for tool in tools]
        }

# Global MongoDB MCP client instance
mongodb_mcp_client = MongoDBMCPClient()
