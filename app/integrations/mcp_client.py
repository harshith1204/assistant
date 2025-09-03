"""Optimized MongoDB MCP Client for high-performance database operations

This client combines the best of both worlds:
- Official MCP protocol compliance with stdio-based communication
- High-performance optimizations (connection pooling, caching, parallel processing)
- LLM integration for intelligent query processing
"""

import json
import asyncio
import uuid
import sys
import mcp
from typing import Dict, Any, List, Optional, AsyncGenerator, Union
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from contextlib import AsyncExitStack
import structlog
from concurrent.futures import ThreadPoolExecutor

from app.config import settings

# Import official MCP components
try:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client, StdioServerParameters
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# Import aiohttp for direct HTTP requests (fallback)
try:
    import aiohttp
    from aiohttp import ClientTimeout
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

logger = structlog.get_logger()


class SmartCache:
    """Intelligent caching system for MCP results"""

    def __init__(self, maxsize: int = 1000, ttl_minutes: int = 30):
        self.cache = {}
        self.maxsize = maxsize
        self.ttl_minutes = ttl_minutes

    def _is_valid(self, entry: Dict[str, Any]) -> bool:
        """Check if cache entry is still valid"""
        if 'timestamp' not in entry:
            return False
        age = datetime.now(timezone.utc) - entry['timestamp']
        return age < timedelta(minutes=self.ttl_minutes)

    def get(self, key: str) -> Optional[Any]:
        """Get cached result if valid"""
        if key in self.cache and self._is_valid(self.cache[key]):
            logger.debug("Cache hit", key=key)
            self.cache[key]['access_count'] += 1  # Track access count
            return self.cache[key]['data']
        elif key in self.cache:
            # Remove expired entry
            del self.cache[key]
        return None

    def set(self, key: str, data: Any) -> None:
        """Cache result with timestamp"""
        if len(self.cache) >= self.maxsize:
            # Remove oldest entry (simple LRU approximation)
            oldest_key = min(self.cache.keys(),
                           key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]

        self.cache[key] = {
            'data': data,
            'timestamp': datetime.now(timezone.utc),
            'access_count': 0,
            'size': len(str(data)) if data else 0  # Rough size estimation
        }
        logger.debug("Cache set", key=key, size=self.cache[key]['size'])

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_size = sum(entry.get('size', 0) for entry in self.cache.values())
        avg_access = sum(entry.get('access_count', 0) for entry in self.cache.values()) / max(len(self.cache), 1)

        return {
            'size': len(self.cache),
            'max_size': self.maxsize,
            'total_data_size': total_size,
            'avg_access_count': avg_access,
            'hit_rate': self._calculate_hit_rate(),
            'oldest_entry': min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp']) if self.cache else None
        }

    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate (simplified)"""
        total_accesses = sum(entry.get('access_count', 0) for entry in self.cache.values())
        if total_accesses == 0:
            return 0.0
        # This is a simplified calculation - in practice you'd track hits vs misses
        return min(total_accesses / len(self.cache), 1.0)


class QueryOptimizer:
    """Optimizes MongoDB queries for better performance"""

    @staticmethod
    def create_combined_analysis_pipeline(collection_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create optimized $facet aggregation for comprehensive analysis"""
        return [{
            "$facet": {
                "overview": [{
                    "$group": {
                        "_id": None,
                        "total_docs": {"$sum": 1},
                        "avg_score": {"$avg": "$score"},
                        "last_updated": {"$max": "$updatedTimeStamp"}
                    }
                }],
                "status_distribution": [{
                    "$group": {"_id": "$status", "count": {"$sum": 1}}
                }],
                "source_analysis": [{
                    "$group": {"_id": "$source", "count": {"$sum": 1}}
                }],
                "temporal_patterns": [{
                    "$group": {
                        "_id": {"$dateToString": {"format": "%Y-%m", "date": "$createdTimeStamp"}},
                        "count": {"$sum": 1}
                    }
                }],
                "quality_metrics": [{
                    "$group": {
                        "_id": None,
                        "spam_rate": {"$avg": {"$cond": ["$isSpamOrBot", 1, 0]}},
                        "conversion_rate": {"$avg": {"$cond": [{"$eq": ["$status", "WON"]}, 1, 0]}},
                        "activity_rate": {"$avg": {"$cond": [{"$or": [{"$gt": ["$emailCount", 0]}, {"$gt": ["$callCount", 0]}]}, 1, 0]}}
                    }
                }]
            }
        }]

class ConnectionPool:
    """Connection pool for efficient HTTP requests with Docker-aware settings"""

    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self._session = None
        self._lock = asyncio.Lock()

    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create a pooled session with Docker-optimized timeouts"""
        async with self._lock:
            if self._session is None or self._session.closed:
                # Docker-optimized timeouts: longer connection time, more retries
                timeout = ClientTimeout(
                    total=60,      # Increased from 30s to 60s for Docker
                    connect=20,    # Increased from 10s to 20s for container startup
                    sock_read=30,  # Socket read timeout
                    sock_connect=15  # Socket connect timeout
                )
                connector = aiohttp.TCPConnector(
                    limit=self.max_connections,
                    # Docker networking improvements
                    ttl_dns_cache=300,  # DNS cache TTL for container networking
                    use_dns_cache=True,
                    keepalive_timeout=60,
                    enable_cleanup_closed=True
                )
                self._session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout
                )
            return self._session

    async def close(self):
        """Close the connection pool"""
        async with self._lock:
            if self._session and not self._session.closed:
                await self._session.close()


class ParallelExecutor:
    """Execute multiple MCP queries in parallel"""

    def __init__(self, max_workers: int = 5):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.semaphore = asyncio.Semaphore(max_workers)

    async def execute_parallel(self, tasks: List[Dict[str, Any]]) -> List[Any]:
        """Execute multiple tasks in parallel"""
        async def run_task(task):
            async with self.semaphore:
                return await task['func'](*task.get('args', []), **task.get('kwargs', {}))

        # Create async tasks
        async_tasks = [run_task(task) for task in tasks]

        # Execute all tasks concurrently
        results = await asyncio.gather(*async_tasks, return_exceptions=True)

        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {i} failed", error=str(result))
                processed_results.append(None)
            else:
                processed_results.append(result)

        return processed_results


class MongoMCP:
    """Optimized MCP client with connection pooling and caching"""

    def __init__(self, base_url: str, max_rows: int = 200):
        self.base_url = base_url.rstrip("/")
        self.session_id: Optional[str] = None
        self.max_rows = max_rows

        # Initialize optimized components
        self.connection_pool = ConnectionPool()
        self.cache = SmartCache()
        self.query_optimizer = QueryOptimizer()
        self.parallel_executor = ParallelExecutor()

    async def _ensure_session(self) -> None:
        """Ensure MCP session with connection pooling"""
        if self.session_id:
            return

        try:
            http = await self.connection_pool.get_session()

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
            async with http.post(f"{self.base_url}/messages?session_id={self.session_id}", json=payload) as r2:
                r2.raise_for_status()
                # Not all servers return a payload here; ignore content
        except Exception as e:
            logger.warning("Failed to establish MCP session, proceeding without session", error=str(e))
            self.session_id = None  # Explicitly set to None to indicate no session

    async def list_tools(self) -> List[str]:
        await self._ensure_session()
        payload = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        url = f"{self.base_url}/messages" + (f"?session_id={self.session_id}" if self.session_id else "")

        http = await self.connection_pool.get_session()
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

        http = await self.connection_pool.get_session()
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

    # Write operations DISABLED for security - MCP client is read-only
    # async def insert_one(self, collection: str, document: Dict[str, Any]):
    #     raise PermissionError("Write operations are disabled. MCP client is read-only.")

    # Optimized Analysis Methods
    async def analyze_collection_smart(self, collection: str, analysis_type: str = "auto") -> AsyncGenerator[Dict[str, Any], None]:
        """Smart collection analysis with caching and optimization"""
        cache_key = f"analysis:{collection}:{analysis_type}"

        # Check cache first
        cached_result = self.cache.get(cache_key)
        if cached_result:
            yield {"type": "cache.hit", "data": cached_result}
            return

        # Determine analysis type
        if analysis_type == "auto":
            analysis_type = self._infer_analysis_type(collection)

        # Get optimized pipeline
        pipeline = self.query_optimizer.create_combined_analysis_pipeline({"type": analysis_type})

        # Execute optimized analysis
        async for result in self.aggregate(collection, pipeline):
            if result.get("type") == "tool.output.data":
                data = result.get("data", {})
                # Cache the result
                self.cache.set(cache_key, data)
                yield {"type": "analysis.result", "data": data}
            else:
                yield result

    async def execute_parallel_queries(self, queries: List[Dict[str, Any]]) -> List[Any]:
        """Execute multiple queries in parallel"""
        tasks = []
        for query in queries:
            if query["type"] == "find":
                task = {
                    "func": self.find,
                    "args": [query["collection"]],
                    "kwargs": query.get("params", {})
                }
            elif query["type"] == "aggregate":
                task = {
                    "func": self.aggregate,
                    "args": [query["collection"], query["pipeline"]],
                    "kwargs": query.get("params", {})
                }
            tasks.append(task)

        return await self.parallel_executor.execute_parallel(tasks)

    async def get_collection_metadata(self, collection: str) -> Dict[str, Any]:
        """Get comprehensive collection metadata"""
        cache_key = f"metadata:{collection}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        # Get sample document for schema inference
        sample_docs = []
        async for doc in self.find(collection, limit=5):
            if doc.get("type") == "tool.output.data":
                sample_docs.append(doc.get("data", {}))

        # Infer schema from samples
        schema = self._infer_schema(sample_docs)

        # Get basic stats
        count_result = await self._get_count_result(collection)

        metadata = {
            "collection": collection,
            "estimated_count": count_result,
            "schema": schema,
            "inferred_type": self._infer_analysis_type(collection),
            "last_updated": datetime.now(timezone.utc).isoformat()
        }

        self.cache.set(cache_key, metadata)
        return metadata

    def _infer_schema(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Infer schema from sample documents"""
        if not documents:
            return {}

        schema = {}
        for doc in documents:
            self._update_schema_with_doc(schema, doc)
        return schema

    def _update_schema_with_doc(self, schema: Dict[str, Any], doc: Dict[str, Any]):
        """Update schema with document fields"""
        for key, value in doc.items():
            if key not in schema:
                schema[key] = {"type": type(value).__name__, "count": 1}
            else:
                schema[key]["count"] += 1

    async def _get_count_result(self, collection: str) -> int:
        """Get document count for collection"""
        count_args = {"collection": collection}
        result = None
        async for item in self.call_tool("mongodb.count", count_args):
            if item.get("type") == "tool.output.data":
                result = item.get("data", 0)
                break
        return result or 0

    async def progressive_analysis(self, collection: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Progressive analysis with streaming results"""
        # Phase 1: Basic metadata
        yield {"phase": "metadata", "status": "starting"}
        metadata = await self.get_collection_metadata(collection)
        yield {"phase": "metadata", "status": "complete", "data": metadata}

        # Phase 2: Schema analysis
        yield {"phase": "schema", "status": "starting"}
        yield {"phase": "schema", "status": "complete", "data": metadata["schema"]}

        # Phase 3: Statistical analysis
        yield {"phase": "statistics", "status": "starting"}
        async for result in self.analyze_collection_smart(collection):
            yield {"phase": "statistics", "status": "streaming", "data": result}

        yield {"phase": "statistics", "status": "complete"}

    async def close(self):
        """Cleanup resources"""
        await self.connection_pool.close()
        self.parallel_executor.executor.shutdown()


class OfficialMCPClient:
    """Official MCP client following the Model Context Protocol specification

    This implementation follows the official MCP client tutorial pattern
    but includes our optimizations for better performance.
    """

    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.connected = False

        # Initialize our optimizations
        self.cache = SmartCache()
        self.query_optimizer = QueryOptimizer()

    async def connect_to_mongodb_server(self, connection_string: str = None, read_only: bool = True):
        """Connect to MongoDB MCP server using stdio transport (READ-ONLY ONLY)

        Args:
            connection_string: MongoDB connection string (uses config if not provided)
            read_only: Whether to run in read-only mode (always enforced as True)
        """
        if not MCP_AVAILABLE:
            raise RuntimeError("Official MCP package not installed. Install with: pip install mcp")

        # Use connection string from config if not provided
        if connection_string is None:
            connection_string = settings.mongodb_connection_string
            if not connection_string:
                raise ValueError("No MongoDB connection string provided and none found in config")

        # SECURITY: Always enforce read-only mode for security
        if not read_only:
            logger.warning("⚠️  Read-only mode was requested as False, but enforcing True for security")
        read_only = True  # Always enforce read-only

        # Set up environment variables for MongoDB MCP server
        env = {
            "MDB_MCP_CONNECTION_STRING": connection_string,
            "MDB_MCP_READ_ONLY": "true",  # Always read-only
        }

        # Use npx to run the MongoDB MCP server
        server_params = StdioServerParameters(
            command="npx",
            args=["-y", "mongodb-mcp-server@latest"],
            env=env
        )

        # Setup stdio transport (official MCP way)
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        read_stream, write_stream = stdio_transport

        # Create and initialize session
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )

        await self.session.initialize()
        self.connected = True

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        logger.info(f"Connected to MongoDB MCP server with {len(tools)} tools",
                   tools=[tool.name for tool in tools])

        return tools

    async def connect(self) -> bool:
        """Connect to MCP server (compatibility method for chat engine)"""
        if not MCP_AVAILABLE:
            logger.warning("Official MCP package not available")
            return False

        if self.connected:
            logger.info("MCP client already connected")
            return True

        try:
            logger.info("🔌 Connecting to MongoDB MCP server...")
            await self.connect_to_mongodb_server()
            logger.info("✅ MCP client connected successfully")
            return True
        except Exception as e:
            logger.error("❌ Failed to connect MCP client", error=str(e))
            return False

    async def connect_to_server(self, server_script_path: str):
        """Connect to MCP server following official protocol

        Args:
            server_script_path: Path to server script (.py or .js)
        """
        if not MCP_AVAILABLE:
            raise RuntimeError("Official MCP package not installed. Install with: pip install mcp")

        # Validate server script
        if not (server_script_path.endswith('.py') or server_script_path.endswith('.js')):
            raise ValueError("Server script must be a .py or .js file")

        command = "python3" if server_script_path.endswith('.py') else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        # Setup stdio transport (official MCP way)
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        read_stream, write_stream = stdio_transport

        # Create and initialize session
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )

        await self.session.initialize()
        self.connected = True

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        logger.info(f"Connected to MCP server with {len(tools)} tools",
                   tools=[tool.name for tool in tools])

        return tools

    async def process_query_with_llm(self, query: str) -> str:
        """Process query using Claude and available tools (Official MCP + LLM pattern)"""
        if not self.anthropic:
            raise RuntimeError("Anthropic client not available for LLM processing")

        messages = [{"role": "user", "content": query}]

        # Get available tools from MCP server
        response = await self.session.list_tools()
        available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        # Initial Claude API call
        response = self.anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=messages,
            tools=available_tools
        )

        # Process response and handle tool calls
        final_text = []
        assistant_message_content = []

        for content in response.content:
            if content.type == 'text':
                final_text.append(content.text)
                assistant_message_content.append(content)
            elif content.type == 'tool_use':
                tool_name = content.name
                tool_args = content.input

                # Execute tool call through MCP session
                result = await self.session.call_tool(tool_name, tool_args)
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                assistant_message_content.append(content)
                messages.append({
                    "role": "assistant",
                    "content": assistant_message_content
                })
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": content.id,
                        "content": result.content
                    }]
                })

                # Get next response from Claude
                response = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    messages=messages,
                    tools=available_tools
                )

                if response.content:
                    final_text.append(response.content[0].text)

        return "\n".join(final_text)

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available MongoDB tools (READ-ONLY FILTERED)"""
        if not self.connected or not self.session:
            logger.warning("MCP client not connected")
            return []

        try:
            response = await self.session.list_tools()
            all_tools = [{"name": tool.name, "description": tool.description} for tool in response.tools]

            # SECURITY: Filter out write operations
            read_only_tools = []
            for tool in all_tools:
                tool_name = tool["name"].lower()
                # Block any write operations
                if any(write_op in tool_name for write_op in ['insert', 'update', 'delete', 'create', 'drop', 'write']):
                    logger.info(f"🔒 Filtered out write tool: {tool['name']}")
                    continue
                read_only_tools.append(tool)

            logger.info(f"✅ Found {len(read_only_tools)} read-only tools", tools=[t["name"] for t in read_only_tools])
            return read_only_tools
        except Exception as e:
            logger.error("Failed to list tools", error=str(e))
            return []

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """Call a MongoDB MCP tool (READ-ONLY ONLY)"""
        if not self.connected or not self.session:
            yield {"type": "error", "message": "MCP client not connected"}
            return

        # SECURITY: Whitelist of allowed read-only tools only
        ALLOWED_READ_TOOLS = {
            "mongodb.find",
            "mongodb.aggregate",
            "mongodb.count",
            "mongodb.listDatabases",
            "mongodb.listCollections",
            "mongodb.listIndexes",
            "mongodb.findOne"  # if available
        }

        # BLOCK any write operations
        if tool_name not in ALLOWED_READ_TOOLS:
            if any(write_op in tool_name.lower() for write_op in ['insert', 'update', 'delete', 'create', 'drop', 'write']):
                yield {"type": "error", "message": f"❌ Write operation '{tool_name}' is BLOCKED. MCP client is read-only."}
                logger.warning(f"Blocked write operation attempt: {tool_name}")
                return
            else:
                yield {"type": "error", "message": f"❌ Tool '{tool_name}' not in allowed read-only tools list"}
                logger.warning(f"Blocked unauthorized tool: {tool_name}")
                return

        try:
            logger.info("Calling MCP read-only tool", tool_name=tool_name, arguments=arguments)
            result = await self.session.call_tool(tool_name, arguments)

            # Process the result content
            if hasattr(result, 'content') and result.content:
                for content_item in result.content:
                    if hasattr(content_item, 'text'):
                        yield {"type": "tool.output", "data": content_item.text}
                    else:
                        yield {"type": "tool.output", "data": str(content_item)}
            else:
                yield {"type": "tool.output", "data": str(result)}

        except Exception as e:
            logger.error("Failed to call read-only tool", tool_name=tool_name, error=str(e))
            yield {"type": "error", "message": f"Tool call failed: {str(e)}"}

    async def smart_query_processing(self, query: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Enhanced query processing with caching and optimization"""
        # Check cache first
        cache_key = f"query:{hash(query)}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            yield {"type": "cache.hit", "data": cached_result}
            return

        # Use LLM for intelligent processing if available
        if self.anthropic:
            try:
                result = await self.process_query_with_llm(query)
                self.cache.set(cache_key, result)
                yield {"type": "llm.response", "data": result}
            except Exception as e:
                logger.warning("LLM processing failed, falling back to direct query", error=str(e))
                # Fall through to direct processing

        # Direct tool processing (fallback)
        try:
            # Parse query to extract tool calls
            tool_calls = self._extract_tool_calls(query)
            for tool_call in tool_calls:
                async for result in self.call_tool(tool_call["name"], tool_call["args"]):
                    yield result
        except Exception as e:
            yield {"type": "error", "message": f"Query processing failed: {str(e)}"}

    def _extract_tool_calls(self, query: str) -> List[Dict[str, Any]]:
        """Simple tool call extraction from natural language query (READ-ONLY ONLY)"""
        # This is a simplified implementation
        # In practice, you'd use more sophisticated NLP or LLM for this
        tool_calls = []

        query_lower = query.lower()

        # SECURITY: Only allow read operations
        if any(write_word in query_lower for write_word in ['insert', 'update', 'delete', 'create', 'drop', 'write', 'modify']):
            logger.warning("⚠️  Write operation detected in query, blocking tool extraction")
            return []

        # Example tool mappings (READ-ONLY only)
        if "find" in query_lower and "lead" in query_lower:
            tool_calls.append({
                "name": "mongodb.find",
                "args": {"collection": "Lead", "limit": 10}
            })
        elif "count" in query_lower and "lead" in query_lower:
            tool_calls.append({
                "name": "mongodb.count",
                "args": {"collection": "Lead"}
            })
        elif "list" in query_lower and "collections" in query_lower:
            tool_calls.append({
                "name": "mongodb.listCollections",
                "args": {"database": settings.mongodb_database}
            })

        return tool_calls

    # Convenience methods for chat engine compatibility
    async def find_documents(self, collection: str, filter_query: Optional[Dict[str, Any]] = None, projection: Optional[Dict[str, Any]] = None, limit: int = 50, user_id: Optional[str] = None):
        """Find documents in collection (read-only)"""
        args = {
            "collection": collection,
            "filter": filter_query or {},
            "projection": projection or {},
            "limit": min(limit, 200)  # Respect max rows limit
        }
        async for result in self.call_tool("mongodb.find", args):
            yield result

    async def aggregate_documents(self, collection: str, pipeline: List[Dict[str, Any]], limit: int = 100, user_id: Optional[str] = None):
        """Aggregate documents in collection (read-only)"""
        args = {
            "collection": collection,
            "pipeline": pipeline,
            "limit": min(limit, 200)  # Respect max rows limit
        }
        async for result in self.call_tool("mongodb.aggregate", args):
            yield result

    # BLOCKED WRITE OPERATIONS (for compatibility)
    async def create_note(self, *args, **kwargs):
        """Create note - BLOCKED for security"""
        raise PermissionError("❌ Write operations are blocked. MCP client is read-only.")

    async def create_task(self, *args, **kwargs):
        """Create task - BLOCKED for security"""
        raise PermissionError("❌ Write operations are blocked. MCP client is read-only.")

    async def create_work_item(self, *args, **kwargs):
        """Create work item - BLOCKED for security"""
        raise PermissionError("❌ Write operations are blocked. MCP client is read-only.")

    async def create_page(self, *args, **kwargs):
        """Create page - BLOCKED for security"""
        raise PermissionError("❌ Write operations are blocked. MCP client is read-only.")

    # CRM-specific read methods
    async def find_crm_leads(self, limit: int = 100, user_id: Optional[str] = None):
        """Find CRM leads (read-only)"""
        filter_query = {}
        if user_id:
            filter_query["business_id"] = user_id

        async for result in self.find_documents("Lead", filter_query, None, limit, user_id):
            yield result

    async def aggregate_crm_stats(self, user_id: Optional[str] = None):
        """Aggregate CRM statistics (read-only)"""
        match_stage = {}
        if user_id:
            match_stage = {"$match": {"business_id": user_id}}

        pipeline = [
            match_stage,
            {"$group": {
                "_id": None,
                "total_leads": {"$sum": 1},
                "active_leads": {"$sum": {"$cond": [{"$eq": ["$status", "ACTIVE"]}, 1, 0]}},
                "won_leads": {"$sum": {"$cond": [{"$eq": ["$status", "WON"]}, 1, 0]}}
            }}
        ]

        async for result in self.aggregate_documents("Lead", pipeline, 1, user_id):
            yield result

    async def find_crm_accounts(self, limit: int = 50, user_id: Optional[str] = None):
        """Find CRM accounts (read-only)"""
        filter_query = {}
        if user_id:
            filter_query["business_id"] = user_id

        async for result in self.find_documents("leadStatus", filter_query, None, limit, user_id):
            yield result

    async def cleanup(self):
        """Clean up resources following official MCP pattern"""
        await self.exit_stack.aclose()
        self.connected = False
        logger.info("Official MCP client cleaned up")

    async def health_check(self) -> Dict[str, Any]:
        """Health check for MongoDB MCP client"""
        if not self.connected:
            return {"status": "disconnected"}

        try:
            tools = await self.list_tools()
            return {
                "status": "connected",
                "tools_count": len(tools),
                "tools": [t["name"] for t in tools],
                "cache_size": len(self.cache.cache) if hasattr(self.cache, 'cache') else 0,
                "transport": "stdio"
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

class MongoDBMCPClient:
    """Optimized MCP client with smart analysis and parallel processing for Docker environments"""

    def __init__(self):
        self.mcp: Optional[MongoMCP] = None
        self.connected = False
        self.last_connection_attempt = None
        self.connection_failures = 0
        self.docker_restart_detected = False

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
        """Setup MCP client with Docker-aware configuration"""
        try:
            # Use resolved URL that handles Docker networking
            base_url = settings.mongodb_mcp_server_url_resolved.rstrip('/')
            self.mcp = MongoMCP(base_url, settings.mongodb_max_rows_per_query)

            # Log Docker-specific configuration
            docker_info = {
                "docker_host": settings.mongodb_mcp_docker_host,
                "docker_port": settings.mongodb_mcp_docker_port,
                "docker_network": settings.mongodb_mcp_docker_network,
                "resolved_url": base_url
            }
            logger.info("MongoDB MCP client configured for Docker", **docker_info)
        except Exception as e:
            logger.error("Failed to setup MCP client", error=str(e))

    def _detect_docker_restart(self) -> bool:
        """Detect if Docker container has restarted based on connection patterns"""
        now = datetime.now(timezone.utc)

        # If we had multiple recent failures, it might indicate a container restart
        if self.connection_failures >= 3:
            # Reset failure counter and mark restart detected
            self.connection_failures = 0
            self.docker_restart_detected = True
            logger.info("🐳 Docker container restart detected - resetting connection state")
            return True

        # Check if the last connection attempt was recent (within 30 seconds)
        if self.last_connection_attempt:
            time_since_last_attempt = (now - self.last_connection_attempt).total_seconds()
            if time_since_last_attempt < 30 and self.connection_failures > 0:
                logger.info("🐳 Rapid connection failures detected - likely Docker container issue")
                return True

        return False

    def _get_docker_optimized_timeout(self) -> float:
        """Get timeout optimized for Docker networking"""
        # Use shorter timeout for Docker to fail fast and retry quickly
        if self.docker_restart_detected:
            return 5.0  # Shorter timeout when restart is detected
        return settings.mongodb_query_timeout_seconds

    async def connect(self, max_retries: int = 5, retry_delay: float = 2.0) -> bool:
        """Connect to MCP server with Docker-aware retry logic"""
        if not self.mcp:
            logger.warning("MCP client not configured")
            return False

        if self.connected:
            logger.info("MCP client already connected")
            return True

        # Update connection attempt timestamp
        self.last_connection_attempt = datetime.now(timezone.utc)

        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting to connect to MCP server (attempt {attempt + 1}/{max_retries})")

                # Test connection by listing tools with Docker-optimized timeout
                timeout = self._get_docker_optimized_timeout()
                try:
                    tools = await asyncio.wait_for(self.mcp.list_tools(), timeout=timeout)
                    logger.info(f"✅ MCP client connected successfully to Docker MCP server, found {len(tools)} tools", tools=tools)
                    self.connected = True
                    self.connection_failures = 0  # Reset failure counter on success
                    self.docker_restart_detected = False  # Reset restart flag
                    return True
                except asyncio.TimeoutError:
                    logger.warning(f"🐳 Docker MCP server connection timed out after {timeout}s")
                    raise aiohttp.ClientError("Connection timeout")

            except aiohttp.ClientError as e:
                self.connection_failures += 1  # Track connection failures
                if "Connection refused" in str(e) or "Connection reset by peer" in str(e):
                    if attempt < max_retries - 1:
                        # Check for Docker restart pattern
                        if self._detect_docker_restart():
                            logger.warning(f"🐳 Docker restart detected, using longer retry delay for container startup...")
                            retry_delay = 8.0  # Longer delay for container startup
                        else:
                            # Use shorter delays for regular connection issues
                            retry_delay = min(retry_delay * 1.5, 10.0)  # Cap at 10 seconds

                        logger.warning(f"🐳 Docker MCP server connection failed (attempt {attempt + 1}), retrying in {retry_delay}s...",
                                     error=str(e), docker_network=settings.mongodb_mcp_docker_network)
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        logger.error(f"Failed to connect to MCP server after {max_retries} attempts", error=str(e))
                else:
                    logger.error("Failed to connect MCP client", error=str(e))
                    break
            except Exception as e:
                self.connection_failures += 1  # Track connection failures
                logger.error("Unexpected error connecting to MCP client", error=str(e))
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 1.5
                    continue
                break

        self.connected = False
        return False

    async def disconnect(self):
        """Disconnect from MCP server and cleanup resources"""
        self.connected = False

        # Cleanup optimized components
        if self.mcp:
            await self.mcp.close()

        logger.info("Disconnected from MongoDB MCP server and cleaned up resources")

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

        except aiohttp.ClientError as e:
            # Docker-specific connection errors
            if "Connection refused" in str(e) or "Connection reset by peer" in str(e):
                logger.warning("Docker MCP server connection error during tool call", error=str(e), tool_name=tool_name)
                # Mark as disconnected to trigger reconnection on next call
                self.connected = False
                yield {
                    "type": "error",
                    "message": "Docker MCP server connection failed. Will retry on next request.",
                    "tool": tool_name,
                    "docker_error": True,
                    "retry_recommended": True
                }
            else:
                logger.error("HTTP error calling MongoDB MCP tool", error=str(e), tool_name=tool_name)
                yield {
                    "type": "error",
                    "message": f"HTTP error calling tool: {str(e)}",
                    "tool": tool_name
                }
        except asyncio.TimeoutError:
            logger.warning("Timeout calling MongoDB MCP tool - possible Docker container issue", tool_name=tool_name)
            yield {
                "type": "error",
                "message": "Request timeout - Docker container may be overloaded",
                "tool": tool_name,
                "timeout": True,
                "docker_error": True
            }
        except Exception as e:
            logger.error("Unexpected error calling MongoDB MCP tool", error=str(e), tool_name=tool_name)
            yield {
                "type": "error",
                "message": f"Unexpected error calling tool: {str(e)}",
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


    # Optimized Analysis Methods
    async def smart_analyze_collection(self, collection: str, analysis_type: str = "auto", user_id: Optional[str] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Smart collection analysis with optimization"""
        if not await self.ensure_connected():
            yield {"type": "error", "message": "Failed to connect to MCP server"}
            return

        try:
            async for result in self.mcp.analyze_collection_smart(collection, analysis_type):
                yield result
        except Exception as e:
            logger.error("Smart analysis failed", error=str(e), collection=collection)
            yield {"type": "error", "message": f"Analysis failed: {str(e)}"}

    async def progressive_analyze_collection(self, collection: str, user_id: Optional[str] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Progressive analysis with streaming results"""
        if not await self.ensure_connected():
            yield {"type": "error", "message": "Failed to connect to MCP server"}
            return

        try:
            async for result in self.mcp.progressive_analysis(collection):
                yield result
        except Exception as e:
            logger.error("Progressive analysis failed", error=str(e), collection=collection)
            yield {"type": "error", "message": f"Progressive analysis failed: {str(e)}"}

    async def get_collection_metadata_smart(self, collection: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get smart collection metadata with caching"""
        if not await self.ensure_connected():
            return {"error": "Failed to connect to MCP server"}

        try:
            return await self.mcp.get_collection_metadata(collection)
        except Exception as e:
            logger.error("Metadata retrieval failed", error=str(e), collection=collection)
            return {"error": f"Metadata retrieval failed: {str(e)}"}

    async def execute_parallel_operations(self, operations: List[Dict[str, Any]], user_id: Optional[str] = None) -> List[Any]:
        """Execute multiple operations in parallel"""
        if not await self.ensure_connected():
            return [{"error": "Failed to connect to MCP server"}]

        try:
            # Convert operations to MCP format
            queries = []
            for op in operations:
                if op["type"] in ["find", "aggregate"]:
                    # Add tenant filtering if needed
                    if user_id and settings.mongodb_tenant_field:
                        if "filter" in op.get("params", {}):
                            op["params"]["filter"][settings.mongodb_tenant_field] = user_id
                        else:
                            op["params"]["filter"] = {settings.mongodb_tenant_field: user_id}

                    queries.append(op)

            return await self.mcp.execute_parallel_queries(queries)
        except Exception as e:
            logger.error("Parallel operations failed", error=str(e))
            return [{"error": f"Parallel operations failed: {str(e)}"}]

    async def batch_analyze_collections(self, collections: List[str], analysis_type: str = "auto", user_id: Optional[str] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Analyze multiple collections with optimized batching"""
        if not await self.ensure_connected():
            yield {"type": "error", "message": "Failed to connect to MCP server"}
            return

        # Create parallel analysis tasks
        operations = []
        for collection in collections:
            operations.append({
                "type": "aggregate",
                "collection": collection,
                "pipeline": self.mcp.query_optimizer.create_combined_analysis_pipeline({"type": analysis_type}),
                "params": {"limit": 100}
            })

        # Execute in parallel
        results = await self.execute_parallel_operations(operations, user_id)

        # Yield results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                yield {
                    "type": "analysis.error",
                    "collection": collections[i],
                    "error": str(result)
                }
            else:
                yield {
                    "type": "analysis.result",
                    "collection": collections[i],
                    "data": result
                }

    async def get_template_suggestions(self, collection_name: str) -> Dict[str, Any]:
        """Get analysis template suggestions for a collection"""
        template = self.template_registry.get_template(collection_name)
        return {
            "collection": collection_name,
            "suggested_template": template["analysis_type"],
            "key_fields": template["key_fields"],
            "recommended_metrics": template["metrics"],
            "temporal_field": template["temporal_field"]
        }

    async def health_check(self) -> Dict[str, Any]:
        """Docker-aware health check for MongoDB MCP server"""
        if not self.mcp:
            return {
                "status": "not_configured",
                "message": "MCP client not initialized"
            }

        if not self.connected:
            # Try to reconnect for Docker environments where containers might restart
            logger.info("MCP client not connected, attempting Docker-aware reconnection...")
            connected = await self.connect(max_retries=2, retry_delay=1.0)
            if not connected:
                            return {
                "status": "disconnected",
                "message": "Failed to reconnect to Docker MCP server",
                "available_tools": 0,
                "docker_environment": True,
                "docker_troubleshooting": {
                    "check_container_running": f"docker ps | grep mongodb-mcp",
                    "check_container_logs": f"docker logs <container_name>",
                    "verify_network_connectivity": f"curl -v {settings.mongodb_mcp_server_url_resolved}/health",
                    "docker_network_mode": settings.mongodb_mcp_docker_network,
                    "resolved_url": settings.mongodb_mcp_server_url_resolved
                },
                "optimizations": {
                    "connection_pooling": False,
                    "caching": False,
                    "parallel_processing": False,
                    "smart_templates": True
                }
            }

        try:
            # Quick connectivity test for Docker
            tools = await asyncio.wait_for(self.list_tools(), timeout=10.0)

            # Check optimization components
            optimizations = {
                "connection_pooling": self.mcp and hasattr(self.mcp, 'connection_pool'),
                "caching": self.mcp and hasattr(self.mcp, 'cache'),
                "parallel_processing": self.mcp and hasattr(self.mcp, 'parallel_executor'),
                "smart_templates": True,
                "query_optimization": self.mcp and hasattr(self.mcp, 'query_optimizer'),
                "docker_optimized": True
            }

            # Get detailed cache stats if available
            cache_stats = None
            if self.mcp and hasattr(self.mcp, 'cache'):
                cache_stats = self.mcp.cache.get_stats()

            # Docker-specific health metrics
            docker_health = {
                "docker_host": settings.mongodb_mcp_docker_host,
                "docker_port": settings.mongodb_mcp_docker_port,
                "docker_network": settings.mongodb_mcp_docker_network,
                "resolved_url": settings.mongodb_mcp_server_url_resolved,
                "connection_pool_active": self.mcp and self.mcp.connection_pool._session is not None,
                "connection_failures": self.connection_failures,
                "docker_restart_detected": self.docker_restart_detected,
                "last_connection_attempt": self.last_connection_attempt.isoformat() if self.last_connection_attempt else None,
                "last_health_check": datetime.now(timezone.utc).isoformat()
            }

            return {
                "status": "connected",
                "message": "Docker MCP server healthy",
                "available_tools": len(tools),
                "tools": [tool.get("name") for tool in tools],
                "optimizations": optimizations,
                "cache_stats": cache_stats,
                "docker_environment": True,
                "docker_health": docker_health,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except asyncio.TimeoutError:
            logger.warning("🐳 Docker MCP server health check timeout - container may be overloaded or restarting")
            return {
                "status": "timeout",
                "message": "Docker container response timeout - check container status",
                "docker_environment": True,
                "docker_troubleshooting": {
                    "check_container_status": "docker ps | grep mongodb-mcp",
                    "check_container_resources": "docker stats <container_name>",
                    "increase_container_resources": "docker update --memory=1g --cpus=1 <container_name>",
                    "resolved_url": settings.mongodb_mcp_server_url_resolved
                }
            }
        except aiohttp.ClientError as e:
            logger.warning("🐳 Docker MCP server connection error during health check", error=str(e))
            # Mark as disconnected so next call will try to reconnect
            self.connected = False
            return {
                "status": "connection_error",
                "message": f"Docker container connection error: {str(e)}",
                "docker_environment": True,
                "docker_troubleshooting": {
                    "check_docker_network": f"docker network ls",
                    "verify_container_network": f"docker inspect <container_name> | grep -A 10 NetworkSettings",
                    "test_connectivity": f"docker exec <container_name> curl -v {settings.mongodb_mcp_server_url_resolved}",
                    "check_container_ports": f"docker port <container_name>",
                    "resolved_url": settings.mongodb_mcp_server_url_resolved
                }
            }
        except Exception as e:
            logger.error("Unexpected error during Docker health check", error=str(e))
            return {
                "status": "error",
                "message": f"Docker health check failed: {str(e)}",
                "docker_environment": True
            }

# Global client instances - using stdio-based client as per MongoDB MCP server design
mongodb_mcp_client = OfficialMCPClient()  # Stdio-based client for MongoDB MCP server

# Auto-connect to MongoDB MCP server on import if connection string is available
async def initialize_mongodb_mcp_client():
    """Initialize and connect the MongoDB MCP client (READ-ONLY ONLY)"""
    if settings.mongodb_connection_string and settings.mongodb_mcp_enabled:
        try:
            logger.info("🔒 Auto-connecting to MongoDB MCP server in READ-ONLY mode...")
            # SECURITY: Always use enforced read-only mode
            await mongodb_mcp_client.connect_to_mongodb_server(read_only=settings.mongodb_readonly_enforced)
            logger.info("✅ MongoDB MCP client connected successfully (READ-ONLY mode)")
        except Exception as e:
            logger.error("❌ Failed to auto-connect MongoDB MCP client", error=str(e))
            logger.info("Note: Make sure Node.js and npm are installed for MCP server")
    else:
        logger.info("MongoDB MCP client not configured (missing connection string or disabled)")

async def create_official_mcp_client():
    """Factory function to create official MCP client with proper error handling"""
    if not MCP_AVAILABLE:
        logger.warning("Official MCP package not available. Use HTTP client instead.")
        return None

    client = OfficialMCPClient()
    return client

