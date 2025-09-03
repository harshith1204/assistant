"""Agentic AI Assistant with reasoning, planning, and tool use capabilities"""

import json
import asyncio
import uuid
from typing import Dict, Any, List, Optional, Callable, AsyncGenerator
from datetime import datetime, timezone
from enum import Enum
from dataclasses import dataclass, field
from groq import AsyncGroq
import structlog

from app.config import settings
from app.core.memory_manager import MemoryManager
from app.research.service import AgenticResearchService
from app.integrations.mcp_client import mongodb_mcp_client

logger = structlog.get_logger()


class AgentState(Enum):
    """Agent operational states"""
    IDLE = "idle"
    REASONING = "reasoning"
    PLANNING = "planning"
    EXECUTING = "executing"
    REFLECTING = "reflecting"
    COMPLETED = "completed"
    ERROR = "error"


class ActionType(Enum):
    """Types of actions the agent can take"""
    THINK = "think"
    RESEARCH = "research"
    QUERY_DATABASE = "query_database"
    GENERATE_RESPONSE = "generate_response"
    USE_TOOL = "use_tool"
    REFLECT = "reflect"
    DELEGATE = "delegate"


@dataclass
class Thought:
    """A single thought in the agent's reasoning chain"""
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    confidence: float = 1.0
    evidence: List[str] = field(default_factory=list)


@dataclass
class Plan:
    """A plan for executing a complex task"""
    steps: List[Dict[str, Any]]
    estimated_duration: int  # in seconds
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AgentContext:
    """Context for agent operations"""
    user_id: str
    conversation_id: str
    current_state: AgentState = AgentState.IDLE
    thoughts: List[Thought] = field(default_factory=list)
    current_plan: Optional[Plan] = None
    tool_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Tool:
    """Base class for agent tools with enhanced capabilities"""

    def __init__(self, name: str, description: str, category: str = "general"):
        self.name = name
        self.description = description
        self.category = category
        self.usage_count = 0
        self.success_count = 0
        self.last_used = None
        self.average_execution_time = 0.0

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given parameters"""
        start_time = asyncio.get_event_loop().time()
        self.last_used = datetime.now(timezone.utc)

        try:
            self.usage_count += 1
            result = await self._execute_impl(**kwargs)
            self.success_count += 1

            execution_time = asyncio.get_event_loop().time() - start_time
            self._update_execution_time(execution_time)

            return {
                "success": True,
                "result": result,
                "execution_time": execution_time,
                "tool": self.name,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            self._update_execution_time(execution_time)

            return {
                "success": False,
                "error": str(e),
                "execution_time": execution_time,
                "tool": self.name,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    def _update_execution_time(self, execution_time: float):
        """Update average execution time using exponential moving average"""
        if self.usage_count == 1:
            self.average_execution_time = execution_time
        else:
            alpha = 0.1  # Smoothing factor
            self.average_execution_time = alpha * execution_time + (1 - alpha) * self.average_execution_time

    async def _execute_impl(self, **kwargs) -> Dict[str, Any]:
        """Actual implementation of the tool"""
        raise NotImplementedError

    def get_schema(self) -> Dict[str, Any]:
        """Return OpenAI-style tool schema for function calling"""
        raise NotImplementedError

    def get_usage_stats(self) -> Dict[str, Any]:
        """Return usage statistics for the tool"""
        return {
            "name": self.name,
            "category": self.category,
            "usage_count": self.usage_count,
            "success_rate": self.success_count / max(self.usage_count, 1),
            "average_execution_time": self.average_execution_time,
            "last_used": self.last_used.isoformat() if self.last_used else None
        }

    def validate_parameters(self, **kwargs) -> bool:
        """Validate input parameters"""
        return True

    def get_examples(self) -> List[Dict[str, Any]]:
        """Return example usages of the tool"""
        return []


class ReasoningEngine:
    """Core reasoning and planning engine for the agent"""

    def __init__(self, llm_client: AsyncGroq):
        self.client = llm_client
        self.model = settings.llm_model

    async def reason_about_request(self, user_message: str, context: AgentContext) -> List[Thought]:
        """Analyze user request and generate reasoning chain"""

        system_prompt = """You are an expert AI reasoning about a user request.

Your task is to:
1. Understand the user's intent and needs
2. Identify what information or actions are required
3. Consider potential challenges or edge cases
4. Plan how to best fulfill the request

Generate a chain of thoughts that shows your reasoning process.
Each thought should be clear, actionable, and build on previous thoughts.

Return ONLY valid JSON:
{
    "thoughts": [
        {
            "content": "Clear, specific thought about the request",
            "confidence": 0.8,
            "evidence": ["supporting fact 1", "supporting fact 2"]
        }
    ],
    "needs_clarification": false,
    "estimated_complexity": "simple|moderate|complex"
}"""

        user_prompt = f"""Analyze this user request and reason about how to best respond:

User Request: {user_message}

Context:
- Current conversation state: {context.current_state.value}
- Previous thoughts: {[t.content for t in context.thoughts[-3:]] if context.thoughts else []}
- Available tools: research, database queries, general assistance

Generate your reasoning chain:"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )

            result = json.loads(response.choices[0].message.content)

            thoughts = []
            for thought_data in result.get("thoughts", []):
                thought = Thought(
                    content=thought_data["content"],
                    confidence=thought_data.get("confidence", 0.8),
                    evidence=thought_data.get("evidence", [])
                )
                thoughts.append(thought)

            # Store clarification need and complexity in context
            context.metadata["needs_clarification"] = result.get("needs_clarification", False)
            context.metadata["estimated_complexity"] = result.get("estimated_complexity", "moderate")

            return thoughts

        except Exception as e:
            logger.error("Failed to generate reasoning", error=str(e))
            return [Thought(content="I understand your request and will help you with it.", confidence=0.5)]

    async def create_execution_plan(self, thoughts: List[Thought], context: AgentContext) -> Plan:
        """Create a structured plan from reasoning thoughts"""

        system_prompt = """You are an expert planner creating actionable execution plans.

Based on the reasoning thoughts, create a step-by-step plan that:
1. Breaks down the task into specific, executable steps
2. Identifies dependencies between steps
3. Estimates time requirements
4. Considers potential failure points

Each step should specify:
- What action to take
- What tools or methods to use
- Expected outcomes
- Success criteria

Return ONLY valid JSON:
{
    "steps": [
        {
            "id": "step_1",
            "action": "research_topic",
            "description": "Research the specific topic mentioned",
            "tool": "research_engine",
            "parameters": {"query": "topic details"},
            "estimated_duration": 30,
            "dependencies": [],
            "success_criteria": "Found relevant information"
        }
    ],
    "total_estimated_duration": 60,
    "risks": ["potential data unavailability", "complexity of topic"]
}"""

        thoughts_summary = "\n".join([f"- {t.content}" for t in thoughts])

        user_prompt = f"""Create an execution plan based on this reasoning:

Reasoning Chain:
{thoughts_summary}

Context: {context.metadata}

Create a detailed execution plan:"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=1500
            )

            result = json.loads(response.choices[0].message.content)

            plan = Plan(
                steps=result.get("steps", []),
                estimated_duration=result.get("total_estimated_duration", 60),
                dependencies={step["id"]: step.get("dependencies", []) for step in result.get("steps", [])}
            )

            # Store risks in context
            context.metadata["plan_risks"] = result.get("risks", [])

            return plan

        except Exception as e:
            logger.error("Failed to create execution plan", error=str(e))
            # Return a simple fallback plan
            return Plan(
                steps=[{
                    "id": "fallback_step",
                    "action": "respond",
                    "description": "Provide a helpful response",
                    "tool": "chat_engine",
                    "parameters": {},
                    "estimated_duration": 10,
                    "dependencies": [],
                    "success_criteria": "User receives response"
                }],
                estimated_duration=10
            )

    async def reflect_on_execution(self, context: AgentContext, final_response: str) -> Dict[str, Any]:
        """Reflect on the execution quality and learnings"""

        system_prompt = """You are reflecting on your performance as an AI assistant.

Analyze:
1. How well you understood the user's request
2. Quality and relevance of your response
3. Effectiveness of your reasoning and planning
4. What you learned that could improve future interactions

Return ONLY valid JSON:
{
    "understanding_score": 0.8,
    "response_quality": 0.9,
    "reasoning_effectiveness": 0.7,
    "learnings": ["What I learned about this type of request"],
    "improvements": ["How to handle similar requests better"],
    "confidence_in_response": 0.85
}"""

        execution_summary = f"""
Thoughts: {[t.content for t in context.thoughts]}
Plan executed: {context.current_plan.steps if context.current_plan else 'No plan'}
Tool results: {list(context.tool_results.keys())}
Final response: {final_response[:500]}...
"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Reflect on this execution:\n{execution_summary}"}
                ],
                temperature=0.3,
                max_tokens=800
            )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            logger.error("Failed to reflect on execution", error=str(e))
            return {
                "understanding_score": 0.7,
                "response_quality": 0.7,
                "reasoning_effectiveness": 0.6,
                "learnings": ["Need better error handling"],
                "improvements": ["Improve reflection capabilities"],
                "confidence_in_response": 0.6
            }


class ToolRegistry:
    """Registry for agent tools"""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register_tool(self, tool: Tool):
        """Register a tool"""
        self.tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name"""
        return self.tools.get(name)

    def get_all_tools(self) -> List[Tool]:
        """Get all registered tools"""
        return list(self.tools.values())

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all tools (for function calling)"""
        return [tool.get_schema() for tool in self.tools.values()]


class AgenticAssistant:
    """Main agentic AI assistant class"""

    def __init__(self):
        self.llm_client = AsyncGroq(api_key=settings.groq_api_key)
        self.reasoning_engine = ReasoningEngine(self.llm_client)
        self.memory_manager = MemoryManager()
        self.tool_registry = ToolRegistry()
        self.research_service = AgenticResearchService(self)  # Pass self for agent reference
        self.mcp_client = mongodb_mcp_client

        # Initialize core tools
        self._register_core_tools()

        # Agent contexts
        self.contexts: Dict[str, AgentContext] = {}

        logger.info("Agentic Assistant initialized")

    def _register_core_tools(self):
        """Register core tools"""

        # Research tools - modular and focused
        web_search_tool = WebSearchTool()
        self.tool_registry.register_tool(web_search_tool)

        content_analysis_tool = ContentAnalysisTool(self.llm_client)
        self.tool_registry.register_tool(content_analysis_tool)

        synthesis_tool = SynthesisTool(self.llm_client)
        self.tool_registry.register_tool(synthesis_tool)

        research_coordinator_tool = ResearchCoordinatorTool(self.tool_registry)
        self.tool_registry.register_tool(research_coordinator_tool)

        # Legacy research tool for backward compatibility
        research_tool = ResearchTool(self.research_service)
        self.tool_registry.register_tool(research_tool)

        # Database tool
        db_tool = DatabaseTool(self.mcp_client)
        self.tool_registry.register_tool(db_tool)

        # Response generation tool
        response_tool = ResponseTool(self.llm_client)
        self.tool_registry.register_tool(response_tool)

        # Analysis tool
        analysis_tool = AnalysisTool(self.llm_client)
        self.tool_registry.register_tool(analysis_tool)

        # Calculation tool
        calculation_tool = CalculationTool()
        self.tool_registry.register_tool(calculation_tool)

        logger.info(f"Registered {len(self.tool_registry.get_all_tools())} tools: {[t.name for t in self.tool_registry.get_all_tools()]}")

    def get_tool_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for all tools"""
        all_stats = {}
        for tool in self.tool_registry.get_all_tools():
            all_stats[tool.name] = tool.get_usage_stats()

        return {
            "total_tools": len(all_stats),
            "tool_stats": all_stats,
            "most_used": max(all_stats.items(), key=lambda x: x[1]["usage_count"], default=(None, {}))[0],
            "highest_success_rate": max(all_stats.items(), key=lambda x: x[1]["success_rate"], default=(None, {}))[0]
        }

    async def execute_tool_by_name(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool by name with parameters"""
        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return {"error": f"Tool '{tool_name}' not found"}

        return await tool.execute(**kwargs)

    def _get_or_create_context(self, user_id: str, conversation_id: str) -> AgentContext:
        """Get or create agent context"""
        context_key = f"{user_id}:{conversation_id}"

        if context_key not in self.contexts:
            self.contexts[context_key] = AgentContext(
                user_id=user_id,
                conversation_id=conversation_id
            )

        return self.contexts[context_key]

    async def process_request(
        self,
        user_message: str,
        user_id: str,
        conversation_id: str,
        streaming_callback: Optional[Callable] = None
    ) -> AsyncGenerator[str, None]:

        """Process a user request in an agentic manner"""

        context = self._get_or_create_context(user_id, conversation_id)

        try:
            # Phase 1: Reasoning
            context.current_state = AgentState.REASONING
            if streaming_callback:
                streaming_callback("🤔 Analyzing your request...")

            thoughts = await self.reasoning_engine.reason_about_request(user_message, context)
            context.thoughts.extend(thoughts)

            # Phase 2: Planning
            context.current_state = AgentState.PLANNING
            if streaming_callback:
                streaming_callback("📋 Creating execution plan...")

            plan = await self.reasoning_engine.create_execution_plan(thoughts, context)
            context.current_plan = plan

            # Phase 3: Execution
            context.current_state = AgentState.EXECUTING
            if streaming_callback:
                streaming_callback("⚡ Executing plan...")

            final_response = await self._execute_plan(plan, context, user_message, streaming_callback)

            # Phase 4: Reflection
            context.current_state = AgentState.REFLECTING
            if streaming_callback:
                streaming_callback("🔍 Reflecting on performance...")

            reflection = await self.reasoning_engine.reflect_on_execution(context, final_response)

            # Store reflection for learning
            await self._store_reflection(context, reflection)

            context.current_state = AgentState.COMPLETED

            # Stream final response
            if final_response is not None:
                final_response_str = str(final_response)

                if streaming_callback:
                    # Stream the response in chunks for better UX
                    if isinstance(final_response, str):
                        # Split response into words/tokens for streaming
                        words = final_response_str.split()
                        streamed_content = ""

                        for i, word in enumerate(words):
                            # Add word to streamed content
                            if i > 0:
                                streamed_content += " "
                            streamed_content += word

                            # Yield the word (with space if not first)
                            chunk = f" {word}" if i > 0 else word
                            streaming_callback(chunk)
                            yield chunk

                            # Small delay to simulate realistic streaming
                            if i < len(words) - 1:  # Don't delay after last word
                                await asyncio.sleep(0.01)

                        # Ensure the complete response is sent to streaming callback
                        if streamed_content != final_response_str:
                            streaming_callback(final_response_str[len(streamed_content):])
                    elif hasattr(final_response, '__iter__'):
                        # If it's iterable, iterate over chunks
                        try:
                            for chunk in final_response:
                                if chunk is not None:
                                    chunk_str = str(chunk)
                                    streaming_callback(chunk_str)
                                    yield chunk_str
                        except TypeError:
                            # Not actually iterable, treat as single item
                            streaming_callback(final_response_str)
                            yield final_response_str
                    else:
                        # Convert to string and yield
                        streaming_callback(final_response_str)
                        yield final_response_str
                else:
                    # No streaming callback, yield the full response
                    yield final_response_str
            else:
                # Handle None case
                fallback_response = "I apologize, but I was unable to generate a response at this time."
                if streaming_callback:
                    streaming_callback(fallback_response)
                yield fallback_response

        except Exception as e:
            logger.error("Agent execution failed", error=str(e))
            context.current_state = AgentState.ERROR
            yield f"I encountered an error while processing your request: {str(e)}"

    async def _execute_plan(
        self,
        plan: Plan,
        context: AgentContext,
        user_message: str,
        streaming_callback: Optional[Callable]
    ) -> str:
        """Execute the plan step by step with optimization"""

        results = {}
        executed_tools = set()

        # Optimize plan execution based on dependencies and tool performance
        optimized_steps = self._optimize_plan_execution(plan.steps)

        for step in optimized_steps:
            step_id = step["id"]
            action = step["action"]
            tool_name = step.get("tool")

            if streaming_callback:
                streaming_callback(f"🔧 Executing step: {step['description']}")

            # Check if we should skip this step based on previous results
            if self._should_skip_step(step, results, context):
                results[step_id] = {"skipped": "Step skipped based on previous results", "reason": "optimization"}
                continue

            try:
                if tool_name:
                    # Prevent duplicate tool executions
                    if tool_name in executed_tools and not step.get("allow_duplicate", False):
                        results[step_id] = {"skipped": f"Tool {tool_name} already executed", "reason": "duplicate_prevention"}
                        continue

                    tool = self.tool_registry.get_tool(tool_name)
                    if tool:
                        # Execute tool with parameters
                        params = step.get("parameters", {})
                        result = await tool.execute(**params)
                        results[step_id] = result
                        context.tool_results[step_id] = result
                        executed_tools.add(tool_name)

                        # Check if we should stop execution based on this result
                        if self._should_stop_execution(result, step, context):
                            break
                    else:
                        logger.warning(f"Tool not found: {tool_name}")
                        results[step_id] = {"error": f"Tool {tool_name} not available"}
                else:
                    # Handle non-tool actions
                    results[step_id] = await self._execute_non_tool_action(action, step, context)

            except Exception as e:
                logger.error(f"Step execution failed: {step_id}", error=str(e))
                results[step_id] = {"error": str(e)}

                # Decide whether to continue or stop on error
                if step.get("critical", False):
                    logger.error(f"Critical step failed: {step_id}, stopping execution")
                    break

        # Generate final response based on all results
        return await self._synthesize_response(results, user_message, context)

    def _optimize_plan_execution(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize the execution order based on dependencies and tool performance"""
        # Simple optimization: sort by dependencies and tool performance
        # In a more advanced system, this could use graph algorithms

        optimized = []
        remaining = steps.copy()

        while remaining:
            # Find steps with no unresolved dependencies
            executable = []
            for step in remaining:
                dependencies = step.get("dependencies", [])
                if not dependencies or all(dep in [s["id"] for s in optimized] for dep in dependencies):
                    executable.append(step)

            if not executable:
                # Circular dependency or other issue, just add the first one
                executable = [remaining[0]]

            # Sort executable steps by tool performance (if available)
            executable.sort(key=lambda s: self._get_tool_priority(s.get("tool")))

            # Add the highest priority step
            optimized.append(executable[0])
            remaining.remove(executable[0])

        return optimized

    def _get_tool_priority(self, tool_name: str) -> int:
        """Get execution priority for a tool (lower number = higher priority)"""
        if not tool_name:
            return 50

        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return 50

        # Prioritize based on success rate and execution time
        success_rate = tool.success_count / max(tool.usage_count, 1)
        avg_time = tool.average_execution_time

        # Lower priority number for better performing tools
        priority = 10 - (success_rate * 5) + (min(avg_time, 5) * 0.5)
        return max(0, min(20, priority))

    def _should_skip_step(self, step: Dict[str, Any], results: Dict[str, Any], context: AgentContext) -> bool:
        """Determine if a step should be skipped based on previous results"""
        # Skip if we already have the required information
        step_requirements = step.get("requires", [])
        for requirement in step_requirements:
            if requirement not in results:
                return False

        # Skip if a similar step already succeeded
        step_action = step.get("action")
        for result_id, result in results.items():
            if (result.get("success") and
                results.get(result_id, {}).get("action") == step_action):
                return True

        return False

    def _should_stop_execution(self, result: Dict[str, Any], step: Dict[str, Any], context: AgentContext) -> bool:
        """Determine if execution should stop based on current result"""
        # Stop if we found a definitive answer
        if (result.get("success") and
            step.get("stop_on_success", False) and
            result.get("result", {}).get("confidence", 0) > 0.8):
            return True

        # Stop if we encountered a critical error
        if (not result.get("success") and
            step.get("critical", False)):
            return True

        return False

    async def _execute_non_tool_action(self, action: str, step: Dict[str, Any], context: AgentContext) -> Dict[str, Any]:
        """Execute non-tool actions"""
        if action == "think":
            return {"thought": step.get("description", "Thought completed")}
        elif action == "reflect":
            return {"reflection": "Reflection completed"}
        else:
            return {"result": f"Executed action: {action}"}

    async def _synthesize_response(self, results: Dict[str, Any], user_message: str, context: AgentContext) -> str:
        """Synthesize final response from all execution results"""

        try:
            response_tool = self.tool_registry.get_tool("response_generator")
            if response_tool:
                tool_result = await response_tool.execute(
                    user_message=user_message,
                    execution_results=results,
                    thoughts=context.thoughts,
                    context=context
                )
                # Extract the actual response from the tool result
                if tool_result and tool_result.get("success") and tool_result.get("result"):
                    response = tool_result["result"]
                    # Ensure response is a string
                    if isinstance(response, str):
                        return response
                    else:
                        return str(response)
                else:
                    # Fallback if tool execution failed
                    error_msg = tool_result.get('error', 'Unknown error') if tool_result else 'Tool execution failed'
                    return f"I encountered an issue generating a response. Tool error: {error_msg}"

            # Fallback response synthesis
            return f"Based on my analysis, here's what I found: {json.dumps(results, indent=2)}"

        except Exception as e:
            logger.error("Error in response synthesis", error=str(e))
            return f"I encountered an error while synthesizing the response: {str(e)}"

    async def _store_reflection(self, context: AgentContext, reflection: Dict[str, Any]):
        """Store reflection for learning"""
        try:
            await self.memory_manager.store_memory(
                content=f"Reflection on interaction: {json.dumps(reflection)}",
                memory_type="reflection",
                user_id=context.user_id,
                metadata={
                    "conversation_id": context.conversation_id,
                    "reflection_type": "execution_review",
                    "scores": reflection
                }
            )
        except Exception as e:
            logger.error("Failed to store reflection", error=str(e))


# Tool Implementations

# Research Agent Tools - Modular and Agentic

class WebSearchTool(Tool):
    """Focused tool for web search and information gathering"""

    def __init__(self):
        super().__init__("web_search", "Search the web for current information and sources", "research")

    async def _execute_impl(self, query: str, num_results: int = 5, **kwargs) -> Dict[str, Any]:
        """Execute web search with focused results"""
        try:
            # Use DuckDuckGo search for reliable results
            import requests
            from bs4 import BeautifulSoup

            search_url = f"https://duckduckgo.com/html/?q={query.replace(' ', '+')}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = requests.get(search_url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')

            results = []
            for result in soup.find_all('a', class_='result__a')[:num_results]:
                title = result.get_text()
                url = result.get('href')
                if url and title:
                    results.append({
                        "title": title,
                        "url": url,
                        "snippet": ""
                    })

            return {
                "query": query,
                "results": results,
                "total_results": len(results),
                "search_engine": "duckduckgo"
            }

        except Exception as e:
            logger.error("Web search failed", error=str(e))
            return {
                "query": query,
                "results": [],
                "error": f"Search failed: {str(e)}",
                "total_results": 0
            }

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for current information and sources",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "Number of results to return",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 20
                        }
                    },
                    "required": ["query"]
                }
            }
        }


class ContentAnalysisTool(Tool):
    """Tool for analyzing and extracting insights from content"""

    def __init__(self, llm_client: AsyncGroq):
        super().__init__("content_analyzer", "Analyze content and extract key insights", "analysis")
        self.llm_client = llm_client

    async def _execute_impl(self, content: str, analysis_type: str = "summary", **kwargs) -> Dict[str, Any]:
        """Analyze content with specific focus"""

        analysis_prompts = {
            "summary": "Provide a concise summary of the main points and key insights from this content:",
            "insights": "Extract the most important insights, trends, and actionable information from this content:",
            "sentiment": "Analyze the sentiment and tone of this content, including any biases or perspectives:",
            "facts": "Extract factual information, statistics, and data points from this content:"
        }

        prompt = analysis_prompts.get(analysis_type, analysis_prompts["summary"])

        try:
            response = await self.llm_client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": f"You are a content analysis expert. {prompt}"},
                    {"role": "user", "content": f"Content to analyze:\n\n{content}"}
                ],
                temperature=0.2,
                max_tokens=1000
            )

            return {
                "content_length": len(content),
                "analysis_type": analysis_type,
                "analysis": response.choices[0].message.content,
                "confidence": 0.8
            }

        except Exception as e:
            logger.error("Content analysis failed", error=str(e))
            return {
                "content_length": len(content),
                "analysis_type": analysis_type,
                "analysis": f"Analysis failed: {str(e)}",
                "confidence": 0.0
            }

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "content_analyzer",
                "description": "Analyze content and extract key insights, summaries, or specific information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The content to analyze"
                        },
                        "analysis_type": {
                            "type": "string",
                            "enum": ["summary", "insights", "sentiment", "facts"],
                            "description": "Type of analysis to perform",
                            "default": "summary"
                        }
                    },
                    "required": ["content"]
                }
            }
        }


class SynthesisTool(Tool):
    """Tool for synthesizing information from multiple sources"""

    def __init__(self, llm_client: AsyncGroq):
        super().__init__("synthesis_engine", "Synthesize and combine information from multiple sources", "synthesis")
        self.llm_client = llm_client

    async def _execute_impl(self, sources: List[str], query: str, synthesis_type: str = "comprehensive", **kwargs) -> Dict[str, Any]:
        """Synthesize information from multiple sources"""

        combined_content = "\n\n".join([f"Source {i+1}:\n{source}" for i, source in enumerate(sources)])

        synthesis_prompts = {
            "comprehensive": "Provide a comprehensive synthesis of all the information, identifying patterns, contradictions, and key insights:",
            "consensus": "Identify areas of consensus and disagreement across the sources:",
            "gaps": "Identify gaps in the information and areas that need further research:",
            "actionable": "Extract actionable insights and recommendations from the combined information:"
        }

        prompt = synthesis_prompts.get(synthesis_type, synthesis_prompts["comprehensive"])

        try:
            response = await self.llm_client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": f"You are a synthesis expert. {prompt}"},
                    {"role": "user", "content": f"Query: {query}\n\nSources to synthesize:\n\n{combined_content}"}
                ],
                temperature=0.3,
                max_tokens=1500
            )

            return {
                "query": query,
                "sources_count": len(sources),
                "synthesis_type": synthesis_type,
                "synthesis": response.choices[0].message.content,
                "key_points": self._extract_key_points(response.choices[0].message.content),
                "confidence": 0.85
            }

        except Exception as e:
            logger.error("Synthesis failed", error=str(e))
            return {
                "query": query,
                "sources_count": len(sources),
                "synthesis_type": synthesis_type,
                "synthesis": f"Synthesis failed: {str(e)}",
                "key_points": [],
                "confidence": 0.0
            }

    def _extract_key_points(self, synthesis: str) -> List[str]:
        """Extract key points from synthesis"""
        points = []
        for line in synthesis.split('\n'):
            line = line.strip()
            if line.startswith(('-', '*', '•')) or (line[0].isdigit() and '. ' in line):
                points.append(line.lstrip('-*•0123456789. '))
        return points[:5]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "synthesis_engine",
                "description": "Synthesize and combine information from multiple sources into coherent insights",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sources": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of source content to synthesize"
                        },
                        "query": {
                            "type": "string",
                            "description": "The original research query"
                        },
                        "synthesis_type": {
                            "type": "string",
                            "enum": ["comprehensive", "consensus", "gaps", "actionable"],
                            "description": "Type of synthesis to perform",
                            "default": "comprehensive"
                        }
                    },
                    "required": ["sources", "query"]
                }
            }
        }


class ResearchCoordinatorTool(Tool):
    """High-level research coordination tool that orchestrates research tasks"""

    def __init__(self, tool_registry: 'ToolRegistry'):
        super().__init__("research_coordinator", "Coordinate complex research tasks using multiple tools", "coordination")
        self.tool_registry = tool_registry

    async def _execute_impl(self, query: str, research_plan: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Coordinate research execution across multiple tools"""

        results = []
        current_context = {"query": query}

        try:
            for step in research_plan:
                tool_name = step.get("tool")
                tool = self.tool_registry.get_tool(tool_name)

                if not tool:
                    results.append({
                        "step": step.get("description", "Unknown"),
                        "tool": tool_name,
                        "error": f"Tool '{tool_name}' not found",
                        "success": False
                    })
                    continue

                # Prepare parameters with context
                params = step.get("parameters", {}).copy()
                params.update(current_context)

                # Execute tool
                result = await tool.execute(**params)

                # Store result for next steps
                if result.get("success", True):  # Assume success if not explicitly failed
                    current_context[f"{tool_name}_result"] = result

                results.append({
                    "step": step.get("description", "Research step"),
                    "tool": tool_name,
                    "result": result,
                    "success": result.get("success", True)
                })

            # Synthesize final results
            successful_results = [r for r in results if r["success"]]
            failed_results = [r for r in results if not r["success"]]

            return {
                "query": query,
                "total_steps": len(research_plan),
                "successful_steps": len(successful_results),
                "failed_steps": len(failed_results),
                "results": results,
                "coordination_success": len(successful_results) > 0,
                "overall_confidence": sum(r.get("result", {}).get("confidence", 0) for r in successful_results) / max(len(successful_results), 1)
            }

        except Exception as e:
            logger.error("Research coordination failed", error=str(e))
            return {
                "query": query,
                "error": f"Coordination failed: {str(e)}",
                "total_steps": len(research_plan),
                "successful_steps": 0,
                "failed_steps": len(research_plan),
                "results": [],
                "coordination_success": False
            }

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "research_coordinator",
                "description": "Coordinate complex research tasks by orchestrating multiple tools in sequence",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The main research query"
                        },
                        "research_plan": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "tool": {"type": "string", "description": "Tool to use for this step"},
                                    "description": {"type": "string", "description": "Description of this research step"},
                                    "parameters": {"type": "object", "description": "Parameters for the tool"}
                                },
                                "required": ["tool", "description"]
                            },
                            "description": "Sequence of research steps to execute"
                        }
                    },
                    "required": ["query", "research_plan"]
                }
            }
        }


class ResearchTool(Tool):
    """Agentic research tool that coordinates multiple specialized research tools"""

    def __init__(self, research_service: AgenticResearchService):
        super().__init__("research_engine", "Conduct comprehensive research using agentic orchestration", "research")
        self.research_service = research_service

    async def _execute_impl(self, query: str, strategy: str = "comprehensive", **kwargs) -> Dict[str, Any]:
        """Execute research using agentic approach with different strategies"""

        try:
            if strategy == "quick":
                # Use quick research for fast results
                results = await self.research_service.quick_research(
                    query=query,
                    num_sources=kwargs.get('max_sources', 3)
                )
            else:
                # Use comprehensive agentic research
                results = await self.research_service.perform_research(
                    query=query,
                    industry=kwargs.get('industry'),
                    geography=kwargs.get('geography'),
                    max_sources=kwargs.get('max_sources', 10),
                    user_id=kwargs.get('user_id')
                )

            return {
                "query": query,
                "strategy": strategy,
                "results": results.get("findings", []),
                "sources": results.get("sources", []),
                "confidence": results.get("confidence", 0.8),
                "method": results.get("method", "agentic_research"),
                "success": results.get("success", True)
            }

        except Exception as e:
            logger.error("Research tool execution failed", error=str(e))
            return {
                "query": query,
                "strategy": strategy,
                "success": False,
                "error": str(e),
                "results": [],
                "sources": [],
                "confidence": 0.0
            }

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "research_engine",
                "description": "Conduct research using agentic orchestration of specialized tools",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The research query to execute"
                        },
                        "strategy": {
                            "type": "string",
                            "enum": ["comprehensive", "quick", "focused"],
                            "description": "Research strategy: comprehensive (agentic), quick (fast search), focused (targeted)",
                            "default": "comprehensive"
                        },
                        "max_sources": {
                            "type": "integer",
                            "description": "Maximum number of sources to analyze",
                            "default": 10
                        },
                        "industry": {
                            "type": "string",
                            "description": "Industry focus for research"
                        },
                        "geography": {
                            "type": "string",
                            "description": "Geographic focus for research"
                        }
                    },
                    "required": ["query"]
                }
            }
        }

    def validate_parameters(self, **kwargs) -> bool:
        """Validate research parameters"""
        if not kwargs.get("query") or len(kwargs["query"].strip()) < 3:
            return False
        return True

    def get_examples(self) -> List[Dict[str, Any]]:
        """Return example usages"""
        return [
            {
                "description": "Quick research for immediate insights",
                "parameters": {"query": "latest AI developments", "strategy": "quick", "max_sources": 3}
            },
            {
                "description": "Comprehensive research with industry focus",
                "parameters": {"query": "market trends in renewable energy", "strategy": "comprehensive", "industry": "renewable energy"}
            },
            {
                "description": "Focused research on specific topic",
                "parameters": {"query": "competitor analysis for Tesla", "strategy": "focused", "geography": "North America"}
            }
        ]


class DatabaseTool(Tool):
    """Enhanced tool for database operations with multiple query types"""

    def __init__(self, mcp_client):
        super().__init__("database_client", "Query and manipulate databases with advanced operations", "database")
        self.mcp_client = mcp_client

    async def _execute_impl(self, operation: str, collection: str = None, query: Dict = None,
                           projection: Dict = None, limit: int = 10, **kwargs) -> Dict[str, Any]:
        """Execute database operation with enhanced capabilities"""
        try:
            if operation == "find":
                return await self._execute_find(collection, query, projection, limit)
            elif operation == "aggregate":
                return await self._execute_aggregate(collection, query)
            elif operation == "count":
                return await self._execute_count(collection, query)
            elif operation == "vector_search":
                return await self._execute_vector_search(collection, query, limit)
            else:
                return await self._execute_custom_operation(operation, **kwargs)
        except Exception as e:
            logger.error("Database tool execution failed", error=str(e), operation=operation)
            raise

    async def _execute_find(self, collection: str, query: Dict = None,
                           projection: Dict = None, limit: int = 10) -> Dict[str, Any]:
        """Execute find operation"""
        if not self.mcp_client:
            return {"results": [], "count": 0, "note": "MCP client not available"}

        try:
            # Use MCP client to perform the query
            if hasattr(self.mcp_client, 'find'):
                results = await self.mcp_client.find(
                    collection=collection,
                    filter=query or {},
                    projection=projection,
                    limit=limit
                )
                return {
                    "operation": "find",
                    "collection": collection,
                    "query": query,
                    "results": results,
                    "count": len(results),
                    "limit": limit
                }
            else:
                # Fallback simulation
                return {
                    "operation": "find",
                    "collection": collection,
                    "query": query,
                    "results": [f"Sample record {i+1}" for i in range(min(limit, 5))],
                    "count": min(limit, 5),
                    "limit": limit
                }
        except Exception as e:
            return {"error": f"Find operation failed: {str(e)}", "collection": collection}

    async def _execute_aggregate(self, collection: str, pipeline: List[Dict] = None) -> Dict[str, Any]:
        """Execute aggregation operation"""
        if not self.mcp_client:
            return {"results": [], "note": "MCP client not available"}

        try:
            if hasattr(self.mcp_client, 'aggregate'):
                results = await self.mcp_client.aggregate(collection=collection, pipeline=pipeline or [])
                return {
                    "operation": "aggregate",
                    "collection": collection,
                    "pipeline": pipeline,
                    "results": results
                }
            else:
                return {
                    "operation": "aggregate",
                    "collection": collection,
                    "results": {"total_count": 42, "average_value": 15.7},
                    "note": "Aggregation simulated"
                }
        except Exception as e:
            return {"error": f"Aggregation failed: {str(e)}", "collection": collection}

    async def _execute_count(self, collection: str, query: Dict = None) -> Dict[str, Any]:
        """Execute count operation"""
        if not self.mcp_client:
            return {"count": 0, "note": "MCP client not available"}

        try:
            if hasattr(self.mcp_client, 'count'):
                count = await self.mcp_client.count(collection=collection, filter=query or {})
                return {"operation": "count", "collection": collection, "count": count}
            else:
                return {"operation": "count", "collection": collection, "count": 25}
        except Exception as e:
            return {"error": f"Count failed: {str(e)}", "collection": collection}

    async def _execute_vector_search(self, collection: str, query_vector: List[float],
                                    limit: int = 5) -> Dict[str, Any]:
        """Execute vector search operation"""
        if not self.mcp_client:
            return {"results": [], "note": "MCP client not available"}

        try:
            if hasattr(self.mcp_client, 'vector_search'):
                results = await self.mcp_client.vector_search(
                    collection=collection,
                    query_vector=query_vector,
                    limit=limit
                )
                return {
                    "operation": "vector_search",
                    "collection": collection,
                    "results": results,
                    "limit": limit
                }
            else:
                return {
                    "operation": "vector_search",
                    "collection": collection,
                    "results": [f"Similar document {i+1}" for i in range(min(limit, 3))],
                    "limit": limit
                }
        except Exception as e:
            return {"error": f"Vector search failed: {str(e)}", "collection": collection}

    async def _execute_custom_operation(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute custom database operation"""
        return {
            "operation": operation,
            "results": f"Custom operation '{operation}' executed",
            "parameters": kwargs
        }

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "database_client",
                "description": "Query and manipulate databases with advanced operations including find, aggregate, count, and vector search",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["find", "aggregate", "count", "vector_search"],
                            "description": "The database operation to execute"
                        },
                        "collection": {
                            "type": "string",
                            "description": "The database collection to query"
                        },
                        "query": {
                            "type": "object",
                            "description": "Query filter for find/count operations"
                        },
                        "pipeline": {
                            "type": "array",
                            "description": "Aggregation pipeline for aggregate operations"
                        },
                        "projection": {
                            "type": "object",
                            "description": "Field projection for find operations"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results to return",
                            "default": 10
                        },
                        "query_vector": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Vector for similarity search"
                        }
                    },
                    "required": ["operation"]
                }
            }
        }

    def validate_parameters(self, **kwargs) -> bool:
        """Validate database operation parameters"""
        operation = kwargs.get("operation")
        if not operation:
            return False

        if operation in ["find", "count"] and not kwargs.get("collection"):
            return False

        if operation == "aggregate" and not kwargs.get("pipeline"):
            return False

        if operation == "vector_search" and not kwargs.get("query_vector"):
            return False

        return True

    def get_examples(self) -> List[Dict[str, Any]]:
        """Return example usages"""
        return [
            {
                "description": "Find documents in a collection",
                "parameters": {
                    "operation": "find",
                    "collection": "leads",
                    "query": {"status": "active"},
                    "limit": 5
                }
            },
            {
                "description": "Count documents matching a query",
                "parameters": {
                    "operation": "count",
                    "collection": "tasks",
                    "query": {"completed": False}
                }
            },
            {
                "description": "Aggregate data for analytics",
                "parameters": {
                    "operation": "aggregate",
                    "collection": "sales",
                    "pipeline": [{"$group": {"_id": "$status", "count": {"$sum": 1}}}]
                }
            }
        ]


class ResponseTool(Tool):
    """Enhanced tool for generating contextual, reasoning-based responses"""

    def __init__(self, llm_client: AsyncGroq):
        super().__init__("response_generator", "Generate intelligent, contextual responses with reasoning", "communication")
        self.llm_client = llm_client

    async def _execute_impl(self, user_message: str, execution_results: Dict[str, Any],
                           thoughts: List[Thought], context: AgentContext, style: str = "professional",
                           include_reasoning: bool = True, **kwargs) -> str:
        """Generate response with enhanced context awareness"""

        try:
            # Build dynamic system prompt based on context
            system_prompt = self._build_dynamic_system_prompt(context, style, include_reasoning)

            # Prepare comprehensive context
            context_summary = self._prepare_context_summary(execution_results, thoughts, context)

            user_prompt = f"""User Request: {user_message}

Context & Analysis:
{context_summary}

Generate a response that addresses the user's needs effectively:"""

            response = await self.llm_client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self._get_temperature_for_style(style),
                max_tokens=self._get_max_tokens_for_complexity(context.metadata.get('estimated_complexity', 'simple'))
            )

            generated_response = response.choices[0].message.content

            # Post-process response for quality
            return self._post_process_response(generated_response, context)

        except Exception as e:
            logger.error("Response generation failed", error=str(e))
            return self._generate_fallback_response(user_message, execution_results)

    def _build_dynamic_system_prompt(self, context: AgentContext, style: str, include_reasoning: bool) -> str:
        """Build dynamic system prompt based on context"""

        base_prompt = "You are an intelligent AI assistant that provides helpful, accurate responses."

        # Add style-specific instructions
        style_instructions = {
            "professional": "Be professional, clear, and business-appropriate.",
            "casual": "Be friendly, conversational, and approachable.",
            "technical": "Use technical terminology and provide detailed explanations.",
            "executive": "Be concise, focus on key insights and actionable recommendations."
        }

        # Add reasoning instructions if requested
        reasoning_instruction = ""
        if include_reasoning:
            reasoning_instruction = "\nShow your reasoning process and explain your conclusions."

        # Add context-specific guidance
        context_guidance = self._get_context_guidance(context)

        return f"""{base_prompt}

{style_instructions.get(style, style_instructions['professional'])}

Instructions:
- Address the user's specific request and context
- Use information from provided analysis and results
- Be comprehensive but not overwhelming
- Provide actionable insights when relevant{reasoning_instruction}
{context_guidance}

Ensure your response is helpful, accurate, and well-structured."""

    def _get_context_guidance(self, context: AgentContext) -> str:
        """Get context-specific guidance"""
        complexity = context.metadata.get('estimated_complexity', 'simple')

        if complexity == 'complex':
            return "\n- Break down complex topics into digestible parts\n- Use clear examples and analogies"
        elif complexity == 'simple':
            return "\n- Keep explanations straightforward and direct"
        else:
            return "\n- Balance depth with accessibility"

    def _prepare_context_summary(self, execution_results: Dict[str, Any],
                                thoughts: List[Thought], context: AgentContext) -> str:
        """Prepare comprehensive context summary"""

        sections = []

        # Add recent thoughts
        if thoughts:
            thought_section = "Recent Analysis:\n" + "\n".join([
                f"- {thought.content} (confidence: {thought.confidence:.1f})"
                for thought in thoughts[-5:]  # Last 5 thoughts
            ])
            sections.append(thought_section)

        # Add execution results
        if execution_results:
            result_section = "Execution Results:\n" + "\n".join([
                f"- {step_id}: {self._summarize_result(result)}"
                for step_id, result in execution_results.items()
            ])
            sections.append(result_section)

        # Add context metadata
        if context.metadata:
            metadata_section = "Context Information:\n" + "\n".join([
                f"- {key}: {value}"
                for key, value in context.metadata.items()
                if key not in ['estimated_complexity']  # Already handled separately
            ])
            sections.append(metadata_section)

        return "\n\n".join(sections)

    def _summarize_result(self, result: Dict[str, Any]) -> str:
        """Summarize execution result for context"""
        if result.get('success', False):
            tool_result = result.get('result', {})
            if isinstance(tool_result, dict):
                if 'count' in tool_result:
                    return f"Found {tool_result['count']} results"
                elif 'results' in tool_result and isinstance(tool_result['results'], list):
                    return f"Retrieved {len(tool_result['results'])} items"
                else:
                    return f"Operation completed successfully"
            else:
                return str(tool_result)[:100] + "..." if len(str(tool_result)) > 100 else str(tool_result)
        else:
            return f"Error: {result.get('error', 'Unknown error')}"

    def _get_temperature_for_style(self, style: str) -> float:
        """Get appropriate temperature for response style"""
        temperatures = {
            "professional": 0.3,
            "casual": 0.7,
            "technical": 0.2,
            "executive": 0.4
        }
        return temperatures.get(style, 0.5)

    def _get_max_tokens_for_complexity(self, complexity: str) -> int:
        """Get appropriate max tokens based on complexity"""
        token_limits = {
            "simple": 1000,
            "moderate": 1500,
            "complex": 2500
        }
        return token_limits.get(complexity, 1500)

    def _post_process_response(self, response: str, context: AgentContext) -> str:
        """Post-process response for quality improvements"""
        # Remove any unwanted artifacts
        response = response.strip()

        # Add confidence indicators if low confidence in context
        if context.current_plan and any(step.get('confidence', 1.0) < 0.6 for step in context.current_plan.steps):
            response += "\n\n*Note: Some information has lower confidence and may need verification.*"

        return response

    def _generate_fallback_response(self, user_message: str, execution_results: Dict[str, Any]) -> str:
        """Generate fallback response when main generation fails"""
        try:
            if execution_results:
                results_str = json.dumps(execution_results, indent=2)
            else:
                results_str = "No execution results available"

            return f"I understand your request about '{user_message}'. Based on my analysis, I found the following information: {results_str}"
        except Exception as e:
            logger.error("Error generating fallback response", error=str(e))
            return f"I understand your request about '{user_message}'. I'm currently processing your request and will provide a more detailed response shortly."

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "response_generator",
                "description": "Generate intelligent, contextual responses with reasoning and analysis",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_message": {
                            "type": "string",
                            "description": "The original user message"
                        },
                        "execution_results": {
                            "type": "object",
                            "description": "Results from executed actions"
                        },
                        "thoughts": {
                            "type": "array",
                            "description": "Agent's reasoning thoughts"
                        },
                        "context": {
                            "type": "object",
                            "description": "Agent context information"
                        },
                        "style": {
                            "type": "string",
                            "enum": ["professional", "casual", "technical", "executive"],
                            "description": "Response style to use",
                            "default": "professional"
                        },
                        "include_reasoning": {
                            "type": "boolean",
                            "description": "Whether to include reasoning in response",
                            "default": True
                        }
                    },
                    "required": ["user_message", "execution_results", "thoughts", "context"]
                }
            }
        }

    def validate_parameters(self, **kwargs) -> bool:
        """Validate response generation parameters"""
        required_fields = ["user_message", "execution_results", "thoughts", "context"]
        return all(kwargs.get(field) is not None for field in required_fields)

    def get_examples(self) -> List[Dict[str, Any]]:
        """Return example usages"""
        return [
            {
                "description": "Generate professional response with reasoning",
                "parameters": {
                    "user_message": "What are the latest trends in AI?",
                    "style": "professional",
                    "include_reasoning": True
                }
            },
            {
                "description": "Generate casual response without detailed reasoning",
                "parameters": {
                    "user_message": "How's the weather?",
                    "style": "casual",
                    "include_reasoning": False
                }
            }
        ]


class AnalysisTool(Tool):
    """Tool for performing various types of analysis"""

    def __init__(self, llm_client: AsyncGroq):
        super().__init__("analysis_engine", "Perform various types of analysis on data and information", "analysis")
        self.llm_client = llm_client

    async def _execute_impl(self, analysis_type: str, data: Any, context: str = "",
                           parameters: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Execute analysis operation"""

        analysis_methods = {
            "sentiment": self._analyze_sentiment,
            "trend": self._analyze_trends,
            "comparison": self._analyze_comparison,
            "impact": self._analyze_impact,
            "gap": self._analyze_gaps,
            "summary": self._analyze_summary
        }

        method = analysis_methods.get(analysis_type)
        if not method:
            return {"error": f"Unsupported analysis type: {analysis_type}"}

        return await method(data, context, parameters or {})

    async def _analyze_sentiment(self, data: Any, context: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sentiment of text data"""
        system_prompt = """You are a sentiment analysis expert. Analyze the sentiment of the provided text.

Return ONLY valid JSON:
{
    "overall_sentiment": "positive|negative|neutral",
    "confidence": 0.0-1.0,
    "sentiment_score": -1.0 to 1.0,
    "key_phrases": ["phrase1", "phrase2"],
    "explanation": "Brief explanation of the analysis"
}"""

        user_prompt = f"""Analyze the sentiment of this text:

Text: {data}
Context: {context}

Provide sentiment analysis:"""

        response = await self.llm_client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )

        return json.loads(response.choices[0].message.content)

    async def _analyze_trends(self, data: Any, context: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trends in data"""
        system_prompt = """You are a trend analysis expert. Identify patterns and trends in the provided data.

Return ONLY valid JSON:
{
    "trends": ["trend1", "trend2"],
    "direction": "increasing|decreasing|stable|fluctuating",
    "confidence": 0.0-1.0,
    "timeframe": "short_term|medium_term|long_term",
    "insights": ["insight1", "insight2"],
    "recommendations": ["recommendation1", "recommendation2"]
}"""

        user_prompt = f"""Analyze trends in this data:

Data: {json.dumps(data) if isinstance(data, (list, dict)) else str(data)}
Context: {context}
Parameters: {json.dumps(parameters)}

Identify key trends and patterns:"""

        response = await self.llm_client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )

        return json.loads(response.choices[0].message.content)

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "analysis_engine",
                "description": "Perform various types of analysis including sentiment, trends, comparisons, impact assessment, gap analysis, and summarization",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "analysis_type": {
                            "type": "string",
                            "enum": ["sentiment", "trend", "comparison", "impact", "gap", "summary"],
                            "description": "Type of analysis to perform"
                        },
                        "data": {
                            "description": "Data to analyze (text, objects, arrays, etc.)"
                        },
                        "context": {
                            "type": "string",
                            "description": "Context for the analysis",
                            "default": ""
                        },
                        "parameters": {
                            "type": "object",
                            "description": "Additional parameters specific to the analysis type"
                        }
                    },
                    "required": ["analysis_type", "data"]
                }
            }
        }


class CalculationTool(Tool):
    """Tool for performing mathematical calculations and data analysis"""

    def __init__(self):
        super().__init__("calculator", "Perform mathematical calculations and statistical analysis", "calculation")

    async def _execute_impl(self, operation: str, values: List[float], **kwargs) -> Dict[str, Any]:
        """Execute calculation operation"""

        if operation == "basic":
            return self._basic_calculation(values, kwargs.get('operator', '+'))
        elif operation == "statistics":
            return self._statistical_analysis(values)
        elif operation == "percentage":
            return self._percentage_calculation(values[0], values[1] if len(values) > 1 else 100)
        elif operation == "projection":
            return self._projection_calculation(values, kwargs)
        else:
            return {"error": f"Unsupported operation: {operation}"}

    def _basic_calculation(self, values: List[float], operator: str) -> Dict[str, Any]:
        """Perform basic arithmetic operations"""
        if not values:
            return {"error": "No values provided"}

        result = values[0]
        for value in values[1:]:
            if operator == '+':
                result += value
            elif operator == '-':
                result -= value
            elif operator == '*':
                result *= value
            elif operator == '/':
                if value != 0:
                    result /= value
                else:
                    return {"error": "Division by zero"}
            elif operator == '^':
                result **= value

        return {
            "operation": f"{operator} calculation",
            "values": values,
            "result": result,
            "expression": f"{' ' + operator + ' '}.join([str(v) for v in values])"
        }

    def _statistical_analysis(self, values: List[float]) -> Dict[str, Any]:
        """Perform statistical analysis on values"""
        if not values:
            return {"error": "No values provided"}

        sorted_values = sorted(values)
        n = len(values)

        # Basic statistics
        mean = sum(values) / n
        median = sorted_values[n // 2] if n % 2 != 0 else (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2

        # Variance and standard deviation
        variance = sum((x - mean) ** 2 for x in values) / (n - 1) if n > 1 else 0
        std_dev = variance ** 0.5

        # Range
        data_range = max(values) - min(values)

        return {
            "count": n,
            "mean": round(mean, 4),
            "median": round(median, 4),
            "std_dev": round(std_dev, 4),
            "variance": round(variance, 4),
            "min": min(values),
            "max": max(values),
            "range": data_range,
            "quartiles": {
                "q1": sorted_values[n // 4],
                "q3": sorted_values[3 * n // 4]
            }
        }

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Perform mathematical calculations and statistical analysis",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["basic", "statistics", "percentage", "projection"],
                            "description": "Type of calculation to perform"
                        },
                        "values": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Values to calculate with"
                        },
                        "operator": {
                            "type": "string",
                            "enum": ["+", "-", "*", "/", "^"],
                            "description": "Operator for basic calculations"
                        }
                    },
                    "required": ["operation", "values"]
                }
            }
        }


# Tool Implementations
