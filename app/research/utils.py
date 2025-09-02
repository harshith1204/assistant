"""Utility functions and helpers for the Deep Research agent."""

import asyncio
import logging
import os
import warnings
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Dict, List, Literal, Optional

import aiohttp
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    MessageLikeRepresentation,
    filter_messages,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import (
    BaseTool,
    InjectedToolArg,
    StructuredTool,
    ToolException,
    tool,
)
from langgraph.config import get_store

from .configuration import Configuration, SearchAPI
from .prompts import summarize_webpage_prompt
from .state import ResearchComplete, Summary

##########################
# DDGS Search Tool Utils - Optimized for Groq + DDGS
##########################
DDGS_SEARCH_DESCRIPTION = (
    "DuckDuckGo search optimized for privacy and comprehensive business research. "
    "Excellent for market analysis, competitor research, and industry insights."
)
@tool(description=DDGS_SEARCH_DESCRIPTION)
async def ddgs_business_search(
    queries: List[str],
    max_results: Annotated[int, InjectedToolArg] = 5,
    config: RunnableConfig = None
) -> str:
    """Fetch and summarize search results from DDGS search API.

    Args:
        queries: List of search queries to execute
        max_results: Maximum number of results to return per query
        config: Runtime configuration for API keys and model settings

    Returns:
        Formatted string containing summarized search results
    """
    # Step 1: Execute DDGS search queries
    search_results = await ddgs_search_async(
        queries,
        max_results=max_results,
        include_raw_content=True,
        config=config
    )

    # Step 2: Deduplicate results by URL
    unique_results = {}
    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = {**result, "query": response['query']}

    # Step 3: Set up the summarization model with configuration
    configurable = Configuration.from_runnable_config(config)

    # Character limit to stay within model token limits
    max_char_to_include = configurable.max_content_length

    # Initialize summarization model with Groq
    summarization_model = init_chat_model(
        model=configurable.summarization_model,
        max_tokens=configurable.summarization_model_max_tokens,
        api_key=get_api_key_for_model(configurable.summarization_model, config),
        tags=["langsmith:nostream"]
    ).with_structured_output(Summary).with_retry(
        stop_after_attempt=configurable.max_structured_output_retries
    )

    # Step 4: Create summarization tasks
    async def noop():
        """No-op function for results without raw content."""
        return None

    summarization_tasks = [
        noop() if not result.get("raw_content")
        else summarize_webpage(
            summarization_model,
            result['raw_content'][:max_char_to_include]
        )
        for result in unique_results.values()
    ]

    # Step 5: Execute all summarization tasks in parallel
    summaries = await asyncio.gather(*summarization_tasks)

    # Step 6: Combine results with their summaries
    summarized_results = {
        url: {
            'title': result['title'],
            'content': result['content'] if summary is None else summary
        }
        for url, result, summary in zip(
            unique_results.keys(),
            unique_results.values(),
            summaries
        )
    }

    # Step 7: Format the final output
    if not summarized_results:
        return "No valid search results found. Please try different search queries."

    formatted_output = "DDGS Search Results:\n\n"
    for i, (url, result) in enumerate(summarized_results.items()):
        formatted_output += f"\n\n--- SOURCE {i+1}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        formatted_output += "\n\n" + "-" * 80 + "\n"

    return formatted_output

async def ddgs_search_async(
    search_queries,
    max_results: int = 5,
    include_raw_content: bool = True,
    config: RunnableConfig = None
):
    """Execute multiple DDGS search queries asynchronously.

    Args:
        search_queries: List of search query strings to execute
        max_results: Maximum number of results per query
        include_raw_content: Whether to include full webpage content
        config: Runtime configuration

    Returns:
        List of search result dictionaries from DDGS
    """
    # Import DDGS here to avoid import issues
    from ddgs import DDGS
    import httpx

    async def search_single_query(query: str):
        """Search for a single query"""
        try:
            # Use synchronous DDGS since it doesn't support async
            ddgs = DDGS()
            results = ddgs.text(query, max_results=max_results)

            formatted_results = []
            for result in results:
                result_dict = {
                    'url': result.get('href', ''),
                    'title': result.get('title', ''),
                    'content': result.get('body', ''),
                    'query': query
                }

                # Fetch content if requested
                if include_raw_content and result_dict['url']:
                    try:
                        async with httpx.AsyncClient(timeout=10.0) as client:
                            response = await client.get(result_dict['url'])
                            if response.status_code == 200:
                                result_dict['raw_content'] = response.text[:5000]  # Limit content
                    except Exception:
                        pass  # Skip content fetch on error

                formatted_results.append(result_dict)

            return {'query': query, 'results': formatted_results}

        except Exception as e:
            return {'query': query, 'results': [], 'error': str(e)}

    # Execute all search queries in parallel
    search_tasks = [search_single_query(query) for query in search_queries]
    search_results = await asyncio.gather(*search_tasks)
    return search_results

async def summarize_webpage(model: BaseChatModel, webpage_content: str) -> str:
    """Summarize webpage content using AI model with timeout protection.
    
    Args:
        model: The chat model configured for summarization
        webpage_content: Raw webpage content to be summarized
        
    Returns:
        Formatted summary with key excerpts, or original content if summarization fails
    """
    try:
        # Create prompt with current date context
        prompt_content = summarize_webpage_prompt.format(
            webpage_content=webpage_content, 
            date=get_today_str()
        )
        
        # Execute summarization with timeout to prevent hanging
        summary = await asyncio.wait_for(
            model.ainvoke([HumanMessage(content=prompt_content)]),
            timeout=60.0  # 60 second timeout for summarization
        )
        
        # Format the summary with structured sections
        formatted_summary = (
            f"<summary>\n{summary.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
        )
        
        return formatted_summary
        
    except asyncio.TimeoutError:
        # Timeout during summarization - return original content
        logging.warning("Summarization timed out after 60 seconds, returning original content")
        return webpage_content
    except Exception as e:
        # Other errors during summarization - log and return original content
        logging.warning(f"Summarization failed with error: {str(e)}, returning original content")
        return webpage_content

##########################
# Reflection Tool Utils
##########################

@tool(description="Strategic reflection tool for research planning")
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"

##########################
# Tool Utils
##########################

async def get_search_tool(search_api: SearchAPI):
    """Configure and return search tools - optimized for Groq + DDGS stack.

    Args:
        search_api: The search API provider to use (DDGS or None)

    Returns:
        List of configured search tool objects
    """
    if search_api == SearchAPI.DDGS:
        # Configure DDGS search tool with metadata
        search_tool = ddgs_business_search
        search_tool.metadata = {
            **(search_tool.metadata or {}),
            "type": "search",
            "name": "ddgs_business_search"
        }
        return [search_tool]

    elif search_api == SearchAPI.NONE:
        # No search functionality configured
        return []

    # Default fallback
    return []
    
async def get_all_tools(config: RunnableConfig):
    """Assemble complete toolkit including research and search tools.

    Args:
        config: Runtime configuration specifying search API settings
        
    Returns:
        List of all configured and available tools for research operations
    """
    # Start with core research tools
    tools = [tool(ResearchComplete), think_tool]
    
    # Add configured search tools
    configurable = Configuration.from_runnable_config(config)
    search_api = SearchAPI(get_config_value(configurable.search_api))
    search_tools = await get_search_tool(search_api)
    tools.extend(search_tools)
    
    # Track existing tool names to prevent conflicts
    existing_tool_names = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search") 
        for tool in tools
    }
    

    
    return tools

def get_notes_from_tool_calls(messages: list[MessageLikeRepresentation]):
    """Extract research notes and findings from tool call messages.

    This function filters through message history to extract content from tool
    execution results, which typically contain research findings, search results,
    and analysis outputs that need to be preserved for synthesis.

    Args:
        messages: List of message representations from the conversation

    Returns:
        List of content strings from tool call messages
    """
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]

##########################
# Groq + DDGS Websearch Utils
##########################

def groq_websearch_called(response):
    """Detect if DDGS web search was used with Groq model.

    Args:
        response: The response object from Groq API

    Returns:
        True if DDGS search was used, False otherwise
    """
    try:
        # Check if DDGS tool was called in the response
        tool_calls = getattr(response, 'tool_calls', [])
        if not tool_calls:
            return False

        # Look for DDGS search tool calls
        for tool_call in tool_calls:
            if tool_call.get("name") == "ddgs_business_search":
                return True

        return False

    except (AttributeError, TypeError):
        return False


##########################
# Token Limit Exceeded Utils
##########################

def is_token_limit_exceeded(exception: Exception, model_name: str = None) -> bool:
    """Determine if an exception indicates a token/context limit was exceeded for Groq.

    Args:
        exception: The exception to analyze
        model_name: Optional model name to optimize provider detection

    Returns:
        True if the exception indicates a token limit was exceeded, False otherwise
    """
    error_str = str(exception).lower()
    exception_type = str(type(exception)).lower()
    class_name = exception.__class__.__name__

    # Check if this is a Groq exception
    is_groq_exception = (
        'groq' in exception_type or
        'groq' in error_str
    )

    # Check for Groq-specific token limit patterns
    if is_groq_exception:
        token_keywords = ['token', 'context', 'length', 'maximum', 'limit', 'exceeded']
        if any(keyword in error_str for keyword in token_keywords):
            return True

        # Check for common HTTP error codes indicating token limits
        if hasattr(exception, 'status_code') and exception.status_code == 429:
            return True

    return False

# Groq model token limits - optimized for high-performance research
MODEL_TOKEN_LIMITS = {
    "groq:llama3-8b-8192": 8192,
    "groq:llama3-70b-8192": 8192,
    "groq:llama3-8b-8192": 8192,  # Default for research tasks
    "groq:mixtral-8x7b-32768": 32768,
    "groq:gemma-7b-it": 8192,
    "groq:gemma2-9b-it": 8192,
}

def get_model_token_limit(model_string):
    """Look up the token limit for a specific model.
    
    Args:
        model_string: The model identifier string to look up
        
    Returns:
        Token limit as integer if found, None if model not in lookup table
    """
    # Search through known model token limits
    for model_key, token_limit in MODEL_TOKEN_LIMITS.items():
        if model_key in model_string:
            return token_limit
    
    # Model not found in lookup table
    return None

def remove_up_to_last_ai_message(messages: list[MessageLikeRepresentation]) -> list[MessageLikeRepresentation]:
    """Truncate message history by removing up to the last AI message.
    
    This is useful for handling token limit exceeded errors by removing recent context.
    
    Args:
        messages: List of message objects to truncate
        
    Returns:
        Truncated message list up to (but not including) the last AI message
    """
    # Search backwards through messages to find the last AI message
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            # Return everything up to (but not including) the last AI message
            return messages[:i]
    
    # No AI messages found, return original list
    return messages

##########################
# Misc Utils
##########################

def get_today_str() -> str:
    """Get current date formatted for display in prompts and outputs.
    
    Returns:
        Human-readable date string in format like 'Mon Jan 15, 2024'
    """
    now = datetime.now()
    return f"{now:%a} {now:%b} {now.day}, {now:%Y}"

def get_config_value(value):
    """Extract and normalize configuration values from various formats.

    This function handles different types of configuration values including
    strings, enums, dictionaries, and None values, converting them to their
    appropriate string or dictionary representations.

    Args:
        value: Configuration value that may be a string, enum, dict, or None

    Returns:
        Normalized configuration value as string, dict, or None
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    elif isinstance(value, dict):
        return value
    else:
        return value.value

def get_api_key_for_model(model_name: str, config: RunnableConfig):
    """Retrieve API key for specified model from environment or configuration.

    This function provides flexible API key retrieval for different model providers,
    supporting both environment variables and runtime configuration. It's optimized
    for research workflows with Groq models but supports other providers as well.

    Args:
        model_name: The model identifier (e.g., 'groq:llama3-8b-8192')
        config: Runtime configuration containing API keys and settings

    Returns:
        API key string if found, None if not available
    """
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    model_name = model_name.lower()

    if should_get_from_config.lower() == "true":
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        if not api_keys:
            return None
        if model_name.startswith("groq:"):
            return api_keys.get("GROQ_API_KEY")
        return None
    else:
        if model_name.startswith("groq:"):
            return os.getenv("GROQ_API_KEY")
        return None
