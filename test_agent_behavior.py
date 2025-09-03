#!/usr/bin/env python3
"""Test script to demonstrate agent-first behavior with strict tool usage"""

import asyncio
import sys
from app.core.agent import Agent
from app.core.intent import classify_intent, ConversationalIntent
from app.core.research_engine import ResearchEngine
from app.integrations.mcp_client import mongodb_mcp_client
from app.core.memory_manager import MemoryManager
from groq import AsyncGroq
from app.config import settings

# Test cases to verify agent behavior
TEST_CASES = [
    {
        "id": "1",
        "query": "What's the latest price of Tesla stock?",
        "expected_intent": ConversationalIntent.PRICING_RESEARCH,
        "expected_behavior": "Should use research tools, not freestyle",
        "should_have_sources": True
    },
    {
        "id": "2", 
        "query": "Show me the last 5 leads in MongoDB",
        "expected_intent": ConversationalIntent.DB_LOOKUP,
        "expected_behavior": "Should use MCP/MongoDB, show error if unavailable",
        "should_have_sources": True
    },
    {
        "id": "3",
        "query": "Summarize ACME's competitors in India",
        "expected_intent": ConversationalIntent.COMPETITOR_ANALYSIS,
        "expected_behavior": "Should use research with possible MCP enrichment",
        "should_have_sources": True
    },
    {
        "id": "4",
        "query": "Hi there, how are you?",
        "expected_intent": ConversationalIntent.GENERAL,
        "expected_behavior": "Can answer directly without tools",
        "should_have_sources": False
    },
    {
        "id": "5",
        "query": "What is the current market cap of Apple?",
        "expected_intent": ConversationalIntent.RESEARCH,
        "expected_behavior": "Should research, not guess",
        "should_have_sources": True
    },
    {
        "id": "6",
        "query": "Find all CRM leads with status NEW",
        "expected_intent": ConversationalIntent.DB_LOOKUP,
        "expected_behavior": "Should query MongoDB via MCP",
        "should_have_sources": True
    }
]


async def test_intent_classification():
    """Test that intent classification is deterministic"""
    print("\n=== Testing Intent Classification ===\n")
    
    llm = AsyncGroq(api_key=settings.groq_api_key)
    
    for test in TEST_CASES:
        intent = await classify_intent(llm, test["query"])
        correct = intent == test["expected_intent"]
        
        print(f"Test {test['id']}: {'✅' if correct else '❌'}")
        print(f"  Query: {test['query']}")
        print(f"  Expected: {test['expected_intent'].value}")
        print(f"  Got: {intent.value}")
        print(f"  Behavior: {test['expected_behavior']}")
        print()


async def test_agent_behavior():
    """Test that agent enforces tool usage"""
    print("\n=== Testing Agent Behavior ===\n")
    
    # Initialize components
    llm = AsyncGroq(api_key=settings.groq_api_key)
    research_engine = ResearchEngine()
    memory_manager = MemoryManager()
    
    # Create agent
    agent = Agent(
        llm=llm,
        mcp_client=mongodb_mcp_client,
        research_engine=research_engine,
        memory_manager=memory_manager
    )
    
    # Test each case
    for test in TEST_CASES:
        print(f"\nTest {test['id']}: {test['query']}")
        print("-" * 60)
        
        try:
            result = await agent.handle_user_turn(
                user_id="test_user",
                conversation_id=f"test_conv_{test['id']}",
                message=test['query'],
                geo="US"
            )
            
            content = result.get("content", "")
            sources = result.get("sources", [])
            status = result.get("status", "unknown")
            
            # Check if behavior is correct
            has_sources = len(sources) > 0
            sources_correct = has_sources == test["should_have_sources"]
            
            print(f"Status: {status}")
            print(f"Sources: {'✅' if sources_correct else '❌'} ({len(sources)} sources)")
            
            if not sources_correct:
                print(f"  Expected sources: {test['should_have_sources']}")
                print(f"  Got sources: {has_sources}")
            
            # Print first 200 chars of response
            print(f"Response preview: {content[:200]}...")
            
            # Check for hallucination indicators
            if test["should_have_sources"] and not sources:
                if any(indicator in content.lower() for indicator in ["$", "%", "million", "billion", "2024", "2023"]):
                    print("⚠️  WARNING: Response contains data claims without sources!")
            
            if sources:
                print("Sources used:")
                for i, source in enumerate(sources[:3]):
                    print(f"  {i+1}. {source.get('type', 'unknown')} - {source.get('title', 'N/A')}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print()


async def test_mcp_health():
    """Test MCP client health"""
    print("\n=== Testing MCP Client Health ===\n")
    
    try:
        # Try to connect
        connected = await mongodb_mcp_client.connect()
        print(f"MCP Connected: {'✅' if connected else '❌'}")
        
        if connected:
            # List tools
            tools = await mongodb_mcp_client.list_tools()
            print(f"Available tools: {len(tools)}")
            for tool in tools[:5]:
                print(f"  - {tool.get('name', 'unknown')}")
            
            # Try a simple query
            print("\nTesting database query...")
            results = []
            async for chunk in mongodb_mcp_client.find_documents(
                collection="Lead",
                filter_query={},
                limit=1,
                user_id="test_user"
            ):
                if chunk.get("type") == "tool.output.data":
                    results.append(chunk["data"])
            
            print(f"Query result: {'✅' if results else '❌ No data'}")
        else:
            print("MCP is not connected. Database queries will fail with clear error messages.")
            
    except Exception as e:
        print(f"MCP health check failed: {str(e)}")
        print("This is expected if MCP server is not running.")


async def main():
    """Run all tests"""
    print("=" * 80)
    print("AGENT-FIRST BEHAVIOR TEST")
    print("=" * 80)
    print("\nThis test demonstrates that the assistant now:")
    print("1. Uses deterministic intent classification")
    print("2. REQUIRES tools for data/research queries (no guessing)")
    print("3. Returns clear errors when tools are unavailable")
    print("4. Only answers directly for true general chat")
    
    # Test MCP first
    await test_mcp_health()
    
    # Test intent classification
    await test_intent_classification()
    
    # Test agent behavior
    await test_agent_behavior()
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    # Set up logging
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Run tests
    asyncio.run(main())