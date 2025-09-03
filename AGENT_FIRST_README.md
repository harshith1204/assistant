# Agent-First Architecture Implementation

## Overview

This implementation transforms the assistant from a "vibes-based" LLM that freestyles answers into a **strict agent** that MUST use tools (MCP/MongoDB, web research) when handling data queries. No more hallucinations or guessing!

## What Changed

### 1. **Strict Agent Loop** (`app/core/agent.py`)
- Every request goes through intent classification first
- Certain intents (research, database, pricing, etc.) REQUIRE tool usage
- If tools fail or return no data, the agent returns an honest error message
- No fallback to LLM guessing

### 2. **Deterministic Intent Router** (`app/core/intent.py`)
- Uses structured JSON output with `temperature=0`
- Fixed enum of intents: `RESEARCH`, `NEWS`, `DB_LOOKUP`, `CRM_ACTION`, `PMS_ACTION`, `PRICING_RESEARCH`, `COMPETITOR_ANALYSIS`, `PROFILE_UPDATE`, `GENERAL`
- Keyword-based fallback when LLM fails
- Only uses `GENERAL` intent for true small talk

### 3. **Source-Required Synthesis** (`app/core/synthesis.py`)
- Refuses to generate answers without sources for data-requiring intents
- Adds source citations automatically
- Prevents fabrication by checking for data claims in responses

### 4. **Enhanced Chat Engine** (`app/core/chat_engine.py`)
- Added "agent-controlled" routing strategy
- Integrates the new Agent class for strict tool usage
- Maintains streaming compatibility while enforcing tool usage

## How It Works

### Intent → Tool Mapping

| User Query | Intent | Required Tool | Behavior |
|------------|--------|---------------|----------|
| "What's the latest price of X?" | `PRICING_RESEARCH` | Web Research | Must fetch real data |
| "Show me leads in MongoDB" | `DB_LOOKUP` | MCP/MongoDB | Query DB or show error |
| "Research competitors in India" | `COMPETITOR_ANALYSIS` | Web Research + MCP | Comprehensive research |
| "Hi, how are you?" | `GENERAL` | None | Can answer directly |
| "What is the market cap of Y?" | `RESEARCH` | Web Research | Must research, not guess |

### Example Flow

1. User: "What's the current price of Tesla stock?"
2. Intent Classification: `PRICING_RESEARCH` (requires sources)
3. Agent routes to Research Engine
4. Research Engine fetches web sources
5. If sources found: Returns answer with citations
6. If no sources: Returns "I need to research that to provide accurate information"
7. **Never**: Makes up a price or uses outdated training data

## Testing

Run the test script to see the new behavior:

```bash
python test_agent_behavior.py
```

This will show:
- Deterministic intent classification
- Tool enforcement for data queries
- Clear error messages when tools unavailable
- Sources attached to all factual responses

## Key Principles

1. **No Guessing**: If the query needs fresh data, MUST use tools
2. **Fail Honestly**: If tools fail, say so clearly
3. **Sources Required**: Research/data claims must have sources
4. **Deterministic Routing**: Same query → same intent → same tool path
5. **Clear Errors**: "Database unavailable" not "I think the data might be..."

## Configuration

The agent respects these settings:
- `groq_model`: LLM model for intent classification and chat
- `mongodb_*`: MCP/MongoDB configuration
- `research_*`: Web research settings
- `chat_temperature`: Set to 0 for deterministic routing

## Extending

To add new tool-required intents:

1. Add to `ConversationalIntent` enum
2. Update `INTENT_SYSTEM_PROMPT` with the new intent
3. Add to `source_required_intents` in synthesis
4. Implement handler in Agent class
5. Add keyword patterns for fallback detection

## Monitoring

The agent logs key decisions:
```
router.intent=research confidence=0.9
mcp.health=true
research.sources.count=5
agent.final sources_count=5 has_content=true
```

This makes it clear when and why tools are used.

## Troubleshooting

**Q: Agent says "no sources" for everything**
A: Check that web search is enabled and MCP is connected

**Q: Still getting hallucinated responses**
A: Check logs for `agent_controlled=true` - ensure new routing is active

**Q: Database queries failing**
A: Run MCP health check, ensure MongoDB server is accessible

**Q: Intents misclassified**
A: Lower temperature further, add more keywords to fallback detection