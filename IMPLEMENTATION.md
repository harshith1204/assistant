# Single WebSocket Agent Implementation

## Overview

This implementation provides a **complete, end-to-end agent** that runs fully on a **single WebSocket connection**, maintaining both short-term context and long-term (Mem0) memory, with integrated research capabilities and all tools in one unified loop.

## Architecture

### Core Principle: One Socket, One Brain

- **WebSocket** is the only API surface - no separate HTTP endpoints for the agent flow
- **WebSocket handler** acts as a simple traffic cop, routing messages by `type`
- **ChatEngine** implements a turn-based FSM with 5 clear stages
- **MemoryManager** provides tiered memory (profile, conversation, user) with composite ranking
- **Typed events** stream back to the client, making the agent's thinking visible

## Message Protocol

### Client → Server Messages

```json
// Send a chat message
{"type": "chat.send", "data": {
  "conversation_id": "c1",
  "message": "Plan a 3-day Goa trip",
  "use_web_search": true,
  "context_window": 4096
}}

// Resume session
{"type": "session.resume", "data": {
  "user_id": "u123",
  "conversation_id": "c1"
}}

// Set profile fact
{"type": "memory.set", "data": {
  "key": "diet",
  "value": "vegetarian",
  "priority": 80
}}

// Get memories
{"type": "memory.get", "data": {
  "conversation_id": "c1",
  "scope": "both",
  "limit": 8
}}

// Cancel generation
{"type": "agent.cancel", "data": {
  "conversation_id": "c1"
}}
```

### Server → Client Events

The server emits these events in order during processing:

1. **`session_info`** - User and connection IDs
2. **`memory.used`** - Exact memories injected this turn
3. **`research.started`** / **`research.chunk`** / **`research.done`** - Research progress
4. **`chat.token`** - Streaming response tokens
5. **`memory.written`** - When facts are persisted
6. **`memory.summary_created`** - Rolling summary created
7. **`chat.final`** - Final message with suggestions
8. **`clarification_needed`** - Low confidence intent
9. **`error`** - Normalized errors

## 5-Stage FSM Pipeline

The `ChatEngine.stream_message` method implements a clear 5-stage pipeline:

### Stage 1: RECEIVE & LOG
- Normalize conversation_id and user_id
- Write user turn to memory (with gates)
- Load context (profile + summary + ranked memories)
- Emit `memory.used` event

### Stage 2: UNDERSTAND INTENT
- Detect intent with confidence scoring
- Extract entities and parameters
- If confidence < 0.5: emit `clarification_needed` and return

### Stage 3: PLAN (Routing Policy)
- Determine strategy based on intent:
  - `research` → research-first
  - `profile_update` → direct profile write
  - Default → direct-answer

### Stage 4: EXECUTE
- **Research-first**: Stream research progress, synthesize findings
- **Profile update**: Write facts, confirm to user
- **Generate**: Stream LLM tokens with enhanced context

### Stage 5: COMPLETE
- Persist assistant turn (with write gates)
- Create rolling summary every ~10 turns
- Extract high-signal facts for long-term storage
- Generate suggestions
- Emit `chat.final` with complete response

## Memory Architecture

### Three-Tier Memory System

1. **Profile (Pinned) Level**
   - High-priority user facts (name, preferences, etc.)
   - Always injected into context
   - Set via `set_profile_fact()`

2. **Conversation Level**
   - Short-term context for current thread
   - Rolling summaries every ~10 turns
   - Stored in memory cache

3. **User Level**
   - Long-term cross-conversation memories
   - Retrieved with semantic search
   - Persisted in Mem0

### Composite Ranking

When retrieving memories, the system applies composite scoring:

```python
score = 0.6 * semantic_score + 0.3 * recency_boost + boosts

where:
- recency_boost = exp(-age_days/30)
- profile/pinned boost: +0.2
- conversation match boost: +0.1
- entity overlap boost: +0.1
```

### Write Gates

To avoid memory spam, writes only occur when:
- Explicit memory cues ("remember", "save", "note that")
- Strong entity patterns ("I am", "I prefer", "My X is Y")
- High-confidence profile_update intent
- Research results or substantial assistant responses

## Token-Tight Prompt Scaffold

The synthesis module builds a structured prompt with strict limits:

```
SYSTEM:
- Role + capabilities

PROFILE:              # ≤ 8 short lines
- Name: Alice
- Diet: vegetarian
- Tone: casual

RECENT SUMMARY:       # ≤ 120 tokens
[Thread summary]

LONG-TERM FACTS:      # 5-7 bullets
- User prefers budget travel
- Allergic to peanuts

CURRENT TASK:
- Intent: research
- Entities: [location: Goa, duration: 3 days]

RESEARCH NOTES:       # ≤ 150 tokens + citations
[Distilled findings]
```

## Key Implementation Files

### Core Components

- **`websocket_handler.py`** - Message router with typed protocol
- **`chat_engine.py`** - 5-stage FSM pipeline implementation
- **`memory_manager.py`** - Tiered memory with composite ranking
- **`synthesis.py`** - Token-tight prompt scaffolding
- **`intent.py`** - Intent detection with confidence scoring

### Helper Methods

- `_determine_routing_strategy()` - Intent → execution strategy
- `_apply_composite_ranking()` - Memory scoring algorithm
- `_should_write_to_memory()` - Write gate logic
- `build_prompt()` - Structured prompt generation
- `_extract_high_signal_facts()` - Fact extraction

## Usage Examples

### Basic Chat with Personalization

```python
client = AgentClient()
await client.connect()

# Set profile
await client.set_profile("name", "Alice")
await client.set_profile("diet", "vegetarian")

# Chat with research
response = await client.chat(
    "Find vegetarian restaurants in Goa",
    use_research=True
)

# Follow-up uses context
response = await client.chat(
    "What about budget options?"
)
```

### Memory Management

```python
# Natural profile updates
await client.chat("Remember that I'm allergic to peanuts")

# Check memories
memories = await client.get_memories(limit=8)

# Explicit profile fact
await client.set_profile("budget", "under $50/day", priority=70)
```

## Testing

Run the test suite to validate the implementation:

```bash
# Complete flow test
python test_websocket_flow.py

# Example conversations
python example_client.py
```

## Observability

The implementation provides visibility through:

- **`memory.used`** events showing exact injected context
- **`research.*`** events for research progress
- **`memory.written`** events for persistence tracking
- **Composite scores** in memory retrieval
- **Intent confidence** in responses

## Configuration

Key settings in `config.py`:

```python
# Memory settings
memory_collection_name = "agent_memories"
memory_temperature = 0.4
summary_max_tokens = 150

# Chat settings
chat_temperature = 0.7
chat_max_tokens = 1000

# Research settings
research_max_sources = 15
```

## Benefits of This Architecture

1. **Single Connection** - No connection overhead, instant responses
2. **Visible Thinking** - Client sees memory usage, research progress
3. **True Personalization** - Profile + ranked memories + summaries
4. **Efficient Context** - Token-tight scaffolds prevent bloat
5. **Cancellable** - Any operation can be cancelled mid-flight
6. **Debuggable** - Typed events make the flow transparent

## Future Enhancements

- [ ] Add tool calling (CRM, PMS, calendar) to the pipeline
- [ ] Implement memory TTL and decay
- [ ] Add conversation branching/forking
- [ ] Support multiple models (fallback chain)
- [ ] Add memory export/import
- [ ] Implement memory privacy controls

## Troubleshooting

### Memory not persisting?
- Check write gates in `_should_write_to_memory()`
- Verify user_id is consistent across sessions
- Check Mem0 configuration

### Research not triggering?
- Ensure `use_web_search=true` in request
- Check intent detection confidence
- Verify research_engine is configured

### Context not being used?
- Check `memory.used` event for injected memories
- Verify profile facts are set with proper priority
- Check composite ranking scores

---

This implementation provides a production-ready foundation for a personalized, research-capable agent that maintains context across conversations while operating entirely through a single WebSocket connection.