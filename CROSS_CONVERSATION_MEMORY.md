# Cross-Conversation Memory Implementation

## ✅ Problem Solved

The system now maintains long-term context for the same user across different conversations. User memories, preferences, and profile facts persist and are accessible regardless of which conversation thread is active.

## 🏗️ Architecture

### Three-Tier Memory System

```
┌─────────────────────────────────────────────────────────┐
│                    USER (user_alice_123)                │
├─────────────────────────────────────────────────────────┤
│  PROFILE FACTS (Always Injected)                       │
│  ├─ name: Alice Chen                                   │
│  ├─ diet: vegetarian                                   │
│  ├─ allergies: peanuts, shellfish                      │
│  └─ location: San Francisco                            │
├─────────────────────────────────────────────────────────┤
│  USER MEMORIES (Cross-Conversation)                    │
│  ├─ "I prefer boutique hotels" (from conv1)            │
│  ├─ "I have a bad knee" (from conv1)                   │
│  └─ Accessible in ALL conversations                    │
├─────────────────────────────────────────────────────────┤
│  CONVERSATION MEMORIES (Thread-Specific)               │
│  ├─ Conv1: Trip planning context                       │
│  ├─ Conv2: Restaurant search context                   │
│  └─ Conv3: Weekend activity context                    │
└─────────────────────────────────────────────────────────┘
```

## 🔧 Key Changes Made

### 1. **MemoryManager** (`app/core/memory_manager.py`)

#### Profile Facts Management
```python
async def set_profile_fact(self, user_id: str, key: str, value: str, priority: int = 50):
    """Store high-priority profile fact that persists across ALL conversations"""
    # Stored with user_id and memory_level="profile"
    # Always retrieved and injected into context
```

#### Composite Ranking
```python
def _apply_composite_ranking(self, results, query, conversation_id, limit):
    """Rank memories with semantic + recency + profile/pinned boosts"""
    # Score = 0.6 * semantic + 0.3 * recency + boosts
    # Profile boost: +0.2
    # Conversation match: +0.1
    # Entity overlap: +0.1
```

#### Write Gates
```python
def _should_write_to_memory(self, message):
    """Only write meaningful information to avoid spam"""
    # Write if: explicit memory cues, strong patterns, research results
    # Skip if: trivial messages, short responses
```

#### Memory Type Determination
```python
def _determine_memory_type(self, message, user_id):
    """Decide whether to store at user, conversation, or both levels"""
    # Profile/personal → user level
    # Research results → both levels
    # Important patterns → both levels
    # Regular chat → conversation level
```

### 2. **ChatEngine** (`app/core/chat_engine.py`)

#### Context Preparation
```python
async def _prepare_context(self, conversation, request, user_id):
    # 1. Load profile facts (always if user_id exists)
    profile = await self.memory_manager.get_profile(user_id)
    
    # 2. Search ALL user memories + conversation memories
    memories = await self.memory_manager.search_memory(
        query=request.message,
        conversation_id=conversation.conversation_id,
        user_id=user_id,  # Critical: searches ALL user memories
        limit=10,
        search_scope="both"  # Both user and conversation levels
    )
    
    # 3. Apply composite ranking
    ranked = sorted(memories, key=_score, reverse=True)
```

#### 5-Stage FSM Pipeline
```python
async def stream_message(self, request, user_id):
    # Stage 1: RECEIVE & LOG
    # - Ensure consistent user_id
    effective_user_id = user_id or request.user_id
    
    # Stage 2: UNDERSTAND INTENT
    # - Detect profile updates, research needs
    
    # Stage 3: PLAN
    # - Route to profile-update, research-first, or direct-answer
    
    # Stage 4: EXECUTE
    # - Process with full cross-conversation context
    
    # Stage 5: COMPLETE
    # - Store high-signal facts at user level
    # - Create rolling summaries
```

### 3. **WebSocket Handler** (`app/websocket_handler.py`)

#### User ID Consistency
```python
# Maintains user_id across all operations
if message_type == "chat.send":
    async for event in manager.chat_engine.stream_message(
        request, 
        user_id or manager.connection_user.get(connection_id)
    ):
        # User_id is consistently passed
```

## 📊 How It Works

### Setting Profile Facts
```json
// Client sends
{
  "type": "memory.set",
  "data": {
    "key": "diet",
    "value": "vegetarian",
    "priority": 85
  }
}

// Stored at user level, accessible in ALL conversations
```

### Cross-Conversation Flow

#### Conversation 1: Trip Planning
```json
// User: "I prefer boutique hotels and have a bad knee"
// System stores at USER level (cross-conversation)
```

#### Conversation 2: Restaurant Search (Different Thread)
```json
// User: "Find me a restaurant"
// System retrieves:
//   - Profile: diet=vegetarian, allergies=peanuts
//   - User memories: boutique hotels, bad knee (from Conv1!)
// Response considers ALL user context
```

#### Conversation 3: Weekend Activities (Another Thread)
```json
// User: "Suggest activities"
// System has access to EVERYTHING:
//   - All profile facts
//   - All user memories from Conv1 and Conv2
//   - Can suggest activities avoiding strenuous knee activities
```

## 🧪 Testing

### Test Files Provided

1. **`test_cross_conversation.py`** - Full WebSocket test
   - Creates multiple conversations with same user
   - Verifies memories persist across conversations
   - Validates profile facts are always available

2. **`demo_cross_conversation.py`** - Conceptual demonstration
   - Shows the memory architecture
   - Demonstrates how memories flow across conversations
   - No external dependencies needed

### Running Tests

```bash
# Demonstration (no server needed)
python3 demo_cross_conversation.py

# Full test (requires server running)
python3 test_cross_conversation.py

# Integration test (requires API keys)
export GROQ_API_KEY="your-key"
export OPENAI_API_KEY="your-key"  # For embeddings
python3 test_memory_integration.py
```

## 🎯 Key Success Indicators

1. **Profile Facts Always Available**
   - Set once, available everywhere
   - Highest priority in context

2. **User Memories Cross Conversations**
   - Facts from Conv1 accessible in Conv2, Conv3, etc.
   - Stored with user_id, not conversation_id

3. **Composite Ranking Works**
   - Recent memories boosted
   - Profile facts prioritized
   - Relevant memories ranked higher

4. **Write Gates Prevent Spam**
   - Only meaningful information saved
   - Deduplication prevents duplicates
   - Trivial messages filtered

## 📝 Event Flow

When processing a message, the system emits:

1. `memory.used` - Shows exactly what memories were injected
   ```json
   {
     "type": "memory.used",
     "items": [
       {"level": "profile", "line": "diet: vegetarian"},
       {"level": "user", "line": "I prefer boutique hotels"},
       {"level": "conversation", "line": "Looking for restaurants"}
     ]
   }
   ```

2. `memory.written` - When new facts are persisted
3. `memory.summary_created` - Rolling summaries every ~10 turns

## 🚀 Benefits

1. **True Personalization** - User context persists forever
2. **Natural Conversations** - No need to repeat preferences
3. **Cross-Thread Intelligence** - Learning from one conversation helps another
4. **Transparent Memory** - Users see what the agent remembers
5. **Efficient Context** - Token-tight scaffolds prevent bloat

## 🔍 Debugging

If memories aren't persisting across conversations:

1. **Check user_id consistency**
   ```python
   # In chat_engine.py
   logger.info(f"Using user_id: {user_id}")
   ```

2. **Verify memory level**
   ```python
   # Memories should have memory_level="user" or "profile"
   ```

3. **Check search scope**
   ```python
   # Must be scope="both" or scope="user"
   ```

4. **Validate write gates**
   ```python
   # Check _should_write_to_memory() isn't filtering too much
   ```

## ✅ Summary

The system now provides **true cross-conversation memory**:

- **Profile facts** persist and are always injected
- **User memories** from any conversation are accessible in all others
- **Composite ranking** ensures relevant memories are prioritized
- **Write gates** prevent memory spam while capturing important facts
- **Transparent events** show what memories are being used

This creates a genuinely personalized experience where the agent remembers users across all interactions, making conversations feel natural and continuous.