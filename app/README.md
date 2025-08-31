Research & Chat Backend - Memory Model

Overview

This backend integrates dual-context memory for conversations:

- Long-term context (user-level): persisted using Mem0 (vector store: Chroma). Memories are scoped by user_id and persist across conversations and sessions.
- Short-term context (conversation-level): in-memory cache per conversation_id with a TTL (default 24h) used as a rolling context buffer.

Key Components

- app/core/memory_manager.py: Orchestrates Mem0 for long-term and maintains a short-term cache.
  - add_to_memory(..., memory_type="user|conversation|both")
  - search_memory(query, conversation_id, user_id, search_scope="user|conversation|both")
  - get_conversation_context(conversation_id, user_id) → returns ConversationContext with short_term and long_term
  - ensure_dual_context(conversation_id, user_id) → generates anon user_id if missing

- app/core/chat_engine.py: Injects both contexts into LLM prompts and stores messages into memory.
  - Uses search_memory when use_long_term_memory is True (default).
  - Includes recent messages (short-term) and relevant user memories (long-term) in system prompt.

Using user_id (to enable long-term memory)

Persist a stable user_id client-side (e.g., localStorage) and pass it on each request. If omitted, the server may generate a temporary anonymous ID that won’t persist across requests.

- HTTP
  - Endpoint: POST /chat/message?user_id={your_user_id}
  - Body: ChatRequest { message: string, conversation_id?: string, ... }

- WebSocket
  - Connect: /ws/chat/{connection_id}?user_id={your_user_id}
  - Send chat messages with type: "chat" and data: ChatRequest payload

Configuration

Environment variables (see app/config.py for defaults):

- GROQ_API_KEY: API key for LLM provider
- MEMORY_DB_PATH: Path for Chroma DB (default: ./chroma_db)
- MEMORY_COLLECTION_NAME: Chroma collection (default: chat_memories)
- MEMORY_EMBEDDER_PROVIDER: embedding provider (default: huggingface)
- MEMORY_EMBEDDER_MODEL: model for embeddings (default: sentence-transformers/all-MiniLM-L6-v2)
- MEMORY_TTL_HOURS: short-term cache TTL hours (default: 24)

Dependencies

- mem0ai (imported as "from mem0 import Memory")
- chromadb

Notes

- Long-term memory only activates when user_id is provided (or consistently persisted). For best results, generate a UUID per user and reuse it.
- Short-term context is per conversation_id; keep conversation_id stable in the UI while a session is active.

