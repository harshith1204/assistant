"""Memory management with Mem0 for context handling"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, timezone
import asyncio
from mem0 import Memory
from pathlib import Path
import structlog
from app.config import settings
from app.chat_models import ConversationContext, ChatMessage, MessageRole
import hashlib

logger = structlog.get_logger()


class MemoryManager:
    """Manages short-term and long-term memory using Mem0"""
    
    def __init__(self):
        """Initialize memory manager with Mem0"""
        # Configure Mem0 - using a simpler configuration without explicit embedder
        # Mem0 will use its default embedding configuration
        # Resolve a stable absolute path for the vector store to persist across restarts
        resolved_db_path = str(Path(settings.memory_db_path).resolve())
        try:
            Path(resolved_db_path).mkdir(parents=True, exist_ok=True)
        except Exception:
            # Directory may already exist or path may be a file; ignore errors here
            pass

        config = {
            "llm": {
                "provider": "groq",
                "config": {
                    "model": settings.llm_model,
                    "api_key": settings.groq_api_key,
                    "temperature": settings.memory_temperature,
                    "max_tokens": settings.memory_max_tokens,
                }
            },
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": settings.memory_collection_name,
                    # Use absolute path to ensure persistence is stable across working directories
                    "path": resolved_db_path,
                }
            },
            "version": "v1.1"
        }
        
        # Try to add embedder configuration if huggingface provider is specified
        if settings.memory_embedder_provider == "huggingface":
            try:
                config["embedder"] = {
                    "provider": "huggingface",
                    "config": {
                        "model": settings.memory_embedder_model,
                    }
                }
                self.memory = Memory.from_config(config)
                logger.info("Memory manager initialized with HuggingFace embeddings", 
                          provider=settings.memory_embedder_provider, 
                          model=settings.memory_embedder_model)
            except Exception as e:
                logger.warning("Failed to initialize with HuggingFace embeddings, using default", 
                             error=str(e))
                # Remove embedder config and use default
                config.pop("embedder", None)
                self.memory = Memory.from_config(config)
                logger.info("Memory manager initialized with default embeddings")
        else:
            # Use default embeddings
            self.memory = Memory.from_config(config)
            logger.info("Memory manager initialized with default embeddings")
            
        self.short_term_cache: Dict[str, Dict] = {}  # In-memory cache for short-term
        self.cache_ttl = timedelta(hours=24)  # Short-term memory TTL
        # Recent content hashes to prevent duplicate long-term writes (per user)
        self.recent_hashes: Dict[str, Dict[str, Any]] = {}
        self.recent_hash_ttl = timedelta(days=7)

        # Memory level constants
        self.PROFILE_LEVEL = "profile"
        self.CONVERSATION_LEVEL = "conversation"
        self.USER_LEVEL = "user"

    def _hash_content(self, content: str) -> str:
        """Create a stable hash for deduplication"""
        return hashlib.sha256(content.strip().lower().encode("utf-8")).hexdigest()
    
    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize metadata to ensure all values are primitive types for mem0"""
        sanitized = {}
        for key, value in metadata.items():
            if value is None or isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, (list, tuple)):
                # Convert lists/tuples to comma-separated strings
                if value:  # Non-empty list
                    sanitized[key] = ", ".join(str(v) for v in value)
                else:  # Empty list
                    sanitized[key] = ""
            elif isinstance(value, dict):
                # Convert dicts to JSON strings
                import json
                sanitized[key] = json.dumps(value)
            else:
                # Convert other types to string
                sanitized[key] = str(value)
        return sanitized
        
    async def add_to_memory(
        self,
        conversation_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        memory_type: str = "both"  # "user", "conversation", or "both"
    ) -> Dict[str, Any]:
        """Add information to memory
        
        Args:
            conversation_id: The conversation identifier
            content: The content to store
            metadata: Additional metadata
            user_id: The user identifier (required for user-level memories)
            memory_type: Where to store - "user" (long-term), "conversation" (short-term), or "both"
        """
        # Prepare metadata
        meta = metadata or {}
        meta.update({
            "conversation_id": conversation_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": user_id or "anonymous"
        })
        
        # Store in short-term cache for conversation context
        if memory_type in ["conversation", "both"]:
            if conversation_id not in self.short_term_cache:
                self.short_term_cache[conversation_id] = {
                    "messages": [],
                    "context": {},
                    "timestamp": datetime.now(timezone.utc)
                }
            
            self.short_term_cache[conversation_id]["messages"].append({
                "content": content,
                "metadata": meta,
                "timestamp": datetime.now(timezone.utc)
            })
        
        # Add to long-term memory (Mem0) for user context
        mem0_result = None
        mem0_error = None
        
        if memory_type in ["user", "both"] and user_id:
            try:
                # Dedup check for long-term writes (avoid flooding)
                content_hash = self._hash_content(content)
                user_hashes = self.recent_hashes.get(user_id, {})
                # Cleanup stale hashes for this user
                now = datetime.now(timezone.utc)
                if user_hashes:
                    stale = [h for h, info in user_hashes.items() if now - info.get("timestamp", now) > self.recent_hash_ttl]
                    for h in stale:
                        user_hashes.pop(h, None)
                # Skip if we recently stored the same fact
                if content_hash in user_hashes:
                    logger.info("Skipping duplicate long-term memory write", user_id=user_id)
                else:
                    # Sanitize metadata for mem0
                    sanitized_meta = self._sanitize_metadata(meta)
                    
                    # Add to Mem0 with user_id for cross-conversation persistence
                    mem0_result = await asyncio.to_thread(
                        self.memory.add,
                        content,
                        user_id=user_id,  # Always use user_id for long-term
                        metadata=sanitized_meta
                    )
                    # Track hash
                    user_hashes[content_hash] = {"timestamp": now}
                    self.recent_hashes[user_id] = user_hashes
            
                logger.info(
                    "Added to long-term memory",
                    conversation_id=conversation_id,
                    memory_id=mem0_result.get("id") if isinstance(mem0_result, dict) else None
                )
            
            except Exception as e:
                mem0_error = str(e)
                logger.warning("Failed to add to long-term memory (Mem0), but short-term cache updated", error=mem0_error)
            
        # Return success if at least short-term cache was updated
        return {
            "success": True,
            "memory_id": mem0_result,
            "short_term_cached": True,
            "long_term_stored": mem0_result is not None,
            "warning": mem0_error if mem0_error else None
        }
    
    async def search_memory(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 5,
        search_scope: str = "both"  # "user", "conversation", or "both"
    ) -> List[Dict[str, Any]]:
        """Search memories
        
        Args:
            query: The search query
            conversation_id: The conversation identifier
            user_id: The user identifier
            limit: Maximum number of results
            search_scope: Where to search - "user" (long-term), "conversation" (short-term), or "both"
        """
        results = []
        
        try:
            # Search user-level long-term memories if user_id is provided
            if search_scope in ["user", "both"] and user_id:
                search_result = await asyncio.to_thread(
                    self.memory.search,
                    query,
                    user_id=user_id,  # Search user memories
                    limit=limit
                )
                
                # Extract results from the response (handles both v1.0 and v1.1 formats)
                if isinstance(search_result, dict) and "results" in search_result:
                    user_results = search_result["results"]
                else:
                    # Fallback for older format or direct list
                    user_results = search_result if isinstance(search_result, list) else []
                
                # Mark these as user-level memories and append one by one
                for result in user_results:
                    try:
                        result["memory_level"] = "user"
                        results.append(result)
                    except Exception:
                        # Be resilient to unexpected formats
                        continue
            
            # Search conversation-level short-term cache
            if search_scope in ["conversation", "both"] and conversation_id and conversation_id in self.short_term_cache:
                cache_data = self.short_term_cache[conversation_id]
                # Add recent messages from cache that match the query
                recent_messages = cache_data.get("messages", [])
                query_lower = query.lower()
                
                for msg in recent_messages[-10:]:  # Check last 10 messages
                    if query_lower in msg["content"].lower():
                        results.append({
                            "memory": msg["content"],
                            "metadata": msg["metadata"],
                            "score": 0.95,  # High score for recent relevant messages
                            "source": "short_term",
                            "memory_level": "conversation"
                        })
            
            return results
            
        except Exception as e:
            logger.error("Failed to search memory", error=str(e))
            return []
    
    async def get_conversation_context(
        self,
        conversation_id: str,
        user_id: Optional[str] = None
    ) -> ConversationContext:
        """Get complete context for a conversation
        
        Returns both user-level (long-term) and conversation-level (short-term) contexts
        """
        try:
            context = ConversationContext()
            
            # Always get conversation-level short-term context
            if conversation_id in self.short_term_cache:
                cache_data = self.short_term_cache[conversation_id]
                context.short_term = {
                    "recent_messages": cache_data.get("messages", [])[-5:],
                    "session_start": cache_data.get("timestamp", datetime.now(timezone.utc)).isoformat(),
                    "message_count": len(cache_data.get("messages", [])),
                    "conversation_id": conversation_id
                }
            
            # Get user-level long-term context only if user_id is provided
            if user_id:
                memories_result = await asyncio.to_thread(
                    self.memory.get_all,
                    user_id=user_id  # Always use user_id for long-term
                )
                
                # Extract memories from the response (handles both v1.0 and v1.1 formats)
                if isinstance(memories_result, dict) and "results" in memories_result:
                    all_memories = memories_result["results"]
                elif isinstance(memories_result, list):
                    all_memories = memories_result
                else:
                    all_memories = []
            else:
                all_memories = []
            
            if all_memories:
                # Extract entities and topics
                entities = set()
                topics = set()
                
                for memory in all_memories:
                    # Handle both dict and object formats
                    if isinstance(memory, dict):
                        memory_text = memory.get("memory", "")
                        meta = memory.get("metadata", {}) or {}
                    else:
                        # Handle object format (if mem0 returns objects)
                        memory_text = getattr(memory, "memory", "")
                        meta = getattr(memory, "metadata", {}) or {}

                    # Normalize meta values that may be JSON-encoded strings
                    def _ensure_list(value):
                        if value is None:
                            return []
                        if isinstance(value, list):
                            return value
                        if isinstance(value, str):
                            try:
                                parsed = json.loads(value)
                                if isinstance(parsed, list):
                                    return parsed
                                return [parsed]
                            except Exception:
                                # Comma-separated string fallback
                                return [v.strip() for v in value.split(",") if v.strip()]
                        return [value]

                    # Extract entities/topics robustly
                    if "entities" in meta:
                        entities.update(_ensure_list(meta.get("entities")))

                    if "topics" in meta:
                        topics.update(_ensure_list(meta.get("topics")))
                
                context.long_term = {
                    "total_memories": len(all_memories),
                    "memories": all_memories[:10],  # Recent 10 memories
                    "first_interaction": self._get_memory_timestamp(all_memories[-1]) if all_memories else None
                }
                
                context.entities = list(entities)
                context.topics = list(topics)
            
            return context
            
        except Exception as e:
            logger.error("Failed to get conversation context", error=str(e))
            return ConversationContext()
    
    async def update_from_message(
        self,
        message: ChatMessage,
        user_id: Optional[str] = None
    ):
        """Update memory from a chat message"""
        try:
            # Prepare content for memory
            content = f"{message.role.value}: {message.content}"
            
            metadata = {
                "role": message.role.value,
                "message_type": message.message_type.value,
                "message_id": message.message_id
            }
            
            # Add research-specific information
            if message.research_brief_id:
                metadata["research_brief_id"] = message.research_brief_id
                content += f" [Research Brief: {message.research_brief_id}]"
            
            # Add to memory
            await self.add_to_memory(
                conversation_id=message.conversation_id,
                content=content,
                metadata=metadata,
                user_id=user_id
            )
            
            # Extract and store entities/topics for important messages
            if message.role in [MessageRole.USER, MessageRole.ASSISTANT]:
                await self._extract_and_store_entities(message, user_id)
                
        except Exception as e:
            logger.error("Failed to update memory from message", error=str(e))

    async def set_profile_fact(
        self,
        user_id: str,
        key: str,
        value: str,
        priority: int = 50
    ) -> Dict[str, Any]:
        """Store or update a high-priority profile fact for a user"""
        content = f"{key}: {value}"
        metadata = {
            "memory_level": self.PROFILE_LEVEL,
            "pinned": True,
            "key": key,
            "priority": priority
        }
        return await self.add_to_memory(
            conversation_id=f"profile-{user_id}",
            content=content,
            metadata=metadata,
            user_id=user_id,
            memory_type="user"
        )

    async def get_profile(self, user_id: str) -> List[Dict[str, Any]]:
        """Fetch profile-level memories for a user"""
        try:
            search_result = await asyncio.to_thread(
                self.memory.get_all,
                user_id=user_id
            )
            if isinstance(search_result, dict) and "results" in search_result:
                all_memories = search_result["results"]
            elif isinstance(search_result, list):
                all_memories = search_result
            else:
                all_memories = []
            profile = []
            for m in all_memories:
                metadata = m.get("metadata", {}) if isinstance(m, dict) else getattr(m, "metadata", {}) or {}
                level = metadata.get("memory_level")
                pinned = metadata.get("pinned")
                if level == self.PROFILE_LEVEL or pinned is True:
                    profile.append(m)
            # Sort by optional priority then key
            def _prio(x):
                meta = x.get("metadata", {}) if isinstance(x, dict) else getattr(x, "metadata", {}) or {}
                return (-(meta.get("priority") or 0), str(meta.get("key") or ""))
            profile.sort(key=_prio)
            return profile
        except Exception as e:
            logger.error("Failed to get profile", error=str(e))
            return []
    
    async def _extract_and_store_entities(
        self,
        message: ChatMessage,
        user_id: Optional[str] = None
    ):
        """Extract and store entities from message"""
        try:
            # Simple entity extraction (can be enhanced with NER)
            # For now, we'll look for capitalized words and common patterns
            
            entities = []
            topics = []
            
            # Extract potential entities (capitalized words)
            words = message.content.split()
            for word in words:
                if word and word[0].isupper() and len(word) > 2:
                    entities.append(word.strip('.,!?'))
            
            # Extract topics based on keywords
            topic_keywords = {
                "research": ["research", "study", "analyze", "investigate"],
                "planning": ["plan", "strategy", "roadmap", "timeline"],
                "market": ["market", "competition", "competitors", "industry"],
                "product": ["product", "feature", "development", "build"],
                "customer": ["customer", "user", "client", "audience"],
                "pricing": ["price", "cost", "budget", "revenue"],
                "technology": ["tech", "software", "platform", "system"]
            }
            
            content_lower = message.content.lower()
            for topic, keywords in topic_keywords.items():
                if any(keyword in content_lower for keyword in keywords):
                    topics.append(topic)
            
            # Store extracted information
            if entities or topics:
                metadata = {
                    "entities": list(set(entities)),
                    "topics": list(set(topics)),
                    "extraction_timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                await self.add_to_memory(
                    conversation_id=message.conversation_id,
                    content=f"Entities and topics from message: {', '.join(entities + topics)}",
                    metadata=metadata,
                    user_id=user_id
                )
                
        except Exception as e:
            logger.error("Failed to extract entities", error=str(e))
    
    async def clear_short_term_memory(self, conversation_id: str):
        """Clear short-term memory for a conversation"""
        if conversation_id in self.short_term_cache:
            del self.short_term_cache[conversation_id]
            logger.info("Cleared short-term memory", conversation_id=conversation_id)
    
    async def cleanup_old_cache(self):
        """Clean up old short-term cache entries"""
        try:
            current_time = datetime.now(timezone.utc)
            expired_conversations = []
            
            for conv_id, cache_data in self.short_term_cache.items():
                if current_time - cache_data.get("timestamp", current_time) > self.cache_ttl:
                    expired_conversations.append(conv_id)
            
            for conv_id in expired_conversations:
                del self.short_term_cache[conv_id]
                
            if expired_conversations:
                logger.info(
                    "Cleaned up expired cache",
                    count=len(expired_conversations)
                )
                
        except Exception as e:
            logger.error("Failed to cleanup cache", error=str(e))
    
    def _get_memory_timestamp(self, memory) -> Optional[str]:
        """Extract timestamp from a memory object or dict"""
        if isinstance(memory, dict):
            metadata = memory.get("metadata", {})
            return metadata.get("timestamp") if metadata else None
        else:
            # Handle object format
            metadata = getattr(memory, "metadata", {})
            return metadata.get("timestamp") if metadata else None
    
    def ensure_dual_context(self, conversation_id: str, user_id: Optional[str] = None) -> tuple[str, str]:
        """Ensure we have both conversation_id and user_id
        
        Returns:
            Tuple of (conversation_id, user_id) where user_id is generated if not provided
        """
        if not user_id:
            # Generate an anonymous user_id that persists for the session
            # This could be stored in browser localStorage for persistence
            import uuid
            user_id = f"anon_{uuid.uuid4().hex[:12]}"
            logger.info("Generated anonymous user_id", user_id=user_id)
        
        return conversation_id, user_id
    
    async def get_memory_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get memory statistics"""
        try:
            memories_result = await asyncio.to_thread(
                self.memory.get_all,
                user_id=user_id or "global"
            )
            
            # Extract memories from the response
            if isinstance(memories_result, dict) and "results" in memories_result:
                all_memories = memories_result["results"]
            elif isinstance(memories_result, list):
                all_memories = memories_result
            else:
                all_memories = []
            
            return {
                "total_memories": len(all_memories),
                "short_term_conversations": len(self.short_term_cache),
                "cache_size": sum(
                    len(cache.get("messages", []))
                    for cache in self.short_term_cache.values()
                ),
                "oldest_memory": self._get_memory_timestamp(all_memories[-1]) if all_memories else None,
                "newest_memory": self._get_memory_timestamp(all_memories[0]) if all_memories else None
            }
            
        except Exception as e:
            logger.error("Failed to get memory stats", error=str(e))
            return {}
