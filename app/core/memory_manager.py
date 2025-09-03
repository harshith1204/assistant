"""Memory management with Mem0 for context handling"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, timezone
import asyncio
from mem0 import Memory
import structlog
from app.config import settings
from app.chat_models import ConversationContext, ChatMessage, MessageRole, MessageType
import hashlib

logger = structlog.get_logger()


class MemoryManager:
    """Manages short-term and long-term memory using Mem0"""
    
    def __init__(self):
        """Initialize memory manager with Mem0"""
        # Configure Mem0 - using a simpler configuration without explicit embedder
        # Mem0 will use its default embedding configuration
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
                    "path": settings.memory_db_path,
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
                    # Mark this as user-level memory in metadata (don't override if already set to profile)
                    if sanitized_meta.get("memory_level") != self.PROFILE_LEVEL:
                        sanitized_meta["memory_level"] = "user"
                    mem0_result = await asyncio.to_thread(
                        self.memory.add,
                        content,
                        user_id=user_id,  # Always use user_id for long-term, NOT conversation_id
                        metadata=sanitized_meta
                    )
                    # Track hash
                    user_hashes[content_hash] = {"timestamp": now}
                    self.recent_hashes[user_id] = user_hashes
            
                logger.info(
                    "Added to long-term memory",
                    user_id=user_id,
                    conversation_id=conversation_id,
                    memory_id=mem0_result.get("id") if isinstance(mem0_result, dict) else None,
                    memory_type=memory_type,
                    content_preview=content[:100]
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
        """Search memories with composite ranking
        
        Args:
            query: The search query
            conversation_id: The conversation identifier
            user_id: The user identifier
            limit: Maximum number of results
            search_scope: Where to search - "user" (long-term), "conversation" (short-term), or "both"
        """
        results = []
        
        logger.info(
            "Searching memory",
            query=query[:50],
            user_id=user_id,
            conversation_id=conversation_id,
            search_scope=search_scope,
            limit=limit
        )
        
        try:
            # Search user-level long-term memories if user_id is provided
            # IMPORTANT: User memories should be searched across ALL conversations
            if search_scope in ["user", "both"] and user_id:
                # Search ALL user memories regardless of conversation
                search_result = await asyncio.to_thread(
                    self.memory.search,
                    query,
                    user_id=user_id,  # Search ALL user memories
                    limit=limit * 3  # Get more for ranking since we'll filter
                )
                
                # Extract results from the response (handles both v1.0 and v1.1 formats)
                if isinstance(search_result, dict) and "results" in search_result:
                    user_results = search_result["results"]
                else:
                    # Fallback for older format or direct list
                    user_results = search_result if isinstance(search_result, list) else []
                
                logger.info(f"Found {len(user_results)} user-level memories for user {user_id}")
                
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
            
            # Apply composite ranking
            ranked_results = self._apply_composite_ranking(
                results, query, conversation_id, limit
            )
            
            return ranked_results
            
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
        """Update memory from a chat message with write gates"""
        try:
            # Apply write gates to avoid memory spam
            should_write = self._should_write_to_memory(message)
            
            if not should_write:
                logger.debug("Skipping memory write due to gates", message_id=message.message_id)
                return
            
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
            
            # Determine memory type based on content and role
            memory_type = self._determine_memory_type(message, user_id)
            
            # Add to memory
            await self.add_to_memory(
                conversation_id=message.conversation_id,
                content=content,
                metadata=metadata,
                user_id=user_id,
                memory_type=memory_type
            )
            
            # Extract and store entities/topics for important messages
            if message.role in [MessageRole.USER, MessageRole.ASSISTANT]:
                await self._extract_and_store_entities(message, user_id)

                # Extract and store profile facts for user messages
                if message.role == MessageRole.USER and user_id:
                    await self._extract_and_store_profile_facts(message, user_id)
                
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

    async def _extract_and_store_profile_facts(
        self,
        message: ChatMessage,
        user_id: str
    ):
        """Extract and store profile facts from user messages"""
        try:
            content_lower = message.content.lower()

            # Profile fact patterns
            profile_patterns = {
                "name": [
                    r"my name is (\w+)",
                    r"i'm (\w+)",
                    r"i am (\w+)",
                    r"call me (\w+)"
                ],
                "occupation": [
                    r"i work as a (.+)",
                    r"i work as an (.+)",
                    r"i'm a (.+)",
                    r"i am a (.+)",
                    r"my job is (.+)",
                    r"i do (.+) for work"
                ],
                "location": [
                    r"i live in (.+)",
                    r"i'm from (.+)",
                    r"my location is (.+)"
                ],
                "interests": [
                    r"i like (.+)",
                    r"i love (.+)",
                    r"i'm interested in (.+)",
                    r"my interests include (.+)"
                ],
                "preferences": [
                    r"i prefer (.+)",
                    r"my preference is (.+)"
                ]
            }

            import re

            # Extract profile facts
            for fact_type, patterns in profile_patterns.items():
                for pattern in patterns:
                    matches = re.findall(pattern, content_lower)
                    for match in matches:
                        # Clean up the extracted fact
                        fact_value = match.strip()

                        # Remove common stop words and punctuation
                        fact_value = re.sub(r'[.,!?]$', '', fact_value)

                        if len(fact_value) > 2:  # Only store meaningful facts
                            await self.set_profile_fact(
                                user_id=user_id,
                                key=fact_type,
                                value=fact_value,
                                priority=70 if fact_type in ["name", "occupation"] else 50
                            )
                            logger.info(f"Extracted profile fact: {fact_type} = {fact_value}", user_id=user_id)

        except Exception as e:
            logger.error("Failed to extract profile facts", error=str(e))
    
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
    
    def _apply_composite_ranking(
        self,
        results: List[Dict[str, Any]],
        query: str,
        conversation_id: Optional[str],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Apply composite ranking to search results"""
        from datetime import datetime
        import math
        
        now = datetime.now(timezone.utc)
        
        for result in results:
            # Start with semantic score
            semantic_score = float(result.get("score", 0.0))
            
            # Calculate recency boost
            metadata = result.get("metadata", {})
            timestamp = metadata.get("timestamp")
            try:
                if timestamp:
                    # Handle both with and without Z suffix
                    ts_norm = timestamp.replace("Z", "+00:00") if isinstance(timestamp, str) else None
                    age_days = (now - datetime.fromisoformat(ts_norm)).total_seconds() / 86400
                else:
                    age_days = 365.0
            except Exception:
                age_days = 365.0
            
            recency_boost = math.exp(-age_days / 30.0)
            
            # Apply boosts
            boost = 0.0
            
            # Profile/pinned boost
            if metadata.get("memory_level") == "profile" or metadata.get("pinned") is True:
                boost += 0.2
            
            # Conversation match boost
            if metadata.get("conversation_id") == conversation_id:
                boost += 0.1
            
            # Intent/entity boost (if metadata contains relevant entities)
            if "entities" in metadata:
                query_lower = query.lower()
                entities = metadata.get("entities", [])
                if isinstance(entities, str):
                    entities = [e.strip() for e in entities.split(",")]
                if any(entity.lower() in query_lower for entity in entities):
                    boost += 0.1
            
            # Calculate composite score
            composite_score = 0.6 * semantic_score + 0.3 * recency_boost + boost
            result["composite_score"] = composite_score
        
        # Sort by composite score
        ranked = sorted(results, key=lambda x: x.get("composite_score", 0), reverse=True)
        
        # Return top-k
        return ranked[:limit]
    
    def _should_write_to_memory(self, message: ChatMessage) -> bool:
        """Determine if a message should be written to memory (write gates)"""
        content_lower = message.content.lower()
        
        # Always write certain types
        if message.message_type in [MessageType.RESEARCH_RESULT]:
            return True
        
        # Write if explicit memory cues
        memory_cues = [
            "remember", "save", "note", "don't forget",
            "keep in mind", "for future reference"
        ]
        if any(cue in content_lower for cue in memory_cues):
            return True
        
        # Write if contains strong entity/value patterns
        import re
        strong_patterns = [
            r"my .+ is",
            r"i prefer",
            r"i am",
            r"i work",
            r"i live",
            r"my goal",
            r"i want",
            r"i need",
            r"i like",
            r"i love"
        ]
        if any(re.search(pattern, content_lower) for pattern in strong_patterns):
            return True
        
        # Skip only very trivial messages
        trivial_patterns = ["ok", "thanks", "yes", "no", "sure", "got it", "hi", "hello", "bye"]
        if content_lower.strip() in trivial_patterns and len(message.content) < 10:
            return False
        
        # More lenient for user messages - store if > 30 chars (was 20)
        if message.role == MessageRole.USER and len(message.content) > 30:
            return True
        
        # For assistant messages, store if substantial (lowered from 100 to 50)
        return len(message.content) > 50
    
    def _determine_memory_type(self, message: ChatMessage, user_id: Optional[str]) -> str:
        """Determine whether to store in user, conversation, or both"""
        content_lower = message.content.lower()
        
        # Profile/personal info goes to user memory
        if any(phrase in content_lower for phrase in ["i am", "my name", "i prefer", "i work", "i live"]):
            return "user" if user_id else "conversation"
        
        # Research results go to both
        if message.message_type == MessageType.RESEARCH_RESULT:
            return "both"
        
        # User messages with important info should go to user level
        if message.role == MessageRole.USER and len(message.content) > 50:
            # Check for important patterns
            important_patterns = [
                "remember", "don't forget", "keep in mind",
                "i like", "i love", "i hate", "i need",
                "allergic", "can't eat", "avoid"
            ]
            if any(pattern in content_lower for pattern in important_patterns):
                return "both" if user_id else "conversation"
        
        # Assistant messages with substantial content
        if message.role == MessageRole.ASSISTANT and len(message.content) > 200:
            return "both" if user_id else "conversation"
        
        # Default: conversation-specific
        return "conversation"
