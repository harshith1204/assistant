"""Memory management with Mem0 for context handling"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio
from mem0 import Memory
import structlog
from app.config import settings
from app.chat_models import ConversationContext, ChatMessage, MessageRole

logger = structlog.get_logger()


class MemoryManager:
    """Manages short-term and long-term memory using Mem0"""
    
    def __init__(self):
        """Initialize memory manager with Mem0"""
        # Configure Mem0 with fallback for embedding provider
        try:
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
                "embedder": {
                    "provider": settings.memory_embedder_provider,
                    "config": {
                        "model": settings.memory_embedder_model,
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
            
            self.memory = Memory.from_config(config)
            logger.info("Memory manager initialized with local embeddings", provider=settings.memory_embedder_provider, model=settings.memory_embedder_model)
            
        except Exception as e:
            logger.warning("Failed to initialize with local embeddings, falling back to Groq", error=str(e))
            # Fallback to Groq embeddings
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
                "embedder": {
                    "provider": "groq",
                    "config": {
                        "model": "llama3-70b-8192",
                        "api_key": settings.groq_api_key,
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
            
            self.memory = Memory.from_config(config)
            logger.info("Memory manager initialized with Groq embeddings fallback")
        self.short_term_cache: Dict[str, Dict] = {}  # In-memory cache for short-term
        self.cache_ttl = timedelta(hours=24)  # Short-term memory TTL
        
    async def add_to_memory(
        self,
        conversation_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Add information to memory"""
        try:
            # Prepare metadata
            meta = metadata or {}
            meta.update({
                "conversation_id": conversation_id,
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": user_id or "anonymous"
            })
            
            # Add to Mem0 (long-term)
            result = await asyncio.to_thread(
                self.memory.add,
                content,
                user_id=user_id or conversation_id,
                metadata=meta
            )
            
            # Update short-term cache
            if conversation_id not in self.short_term_cache:
                self.short_term_cache[conversation_id] = {
                    "messages": [],
                    "context": {},
                    "timestamp": datetime.utcnow()
                }
            
            self.short_term_cache[conversation_id]["messages"].append({
                "content": content,
                "metadata": meta,
                "timestamp": datetime.utcnow()
            })
            
            logger.info(
                "Added to memory",
                conversation_id=conversation_id,
                memory_id=result.get("id") if isinstance(result, dict) else None
            )
            
            return {"success": True, "memory_id": result}
            
        except Exception as e:
            logger.error("Failed to add to memory", error=str(e))
            return {"success": False, "error": str(e)}
    
    async def search_memory(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search memories"""
        try:
            # Search in Mem0
            results = await asyncio.to_thread(
                self.memory.search,
                query,
                user_id=user_id or conversation_id,
                limit=limit
            )
            
            # Combine with short-term cache if available
            if conversation_id and conversation_id in self.short_term_cache:
                cache_data = self.short_term_cache[conversation_id]
                # Add recent messages from cache
                recent_messages = cache_data.get("messages", [])[-3:]
                for msg in recent_messages:
                    results.append({
                        "memory": msg["content"],
                        "metadata": msg["metadata"],
                        "score": 0.9,  # High score for recent messages
                        "source": "short_term"
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
        """Get complete context for a conversation"""
        try:
            context = ConversationContext()
            
            # Get short-term context from cache
            if conversation_id in self.short_term_cache:
                cache_data = self.short_term_cache[conversation_id]
                context.short_term = {
                    "recent_messages": cache_data.get("messages", [])[-5:],
                    "session_start": cache_data.get("timestamp", datetime.utcnow()).isoformat(),
                    "message_count": len(cache_data.get("messages", []))
                }
            
            # Get long-term context from Mem0
            all_memories = await asyncio.to_thread(
                self.memory.get_all,
                user_id=user_id or conversation_id
            )
            
            if all_memories:
                # Extract entities and topics
                entities = set()
                topics = set()
                
                for memory in all_memories:
                    memory_text = memory.get("memory", "")
                    meta = memory.get("metadata", {})
                    
                    # Extract entities (simple approach - can be enhanced with NER)
                    if "entities" in meta:
                        entities.update(meta["entities"])
                    
                    # Extract topics
                    if "topics" in meta:
                        topics.update(meta["topics"])
                
                context.long_term = {
                    "total_memories": len(all_memories),
                    "memories": all_memories[:10],  # Recent 10 memories
                    "first_interaction": all_memories[-1].get("metadata", {}).get("timestamp") if all_memories else None
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
                    "extraction_timestamp": datetime.utcnow().isoformat()
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
            current_time = datetime.utcnow()
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
    
    async def get_memory_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get memory statistics"""
        try:
            all_memories = await asyncio.to_thread(
                self.memory.get_all,
                user_id=user_id or "global"
            )
            
            return {
                "total_memories": len(all_memories),
                "short_term_conversations": len(self.short_term_cache),
                "cache_size": sum(
                    len(cache.get("messages", []))
                    for cache in self.short_term_cache.values()
                ),
                "oldest_memory": all_memories[-1].get("metadata", {}).get("timestamp") if all_memories else None,
                "newest_memory": all_memories[0].get("metadata", {}).get("timestamp") if all_memories else None
            }
            
        except Exception as e:
            logger.error("Failed to get memory stats", error=str(e))
            return {}
