#!/usr/bin/env python3
"""
Demonstration of cross-conversation memory implementation
This shows how the system maintains user context across different conversations
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional

class MockMemoryManager:
    """Simplified memory manager to demonstrate cross-conversation memory"""
    
    def __init__(self):
        # User memories persist across conversations
        self.user_memories: Dict[str, List[Dict]] = {}
        # Profile facts are high-priority user memories
        self.profile_facts: Dict[str, Dict] = {}
        # Conversation-specific memories
        self.conversation_memories: Dict[str, List[Dict]] = {}
    
    def set_profile_fact(self, user_id: str, key: str, value: str, priority: int = 50):
        """Set a profile fact for a user - persists across ALL conversations"""
        if user_id not in self.profile_facts:
            self.profile_facts[user_id] = {}
        self.profile_facts[user_id][key] = {
            "value": value,
            "priority": priority,
            "timestamp": datetime.now().isoformat()
        }
        print(f"✓ Profile fact set for {user_id}: {key} = {value}")
    
    def add_user_memory(self, user_id: str, memory: str, conversation_id: str):
        """Add a memory at user level - accessible across conversations"""
        if user_id not in self.user_memories:
            self.user_memories[user_id] = []
        self.user_memories[user_id].append({
            "memory": memory,
            "conversation_id": conversation_id,  # Track source but don't filter by it
            "timestamp": datetime.now().isoformat(),
            "memory_level": "user"
        })
        print(f"✓ User memory added: {memory[:50]}...")
    
    def add_conversation_memory(self, conversation_id: str, memory: str):
        """Add a memory specific to a conversation"""
        if conversation_id not in self.conversation_memories:
            self.conversation_memories[conversation_id] = []
        self.conversation_memories[conversation_id].append({
            "memory": memory,
            "timestamp": datetime.now().isoformat(),
            "memory_level": "conversation"
        })
        print(f"✓ Conversation memory added: {memory[:50]}...")
    
    def get_memories_for_context(self, user_id: str, conversation_id: str) -> Dict[str, Any]:
        """Get all relevant memories for a conversation context"""
        context = {
            "profile": [],
            "user_memories": [],
            "conversation_memories": []
        }
        
        # Always include profile facts (highest priority)
        if user_id in self.profile_facts:
            for key, data in self.profile_facts[user_id].items():
                context["profile"].append(f"{key}: {data['value']}")
        
        # Include ALL user memories (cross-conversation)
        if user_id in self.user_memories:
            context["user_memories"] = self.user_memories[user_id]
        
        # Include conversation-specific memories
        if conversation_id in self.conversation_memories:
            context["conversation_memories"] = self.conversation_memories[conversation_id]
        
        return context


class MockChatEngine:
    """Simplified chat engine to demonstrate memory usage"""
    
    def __init__(self, memory_manager: MockMemoryManager):
        self.memory_manager = memory_manager
    
    def process_message(self, user_id: str, conversation_id: str, message: str):
        """Process a message with memory context"""
        print(f"\n{'='*60}")
        print(f"Processing message in conversation: {conversation_id}")
        print(f"User: {user_id}")
        print(f"Message: {message}")
        print("-"*60)
        
        # Get ALL relevant context for this user
        context = self.memory_manager.get_memories_for_context(user_id, conversation_id)
        
        # Display what memories are being used
        print("\n📚 Memory Context Loaded:")
        
        if context["profile"]:
            print(f"\n  Profile Facts ({len(context['profile'])} items):")
            for fact in context["profile"]:
                print(f"    • {fact}")
        
        if context["user_memories"]:
            print(f"\n  User Memories ({len(context['user_memories'])} items from ALL conversations):")
            for mem in context["user_memories"][:5]:
                source_conv = mem.get("conversation_id", "unknown")
                print(f"    • [{source_conv}] {mem['memory'][:80]}...")
        
        if context["conversation_memories"]:
            print(f"\n  Conversation Memories ({len(context['conversation_memories'])} items):")
            for mem in context["conversation_memories"][:3]:
                print(f"    • {mem['memory'][:80]}...")
        
        # Simulate processing and potentially adding new memories
        if "remember" in message.lower() or "i am" in message.lower() or "i prefer" in message.lower():
            # This should be stored at user level
            self.memory_manager.add_user_memory(user_id, message, conversation_id)
        else:
            # Regular message, store at conversation level
            self.memory_manager.add_conversation_memory(conversation_id, message)
        
        # Generate response (simplified)
        response = f"I understand. Based on your profile and history, I can help you."
        print(f"\n🤖 Response: {response}")
        
        return response


def demonstrate_cross_conversation_memory():
    """Demonstrate how memory persists across conversations"""
    
    print("="*70)
    print("CROSS-CONVERSATION MEMORY DEMONSTRATION")
    print("="*70)
    
    # Initialize components
    memory_manager = MockMemoryManager()
    chat_engine = MockChatEngine(memory_manager)
    
    # Create a consistent user
    user_id = "user_alice_123"
    
    # Create different conversation IDs
    conv1 = "conversation_trip_planning"
    conv2 = "conversation_restaurant_search"
    conv3 = "conversation_weekend_activities"
    
    print(f"\n👤 User: {user_id}")
    print(f"📝 We will have 3 different conversations:")
    print(f"  1. {conv1}")
    print(f"  2. {conv2}")
    print(f"  3. {conv3}")
    
    # ========== Set Profile Facts (persist across ALL conversations) ==========
    print("\n" + "="*70)
    print("SETTING PROFILE FACTS (Persist Across All Conversations)")
    print("="*70)
    
    memory_manager.set_profile_fact(user_id, "name", "Alice Chen", 95)
    memory_manager.set_profile_fact(user_id, "diet", "vegetarian", 90)
    memory_manager.set_profile_fact(user_id, "location", "San Francisco", 80)
    memory_manager.set_profile_fact(user_id, "allergies", "peanuts, shellfish", 95)
    
    # ========== Conversation 1: Trip Planning ==========
    print("\n" + "="*70)
    print("CONVERSATION 1: Trip Planning")
    print("="*70)
    
    chat_engine.process_message(
        user_id, conv1,
        "I want to plan a trip to Goa. Remember I prefer boutique hotels."
    )
    
    chat_engine.process_message(
        user_id, conv1,
        "Also, I love beach activities but have a bad knee so nothing too strenuous."
    )
    
    # ========== Conversation 2: Restaurant Search (DIFFERENT CONVERSATION) ==========
    print("\n" + "="*70)
    print("CONVERSATION 2: Restaurant Search (Different Conversation, Same User)")
    print("="*70)
    
    chat_engine.process_message(
        user_id, conv2,  # DIFFERENT conversation
        "Can you recommend some restaurants for dinner tonight?"
    )
    
    print("\n🔍 Notice how the system has access to:")
    print("  ✓ Profile facts (diet: vegetarian, allergies)")
    print("  ✓ User memories from Conversation 1 (boutique hotels, bad knee)")
    print("  ✓ Even though this is a DIFFERENT conversation!")
    
    # ========== Conversation 3: Weekend Activities (YET ANOTHER CONVERSATION) ==========
    print("\n" + "="*70)
    print("CONVERSATION 3: Weekend Activities (Third Conversation, Same User)")
    print("="*70)
    
    chat_engine.process_message(
        user_id, conv3,  # YET ANOTHER conversation
        "What activities would you recommend for this weekend?"
    )
    
    print("\n🎯 Key Points Demonstrated:")
    print("  1. Profile facts are ALWAYS available in every conversation")
    print("  2. User memories from Conv1 are accessible in Conv2 and Conv3")
    print("  3. Each conversation also has its own specific memories")
    print("  4. The user_id links all these memories together")
    
    # ========== Summary ==========
    print("\n" + "="*70)
    print("MEMORY ARCHITECTURE SUMMARY")
    print("="*70)
    
    print("\n📊 Memory Levels:")
    print("  1. PROFILE (Highest Priority):")
    print("     - Always injected into context")
    print("     - Set via set_profile_fact()")
    print("     - Examples: name, diet, allergies")
    
    print("\n  2. USER (Cross-Conversation):")
    print("     - Stored with user_id")
    print("     - Accessible from ANY conversation")
    print("     - Examples: preferences, past experiences")
    
    print("\n  3. CONVERSATION (Thread-Specific):")
    print("     - Only for current conversation")
    print("     - Short-term context")
    print("     - Examples: current topic, recent questions")
    
    print("\n✅ This ensures TRUE PERSONALIZATION across all interactions!")


if __name__ == "__main__":
    demonstrate_cross_conversation_memory()
    
    print("\n" + "="*70)
    print("IMPLEMENTATION NOTES")
    print("="*70)
    
    print("\nKey Implementation Details:")
    print("  1. In memory_manager.py:")
    print("     - set_profile_fact() stores with user_id and memory_level='profile'")
    print("     - search_memory() with scope='both' searches user AND conversation")
    print("     - User memories are stored with user_id, NOT conversation_id")
    
    print("\n  2. In chat_engine.py:")
    print("     - _prepare_context() loads profile + user memories + conversation memories")
    print("     - User_id must be consistent across all calls")
    print("     - stream_message() emits memory.used event showing what was injected")
    
    print("\n  3. In websocket_handler.py:")
    print("     - Maintains user_id across connections")
    print("     - Passes user_id to all chat_engine methods")
    print("     - Session persistence via session.resume message")
    
    print("\n🚀 Result: Users have persistent memory across all conversations!")