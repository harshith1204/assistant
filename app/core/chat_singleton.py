"""Singleton ChatEngine instance shared across the app"""

from app.core.chat_engine import ChatEngine

# Global shared chat engine to keep conversations and memory consistent
chat_engine = ChatEngine()

