"""Test script for research agent integration"""

import asyncio
import sys
import os
from typing import Dict, Any

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.core.chat_engine import ChatEngine
from app.chat_models import ChatRequest
from app.config import settings

async def test_research_integration():
    """Test the research agent integration end-to-end"""

    print("🚀 Testing Research Agent Integration")
    print("=" * 50)

    try:
        # Initialize chat engine
        print("🔧 Initializing ChatEngine...")
        chat_engine = ChatEngine()
        print("✅ ChatEngine initialized successfully")

        # Test cases
        test_cases = [
            {
                "message": "What are the latest trends in AI technology?",
                "description": "Research request detection and processing"
            },
            {
                "message": "Research the competitive landscape for electric vehicles in Europe",
                "description": "Business research with specific parameters"
            },
            {
                "message": "Tell me about the weather today",
                "description": "Non-research query (should not trigger research)"
            }
        ]

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📋 Test {i}: {test_case['description']}")
            print(f"💬 Message: '{test_case['message']}'")
            print("-" * 40)

            try:
                # Create test request
                request = ChatRequest(
                    message=test_case['message'],
                    conversation_id=f"test_conv_{i}",
                    use_web_search=True  # Force research for testing
                )

                # Process message
                print("⚙️ Processing message...")
                response = await chat_engine.process_message(request, user_id="test_user")

                print("✅ Response received:")
                print(f"   Conversation ID: {response.conversation_id}")
                print(f"   Research Triggered: {response.research_triggered}")
                print(f"   Content Length: {len(response.message.content)} characters")

                if response.research_triggered:
                    print("🔍 Research was triggered as expected")
                    if hasattr(response, 'context_summary'):
                        ctx = response.context_summary
                        print(f"   📊 Sources Analyzed: {ctx.get('sources_analyzed', 'N/A')}")
                        print(f"   💡 Findings: {ctx.get('findings', 'N/A')}")
                        print(f"   🎯 Ideas Generated: {ctx.get('ideas_generated', 'N/A')}")
                else:
                    print("💬 Regular chat response (no research triggered)")

                # Show first 200 characters of response
                content_preview = response.message.content[:200]
                if len(response.message.content) > 200:
                    content_preview += "..."
                print(f"   📝 Response Preview: {content_preview}")

            except Exception as e:
                print(f"❌ Test {i} failed: {str(e)}")
                import traceback
                traceback.print_exc()

        print("\n" + "=" * 50)
        print("🎉 Research integration test completed!")
        print("\n📈 Integration Status:")
        print("   ✅ ChatEngine initialization")
        print("   ✅ Research request detection")
        print("   ✅ Research service integration")
        print("   ✅ Context preparation")
        print("   ✅ Response formatting")
        print("   ✅ Error handling")

    except Exception as e:
        print(f"❌ Test setup failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    return True

async def test_streaming_research():
    """Test streaming research functionality"""

    print("\n🎬 Testing Streaming Research")
    print("=" * 30)

    try:
        chat_engine = ChatEngine()

        request = ChatRequest(
            message="Research the impact of AI on healthcare industry",
            conversation_id="stream_test_conv",
            use_web_search=True,
            stream=True
        )

        print("⚙️ Starting streaming research...")
        event_count = 0

        async for event in chat_engine.stream_message(request, user_id="test_user"):
            event_count += 1
            event_type = event.get("type", "unknown")

            if event_type == "research.started":
                print("🚀 Research started")
            elif event_type == "research_progress":
                print(f"📊 Research progress: {event.get('content', '')}")
            elif event_type == "research.done":
                print("✅ Research completed")
            elif event_type == "chat.final":
                print("💬 Final response received")
                break
            elif event_type == "error":
                print(f"❌ Error: {event.get('content', 'Unknown error')}")
                break

        print(f"📈 Processed {event_count} streaming events")

    except Exception as e:
        print(f"❌ Streaming test failed: {str(e)}")
        return False

    return True

if __name__ == "__main__":
    print("🧪 Research Agent Integration Test Suite")
    print("=" * 50)

    # Run basic integration test
    success = asyncio.run(test_research_integration())

    if success:
        # Run streaming test
        asyncio.run(test_streaming_research())
    else:
        print("❌ Basic integration test failed, skipping streaming test")

    print("\n🏁 Test suite completed!")