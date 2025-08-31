#!/usr/bin/env python3
"""Test script for conversational AI system"""

import asyncio
import json
from app.core.conversational_intent import (
    ConversationalIntentDetector, 
    ConversationalRouter,
    ConversationalFlowManager
)
from app.core.enhanced_chat_engine import EnhancedChatEngine
from app.chat_models import ChatRequest

async def test_intent_detection():
    """Test intent detection"""
    detector = ConversationalIntentDetector()
    
    test_messages = [
        "Hi there, how are you?",
        "I need to research market trends for SaaS companies in Europe",
        "Create a note in CRM about our meeting with John from Acme Corp",
        "Add a task to the project for implementing the new feature",
        "Schedule a meeting with the team for tomorrow at 2pm",
        "Generate a report of our Q3 performance",
        "What's the weather like?",
        "Find information about competitor pricing",
        "Update the lead status to qualified",
        "Create a documentation page for the API"
    ]
    
    print("=" * 60)
    print("INTENT DETECTION TEST")
    print("=" * 60)
    
    for message in test_messages:
        result = await detector.detect_intent(message)
        print(f"\nMessage: {message}")
        print(f"Intent: {result.get('intent')} (confidence: {result.get('confidence', 0):.2f})")
        if result.get('action'):
            print(f"Action: {result['action']}")
        if result.get('entities'):
            print(f"Entities: {json.dumps(result['entities'], indent=2)}")
        if result.get('needs_clarification'):
            print("⚠️  Needs clarification")
        print("-" * 40)

async def test_conversational_flow():
    """Test conversational flow"""
    flow_manager = ConversationalFlowManager()
    
    test_conversations = [
        {
            "id": "conv1",
            "messages": [
                "Research the best practices for customer onboarding in SaaS",
                "Focus on B2B companies with enterprise clients",
                "Include pricing models and retention strategies"
            ]
        },
        {
            "id": "conv2", 
            "messages": [
                "Create a task for following up with the lead from yesterday",
                "Set it as high priority",
                "Assign it to Sarah"
            ]
        }
    ]
    
    print("\n" + "=" * 60)
    print("CONVERSATIONAL FLOW TEST")
    print("=" * 60)
    
    for conv in test_conversations:
        print(f"\nConversation: {conv['id']}")
        print("-" * 40)
        
        for message in conv['messages']:
            result = await flow_manager.process_conversation_turn(
                message,
                conv['id'],
                user_id="test_user"
            )
            
            print(f"\nUser: {message}")
            print(f"Intent: {result['routing']['intent']}")
            print(f"Handler: {result['routing']['handler']}")
            print(f"Strategy: {result['strategy']['type']}")
            
            if result['routing'].get('parameters'):
                print(f"Parameters: {json.dumps(result['routing']['parameters'], indent=2)}")

async def test_enhanced_engine():
    """Test enhanced chat engine"""
    engine = EnhancedChatEngine()
    
    print("\n" + "=" * 60)
    print("ENHANCED ENGINE TEST")
    print("=" * 60)
    
    request = ChatRequest(
        message="I need to analyze our competitor landscape in the project management space",
        conversation_id="test_conv",
        use_web_search=False  # Disable for testing
    )
    
    print(f"\nProcessing: {request.message}")
    print("-" * 40)
    
    updates = []
    async for update in engine.process_conversational_message(request, "test_user"):
        updates.append(update)
        print(f"\nUpdate Type: {update['type']}")
        if update.get('content'):
            print(f"Content: {update['content'][:100]}...")
        if update.get('intent'):
            print(f"Intent: {update['intent']}")
        if update.get('confidence'):
            print(f"Confidence: {update['confidence']:.2f}")
    
    print(f"\nTotal updates received: {len(updates)}")

async def main():
    """Run all tests"""
    print("\n🚀 CONVERSATIONAL AI SYSTEM TEST\n")
    
    try:
        # Test intent detection
        await test_intent_detection()
        
        # Test conversational flow
        await test_conversational_flow()
        
        # Test enhanced engine
        await test_enhanced_engine()
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())