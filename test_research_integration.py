#!/usr/bin/env python3
"""Test script for research integration"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_research_imports():
    """Test that all research imports work"""
    try:
        from app.research import ResearchService, run_research, stream_research
        from app.research import BusinessResearchBrief, BusinessResearchType
        from app.research.service import ResearchService as DirectService

        print("✅ Research module imports successful!")
        print("📦 ResearchService:", ResearchService)
        print("🔍 run_research function:", run_research)
        print("📊 BusinessResearchBrief:", BusinessResearchBrief)
        print("🎯 BusinessResearchType:", BusinessResearchType)

        return True
    except ImportError as e:
        print("❌ Import failed:", str(e))
        return False

def test_chat_engine_integration():
    """Test that chat engine uses new research service"""
    try:
        from app.core.chat_engine import ChatEngine

        # Check that ChatEngine has research_service attribute
        engine = ChatEngine()
        if hasattr(engine, 'research_service'):
            print("✅ ChatEngine has research_service")
            print("🔄 Old research_engine:", hasattr(engine, 'research_engine'))
            return True
        else:
            print("❌ ChatEngine missing research_service")
            return False
    except Exception as e:
        print("❌ ChatEngine test failed:", str(e))
        return False

if __name__ == "__main__":
    print("🧪 Testing Research Integration")
    print("=" * 40)

    success = True

    print("\n1. Testing Research Module Imports...")
    success &= test_research_imports()

    print("\n2. Testing Chat Engine Integration...")
    success &= test_chat_engine_integration()

    print("\n" + "=" * 40)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("🚀 Research integration is working correctly")
    else:
        print("❌ SOME TESTS FAILED!")
        print("🔧 Please check the integration setup")
