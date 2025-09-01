#!/usr/bin/env python3
"""Simple test to verify imports work"""

try:
    print("Testing imports...")
    import asyncio
    print("✓ asyncio")
    
    from app.config import settings
    print("✓ config")
    
    from app.chat_models import ChatRequest, MessageRole
    print("✓ chat_models")
    
    from app.core.memory_manager import MemoryManager
    print("✓ memory_manager")
    
    from app.core.chat_engine import ChatEngine
    print("✓ chat_engine")
    
    print("\n✅ All imports successful!")
    
    # Quick functionality test
    print("\nTesting basic functionality...")
    
    async def quick_test():
        try:
            mm = MemoryManager()
            print("✓ MemoryManager initialized")
            
            ce = ChatEngine()
            print("✓ ChatEngine initialized")
            
            # Test profile fact
            user_id = "test_user_123"
            await mm.set_profile_fact(user_id, "name", "Test User", 90)
            print("✓ Profile fact set")
            
            profile = await mm.get_profile(user_id)
            print(f"✓ Profile retrieved: {len(profile)} facts")
            
            return True
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    result = asyncio.run(quick_test())
    
    if result:
        print("\n🎉 Basic test passed!")
    else:
        print("\n❌ Basic test failed")
        
except Exception as e:
    print(f"❌ Import or test failed: {e}")
    import traceback
    traceback.print_exc()