#!/usr/bin/env python3
"""
Direct import test to bypass __init__.py issues
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_direct_import():
    """Test importing the ZeroMQ client directly."""
    print("🧪 Testing direct import...")
    
    try:
        # Import directly from the file, bypassing __init__.py
        import src.clients.coppeliasim_zmq_client
        from src.clients.coppeliasim_zmq_client import CoppeliaSimZMQClient
        
        print("✅ Direct import successful!")
        
        # Test creating instance
        client = CoppeliaSimZMQClient()
        print("✅ ZeroMQ client instance created!")
        
        return True
        
    except Exception as e:
        print(f"❌ Direct import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_connection():
    """Test connection to CoppeliaSim."""
    print("\n🔌 Testing connection...")
    
    try:
        from src.clients.coppeliasim_zmq_client import CoppeliaSimZMQClient
        
        client = CoppeliaSimZMQClient()
        success = client.connect()
        
        if success:
            print("✅ Connection successful!")
            client.disconnect()
            return True
        else:
            print("❌ Connection failed!")
            return False
            
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🤖 Direct Import and Connection Test")
    print("=" * 40)
    
    # Test direct import
    if not test_direct_import():
        return
    
    # Test connection
    if not test_connection():
        print("\n⚠️ Connection failed - make sure CoppeliaSim is running!")
        return
    
    print("\n🎉 All tests passed!")

if __name__ == "__main__":
    main()







