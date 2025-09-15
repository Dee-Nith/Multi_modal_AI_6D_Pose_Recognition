#!/usr/bin/env python3
"""
Simple test to isolate import issues
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test importing the ZeroMQ client."""
    print("🧪 Testing imports...")
    
    try:
        from src.clients.coppeliasim_zmq_client import CoppeliaSimZMQClient
        print("✅ ZeroMQ client imported successfully!")
        
        # Test creating instance
        client = CoppeliaSimZMQClient()
        print("✅ ZeroMQ client instance created!")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
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
    print("🤖 Simple Import and Connection Test")
    print("=" * 40)
    
    # Test imports
    if not test_imports():
        return
    
    # Test connection
    if not test_connection():
        print("\n⚠️ Connection failed - make sure CoppeliaSim is running!")
        return
    
    print("\n🎉 All tests passed!")

if __name__ == "__main__":
    main()







