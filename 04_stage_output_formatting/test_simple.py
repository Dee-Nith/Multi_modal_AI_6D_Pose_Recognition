#!/usr/bin/env python3
"""
Simple test using only the ZeroMQ client
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import only the ZeroMQ client directly
from clients.coppeliasim_zmq_client import CoppeliaSimZMQClient

def main():
    """Test CoppeliaSim connection."""
    print("🤖 Testing CoppeliaSim ZeroMQ Connection")
    print("=" * 40)
    
    try:
        with CoppeliaSimZMQClient() as client:
            print("Testing connection...")
            
            if client.test_connection():
                print("✅ Connection test successful!")
                
                # Try to get simulation time
                sim_time = client.get_simulation_time()
                if sim_time is not None:
                    print(f"✅ Simulation time response received (length: {sim_time})")
                else:
                    print("⚠️ Could not get simulation time (may need proper message format)")
                
                print("\n🎉 ZeroMQ connection is working!")
                print("You can now use this client for your robotic grasping project.")
                
            else:
                print("❌ Connection test failed")
                
    except Exception as e:
        print(f"❌ Connection error: {e}")

if __name__ == "__main__":
    main()







