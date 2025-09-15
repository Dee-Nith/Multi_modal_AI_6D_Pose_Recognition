#!/usr/bin/env python3
"""
Standalone ZeroMQ Test
Tests the ZeroMQ connection without package imports
"""

import zmq
import time

def test_zmq_connection():
    """Test ZeroMQ connection directly."""
    print("🤖 Testing CoppeliaSim ZeroMQ Connection")
    print("=" * 40)

    try:
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, 5000)
        socket.setsockopt(zmq.SNDTIMEO, 5000)
        
        socket.connect("tcp://localhost:23000")
        print("✅ Connected to CoppeliaSim ZeroMQ server at localhost:23000")
        
        test_msg = b"test"
        socket.send(test_msg)
        
        try:
            response = socket.recv()
            print(f"✅ Test successful! Response length: {len(response)}")
            print("🎉 ZeroMQ connection is working!")
            return True
        except zmq.error.Again:
            print("❌ Timeout waiting for response")
            return False
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False
    finally:
        if 'socket' in locals():
            socket.close()
        if 'context' in locals():
            context.term()

def main():
    """Main test function."""
    print("Make sure CoppeliaSim is running with Web Server enabled!")
    print("Tools > Developer > Web Server (port 23000)")
    print()
    
    success = test_zmq_connection()
    
    if success:
        print("\n🎊 ZeroMQ connection test PASSED!")
        print("✅ Ready to proceed with the robotic grasping pipeline!")
    else:
        print("\n💥 ZeroMQ connection test FAILED!")
        print("❌ Check that CoppeliaSim is running and Web Server is enabled.")

if __name__ == "__main__":
    main()







