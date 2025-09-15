#!/usr/bin/env python3
"""
Standalone test - no imports from clients package
"""

import zmq
import time

def test_zmq_connection():
    """Test ZeroMQ connection directly."""
    print("🤖 Testing CoppeliaSim ZeroMQ Connection")
    print("=" * 40)
    
    try:
        # Create ZeroMQ context and socket
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
        socket.setsockopt(zmq.SNDTIMEO, 5000)  # 5 second timeout
        
        # Connect to CoppeliaSim
        socket.connect("tcp://localhost:23000")
        print("✅ Connected to CoppeliaSim ZeroMQ server at localhost:23000")
        
        # Test connection
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

if __name__ == "__main__":
    test_zmq_connection()







