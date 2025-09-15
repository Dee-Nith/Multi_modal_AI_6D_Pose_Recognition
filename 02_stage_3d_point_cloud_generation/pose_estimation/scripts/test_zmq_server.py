#!/usr/bin/env python3
"""
Test ZeroMQ Server Connection
Simple test to check if CoppeliaSim ZeroMQ server is running
"""

import zmq
import time

def test_zmq_server():
    """Test if ZeroMQ server is running."""
    print("🔍 Testing ZeroMQ server connection...")
    
    try:
        # Create context and socket
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, 3000)  # 3 second timeout
        socket.setsockopt(zmq.SNDTIMEO, 3000)  # 3 second timeout
        
        # Try to connect
        print("🔌 Attempting to connect to localhost:23000...")
        socket.connect("tcp://localhost:23000")
        
        # Test different message formats
        test_messages = [
            "ping",
            "test",
            "sim.getSimulationTime",
            "sim.getSimulationTime()",
            "getSimulationTime",
            "getSimulationTime()"
        ]
        
        for msg in test_messages:
            try:
                print(f"📤 Sending: '{msg}'")
                socket.send_string(msg)
                
                # Try to receive response
                response = socket.recv_string()
                print(f"📥 Received: '{response}'")
                
                if response:
                    print(f"✅ SUCCESS! Server responded to: '{msg}'")
                    socket.close()
                    context.term()
                    return True
                    
            except zmq.error.Again:
                print(f"⏰ Timeout for: '{msg}'")
                continue
            except Exception as e:
                print(f"❌ Error with '{msg}': {e}")
                continue
        
        print("❌ No successful communication with server")
        socket.close()
        context.term()
        return False
        
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

def test_port_availability():
    """Test if port 23000 is open."""
    import socket
    
    print("🔍 Testing port 23000 availability...")
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex(('localhost', 23000))
        sock.close()
        
        if result == 0:
            print("✅ Port 23000 is open and accepting connections")
            return True
        else:
            print("❌ Port 23000 is not accessible")
            return False
            
    except Exception as e:
        print(f"❌ Port test error: {e}")
        return False

def main():
    """Main test function."""
    print("🤖 ZeroMQ Server Connection Test")
    print("=" * 40)
    
    # Test port availability
    if not test_port_availability():
        print("\n💡 SUGGESTIONS:")
        print("   1. Make sure CoppeliaSim is running")
        print("   2. Check if ZeroMQ server is enabled in CoppeliaSim")
        print("   3. Look for 'ZeroMQ Remote API server starting' in CoppeliaSim console")
        return
    
    print("\n" + "=" * 40)
    
    # Test ZeroMQ connection
    if test_zmq_server():
        print("\n🎉 ZeroMQ server is working!")
        print("💡 The server is responding to commands")
    else:
        print("\n❌ ZeroMQ server test failed")
        print("\n💡 TROUBLESHOOTING:")
        print("   1. Check CoppeliaSim console for ZeroMQ errors")
        print("   2. Restart CoppeliaSim")
        print("   3. Check if ZeroMQ add-on is properly installed")
        print("   4. Try different message formats")

if __name__ == "__main__":
    main()




