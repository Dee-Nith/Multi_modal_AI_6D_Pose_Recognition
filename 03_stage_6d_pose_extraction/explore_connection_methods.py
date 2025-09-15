#!/usr/bin/env python3
"""
Explore All Connection Methods to CoppeliaSim
Test different ways to connect and communicate with CoppeliaSim
"""

import requests
import socket
import time
import json
import subprocess
import os

def test_http_connection():
    """Test HTTP connection to CoppeliaSim."""
    print("🌐 Testing HTTP Connection...")
    
    try:
        # Try different HTTP endpoints
        endpoints = [
            "http://localhost:23000",
            "http://localhost:23000/api",
            "http://localhost:23000/status",
            "http://localhost:23000/sim",
            "http://localhost:23000/vrep",
            "http://localhost:23000/coppelia"
        ]
        
        for endpoint in endpoints:
            try:
                print(f"   Testing: {endpoint}")
                response = requests.get(endpoint, timeout=3)
                print(f"   ✅ Status: {response.status_code}")
                print(f"   📄 Response: {response.text[:100]}...")
                return True
            except requests.exceptions.RequestException as e:
                print(f"   ❌ Failed: {e}")
                continue
        
        return False
        
    except Exception as e:
        print(f"❌ HTTP test error: {e}")
        return False

def test_websocket_connection():
    """Test WebSocket connection to CoppeliaSim."""
    print("🔌 Testing WebSocket Connection...")
    
    try:
        import websocket
        
        # Try to connect to WebSocket
        ws_url = "ws://localhost:23000"
        
        def on_message(ws, message):
            print(f"📥 WebSocket message: {message}")
        
        def on_error(ws, error):
            print(f"❌ WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            print("🔌 WebSocket connection closed")
        
        def on_open(ws):
            print("✅ WebSocket connection opened")
            ws.send("ping")
        
        # Create WebSocket connection
        ws = websocket.WebSocketApp(ws_url,
                                   on_open=on_open,
                                   on_message=on_message,
                                   on_error=on_error,
                                   on_close=on_close)
        
        # Run for a short time
        ws.run_forever(timeout=5)
        return True
        
    except ImportError:
        print("   ⚠️ websocket-client not installed")
        return False
    except Exception as e:
        print(f"   ❌ WebSocket error: {e}")
        return False

def test_tcp_connection():
    """Test direct TCP connection to CoppeliaSim."""
    print("🔌 Testing TCP Connection...")
    
    try:
        # Try different ports
        ports = [23000, 23001, 23002, 19997, 19998, 19999]
        
        for port in ports:
            try:
                print(f"   Testing port {port}...")
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(3)
                result = sock.connect_ex(('localhost', port))
                sock.close()
                
                if result == 0:
                    print(f"   ✅ Port {port} is open")
                    
                    # Try to send a simple message
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(3)
                    sock.connect(('localhost', port))
                    sock.send(b"ping")
                    
                    try:
                        response = sock.recv(1024)
                        print(f"   📥 Response: {response}")
                    except:
                        print("   ⏰ No response")
                    
                    sock.close()
                    return True
                else:
                    print(f"   ❌ Port {port} is closed")
                    
            except Exception as e:
                print(f"   ❌ Port {port} error: {e}")
                continue
        
        return False
        
    except Exception as e:
        print(f"❌ TCP test error: {e}")
        return False

def test_shared_memory():
    """Test shared memory communication with CoppeliaSim."""
    print("💾 Testing Shared Memory...")
    
    try:
        # Check if shared memory files exist
        shared_memory_paths = [
            "/tmp/coppelia_shared",
            "/tmp/vrep_shared",
            "/dev/shm/coppelia",
            "/dev/shm/vrep"
        ]
        
        for path in shared_memory_paths:
            if os.path.exists(path):
                print(f"   ✅ Found shared memory: {path}")
                return True
        
        print("   ❌ No shared memory found")
        return False
        
    except Exception as e:
        print(f"❌ Shared memory test error: {e}")
        return False

def test_file_based_communication():
    """Test file-based communication with CoppeliaSim."""
    print("📁 Testing File-Based Communication...")
    
    try:
        # Create a test file
        test_file = "coppelia_test_command.txt"
        with open(test_file, "w") as f:
            f.write("getSimulationTime")
        
        print(f"   📝 Created test command file: {test_file}")
        
        # Wait for response
        time.sleep(2)
        
        # Check for response file
        response_file = "coppelia_test_response.txt"
        if os.path.exists(response_file):
            with open(response_file, "r") as f:
                response = f.read()
            print(f"   📥 Response: {response}")
            return True
        else:
            print("   ❌ No response file found")
            return False
            
    except Exception as e:
        print(f"❌ File-based test error: {e}")
        return False

def test_ros_connection():
    """Test ROS connection to CoppeliaSim."""
    print("🤖 Testing ROS Connection...")
    
    try:
        # Check if ROS is available
        result = subprocess.run(["rosnode", "list"], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            print("   ✅ ROS is available")
            nodes = result.stdout.strip().split('\n')
            print(f"   📋 ROS nodes: {nodes}")
            
            # Check for CoppeliaSim ROS nodes
            coppelia_nodes = [node for node in nodes if 'coppelia' in node.lower() or 'vrep' in node.lower()]
            if coppelia_nodes:
                print(f"   ✅ Found CoppeliaSim ROS nodes: {coppelia_nodes}")
                return True
            else:
                print("   ❌ No CoppeliaSim ROS nodes found")
                return False
        else:
            print("   ❌ ROS not available")
            return False
            
    except Exception as e:
        print(f"   ❌ ROS test error: {e}")
        return False

def test_python_api():
    """Test Python API connection to CoppeliaSim."""
    print("🐍 Testing Python API...")
    
    try:
        # Try to import and use the traditional CoppeliaSim API
        import sys
        sys.path.append('/Applications/coppeliaSim.app/Contents/Resources/programming/remoteApiBindings/python/python')
        
        import sim
        import simConst
        
        print("   ✅ CoppeliaSim API imported")
        
        # Try to connect
        client_id = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
        
        if client_id != -1:
            print(f"   ✅ Connected with client ID: {client_id}")
            
            # Test a simple command
            result, time = sim.simxGetSimulationTime(client_id, sim.simx_opmode_blocking)
            if result == sim.simx_return_ok:
                print(f"   ✅ Simulation time: {time}")
                sim.simxFinish(client_id)
                return True
            else:
                print(f"   ❌ Command failed: {result}")
                sim.simxFinish(client_id)
                return False
        else:
            print("   ❌ Failed to connect")
            return False
            
    except ImportError as e:
        print(f"   ❌ API import error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ API test error: {e}")
        return False

def test_zeromq_alternatives():
    """Test alternative ZeroMQ approaches."""
    print("🔌 Testing ZeroMQ Alternatives...")
    
    try:
        import zmq
        
        # Test different ZeroMQ patterns
        patterns = [
            ("REQ-REP", zmq.REQ),
            ("PUB-SUB", zmq.SUB),
            ("PUSH-PULL", zmq.PUSH)
        ]
        
        for pattern_name, socket_type in patterns:
            try:
                print(f"   Testing {pattern_name} pattern...")
                
                context = zmq.Context()
                socket = context.socket(socket_type)
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.RCVTIMEO, 2000)
                socket.setsockopt(zmq.SNDTIMEO, 2000)
                
                # Try different connection strings
                connections = [
                    "tcp://localhost:23000",
                    "tcp://127.0.0.1:23000",
                    "ipc:///tmp/coppelia",
                    "inproc://coppelia"
                ]
                
                for conn_str in connections:
                    try:
                        socket.connect(conn_str)
                        print(f"     ✅ Connected to {conn_str}")
                        
                        # Send test message
                        socket.send(b"test")
                        
                        try:
                            response = socket.recv()
                            print(f"     📥 Response: {response}")
                            socket.close()
                            context.term()
                            return True
                        except zmq.error.Again:
                            print(f"     ⏰ No response from {conn_str}")
                            continue
                            
                    except Exception as e:
                        print(f"     ❌ {conn_str} failed: {e}")
                        continue
                
                socket.close()
                context.term()
                
            except Exception as e:
                print(f"   ❌ {pattern_name} error: {e}")
                continue
        
        return False
        
    except ImportError:
        print("   ❌ ZeroMQ not available")
        return False
    except Exception as e:
        print(f"❌ ZeroMQ alternatives error: {e}")
        return False

def main():
    """Test all connection methods."""
    print("🔍 Exploring All Connection Methods to CoppeliaSim")
    print("=" * 60)
    
    methods = [
        ("HTTP", test_http_connection),
        ("WebSocket", test_websocket_connection),
        ("TCP", test_tcp_connection),
        ("Shared Memory", test_shared_memory),
        ("File-Based", test_file_based_communication),
        ("ROS", test_ros_connection),
        ("Python API", test_python_api),
        ("ZeroMQ Alternatives", test_zeromq_alternatives)
    ]
    
    working_methods = []
    
    for method_name, test_func in methods:
        print(f"\n{'='*20} {method_name} {'='*20}")
        try:
            if test_func():
                working_methods.append(method_name)
                print(f"✅ {method_name} - WORKING")
            else:
                print(f"❌ {method_name} - FAILED")
        except Exception as e:
            print(f"❌ {method_name} - ERROR: {e}")
    
    print(f"\n{'='*60}")
    print("📊 CONNECTION TEST RESULTS")
    print("=" * 60)
    
    if working_methods:
        print(f"✅ Working methods ({len(working_methods)}):")
        for method in working_methods:
            print(f"   - {method}")
    else:
        print("❌ No working connection methods found")
    
    print(f"\n💡 RECOMMENDATIONS:")
    if "HTTP" in working_methods:
        print("   - Use HTTP API for simple commands")
    if "WebSocket" in working_methods:
        print("   - Use WebSocket for real-time communication")
    if "Python API" in working_methods:
        print("   - Use Python API for full functionality")
    if "File-Based" in working_methods:
        print("   - Use file-based communication as fallback")
    
    if not working_methods:
        print("   - Check if CoppeliaSim is running")
        print("   - Verify network settings")
        print("   - Try restarting CoppeliaSim")
        print("   - Check CoppeliaSim console for errors")

if __name__ == "__main__":
    main()




