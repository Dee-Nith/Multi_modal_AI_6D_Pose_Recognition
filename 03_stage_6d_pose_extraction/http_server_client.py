#!/usr/bin/env python3
"""
HTTP Server Client for CoppeliaSim
Creates an HTTP server that CoppeliaSim can connect to
"""

import http.server
import socketserver
import json
import threading
import time
import os
from urllib.parse import urlparse, parse_qs

class CoppeliaSimHTTPHandler(http.server.BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.camera_data = None
        self.last_command = None
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests from CoppeliaSim."""
        try:
            parsed_url = urlparse(self.path)
            path = parsed_url.path
            query_params = parse_qs(parsed_url.query)
            
            print(f"📥 GET request: {path}")
            
            if path == "/status":
                # Return server status
                response = {
                    "status": "running",
                    "timestamp": time.time(),
                    "camera_data_available": self.camera_data is not None
                }
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
                
            elif path == "/camera":
                # Return camera data
                if self.camera_data:
                    self.send_response(200)
                    self.send_header('Content-type', 'application/octet-stream')
                    self.end_headers()
                    self.wfile.write(self.camera_data)
                else:
                    self.send_response(404)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "No camera data available"}).encode())
                    
            elif path == "/command":
                # Return last command
                if self.last_command:
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"command": self.last_command}).encode())
                else:
                    self.send_response(404)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "No command available"}).encode())
                    
            else:
                # Return 404 for unknown paths
                self.send_response(404)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Not found"}).encode())
                
        except Exception as e:
            print(f"❌ GET request error: {e}")
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
    
    def do_POST(self):
        """Handle POST requests from CoppeliaSim."""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            parsed_url = urlparse(self.path)
            path = parsed_url.path
            
            print(f"📥 POST request: {path}")
            
            if path == "/camera":
                # Receive camera data
                self.camera_data = post_data
                print(f"✅ Received camera data: {len(post_data)} bytes")
                
                response = {"status": "success", "bytes_received": len(post_data)}
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
                
            elif path == "/command":
                # Receive command
                try:
                    command_data = json.loads(post_data.decode('utf-8'))
                    self.last_command = command_data
                    print(f"✅ Received command: {command_data}")
                    
                    response = {"status": "success", "command_received": True}
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(response).encode())
                except json.JSONDecodeError:
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode())
                    
            else:
                # Return 404 for unknown paths
                self.send_response(404)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Not found"}).encode())
                
        except Exception as e:
            print(f"❌ POST request error: {e}")
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

class CoppeliaSimHTTPServer:
    def __init__(self, host='localhost', port=8080):
        self.host = host
        self.port = port
        self.server = None
        self.server_thread = None
        self.running = False
        
    def start_server(self):
        """Start the HTTP server in a separate thread."""
        try:
            # Create server
            handler = CoppeliaSimHTTPHandler
            self.server = socketserver.TCPServer((self.host, self.port), handler)
            
            # Start server in a separate thread
            self.server_thread = threading.Thread(target=self.server.serve_forever)
            self.server_thread.daemon = True
            self.server_thread.start()
            
            self.running = True
            print(f"✅ HTTP server started on {self.host}:{self.port}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start HTTP server: {e}")
            return False
    
    def stop_server(self):
        """Stop the HTTP server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.running = False
            print("🛑 HTTP server stopped")
    
    def get_camera_data(self):
        """Get camera data from the server."""
        try:
            import requests
            response = requests.get(f"http://{self.host}:{self.port}/camera", timeout=5)
            
            if response.status_code == 200:
                return response.content
            else:
                print(f"❌ Failed to get camera data: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error getting camera data: {e}")
            return None
    
    def send_command(self, command):
        """Send command to CoppeliaSim."""
        try:
            import requests
            data = {"command": command, "timestamp": time.time()}
            response = requests.post(f"http://{self.host}:{self.port}/command", 
                                   json=data, timeout=5)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Failed to send command: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error sending command: {e}")
            return None

def create_coppelia_lua_script():
    """Create a Lua script for CoppeliaSim to communicate with our HTTP server."""
    lua_script = '''
-- CoppeliaSim HTTP Client Script
-- Attach this to the sphericalVisionRGBAndDepth object

local http = require("socket.http")
local ltn12 = require("ltn12")

function sysCall_init()
    print("🌐 HTTP client initialized")
    server_url = "http://localhost:8080"
    rgbSensor = sim.getObject("./sensorRGB")
    depthSensor = sim.getObject("./sensorDepth")
end

function sysCall_sensing()
    -- Capture camera data every 10 simulation steps
    if sim.getSimulationTime() % 0.1 < 0.01 then
        if rgbSensor ~= -1 then
            local rgbImage = sim.getVisionSensorImg(rgbSensor)
            if rgbImage then
                -- Send camera data to HTTP server
                local response_body = {}
                local res, code, response_headers = http.request{
                    url = server_url .. "/camera",
                    method = "POST",
                    headers = {
                        ["Content-Type"] = "application/octet-stream",
                        ["Content-Length"] = #rgbImage
                    },
                    source = ltn12.source.string(rgbImage),
                    sink = ltn12.sink.table(response_body)
                }
                
                if res then
                    print("✅ Camera data sent to HTTP server")
                else
                    print("❌ Failed to send camera data")
                end
            end
        end
    end
end

function sysCall_cleanup()
    print("🌐 HTTP client stopped")
end
'''
    
    with open("coppelia_http_client.lua", "w") as f:
        f.write(lua_script)
    
    print("📝 Created CoppeliaSim HTTP client script: coppelia_http_client.lua")
    return lua_script

def main():
    """Test HTTP server communication."""
    print("🌐 Testing HTTP Server Communication with CoppeliaSim")
    print("=" * 60)
    
    # Create CoppeliaSim Lua script
    lua_script = create_coppelia_lua_script()
    
    # Start HTTP server
    server = CoppeliaSimHTTPServer()
    if server.start_server():
        print("✅ HTTP server is running")
        print("\n💡 Instructions:")
        print("   1. Copy the Lua script from coppelia_http_client.lua")
        print("   2. Paste it into CoppeliaSim console or attach to an object")
        print("   3. The script will automatically send camera data to the server")
        print("   4. Press Ctrl+C to stop the server")
        
        try:
            while True:
                time.sleep(1)
                
                # Check for camera data
                camera_data = server.get_camera_data()
                if camera_data:
                    print(f"📸 Received camera data: {len(camera_data)} bytes")
                    
                    # Save camera data
                    with open("http_camera_data.bin", "wb") as f:
                        f.write(camera_data)
                    
                    print("💾 Camera data saved to: http_camera_data.bin")
                    
        except KeyboardInterrupt:
            print("\n🛑 Stopping server...")
        finally:
            server.stop_server()
    else:
        print("❌ Failed to start HTTP server")

if __name__ == "__main__":
    main()




