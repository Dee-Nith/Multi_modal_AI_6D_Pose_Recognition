#!/usr/bin/env python3
"""
CoppeliaSim Protocol Client
Uses the correct ZeroMQ protocol that CoppeliaSim expects
"""

import zmq
import struct
import time
import numpy as np
import cv2

class CoppeliaSimProtocolClient:
    def __init__(self, host='localhost', port=23000):
        self.host = host
        self.port = port
        self.context = None
        self.socket = None
        self.connected = False
        
    def connect(self):
        """Connect to CoppeliaSim using proper protocol."""
        try:
            print(f"🔌 Connecting to CoppeliaSim at {self.host}:{self.port}...")
            
            # Create ZeroMQ context
            self.context = zmq.Context()
            
            # Create socket
            self.socket = self.context.socket(zmq.REQ)
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.setsockopt(zmq.RCVTIMEO, 5000)
            self.socket.setsockopt(zmq.SNDTIMEO, 5000)
            
            # Connect to CoppeliaSim
            self.socket.connect(f"tcp://{self.host}:{self.port}")
            
            # Test connection
            if self.test_connection():
                self.connected = True
                print("✅ Successfully connected to CoppeliaSim!")
                return True
            else:
                print("❌ Connection test failed")
                return False
                
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def test_connection(self):
        """Test connection with proper protocol."""
        try:
            # Try to get simulation time
            response = self.send_protocol_command("sim.getSimulationTime")
            
            if response is not None and "error" not in str(response).lower():
                print(f"✅ Connection test successful: {response}")
                return True
            else:
                print(f"❌ Connection test failed: {response}")
                return False
                
        except Exception as e:
            print(f"❌ Connection test error: {e}")
            return False
    
    def send_protocol_command(self, command, *args):
        """Send command using CoppeliaSim's protocol."""
        if not self.connected:
            print("❌ Not connected to CoppeliaSim")
            return None
        
        try:
            # Prepare the command
            if args:
                arg_str = ",".join(str(arg) for arg in args)
                full_command = f"{command}({arg_str})"
            else:
                full_command = f"{command}()"
            
            print(f"📤 Sending: {full_command}")
            
            # Send command
            self.socket.send_string(full_command)
            
            # Receive response
            response = self.socket.recv()
            
            # Try to decode response
            try:
                response_str = response.decode('utf-8')
                print(f"📥 Received: {response_str}")
                return response_str
            except UnicodeDecodeError:
                # If it's binary data, return raw bytes
                print(f"📥 Received binary: {len(response)} bytes")
                return response
                
        except zmq.error.Again:
            print("❌ Timeout waiting for response")
            return None
        except Exception as e:
            print(f"❌ Command error: {e}")
            return None
    
    def get_camera_handles(self):
        """Get camera handles using proper protocol."""
        try:
            # Get handles for the spherical vision sensor
            rgb_handle = self.send_protocol_command("sim.getObject", "./sensorRGB")
            depth_handle = self.send_protocol_command("sim.getObject", "./sensorDepth")
            
            if rgb_handle and depth_handle and "error" not in str(rgb_handle).lower():
                print(f"✅ Camera handles: RGB={rgb_handle}, Depth={depth_handle}")
                return rgb_handle, depth_handle
            else:
                print("❌ Failed to get camera handles")
                return None, None
                
        except Exception as e:
            print(f"❌ Error getting camera handles: {e}")
            return None, None
    
    def capture_camera_data(self):
        """Capture camera data using file-based approach (fallback)."""
        try:
            # Since direct protocol is having issues, let's use the file-based approach
            # but trigger the capture from CoppeliaSim
            
            # Send command to capture camera data
            response = self.send_protocol_command("sim.callScriptFunction", "captureCamera", "sphericalVisionRGBAndDepth")
            
            if response and "error" not in str(response).lower():
                print("✅ Camera capture triggered")
                
                # Wait a moment for file to be written
                time.sleep(0.5)
                
                # Read the captured data
                rgb_file = "current_rgb.txt"
                if os.path.exists(rgb_file):
                    with open(rgb_file, 'rb') as f:
                        content = f.read()
                    
                    data = np.frombuffer(content, dtype=np.uint8)
                    if len(data) >= 512 * 256 * 3:
                        image = data[:512*256*3].reshape((256, 512, 3))
                        return image
                
                print("❌ Could not read captured data")
                return None
            else:
                print("❌ Failed to trigger camera capture")
                return None
                
        except Exception as e:
            print(f"❌ Error capturing camera data: {e}")
            return None
    
    def get_live_camera_feed(self):
        """Get live camera feed."""
        try:
            # Try direct capture first
            rgb_image = self.capture_camera_data()
            
            if rgb_image is not None:
                return rgb_image, None  # Depth not implemented yet
            else:
                print("❌ Could not capture camera data")
                return None, None
                
        except Exception as e:
            print(f"❌ Error getting live camera feed: {e}")
            return None, None
    
    def disconnect(self):
        """Disconnect from CoppeliaSim."""
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        self.connected = False
        print("🔌 Disconnected from CoppeliaSim")
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

def main():
    """Test the CoppeliaSim protocol client."""
    print("🤖 Testing CoppeliaSim Protocol Client")
    print("=" * 50)
    
    import os
    
    with CoppeliaSimProtocolClient() as client:
        if client.connected:
            print("\n📸 Testing camera capture...")
            
            # Get live camera feed
            rgb_image, depth_image = client.get_live_camera_feed()
            
            if rgb_image is not None:
                print(f"✅ RGB image captured: {rgb_image.shape}")
                
                # Save the image
                cv2.imwrite("protocol_rgb_capture.jpg", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
                print("💾 RGB image saved as: protocol_rgb_capture.jpg")
                
                # Display image info
                print(f"📊 Image info:")
                print(f"   - Shape: {rgb_image.shape}")
                print(f"   - Data type: {rgb_image.dtype}")
                print(f"   - Value range: {rgb_image.min()} - {rgb_image.max()}")
                print(f"   - Mean value: {rgb_image.mean():.2f}")
            else:
                print("❌ Failed to capture RGB image")
        else:
            print("❌ Could not connect to CoppeliaSim")

if __name__ == "__main__":
    main()




