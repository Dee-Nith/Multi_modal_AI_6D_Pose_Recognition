#!/usr/bin/env python3
"""
Fixed ZeroMQ Client for CoppeliaSim
Uses the correct protocol for direct communication
"""

import zmq
import struct
import time
import numpy as np
import cv2

class FixedCoppeliaSimZMQClient:
    def __init__(self, host='localhost', port=23000):
        self.host = host
        self.port = port
        self.context = None
        self.socket = None
        self.connected = False
        
    def connect(self):
        """Connect to CoppeliaSim ZeroMQ server."""
        try:
            print(f"🔌 Connecting to CoppeliaSim at {self.host}:{self.port}...")
            
            # Create ZeroMQ context
            self.context = zmq.Context()
            
            # Create socket
            self.socket = self.context.socket(zmq.REQ)
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
            self.socket.setsockopt(zmq.SNDTIMEO, 5000)  # 5 second timeout
            
            # Connect to CoppeliaSim
            self.socket.connect(f"tcp://{self.host}:{self.port}")
            
            # Test connection with a simple command
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
        """Test the connection with a simple command."""
        try:
            # Send a simple command to test connection
            command = "sim.getSimulationTime"
            response = self.send_command(command)
            
            if response is not None:
                print(f"✅ Connection test successful: {response}")
                return True
            else:
                print("❌ No response from CoppeliaSim")
                return False
                
        except Exception as e:
            print(f"❌ Connection test error: {e}")
            return False
    
    def send_command(self, command, *args):
        """Send a command to CoppeliaSim and get response."""
        if not self.connected:
            print("❌ Not connected to CoppeliaSim")
            return None
        
        try:
            # Prepare the message
            if args:
                # Convert args to strings and join
                arg_str = ",".join(str(arg) for arg in args)
                full_command = f"{command}({arg_str})"
            else:
                full_command = f"{command}()"
            
            print(f"📤 Sending: {full_command}")
            
            # Send command
            self.socket.send_string(full_command)
            
            # Receive response
            response = self.socket.recv_string()
            
            print(f"📥 Received: {response}")
            return response
            
        except zmq.error.Again:
            print("❌ Timeout waiting for response")
            return None
        except Exception as e:
            print(f"❌ Command error: {e}")
            return None
    
    def get_camera_handles(self):
        """Get handles for RGB and depth cameras."""
        try:
            # Get the spherical vision sensor handle
            rgb_handle = self.send_command("sim.getObject", "./sensorRGB")
            depth_handle = self.send_command("sim.getObject", "./sensorDepth")
            
            if rgb_handle and depth_handle:
                print(f"✅ Camera handles: RGB={rgb_handle}, Depth={depth_handle}")
                return rgb_handle, depth_handle
            else:
                print("❌ Failed to get camera handles")
                return None, None
                
        except Exception as e:
            print(f"❌ Error getting camera handles: {e}")
            return None, None
    
    def capture_rgb_image(self, rgb_handle):
        """Capture RGB image from camera."""
        try:
            # Get RGB image
            rgb_data = self.send_command("sim.getVisionSensorImg", rgb_handle)
            
            if rgb_data and rgb_data != "nil":
                # Parse the response (it should be a table of numbers)
                # Remove the table brackets and split by commas
                rgb_data = rgb_data.strip("{}")
                rgb_values = [int(x.strip()) for x in rgb_data.split(",") if x.strip().isdigit()]
                
                if len(rgb_values) >= 512 * 256 * 3:
                    # Reshape to image
                    image = np.array(rgb_values[:512*256*3], dtype=np.uint8)
                    image = image.reshape((256, 512, 3))
                    return image
                else:
                    print(f"❌ Invalid RGB data size: {len(rgb_values)}")
                    return None
            else:
                print("❌ No RGB data received")
                return None
                
        except Exception as e:
            print(f"❌ Error capturing RGB image: {e}")
            return None
    
    def capture_depth_image(self, depth_handle):
        """Capture depth image from camera."""
        try:
            # Get depth image
            depth_data = self.send_command("sim.getVisionSensorDepth", depth_handle)
            
            if depth_data and depth_data != "nil":
                # Parse the response
                depth_data = depth_data.strip("{}")
                depth_values = [float(x.strip()) for x in depth_data.split(",") if x.strip().replace('.', '').replace('-', '').isdigit()]
                
                if len(depth_values) >= 512 * 256:
                    # Reshape to image
                    depth_image = np.array(depth_values[:512*256], dtype=np.float32)
                    depth_image = depth_image.reshape((256, 512))
                    return depth_image
                else:
                    print(f"❌ Invalid depth data size: {len(depth_values)}")
                    return None
            else:
                print("❌ No depth data received")
                return None
                
        except Exception as e:
            print(f"❌ Error capturing depth image: {e}")
            return None
    
    def get_live_camera_feed(self):
        """Get live camera feed from CoppeliaSim."""
        try:
            # Get camera handles
            rgb_handle, depth_handle = self.get_camera_handles()
            
            if rgb_handle and depth_handle:
                # Capture images
                rgb_image = self.capture_rgb_image(rgb_handle)
                depth_image = self.capture_depth_image(depth_handle)
                
                return rgb_image, depth_image
            else:
                print("❌ Could not get camera handles")
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
    """Test the fixed ZeroMQ client."""
    print("🤖 Testing Fixed ZeroMQ Client for CoppeliaSim")
    print("=" * 50)
    
    with FixedCoppeliaSimZMQClient() as client:
        if client.connected:
            print("\n📸 Testing live camera feed...")
            
            # Get live camera feed
            rgb_image, depth_image = client.get_live_camera_feed()
            
            if rgb_image is not None:
                print(f"✅ RGB image captured: {rgb_image.shape}")
                
                # Save the image
                cv2.imwrite("direct_rgb_capture.jpg", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
                print("💾 RGB image saved as: direct_rgb_capture.jpg")
                
                # Display image info
                print(f"📊 Image info:")
                print(f"   - Shape: {rgb_image.shape}")
                print(f"   - Data type: {rgb_image.dtype}")
                print(f"   - Value range: {rgb_image.min()} - {rgb_image.max()}")
                print(f"   - Mean value: {rgb_image.mean():.2f}")
            else:
                print("❌ Failed to capture RGB image")
            
            if depth_image is not None:
                print(f"✅ Depth image captured: {depth_image.shape}")
                
                # Save the depth image
                depth_normalized = ((depth_image - depth_image.min()) / (depth_image.max() - depth_image.min()) * 255).astype(np.uint8)
                cv2.imwrite("direct_depth_capture.jpg", depth_normalized)
                print("💾 Depth image saved as: direct_depth_capture.jpg")
                
                print(f"📊 Depth info:")
                print(f"   - Shape: {depth_image.shape}")
                print(f"   - Value range: {depth_image.min():.3f} - {depth_image.max():.3f}")
                print(f"   - Mean value: {depth_image.mean():.3f}")
            else:
                print("❌ Failed to capture depth image")
        else:
            print("❌ Could not connect to CoppeliaSim")

if __name__ == "__main__":
    main()




