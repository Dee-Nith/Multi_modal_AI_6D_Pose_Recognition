#!/usr/bin/env python3
"""
CoppeliaSim Client using Rosetta 2
Runs the traditional CoppeliaSim API using Rosetta 2 for Apple Silicon compatibility
"""

import subprocess
import sys
import os
import time
import json

class RosettaCoppeliaClient:
    def __init__(self):
        self.client_id = None
        self.connected = False
        
    def setup_rosetta_environment(self):
        """Set up the environment for Rosetta 2 execution."""
        try:
            # Check if we're on Apple Silicon
            import platform
            if platform.machine() == 'arm64':
                print("🍎 Detected Apple Silicon (ARM64)")
                print("🔄 Setting up Rosetta 2 environment...")
                
                # Set environment variables for Rosetta 2
                os.environ['ARCHFLAGS'] = '-arch x86_64'
                os.environ['PYTHONPATH'] = '/Applications/coppeliaSim.app/Contents/Resources/programming/remoteApiBindings/python/python'
                
                return True
            else:
                print("✅ Running on x86_64 architecture")
                return True
                
        except Exception as e:
            print(f"❌ Error setting up Rosetta environment: {e}")
            return False
    
    def run_with_rosetta(self, script_content):
        """Run a Python script using Rosetta 2."""
        try:
            # Create a temporary script file
            script_file = "temp_rosetta_script.py"
            with open(script_file, "w") as f:
                f.write(script_content)
            
            # Run with Rosetta 2
            if sys.platform == "darwin":
                # Use arch command to run with x86_64 architecture
                cmd = ["arch", "-x86_64", "python3", script_file]
            else:
                cmd = ["python3", script_file]
            
            print(f"🚀 Running with command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            # Clean up
            os.remove(script_file)
            
            if result.returncode == 0:
                print("✅ Rosetta execution successful")
                return result.stdout
            else:
                print(f"❌ Rosetta execution failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print("⏰ Rosetta execution timed out")
            return None
        except Exception as e:
            print(f"❌ Rosetta execution error: {e}")
            return None
    
    def test_connection(self):
        """Test connection to CoppeliaSim using Rosetta 2."""
        script = '''
import sys
import os

# Add CoppeliaSim API path
sys.path.append('/Applications/coppeliaSim.app/Contents/Resources/programming/remoteApiBindings/python/python')

try:
    import sim
    import simConst
    
    print("✅ CoppeliaSim API imported successfully")
    
    # Try to connect
    client_id = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    
    if client_id != -1:
        print(f"✅ Connected with client ID: {client_id}")
        
        # Test a simple command
        result, time = sim.simxGetSimulationTime(client_id, sim.simx_opmode_blocking)
        if result == sim.simx_return_ok:
            print(f"✅ Simulation time: {time}")
            sim.simxFinish(client_id)
            print("SUCCESS")
        else:
            print(f"❌ Command failed: {result}")
            sim.simxFinish(client_id)
            print("FAILED")
    else:
        print("❌ Failed to connect")
        print("FAILED")
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("FAILED")
except Exception as e:
    print(f"❌ Error: {e}")
    print("FAILED")
'''
        
        result = self.run_with_rosetta(script)
        if result and "SUCCESS" in result:
            print("✅ Connection test successful!")
            return True
        else:
            print("❌ Connection test failed")
            return False
    
    def get_camera_data(self):
        """Get camera data using Rosetta 2."""
        script = '''
import sys
import os
import numpy as np

# Add CoppeliaSim API path
sys.path.append('/Applications/coppeliaSim.app/Contents/Resources/programming/remoteApiBindings/python/python')

try:
    import sim
    import simConst
    
    # Connect to CoppeliaSim
    client_id = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    
    if client_id != -1:
        print("✅ Connected to CoppeliaSim")
        
        # Get camera handle
        result, rgb_handle = sim.simxGetObjectHandle(client_id, 'sensorRGB', sim.simx_opmode_blocking)
        
        if result == sim.simx_return_ok:
            print(f"✅ Got RGB camera handle: {rgb_handle}")
            
            # Get RGB image
            result, resolution, image = sim.simxGetVisionSensorImage(client_id, rgb_handle, 0, sim.simx_opmode_blocking)
            
            if result == sim.simx_return_ok:
                print(f"✅ Got RGB image: {len(image)} pixels")
                
                # Save image data
                with open("rosetta_rgb_data.txt", "w") as f:
                    for pixel in image:
                        f.write(str(pixel) + "\\n")
                
                print("SUCCESS")
            else:
                print(f"❌ Failed to get RGB image: {result}")
                print("FAILED")
        else:
            print(f"❌ Failed to get camera handle: {result}")
            print("FAILED")
        
        sim.simxFinish(client_id)
    else:
        print("❌ Failed to connect")
        print("FAILED")
        
except Exception as e:
    print(f"❌ Error: {e}")
    print("FAILED")
'''
        
        result = self.run_with_rosetta(script)
        if result and "SUCCESS" in result:
            print("✅ Camera data captured successfully!")
            return True
        else:
            print("❌ Camera data capture failed")
            return False
    
    def run_object_detection(self):
        """Run object detection on captured camera data."""
        try:
            # Read the captured data
            if os.path.exists("rosetta_rgb_data.txt"):
                with open("rosetta_rgb_data.txt", "r") as f:
                    data = [int(line.strip()) for line in f.readlines()]
                
                # Convert to numpy array
                data = np.array(data, dtype=np.uint8)
                
                if len(data) >= 512 * 256 * 3:
                    # Reshape to image
                    image = data[:512*256*3].reshape((256, 512, 3))
                    
                    # Save image
                    import cv2
                    cv2.imwrite("rosetta_camera_capture.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    
                    print(f"✅ Image saved: {image.shape}")
                    return image
                else:
                    print(f"❌ Invalid data size: {len(data)}")
                    return None
            else:
                print("❌ No camera data file found")
                return None
                
        except Exception as e:
            print(f"❌ Error processing camera data: {e}")
            return None

def main():
    """Test Rosetta 2 CoppeliaSim client."""
    print("🍎 Testing CoppeliaSim Client with Rosetta 2")
    print("=" * 50)
    
    client = RosettaCoppeliaClient()
    
    # Set up Rosetta environment
    if client.setup_rosetta_environment():
        print("✅ Rosetta environment set up")
        
        # Test connection
        if client.test_connection():
            print("✅ Connection successful!")
            
            # Get camera data
            if client.get_camera_data():
                # Run object detection
                image = client.run_object_detection()
                if image is not None:
                    print("🎉 Successfully captured and processed camera data!")
                    print(f"📊 Image shape: {image.shape}")
                    print("💾 Saved as: rosetta_camera_capture.jpg")
                else:
                    print("❌ Failed to process camera data")
            else:
                print("❌ Failed to get camera data")
        else:
            print("❌ Connection failed")
    else:
        print("❌ Failed to set up Rosetta environment")

if __name__ == "__main__":
    main()




