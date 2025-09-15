#!/usr/bin/env python3
"""
CoppeliaSim Connection for Real-time 6D Pose Estimation
Connects to CoppeliaSim and captures live video feed
"""

import sim
import numpy as np
import cv2
import time
import sys
import os
from pathlib import Path

class CoppeliaSimConnection:
    def __init__(self):
        """Initialize CoppeliaSim connection"""
        self.client_id = None
        self.kinect_handle = None
        self.rgb_handle = None
        self.depth_handle = None
        
    def connect_to_coppelia(self, ip='127.0.0.1', port=23000):
        """Connect to CoppeliaSim"""
        print(f"🔗 Connecting to CoppeliaSim at {ip}:{port}...")
        
        # Close any existing connections
        sim.simxFinish(-1)
        
        # Connect to CoppeliaSim
        self.client_id = sim.simxStart(ip, port, True, True, 5000, 5)
        
        if self.client_id != -1:
            print("✅ Successfully connected to CoppeliaSim!")
            
            # Start simulation
            sim.simxStartSimulation(self.client_id, sim.simx_opmode_oneshot)
            time.sleep(1)  # Wait for simulation to start
            
            # Get Kinect handles
            self.get_kinect_handles()
            
            return True
        else:
            print("❌ Failed to connect to CoppeliaSim!")
            print("💡 Make sure CoppeliaSim is running and the scene is loaded")
            return False
    
    def get_kinect_handles(self):
        """Get handles for Kinect camera components"""
        print("📷 Getting Kinect camera handles...")
        
        # Get Kinect object handle
        ret, self.kinect_handle = sim.simxGetObjectHandle(
            self.client_id, 'kinect', sim.simx_opmode_blocking
        )
        
        if ret == sim.simx_return_ok:
            print("✅ Found Kinect object")
            
            # Get RGB camera handle
            ret, self.rgb_handle = sim.simxGetObjectHandle(
                self.client_id, 'kinect_rgb', sim.simx_opmode_blocking
            )
            
            if ret == sim.simx_return_ok:
                print("✅ Found RGB camera")
            else:
                print("⚠️ RGB camera not found, trying alternative name...")
                ret, self.rgb_handle = sim.simxGetObjectHandle(
                    self.client_id, 'kinect#0', sim.simx_opmode_blocking
                )
                if ret == sim.simx_return_ok:
                    print("✅ Found RGB camera (alternative name)")
                else:
                    print("❌ Could not find RGB camera")
            
            # Get depth camera handle
            ret, self.depth_handle = sim.simxGetObjectHandle(
                self.client_id, 'kinect_depth', sim.simx_opmode_blocking
            )
            
            if ret == sim.simx_return_ok:
                print("✅ Found depth camera")
            else:
                print("⚠️ Depth camera not found")
        else:
            print("❌ Could not find Kinect object")
    
    def capture_rgb_frame(self):
        """Capture RGB frame from Kinect"""
        if self.rgb_handle is None:
            return None
        
        # Get image from vision sensor
        ret, resolution, image = sim.simxGetVisionSensorImage(
            self.client_id, self.rgb_handle, 0, sim.simx_opmode_blocking
        )
        
        if ret == sim.simx_return_ok:
            # Convert to numpy array
            image = np.array(image, dtype=np.uint8)
            
            # Reshape based on resolution
            if len(resolution) >= 2:
                height, width = resolution[1], resolution[0]
                image = image.reshape(height, width, 3)
                
                # Convert from RGB to BGR (OpenCV format)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                return image
            else:
                print("⚠️ Invalid image resolution")
                return None
        else:
            print("❌ Failed to capture RGB frame")
            return None
    
    def capture_depth_frame(self):
        """Capture depth frame from Kinect"""
        if self.depth_handle is None:
            return None
        
        # Get depth image from vision sensor
        ret, resolution, depth_data = sim.simxGetVisionSensorImage(
            self.client_id, self.depth_handle, 1, sim.simx_opmode_blocking
        )
        
        if ret == sim.simx_return_ok:
            # Convert to numpy array
            depth_data = np.array(depth_data, dtype=np.float32)
            
            # Reshape based on resolution
            if len(resolution) >= 2:
                height, width = resolution[1], resolution[0]
                depth_data = depth_data.reshape(height, width)
                
                return depth_data
            else:
                print("⚠️ Invalid depth resolution")
                return None
        else:
            print("❌ Failed to capture depth frame")
            return None
    
    def get_object_poses(self):
        """Get poses of objects in the scene"""
        print("🎯 Getting object poses from CoppeliaSim...")
        
        object_poses = {}
        object_names = ['master_chef_can', 'cracker_box', 'mug', 'banana', 'mustard_bottle']
        
        for obj_name in object_names:
            # Try to get object handle
            ret, obj_handle = sim.simxGetObjectHandle(
                self.client_id, obj_name, sim.simx_opmode_blocking
            )
            
            if ret == sim.simx_return_ok:
                # Get object position
                ret, position = sim.simxGetObjectPosition(
                    self.client_id, obj_handle, -1, sim.simx_opmode_blocking
                )
                
                # Get object orientation
                ret, orientation = sim.simxGetObjectOrientation(
                    self.client_id, obj_handle, -1, sim.simx_opmode_blocking
                )
                
                if ret == sim.simx_return_ok:
                    object_poses[obj_name] = {
                        'position': position,
                        'orientation': orientation,
                        'handle': obj_handle
                    }
                    print(f"✅ Found {obj_name}: pos={position}, rot={orientation}")
                else:
                    print(f"⚠️ Could not get pose for {obj_name}")
            else:
                print(f"⚠️ Could not find object: {obj_name}")
        
        return object_poses
    
    def move_object(self, obj_name, position, orientation=None):
        """Move an object in CoppeliaSim"""
        ret, obj_handle = sim.simxGetObjectHandle(
            self.client_id, obj_name, sim.simx_opmode_blocking
        )
        
        if ret == sim.simx_return_ok:
            # Set position
            sim.simxSetObjectPosition(
                self.client_id, obj_handle, -1, position, sim.simx_opmode_oneshot
            )
            
            # Set orientation if provided
            if orientation is not None:
                sim.simxSetObjectOrientation(
                    self.client_id, obj_handle, -1, orientation, sim.simx_opmode_oneshot
                )
            
            print(f"✅ Moved {obj_name} to position {position}")
            return True
        else:
            print(f"❌ Could not find object: {obj_name}")
            return False
    
    def start_realtime_capture(self, duration=30):
        """Start real-time capture for pose estimation"""
        print(f"🎥 Starting real-time capture for {duration} seconds...")
        print("📹 Press 'q' to quit early")
        
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < duration:
            # Capture RGB frame
            rgb_frame = self.capture_rgb_frame()
            
            if rgb_frame is not None:
                # Display frame
                cv2.imshow('CoppeliaSim Live Feed', rgb_frame)
                
                # Save frame for pose estimation
                frame_path = f"../results/coppelia_frame_{frame_count:04d}.jpg"
                cv2.imwrite(frame_path, rgb_frame)
                
                frame_count += 1
                
                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("🛑 Stopping capture...")
                    break
                
                # Small delay
                time.sleep(0.033)  # ~30 FPS
            else:
                print("⚠️ Failed to capture frame")
                time.sleep(0.1)
        
        cv2.destroyAllWindows()
        print(f"✅ Captured {frame_count} frames")
        return frame_count
    
    def disconnect(self):
        """Disconnect from CoppeliaSim"""
        if self.client_id is not None:
            sim.simxStopSimulation(self.client_id, sim.simx_opmode_oneshot)
            sim.simxFinish(self.client_id)
            print("🔌 Disconnected from CoppeliaSim")

def main():
    """Main function to test CoppeliaSim connection"""
    print("🎯 CoppeliaSim Connection Test")
    print("=" * 40)
    
    # Initialize connection
    coppelia = CoppeliaSimConnection()
    
    # Connect to CoppeliaSim
    if coppelia.connect_to_coppelia():
        print("\n🎉 Successfully connected to CoppeliaSim!")
        
        # Get object poses
        object_poses = coppelia.get_object_poses()
        
        # Test capture
        print("\n📸 Testing frame capture...")
        rgb_frame = coppelia.capture_rgb_frame()
        
        if rgb_frame is not None:
            print(f"✅ Captured RGB frame: {rgb_frame.shape}")
            
            # Save test frame
            cv2.imwrite("../results/coppelia_test_frame.jpg", rgb_frame)
            print("📸 Test frame saved to: ../results/coppelia_test_frame.jpg")
            
            # Display frame
            cv2.imshow('CoppeliaSim Test Frame', rgb_frame)
            cv2.waitKey(2000)  # Show for 2 seconds
            cv2.destroyAllWindows()
            
            # Ask if user wants to start real-time capture
            choice = input("\n🎥 Start real-time capture? (y/n): ").strip().lower()
            
            if choice == 'y':
                duration = int(input("Enter capture duration in seconds (default 30): ") or "30")
                coppelia.start_realtime_capture(duration)
        else:
            print("❌ Failed to capture test frame")
        
        # Disconnect
        coppelia.disconnect()
    else:
        print("❌ Could not connect to CoppeliaSim")

if __name__ == "__main__":
    main()




