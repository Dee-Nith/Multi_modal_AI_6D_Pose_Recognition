#!/usr/bin/env python3
"""
Live Traditional CoppeliaSim Connection
Using the traditional CoppeliaSim Python API for real-time camera feed
"""

import sys
import os
import numpy as np
import cv2
import time
from ultralytics import YOLO

# Add CoppeliaSim API to path
sys.path.insert(0, 'src/utils/coppeliasim_api')

try:
    import sim
    print("✅ CoppeliaSim API imported successfully")
except ImportError as e:
    print(f"❌ Failed to import CoppeliaSim API: {e}")
    sys.exit(1)

class LiveTraditionalConnection:
    def __init__(self):
        self.client_id = None
        self.yolo_model = None
        
        # Load YOLO model
        try:
            self.yolo_model = YOLO('yolov8n.pt')
            print("✅ YOLO model loaded successfully")
        except Exception as e:
            print(f"⚠️ Could not load YOLO model: {e}")
    
    def connect(self):
        """Connect to CoppeliaSim."""
        try:
            # Close any existing connections
            sim.simxFinish(-1)
            
            # Connect to CoppeliaSim
            self.client_id = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
            
            if self.client_id != -1:
                print("✅ Connected to CoppeliaSim")
                return True
            else:
                print("❌ Failed to connect to CoppeliaSim")
                return False
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def get_camera_handles(self):
        """Get handles for RGB and depth cameras."""
        print("🔍 Getting camera handles...")
        
        try:
            # Get RGB camera handle
            res, rgb_handle = sim.simxGetObjectHandle(self.client_id, 'sensorRGB', sim.simx_opmode_blocking)
            if res == sim.simx_return_ok:
                print(f"✅ RGB camera handle: {rgb_handle}")
            else:
                print(f"❌ Failed to get RGB camera handle: {res}")
                return None, None
            
            # Get depth camera handle
            res, depth_handle = sim.simxGetObjectHandle(self.client_id, 'sensorDepth', sim.simx_opmode_blocking)
            if res == sim.simx_return_ok:
                print(f"✅ Depth camera handle: {depth_handle}")
            else:
                print(f"❌ Failed to get depth camera handle: {res}")
                return None, None
            
            return rgb_handle, depth_handle
        except Exception as e:
            print(f"❌ Error getting camera handles: {e}")
            return None, None
    
    def get_camera_image(self, handle):
        """Get image from camera handle."""
        if handle is None:
            return None
        
        try:
            # Get image data
            res, resolution, image = sim.simxGetVisionSensorImage(self.client_id, handle, 0, sim.simx_opmode_blocking)
            
            if res == sim.simx_return_ok:
                # Convert to numpy array
                img_array = np.array(image, dtype=np.uint8)
                img_array = img_array.reshape(resolution[1], resolution[0], 3)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                return img_array
            else:
                print(f"❌ Failed to get image: {res}")
                return None
        except Exception as e:
            print(f"❌ Error getting camera image: {e}")
            return None
    
    def get_camera_depth(self, handle):
        """Get depth data from camera handle."""
        if handle is None:
            return None
        
        try:
            # Get depth data
            res, resolution, depth = sim.simxGetVisionSensorDepthBuffer(self.client_id, handle, sim.simx_opmode_blocking)
            
            if res == sim.simx_return_ok:
                # Convert to numpy array
                depth_array = np.array(depth, dtype=np.float32)
                depth_array = depth_array.reshape(resolution[1], resolution[0])
                return depth_array
            else:
                print(f"❌ Failed to get depth: {res}")
                return None
        except Exception as e:
            print(f"❌ Error getting camera depth: {e}")
            return None
    
    def detect_objects(self, image):
        """Detect objects using YOLO."""
        if self.yolo_model is None or image is None:
            return []
        
        try:
            results = self.yolo_model(image, verbose=False)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        class_name = result.names[cls]
                        
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'class': class_name,
                            'class_id': cls
                        })
            
            return detections
        except Exception as e:
            print(f"❌ Error in object detection: {e}")
            return []
    
    def draw_detections(self, image, detections):
        """Draw bounding boxes on image."""
        if image is None:
            return image
        
        img_copy = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_name = det['class']
            
            # Draw bounding box
            cv2.rectangle(img_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(img_copy, label, (int(x1), int(y1)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return img_copy
    
    def start_live_feed(self):
        """Start live camera feed with object detection."""
        print("🎥 Starting live camera feed via traditional API...")
        print("📋 Instructions:")
        print("   - Press 'q' to quit")
        print("   - Press 's' to save current frame")
        print("   - Press 'd' to toggle object detection")
        
        if not self.connect():
            print("❌ Could not connect to CoppeliaSim")
            return
        
        # Get camera handles
        rgb_handle, depth_handle = self.get_camera_handles()
        if rgb_handle is None or depth_handle is None:
            print("❌ Could not get camera handles")
            return
        
        show_detections = True
        frame_count = 0
        
        while True:
            try:
                # Get camera images
                rgb_image = self.get_camera_image(rgb_handle)
                depth_image = self.get_camera_depth(depth_handle)
                
                if rgb_image is not None:
                    # Create display image
                    display_img = rgb_image.copy()
                    
                    # Perform object detection if enabled
                    if show_detections:
                        detections = self.detect_objects(rgb_image)
                        display_img = self.draw_detections(display_img, detections)
                        
                        # Print detection info
                        if detections:
                            print(f"🔍 Frame {frame_count}: Detected {len(detections)} objects")
                            for det in detections:
                                print(f"   - {det['class']} (conf: {det['confidence']:.2f})")
                    
                    # Display the image
                    cv2.imshow('CoppeliaSim Live Camera Feed (Traditional)', cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        # Save current frame
                        timestamp = int(time.time())
                        cv2.imwrite(f'live_frame_{timestamp}.jpg', cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))
                        print(f"💾 Saved frame: live_frame_{timestamp}.jpg")
                    elif key == ord('d'):
                        show_detections = not show_detections
                        print(f"🔍 Object detection: {'ON' if show_detections else 'OFF'}")
                    
                    frame_count += 1
                else:
                    print("❌ No camera data received")
                
                # Small delay
                time.sleep(0.1)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error in live feed: {e}")
                break
        
        cv2.destroyAllWindows()
        self.disconnect()
    
    def disconnect(self):
        """Disconnect from CoppeliaSim."""
        if self.client_id is not None:
            sim.simxFinish(self.client_id)
        print("🔌 Disconnected from CoppeliaSim")

def main():
    """Main function."""
    print("🤖 Live CoppeliaSim Camera Connection (Traditional API)")
    print("=" * 55)
    
    # Create connection
    connection = LiveTraditionalConnection()
    
    try:
        connection.start_live_feed()
    except KeyboardInterrupt:
        print("\n🛑 Stopping live feed...")
    finally:
        connection.disconnect()

if __name__ == "__main__":
    main()





