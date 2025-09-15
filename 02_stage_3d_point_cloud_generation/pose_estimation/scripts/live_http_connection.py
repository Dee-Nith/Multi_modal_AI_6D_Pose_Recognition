#!/usr/bin/env python3
"""
Live HTTP Connection to CoppeliaSim
Direct HTTP connection for real-time camera feed
"""

import requests
import numpy as np
import cv2
import time
import json
from ultralytics import YOLO
import base64

class LiveHTTPConnection:
    def __init__(self, port=23000):
        self.port = port
        self.base_url = f"http://localhost:{port}"
        self.yolo_model = None
        
        # Load YOLO model
        try:
            self.yolo_model = YOLO('yolov8n.pt')
            print("✅ YOLO model loaded successfully")
        except Exception as e:
            print(f"⚠️ Could not load YOLO model: {e}")
    
    def test_connection(self):
        """Test connection to CoppeliaSim."""
        try:
            response = requests.get(f"{self.base_url}/api/v1/sim", timeout=5)
            if response.status_code == 200:
                print("✅ Connected to CoppeliaSim HTTP API")
                return True
            else:
                print(f"❌ HTTP API returned status {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Failed to connect to CoppeliaSim: {e}")
            return False
    
    def get_camera_data(self):
        """Get camera data from CoppeliaSim."""
        try:
            # Get RGB image
            rgb_response = requests.get(f"{self.base_url}/api/v1/vision/rgb", timeout=5)
            if rgb_response.status_code == 200:
                # Decode base64 image
                img_data = base64.b64decode(rgb_response.json()['data'])
                img_array = np.frombuffer(img_data, dtype=np.uint8)
                rgb_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
                return rgb_image, None
            else:
                print(f"❌ Failed to get RGB image: {rgb_response.status_code}")
                return None, None
        except Exception as e:
            print(f"❌ Error getting camera data: {e}")
            return None, None
    
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
        print("🎥 Starting live camera feed via HTTP...")
        print("📋 Instructions:")
        print("   - Press 'q' to quit")
        print("   - Press 's' to save current frame")
        print("   - Press 'd' to toggle object detection")
        
        if not self.test_connection():
            print("❌ Could not connect to CoppeliaSim")
            print("💡 Make sure CoppeliaSim is running with Web Server enabled")
            return
        
        show_detections = True
        frame_count = 0
        
        while True:
            try:
                # Get camera data
                rgb_image, depth_image = self.get_camera_data()
                
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
                    cv2.imshow('CoppeliaSim Live Camera Feed (HTTP)', cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))
                    
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

def main():
    """Main function."""
    print("🤖 Live CoppeliaSim Camera Connection (HTTP)")
    print("=" * 50)
    
    # Create connection
    connection = LiveHTTPConnection()
    
    try:
        connection.start_live_feed()
    except KeyboardInterrupt:
        print("\n🛑 Stopping live feed...")

if __name__ == "__main__":
    main()





