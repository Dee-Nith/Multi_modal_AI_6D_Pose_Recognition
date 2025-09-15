#!/usr/bin/env python3
"""
Live Camera with Trained YOLO Model
Uses the trained YCB texture model for real-time object detection
"""

import numpy as np
import cv2
import time
import os
from pathlib import Path
from ultralytics import YOLO

class LiveCameraWithTrainedModel:
    def __init__(self):
        self.running = False
        self.yolo_model = None
        self.last_rgb_time = 0
        
        # Load the trained YOLO model
        try:
            model_path = "ycb_texture_training/ycb_texture_detector/weights/best.pt"
            if os.path.exists(model_path):
                self.yolo_model = YOLO(model_path)
                print(f"✅ Trained YOLO model loaded: {model_path}")
            else:
                # Fallback to pre-trained model
                self.yolo_model = YOLO('yolov8n.pt')
                print("⚠️ Using pre-trained YOLO model (trained model not found)")
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            self.yolo_model = None
    
    def read_camera_data_simple(self):
        """Read camera data with minimal processing."""
        try:
            # Check for RGB data
            rgb_file = Path("current_rgb.txt")
            
            if rgb_file.exists():
                file_time = os.path.getmtime(rgb_file)
                if file_time > self.last_rgb_time:
                    with open(rgb_file, 'rb') as f:
                        content = f.read()
                    
                    # Decode with error handling
                    try:
                        text = content.decode('utf-8')
                    except UnicodeDecodeError:
                        text = content.decode('utf-8', errors='ignore')
                    
                    lines = text.split('\n')
                    if len(lines) >= 2:
                        # Parse dimensions and data
                        dims = lines[0].strip().split(',')
                        width, height = int(dims[0]), int(dims[1])
                        
                        rgb_data = lines[1].strip().split(',')
                        rgb_values = []
                        
                        # Take all values without filtering
                        for item in rgb_data:
                            item = item.strip()
                            if item:
                                try:
                                    value = int(item)
                                    rgb_values.append(value)
                                except ValueError:
                                    continue
                        
                        # Try to reshape based on data size
                        if len(rgb_values) > 0:
                            # Try common resolutions
                            possible_resolutions = [
                                (512, 256), (640, 480), (800, 600), 
                                (1024, 768), (256, 256), (512, 512)
                            ]
                            
                            for w, h in possible_resolutions:
                                expected_size = w * h * 3
                                if len(rgb_values) >= expected_size:
                                    rgb_array = np.array(rgb_values[:expected_size], dtype=np.uint8)
                                    rgb_image = rgb_array.reshape((h, w, 3))
                                    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
                                    self.last_rgb_time = file_time
                                    return rgb_image
                            
                            # If no standard resolution fits, try square
                            side = int(np.sqrt(len(rgb_values) / 3))
                            if side > 0:
                                expected_size = side * side * 3
                                if len(rgb_values) >= expected_size:
                                    rgb_array = np.array(rgb_values[:expected_size], dtype=np.uint8)
                                    rgb_image = rgb_array.reshape((side, side, 3))
                                    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
                                    self.last_rgb_time = file_time
                                    return rgb_image
            
            return None
            
        except Exception as e:
            print(f"❌ Error reading camera data: {e}")
            return None
    
    def detect_objects(self, image):
        """Detect objects using the trained YOLO model."""
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
                        
                        # Get class name
                        if hasattr(result, 'names') and cls in result.names:
                            class_name = result.names[cls]
                        else:
                            class_name = f"Class_{cls}"
                        
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
        """Start live camera feed with trained model detection."""
        print("🎥 Starting live camera feed with trained YOLO model...")
        print("📋 Instructions:")
        print("   - Press 'q' to quit")
        print("   - Press 's' to save current frame")
        print("   - Press 'd' to toggle object detection")
        print("   - CoppeliaSim should be saving current_rgb.txt")
        
        show_detections = True
        frame_count = 0
        last_frame_time = 0
        
        while self.running:
            try:
                # Read camera data
                rgb_image = self.read_camera_data_simple()
                
                current_time = time.time()
                
                if rgb_image is not None and current_time - last_frame_time > 0.1:  # 10 FPS max
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
                    cv2.imshow('CoppeliaSim Live Camera (Trained Model)', cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        # Save current frame
                        timestamp = int(time.time())
                        cv2.imwrite(f'live_frame_trained_{timestamp}.jpg', cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))
                        print(f"💾 Saved frame: live_frame_trained_{timestamp}.jpg")
                    elif key == ord('d'):
                        show_detections = not show_detections
                        print(f"🔍 Object detection: {'ON' if show_detections else 'OFF'}")
                    
                    frame_count += 1
                    last_frame_time = current_time
                else:
                    # Show a placeholder if no camera data
                    if current_time - last_frame_time > 2.0:  # Show message every 2 seconds
                        print("⏳ Waiting for camera data from CoppeliaSim...")
                        last_frame_time = current_time
                
                # Small delay
                time.sleep(0.05)  # 20 FPS monitoring
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error in live feed: {e}")
                break
        
        cv2.destroyAllWindows()
    
    def start(self):
        """Start the live monitor."""
        self.running = True
        self.start_live_feed()
    
    def stop(self):
        """Stop the live monitor."""
        self.running = False

def main():
    """Main function."""
    print("🤖 Live CoppeliaSim Camera with Trained YOLO Model")
    print("=" * 55)
    
    # Create monitor
    monitor = LiveCameraWithTrainedModel()
    
    try:
        monitor.start()
    except KeyboardInterrupt:
        print("\n🛑 Stopping live feed...")
    finally:
        monitor.stop()

if __name__ == "__main__":
    main()




