#!/usr/bin/env python3
"""
Live Camera Feed with YOLO Detection
Uses the working camera data parser with real-time YOLO detection
"""

import numpy as np
import cv2
import time
import os
from pathlib import Path
from ultralytics import YOLO

class LiveCameraYOLODetection:
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
    
    def read_binary_camera_data(self, filename):
        """Read camera data as binary and convert to image."""
        try:
            with open(filename, 'rb') as f:
                content = f.read()
            
            # Try to decode as text first to see the format
            try:
                text_content = content.decode('utf-8', errors='ignore')
                
                # If it's mostly commas and binary data, treat as raw binary
                if text_content.count(',') > len(text_content) * 0.3:
                    return self.parse_binary_data(content)
                else:
                    return self.parse_text_data(text_content)
                    
            except UnicodeDecodeError:
                return self.parse_binary_data(content)
                
        except Exception as e:
            return None

    def parse_binary_data(self, content):
        """Parse binary data directly."""
        try:
            data = np.frombuffer(content, dtype=np.uint8)
            
            # Try common image sizes
            possible_sizes = [
                (512, 256, 3),  # 512x256 RGB
                (640, 480, 3),  # 640x480 RGB
                (800, 600, 3),  # 800x600 RGB
                (1024, 768, 3), # 1024x768 RGB
                (256, 256, 3),  # 256x256 RGB
                (512, 512, 3),  # 512x512 RGB
            ]
            
            for width, height, channels in possible_sizes:
                expected_size = width * height * channels
                if len(data) >= expected_size:
                    try:
                        image_data = data[:expected_size]
                        image = image_data.reshape((height, width, channels))
                        
                        # Check if it looks like a valid image
                        if np.mean(image) > 0 and np.std(image) > 0:
                            return image
                    except:
                        continue
            
            # If no standard size works, try to find a reasonable square size
            total_pixels = len(data) // 3  # Assuming RGB
            side = int(np.sqrt(total_pixels))
            if side > 0:
                expected_size = side * side * 3
                if len(data) >= expected_size:
                    image_data = data[:expected_size]
                    image = image_data.reshape((side, side, 3))
                    return image
            
            return None
            
        except Exception as e:
            return None

    def parse_text_data(self, text_content):
        """Parse text-based data."""
        try:
            lines = text_content.split('\n')
            if len(lines) < 2:
                return None
            
            # Try to extract numeric values
            values = []
            for line in lines:
                for item in line.split(','):
                    item = item.strip()
                    if item and item.replace('-', '').replace('.', '').isdigit():
                        try:
                            values.append(int(float(item)))
                        except:
                            continue
            
            if len(values) == 0:
                return None
            
            # Try to reshape into image
            possible_sizes = [
                (512, 256, 3),
                (640, 480, 3),
                (800, 600, 3),
                (1024, 768, 3),
                (256, 256, 3),
                (512, 512, 3),
            ]
            
            for width, height, channels in possible_sizes:
                expected_size = width * height * channels
                if len(values) >= expected_size:
                    try:
                        image_data = np.array(values[:expected_size], dtype=np.uint8)
                        image = image_data.reshape((height, width, channels))
                        
                        # Check if it looks like a valid image
                        if np.mean(image) > 0 and np.std(image) > 0:
                            return image
                    except:
                        continue
            
            return None
            
        except Exception as e:
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
        """Start live camera feed with YOLO detection."""
        print("🎥 Starting live camera feed with YOLO detection...")
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
                rgb_file = "current_rgb.txt"
                
                if os.path.exists(rgb_file):
                    file_time = os.path.getmtime(rgb_file)
                    if file_time > self.last_rgb_time:
                        rgb_image = self.read_binary_camera_data(rgb_file)
                        self.last_rgb_time = file_time
                    else:
                        rgb_image = None
                else:
                    rgb_image = None
                
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
                    
                    # Resize for display
                    height, width = display_img.shape[:2]
                    if width > 800 or height > 600:
                        scale = min(800/width, 600/height)
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        display_img = cv2.resize(display_img, (new_width, new_height))
                    
                    # Display the image
                    cv2.imshow('CoppeliaSim Live Camera (YOLO Detection)', cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        # Save current frame
                        timestamp = int(time.time())
                        cv2.imwrite(f'live_frame_yolo_{timestamp}.jpg', cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))
                        print(f"💾 Saved frame: live_frame_yolo_{timestamp}.jpg")
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
    print("🤖 Live CoppeliaSim Camera with YOLO Detection")
    print("=" * 55)
    
    # Create monitor
    monitor = LiveCameraYOLODetection()
    
    try:
        monitor.start()
    except KeyboardInterrupt:
        print("\n🛑 Stopping live feed...")
    finally:
        monitor.stop()

if __name__ == "__main__":
    main()




