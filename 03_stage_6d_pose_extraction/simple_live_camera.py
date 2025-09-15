#!/usr/bin/env python3
"""
Simple Live Camera with YOLO Detection
Continuously monitors camera data files from CoppeliaSim
"""

import numpy as np
import cv2
import time
import os
from ultralytics import YOLO

def read_camera_data(filename):
    """Read camera data and convert to image."""
    try:
        with open(filename, 'rb') as f:
            content = f.read()
        
        # Convert to numpy array
        data = np.frombuffer(content, dtype=np.uint8)
        
        # Try 512x256 first (most common for CoppeliaSim)
        if len(data) >= 512 * 256 * 3:
            image = data[:512*256*3].reshape((256, 512, 3))
            return image
        
        # Try other common sizes
        sizes = [(640, 480, 3), (800, 600, 3), (1024, 768, 3), (256, 256, 3)]
        for width, height, channels in sizes:
            expected_size = width * height * channels
            if len(data) >= expected_size:
                image = data[:expected_size].reshape((height, width, channels))
                return image
        
        return None
    except Exception as e:
        return None

def main():
    """Main function."""
    print("🎥 Simple Live Camera with YOLO Detection")
    print("=" * 45)
    
    # Load YOLO model
    try:
        model_path = "ycb_texture_training/ycb_texture_detector/weights/best.pt"
        if os.path.exists(model_path):
            yolo_model = YOLO(model_path)
            print(f"✅ Trained YOLO model loaded: {model_path}")
        else:
            yolo_model = YOLO('yolov8n.pt')
            print("⚠️ Using pre-trained YOLO model")
    except Exception as e:
        print(f"❌ Error loading YOLO model: {e}")
        yolo_model = None
    
    print("\n📋 Instructions:")
    print("   - Press 'q' to quit")
    print("   - Press 's' to save current frame")
    print("   - Press 'd' to toggle object detection")
    print("   - CoppeliaSim should be saving current_rgb.txt")
    print("\n⏳ Waiting for camera data...")
    
    show_detections = True
    last_file_time = 0
    frame_count = 0
    
    while True:
        try:
            # Check for camera data file
            rgb_file = "current_rgb.txt"
            
            if os.path.exists(rgb_file):
                file_time = os.path.getmtime(rgb_file)
                
                if file_time > last_file_time:
                    # Read and display image
                    image = read_camera_data(rgb_file)
                    
                    if image is not None:
                        display_img = image.copy()
                        
                        # Perform object detection
                        if show_detections and yolo_model is not None:
                            try:
                                results = yolo_model(image, verbose=False)
                                
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
                                            
                                            # Draw bounding box
                                            cv2.rectangle(display_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                                            
                                            # Draw label
                                            label = f"{class_name}: {conf:.2f}"
                                            cv2.putText(display_img, label, (int(x1), int(y1)-10), 
                                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                            
                                            print(f"🔍 Frame {frame_count}: {class_name} (conf: {conf:.2f})")
                            except Exception as e:
                                print(f"❌ Detection error: {e}")
                        
                        # Resize for display
                        height, width = display_img.shape[:2]
                        if width > 800 or height > 600:
                            scale = min(800/width, 600/height)
                            new_width = int(width * scale)
                            new_height = int(height * scale)
                            display_img = cv2.resize(display_img, (new_width, new_height))
                        
                        # Display image
                        cv2.imshow('CoppeliaSim Live Camera', cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))
                        
                        # Handle key presses
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                        elif key == ord('s'):
                            timestamp = int(time.time())
                            cv2.imwrite(f'live_frame_{timestamp}.jpg', cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))
                            print(f"💾 Saved: live_frame_{timestamp}.jpg")
                        elif key == ord('d'):
                            show_detections = not show_detections
                            print(f"🔍 Detection: {'ON' if show_detections else 'OFF'}")
                        
                        frame_count += 1
                        last_file_time = file_time
                        
                        # Print status
                        print(f"📸 Frame {frame_count} displayed ({width}x{height})")
            
            # Small delay
            time.sleep(0.1)  # 10 FPS
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(1)
    
    cv2.destroyAllWindows()
    print("\n🛑 Live camera stopped")

if __name__ == "__main__":
    main()




