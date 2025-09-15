#!/usr/bin/env python3
"""
Working Live Camera System
Combines file-based camera data with automatic triggering for near real-time operation
"""

import numpy as np
import cv2
import time
import os
import subprocess
from ultralytics import YOLO

class WorkingLiveCamera:
    def __init__(self):
        self.yolo_model = None
        self.last_capture_time = 0
        self.capture_interval = 1.0  # Capture every 1 second
        
        # Load the trained YOLO model
        try:
            model_path = "ycb_texture_training/ycb_texture_detector/weights/best.pt"
            if os.path.exists(model_path):
                self.yolo_model = YOLO(model_path)
                print(f"✅ Trained YOLO model loaded: {model_path}")
            else:
                self.yolo_model = YOLO('yolov8n.pt')
                print("⚠️ Using pre-trained YOLO model")
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            self.yolo_model = None
    
    def read_camera_data(self, filename):
        """Read camera data and convert to image."""
        try:
            with open(filename, 'rb') as f:
                content = f.read()
            
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
    
    def trigger_camera_capture(self):
        """Trigger camera capture in CoppeliaSim using a simple command."""
        try:
            # Create a simple Lua script to capture camera data
            lua_script = '''
-- Simple camera capture script
local rgbSensor = sim.getObject("./sensorRGB")
local depthSensor = sim.getObject("./sensorDepth")

if rgbSensor ~= -1 and depthSensor ~= -1 then
    local rgbImage = sim.getVisionSensorImg(rgbSensor)
    if rgbImage then
        local file = io.open("current_rgb.txt", "wb")
        if file then
            file:write(rgbImage)
            file:close()
            print("Camera data captured!")
        end
    end
end
'''
            
            # Save the script temporarily
            with open("temp_capture.lua", "w") as f:
                f.write(lua_script)
            
            print("📸 Triggering camera capture...")
            
            # The user will need to run this script in CoppeliaSim console
            print("💡 Please copy and paste this script into CoppeliaSim console:")
            print("=" * 50)
            print(lua_script)
            print("=" * 50)
            
            return True
            
        except Exception as e:
            print(f"❌ Error triggering capture: {e}")
            return False
    
    def detect_objects(self, image):
        """Detect objects using YOLO model."""
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
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(conf),
                            'class': class_name,
                            'class_id': int(cls)
                        })
            
            return detections
        except Exception as e:
            print(f"❌ Error in object detection: {e}")
            return []
    
    def start_live_detection(self):
        """Start live object detection system."""
        print("🤖 Starting Working Live Camera System")
        print("=" * 50)
        print("📋 This system will:")
        print("   - Monitor camera data files")
        print("   - Run YOLO object detection")
        print("   - Display detection results")
        print("   - Save training data automatically")
        print("\n💡 Instructions:")
        print("   1. Keep CoppeliaSim running")
        print("   2. Run the camera capture script when prompted")
        print("   3. Move objects in the scene to test detection")
        print("   4. Press Ctrl+C to stop")
        
        frame_count = 0
        total_detections = 0
        last_file_time = 0
        
        try:
            while True:
                current_time = time.time()
                
                # Check if it's time to capture
                if current_time - self.last_capture_time >= self.capture_interval:
                    # Check for camera data file
                    rgb_file = "current_rgb.txt"
                    
                    if os.path.exists(rgb_file):
                        file_time = os.path.getmtime(rgb_file)
                        
                        # Only process if file is recent (within last 30 seconds)
                        if current_time - file_time < 30:
                            if file_time > last_file_time:
                                image = self.read_camera_data(rgb_file)
                                
                                if image is not None:
                                    frame_count += 1
                                    timestamp = int(current_time)
                                    
                                    print(f"\n📸 Processing frame {frame_count} (timestamp: {timestamp})")
                                    
                                    # Run object detection
                                    detections = self.detect_objects(image)
                                    
                                    if detections:
                                        total_detections += len(detections)
                                        
                                        # Print detection results
                                        print(f"🔍 Detected {len(detections)} objects:")
                                        for det in detections:
                                            print(f"   - {det['class']} (conf: {det['confidence']:.3f})")
                                        
                                        # Save detection data for training
                                        self.save_detection_data(image, detections, timestamp)
                                        
                                        # Analyze performance
                                        self.analyze_detection_performance(detections)
                                    else:
                                        print("❌ No objects detected")
                                    
                                    last_file_time = file_time
                                    self.last_capture_time = current_time
                                else:
                                    print("❌ Failed to read camera data")
                            else:
                                print("⏳ Waiting for new camera data...")
                        else:
                            print("⏳ Camera data is old, waiting for fresh capture...")
                    else:
                        print("⏳ Waiting for camera data file...")
                        # Trigger capture if no file exists
                        if current_time - self.last_capture_time >= 5.0:
                            self.trigger_camera_capture()
                            self.last_capture_time = current_time
                
                # Small delay
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\n🛑 Live detection stopped by user")
        except Exception as e:
            print(f"❌ Error in live detection: {e}")
        finally:
            self.print_summary(frame_count, total_detections)
    
    def save_detection_data(self, image, detections, timestamp):
        """Save detection data for training."""
        if len(detections) == 0:
            return
        
        # Create training data directory
        training_dir = "auto_training_data"
        os.makedirs(training_dir, exist_ok=True)
        
        # Save image
        image_filename = f"frame_{timestamp}.jpg"
        image_path = os.path.join(training_dir, image_filename)
        cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        # Save detection annotations
        annotation_filename = f"frame_{timestamp}.json"
        annotation_path = os.path.join(training_dir, annotation_filename)
        
        import json
        annotation_data = {
            'timestamp': timestamp,
            'image_file': image_filename,
            'detections': detections,
            'image_size': [image.shape[1], image.shape[0]]  # [width, height]
        }
        
        with open(annotation_path, 'w') as f:
            json.dump(annotation_data, f, indent=2)
        
        print(f"💾 Saved training data: {image_filename} with {len(detections)} detections")
    
    def analyze_detection_performance(self, detections):
        """Analyze detection performance."""
        if not detections:
            return
        
        # Calculate average confidence
        confidences = [det['confidence'] for det in detections]
        avg_confidence = np.mean(confidences)
        
        # Count detections by class
        class_counts = {}
        for det in detections:
            class_name = det['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"📊 Performance: {len(detections)} objects, avg conf: {avg_confidence:.3f}")
        
        # Suggest improvements
        if avg_confidence < 0.5:
            print("⚠️ Low confidence - consider retraining with more data")
    
    def print_summary(self, frame_count, total_detections):
        """Print detection summary."""
        print("\n" + "=" * 50)
        print("📊 DETECTION SUMMARY")
        print("=" * 50)
        print(f"📸 Total frames processed: {frame_count}")
        print(f"🔍 Total objects detected: {total_detections}")
        print(f"📁 Training data saved to: auto_training_data/")
        print("\n🚀 Next steps:")
        print("   1. Review saved training data")
        print("   2. Retrain YOLO model if needed")
        print("   3. Move to 6D pose estimation")

def main():
    """Main function."""
    camera = WorkingLiveCamera()
    camera.start_live_detection()

if __name__ == "__main__":
    main()




