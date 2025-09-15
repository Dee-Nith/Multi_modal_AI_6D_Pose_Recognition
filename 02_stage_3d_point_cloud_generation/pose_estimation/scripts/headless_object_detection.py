#!/usr/bin/env python3
"""
Headless Object Detection and Training System
Automatically captures camera data, runs YOLO detection, and improves the model
"""

import numpy as np
import cv2
import time
import os
import json
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime

class HeadlessObjectDetector:
    def __init__(self):
        self.yolo_model = None
        self.detection_history = []
        self.training_data = []
        self.last_capture_time = 0
        self.capture_interval = 2.0  # Capture every 2 seconds
        
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
    
    def save_detection_data(self, image, detections, timestamp):
        """Save detection data for training."""
        if len(detections) == 0:
            return
        
        # Create training data directory
        training_dir = Path("auto_training_data")
        training_dir.mkdir(exist_ok=True)
        
        # Save image
        image_filename = f"frame_{timestamp}.jpg"
        image_path = training_dir / image_filename
        cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        # Save detection annotations
        annotation_filename = f"frame_{timestamp}.json"
        annotation_path = training_dir / annotation_filename
        
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
        """Analyze detection performance and suggest improvements."""
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
        
        print(f"📊 Detection Analysis:")
        print(f"   - Total detections: {len(detections)}")
        print(f"   - Average confidence: {avg_confidence:.3f}")
        print(f"   - Classes detected: {list(class_counts.keys())}")
        
        # Suggest improvements
        if avg_confidence < 0.5:
            print("⚠️ Low confidence detections - consider retraining with more data")
        
        if len(class_counts) < 2:
            print("⚠️ Limited class diversity - consider adding more object types")
        
        return {
            'total_detections': len(detections),
            'avg_confidence': avg_confidence,
            'class_counts': class_counts
        }
    
    def start_headless_detection(self):
        """Start headless object detection and data collection."""
        print("🤖 Starting Headless Object Detection and Training System")
        print("=" * 60)
        print("📋 System will:")
        print("   - Automatically capture camera data every 2 seconds")
        print("   - Run YOLO object detection")
        print("   - Save detection data for training")
        print("   - Analyze detection performance")
        print("   - Provide training recommendations")
        print("\n⏳ Starting detection loop...")
        
        frame_count = 0
        total_detections = 0
        
        try:
            while True:
                current_time = time.time()
                
                # Check if it's time to capture
                if current_time - self.last_capture_time >= self.capture_interval:
                    # Read camera data
                    rgb_file = "current_rgb.txt"
                    
                    if os.path.exists(rgb_file):
                        file_time = os.path.getmtime(rgb_file)
                        
                        # Only process if file is recent (within last 10 seconds)
                        if current_time - file_time < 10:
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
                                    performance = self.analyze_detection_performance(detections)
                                    
                                    # Store in history
                                    self.detection_history.append({
                                        'frame': frame_count,
                                        'timestamp': timestamp,
                                        'detections': detections,
                                        'performance': performance
                                    })
                                else:
                                    print("❌ No objects detected")
                                
                                self.last_capture_time = current_time
                            else:
                                print("❌ Failed to read camera data")
                        else:
                            print("⏳ Waiting for fresh camera data...")
                    else:
                        print("⏳ Waiting for camera data file...")
                
                # Small delay
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\n🛑 Detection stopped by user")
        except Exception as e:
            print(f"❌ Error in detection loop: {e}")
        finally:
            self.print_summary(frame_count, total_detections)
    
    def print_summary(self, frame_count, total_detections):
        """Print detection summary."""
        print("\n" + "=" * 60)
        print("📊 DETECTION SUMMARY")
        print("=" * 60)
        print(f"📸 Total frames processed: {frame_count}")
        print(f"🔍 Total objects detected: {total_detections}")
        print(f"📁 Training data saved to: auto_training_data/")
        
        if self.detection_history:
            # Calculate statistics
            confidences = []
            class_counts = {}
            
            for entry in self.detection_history:
                for det in entry['detections']:
                    confidences.append(det['confidence'])
                    class_name = det['class']
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            if confidences:
                print(f"📈 Average confidence: {np.mean(confidences):.3f}")
                print(f"📈 Best confidence: {np.max(confidences):.3f}")
                print(f"📈 Classes detected: {list(class_counts.keys())}")
        
        print("\n🚀 Next steps:")
        print("   1. Review saved training data in auto_training_data/")
        print("   2. Retrain YOLO model with new data if needed")
        print("   3. Adjust detection parameters based on performance")

def main():
    """Main function."""
    detector = HeadlessObjectDetector()
    detector.start_headless_detection()

if __name__ == "__main__":
    main()




