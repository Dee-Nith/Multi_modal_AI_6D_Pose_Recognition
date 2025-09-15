#!/usr/bin/env python3
"""
Real-time 6D Pose Estimation Demo
Simplified version for testing with camera or video files
"""

import cv2
import numpy as np
import json
import time
import sys
import os
from pathlib import Path

# Add YOLO path
sys.path.append('../../coppelia_sim_dataset')
from ultralytics import YOLO

class RealtimePoseDemo:
    def __init__(self, model_path, calibration_path):
        """Initialize real-time pose demo"""
        
        # Load YOLO model
        self.yolo_model = YOLO(model_path)
        print(f"✅ Loaded YOLO model: {model_path}")
        
        # Load camera calibration
        self.camera_matrix, self.dist_coeffs = self.load_calibration(calibration_path)
        print(f"✅ Loaded camera calibration: {calibration_path}")
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.avg_fps = 0
        
    def load_calibration(self, calibration_path):
        """Load camera calibration from JSON file"""
        with open(calibration_path, 'r') as f:
            data = json.load(f)
        
        camera_matrix = np.array(data['camera_matrix'])
        dist_coeffs = np.array(data['dist_coeffs'])
        
        return camera_matrix, dist_coeffs
    
    def detect_and_estimate_pose(self, frame):
        """Detect objects and estimate poses in real-time"""
        # Detect objects
        results = self.yolo_model(frame, conf=0.3, verbose=False)
        poses = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Get class
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Get class name
                    if hasattr(result, 'names') and class_id < len(result.names):
                        class_name = result.names[class_id]
                    else:
                        class_name = f"object_{class_id}"
                    
                    # Get confidence
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    # Simple pose estimation (for demo)
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    
                    # Estimate depth (simplified)
                    focal_length = self.camera_matrix[0, 0]
                    estimated_depth = 0.5  # Default 50cm
                    
                    # Create pose info
                    pose = {
                        'bbox': [x1, y1, x2, y2],
                        'class_name': class_name,
                        'confidence': confidence,
                        'center': [center_x, center_y],
                        'size': [bbox_width, bbox_height],
                        'estimated_depth': estimated_depth
                    }
                    
                    poses.append(pose)
        
        return poses
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.avg_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def visualize_poses(self, frame, poses):
        """Visualize poses with real-time info"""
        result_frame = frame.copy()
        
        for i, pose in enumerate(poses):
            bbox = pose['bbox']
            class_name = pose['class_name']
            confidence = pose['confidence']
            center = pose['center']
            size = pose['size']
            depth = pose['estimated_depth']
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw center point
            center_x, center_y = map(int, center)
            cv2.circle(result_frame, (center_x, center_y), 5, (255, 0, 0), -1)
            
            # Draw labels
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(result_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw pose info
            pose_text = f"Center: ({center_x}, {center_y})"
            cv2.putText(result_frame, pose_text, (x1, y2+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            size_text = f"Size: {size[0]:.0f}x{size[1]:.0f}px"
            cv2.putText(result_frame, size_text, (x1, y2+35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
            
            depth_text = f"Depth: {depth:.2f}m"
            cv2.putText(result_frame, depth_text, (x1, y2+50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Performance overlay
        cv2.putText(result_frame, f"FPS: {self.avg_fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(result_frame, f"Objects: {len(poses)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Instructions
        cv2.putText(result_frame, "Press 'q' to quit, 's' to save", (10, frame.shape[0]-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result_frame
    
    def run_demo(self, video_source=0):
        """Run real-time pose estimation demo"""
        print("🎥 Starting Real-time 6D Pose Estimation Demo...")
        print("📹 Press 'q' to quit, 's' to save current frame")
        
        # Open video source
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open video source {video_source}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Real-time pose estimation demo started!")
        print("🎯 Show objects to your camera to see real-time detection!")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            # Process frame
            poses = self.detect_and_estimate_pose(frame)
            
            # Update FPS
            self.update_fps()
            
            # Visualize results
            result_frame = self.visualize_poses(frame, poses)
            
            # Display frame
            cv2.imshow('Real-time 6D Pose Estimation Demo', result_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🛑 Stopping real-time pose estimation demo...")
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = int(time.time())
                save_path = f"../results/realtime_demo_frame_{timestamp}.jpg"
                cv2.imwrite(save_path, result_frame)
                print(f"📸 Saved frame to: {save_path}")
            
            frame_count += 1
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print(f"✅ Processed {frame_count} frames")
        print(f"📊 Average FPS: {self.avg_fps:.1f}")

def main():
    """Main function for real-time pose demo"""
    # Paths
    model_path = "../../coppelia_sim_dataset/runs/detect/train/weights/best.pt"
    calibration_path = "../calibration/coppelia_camera_calibration.json"
    
    # Initialize demo
    demo = RealtimePoseDemo(model_path, calibration_path)
    
    print("🎯 Real-time 6D Pose Estimation Demo")
    print("=" * 40)
    print("1. Use your computer's camera")
    print("2. Use a video file")
    print("3. Test with CoppeliaSim image")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        print("📹 Using computer camera...")
        demo.run_demo(0)  # Default camera
    elif choice == "2":
        video_file = input("Enter video file path: ").strip()
        print(f"📹 Using video file: {video_file}")
        demo.run_demo(video_file)
    elif choice == "3":
        print("📹 Testing with CoppeliaSim image...")
        # For testing, we'll use a simple approach
        test_image = cv2.imread("../../enhanced_debug_kinect_rgb.jpg")
        if test_image is not None:
            poses = demo.detect_and_estimate_pose(test_image)
            result_frame = demo.visualize_poses(test_image, poses)
            
            # Save result
            save_path = "../results/realtime_demo_test.jpg"
            cv2.imwrite(save_path, result_frame)
            print(f"📸 Test result saved to: {save_path}")
            
            # Show result
            cv2.imshow('Test Result', result_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("❌ Could not load test image")
    else:
        print("Using default camera")
        demo.run_demo(0)

if __name__ == "__main__":
    main()




