#!/usr/bin/env python3
"""
Real-time 6D Pose Estimation for CoppeliaSim
Processes live video feed and displays real-time 6D poses
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

class RealtimePoseEstimator:
    def __init__(self, model_path, calibration_path, models_dir):
        """Initialize real-time pose estimator"""
        
        # Class mapping
        self.class_names = ['banana', 'cracker_box', 'master_chef_can', 'mug', 'mustard_bottle']
        
        # Load YOLO model
        self.yolo_model = YOLO(model_path)
        print(f"✅ Loaded YOLO model: {model_path}")
        
        # Load camera calibration
        self.camera_matrix, self.dist_coeffs = self.load_calibration(calibration_path)
        print(f"✅ Loaded camera calibration: {calibration_path}")
        
        # Load 3D models
        self.models = self.load_3d_models(models_dir)
        print(f"✅ Loaded {len(self.models)} 3D models")
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.avg_fps = 0
        
        # Object tracking for smooth poses
        self.tracked_objects = {}
        self.track_id = 0
        
    def load_calibration(self, calibration_path):
        """Load camera calibration from JSON file"""
        with open(calibration_path, 'r') as f:
            data = json.load(f)
        
        camera_matrix = np.array(data['camera_matrix'])
        dist_coeffs = np.array(data['dist_coeffs'])
        
        return camera_matrix, dist_coeffs
    
    def load_3d_models(self, models_dir):
        """Load 3D models and extract keypoints"""
        models = {}
        
        for class_name in self.class_names:
            model_path = Path(models_dir) / f"{class_name}.obj"
            if model_path.exists():
                # Load mesh
                import trimesh
                mesh = trimesh.load(str(model_path))
                
                # Extract keypoints (simplified - using mesh vertices)
                vertices = np.array(mesh.vertices)
                
                # Normalize and scale vertices
                center = np.mean(vertices, axis=0)
                vertices = vertices - center
                
                # Scale to reasonable size (meters)
                max_dim = np.max(np.linalg.norm(vertices, axis=1))
                scale = 0.1 / max_dim  # 10cm max dimension
                vertices = vertices * scale
                
                models[class_name] = {
                    'mesh': mesh,
                    'vertices': vertices,
                    'center': center,
                    'scale': scale
                }
                
                print(f"  📦 {class_name}: {len(vertices)} vertices")
        
        return models
    
    def detect_objects_realtime(self, image):
        """Detect objects using YOLO with real-time optimization"""
        results = self.yolo_model(image, conf=0.3, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Get class
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Get class name from YOLO model
                    if hasattr(result, 'names') and class_id < len(result.names):
                        class_name = result.names[class_id]
                    else:
                        class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                    
                    # Get confidence
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'class_name': class_name,
                        'class_id': class_id,
                        'confidence': confidence
                    })
        
        return detections
    
    def estimate_pose_realtime(self, image, detection):
        """Estimate 6D pose optimized for real-time performance"""
        class_name = detection['class_name']
        bbox = detection['bbox']
        
        if class_name not in self.models:
            return None
        
        # Extract region of interest
        x1, y1, x2, y2 = map(int, bbox)
        
        # Fast depth estimation based on object size
        object_real_sizes = {
            'master_chef_can': 0.10,    # 10cm height
            'cracker_box': 0.16,        # 16cm height
            'mug': 0.12,                # 12cm height
            'banana': 0.20,             # 20cm length
            'mustard_bottle': 0.19      # 19cm height
        }
        
        if class_name in object_real_sizes:
            real_size = object_real_sizes[class_name]
            pixel_size = max(x2 - x1, y2 - y1)
            focal_length = self.camera_matrix[0, 0]  # fx
            estimated_depth = (real_size * focal_length) / pixel_size
        else:
            estimated_depth = 0.5  # Default depth
        
        # Simplified 3D-2D correspondences for speed
        half_size = object_real_sizes.get(class_name, 0.05) / 2
        
        bbox_3d = np.array([
            [-half_size, -half_size, estimated_depth],
            [half_size, -half_size, estimated_depth],
            [half_size, half_size, estimated_depth],
            [-half_size, half_size, estimated_depth]
        ], dtype=np.float32)
        
        bbox_2d = np.array([
            [x1, y2], [x2, y2], [x2, y1], [x1, y1]
        ], dtype=np.float32)
        
        # Fast PnP solve
        success, rvec, tvec = cv2.solvePnP(
            bbox_3d, bbox_2d, self.camera_matrix, self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success:
            # Convert rotation vector to rotation matrix
            rmat, _ = cv2.Rodrigues(rvec)
            
            # Convert to Euler angles
            euler_angles = self.rotation_matrix_to_euler_angles(rmat)
            
            pose = {
                'translation': tvec.flatten(),
                'rotation_matrix': rmat,
                'rotation_vector': rvec.flatten(),
                'euler_angles': euler_angles,
                'euler_degrees': np.degrees(euler_angles),
                'bbox': bbox,
                'class_name': class_name,
                'confidence': detection['confidence'],
                'distance': np.linalg.norm(tvec)
            }
            
            return pose
        
        return None
    
    def rotation_matrix_to_euler_angles(self, R):
        """Convert rotation matrix to Euler angles (roll, pitch, yaw)"""
        import math
        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        
        singular = sy < 1e-6
        
        if not singular:
            x = math.atan2(R[2,1], R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else:
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
        
        return np.array([x, y, z])
    
    def draw_3d_axes_realtime(self, image, rvec, tvec, length=0.05):
        """Draw 3D coordinate axes optimized for real-time"""
        # Define 3D points for axes
        axes_3d = np.array([
            [0, 0, 0],          # Origin
            [length, 0, 0],     # X-axis (red)
            [0, length, 0],     # Y-axis (green)
            [0, 0, length]      # Z-axis (blue)
        ], dtype=np.float32)
        
        # Project 3D points to 2D
        axes_2d, _ = cv2.projectPoints(axes_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        axes_2d = axes_2d.reshape(-1, 2).astype(int)
        
        origin = tuple(axes_2d[0])
        x_point = tuple(axes_2d[1])
        y_point = tuple(axes_2d[2])
        z_point = tuple(axes_2d[3])
        
        # Draw axes with different colors
        cv2.arrowedLine(image, origin, x_point, (0, 0, 255), 2, tipLength=0.3)  # X: Red
        cv2.arrowedLine(image, origin, y_point, (0, 255, 0), 2, tipLength=0.3)  # Y: Green
        cv2.arrowedLine(image, origin, z_point, (255, 0, 0), 2, tipLength=0.3)  # Z: Blue
        
        return origin
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.avg_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def process_frame_realtime(self, frame):
        """Process a single frame in real-time"""
        # Detect objects
        detections = self.detect_objects_realtime(frame)
        
        poses = []
        for detection in detections:
            pose = self.estimate_pose_realtime(frame, detection)
            if pose:
                poses.append(pose)
        
        # Update FPS
        self.update_fps()
        
        return poses
    
    def visualize_realtime(self, frame, poses):
        """Real-time visualization with performance info"""
        result_frame = frame.copy()
        
        # Draw poses
        for i, pose in enumerate(poses):
            bbox = pose['bbox']
            class_name = pose['class_name']
            confidence = pose['confidence']
            translation = pose['translation']
            euler_degrees = pose['euler_degrees']
            distance = pose['distance']
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw 3D coordinate axes
            rvec = pose['rotation_vector'].reshape(3, 1)
            tvec = translation.reshape(3, 1)
            self.draw_3d_axes_realtime(result_frame, rvec, tvec, length=0.05)
            
            # Compact labels for real-time display
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(result_frame, label, (x1, y1-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Position (compact)
            pos_text = f"P: [{translation[0]:.2f}, {translation[1]:.2f}, {translation[2]:.2f}]"
            cv2.putText(result_frame, pos_text, (x1, y1-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Rotation (compact)
            rot_text = f"R: [{euler_degrees[0]:.0f}°, {euler_degrees[1]:.0f}°, {euler_degrees[2]:.0f}°]"
            cv2.putText(result_frame, rot_text, (x1, y2+15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        
        # Performance overlay
        cv2.putText(result_frame, f"FPS: {self.avg_fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(result_frame, f"Objects: {len(poses)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Instructions
        cv2.putText(result_frame, "Press 'q' to quit, 's' to save", (10, frame.shape[0]-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result_frame
    
    def run_realtime(self, video_source=0):
        """Run real-time pose estimation"""
        print("🎥 Starting Real-time 6D Pose Estimation...")
        print("📹 Press 'q' to quit, 's' to save current frame")
        
        # Open video source
        if isinstance(video_source, str):
            cap = cv2.VideoCapture(video_source)
        else:
            cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open video source {video_source}")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Real-time pose estimation started!")
        print("🎯 Move objects in CoppeliaSim to see real-time 6D poses!")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            # Process frame
            poses = self.process_frame_realtime(frame)
            
            # Visualize results
            result_frame = self.visualize_realtime(frame, poses)
            
            # Display frame
            cv2.imshow('Real-time 6D Pose Estimation', result_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🛑 Stopping real-time pose estimation...")
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = int(time.time())
                save_path = f"../results/realtime_frame_{timestamp}.jpg"
                cv2.imwrite(save_path, result_frame)
                print(f"📸 Saved frame to: {save_path}")
            
            frame_count += 1
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print(f"✅ Processed {frame_count} frames")
        print(f"📊 Average FPS: {self.avg_fps:.1f}")

def main():
    """Main function for real-time pose estimation"""
    # Paths
    model_path = "../../coppelia_sim_dataset/runs/detect/train/weights/best.pt"
    calibration_path = "../calibration/coppelia_camera_calibration.json"
    models_dir = "../models"
    
    # Initialize real-time pose estimator
    pose_estimator = RealtimePoseEstimator(model_path, calibration_path, models_dir)
    
    # Run real-time estimation
    # You can specify different video sources:
    # - 0: Default camera
    # - "path/to/video.mp4": Video file
    # - "rtsp://...": IP camera stream
    
    print("🎯 Choose video source:")
    print("1. Default camera (0)")
    print("2. Video file")
    print("3. CoppeliaSim video file")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        video_source = 0
    elif choice == "2":
        video_file = input("Enter video file path: ").strip()
        video_source = video_file
    elif choice == "3":
        # Use a CoppeliaSim video file
        video_source = "../../enhanced_debug_kinect_rgb.jpg"  # For testing
        print("⚠️ Using test image for demonstration")
    else:
        video_source = 0
        print("Using default camera")
    
    # Start real-time pose estimation
    pose_estimator.run_realtime(video_source)

if __name__ == "__main__":
    main()




