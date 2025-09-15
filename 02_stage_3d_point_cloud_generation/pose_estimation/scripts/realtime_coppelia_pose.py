#!/usr/bin/env python3
"""
Real-time 6D Pose Estimation for CoppeliaSim
Uses file-based communication for live pose estimation
"""

import cv2
import numpy as np
import json
import time
import sys
import os
import glob
from pathlib import Path

# Add YOLO path
sys.path.append('../../coppelia_sim_dataset')
from ultralytics import YOLO

class RealtimeCoppeliaPose:
    def __init__(self, model_path, calibration_path, models_dir):
        """Initialize real-time CoppeliaSim pose estimator"""
        
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
        
        # File monitoring
        self.last_processed_file = None
        self.monitoring_dir = "/tmp"  # CoppeliaSim saves here
        
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
    
    def find_latest_coppelia_image(self):
        """Find the latest image captured by CoppeliaSim"""
        # Look for auto_kinect_*_rgb.txt files in /tmp and current directory
        patterns = [
            "/tmp/auto_kinect_*_rgb.txt",
            "auto_kinect_*_rgb.txt",
            "/tmp/coppelia_*_rgb.txt",
            "coppelia_*_rgb.txt"
        ]
        
        latest_file = None
        latest_time = 0
        
        for pattern in patterns:
            files = glob.glob(pattern)
            for file in files:
                file_time = os.path.getmtime(file)
                if file_time > latest_time:
                    latest_time = file_time
                    latest_file = file
        
        return latest_file
    
    def process_coppelia_image(self, file_path):
        """Process a CoppeliaSim image file"""
        try:
            # Read the raw image data
            with open(file_path, 'r') as f:
                data = f.read().strip()
            
            # Convert string data to bytes
            if data.startswith('[') and data.endswith(']'):
                # Remove brackets and split by commas
                data = data[1:-1]
                values = [int(x.strip()) for x in data.split(',') if x.strip()]
                image_data = np.array(values, dtype=np.uint8)
            else:
                # Try direct conversion
                image_data = np.frombuffer(data.encode(), dtype=np.uint8)
            
            # Reshape based on expected size
            if len(image_data) == 921600:  # 640x480x3
                image = image_data.reshape(480, 640, 3)
            elif len(image_data) == 9216:  # 64x48x3
                image = image_data.reshape(48, 64, 3)
                # Resize to standard size
                image = cv2.resize(image, (640, 480))
            else:
                print(f"⚠️ Unexpected image size: {len(image_data)}")
                return None
            
            # Convert from RGB to BGR (OpenCV format)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            return image
            
        except Exception as e:
            print(f"❌ Error processing image file {file_path}: {e}")
            return None
    
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
    
    def run_realtime_monitoring(self):
        """Run real-time monitoring of CoppeliaSim images"""
        print("🎥 Starting Real-time CoppeliaSim 6D Pose Estimation...")
        print("📹 Press 'q' to quit, 's' to save current frame")
        print("💡 Run your CoppeliaSim capture script to see real-time poses!")
        
        frame_count = 0
        last_processed = None
        
        while True:
            # Find latest CoppeliaSim image
            latest_file = self.find_latest_coppelia_image()
            
            if latest_file and latest_file != last_processed:
                # Process new image
                frame = self.process_coppelia_image(latest_file)
                
                if frame is not None:
                    # Detect objects and estimate poses
                    detections = self.detect_objects_realtime(frame)
                    
                    poses = []
                    for detection in detections:
                        pose = self.estimate_pose_realtime(frame, detection)
                        if pose:
                            poses.append(pose)
                    
                    # Update FPS
                    self.update_fps()
                    
                    # Visualize results
                    result_frame = self.visualize_realtime(frame, poses)
                    
                    # Display frame
                    cv2.imshow('Real-time CoppeliaSim 6D Pose Estimation', result_frame)
                    
                    # Save frame for reference
                    timestamp = int(time.time())
                    save_path = f"../results/realtime_coppelia_{timestamp}.jpg"
                    cv2.imwrite(save_path, result_frame)
                    
                    last_processed = latest_file
                    frame_count += 1
                    
                    print(f"📸 Processed frame {frame_count}: {len(poses)} objects detected")
            
            # Handle key presses
            key = cv2.waitKey(100) & 0xFF  # 100ms delay for file monitoring
            if key == ord('q'):
                print("🛑 Stopping real-time pose estimation...")
                break
            elif key == ord('s'):
                if 'result_frame' in locals():
                    timestamp = int(time.time())
                    save_path = f"../results/realtime_coppelia_saved_{timestamp}.jpg"
                    cv2.imwrite(save_path, result_frame)
                    print(f"📸 Saved frame to: {save_path}")
        
        # Cleanup
        cv2.destroyAllWindows()
        print(f"✅ Processed {frame_count} frames")
        print(f"📊 Average FPS: {self.avg_fps:.1f}")

def main():
    """Main function for real-time CoppeliaSim pose estimation"""
    # Paths
    model_path = "../../coppelia_sim_dataset/runs/detect/train/weights/best.pt"
    calibration_path = "../calibration/coppelia_camera_calibration.json"
    models_dir = "../models"
    
    # Initialize real-time pose estimator
    pose_estimator = RealtimeCoppeliaPose(model_path, calibration_path, models_dir)
    
    print("🎯 Real-time CoppeliaSim 6D Pose Estimation")
    print("=" * 50)
    print("📋 Instructions:")
    print("1. Start CoppeliaSim with your scene")
    print("2. Run your capture script (e.g., auto_kinect_capture_increment.lua)")
    print("3. This system will automatically detect new images and estimate poses")
    print("4. Press 'q' to quit, 's' to save current frame")
    print()
    
    # Start real-time monitoring
    pose_estimator.run_realtime_monitoring()

if __name__ == "__main__":
    main()




