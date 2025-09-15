#!/usr/bin/env python3
"""
6D Pose Estimation for CoppeliaSim Objects
Combines YOLO object detection with PnP pose estimation
"""

import cv2
import numpy as np
import json
import trimesh
from pathlib import Path
import sys
import os

# Add YOLO path
sys.path.append('../../coppelia_sim_dataset')
from ultralytics import YOLO

class PoseEstimator:
    def __init__(self, model_path, calibration_path, models_dir):
        """
        Initialize 6D pose estimator
        
        Args:
            model_path: Path to trained YOLO model
            calibration_path: Path to camera calibration file
            models_dir: Directory containing 3D models
        """
        # Class mapping (define first)
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
    
    def detect_objects(self, image):
        """Detect objects using YOLO"""
        results = self.yolo_model(image, conf=0.3)
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
                        # Fallback to our class names
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
    
    def estimate_pose(self, image, detection):
        """Estimate 6D pose for a detected object"""
        class_name = detection['class_name']
        bbox = detection['bbox']
        
        if class_name not in self.models:
            return None
        
        # Extract region of interest
        x1, y1, x2, y2 = map(int, bbox)
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return None
        
        # Get 3D model vertices
        model_vertices = self.models[class_name]['vertices']
        
        # For simplicity, we'll use a basic approach:
        # 1. Find keypoints in the ROI
        # 2. Match with 3D model keypoints
        # 3. Solve PnP
        
        # Convert to grayscale for keypoint detection
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Detect keypoints (using SIFT)
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray_roi, None)
        
        if len(keypoints) < 4:
            return None
        
        # For now, use a simplified approach with bounding box center
        # In a real implementation, you'd match keypoints properly
        
        # Get bounding box center
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Estimate depth based on bounding box size (simplified)
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        bbox_size = max(bbox_width, bbox_height)
        
        # Rough depth estimation (this is simplified)
        # In reality, you'd use stereo vision or depth sensors
        estimated_depth = 0.5  # meters (simplified)
        
        # Create 2D-3D correspondences
        # For now, use bounding box corners as 2D points
        bbox_2d = np.array([
            [x1, y1], [x2, y1], [x2, y2], [x1, y2]
        ], dtype=np.float32)
        
        # Create corresponding 3D points (simplified)
        # In reality, these would come from the 3D model
        bbox_3d = np.array([
            [-0.05, -0.05, estimated_depth],
            [0.05, -0.05, estimated_depth],
            [0.05, 0.05, estimated_depth],
            [-0.05, 0.05, estimated_depth]
        ], dtype=np.float32)
        
        # Solve PnP
        success, rvec, tvec = cv2.solvePnP(
            bbox_3d, bbox_2d, self.camera_matrix, self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success:
            # Convert rotation vector to rotation matrix
            rmat, _ = cv2.Rodrigues(rvec)
            
            pose = {
                'translation': tvec.flatten(),
                'rotation_matrix': rmat,
                'rotation_vector': rvec.flatten(),
                'bbox': bbox,
                'class_name': class_name,
                'confidence': detection['confidence']
            }
            
            return pose
        
        return None
    
    def process_image(self, image):
        """Process image and estimate poses for all detected objects"""
        # Detect objects
        detections = self.detect_objects(image)
        
        poses = []
        for detection in detections:
            pose = self.estimate_pose(image, detection)
            if pose:
                poses.append(pose)
        
        return poses
    
    def visualize_poses(self, image, poses):
        """Visualize 6D poses on image"""
        result_image = image.copy()
        
        for pose in poses:
            bbox = pose['bbox']
            class_name = pose['class_name']
            confidence = pose['confidence']
            translation = pose['translation']
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(result_image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw pose information
            pose_text = f"T: [{translation[0]:.2f}, {translation[1]:.2f}, {translation[2]:.2f}]"
            cv2.putText(result_image, pose_text, (x1, y2+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        return result_image

def main():
    """Main function to test 6D pose estimation"""
    # Paths
    model_path = "../../coppelia_sim_dataset/runs/detect/train/weights/best.pt"
    calibration_path = "../calibration/coppelia_camera_calibration.json"
    models_dir = "../models"
    
    # Initialize pose estimator
    pose_estimator = PoseEstimator(model_path, calibration_path, models_dir)
    
    # Test with a sample image
    test_image_path = "../../enhanced_debug_kinect_rgb.jpg"
    
    if os.path.exists(test_image_path):
        print(f"🎯 Testing with image: {test_image_path}")
        
        # Load image
        image = cv2.imread(test_image_path)
        if image is not None:
            # Process image
            poses = pose_estimator.process_image(image)
            
            print(f"✅ Detected {len(poses)} objects with poses:")
            for i, pose in enumerate(poses):
                print(f"  {i+1}. {pose['class_name']}: T={pose['translation']}")
            
            # Visualize results
            result_image = pose_estimator.visualize_poses(image, poses)
            
            # Save result
            output_path = "../results/pose_estimation_result.jpg"
            cv2.imwrite(output_path, result_image)
            print(f"📸 Result saved to: {output_path}")
            
        else:
            print(f"❌ Failed to load image: {test_image_path}")
    else:
        print(f"⚠️ Test image not found: {test_image_path}")
        print("💡 You can test with any of your CoppeliaSim images")

if __name__ == "__main__":
    main()
