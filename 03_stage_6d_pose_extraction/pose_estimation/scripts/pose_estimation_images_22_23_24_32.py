#!/usr/bin/env python3
"""
🎯 6D Pose Estimation for Images 22, 23, 24, 32
===============================================
Run object detection and 6D pose estimation on specific images.
"""

import cv2
import numpy as np
import json
import trimesh
from ultralytics import YOLO
import os
import time

class PoseEstimationForSpecificImages:
    """Run 6D pose estimation on specific images."""
    
    def __init__(self):
        """Initialize the pose estimation system."""
        print("🎯 Initializing 6D Pose Estimation System...")
        
        # Load YOLO model
        self.model = YOLO('../../coppelia_sim_results/weights/best.pt')
        
        # Load camera calibration
        with open('../calibration/coppelia_camera_calibration.json', 'r') as f:
            calibration = json.load(f)
        
        self.camera_matrix = np.array(calibration['camera_matrix'])
        self.dist_coeffs = np.array(calibration['dist_coeffs'])
        
        # Load 3D models
        self.models = {}
        model_paths = {
            'master_chef_can': '../models/master_chef_can.obj',
            'cracker_box': '../models/cracker_box.obj',
            'mug': '../models/mug.obj',
            'banana': '../models/banana.obj',
            'mustard_bottle': '../models/mustard_bottle.obj'
        }
        
        for name, path in model_paths.items():
            if os.path.exists(path):
                mesh = trimesh.load(path)
                self.models[name] = mesh
                print(f"  📦 {name}: {len(mesh.vertices)} vertices")
            else:
                print(f"  ❌ Missing model: {path}")
        
        print("✅ 6D Pose Estimation System initialized!")
    
    def load_rgb_image(self, rgb_file):
        """Load RGB image from file."""
        try:
            if rgb_file.endswith('.txt'):
                with open(rgb_file, 'rb') as f:
                    rgb_data = np.frombuffer(f.read(), dtype=np.uint8)
                rgb_data = rgb_data.reshape(480, 640, 3)
                return rgb_data
            else:
                return cv2.imread(rgb_file)
        except Exception as e:
            print(f"❌ Error loading RGB image: {e}")
            return None
    
    def load_depth_image(self, depth_file):
        """Load depth image from file."""
        try:
            if depth_file.endswith('.txt'):
                with open(depth_file, 'r') as f:
                    content = f.read().strip()
                depth_values = [float(x) for x in content.split(',') if x.strip()]
                depth_data = np.array(depth_values, dtype=np.float32)
                depth_data = depth_data.reshape(480, 640)
                return depth_data
            else:
                depth_data = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
                return depth_data.astype(np.float32) / 1000.0
        except Exception as e:
            print(f"❌ Error loading depth image: {e}")
            return None
    
    def detect_objects(self, rgb_image):
        """Detect objects in RGB image."""
        results = self.model(rgb_image, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    name = result.names[cls]
                    
                    detections.append({
                        'name': name,
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class_id': cls
                    })
        
        return detections
    
    def estimate_6d_pose(self, rgb_image, depth_image, detection):
        """Estimate 6D pose for a single object."""
        name = detection['name']
        bbox = detection['bbox']
        
        if name not in self.models:
            print(f"  ❌ No 3D model for {name}")
            return None
        
        # Get 3D model points
        model = self.models[name]
        model_points = np.array(model.vertices)
        
        # Get 2D bounding box
        x1, y1, x2, y2 = bbox
        x1, x2 = int(x1), int(x2)
        y1, y2 = int(y1), int(y2)
        
        # Sample points from depth image within bbox
        sample_points = []
        sample_colors = []
        
        for v in range(y1, y2, 5):  # Sample every 5th pixel
            for u in range(x1, x2, 5):
                if 0 <= v < depth_image.shape[0] and 0 <= u < depth_image.shape[1]:
                    depth = depth_image[v, u]
                    if depth > 0.01 and depth < 3.0:
                        # Convert to 3D
                        x = (u - self.camera_matrix[0, 2]) * depth / self.camera_matrix[0, 0]
                        y = (v - self.camera_matrix[1, 2]) * depth / self.camera_matrix[1, 1]
                        z = depth
                        sample_points.append([x, y, z])
                        sample_colors.append(rgb_image[v, u])
        
        if len(sample_points) < 10:
            print(f"  ❌ Not enough depth points for {name}")
            return None
        
        sample_points = np.array(sample_points)
        
        # Use PnP to estimate pose
        try:
            # Find correspondences (simplified - using closest points)
            # In practice, you'd use feature matching
            if len(model_points) > len(sample_points):
                indices = np.random.choice(len(model_points), len(sample_points), replace=False)
                model_points_subset = model_points[indices]
            else:
                model_points_subset = model_points
            
            # Ensure we have enough points
            min_points = min(len(model_points_subset), len(sample_points))
            if min_points < 4:
                print(f"  ❌ Not enough points for PnP: {min_points}")
                return None
            
            model_points_subset = model_points_subset[:min_points]
            sample_points_subset = sample_points[:min_points]
            
            # Solve PnP
            success, rotation_vec, translation_vec = cv2.solvePnP(
                model_points_subset, 
                sample_points_subset[:, :2],  # Use only X,Y for 2D projection
                self.camera_matrix, 
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                # Convert rotation vector to rotation matrix
                rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
                
                # Create transformation matrix
                transform = np.eye(4)
                transform[:3, :3] = rotation_matrix
                transform[:3, 3] = translation_vec.flatten()
                
                return {
                    'name': name,
                    'translation': translation_vec.flatten(),
                    'rotation': rotation_matrix,
                    'transform': transform,
                    'confidence': detection['confidence'],
                    'sample_points': len(sample_points)
                }
            else:
                print(f"  ❌ PnP failed for {name}")
                return None
                
        except Exception as e:
            print(f"  ❌ Error in pose estimation for {name}: {e}")
            return None
    
    def process_image(self, image_id):
        """Process a single image for 6D pose estimation."""
        print(f"\n📸 Processing image {image_id}...")
        
        # Load images
        rgb_file = f"/tmp/auto_kinect_{image_id}_rgb.txt"
        depth_file = f"/tmp/auto_kinect_{image_id}_depth.txt"
        
        if not os.path.exists(rgb_file) or not os.path.exists(depth_file):
            print(f"❌ Missing files for image {image_id}")
            return None
        
        rgb_image = self.load_rgb_image(rgb_file)
        depth_image = self.load_depth_image(depth_file)
        
        if rgb_image is None or depth_image is None:
            print(f"❌ Failed to load images for {image_id}")
            return None
        
        # Detect objects
        detections = self.detect_objects(rgb_image)
        print(f"🎯 Detected {len(detections)} objects")
        
        # Show detections
        for i, detection in enumerate(detections):
            print(f"  {i+1}. {detection['name']}: {detection['confidence']:.2f}")
        
        # Estimate 6D pose for each detection
        pose_results = []
        for detection in detections:
            print(f"\n🔄 Estimating 6D pose for {detection['name']}...")
            pose_result = self.estimate_6d_pose(rgb_image, depth_image, detection)
            
            if pose_result:
                pose_results.append(pose_result)
                print(f"  ✅ Pose estimated successfully!")
                print(f"     Translation: [{pose_result['translation'][0]:.3f}, {pose_result['translation'][1]:.3f}, {pose_result['translation'][2]:.3f}]")
                print(f"     Sample points: {pose_result['sample_points']}")
            else:
                print(f"  ❌ Failed to estimate pose for {detection['name']}")
        
        return {
            'image_id': image_id,
            'detections': detections,
            'pose_results': pose_results
        }
    
    def run_pose_estimation_on_images(self, image_ids=[22, 23, 24, 32]):
        """Run 6D pose estimation on specific images."""
        print("🚀 Running 6D Pose Estimation on Images 22, 23, 24, 32...")
        
        all_results = {}
        
        for image_id in image_ids:
            result = self.process_image(image_id)
            if result:
                all_results[image_id] = result
        
        # Print summary
        print("\n" + "="*60)
        print("🎯 6D POSE ESTIMATION SUMMARY")
        print("="*60)
        
        for image_id, result in all_results.items():
            print(f"\n📸 Image {image_id}:")
            print(f"  🎯 Objects Detected: {len(result['detections'])}")
            print(f"  ✅ Poses Estimated: {len(result['pose_results'])}")
            
            for pose_result in result['pose_results']:
                name = pose_result['name']
                trans = pose_result['translation']
                conf = pose_result['confidence']
                points = pose_result['sample_points']
                
                print(f"    • {name}:")
                print(f"      - Confidence: {conf:.2f}")
                print(f"      - Translation: [{trans[0]:.3f}, {trans[1]:.3f}, {trans[2]:.3f}]")
                print(f"      - Sample Points: {points}")
        
        return all_results

def main():
    """Main function."""
    print("🎯 6D Pose Estimation for Images 22, 23, 24, 32")
    print("=" * 50)
    
    # Initialize system
    system = PoseEstimationForSpecificImages()
    
    # Run pose estimation
    results = system.run_pose_estimation_on_images()
    
    print(f"\n🎉 Completed 6D pose estimation on {len(results)} images!")

if __name__ == "__main__":
    main()




