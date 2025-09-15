#!/usr/bin/env python3
"""
🎯 Simple 6D Pose Estimation for Images 22, 23, 24, 32
=====================================================
Run object detection and simplified 6D pose estimation.
"""

import cv2
import numpy as np
import json
import trimesh
from ultralytics import YOLO
import os
import time

class SimplePoseEstimation:
    """Run simplified 6D pose estimation on specific images."""
    
    def __init__(self):
        """Initialize the pose estimation system."""
        print("🎯 Initializing Simple 6D Pose Estimation System...")
        
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
        
        print("✅ Simple 6D Pose Estimation System initialized!")
    
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
    
    def estimate_simple_pose(self, rgb_image, depth_image, detection):
        """Estimate simplified 6D pose for a single object."""
        name = detection['name']
        bbox = detection['bbox']
        
        if name not in self.models:
            print(f"  ❌ No 3D model for {name}")
            return None
        
        # Get 2D bounding box
        x1, y1, x2, y2 = bbox
        x1, x2 = int(x1), int(x2)
        y1, y2 = int(y1), int(y2)
        
        # Calculate center of bounding box
        center_u = (x1 + x2) / 2
        center_v = (y1 + y2) / 2
        
        # Get depth at center
        if 0 <= int(center_v) < depth_image.shape[0] and 0 <= int(center_u) < depth_image.shape[1]:
            center_depth = depth_image[int(center_v), int(center_u)]
        else:
            # Use average depth in bbox
            depths = []
            for v in range(y1, y2, 5):
                for u in range(x1, x2, 5):
                    if 0 <= v < depth_image.shape[0] and 0 <= u < depth_image.shape[1]:
                        depth = depth_image[v, u]
                        if depth > 0.01 and depth < 3.0:
                            depths.append(depth)
            
            if len(depths) == 0:
                print(f"  ❌ No valid depth for {name}")
                return None
            
            center_depth = np.mean(depths)
        
        # Convert to 3D coordinates
        center_x = (center_u - self.camera_matrix[0, 2]) * center_depth / self.camera_matrix[0, 0]
        center_y = (center_v - self.camera_matrix[1, 2]) * center_depth / self.camera_matrix[1, 1]
        center_z = center_depth
        
        # Get model dimensions for orientation estimation
        model = self.models[name]
        model_vertices = np.array(model.vertices)
        
        # Calculate model center and dimensions
        model_center = np.mean(model_vertices, axis=0)
        model_dimensions = np.max(model_vertices, axis=0) - np.min(model_vertices, axis=0)
        
        # Simple orientation estimation (assuming upright objects)
        # This is a simplified approach - in practice you'd use more sophisticated methods
        rotation_matrix = np.eye(3)  # Identity matrix (no rotation)
        
        # Create transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = [center_x, center_y, center_z]
        
        # Calculate bounding box area for confidence
        bbox_area = (x2 - x1) * (y2 - y1)
        max_area = 640 * 480
        area_confidence = min(bbox_area / max_area, 1.0)
        
        return {
            'name': name,
            'translation': [center_x, center_y, center_z],
            'rotation': rotation_matrix,
            'transform': transform,
            'confidence': detection['confidence'],
            'bbox_center': [center_u, center_v],
            'depth': center_depth,
            'bbox_area': bbox_area,
            'area_confidence': area_confidence,
            'model_dimensions': model_dimensions
        }
    
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
            pose_result = self.estimate_simple_pose(rgb_image, depth_image, detection)
            
            if pose_result:
                pose_results.append(pose_result)
                print(f"  ✅ Pose estimated successfully!")
                trans = pose_result['translation']
                print(f"     Translation: [{trans[0]:.3f}, {trans[1]:.3f}, {trans[2]:.3f}]")
                print(f"     Depth: {pose_result['depth']:.3f}m")
                print(f"     BBox Area: {pose_result['bbox_area']} pixels")
            else:
                print(f"  ❌ Failed to estimate pose for {detection['name']}")
        
        return {
            'image_id': image_id,
            'detections': detections,
            'pose_results': pose_results
        }
    
    def run_pose_estimation_on_images(self, image_ids=[22, 23, 24, 32]):
        """Run 6D pose estimation on specific images."""
        print("🚀 Running Simple 6D Pose Estimation on Images 22, 23, 24, 32...")
        
        all_results = {}
        
        for image_id in image_ids:
            result = self.process_image(image_id)
            if result:
                all_results[image_id] = result
        
        # Print summary
        print("\n" + "="*70)
        print("🎯 SIMPLE 6D POSE ESTIMATION SUMMARY")
        print("="*70)
        
        for image_id, result in all_results.items():
            print(f"\n📸 Image {image_id}:")
            print(f"  🎯 Objects Detected: {len(result['detections'])}")
            print(f"  ✅ Poses Estimated: {len(result['pose_results'])}")
            
            for pose_result in result['pose_results']:
                name = pose_result['name']
                trans = pose_result['translation']
                conf = pose_result['confidence']
                depth = pose_result['depth']
                bbox_area = pose_result['bbox_area']
                
                print(f"    • {name}:")
                print(f"      - YOLO Confidence: {conf:.2f}")
                print(f"      - Translation: [{trans[0]:.3f}, {trans[1]:.3f}, {trans[2]:.3f}]")
                print(f"      - Depth: {depth:.3f}m")
                print(f"      - BBox Area: {bbox_area} pixels")
        
        return all_results

def main():
    """Main function."""
    print("🎯 Simple 6D Pose Estimation for Images 22, 23, 24, 32")
    print("=" * 60)
    
    # Initialize system
    system = SimplePoseEstimation()
    
    # Run pose estimation
    results = system.run_pose_estimation_on_images()
    
    print(f"\n🎉 Completed simple 6D pose estimation on {len(results)} images!")

if __name__ == "__main__":
    main()




