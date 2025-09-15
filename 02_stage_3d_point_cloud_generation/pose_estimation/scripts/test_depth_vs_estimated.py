#!/usr/bin/env python3
"""
Test Depth vs Estimated Depth for 6D Pose Estimation
Compares accuracy between real depth data and size-based estimation
"""

import cv2
import numpy as np
import sys
import os
import time
import glob
import json
from ultralytics import YOLO
import trimesh

class DepthComparisonTest:
    def __init__(self, model_path, calibration_path, models_dir):
        """Initialize with YOLO model, camera calibration, and 3D models"""
        print("🔄 Loading AI system for depth comparison...")
        
        # Load YOLO model
        self.yolo_model = YOLO(model_path)
        print(f"✅ Loaded YOLO model: {model_path}")
        
        # Load camera calibration
        with open(calibration_path, 'r') as f:
            calib_data = json.load(f)
        self.camera_matrix = np.array(calib_data['camera_matrix'])
        self.dist_coeffs = np.array(calib_data['dist_coeffs'])
        print(f"✅ Loaded camera calibration: {calibration_path}")
        
        # Load 3D models
        self.models = {}
        self.class_names = ['master_chef_can', 'cracker_box', 'mug', 'banana', 'mustard_bottle']
        
        for class_name in self.class_names:
            model_path = os.path.join(models_dir, f"{class_name}.obj")
            if os.path.exists(model_path):
                mesh = trimesh.load(model_path)
                self.models[class_name] = mesh
                print(f"  📦 {class_name}: {len(mesh.vertices)} vertices")
        
        print(f"✅ Loaded {len(self.models)} 3D models")
        
        # Object real sizes for estimation
        self.object_real_sizes = {
            'master_chef_can': 0.10,    # 10cm height
            'cracker_box': 0.16,        # 16cm height
            'mug': 0.12,                # 12cm height
            'banana': 0.20,             # 20cm length
            'mustard_bottle': 0.19      # 19cm height
        }
        
        print("✅ AI system loaded successfully!")
    
    def process_coppelia_image(self, file_path):
        """Process CoppeliaSim image file (RGB)"""
        try:
            # Read as binary first (most reliable for RGB data)
            with open(file_path, 'rb') as f:
                data = np.frombuffer(f.read(), dtype=np.uint8)
            
            # Reshape based on expected size
            if len(data) == 640 * 480 * 3:
                image = data.reshape(480, 640, 3)
            elif len(data) == 64 * 48 * 3:
                image = data.reshape(48, 64, 3)
                # Resize to standard size
                image = cv2.resize(image, (640, 480))
            else:
                print(f"⚠️ Unexpected image size: {len(data)} bytes")
                return None
            
            # Convert RGB to BGR for OpenCV
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image
            
        except Exception as e:
            print(f"❌ Error processing image {file_path}: {e}")
            return None
    
    def process_depth_image(self, file_path):
        """Process CoppeliaSim depth image file"""
        try:
            # Try reading as text first
            with open(file_path, 'r') as f:
                content = f.read().strip()
            
            # Try to parse as comma-separated values
            try:
                # Split by comma and convert to float
                depth_values = [float(x.strip()) for x in content.split(',')]
                data = np.array(depth_values, dtype=np.float32)
            except:
                # If comma parsing fails, try to parse as list
                try:
                    data = eval(content)
                    if isinstance(data, list):
                        data = np.array(data, dtype=np.float32)
                    else:
                        raise ValueError("Not a list")
                except:
                    # If text parsing fails, read as binary
                    with open(file_path, 'rb') as f:
                        data = np.frombuffer(f.read(), dtype=np.float32)
            
            # Reshape based on expected size
            if len(data) == 640 * 480:
                depth = data.reshape(480, 640)
            elif len(data) == 64 * 48:
                depth = data.reshape(48, 64)
                # Resize to standard size
                depth = cv2.resize(depth, (640, 480), interpolation=cv2.INTER_NEAREST)
            else:
                print(f"⚠️ Unexpected depth size: {len(data)} values")
                return None
            
            return depth
            
        except Exception as e:
            print(f"❌ Error processing depth image {file_path}: {e}")
            return None
    
    def detect_objects(self, image):
        """Detect objects using YOLO"""
        results = self.yolo_model(image, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Get class
                    class_id = int(box.cls[0].cpu().numpy())
                    if hasattr(result, 'names') and result.names:
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
    
    def estimate_depth_from_size(self, class_name, bbox):
        """Estimate depth from object size in pixels"""
        if class_name in self.object_real_sizes:
            real_size = self.object_real_sizes[class_name]
            x1, y1, x2, y2 = bbox
            pixel_size = max(x2 - x1, y2 - y1)
            focal_length = self.camera_matrix[0, 0]  # fx
            estimated_depth = (real_size * focal_length) / pixel_size
            return estimated_depth
        else:
            return 0.5  # Default depth
    
    def estimate_pose_with_depth(self, image, depth_image, detection):
        """Estimate 6D pose using real depth data"""
        class_name = detection['class_name']
        bbox = detection['bbox']
        
        if class_name not in self.models:
            return None
        
        # Extract region of interest
        x1, y1, x2, y2 = map(int, bbox)
        
        # Get depth at object center
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        if depth_image is not None:
            # Use real depth data
            depth_value = depth_image[center_y, center_x]
            
            # Validate depth value
            if depth_value > 0 and depth_value < 10.0:  # Reasonable depth range
                estimated_depth = depth_value
                depth_source = "real"
            else:
                # Fallback to size-based estimation
                estimated_depth = self.estimate_depth_from_size(class_name, bbox)
                depth_source = "estimated (invalid real depth)"
        else:
            # No depth image available
            estimated_depth = self.estimate_depth_from_size(class_name, bbox)
            depth_source = "estimated (no depth data)"
        
        # Use simplified pose estimation for comparison
        return self.estimate_pose_simplified(class_name, bbox, estimated_depth, depth_source)
    
    def estimate_pose_simplified(self, class_name, bbox, depth, depth_source):
        """Simplified pose estimation for comparison"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Create simplified 3D bounding box
        half_size = self.object_real_sizes.get(class_name, 0.05) / 2
        
        bbox_3d = np.array([
            [-half_size, -half_size, depth],
            [half_size, -half_size, depth],
            [half_size, half_size, depth],
            [-half_size, half_size, depth]
        ], dtype=np.float32)
        
        bbox_2d = np.array([
            [x1, y2], [x2, y2], [x2, y1], [x1, y1]
        ], dtype=np.float32)
        
        success, rvec, tvec = cv2.solvePnP(
            bbox_3d, bbox_2d, self.camera_matrix, self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success:
            # Convert rotation vector to rotation matrix
            rmat, _ = cv2.Rodrigues(rvec)
            
            # Convert to Euler angles
            euler_angles = self.rotation_matrix_to_euler_angles(rmat)
            
            return {
                'translation': tvec.flatten(),
                'rotation_matrix': rmat,
                'rotation_vector': rvec.flatten(),
                'euler_angles': euler_angles,
                'euler_degrees': np.degrees(euler_angles),
                'bbox': bbox,
                'class_name': class_name,
                'depth': depth,
                'depth_source': depth_source,
                'distance': np.linalg.norm(tvec)
            }
        
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
    
    def visualize_comparison(self, image, poses):
        """Visualize results with depth comparison"""
        result = image.copy()
        
        # Draw detections and poses
        for pose in poses:
            bbox = pose['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw bounding box with color based on depth source
            if "real" in pose['depth_source']:
                color = (0, 255, 0)  # Green for real depth
            else:
                color = (0, 165, 255)  # Orange for estimated depth
            
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with depth info
            label = f"{pose['class_name']} ({pose['depth_source']})"
            cv2.putText(result, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw pose info
            pos = pose['translation']
            rot = pose['euler_degrees']
            depth_val = pose['depth']
            
            pose_text = f"Pos: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
            depth_text = f"Depth: {depth_val:.3f}m"
            rot_text = f"Rot: ({rot[0]:.1f}°, {rot[1]:.1f}°, {rot[2]:.1f}°)"
            
            cv2.putText(result, pose_text, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.putText(result, depth_text, (x1, y2+35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.putText(result, rot_text, (x1, y2+50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Add legend
        cv2.putText(result, "Green: Real Depth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(result, "Orange: Estimated Depth", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        return result
    
    def test_with_image_pair(self, rgb_file, depth_file=None):
        """Test pose estimation with RGB and optional depth image"""
        print(f"\n📸 Testing with RGB image: {os.path.basename(rgb_file)}")
        if depth_file:
            print(f"📏 Testing with depth image: {os.path.basename(depth_file)}")
        
        # Process RGB image
        image = self.process_coppelia_image(rgb_file)
        if image is None:
            print("❌ Failed to process RGB image")
            return None
        
        print(f"✅ RGB image processed: {image.shape}")
        
        # Process depth image if available
        depth_image = None
        if depth_file and os.path.exists(depth_file):
            depth_image = self.process_depth_image(depth_file)
            if depth_image is not None:
                print(f"✅ Depth image processed: {depth_image.shape}")
                print(f"📊 Depth range: {np.min(depth_image):.3f}m - {np.max(depth_image):.3f}m")
            else:
                print("⚠️ Failed to process depth image")
        else:
            print("⚠️ No depth image provided")
        
        # Detect objects
        detections = self.detect_objects(image)
        print(f"🎯 Detected {len(detections)} objects")
        
        # Show detection results
        for i, detection in enumerate(detections):
            print(f"  {i+1}. {detection['class_name']} (confidence: {detection['confidence']:.2f})")
        
        # Estimate poses with depth comparison
        poses = []
        for detection in detections:
            pose = self.estimate_pose_with_depth(image, depth_image, detection)
            if pose:
                poses.append(pose)
                depth_source = pose['depth_source']
                pos = pose['translation']
                depth_val = pose['depth']
                print(f"  📐 {pose['class_name']}: pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) depth={depth_val:.3f}m [{depth_source}]")
        
        # Create visualization
        result = self.visualize_comparison(image, poses)
        
        # Save result
        timestamp = int(time.time())
        save_path = f"../results/depth_comparison_{timestamp}.jpg"
        cv2.imwrite(save_path, result)
        print(f"📸 Result saved to: {save_path}")
        
        # Show the result
        print("🖼️ Displaying comparison...")
        print("Press any key to continue...")
        cv2.imshow('Depth vs Estimated Depth Comparison', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return poses

def main():
    """Main function to test depth comparison"""
    print("🔬 Depth vs Estimated Depth 6D Pose Estimation Test")
    print("=" * 60)
    
    # Paths
    model_path = "../../coppelia_sim_dataset/runs/detect/train/weights/best.pt"
    calibration_path = "../calibration/coppelia_camera_calibration.json"
    models_dir = "../models"
    
    # Initialize test system
    test_system = DepthComparisonTest(model_path, calibration_path, models_dir)
    
    # Test options
    print("\n🎯 Test Options:")
    print("1. Test with latest captured images")
    print("2. Test with specific image pair")
    print("3. Test with RGB only (no depth)")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        # Find latest images in organized folders
        rgb_pattern = "../rgb_captures/auto_kinect_*_rgb.txt"
        rgb_files = glob.glob(rgb_pattern)
        
        if not rgb_files:
            print("❌ No captured images found!")
            print("💡 Please run the CoppeliaSim capture script first")
            return
        
        # Get most recent RGB file
        latest_rgb = max(rgb_files, key=os.path.getmtime)
        
        # Find corresponding depth file
        rgb_base = latest_rgb.replace('_rgb.txt', '').replace('../rgb_captures/', '../depth_captures/')
        corresponding_depth = rgb_base + '_depth.txt'
        
        if os.path.exists(corresponding_depth):
            print(f"📸 Found image pair:")
            print(f"  RGB: {latest_rgb}")
            print(f"  Depth: {corresponding_depth}")
            test_system.test_with_image_pair(latest_rgb, corresponding_depth)
        else:
            print("⚠️ No corresponding depth file found, testing with RGB only")
            test_system.test_with_image_pair(latest_rgb)
    
    elif choice == "2":
        # Test with specific image
        rgb_file = input("Enter RGB file path: ").strip()
        depth_file = input("Enter depth file path (or press Enter for none): ").strip()
        
        if not depth_file:
            depth_file = None
        
        test_system.test_with_image_pair(rgb_file, depth_file)
    
    elif choice == "3":
        # Test with RGB only
        rgb_file = input("Enter RGB file path: ").strip()
        test_system.test_with_image_pair(rgb_file)
    
    print("\n🎯 Test completed!")

if __name__ == "__main__":
    main()
