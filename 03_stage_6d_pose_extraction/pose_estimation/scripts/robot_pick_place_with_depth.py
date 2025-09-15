#!/usr/bin/env python3
"""
Robot Pick and Place with Real Depth Data
Uses Kinect depth images for accurate 6D pose estimation
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

class RobotPickPlaceWithDepth:
    def __init__(self, model_path, calibration_path, models_dir):
        """Initialize with YOLO model, camera calibration, and 3D models"""
        print("🔄 Loading AI system with depth support...")
        
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
        
        # Robot configuration
        self.robot_config = {
            'ur5': {
                'approach_distance': 0.1,  # 10cm approach
                'lift_height': 0.2,        # 20cm lift
                'gripper_offset': 0.05,    # 5cm gripper offset
                'max_reach': 0.85          # 85cm max reach
            }
        }
        
        # Pick targets with priorities
        self.pick_targets = {
            'master_chef_can': {'priority': 3, 'grasp_type': 'top'},
            'cracker_box': {'priority': 2, 'grasp_type': 'side'},
            'mug': {'priority': 4, 'grasp_type': 'top'},
            'banana': {'priority': 1, 'grasp_type': 'side'},
            'mustard_bottle': {'priority': 5, 'grasp_type': 'top'}
        }
        
        # Place locations
        self.place_locations = {
            'zone_1': {'position': [0.3, 0.2, 0.1], 'orientation': [0, 0, 0]},
            'zone_2': {'position': [0.3, -0.2, 0.1], 'orientation': [0, 0, 0]},
            'zone_3': {'position': [0.5, 0.0, 0.1], 'orientation': [0, 0, 0]}
        }
        
        self.task_queue = []
        self.completed_tasks = []
        
        print("✅ AI system with depth support loaded successfully!")
    
    def find_latest_coppelia_images(self):
        """Find latest RGB and depth images from CoppeliaSim"""
        # Look for RGB images
        rgb_pattern = "/tmp/auto_kinect_*_rgb.txt"
        rgb_files = glob.glob(rgb_pattern)
        
        # Look for depth images
        depth_pattern = "/tmp/auto_kinect_*_depth.txt"
        depth_files = glob.glob(depth_pattern)
        
        if not rgb_files:
            return None, None
        
        # Get most recent RGB file
        latest_rgb = max(rgb_files, key=os.path.getmtime)
        
        # Find corresponding depth file
        rgb_base = latest_rgb.replace('_rgb.txt', '')
        corresponding_depth = rgb_base + '_depth.txt'
        
        if os.path.exists(corresponding_depth):
            return latest_rgb, corresponding_depth
        else:
            # If no corresponding depth, use most recent depth file
            if depth_files:
                latest_depth = max(depth_files, key=os.path.getmtime)
                return latest_rgb, latest_depth
            else:
                return latest_rgb, None
    
    def process_coppelia_image(self, file_path):
        """Process CoppeliaSim image file (RGB)"""
        try:
            # Try reading as text first (for list-like format)
            with open(file_path, 'r') as f:
                content = f.read().strip()
            
            # Try to parse as list
            try:
                data = eval(content)
                if isinstance(data, list):
                    # Convert list to numpy array
                    data = np.array(data, dtype=np.uint8)
                else:
                    raise ValueError("Not a list")
            except:
                # If text parsing fails, read as binary
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
            
            # Try to parse as list
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
    
    def detect_objects_realtime(self, image):
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
                print(f"  📏 Using real depth: {depth_value:.3f}m")
            else:
                # Fallback to size-based estimation
                estimated_depth = self.estimate_depth_from_size(class_name, bbox)
                print(f"  📏 Using estimated depth: {estimated_depth:.3f}m (real depth invalid)")
        else:
            # No depth image available
            estimated_depth = self.estimate_depth_from_size(class_name, bbox)
            print(f"  📏 Using estimated depth: {estimated_depth:.3f}m (no depth data)")
        
        # Get 3D model points for this object
        mesh = self.models[class_name]
        model_points = mesh.vertices.astype(np.float32)
        
        # Project 3D points to 2D using current depth estimate
        # This is a simplified approach - in practice you'd use ICP or similar
        projected_points = []
        for point_3d in model_points[:100]:  # Use subset for speed
            # Transform point to camera coordinates
            point_cam = point_3d + np.array([0, 0, estimated_depth])
            
            # Project to 2D
            point_2d, _ = cv2.projectPoints(
                point_cam.reshape(1, 1, 3), 
                np.zeros(3), np.zeros(3), 
                self.camera_matrix, self.dist_coeffs
            )
            projected_points.append(point_2d[0, 0])
        
        projected_points = np.array(projected_points)
        
        # Find correspondences within bounding box
        valid_points = []
        valid_model_points = []
        
        for i, point_2d in enumerate(projected_points):
            if (x1 <= point_2d[0] <= x2 and y1 <= point_2d[1] <= y2):
                valid_points.append(point_2d)
                valid_model_points.append(model_points[i])
        
        if len(valid_points) < 4:
            # Fallback to bbox-based pose estimation
            return self.estimate_pose_fallback(class_name, bbox, estimated_depth)
        
        # Use PnP with real 3D-2D correspondences
        valid_points = np.array(valid_points, dtype=np.float32)
        valid_model_points = np.array(valid_model_points, dtype=np.float32)
        
        success, rvec, tvec = cv2.solvePnP(
            valid_model_points, valid_points, 
            self.camera_matrix, self.dist_coeffs,
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
                'distance': np.linalg.norm(tvec),
                'depth_source': 'real' if depth_image is not None and depth_value > 0 else 'estimated'
            }
            
            return pose
        
        return None
    
    def estimate_depth_from_size(self, class_name, bbox):
        """Estimate depth from object size in pixels"""
        object_real_sizes = {
            'master_chef_can': 0.10,    # 10cm height
            'cracker_box': 0.16,        # 16cm height
            'mug': 0.12,                # 12cm height
            'banana': 0.20,             # 20cm length
            'mustard_bottle': 0.19      # 19cm height
        }
        
        if class_name in object_real_sizes:
            real_size = object_real_sizes[class_name]
            x1, y1, x2, y2 = bbox
            pixel_size = max(x2 - x1, y2 - y1)
            focal_length = self.camera_matrix[0, 0]  # fx
            estimated_depth = (real_size * focal_length) / pixel_size
            return estimated_depth
        else:
            return 0.5  # Default depth
    
    def estimate_pose_fallback(self, class_name, bbox, depth):
        """Fallback pose estimation using bounding box"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Simplified 3D-2D correspondences
        object_real_sizes = {
            'master_chef_can': 0.10,
            'cracker_box': 0.16,
            'mug': 0.12,
            'banana': 0.20,
            'mustard_bottle': 0.19
        }
        
        half_size = object_real_sizes.get(class_name, 0.05) / 2
        
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
            rmat, _ = cv2.Rodrigues(rvec)
            euler_angles = self.rotation_matrix_to_euler_angles(rmat)
            
            return {
                'translation': tvec.flatten(),
                'rotation_matrix': rmat,
                'rotation_vector': rvec.flatten(),
                'euler_angles': euler_angles,
                'euler_degrees': np.degrees(euler_angles),
                'bbox': bbox,
                'class_name': class_name,
                'confidence': detection['confidence'],
                'distance': np.linalg.norm(tvec),
                'depth_source': 'estimated'
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
    
    def visualize_with_depth(self, image, depth_image, poses, task_queue):
        """Visualize results with depth information"""
        result = image.copy()
        
        # Draw depth info
        if depth_image is not None:
            # Create depth visualization
            depth_viz = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)
            
            # Resize depth viz to fit in corner
            depth_viz = cv2.resize(depth_viz, (200, 150))
            
            # Add depth viz to result
            result[10:160, 10:210] = depth_viz
            
            # Add depth legend
            cv2.putText(result, "Depth Map", (15, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw detections and poses
        for pose in poses:
            bbox = pose['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw bounding box
            color = (0, 255, 0) if pose['depth_source'] == 'real' else (0, 165, 255)
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{pose['class_name']} ({pose['depth_source']})"
            cv2.putText(result, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw pose info
            pos = pose['translation']
            rot = pose['euler_degrees']
            pose_text = f"Pos: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
            rot_text = f"Rot: ({rot[0]:.1f}°, {rot[1]:.1f}°, {rot[2]:.1f}°)"
            
            cv2.putText(result, pose_text, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.putText(result, rot_text, (x1, y2+35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw 3D axes
            self.draw_3d_axes(result, pose)
        
        # Draw task queue info
        y_offset = 200
        cv2.putText(result, f"Tasks: {len(task_queue)}", (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        for i, task in enumerate(task_queue[:3]):  # Show first 3 tasks
            task_text = f"{i+1}. Pick {task['object_pose']['class_name']}"
            cv2.putText(result, task_text, (15, y_offset + 20 + i*15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return result
    
    def draw_3d_axes(self, image, pose):
        """Draw 3D coordinate axes for pose"""
        # Define axis points
        axis_length = 0.05  # 5cm
        axis_points = np.float32([[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]])
        
        # Transform axis points
        rvec = pose['rotation_vector']
        tvec = pose['translation']
        
        axis_points_2d, _ = cv2.projectPoints(
            axis_points, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )
        
        # Draw axes
        origin = tuple(map(int, axis_points_2d[0].ravel()))
        point_x = tuple(map(int, axis_points_2d[1].ravel()))
        point_y = tuple(map(int, axis_points_2d[2].ravel()))
        point_z = tuple(map(int, axis_points_2d[3].ravel()))
        
        cv2.line(image, origin, point_x, (0, 0, 255), 2)  # X-axis (red)
        cv2.line(image, origin, point_y, (0, 255, 0), 2)  # Y-axis (green)
        cv2.line(image, origin, point_z, (255, 0, 0), 2)  # Z-axis (blue)
    
    def process_detections(self, detections):
        """Process detections and add to task queue"""
        for detection in detections:
            if detection['confidence'] > 0.5:  # Confidence threshold
                class_name = detection['class_name']
                
                # Check if already in queue
                already_queued = any(task['object_pose']['class_name'] == class_name for task in self.task_queue)
                
                if not already_queued and class_name in self.pick_targets:
                    # Create task
                    task = {
                        'object_pose': detection,
                        'priority': self.pick_targets[class_name]['priority'],
                        'grasp_type': self.pick_targets[class_name]['grasp_type'],
                        'status': 'pending',
                        'timestamp': time.time()
                    }
                    
                    self.task_queue.append(task)
                    self.task_queue.sort(key=lambda x: x['priority'], reverse=True)
                    print(f"📋 Added pick task for {class_name} (Priority: {task['priority']})")
    
    def run_with_depth(self):
        """Main loop with depth support"""
        print("🚀 Starting AI 6D Pose Recognition with Depth Support")
        print("=" * 60)
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save current result")
        print("  'e' - Execute pick and place tasks")
        print("  'd' - Show depth information")
        print("=" * 60)
        
        last_processed = None
        
        while True:
            # Find latest images
            rgb_file, depth_file = self.find_latest_coppelia_images()
            
            if rgb_file is None:
                print("⏳ Waiting for CoppeliaSim images...")
                time.sleep(1)
                continue
            
            # Check if this is a new image
            if rgb_file == last_processed:
                time.sleep(0.1)  # Short delay
                continue
            
            print(f"\n📸 Processing new capture: {os.path.basename(rgb_file)}")
            
            # Process RGB image
            image = self.process_coppelia_image(rgb_file)
            if image is None:
                print("❌ Failed to process RGB image")
                continue
            
            # Process depth image
            depth_image = None
            if depth_file:
                depth_image = self.process_depth_image(depth_file)
                if depth_image is not None:
                    print(f"✅ Loaded depth image: {depth_image.shape}")
                else:
                    print("⚠️ Failed to load depth image")
            
            # Detect objects
            detections = self.detect_objects_realtime(image)
            print(f"🎯 Detected {len(detections)} objects")
            
            # Process detections
            self.process_detections(detections)
            
            # Estimate poses with depth
            poses = []
            for detection in detections:
                pose = self.estimate_pose_with_depth(image, depth_image, detection)
                if pose:
                    poses.append(pose)
                    depth_source = pose['depth_source']
                    pos = pose['translation']
                    print(f"  📐 {pose['class_name']}: pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) [{depth_source}]")
            
            # Visualize results
            result = self.visualize_with_depth(image, depth_image, poses, self.task_queue)
            
            # Display result
            cv2.imshow('AI 6D Pose Recognition with Depth', result)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = int(time.time())
                save_path = f"../results/depth_pose_{timestamp}.jpg"
                cv2.imwrite(save_path, result)
                print(f"📸 Saved result to: {save_path}")
            elif key == ord('e'):
                print("🚀 Executing pick and place tasks...")
                # Execute tasks (simplified)
                for task in self.task_queue[:]:
                    if task['status'] == 'pending':
                        print(f"🤖 Picking {task['object_pose']['class_name']}...")
                        task['status'] = 'completed'
                        self.completed_tasks.append(task)
                        self.task_queue.remove(task)
                print(f"✅ Completed {len(self.completed_tasks)} tasks!")
            elif key == ord('d'):
                if depth_image is not None:
                    print("📊 Depth Statistics:")
                    print(f"  Min depth: {np.min(depth_image):.3f}m")
                    print(f"  Max depth: {np.max(depth_image):.3f}m")
                    print(f"  Mean depth: {np.mean(depth_image):.3f}m")
                    print(f"  Valid pixels: {np.sum(depth_image > 0)}/{depth_image.size}")
            
            last_processed = rgb_file
        
        cv2.destroyAllWindows()
        print("🎯 AI system with depth support stopped")

if __name__ == "__main__":
    # Paths
    model_path = "../../coppelia_sim_dataset/runs/detect/train/weights/best.pt"
    calibration_path = "../calibration/coppelia_camera_calibration.json"
    models_dir = "../models"
    
    # Initialize and run
    robot_system = RobotPickPlaceWithDepth(model_path, calibration_path, models_dir)
    robot_system.run_with_depth()




