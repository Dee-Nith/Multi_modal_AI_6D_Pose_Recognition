#!/usr/bin/env python3
"""
Robot Pick and Place with 6D Pose Estimation
Controls UR5 robot to pick and place objects based on detected poses
"""

import cv2
import numpy as np
import json
import time
import sys
import os
import glob
import math
from pathlib import Path

# Add YOLO path
sys.path.append('../../coppelia_sim_dataset')
from ultralytics import YOLO

class RobotPickPlace:
    def __init__(self, model_path, calibration_path, models_dir):
        """Initialize robot pick and place system"""
        
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
        
        # Robot configuration
        self.robot_config = {
            'ur5': {
                'base_height': 0.0,  # Robot base height
                'gripper_offset': 0.15,  # Distance from end-effector to gripper
                'approach_distance': 0.05,  # Approach distance for picking
                'lift_height': 0.1,  # Height to lift after grasping
                'place_height': 0.05,  # Height above placement surface
                'gripper_open': 0.085,  # Open gripper width
                'gripper_close': 0.0,   # Closed gripper width
            }
        }
        
        # Pick and place targets
        self.pick_targets = {
            'banana': {'priority': 1, 'grasp_type': 'top'},
            'cracker_box': {'priority': 2, 'grasp_type': 'side'},
            'master_chef_can': {'priority': 3, 'grasp_type': 'top'},
            'mug': {'priority': 4, 'grasp_type': 'handle'},
            'mustard_bottle': {'priority': 5, 'grasp_type': 'top'}
        }
        
        # Place locations (relative to robot base)
        self.place_locations = {
            'zone_1': [0.3, 0.2, 0.0],   # Right side
            'zone_2': [0.3, -0.2, 0.0],  # Left side
            'zone_3': [0.5, 0.0, 0.0],   # Front
            'zone_4': [0.1, 0.3, 0.0],   # Back right
            'zone_5': [0.1, -0.3, 0.0]   # Back left
        }
        
        # Robot state
        self.robot_state = {
            'current_pose': None,
            'gripper_state': 'open',
            'is_moving': False,
            'current_task': None
        }
        
        # Task queue
        self.task_queue = []
        self.completed_tasks = []
        
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
    
    def camera_to_robot_transform(self, camera_pose):
        """Transform camera coordinates to robot coordinates"""
        # Camera to robot transformation (simplified)
        # In a real system, this would be calibrated
        
        # Camera is mounted above and behind the robot
        camera_offset = np.array([0.0, 0.0, 0.5])  # 50cm above robot base
        camera_rotation = np.array([0, 0, 0])  # Camera pointing down
        
        # Transform camera pose to robot coordinates
        robot_translation = camera_pose['translation'] + camera_offset
        
        # For simplicity, keep the same rotation
        robot_rotation = camera_pose['euler_angles']
        
        return {
            'translation': robot_translation,
            'rotation': robot_rotation,
            'class_name': camera_pose['class_name'],
            'confidence': camera_pose['confidence']
        }
    
    def plan_pick_trajectory(self, object_pose):
        """Plan trajectory for picking an object"""
        robot_pose = self.camera_to_robot_transform(object_pose)
        
        # Define waypoints for pick trajectory
        approach_distance = self.robot_config['ur5']['approach_distance']
        lift_height = self.robot_config['ur5']['lift_height']
        
        # Approach position (above object)
        approach_pos = robot_pose['translation'].copy()
        approach_pos[2] += approach_distance
        
        # Pick position (at object)
        pick_pos = robot_pose['translation'].copy()
        
        # Lift position (after grasping)
        lift_pos = pick_pos.copy()
        lift_pos[2] += lift_height
        
        trajectory = {
            'waypoints': [
                {'position': approach_pos, 'orientation': robot_pose['rotation'], 'type': 'approach'},
                {'position': pick_pos, 'orientation': robot_pose['rotation'], 'type': 'pick'},
                {'position': lift_pos, 'orientation': robot_pose['rotation'], 'type': 'lift'}
            ],
            'object_class': robot_pose['class_name'],
            'confidence': robot_pose['confidence']
        }
        
        return trajectory
    
    def plan_place_trajectory(self, place_zone):
        """Plan trajectory for placing an object"""
        place_pos = np.array(self.place_locations[place_zone])
        place_height = self.robot_config['ur5']['place_height']
        
        # Approach position (above place location)
        approach_pos = place_pos.copy()
        approach_pos[2] += place_height + 0.05
        
        # Place position (at surface)
        place_pos[2] += place_height
        
        trajectory = {
            'waypoints': [
                {'position': approach_pos, 'orientation': [0, 0, 0], 'type': 'approach'},
                {'position': place_pos, 'orientation': [0, 0, 0], 'type': 'place'},
                {'position': approach_pos, 'orientation': [0, 0, 0], 'type': 'retreat'}
            ],
            'place_zone': place_zone
        }
        
        return trajectory
    
    def generate_robot_commands(self, trajectory, action_type='pick'):
        """Generate robot commands for execution"""
        commands = []
        
        for i, waypoint in enumerate(trajectory['waypoints']):
            # Joint positions (simplified IK)
            joint_positions = self.inverse_kinematics(waypoint['position'], waypoint['orientation'])
            
            command = {
                'type': 'move_joints',
                'joints': joint_positions,
                'speed': 0.5,  # 50% speed
                'waypoint_type': waypoint['type']
            }
            commands.append(command)
            
            # Add gripper commands
            if action_type == 'pick':
                if waypoint['type'] == 'pick':
                    commands.append({
                        'type': 'gripper',
                        'action': 'close',
                        'width': self.robot_config['ur5']['gripper_close']
                    })
            elif action_type == 'place':
                if waypoint['type'] == 'place':
                    commands.append({
                        'type': 'gripper',
                        'action': 'open',
                        'width': self.robot_config['ur5']['gripper_open']
                    })
        
        return commands
    
    def inverse_kinematics(self, position, orientation):
        """Simplified inverse kinematics for UR5"""
        # This is a simplified IK - in practice, you'd use a proper IK solver
        
        # For demonstration, return approximate joint angles
        # These would be calculated based on the target position and orientation
        
        # Simplified 6-DOF IK approximation
        x, y, z = position
        roll, pitch, yaw = orientation
        
        # Base rotation (yaw)
        base_angle = math.atan2(y, x)
        
        # Shoulder and elbow angles (simplified)
        shoulder_angle = math.pi/4  # 45 degrees
        elbow_angle = -math.pi/4    # -45 degrees
        
        # Wrist angles
        wrist_1 = pitch
        wrist_2 = roll
        wrist_3 = yaw
        
        return [base_angle, shoulder_angle, elbow_angle, wrist_1, wrist_2, wrist_3]
    
    def add_pick_task(self, object_pose):
        """Add a pick task to the queue"""
        priority = self.pick_targets[object_pose['class_name']]['priority']
        
        task = {
            'id': len(self.task_queue) + 1,
            'type': 'pick',
            'object_pose': object_pose,
            'priority': priority,
            'status': 'pending',
            'created_time': time.time()
        }
        
        self.task_queue.append(task)
        
        # Sort by priority
        self.task_queue.sort(key=lambda x: x['priority'])
        
        print(f"📋 Added pick task for {object_pose['class_name']} (Priority: {priority})")
    
    def execute_pick_task(self, task):
        """Execute a pick task"""
        print(f"🤖 Executing pick task for {task['object_pose']['class_name']}")
        
        # Plan trajectory
        pick_trajectory = self.plan_pick_trajectory(task['object_pose'])
        
        # Generate commands
        commands = self.generate_robot_commands(pick_trajectory, 'pick')
        
        # Execute commands (simulated)
        for command in commands:
            print(f"  📡 Command: {command['type']} - {command.get('waypoint_type', 'N/A')}")
            time.sleep(0.5)  # Simulate execution time
        
        # Mark task as completed
        task['status'] = 'completed'
        task['completed_time'] = time.time()
        self.completed_tasks.append(task)
        
        print(f"✅ Pick task completed for {task['object_pose']['class_name']}")
        
        # Add place task
        place_zone = f"zone_{len(self.completed_tasks)}"
        place_trajectory = self.plan_place_trajectory(place_zone)
        place_commands = self.generate_robot_commands(place_trajectory, 'place')
        
        print(f"📦 Placing {task['object_pose']['class_name']} in {place_zone}")
        
        for command in place_commands:
            print(f"  📡 Command: {command['type']} - {command.get('waypoint_type', 'N/A')}")
            time.sleep(0.5)  # Simulate execution time
        
        print(f"✅ Place task completed for {task['object_pose']['class_name']}")
    
    def process_detections(self, detections):
        """Process object detections and add to task queue"""
        for detection in detections:
            pose = self.estimate_pose_realtime(None, detection)
            if pose and pose['confidence'] > 0.7:  # High confidence threshold
                # Check if object is already in queue
                object_name = pose['class_name']
                already_queued = any(task['object_pose']['class_name'] == object_name 
                                   for task in self.task_queue)
                
                if not already_queued:
                    self.add_pick_task(pose)
    
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
            # Read as binary data first
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # Try to decode as string first (for text format)
            try:
                text_data = data.decode('utf-8').strip()
                
                if text_data.startswith('[') and text_data.endswith(']'):
                    # Remove brackets and split by commas
                    text_data = text_data[1:-1]
                    values = [int(x.strip()) for x in text_data.split(',') if x.strip()]
                    image_data = np.array(values, dtype=np.uint8)
                else:
                    # Try direct conversion
                    image_data = np.frombuffer(data, dtype=np.uint8)
            except UnicodeDecodeError:
                # If text decoding fails, treat as binary
                image_data = np.frombuffer(data, dtype=np.uint8)
            
            # Reshape based on expected size
            if len(image_data) == 921600:  # 640x480x3
                image = image_data.reshape(480, 640, 3)
            elif len(image_data) == 9216:  # 64x48x3
                image = image_data.reshape(48, 64, 3)
                # Resize to standard size
                image = cv2.resize(image, (640, 480))
            else:
                print(f"⚠️ Unexpected image size: {len(image_data)} bytes")
                return None
            
            # Convert from RGB to BGR (OpenCV format)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            return image
            
        except Exception as e:
            print(f"❌ Error processing image file {file_path}: {e}")
            return None
    
    def visualize_robot_control(self, frame, poses, tasks):
        """Visualize robot control with task information"""
        result_frame = frame.copy()
        
        # Draw poses
        for i, pose in enumerate(poses):
            bbox = pose['bbox']
            class_name = pose['class_name']
            confidence = pose['confidence']
            translation = pose['translation']
            euler_degrees = pose['euler_degrees']
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw labels
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(result_frame, label, (x1, y1-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Position
            pos_text = f"P: [{translation[0]:.2f}, {translation[1]:.2f}, {translation[2]:.2f}]"
            cv2.putText(result_frame, pos_text, (x1, y1-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Rotation
            rot_text = f"R: [{euler_degrees[0]:.0f}°, {euler_degrees[1]:.0f}°, {euler_degrees[2]:.0f}°]"
            cv2.putText(result_frame, rot_text, (x1, y2+15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        
        # Task information overlay
        cv2.putText(result_frame, f"FPS: {self.avg_fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(result_frame, f"Objects: {len(poses)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(result_frame, f"Tasks: {len(self.task_queue)}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(result_frame, f"Completed: {len(self.completed_tasks)}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Instructions
        cv2.putText(result_frame, "Press 'q' to quit, 's' to save, 'e' to execute tasks", 
                   (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result_frame
    
    def run_robot_control(self):
        """Run the complete robot pick and place system"""
        print("🤖 Starting Robot Pick and Place System...")
        print("📹 Press 'q' to quit, 's' to save, 'e' to execute tasks")
        print("💡 Run your CoppeliaSim capture script to detect objects!")
        
        frame_count = 0
        last_processed = None
        
        while True:
            # Find latest CoppeliaSim image
            latest_file = self.find_latest_coppelia_image()
            
            if latest_file and latest_file != last_processed:
                # Process new image
                frame = self.process_coppelia_image(latest_file)
                
                if frame is not None:
                    # Detect objects
                    detections = self.detect_objects_realtime(frame)
                    
                    # Process detections and add to task queue
                    self.process_detections(detections)
                    
                    # Estimate poses for visualization
                    poses = []
                    for detection in detections:
                        pose = self.estimate_pose_realtime(frame, detection)
                        if pose:
                            poses.append(pose)
                    
                    # Update FPS
                    self.update_fps()
                    
                    # Visualize results
                    result_frame = self.visualize_robot_control(frame, poses, self.task_queue)
                    
                    # Display frame
                    cv2.imshow('Robot Pick and Place System', result_frame)
                    
                    last_processed = latest_file
                    frame_count += 1
                    
                    if frame_count % 10 == 0:
                        print(f"📸 Frame {frame_count}: {len(poses)} objects, {len(self.task_queue)} tasks")
            
            # Handle key presses
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q'):
                print("🛑 Stopping robot control system...")
                break
            elif key == ord('s'):
                if 'result_frame' in locals():
                    timestamp = int(time.time())
                    save_path = f"../results/robot_control_{timestamp}.jpg"
                    cv2.imwrite(save_path, result_frame)
                    print(f"📸 Saved frame to: {save_path}")
            elif key == ord('e'):
                # Execute pending tasks
                if self.task_queue:
                    print(f"🚀 Executing {len(self.task_queue)} pending tasks...")
                    for task in self.task_queue[:]:  # Copy list to avoid modification during iteration
                        if task['status'] == 'pending':
                            self.execute_pick_task(task)
                            self.task_queue.remove(task)
                else:
                    print("📋 No pending tasks to execute")
        
        # Cleanup
        cv2.destroyAllWindows()
        print(f"✅ Processed {frame_count} frames")
        print(f"📊 Completed {len(self.completed_tasks)} pick and place tasks")

def main():
    """Main function for robot pick and place system"""
    # Paths
    model_path = "../../coppelia_sim_dataset/runs/detect/train/weights/best.pt"
    calibration_path = "../calibration/coppelia_camera_calibration.json"
    models_dir = "../models"
    
    # Initialize robot pick and place system
    robot_system = RobotPickPlace(model_path, calibration_path, models_dir)
    
    print("🎯 Robot Pick and Place with 6D Pose Estimation")
    print("=" * 55)
    print("📋 Instructions:")
    print("1. Start CoppeliaSim with your scene")
    print("2. Run your capture script to detect objects")
    print("3. Press 'e' to execute pick and place tasks")
    print("4. Press 'q' to quit, 's' to save current frame")
    print()
    print("🤖 Robot Configuration:")
    print(f"   - Gripper open width: {robot_system.robot_config['ur5']['gripper_open']}m")
    print(f"   - Approach distance: {robot_system.robot_config['ur5']['approach_distance']}m")
    print(f"   - Lift height: {robot_system.robot_config['ur5']['lift_height']}m")
    print()
    print("📦 Pick Priorities:")
    for obj, config in robot_system.pick_targets.items():
        print(f"   {config['priority']}. {obj} ({config['grasp_type']} grasp)")
    print()
    
    # Start robot control system
    robot_system.run_robot_control()

if __name__ == "__main__":
    main()
