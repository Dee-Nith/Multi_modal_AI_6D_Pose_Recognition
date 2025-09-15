#!/usr/bin/env python3
"""
Real-Time Camera-Robot Integration for 6D Pose Estimation and Pick & Place
Combines YOLO detection, depth-based pose estimation, coordinate transformation, and robot control
"""

import cv2
import numpy as np
import glob
import os
import json
import time
from ultralytics import YOLO
import math

class RealTimeCameraRobotIntegration:
    def __init__(self):
        """Initialize the complete camera-robot integration system"""
        print("🚀 Initializing Real-Time Camera-Robot Integration System...")
        
        # Load YOLO model
        self.model = YOLO('../../coppelia_sim_results/weights/best.pt')
        print("✅ YOLO model loaded successfully")
        
        # Camera to Robot coordinate transformation offsets
        # Calculated from CoppeliaSim scene analysis
        self.OFFSET_X = -0.325  # meters (camera is 32.5cm to the left of robot)
        self.OFFSET_Y = -0.575  # meters (camera is 57.5cm behind robot)
        self.OFFSET_Z = +0.04561  # meters (camera is 4.6cm above robot)
        
        # Known object positions on pick conveyor (from CoppeliaSim)
        self.known_object_positions = {
            'master_chef_can': {'x': -0.625, 'y': -0.275, 'z': 0.750},
            'cracker_box': {'x': -0.625, 'y': -0.100, 'z': 0.825},
            'mug': {'x': -0.650, 'y': 0.075, 'z': 0.750},
            'banana': {'x': -0.625, 'y': 0.275, 'z': 0.725},
            'mustard_bottle': {'x': -0.625, 'y': 0.425, 'z': 0.800}
        }
        
        # Place conveyor position (where to put objects)
        self.place_conveyor = {'x': 0.100, 'y': -0.625, 'z': 0.700}
        
        # Robot base position
        self.robot_base = {'x': -0.625, 'y': 0.075, 'z': 0.700}
        
        print("✅ Coordinate system initialized")
        print("✅ Known object positions loaded")
        
        # Camera intrinsic parameters (from calibration)
        self.camera_matrix = np.array([
            [800, 0, 320],
            [0, 800, 240],
            [0, 0, 1]
        ])
        
        self.dist_coeffs = np.zeros(4)
        
        # 3D model points for PnP (simplified bounding box)
        self.model_points = np.array([
            [0, 0, 0],    # top-left
            [1, 0, 0],    # top-right
            [1, 1, 0],    # bottom-right
            [0, 1, 0],    # bottom-left
        ], dtype=np.float32)
        
        print("🚀 System initialization complete!")
        
    def capture_from_kinect(self):
        """Capture RGB and depth images from Kinect in CoppeliaSim"""
        try:
            # Find latest RGB image
            rgb_pattern = "/tmp/auto_kinect_*_rgb.txt"
            rgb_files = glob.glob(rgb_pattern)
            if not rgb_files:
                print("❌ No RGB images found in /tmp/")
                return None, None
                
            latest_rgb = max(rgb_files, key=os.path.getmtime)
            
            # Find corresponding depth image
            base_name = latest_rgb.replace("_rgb.txt", "")
            depth_file = base_name + "_depth.txt"
            
            if not os.path.exists(depth_file):
                print(f"❌ No depth image found for {base_name}")
                return self.process_coppelia_image(latest_rgb), None
            
            # Process both images
            rgb_image = self.process_coppelia_image(latest_rgb)
            depth_image = self.process_depth_image(depth_file)
            
            if rgb_image is not None:
                print(f"✅ Captured RGB image: {os.path.basename(latest_rgb)}")
                if depth_image is not None:
                    print(f"✅ Captured depth image: {os.path.basename(depth_file)}")
                else:
                    print("⚠️ Depth image processing failed")
            else:
                print("❌ RGB image processing failed")
                
            return rgb_image, depth_image
            
        except Exception as e:
            print(f"❌ Error capturing from Kinect: {e}")
            return None, None
    
    def process_coppelia_image(self, filepath):
        """Process CoppeliaSim image file to OpenCV format"""
        try:
            # Read as binary first (raw image data)
            with open(filepath, 'rb') as f:
                data = f.read()
            
            # Convert to numpy array
            if len(data) > 0:
                # Assume 640x480 RGB image
                height, width = 480, 640
                channels = 3
                
                # Reshape data to image dimensions
                if len(data) >= height * width * channels:
                    image_data = np.frombuffer(data[:height * width * channels], dtype=np.uint8)
                    image = image_data.reshape((height, width, channels))
                    
                    # Convert BGR to RGB if needed
                    if image.shape[2] == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    return image
                else:
                    print(f"❌ Insufficient data: {len(data)} bytes for {height}x{width}x{channels}")
                    return None
            else:
                print("❌ Empty file")
                return None
                
        except Exception as e:
            print(f"❌ Error processing image {filepath}: {e}")
            return None
    
    def process_depth_image(self, filepath):
        """Process CoppeliaSim depth file to numpy array"""
        try:
            with open(filepath, 'r') as f:
                content = f.read().strip()
            
            if content:
                # Parse comma-separated depth values
                depth_values = [float(x) for x in content.split(',') if x.strip()]
                
                if depth_values:
                    # Assume 640x480 depth image
                    width, height = 640, 480
                    
                    if len(depth_values) >= width * height:
                        # Reshape to 2D array
                        depth_array = np.array(depth_values[:width * height], dtype=np.float32)
                        depth_image = depth_array.reshape((height, width))
                        
                        print(f"✅ Depth image processed: {depth_image.shape}, range: {depth_image.min():.3f}-{depth_image.max():.3f}")
                        return depth_image
                    else:
                        print(f"❌ Insufficient depth data: {len(depth_values)} for {width}x{height}")
                        return None
                else:
                    print("❌ No valid depth values found")
                    return None
            else:
                print("❌ Empty depth file")
                return None
                
        except Exception as e:
            print(f"❌ Error processing depth file {filepath}: {e}")
            return None
    
    def detect_objects(self, image):
        """Detect objects using YOLO model"""
        try:
            results = self.model(image)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.model.names[class_id]
                        
                        if confidence > 0.5:  # Confidence threshold
                            detection = {
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(confidence),
                                'class_id': class_id,
                                'class_name': class_name
                            }
                            detections.append(detection)
            
            print(f"✅ Detected {len(detections)} objects")
            return detections
            
        except Exception as e:
            print(f"❌ Error in object detection: {e}")
            return []
    
    def estimate_pose_with_depth(self, image, depth_image, detection):
        """Estimate 6D pose using real depth data and PnP"""
        try:
            bbox = detection['bbox']
            class_name = detection['class_name']
            x1, y1, x2, y2 = bbox
            
            # Get object center
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Use real depth if available
            if depth_image is not None and 0 <= center_y < depth_image.shape[0] and 0 <= center_x < depth_image.shape[1]:
                depth_value = depth_image[center_y, center_x]
                if depth_value > 0 and depth_value < 10.0:  # Valid depth range
                    estimated_depth = depth_value
                    depth_source = "real"
                    print(f"✅ Using real depth: {depth_value:.3f}m")
                else:
                    estimated_depth = self.estimate_depth_from_size(class_name, bbox)
                    depth_source = "estimated (invalid real depth)"
                    print(f"⚠️ Invalid real depth, using estimated: {estimated_depth:.3f}m")
            else:
                estimated_depth = self.estimate_depth_from_size(class_name, bbox)
                depth_source = "estimated (no depth data)"
                print(f"⚠️ No depth data, using estimated: {estimated_depth:.3f}m")
            
            # Calculate 3D position using depth
            fx = self.camera_matrix[0, 0]
            fy = self.camera_matrix[1, 1]
            cx = self.camera_matrix[0, 2]
            cy = self.camera_matrix[1, 2]
            
            # Convert pixel coordinates to 3D world coordinates
            world_x = (center_x - cx) * estimated_depth / fx
            world_y = (center_y - cy) * estimated_depth / fy
            world_z = estimated_depth
            
            # Convert to meters and adjust for camera orientation
            world_x = world_x / 100.0  # Convert cm to meters
            world_y = world_y / 100.0
            world_z = world_z
            
            # Apply camera orientation correction (45° to the left)
            angle_rad = math.radians(45)
            corrected_x = world_x * math.cos(angle_rad) - world_y * math.sin(angle_rad)
            corrected_y = world_x * math.sin(angle_rad) + world_y * math.cos(angle_rad)
            
            # Convert camera coordinates to robot coordinates
            robot_x = corrected_x - self.OFFSET_X
            robot_y = corrected_y - self.OFFSET_Y
            robot_z = world_z - self.OFFSET_Z
            
            # Estimate rotation (simplified - can be enhanced with PnP)
            # For now, use object orientation based on bounding box
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 1.0
            
            # Simple rotation estimation based on aspect ratio
            if aspect_ratio > 1.5:  # Wide object
                yaw = 0  # Horizontal
            elif aspect_ratio < 0.7:  # Tall object
                yaw = 90  # Vertical
            else:
                yaw = 45  # Diagonal
            
            pose_6d = {
                'translation': [robot_x, robot_y, robot_z],
                'rotation': [0, 0, yaw],  # Roll, Pitch, Yaw in degrees
                'depth_source': depth_source,
                'camera_coords': [corrected_x, corrected_y, world_z],
                'robot_coords': [robot_x, robot_y, robot_z]
            }
            
            print(f"✅ 6D pose estimated: T({robot_x:.3f}, {robot_y:.3f}, {robot_z:.3f}), R(0, 0, {yaw})")
            return pose_6d
            
        except Exception as e:
            print(f"❌ Error in pose estimation: {e}")
            return None
    
    def estimate_depth_from_size(self, class_name, bbox):
        """Estimate depth based on known object sizes"""
        # Known object sizes in meters
        object_real_sizes = {
            'master_chef_can': 0.10,    # 10cm height
            'cracker_box': 0.16,        # 16cm height
            'mug': 0.12,                # 12cm height
            'banana': 0.20,             # 20cm length
            'mustard_bottle': 0.19      # 19cm height
        }
        
        real_size = object_real_sizes.get(class_name, 0.15)  # Default 15cm
        
        # Calculate depth using focal length and object size
        focal_length = 800  # pixels (from camera matrix)
        pixel_size = bbox[3] - bbox[1]  # Height in pixels
        
        if pixel_size > 0:
            estimated_depth = (real_size * focal_length) / pixel_size
            return estimated_depth / 100.0  # Convert to meters
        else:
            return 1.0  # Default 1 meter
    
    def plan_pick_and_place(self, detected_objects):
        """Plan robot pick and place sequence"""
        try:
            if not detected_objects:
                print("❌ No objects detected for planning")
                return []
            
            # Sort objects by priority (closest to robot first)
            robot_x, robot_y = self.robot_base['x'], self.robot_base['y']
            
            for obj in detected_objects:
                pose = obj['pose']
                robot_coords = pose['robot_coords']
                
                # Calculate distance to robot
                distance = math.sqrt((robot_coords[0] - robot_x)**2 + (robot_coords[1] - robot_y)**2)
                obj['distance_to_robot'] = distance
            
            # Sort by distance (closest first)
            detected_objects.sort(key=lambda x: x['distance_to_robot'])
            
            # Create pick and place sequence
            sequence = []
            for i, obj in enumerate(detected_objects):
                pick_operation = {
                    'type': 'pick',
                    'object_name': obj['detection']['class_name'],
                    'position': obj['pose']['robot_coords'],
                    'rotation': obj['pose']['rotation'],
                    'priority': i + 1
                }
                
                place_operation = {
                    'type': 'place',
                    'object_name': obj['detection']['class_name'],
                    'position': [self.place_conveyor['x'], self.place_conveyor['y'], self.place_conveyor['z']],
                    'rotation': [0, 0, 0],  # Standard orientation for placement
                    'priority': i + 1
                }
                
                sequence.append(pick_operation)
                sequence.append(place_operation)
            
            print(f"✅ Planned {len(sequence)} operations for {len(detected_objects)} objects")
            return sequence
            
        except Exception as e:
            print(f"❌ Error in planning: {e}")
            return []
    
    def visualize_detection_and_pose(self, image, detected_objects):
        """Visualize object detection and 6D poses"""
        try:
            vis_image = image.copy()
            
            for obj in detected_objects:
                detection = obj['detection']
                pose = obj['pose']
                
                # Draw bounding box
                bbox = detection['bbox']
                x1, y1, x2, y2 = bbox
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{detection['class_name']} ({detection['confidence']:.2f})"
                cv2.putText(vis_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw pose information
                robot_coords = pose['robot_coords']
                pose_text = f"Robot: ({robot_coords[0]:.3f}, {robot_coords[1]:.3f}, {robot_coords[2]:.3f})"
                cv2.putText(vis_image, pose_text, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
                # Draw depth source
                depth_text = f"Depth: {pose['depth_source']}"
                cv2.putText(vis_image, depth_text, (x1, y2+40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
                # Draw rotation
                rotation = pose['rotation']
                rot_text = f"Rotation: ({rotation[0]:.1f}°, {rotation[1]:.1f}°, {rotation[2]:.1f}°)"
                cv2.putText(vis_image, rot_text, (x1, y2+60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Add legend
            legend_text = "Green: Detection, Yellow: Pose Info, Blue: Robot Coordinates"
            cv2.putText(vis_image, legend_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return vis_image
            
        except Exception as e:
            print(f"❌ Error in visualization: {e}")
            return image
    
    def generate_robot_commands(self, sequence):
        """Generate robot command files for CoppeliaSim to execute"""
        try:
            if not sequence:
                print("❌ No sequence to generate commands for")
                return None
            
            # Create command directory if it doesn't exist
            command_dir = "/tmp/robot_commands"
            os.makedirs(command_dir, exist_ok=True)
            
            # Generate timestamp for this command set
            timestamp = int(time.time())
            command_file = f"{command_dir}/pick_place_commands_{timestamp}.json"
            
            # Create command structure
            commands = {
                "timestamp": timestamp,
                "total_operations": len(sequence),
                "robot_base": self.robot_base,
                "place_conveyor": self.place_conveyor,
                "operations": []
            }
            
            # Convert sequence to robot commands
            for i, operation in enumerate(sequence):
                op_type = operation['type']
                object_name = operation['object_name']
                position = operation['position']
                rotation = operation['rotation']
                
                if op_type == 'pick':
                    # Pick operation: Move to object, open gripper, close gripper, lift
                    pick_commands = [
                        {
                            "step": 1,
                            "action": "move_to_approach",
                            "position": [position[0], position[1], position[2] + 0.1],  # 10cm above object
                            "rotation": rotation,
                            "description": f"Move above {object_name} for pick"
                        },
                        {
                            "step": 2,
                            "action": "open_gripper",
                            "description": f"Open gripper to pick {object_name}"
                        },
                        {
                            "step": 3,
                            "action": "move_to_pick",
                            "position": position,
                            "rotation": rotation,
                            "description": f"Move to pick position for {object_name}"
                        },
                        {
                            "step": 4,
                            "action": "close_gripper",
                            "description": f"Close gripper to grasp {object_name}"
                        },
                        {
                            "step": 5,
                            "action": "move_to_lift",
                            "position": [position[0], position[1], position[2] + 0.15],  # 15cm above object
                            "rotation": rotation,
                            "description": f"Lift {object_name} after pick"
                        }
                    ]
                    
                    commands["operations"].extend(pick_commands)
                    
                elif op_type == 'place':
                    # Place operation: Move to place position, open gripper, move away
                    place_commands = [
                        {
                            "step": 1,
                            "action": "move_to_place_approach",
                            "position": [position[0], position[1], position[2] + 0.1],  # 10cm above place
                            "rotation": [0, 0, 0],
                            "description": f"Move above place position for {object_name}"
                        },
                        {
                            "step": 2,
                            "action": "move_to_place",
                            "position": position,
                            "rotation": [0, 0, 0],
                            "description": f"Move to place position for {object_name}"
                        },
                        {
                            "step": 3,
                            "action": "open_gripper",
                            "description": f"Release {object_name} at place position"
                        },
                        {
                            "step": 4,
                            "action": "move_to_place_approach",
                            "position": [position[0], position[1], position[2] + 0.15],  # 15cm above place
                            "rotation": [0, 0, 0],
                            "description": f"Move away from place position for {object_name}"
                        }
                    ]
                    
                    commands["operations"].extend(place_commands)
            
            # Add final return to home position
            home_command = {
                "step": len(commands["operations"]) + 1,
                "action": "return_home",
                "position": [self.robot_base['x'], self.robot_base['y'], self.robot_base['z'] + 0.3],
                "rotation": [0, 0, 0],
                "description": "Return robot to home position"
            }
            commands["operations"].append(home_command)
            
            # Save commands to JSON file
            with open(command_file, 'w') as f:
                json.dump(commands, f, indent=2)
            
            print(f"✅ Robot commands generated: {command_file}")
            print(f"📋 Total robot movements: {len(commands['operations'])}")
            
            # Also create a status file to track execution
            status_file = f"{command_dir}/pick_place_status_{timestamp}.json"
            status = {
                "command_file": command_file,
                "timestamp": timestamp,
                "status": "ready",
                "current_step": 0,
                "total_steps": len(commands["operations"]),
                "completed_operations": [],
                "errors": []
            }
            
            with open(status_file, 'w') as f:
                json.dump(status, f, indent=2)
            
            print(f"✅ Status file created: {status_file}")
            return command_file
            
        except Exception as e:
            print(f"❌ Error generating robot commands: {e}")
            return None

    def save_robot_commands_to_file(self, sequence):
        """Save robot commands to a simple text file for easy CoppeliaSim reading"""
        try:
            if not sequence:
                print("❌ No sequence to save")
                return None
            
            # Create simple command file
            command_dir = "/tmp/robot_commands"
            os.makedirs(command_dir, exist_ok=True)
            
            timestamp = int(time.time())
            command_file = f"{command_dir}/robot_commands_{timestamp}.txt"
            
            with open(command_file, 'w') as f:
                f.write(f"# Robot Pick and Place Commands - Generated at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Total Operations: {len(sequence)}\n")
                f.write(f"# Robot Base: {self.robot_base}\n")
                f.write(f"# Place Conveyor: {self.place_conveyor}\n\n")
                
                for i, operation in enumerate(sequence):
                    op_type = operation['type']
                    object_name = operation['object_name']
                    position = operation['position']
                    rotation = operation['rotation']
                    
                    f.write(f"# Operation {i+1}: {op_type.upper()} {object_name}\n")
                    
                    if op_type == 'pick':
                        # Pick sequence
                        f.write(f"PICK_START {object_name}\n")
                        f.write(f"MOVE_APPROACH {position[0]:.6f} {position[1]:.6f} {position[2]+0.1:.6f} {rotation[0]:.1f} {rotation[1]:.1f} {rotation[2]:.1f}\n")
                        f.write(f"GRIPPER_OPEN\n")
                        f.write(f"MOVE_PICK {position[0]:.6f} {position[1]:.6f} {position[2]:.6f} {rotation[0]:.1f} {rotation[1]:.1f} {rotation[2]:.1f}\n")
                        f.write(f"GRIPPER_CLOSE\n")
                        f.write(f"MOVE_LIFT {position[0]:.6f} {position[1]:.6f} {position[2]+0.15:.6f} {rotation[0]:.1f} {rotation[1]:.1f} {rotation[2]:.1f}\n")
                        f.write(f"PICK_END {object_name}\n\n")
                        
                    elif op_type == 'place':
                        # Place sequence
                        f.write(f"PLACE_START {object_name}\n")
                        f.write(f"MOVE_PLACE_APPROACH {position[0]:.6f} {position[1]:.6f} {position[2]+0.1:.6f} 0.0 0.0 0.0\n")
                        f.write(f"MOVE_PLACE {position[0]:.6f} {position[1]:.6f} {position[2]:.6f} 0.0 0.0 0.0\n")
                        f.write(f"GRIPPER_OPEN\n")
                        f.write(f"MOVE_PLACE_APPROACH {position[0]:.6f} {position[1]:.6f} {position[2]+0.15:.6f} 0.0 0.0 0.0\n")
                        f.write(f"PLACE_END {object_name}\n\n")
                
                # Add return home command
                f.write(f"# Final: Return to home position\n")
                f.write(f"RETURN_HOME {self.robot_base['x']:.6f} {self.robot_base['y']:.6f} {self.robot_base['z']+0.3:.6f} 0.0 0.0 0.0\n")
                f.write(f"SYSTEM_READY\n")
            
            print(f"✅ Robot commands saved to: {command_file}")
            return command_file
            
        except Exception as e:
            print(f"❌ Error saving robot commands: {e}")
            return None
    
    def run_real_time_system(self):
        """Run the complete real-time camera-robot integration system"""
        print("🚀 Starting Real-Time Camera-Robot Integration System...")
        print("📷 Camera: Kinect in CoppeliaSim")
        print("🤖 Robot: UR5 with RG2 gripper")
        print("🎯 Mode: Real-time 6D pose estimation + Pick & Place planning")
        print("=" * 60)
        
        try:
            while True:
                print(f"\n🔄 Cycle {time.strftime('%H:%M:%S')}")
                print("-" * 40)
                
                # Step 1: Capture from Kinect
                print("📷 Capturing from Kinect...")
                rgb_image, depth_image = self.capture_from_kinect()
                
                if rgb_image is None:
                    print("❌ Failed to capture RGB image, retrying...")
                    time.sleep(2)
                    continue
                
                # Step 2: Detect objects
                print("🔍 Detecting objects...")
                detections = self.detect_objects(rgb_image)
                
                if not detections:
                    print("⚠️ No objects detected, retrying...")
                    time.sleep(2)
                    continue
                
                # Step 3: Estimate 6D poses
                print("📐 Estimating 6D poses...")
                detected_objects = []
                
                for detection in detections:
                    pose = self.estimate_pose_with_depth(rgb_image, depth_image, detection)
                    if pose:
                        detected_objects.append({
                            'detection': detection,
                            'pose': pose
                        })
                
                if not detected_objects:
                    print("⚠️ No poses estimated, retrying...")
                    time.sleep(2)
                    continue
                
                # Step 4: Plan pick and place sequence
                print("🤖 Planning pick and place sequence...")
                sequence = self.plan_pick_and_place(detected_objects)
                
                if not sequence:
                    print("⚠️ No sequence planned, retrying...")
                    time.sleep(2)
                    continue
                
                # Step 5: Generate and save robot commands
                print("🤖 Generating robot commands...")
                command_file = self.save_robot_commands_to_file(sequence)
                
                if command_file:
                    print(f"🚀 Robot commands ready for CoppeliaSim: {command_file}")
                    print("📋 Instructions:")
                    print("  1. Copy the command file to CoppeliaSim")
                    print("  2. Run the robot controller script")
                    print("  3. Watch the robot execute pick and place!")
                else:
                    print("❌ Failed to generate robot commands")
                
                # Step 6: Visualize results
                print("🎨 Visualizing results...")
                vis_image = self.visualize_detection_and_pose(rgb_image, detected_objects)
                
                # Display results
                cv2.imshow('Real-Time Camera-Robot Integration', vis_image)
                
                # Print summary
                print(f"✅ Summary: {len(detected_objects)} objects detected, {len(sequence)} operations planned")
                print("📋 Operations:")
                for op in sequence:
                    print(f"  {op['type'].upper()}: {op['object_name']} at {op['position']}")
                
                # Check for exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("🛑 User requested exit")
                    break
                
                # Wait before next cycle
                print("⏳ Waiting 5 seconds before next cycle...")
                time.sleep(5)
                
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        except Exception as e:
            print(f"❌ System error: {e}")
        finally:
            cv2.destroyAllWindows()
            print("🏁 System shutdown complete")

def main():
    """Main function to run the integration system"""
    print("🚀 Real-Time Camera-Robot Integration System")
    print("=" * 50)
    
    # Initialize system
    system = RealTimeCameraRobotIntegration()
    
    # Run real-time system
    system.run_real_time_system()

if __name__ == "__main__":
    main()
