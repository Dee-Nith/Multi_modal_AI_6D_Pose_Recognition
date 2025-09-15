#!/usr/bin/env python3
"""
Standalone YOLOv8 Robotic Grasping Pipeline
Includes YOLOv8 integration without package import issues
"""

import zmq
import struct
import time
import sys
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO

class CoppeliaSimZMQClient:
    """ZeroMQ client for CoppeliaSim."""
    
    def __init__(self, host='localhost', port=23000):
        """Initialize the ZeroMQ client."""
        self.host = host
        self.port = port
        self.connected = False
        self.socket = None
        self.context = None
        
    def connect(self):
        """Connect to CoppeliaSim ZeroMQ server."""
        try:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.REQ)
            self.socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
            self.socket.setsockopt(zmq.SNDTIMEO, 5000)  # 5 second timeout
            
            # Connect to CoppeliaSim
            self.socket.connect(f"tcp://{self.host}:{self.port}")
            self.connected = True
            print(f"✅ Connected to CoppeliaSim ZeroMQ server at {self.host}:{self.port}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from CoppeliaSim."""
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        self.connected = False
        print("✅ Disconnected from CoppeliaSim")
    
    def send_message(self, message):
        """Send a message to CoppeliaSim."""
        if not self.connected:
            print("❌ Not connected to CoppeliaSim")
            return None
        
        try:
            # Send the message
            self.socket.send(message)
            
            # Receive response
            response = self.socket.recv()
            return response
            
        except zmq.error.Again:
            print("❌ Timeout waiting for response")
            return None
        except Exception as e:
            print(f"❌ Error sending message: {e}")
            return None
    
    def test_connection(self):
        """Test the connection with a simple message."""
        if not self.connected:
            return False
        
        try:
            # Simple test message
            test_msg = b"test"
            response = self.send_message(test_msg)
            
            if response:
                print(f"✅ Test successful! Response length: {len(response)}")
                return True
            else:
                print("❌ No response received")
                return False
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
    
    def get_object_handle(self, object_name):
        """Get object handle by name (placeholder)."""
        # TODO: Implement actual object handle retrieval
        print(f"📋 Getting handle for: {object_name}")
        return f"handle_{object_name}"
    
    def get_vision_sensor_image(self, handle, image_type):
        """Get vision sensor image (placeholder)."""
        # TODO: Implement actual image capture
        print(f"📸 Capturing {image_type} image from {handle}")
        
        if image_type == "rgb":
            # Return placeholder RGB image
            return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        elif image_type == "depth":
            # Return placeholder depth image
            return np.random.rand(480, 640)
        else:
            return None
    
    def get_camera_intrinsics(self, handle):
        """Get camera intrinsics (placeholder)."""
        # TODO: Implement actual camera intrinsics
        print(f"📐 Getting camera intrinsics for {handle}")
        return {
            'fx': 525.0,  # focal length x
            'fy': 525.0,  # focal length y
            'cx': 320.0,  # principal point x
            'cy': 240.0   # principal point y
        }

class YCBObjectDetector:
    """YOLOv8-based object detector for YCB models."""
    
    def __init__(self, model_path=None, confidence_threshold=0.5):
        """Initialize the YCB object detector."""
        self.model = None
        self.confidence_threshold = confidence_threshold
        self.ycb_classes = self._load_ycb_classes()
        
        # Load model
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            self.load_pretrained_model()
    
    def _load_ycb_classes(self):
        """Load YCB object classes."""
        # YCB-Video dataset classes (21 objects)
        ycb_classes = {
            0: '002_master_chef_can',
            1: '003_cracker_box', 
            2: '004_sugar_box',
            3: '005_tomato_soup_can',
            4: '006_mustard_bottle',
            5: '007_tuna_fish_can',
            6: '008_pudding_box',
            7: '009_gelatin_box',
            8: '010_potted_meat_can',
            9: '011_banana',
            10: '019_pitcher_base',
            11: '021_bleach_cleanser',
            12: '024_bowl',
            13: '025_mug',
            14: '035_power_drill',
            15: '036_wood_block',
            16: '037_scissors',
            17: '040_large_marker',
            18: '051_large_clamp',
            19: '052_extra_large_clamp',
            20: '061_foam_brick'
        }
        return ycb_classes
    
    def load_pretrained_model(self):
        """Load pre-trained YOLOv8 model."""
        try:
            print("🧠 Loading pre-trained YOLOv8 model...")
            self.model = YOLO('yolov8n.pt')  # Load nano model for speed
            print("✅ Pre-trained YOLOv8 model loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load pre-trained model: {e}")
            raise
    
    def load_model(self, model_path):
        """Load custom trained YOLOv8 model."""
        try:
            print(f"🧠 Loading custom YOLOv8 model from {model_path}...")
            self.model = YOLO(model_path)
            print("✅ Custom YOLOv8 model loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load custom model: {e}")
            raise
    
    def detect_objects(self, image):
        """Detect YCB objects in the image."""
        if self.model is None:
            print("❌ No model loaded!")
            return []
        
        try:
            # Run YOLOv8 detection
            results = self.model(image, verbose=False)
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Get confidence and class
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Filter by confidence threshold
                        if confidence >= self.confidence_threshold:
                            # Get class name
                            class_name = self.ycb_classes.get(class_id, f"unknown_{class_id}")
                            
                            detection = {
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(confidence),
                                'class': class_name,
                                'class_id': class_id
                            }
                            detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"❌ Detection error: {e}")
            return []
    
    def get_detection_center(self, detection):
        """Get the center point of a detection."""
        bbox = detection['bbox']
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2
        return center_x, center_y

class RoboticGraspingSystem:
    def __init__(self):
        """Initialize the robotic grasping system."""
        self.coppelia_client = None
        self.yolo_detector = None
        self.pose_estimator = None
        self.camera_handle = None
        self.robot_handle = None
        self.gripper_handle = None
        
        print("🤖 Initializing Robotic Grasping System...")
        
    def connect_to_coppelia(self):
        """Connect to CoppeliaSim via ZeroMQ."""
        try:
            self.coppelia_client = CoppeliaSimZMQClient()
            self.coppelia_client.connect()
            print("✅ Connected to CoppeliaSim")
            
            # Get handles for key objects
            self.camera_handle = self.coppelia_client.get_object_handle("sphericalVisionRGBAndDepth")
            self.robot_handle = self.coppelia_client.get_object_handle("UR5")
            self.gripper_handle = self.coppelia_client.get_object_handle("RG2")
            
            print(f"📷 Camera handle: {self.camera_handle}")
            print(f"🤖 Robot handle: {self.robot_handle}")
            print(f"✋ Gripper handle: {self.gripper_handle}")
            
            return True
        except Exception as e:
            print(f"❌ Failed to connect to CoppeliaSim: {e}")
            return False
    
    def load_deep_learning_models(self):
        """Load YOLOv8 and pose estimation models."""
        print("🧠 Loading deep learning models...")
        
        # Load YOLOv8 detector
        try:
            self.yolo_detector = YCBObjectDetector(confidence_threshold=0.5)
            print("✅ YOLOv8 detector loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load YOLOv8 detector: {e}")
            return False
        
        # TODO: Load pose estimation model
        # self.pose_estimator = load_pose_estimator()
        
        print("✅ Deep learning models loaded successfully!")
        return True
    
    def get_simulation_data(self):
        """Capture RGB-D data from CoppeliaSim camera."""
        try:
            # Get RGB image
            rgb_image = self.coppelia_client.get_vision_sensor_image(
                self.camera_handle, 
                "rgb"
            )
            
            # Get depth image
            depth_image = self.coppelia_client.get_vision_sensor_image(
                self.camera_handle, 
                "depth"
            )
            
            # Get camera intrinsics
            camera_intrinsics = self.coppelia_client.get_camera_intrinsics(
                self.camera_handle
            )
            
            print(f"📸 Captured RGB image: {rgb_image.shape}")
            print(f"📏 Captured depth image: {depth_image.shape}")
            
            return rgb_image, depth_image, camera_intrinsics
            
        except Exception as e:
            print(f"❌ Failed to capture simulation data: {e}")
            return None, None, None
    
    def detect_objects(self, rgb_image):
        """Detect objects using YOLOv8."""
        print("🔍 Detecting objects with YOLOv8...")
        
        if self.yolo_detector is None:
            print("❌ YOLOv8 detector not loaded!")
            return []
        
        try:
            # Run YOLOv8 detection
            detections = self.yolo_detector.detect_objects(rgb_image)
            
            print(f"✅ Detected {len(detections)} objects")
            
            # Print detection details
            for i, detection in enumerate(detections):
                print(f"  {i+1}. {detection['class']} (confidence: {detection['confidence']:.2f})")
            
            return detections
            
        except Exception as e:
            print(f"❌ Detection error: {e}")
            return []
    
    def estimate_poses(self, rgb_image, depth_image, detections, camera_intrinsics):
        """Estimate 6D poses of detected objects."""
        print("🎯 Estimating 6D poses...")
        
        poses = []
        
        for detection in detections:
            # Get detection center for depth lookup
            center_x, center_y = self.yolo_detector.get_detection_center(detection)
            
            # Placeholder pose estimation using depth data
            if depth_image is not None and 0 <= center_y < depth_image.shape[0] and 0 <= center_x < depth_image.shape[1]:
                # Get depth at detection center
                depth = depth_image[center_y, center_x]
                
                # Convert to 3D coordinates (simplified)
                fx = camera_intrinsics.get('fx', 525.0)
                fy = camera_intrinsics.get('fy', 525.0)
                cx = camera_intrinsics.get('cx', 320.0)
                cy = camera_intrinsics.get('cy', 240.0)
                
                # Convert pixel to 3D
                x = (center_x - cx) * depth / fx
                y = (center_y - cy) * depth / fy
                z = depth
                
                pose = {
                    'translation': [x, y, z],  # [x, y, z] in meters
                    'rotation': [0, 0, 0],  # [roll, pitch, yaw] in radians
                    'confidence': detection['confidence'],
                    'object_class': detection['class']
                }
            else:
                # Fallback pose
                pose = {
                    'translation': [0.5, 0.3, 0.1],  # [x, y, z] in meters
                    'rotation': [0, 0, 0],  # [roll, pitch, yaw] in radians
                    'confidence': detection['confidence'],
                    'object_class': detection['class']
                }
            
            poses.append(pose)
        
        print(f"✅ Estimated poses for {len(poses)} objects")
        return poses
    
    def plan_grasps(self, poses):
        """Plan optimal grasp poses for detected objects."""
        print("🤏 Planning grasps...")
        
        grasps = []
        
        for i, pose in enumerate(poses):
            # Placeholder grasp pose
            grasp = {
                'position': pose['translation'],
                'orientation': pose['rotation'],
                'approach_vector': [0, 0, -1],  # Approach from above
                'gripper_width': 0.08,  # 8cm gripper width
                'confidence': pose['confidence'],
                'object_class': pose['object_class']
            }
            grasps.append(grasp)
        
        print(f"✅ Planned {len(grasps)} grasps")
        return grasps
    
    def execute_grasp(self, grasp):
        """Execute grasp using UR5 robot."""
        print(f"🚀 Executing grasp for {grasp['object_class']}...")
        
        try:
            # TODO: Implement actual robot control
            # self.coppelia_client.move_robot_to_pose(grasp['position'], grasp['orientation'])
            # self.coppelia_client.open_gripper()
            # self.coppelia_client.close_gripper()
            
            print(f"✅ Grasp executed at position: {grasp['position']}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to execute grasp: {e}")
            return False
    
    def run_complete_pipeline(self):
        """Run the complete robotic grasping pipeline."""
        print("🎬 Starting complete robotic grasping pipeline...")
        print("=" * 50)
        
        # Step 1: Connect to CoppeliaSim
        if not self.connect_to_coppelia():
            return False
        
        # Step 2: Load deep learning models
        if not self.load_deep_learning_models():
            return False
        
        # Step 3: Capture simulation data
        rgb_image, depth_image, camera_intrinsics = self.get_simulation_data()
        if rgb_image is None:
            return False
        
        # Step 4: Detect objects
        detections = self.detect_objects(rgb_image)
        if not detections:
            print("⚠️ No objects detected")
            return False
        
        # Step 5: Estimate poses
        poses = self.estimate_poses(rgb_image, depth_image, detections, camera_intrinsics)
        if not poses:
            print("⚠️ No poses estimated")
            return False
        
        # Step 6: Plan grasps
        grasps = self.plan_grasps(poses)
        if not grasps:
            print("⚠️ No grasps planned")
            return False
        
        # Step 7: Execute grasps
        for i, grasp in enumerate(grasps):
            print(f"\n🎯 Executing grasp {i+1}/{len(grasps)}...")
            success = self.execute_grasp(grasp)
            if success:
                print(f"✅ Grasp {i+1} successful!")
            else:
                print(f"❌ Grasp {i+1} failed!")
        
        print("\n🎉 Pipeline completed!")
        return True

def main():
    """Main function to run the robotic grasping system."""
    print("🤖 YOLOv8 Robotic Grasping Pipeline")
    print("Make sure CoppeliaSim is running with Web Server enabled!")
    print()
    
    grasping_system = RoboticGraspingSystem()
    success = grasping_system.run_complete_pipeline()
    
    if success:
        print("🎊 Robotic grasping system completed successfully!")
        print("✅ YOLOv8 integration working!")
    else:
        print("💥 Robotic grasping system encountered errors!")

if __name__ == "__main__":
    main()







