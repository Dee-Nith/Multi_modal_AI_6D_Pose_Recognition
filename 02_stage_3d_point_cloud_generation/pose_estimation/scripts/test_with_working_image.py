#!/usr/bin/env python3
"""
Test Robot Pick and Place with a working image
"""

import cv2
import numpy as np
import sys
import os

# Add the robot pick place system
sys.path.append('.')
from robot_pick_place import RobotPickPlace

def test_with_working_image():
    """Test the robot system with a working image"""
    
    # Paths
    model_path = "../../coppelia_sim_dataset/runs/detect/train/weights/best.pt"
    calibration_path = "../calibration/coppelia_camera_calibration.json"
    models_dir = "../models"
    
    # Initialize robot system
    print("🤖 Initializing Robot Pick and Place System...")
    robot_system = RobotPickPlace(model_path, calibration_path, models_dir)
    
    # Use a working image file
    working_image_path = "/tmp/simple_kinect_1_rgb.txt"
    
    if not os.path.exists(working_image_path):
        print(f"❌ Working image not found: {working_image_path}")
        return
    
    print(f"📸 Testing with image: {working_image_path}")
    
    # Process the image
    frame = robot_system.process_coppelia_image(working_image_path)
    
    if frame is None:
        print("❌ Failed to process image")
        return
    
    print(f"✅ Image processed successfully: {frame.shape}")
    
    # Detect objects
    detections = robot_system.detect_objects_realtime(frame)
    print(f"🎯 Detected {len(detections)} objects")
    
    # Process detections and add to task queue
    robot_system.process_detections(detections)
    print(f"📋 Added {len(robot_system.task_queue)} tasks to queue")
    
    # Estimate poses for visualization
    poses = []
    for detection in detections:
        pose = robot_system.estimate_pose_realtime(frame, detection)
        if pose:
            poses.append(pose)
            print(f"📐 Pose for {pose['class_name']}: pos={pose['translation'][:2]}, rot={pose['euler_degrees'][:2]}")
    
    # Visualize results
    result_frame = robot_system.visualize_robot_control(frame, poses, robot_system.task_queue)
    
    # Save result
    save_path = "../results/test_robot_system.jpg"
    cv2.imwrite(save_path, result_frame)
    print(f"📸 Result saved to: {save_path}")
    
    # Show the result
    cv2.imshow('Robot Pick and Place Test', result_frame)
    print("🖼️ Press any key to close the image window")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Execute tasks if any
    if robot_system.task_queue:
        print(f"\n🚀 Executing {len(robot_system.task_queue)} tasks...")
        for task in robot_system.task_queue[:]:
            if task['status'] == 'pending':
                robot_system.execute_pick_task(task)
                robot_system.task_queue.remove(task)
        
        print(f"✅ Completed {len(robot_system.completed_tasks)} pick and place tasks")
    else:
        print("📋 No tasks to execute")
    
    print("\n🎯 Test completed successfully!")

if __name__ == "__main__":
    print("🧪 Testing Robot Pick and Place System")
    print("=" * 40)
    test_with_working_image()




