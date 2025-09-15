#!/usr/bin/env python3
"""
Use Latest CoppeliaSim Capture
Processes the most recent image you captured
"""

import cv2
import numpy as np
import sys
import os
import time
import glob

# Add the robot pick place system
sys.path.append('.')
from robot_pick_place import RobotPickPlace

def use_latest_capture():
    """Use the latest captured image from CoppeliaSim"""
    
    print("📸 Using Latest CoppeliaSim Capture")
    print("=" * 40)
    
    # Paths
    model_path = "../../coppelia_sim_dataset/runs/detect/train/weights/best.pt"
    calibration_path = "../calibration/coppelia_camera_calibration.json"
    models_dir = "../models"
    
    # Initialize robot system
    print("🔄 Loading AI system...")
    robot_system = RobotPickPlace(model_path, calibration_path, models_dir)
    print("✅ AI system loaded successfully!")
    
    # Find the latest captured image
    print("🔍 Looking for latest captured image...")
    
    # Look for auto_kinect files in /tmp
    pattern = "/tmp/auto_kinect_*_rgb.txt"
    files = glob.glob(pattern)
    
    if not files:
        print("❌ No captured images found!")
        print("💡 Please run your CoppeliaSim capture script first")
        return
    
    # Get the most recent file
    latest_file = max(files, key=os.path.getmtime)
    file_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(latest_file)))
    
    print(f"📸 Found latest capture: {latest_file}")
    print(f"📅 Captured at: {file_time}")
    
    # Process the image
    print("🔄 Processing image...")
    frame = robot_system.process_coppelia_image(latest_file)
    
    if frame is None:
        print("❌ Failed to process image")
        print("💡 The image might be corrupted or in wrong format")
        return
    
    print(f"✅ Image processed: {frame.shape}")
    
    # Detect objects
    print("🎯 Detecting objects...")
    detections = robot_system.detect_objects_realtime(frame)
    print(f"✅ Detected {len(detections)} objects")
    
    # Show detection results
    for i, detection in enumerate(detections):
        print(f"  {i+1}. {detection['class_name']} (confidence: {detection['confidence']:.2f})")
    
    # Process detections and add to task queue
    robot_system.process_detections(detections)
    print(f"📋 Added {len(robot_system.task_queue)} tasks to queue")
    
    # Estimate poses
    print("📐 Estimating 6D poses...")
    poses = []
    for detection in detections:
        pose = robot_system.estimate_pose_realtime(frame, detection)
        if pose:
            poses.append(pose)
            print(f"  📐 {pose['class_name']}: pos=({pose['translation'][0]:.3f}, {pose['translation'][1]:.3f}, {pose['translation'][2]:.3f})")
    
    # Create visualization
    print("🎨 Creating visualization...")
    result_frame = robot_system.visualize_robot_control(frame, poses, robot_system.task_queue)
    
    # Save result
    timestamp = int(time.time())
    save_path = f"../results/latest_capture_{timestamp}.jpg"
    cv2.imwrite(save_path, result_frame)
    print(f"📸 Result saved to: {save_path}")
    
    # Show the result
    print("🖼️ Displaying your captured image with detections...")
    print("Press any key to continue...")
    cv2.imshow('Your Latest CoppeliaSim Capture', result_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Ask user if they want to execute tasks
    print("\n🤖 Robot Control Options:")
    print("1. Execute pick and place tasks")
    print("2. Show task details")
    print("3. Capture new image and process")
    print("4. Exit")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        print("\n🚀 Executing pick and place tasks...")
        if robot_system.task_queue:
            for task in robot_system.task_queue[:]:
                if task['status'] == 'pending':
                    print(f"\n🤖 Picking {task['object_pose']['class_name']}...")
                    robot_system.execute_pick_task(task)
                    robot_system.task_queue.remove(task)
            
            print(f"\n✅ Completed {len(robot_system.completed_tasks)} pick and place tasks!")
        else:
            print("📋 No tasks to execute")
    
    elif choice == "2":
        print("\n📋 Task Details:")
        for i, task in enumerate(robot_system.task_queue):
            print(f"  {i+1}. Pick {task['object_pose']['class_name']} (Priority: {task['priority']})")
            pose = task['object_pose']
            print(f"     Position: ({pose['translation'][0]:.3f}, {pose['translation'][1]:.3f}, {pose['translation'][2]:.3f})")
            print(f"     Rotation: ({pose['euler_degrees'][0]:.1f}°, {pose['euler_degrees'][1]:.1f}°, {pose['euler_degrees'][2]:.1f}°)")
    
    elif choice == "3":
        print("\n📸 Please capture a new image in CoppeliaSim...")
        input("Press Enter when you've captured the new image...")
        use_latest_capture()  # Recursive call to process new image
    
    print("\n🎯 Processing completed!")

if __name__ == "__main__":
    use_latest_capture()




