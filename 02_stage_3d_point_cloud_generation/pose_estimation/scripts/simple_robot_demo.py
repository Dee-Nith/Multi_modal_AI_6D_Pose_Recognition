#!/usr/bin/env python3
"""
Simple Robot Pick and Place Demo
Shows the system working with clear visual output
"""

import cv2
import numpy as np
import sys
import os
import time

# Add the robot pick place system
sys.path.append('.')
from robot_pick_place import RobotPickPlace

def simple_demo():
    """Simple demo showing robot pick and place system"""
    
    print("🤖 Simple Robot Pick and Place Demo")
    print("=" * 40)
    
    # Paths
    model_path = "../../coppelia_sim_dataset/runs/detect/train/weights/best.pt"
    calibration_path = "../calibration/coppelia_camera_calibration.json"
    models_dir = "../models"
    
    # Initialize robot system
    print("🔄 Loading AI system...")
    robot_system = RobotPickPlace(model_path, calibration_path, models_dir)
    print("✅ AI system loaded successfully!")
    
    # Test with working image
    working_image_path = "/tmp/simple_kinect_1_rgb.txt"
    
    if not os.path.exists(working_image_path):
        print(f"❌ Working image not found: {working_image_path}")
        print("💡 Please run your CoppeliaSim capture script first")
        return
    
    print(f"📸 Processing image: {working_image_path}")
    
    # Process the image
    frame = robot_system.process_coppelia_image(working_image_path)
    
    if frame is None:
        print("❌ Failed to process image")
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
    save_path = "../results/simple_demo_result.jpg"
    cv2.imwrite(save_path, result_frame)
    print(f"📸 Result saved to: {save_path}")
    
    # Show the result
    print("🖼️ Displaying result (press any key to continue)...")
    cv2.imshow('Robot Pick and Place Demo', result_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Ask user if they want to execute tasks
    print("\n🤖 Robot Control Options:")
    print("1. Execute pick and place tasks")
    print("2. Show task details")
    print("3. Exit")
    
    choice = input("Enter choice (1-3): ").strip()
    
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
    
    print("\n🎯 Demo completed successfully!")

if __name__ == "__main__":
    simple_demo()




