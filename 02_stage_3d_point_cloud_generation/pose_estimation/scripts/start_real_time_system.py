#!/usr/bin/env python3
"""
Start Real-time Robot Pick and Place System
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

def start_real_time_system():
    """Start the real-time robot pick and place system"""
    
    # Paths
    model_path = "../../coppelia_sim_dataset/runs/detect/train/weights/best.pt"
    calibration_path = "../calibration/coppelia_camera_calibration.json"
    models_dir = "../models"
    
    # Initialize robot system
    print("🤖 Starting Real-time Robot Pick and Place System...")
    robot_system = RobotPickPlace(model_path, calibration_path, models_dir)
    
    print("🎯 Real-time Robot Pick and Place System")
    print("=" * 50)
    print("📋 Instructions:")
    print("1. Start CoppeliaSim with your scene")
    print("2. Run your capture script to detect objects")
    print("3. Press 'e' to execute pick and place tasks")
    print("4. Press 'q' to quit, 's' to save current frame")
    print("5. Press 't' to test with working image")
    print()
    
    # Start monitoring loop
    frame_count = 0
    last_processed = None
    
    while True:
        # Find latest CoppeliaSim image
        latest_file = robot_system.find_latest_coppelia_image()
        
        if latest_file and latest_file != last_processed:
            # Process new image
            frame = robot_system.process_coppelia_image(latest_file)
            
            if frame is not None:
                # Detect objects
                detections = robot_system.detect_objects_realtime(frame)
                
                # Process detections and add to task queue
                robot_system.process_detections(detections)
                
                # Estimate poses for visualization
                poses = []
                for detection in detections:
                    pose = robot_system.estimate_pose_realtime(frame, detection)
                    if pose:
                        poses.append(pose)
                
                # Update FPS
                robot_system.update_fps()
                
                # Visualize results
                result_frame = robot_system.visualize_robot_control(frame, poses, robot_system.task_queue)
                
                # Display frame
                cv2.imshow('Real-time Robot Pick and Place System', result_frame)
                
                last_processed = latest_file
                frame_count += 1
                
                if frame_count % 10 == 0:
                    print(f"📸 Frame {frame_count}: {len(poses)} objects, {len(robot_system.task_queue)} tasks")
        
        # Handle key presses
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            print("🛑 Stopping real-time system...")
            break
        elif key == ord('s'):
            if 'result_frame' in locals():
                timestamp = int(time.time())
                save_path = f"../results/realtime_robot_{timestamp}.jpg"
                cv2.imwrite(save_path, result_frame)
                print(f"📸 Saved frame to: {save_path}")
        elif key == ord('e'):
            # Execute pending tasks
            if robot_system.task_queue:
                print(f"🚀 Executing {len(robot_system.task_queue)} pending tasks...")
                for task in robot_system.task_queue[:]:
                    if task['status'] == 'pending':
                        robot_system.execute_pick_task(task)
                        robot_system.task_queue.remove(task)
            else:
                print("📋 No pending tasks to execute")
        elif key == ord('t'):
            # Test with working image
            print("🧪 Testing with working image...")
            test_with_working_image(robot_system)
    
    # Cleanup
    cv2.destroyAllWindows()
    print(f"✅ Processed {frame_count} frames")
    print(f"📊 Completed {len(robot_system.completed_tasks)} pick and place tasks")

def test_with_working_image(robot_system):
    """Test with a working image"""
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
    save_path = "../results/test_realtime_system.jpg"
    cv2.imwrite(save_path, result_frame)
    print(f"📸 Result saved to: {save_path}")
    
    # Show the result
    cv2.imshow('Test Result', result_frame)
    cv2.waitKey(2000)  # Show for 2 seconds
    cv2.destroyAllWindows()
    
    print("✅ Test completed!")

if __name__ == "__main__":
    start_real_time_system()




