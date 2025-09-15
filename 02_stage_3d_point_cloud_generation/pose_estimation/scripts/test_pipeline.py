#!/usr/bin/env python3
"""
Test Robotic Grasping Pipeline
Tests the complete pipeline with CoppeliaSim
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.robotic_grasping_pipeline import RoboticGraspingSystem

def test_pipeline():
    """Test the complete robotic grasping pipeline."""
    print("🧪 Testing Robotic Grasping Pipeline")
    print("=" * 40)
    
    # Create grasping system
    grasping_system = RoboticGraspingSystem()
    
    # Test connection
    print("\n1️⃣ Testing CoppeliaSim connection...")
    if not grasping_system.connect_to_coppelia():
        print("❌ Connection failed - make sure CoppeliaSim is running!")
        return False
    print("✅ Connection successful!")
    
    # Test model loading
    print("\n2️⃣ Testing deep learning model loading...")
    if not grasping_system.load_deep_learning_models():
        print("❌ Model loading failed!")
        return False
    print("✅ Models loaded successfully!")
    
    # Test data capture
    print("\n3️⃣ Testing simulation data capture...")
    rgb_image, depth_image, camera_intrinsics = grasping_system.get_simulation_data()
    if rgb_image is None:
        print("❌ Data capture failed!")
        return False
    print("✅ Data capture successful!")
    
    # Test object detection
    print("\n4️⃣ Testing object detection...")
    detections = grasping_system.detect_objects(rgb_image)
    if not detections:
        print("❌ Object detection failed!")
        return False
    print("✅ Object detection successful!")
    
    # Test pose estimation
    print("\n5️⃣ Testing pose estimation...")
    poses = grasping_system.estimate_poses(rgb_image, depth_image, detections, camera_intrinsics)
    if not poses:
        print("❌ Pose estimation failed!")
        return False
    print("✅ Pose estimation successful!")
    
    # Test grasp planning
    print("\n6️⃣ Testing grasp planning...")
    grasps = grasping_system.plan_grasps(poses)
    if not grasps:
        print("❌ Grasp planning failed!")
        return False
    print("✅ Grasp planning successful!")
    
    # Test grasp execution (simulation only)
    print("\n7️⃣ Testing grasp execution (simulation)...")
    for i, grasp in enumerate(grasps):
        success = grasping_system.execute_grasp(grasp)
        if success:
            print(f"✅ Grasp {i+1} execution successful!")
        else:
            print(f"❌ Grasp {i+1} execution failed!")
    
    print("\n🎉 All pipeline tests completed!")
    return True

def main():
    """Main test function."""
    print("🤖 Robotic Grasping Pipeline Test")
    print("Make sure CoppeliaSim is running with your scene loaded!")
    print()
    
    success = test_pipeline()
    
    if success:
        print("\n🎊 Pipeline test PASSED!")
        print("✅ Ready for full robotic grasping!")
    else:
        print("\n💥 Pipeline test FAILED!")
        print("❌ Check the errors above and fix them.")

if __name__ == "__main__":
    main()
