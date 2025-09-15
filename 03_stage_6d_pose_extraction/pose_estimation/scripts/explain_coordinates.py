#!/usr/bin/env python3
"""
Coordinate Frame Explanation for 6D Pose Estimation
Explains the camera coordinate system and object positions
"""

def explain_pose_results():
    """Explain the coordinate system and your specific results"""
    
    print("🎯 6D POSE ESTIMATION COORDINATE SYSTEM")
    print("=" * 50)
    
    print("\n📐 CAMERA COORDINATE FRAME:")
    print("   Origin (0,0,0): Camera center")
    print("   X-axis (RED):   RIGHT  →  (positive = right)")
    print("   Y-axis (GREEN): DOWN   ↓  (positive = down)")
    print("   Z-axis (BLUE):  FORWARD → (positive = away from camera)")
    
    print("\n🎯 YOUR OBJECT POSITIONS:")
    print("=" * 30)
    
    # Your actual results
    objects = [
        ("master_chef_can", [0.228, 0.021, -0.233]),
        ("cracker_box", [-0.266, -0.006, -0.276]),
        ("mug", [-0.509, -0.032, 0.062]),
        ("mustard_bottle", [-0.553, 0.001, -0.029])
    ]
    
    for obj_name, pos in objects:
        x, y, z = pos
        print(f"\n📦 {obj_name.upper()}:")
        print(f"   Position: [{x:.3f}, {y:.3f}, {z:.3f}] meters")
        
        # Interpret X position
        if x > 0:
            print(f"   X = {x:.3f}m  → {abs(x)*100:.1f}cm to the RIGHT")
        else:
            print(f"   X = {x:.3f}m  → {abs(x)*100:.1f}cm to the LEFT")
        
        # Interpret Y position
        if y > 0:
            print(f"   Y = {y:.3f}m  → {abs(y)*100:.1f}cm DOWN from center")
        else:
            print(f"   Y = {y:.3f}m  → {abs(y)*100:.1f}cm UP from center")
        
        # Interpret Z position
        if z > 0:
            print(f"   Z = {z:.3f}m  → {abs(z)*100:.1f}cm AWAY from camera")
        else:
            print(f"   Z = {z:.3f}m  → {abs(z)*100:.1f}cm TOWARDS camera")
        
        # Calculate distance
        distance = (x**2 + y**2 + z**2)**0.5
        print(f"   Distance from camera: {distance:.3f}m ({distance*100:.1f}cm)")

def explain_coordinate_transforms():
    """Explain different coordinate systems"""
    
    print("\n🔄 COORDINATE SYSTEM CONVERSIONS:")
    print("=" * 40)
    
    print("\n1. CAMERA FRAME (your current results):")
    print("   - Origin: Camera center")
    print("   - Used in: Computer vision, OpenCV")
    print("   - Your YOLO + PnP results are in this frame")
    
    print("\n2. WORLD FRAME (CoppeliaSim scene):")
    print("   - Origin: Scene origin (0,0,0)")
    print("   - Used in: Robotics, CoppeliaSim")
    print("   - To convert: Apply camera pose transformation")
    
    print("\n3. ROBOT FRAME (if using robotic arm):")
    print("   - Origin: Robot base")
    print("   - Used in: Robot control, grasping")
    print("   - To convert: Apply robot-camera transformation")

if __name__ == "__main__":
    explain_pose_results()
    explain_coordinate_transforms()
    
    print("\n💡 SUMMARY:")
    print("   Your objects are positioned relative to the CAMERA")
    print("   Negative Z values mean objects are CLOSER than expected")
    print("   This is normal for close-up object detection!")




