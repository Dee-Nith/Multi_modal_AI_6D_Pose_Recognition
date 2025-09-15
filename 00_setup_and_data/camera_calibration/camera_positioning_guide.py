#!/usr/bin/env python3
"""
Camera Positioning Guide for CoppeliaSim
Helps position the camera to capture objects properly
"""

import cv2
import numpy as np
import os

def show_camera_guide():
    """Show camera view and provide positioning guidance."""
    
    image_path = "http_camera_capture.jpg"
    
    if not os.path.exists(image_path):
        print(f"❌ No camera image found: {image_path}")
        print(f"💡 Please run the Lua script in CoppeliaSim first")
        return
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return
    
    print(f"📸 Camera View Analysis:")
    print(f"=" * 40)
    print(f"   Image size: {image.shape[1]}x{image.shape[0]} pixels")
    print(f"   Aspect ratio: {image.shape[1]/image.shape[0]:.2f}")
    
    # Analyze image content
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Count edge pixels (indicates object boundaries)
    edge_pixels = np.sum(edges > 0)
    total_pixels = edges.shape[0] * edges.shape[1]
    edge_density = edge_pixels / total_pixels
    
    print(f"   Edge density: {edge_density:.3f} ({edge_pixels} edge pixels)")
    
    if edge_density < 0.01:
        print(f"   ⚠️  Very few edges detected - camera might be pointing at empty space")
    elif edge_density < 0.05:
        print(f"   ⚠️  Low edge density - objects might be too small or far away")
    elif edge_density > 0.2:
        print(f"   ✅ Good edge density - objects should be visible")
    else:
        print(f"   ✅ Moderate edge density - objects might be visible")
    
    # Show the image with grid overlay
    display_image = image.copy()
    height, width = display_image.shape[:2]
    
    # Draw grid
    grid_size = 64  # 8x4 grid for 512x256 image
    for i in range(0, width, grid_size):
        cv2.line(display_image, (i, 0), (i, height), (128, 128, 128), 1)
    for i in range(0, height, grid_size):
        cv2.line(display_image, (0, i), (width, i), (128, 128, 128), 1)
    
    # Add text overlay
    cv2.putText(display_image, "Camera View", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display_image, f"Size: {width}x{height}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(display_image, f"Edges: {edge_pixels}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Save the annotated image
    cv2.imwrite("camera_analysis.jpg", display_image)
    print(f"   💾 Analysis saved as: camera_analysis.jpg")
    
    # Show positioning tips
    print(f"\n🎯 Camera Positioning Tips:")
    print(f"=" * 40)
    print(f"   1. 📍 Move camera closer to objects")
    print(f"   2. 📐 Point camera directly at objects (not at angle)")
    print(f"   3. 💡 Ensure good lighting in the scene")
    print(f"   4. 🎯 Focus on the blue can and red Cheez-It box")
    print(f"   5. 📏 Objects should take up at least 10% of the image")
    print(f"   6. 🔄 Try different camera positions")
    
    # Show the image
    try:
        # Resize for display
        if width > 800 or height > 600:
            scale = min(800/width, 600/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            display_image = cv2.resize(display_image, (new_width, new_height))
        
        cv2.imshow("Camera Analysis", display_image)
        print(f"\n🖼️  Camera analysis displayed. Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"⚠️ Could not display image: {e}")
        print(f"💡 Check the saved image: camera_analysis.jpg")

if __name__ == "__main__":
    show_camera_guide()




