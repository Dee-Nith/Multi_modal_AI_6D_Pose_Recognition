#!/usr/bin/env python3
"""
📁 Organize Captured Images from CoppeliaSim
===========================================
Move all captured images from /tmp to a separate organized folder.
"""

import os
import shutil
import cv2
import numpy as np
import glob
from pathlib import Path

def organize_captured_images():
    """Organize all captured images from CoppeliaSim."""
    print("📁 Organizing Captured Images from CoppeliaSim")
    print("=" * 50)
    
    # Create organized directory structure
    organized_dir = "organized_captured_images"
    rgb_dir = os.path.join(organized_dir, "rgb_images")
    depth_dir = os.path.join(organized_dir, "depth_images")
    jpg_dir = os.path.join(organized_dir, "jpg_images")
    
    # Create directories
    os.makedirs(organized_dir, exist_ok=True)
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(jpg_dir, exist_ok=True)
    
    print(f"📂 Created organized directory: {organized_dir}")
    
    # Find all captured files in /tmp
    tmp_dir = "/tmp"
    rgb_files = glob.glob(os.path.join(tmp_dir, "auto_kinect_*_rgb.txt"))
    depth_files = glob.glob(os.path.join(tmp_dir, "auto_kinect_*_depth.txt"))
    
    print(f"🔍 Found {len(rgb_files)} RGB files and {len(depth_files)} depth files")
    
    # Process RGB files
    processed_images = []
    
    for rgb_file in rgb_files:
        filename = os.path.basename(rgb_file)
        image_id = filename.replace("auto_kinect_", "").replace("_rgb.txt", "")
        
        try:
            image_id = int(image_id)
        except ValueError:
            continue
        
        print(f"\n📸 Processing image {image_id}...")
        
        # Copy RGB file
        dst_rgb = os.path.join(rgb_dir, f"auto_kinect_{image_id}_rgb.txt")
        shutil.copy2(rgb_file, dst_rgb)
        print(f"  ✅ Copied RGB: {filename}")
        
        # Convert to JPG
        try:
            # Load raw RGB data
            with open(rgb_file, 'rb') as f:
                rgb_data = np.frombuffer(f.read(), dtype=np.uint8)
            
            # Reshape to image
            rgb_image = rgb_data.reshape(480, 640, 3)
            
            # Save as JPG
            jpg_path = os.path.join(jpg_dir, f"auto_kinect_{image_id}_rgb.jpg")
            cv2.imwrite(jpg_path, rgb_image)
            print(f"  ✅ Converted to JPG: auto_kinect_{image_id}_rgb.jpg")
            
        except Exception as e:
            print(f"  ❌ Error converting to JPG: {e}")
        
        # Find corresponding depth file
        depth_file = os.path.join(tmp_dir, f"auto_kinect_{image_id}_depth.txt")
        if os.path.exists(depth_file):
            dst_depth = os.path.join(depth_dir, f"auto_kinect_{image_id}_depth.txt")
            shutil.copy2(depth_file, dst_depth)
            print(f"  ✅ Copied depth: auto_kinect_{image_id}_depth.txt")
        else:
            print(f"  ⚠️ No depth file found for image {image_id}")
        
        processed_images.append(image_id)
    
    # Process depth files (in case there are extra ones)
    for depth_file in depth_files:
        filename = os.path.basename(depth_file)
        image_id = filename.replace("auto_kinect_", "").replace("_depth.txt", "")
        
        try:
            image_id = int(image_id)
        except ValueError:
            continue
        
        # Check if we already processed this
        if image_id not in processed_images:
            print(f"\n📸 Processing depth-only image {image_id}...")
            
            dst_depth = os.path.join(depth_dir, f"auto_kinect_{image_id}_depth.txt")
            shutil.copy2(depth_file, dst_depth)
            print(f"  ✅ Copied depth: {filename}")
    
    # Create summary
    print(f"\n📊 Organization Summary:")
    print(f"  📁 Organized directory: {organized_dir}")
    print(f"  📸 Total images processed: {len(processed_images)}")
    print(f"  🖼️ RGB files: {len(os.listdir(rgb_dir))}")
    print(f"  📏 Depth files: {len(os.listdir(depth_dir))}")
    print(f"  🖼️ JPG files: {len(os.listdir(jpg_dir))}")
    
    # Create a summary file
    summary_file = os.path.join(organized_dir, "summary.txt")
    with open(summary_file, 'w') as f:
        f.write("Captured Images Summary\n")
        f.write("=" * 30 + "\n")
        f.write(f"Total images: {len(processed_images)}\n")
        f.write(f"Image IDs: {sorted(processed_images)}\n")
        f.write(f"RGB files: {len(os.listdir(rgb_dir))}\n")
        f.write(f"Depth files: {len(os.listdir(depth_dir))}\n")
        f.write(f"JPG files: {len(os.listdir(jpg_dir))}\n")
    
    print(f"  📄 Summary saved: {summary_file}")
    
    # List all organized images
    print(f"\n📋 Organized Images:")
    for image_id in sorted(processed_images):
        print(f"  • Image {image_id}: RGB + Depth + JPG")
    
    print(f"\n🎉 Organization complete!")
    print(f"📁 All images are now in: {organized_dir}")
    print(f"🖼️ JPG images ready for annotation: {jpg_dir}")

if __name__ == "__main__":
    organize_captured_images()




