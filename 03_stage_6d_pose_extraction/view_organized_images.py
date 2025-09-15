#!/usr/bin/env python3
"""
👁️ View Organized Images
=======================
View the organized images and show the difference between txt and jpg formats.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

def view_organized_images():
    """View the organized images and show format differences."""
    print("👁️ Viewing Organized Images")
    print("=" * 40)
    
    organized_dir = "organized_captured_images"
    rgb_dir = os.path.join(organized_dir, "rgb_images")
    jpg_dir = os.path.join(organized_dir, "jpg_images")
    
    # Check what we have
    rgb_files = glob.glob(os.path.join(rgb_dir, "*.txt"))
    jpg_files = glob.glob(os.path.join(jpg_dir, "*.jpg"))
    
    print(f"📊 Found {len(rgb_files)} RGB .txt files and {len(jpg_files)} JPG files")
    
    # Show a few examples
    sample_images = [1, 22, 23, 24, 32]  # Your test images
    
    for image_id in sample_images:
        print(f"\n📸 Image {image_id}:")
        
        # Check RGB .txt file
        rgb_txt_file = os.path.join(rgb_dir, f"auto_kinect_{image_id}_rgb.txt")
        if os.path.exists(rgb_txt_file):
            # Get file size
            file_size = os.path.getsize(rgb_txt_file)
            print(f"  📄 RGB .txt file: {file_size:,} bytes")
            
            # Read and convert to image
            try:
                with open(rgb_txt_file, 'rb') as f:
                    rgb_data = np.frombuffer(f.read(), dtype=np.uint8)
                
                # Reshape to image (480x640x3)
                rgb_image = rgb_data.reshape(480, 640, 3)
                print(f"  ✅ Successfully converted to image: {rgb_image.shape}")
                
            except Exception as e:
                print(f"  ❌ Error reading RGB file: {e}")
        else:
            print(f"  ❌ RGB .txt file not found")
        
        # Check JPG file
        jpg_file = os.path.join(jpg_dir, f"auto_kinect_{image_id}_rgb.jpg")
        if os.path.exists(jpg_file):
            file_size = os.path.getsize(jpg_file)
            print(f"  🖼️ JPG file: {file_size:,} bytes")
            
            # Load JPG image
            try:
                jpg_image = cv2.imread(jpg_file)
                jpg_image_rgb = cv2.cvtColor(jpg_image, cv2.COLOR_BGR2RGB)
                print(f"  ✅ Successfully loaded JPG: {jpg_image_rgb.shape}")
                
            except Exception as e:
                print(f"  ❌ Error reading JPG file: {e}")
        else:
            print(f"  ❌ JPG file not found")
    
    # Create a visual comparison
    print(f"\n🖼️ Creating visual comparison...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for i, image_id in enumerate([22, 23, 24]):
        # Load from RGB .txt
        rgb_txt_file = os.path.join(rgb_dir, f"auto_kinect_{image_id}_rgb.txt")
        jpg_file = os.path.join(jpg_dir, f"auto_kinect_{image_id}_rgb.jpg")
        
        if os.path.exists(rgb_txt_file) and os.path.exists(jpg_file):
            # Load from .txt
            with open(rgb_txt_file, 'rb') as f:
                rgb_data = np.frombuffer(f.read(), dtype=np.uint8)
            rgb_image = rgb_data.reshape(480, 640, 3)
            
            # Load from .jpg
            jpg_image = cv2.imread(jpg_file)
            jpg_image_rgb = cv2.cvtColor(jpg_image, cv2.COLOR_BGR2RGB)
            
            # Display
            axes[0, i].imshow(rgb_image)
            axes[0, i].set_title(f'Image {image_id}: From .txt file')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(jpg_image_rgb)
            axes[1, i].set_title(f'Image {image_id}: From .jpg file')
            axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('image_format_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n📋 Summary:")
    print(f"  📄 RGB .txt files: Raw binary data (need conversion)")
    print(f"  🖼️ JPG files: Ready for annotation in Roboflow")
    print(f"  📊 Both formats contain the same image data")
    print(f"  🎯 Use JPG files for annotation in Roboflow")

if __name__ == "__main__":
    view_organized_images()




