#!/usr/bin/env python3
"""
📸 Show Raw Image
================
Display the raw image being used for detection without any processing.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def show_raw_image(image_id):
    """Show the raw image without any processing."""
    print(f"📸 Loading raw image {image_id}...")
    
    # Load raw RGB image
    rgb_file = f"/tmp/auto_kinect_{image_id}_rgb.txt"
    
    if not os.path.exists(rgb_file):
        print(f"❌ Image file not found: {rgb_file}")
        return None
    
    # Load raw RGB data
    with open(rgb_file, 'rb') as f:
        rgb_data = np.frombuffer(f.read(), dtype=np.uint8)
    
    print(f"📊 Raw data shape: {rgb_data.shape}")
    print(f"📊 Raw data type: {rgb_data.dtype}")
    print(f"📊 Raw data range: {rgb_data.min()} to {rgb_data.max()}")
    
    # Reshape to image
    rgb_image = rgb_data.reshape(480, 640, 3)
    print(f"📸 Reshaped image shape: {rgb_image.shape}")
    
    # Show raw image statistics
    print(f"📊 Image statistics:")
    print(f"  - Mean RGB: [{np.mean(rgb_image[:,:,0]):.1f}, {np.mean(rgb_image[:,:,1]):.1f}, {np.mean(rgb_image[:,:,2]):.1f}]")
    print(f"  - Min RGB: [{rgb_image[:,:,0].min()}, {rgb_image[:,:,1].min()}, {rgb_image[:,:,2].min()}]")
    print(f"  - Max RGB: [{rgb_image[:,:,0].max()}, {rgb_image[:,:,1].max()}, {rgb_image[:,:,2].max()}]")
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Raw BGR image (as loaded)
    axes[0].imshow(rgb_image)
    axes[0].set_title(f'Raw Image {image_id} (BGR as loaded)')
    axes[0].axis('off')
    
    # 2. Converted to RGB
    rgb_image_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    axes[1].imshow(rgb_image_rgb)
    axes[1].set_title(f'Image {image_id} (Converted to RGB)')
    axes[1].axis('off')
    
    # 3. Individual color channels
    # Create a composite showing R, G, B channels
    rgb_channels = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb_channels[:,:,0] = rgb_image[:,:,2]  # Red channel
    rgb_channels[:,:,1] = rgb_image[:,:,1]  # Green channel  
    rgb_channels[:,:,2] = rgb_image[:,:,0]  # Blue channel
    
    axes[2].imshow(rgb_channels)
    axes[2].set_title(f'Image {image_id} (RGB Channels)')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save the raw image
    output_path = f"raw_image_{image_id}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved raw image: {output_path}")
    
    plt.show()
    
    return rgb_image

def show_all_raw_images(image_ids=[22, 23, 24, 32]):
    """Show all raw images."""
    print("📸 Showing Raw Images...")
    
    for image_id in image_ids:
        print(f"\n{'='*50}")
        print(f"📸 IMAGE {image_id}")
        print(f"{'='*50}")
        show_raw_image(image_id)

if __name__ == "__main__":
    show_all_raw_images()




