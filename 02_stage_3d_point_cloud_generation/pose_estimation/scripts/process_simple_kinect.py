#!/usr/bin/env python3
"""
Process Simple Kinect Captures
Process captured data from simple kinect captures
"""

import cv2
import numpy as np
import os
import glob
from pathlib import Path

def process_simple_kinect():
    """Process simple kinect capture data."""
    
    print("🔄 Processing Simple Kinect Captures")
    print("=" * 35)
    
    # Look for captured files
    captured_files = glob.glob("kinect_capture_*_rgb.txt")
    
    if not captured_files:
        print("❌ No captured files found!")
        print("💡 Please run the simple kinect capture script in CoppeliaSim first")
        return
    
    print(f"📊 Found {len(captured_files)} captured files")
    
    # Create output directory
    output_dir = "simple_kinect_captures/processed_images"
    os.makedirs(output_dir, exist_ok=True)
    
    processed_count = 0
    failed_count = 0
    
    for file_path in sorted(captured_files):
        try:
            print(f"\n📸 Processing: {file_path}")
            
            # Read data
            with open(file_path, 'rb') as f:
                data = f.read()
            
            print(f"   Data size: {len(data)} bytes")
            
            # Check if it's the right size for 64x48x3 (from your working image)
            if len(data) == 9216:  # 64x48x3 = 9216 bytes
                # Convert to image
                data_array = np.frombuffer(data, dtype=np.uint8)
                data_array = data_array.reshape(48, 64, 3)
                
                # Convert BGR to RGB
                rgb_image = cv2.cvtColor(data_array, cv2.COLOR_BGR2RGB)
                
                # Save image
                filename = Path(file_path).stem + ".jpg"
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, rgb_image)
                
                print(f"   ✅ Saved: {filename}")
                print(f"   Shape: {rgb_image.shape}")
                print(f"   Min/Max: {rgb_image.min()}/{rgb_image.max()}")
                
                processed_count += 1
                
            else:
                print(f"   ❌ Unexpected data size: {len(data)} bytes")
                print(f"   💡 Expected 9216 bytes for 64x48x3 resolution")
                failed_count += 1
                
        except Exception as e:
            print(f"   ❌ Error processing {file_path}: {e}")
            failed_count += 1
    
    print(f"\n📊 Processing Summary:")
    print(f"   - Processed: {processed_count} images")
    print(f"   - Failed: {failed_count} images")
    print(f"   - Output directory: {output_dir}")
    
    if processed_count > 0:
        print(f"\n🎯 Images ready for annotation!")
        print(f"💡 Check the images in {output_dir}/")
        print(f"💡 Annotate them with bounding boxes for your objects:")
        print(f"   - Master Chef can (blue can)")
        print(f"   - Cracker box (red CHEEZ-IT box)")
        print(f"   - Mug (red mug)")
        print(f"   - Banana (yellow banana)")
        print(f"   - Mustard bottle (yellow bottle)")
        
        # Show first few images
        print(f"\n📸 First few captured images:")
        image_files = sorted(glob.glob(f"{output_dir}/*.jpg"))
        for i, img_file in enumerate(image_files[:5]):
            print(f"   {i+1}. {os.path.basename(img_file)}")
        
        if len(image_files) > 5:
            print(f"   ... and {len(image_files) - 5} more")
            
        # Clean up raw files
        print(f"\n🧹 Cleaning up raw files...")
        for file_path in captured_files:
            try:
                os.remove(file_path)
                print(f"   Removed: {file_path}")
            except:
                pass
                
    else:
        print(f"\n❌ No images processed")

if __name__ == "__main__":
    process_simple_kinect()




