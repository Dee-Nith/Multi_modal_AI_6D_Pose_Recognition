#!/usr/bin/env python3
"""
Test Label Format
Debug the YCB label image format
"""

import cv2
import numpy as np
from pathlib import Path

def test_label_image():
    """Test a sample label image."""
    print("🧪 Testing YCB Label Image Format")
    print("=" * 40)
    
    # Find a sample label image
    data_root = Path("Data_sets/data_syn")
    label_files = list(data_root.glob("*label.png"))
    
    if not label_files:
        print("❌ No label files found!")
        return
    
    # Test first few label files
    for i, label_path in enumerate(label_files[:3]):
        print(f"\n📄 Testing {label_path.name}...")
        
        # Read label image
        label_img = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        if label_img is None:
            print("  ❌ Could not read image")
            continue
        
        print(f"  📏 Image shape: {label_img.shape}")
        print(f"  📊 Value range: {label_img.min()} to {label_img.max()}")
        
        # Find unique values
        unique_values = np.unique(label_img)
        print(f"  🔢 Unique values: {unique_values}")
        
        # Count non-zero pixels
        non_zero = np.count_nonzero(label_img)
        total_pixels = label_img.size
        print(f"  📈 Non-zero pixels: {non_zero}/{total_pixels} ({non_zero/total_pixels*100:.1f}%)")
        
        # Check if there are objects (non-zero values)
        if len(unique_values) > 1:  # More than just background
            print(f"  ✅ Found objects with IDs: {unique_values[unique_values > 0]}")
        else:
            print(f"  ⚠️ No objects found (only background)")
    
    # Test corresponding color image
    print(f"\n🎨 Testing corresponding color image...")
    color_path = data_root / "000003-color.jpg"
    if color_path.exists():
        color_img = cv2.imread(str(color_path))
        print(f"  📏 Color image shape: {color_img.shape}")
        print(f"  📊 Color image type: {color_img.dtype}")
    else:
        print("  ❌ Color image not found")

if __name__ == "__main__":
    test_label_image()







