#!/usr/bin/env python3
"""
Debug script to check CoppeliaSim image files
"""

import os
import glob
import time

def check_image_files():
    """Check what image files are available"""
    print("🔍 Checking for CoppeliaSim image files...")
    
    # Check different locations
    locations = [
        "/tmp/",
        "./",
        "../",
        "../../"
    ]
    
    patterns = [
        "auto_kinect_*_rgb.txt",
        "coppelia_*_rgb.txt",
        "kinect_*_rgb.txt",
        "*_rgb.txt"
    ]
    
    found_files = []
    
    for location in locations:
        for pattern in patterns:
            full_pattern = os.path.join(location, pattern)
            files = glob.glob(full_pattern)
            
            for file in files:
                try:
                    file_size = os.path.getsize(file)
                    mod_time = os.path.getmtime(file)
                    mod_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mod_time))
                    
                    found_files.append({
                        'path': file,
                        'size': file_size,
                        'modified': mod_time_str
                    })
                    
                    print(f"📁 Found: {file}")
                    print(f"   Size: {file_size} bytes")
                    print(f"   Modified: {mod_time_str}")
                    
                    # Try to read first few bytes
                    try:
                        with open(file, 'rb') as f:
                            first_bytes = f.read(100)
                            print(f"   First 100 bytes: {first_bytes[:50]}...")
                    except Exception as e:
                        print(f"   Error reading file: {e}")
                    
                    print()
                    
                except Exception as e:
                    print(f"❌ Error checking {file}: {e}")
    
    if not found_files:
        print("❌ No image files found!")
        print("💡 Make sure CoppeliaSim is capturing images")
        print("💡 Check if your capture script is running")
    else:
        print(f"✅ Found {len(found_files)} image files")
    
    return found_files

def test_image_processing(file_path):
    """Test processing a specific image file"""
    print(f"🧪 Testing image processing for: {file_path}")
    
    try:
        # Read as binary
        with open(file_path, 'rb') as f:
            data = f.read()
        
        print(f"   File size: {len(data)} bytes")
        
        # Try different interpretations
        interpretations = [
            ("Raw binary", data),
            ("UTF-8 text", data.decode('utf-8', errors='ignore')),
            ("ASCII text", data.decode('ascii', errors='ignore'))
        ]
        
        for name, interpretation in interpretations:
            if isinstance(interpretation, str):
                print(f"   {name}: {interpretation[:100]}...")
            else:
                print(f"   {name}: {interpretation[:50]}...")
        
        # Try to convert to numpy array
        try:
            import numpy as np
            image_data = np.frombuffer(data, dtype=np.uint8)
            print(f"   As numpy array: {image_data.shape}, min={image_data.min()}, max={image_data.max()}")
            
            # Try different shapes
            possible_shapes = [
                (480, 640, 3),  # 640x480x3
                (48, 64, 3),    # 64x48x3
                (640, 480, 3),  # 480x640x3
                (64, 48, 3),    # 48x64x3
            ]
            
            for shape in possible_shapes:
                total_pixels = shape[0] * shape[1] * shape[2]
                if len(image_data) == total_pixels:
                    print(f"   ✅ Possible shape: {shape}")
                    break
            else:
                print(f"   ⚠️ No matching shape found for {len(image_data)} pixels")
                
        except Exception as e:
            print(f"   ❌ Error converting to numpy: {e}")
            
    except Exception as e:
        print(f"❌ Error testing file: {e}")

if __name__ == "__main__":
    print("🔍 CoppeliaSim Image File Debug")
    print("=" * 40)
    
    # Check for image files
    files = check_image_files()
    
    if files:
        # Test the most recent file
        most_recent = max(files, key=lambda x: x['modified'])
        print(f"\n🧪 Testing most recent file: {most_recent['path']}")
        test_image_processing(most_recent['path'])
    
    print("\n💡 Next steps:")
    print("1. Make sure CoppeliaSim is running")
    print("2. Run your capture script (e.g., auto_kinect_capture_increment.lua)")
    print("3. Check if new image files are created")
    print("4. Run this debug script again to verify")




