#!/usr/bin/env python3
"""
Auto Capture Training Data
Automatically captures and saves images from CoppeliaSim for training
"""

import os
import time
import shutil
from datetime import datetime
import cv2
import numpy as np

class AutoCaptureTrainingData:
    def __init__(self):
        self.captured_dir = "captured_images"
        self.max_images = 100  # Maximum number of images to capture
        self.capture_interval = 2  # Seconds between captures
        
        # Create captured images directory
        os.makedirs(self.captured_dir, exist_ok=True)
        
    def start_auto_capture(self):
        """Start automatic image capture for training data."""
        print("🎯 Auto Capture Training Data")
        print("=" * 40)
        print(f"📁 Saving images to: {self.captured_dir}")
        print(f"⏱️  Capture interval: {self.capture_interval} seconds")
        print(f"📸 Max images: {self.max_images}")
        print("\n💡 Instructions:")
        print("   1. Position camera in CoppeliaSim")
        print("   2. Run the Lua script to capture")
        print("   3. Move camera to new position")
        print("   4. Repeat until you have enough images")
        print("   5. Press Ctrl+C to stop")
        
        captured_count = 0
        
        try:
            while captured_count < self.max_images:
                print(f"\n📸 Ready for capture #{captured_count + 1}/{self.max_images}")
                print("🔄 Run the Lua script in CoppeliaSim now...")
                
                # Wait for user to run Lua script
                input("⏸️  Press Enter when ready to capture...")
                
                # Check for new camera capture
                if self._check_for_new_capture():
                    captured_count += 1
                    print(f"✅ Captured image #{captured_count}")
                else:
                    print("❌ No new capture detected")
                
                # Wait before next capture
                if captured_count < self.max_images:
                    print(f"⏳ Waiting {self.capture_interval} seconds...")
                    time.sleep(self.capture_interval)
                    
        except KeyboardInterrupt:
            print("\n🛑 Auto capture stopped by user")
        
        print(f"\n✅ Captured {captured_count} images for training")
        print("💡 Next steps:")
        print("   1. Run: python enhanced_training_pipeline.py")
        print("   2. Annotate images if needed")
        print("   3. Train enhanced model")
    
    def _check_for_new_capture(self):
        """Check if a new camera capture is available."""
        # Look for the latest camera capture (both spherical and Kinect)
        camera_files = [
            "http_camera_capture.jpg",
            "current_rgb.txt",
            "kinect_rgb.txt"
        ]
        
        for file in camera_files:
            if os.path.exists(file):
                # Generate timestamp for unique filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                if file.endswith('.jpg'):
                    # Copy image file
                    new_filename = f"captured_{timestamp}_{len(os.listdir(self.captured_dir)):04d}.jpg"
                    new_path = os.path.join(self.captured_dir, new_filename)
                    shutil.copy2(file, new_path)
                    return True
                
                elif file.endswith('.txt'):
                    # Convert raw data to image
                    try:
                        with open(file, 'rb') as f:
                            raw_data = f.read()
                        
                        # Convert raw data to image (assuming RGB format)
                        if len(raw_data) == 393216:  # 512x256x3
                            # Reshape to image
                            img_array = np.frombuffer(raw_data, dtype=np.uint8)
                            img_array = img_array.reshape(256, 512, 3)
                            
                            # Save as image
                            new_filename = f"captured_{timestamp}_{len(os.listdir(self.captured_dir)):04d}.jpg"
                            new_path = os.path.join(self.captured_dir, new_filename)
                            cv2.imwrite(new_path, img_array)
                            return True
                    except Exception as e:
                        print(f"❌ Error converting raw data: {e}")
        
        return False
    
    def create_capture_script(self):
        """Create an improved Lua script for easier capture."""
        lua_script = '''-- Enhanced CoppeliaSim Capture Script
-- Copy and paste this into CoppeliaSim console

print("📸 Enhanced Camera Capture Script")
print("=" * 40)

-- Get camera handles
local rgbSensor = sim.getObject("./sensorRGB")
local depthSensor = sim.getObject("./sensorDepth")

if rgbSensor ~= -1 and depthSensor ~= -1 then
    print("✅ Found cameras!")
    
    -- Capture RGB image
    local rgbImage = sim.getVisionSensorImg(rgbSensor)
    if rgbImage then
        print("✅ RGB captured: " .. #rgbImage .. " pixels")
        
        -- Save to file
        local file = io.open("current_rgb.txt", "wb")
        if file then
            file:write(rgbImage)
            file:close()
            print("💾 RGB data saved to current_rgb.txt")
        end
        
        -- Try to send via HTTP
        local http = require("socket.http")
        local ltn12 = require("ltn12")
        
        local response_body = {}
        local res, code, response_headers = http.request{
            url = "http://localhost:8080/camera",
            method = "POST",
            headers = {
                ["Content-Type"] = "application/octet-stream",
                ["Content-Length"] = #rgbImage
            },
            source = ltn12.source.string(rgbImage),
            sink = ltn12.sink.table(response_body)
        }
        
        if res and code == 200 then
            print("✅ Camera data sent to HTTP server!")
        else
            print("❌ Failed to send camera data: " .. (code or "unknown error"))
        end
    else
        print("❌ Failed to capture RGB image")
    end
    
    print("🎯 Capture completed!")
else
    print("❌ Camera sensors not found!")
end
'''
        
        with open('enhanced_capture_script.lua', 'w') as f:
            f.write(lua_script)
        
        print("✅ Enhanced capture script created: enhanced_capture_script.lua")
    
    def analyze_captured_images(self):
        """Analyze captured images for quality and diversity."""
        print("🔍 Analyzing captured images...")
        
        if not os.path.exists(self.captured_dir):
            print("❌ No captured images directory found")
            return
        
        image_files = [f for f in os.listdir(self.captured_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        if not image_files:
            print("❌ No captured images found")
            return
        
        print(f"📸 Found {len(image_files)} captured images")
        
        # Analyze image quality
        total_pixels = 0
        edge_densities = []
        
        for img_file in image_files:
            img_path = os.path.join(self.captured_dir, img_file)
            image = cv2.imread(img_path)
            
            if image is not None:
                # Calculate edge density
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / (image.shape[0] * image.shape[1])
                edge_densities.append(edge_density)
                
                total_pixels += image.shape[0] * image.shape[1]
        
        if edge_densities:
            avg_edge_density = np.mean(edge_densities)
            print(f"📊 Analysis Results:")
            print(f"   Average edge density: {avg_edge_density:.4f}")
            print(f"   Total pixels: {total_pixels:,}")
            print(f"   Average image size: {total_pixels // len(image_files):,} pixels")
            
            # Quality assessment
            if avg_edge_density > 0.1:
                print("✅ Good image quality - objects should be detectable")
            elif avg_edge_density > 0.05:
                print("⚠️  Moderate image quality - may need more training")
            else:
                print("❌ Low image quality - consider repositioning camera")
        
        print("✅ Image analysis complete!")

def main():
    """Main function."""
    capture_tool = AutoCaptureTrainingData()
    
    print("🎯 Auto Capture Training Data Tool")
    print("=" * 40)
    
    # Create enhanced capture script
    capture_tool.create_capture_script()
    
    # Analyze existing images if any
    capture_tool.analyze_captured_images()
    
    # Start auto capture
    response = input("\n🤔 Would you like to start auto capture? (y/n): ")
    if response.lower() == 'y':
        capture_tool.start_auto_capture()
    else:
        print("💡 You can run auto capture later with: capture_tool.start_auto_capture()")

if __name__ == "__main__":
    main()
