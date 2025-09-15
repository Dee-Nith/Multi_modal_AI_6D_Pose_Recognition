#!/usr/bin/env python3
"""
Claude Workflow Runner for AI 6D Pose Recognition
Automatically runs the complete pipeline using Claude-generated code
"""

import os
import subprocess
import time
import json
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

class ClaudeWorkflowRunner:
    def __init__(self):
        """Initialize the workflow runner."""
        self.project_root = Path.cwd()
        self.results = {}
        print("🤖 Claude Workflow Runner Initialized")
    
    def step_1_analyze_environment(self):
        """Step 1: Analyze the current environment and setup."""
        print("\n🔍 Step 1: Analyzing Environment")
        print("=" * 40)
        
        # Check for required files
        required_files = [
            "ycb_texture_training/ycb_texture_detector/weights/best.pt",
            "improved_http_server.py",
            "process_kinect_data.py"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
            else:
                print(f"✅ Found: {file_path}")
        
        if missing_files:
            print(f"❌ Missing files: {missing_files}")
            return False
        
        # Check Python environment
        try:
            import ultralytics
            import cv2
            import numpy
            print("✅ All required Python packages available")
        except ImportError as e:
            print(f"❌ Missing package: {e}")
            return False
        
        return True
    
    def step_2_generate_optimized_lua(self):
        """Step 2: Generate optimized Lua script for CoppeliaSim."""
        print("\n📝 Step 2: Generating Optimized Lua Script")
        print("=" * 40)
        
        lua_script = '''-- Claude-Optimized CoppeliaSim Capture Script
-- Enhanced stability and error handling

print("🤖 Claude-optimized capture script starting...")

-- Function to safely get camera handles
function getCameraHandles()
    local rgbSensor = sim.getObject("./rgb")
    local depthSensor = sim.getObject("./depth")
    
    if rgbSensor == -1 or depthSensor == -1 then
        print("❌ Camera sensors not found!")
        return nil, nil
    end
    
    print("✅ Camera handles obtained successfully")
    return rgbSensor, depthSensor
end

-- Function to capture and save RGB data
function captureRGB(rgbSensor)
    if rgbSensor == nil then return false end
    
    local rgbImage = sim.getVisionSensorImg(rgbSensor)
    if not rgbImage then
        print("❌ Failed to capture RGB image")
        return false
    end
    
    print("✅ RGB image captured: " .. #rgbImage .. " pixels")
    
    -- Save RGB data
    local file = io.open("claude_workflow_rgb.txt", "wb")
    if file then
        file:write(rgbImage)
        file:close()
        print("💾 RGB data saved to claude_workflow_rgb.txt")
        return true
    else
        print("❌ Failed to save RGB data")
        return false
    end
end

-- Function to capture and save depth data
function captureDepth(depthSensor)
    if depthSensor == nil then return false end
    
    local depthImage = sim.getVisionSensorDepthBuffer(depthSensor)
    if not depthImage then
        print("❌ Failed to capture depth image")
        return false
    end
    
    print("✅ Depth image captured: " .. #depthImage .. " pixels")
    
    -- Save depth data
    local file = io.open("claude_workflow_depth.txt", "wb")
    if file then
        file:write(depthImage)
        file:close()
        print("💾 Depth data saved to claude_workflow_depth.txt")
        return true
    else
        print("❌ Failed to save depth data")
        return false
    end
end

-- Main execution
print("🎯 Starting Claude-optimized capture workflow...")

local rgbSensor, depthSensor = getCameraHandles()
if rgbSensor and depthSensor then
    local rgbSuccess = captureRGB(rgbSensor)
    local depthSuccess = captureDepth(depthSensor)
    
    if rgbSuccess and depthSuccess then
        print("🎉 Claude workflow capture completed successfully!")
        print("📁 Files created:")
        print("   - claude_workflow_rgb.txt")
        print("   - claude_workflow_depth.txt")
    else
        print("⚠️  Partial capture completed")
    end
else
    print("❌ Camera setup failed")
end

print("🤖 Claude workflow script finished!")
'''
        
        with open("claude_workflow_capture.lua", "w") as f:
            f.write(lua_script)
        
        print("✅ Generated: claude_workflow_capture.lua")
        print("📋 Copy this script into CoppeliaSim console")
        return True
    
    def step_3_process_camera_data(self):
        """Step 3: Process camera data and run object detection."""
        print("\n🖼️  Step 3: Processing Camera Data")
        print("=" * 40)
        
        # Check for camera data files
        rgb_file = "claude_workflow_rgb.txt"
        depth_file = "claude_workflow_depth.txt"
        
        if not os.path.exists(rgb_file):
            print(f"❌ Camera data not found: {rgb_file}")
            print("💡 Please run the Lua script in CoppeliaSim first")
            return False
        
        print(f"✅ Found camera data: {rgb_file}")
        
        # Process RGB data
        try:
            with open(rgb_file, 'rb') as f:
                rgb_data = f.read()
            
            print(f"📊 RGB data size: {len(rgb_data)} bytes")
            
            # Try different resolutions for Kinect (64x48x3 = 9216 bytes)
            possible_resolutions = [
                (64, 48, 3),   # 9216 bytes
                (96, 32, 3),   # 9216 bytes
                (48, 64, 3),   # 9216 bytes
            ]
            
            image_created = False
            for width, height, channels in possible_resolutions:
                expected_size = width * height * channels
                if len(rgb_data) == expected_size:
                    print(f"🎯 Detected resolution: {width}x{height}")
                    
                    # Convert to numpy array
                    rgb_array = np.frombuffer(rgb_data, dtype=np.uint8)
                    rgb_array = rgb_array.reshape(height, width, channels)
                    
                    # Convert BGR to RGB
                    rgb_array = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2RGB)
                    
                    # Save as image
                    cv2.imwrite("claude_workflow_rgb_image.jpg", rgb_array)
                    print("💾 RGB image saved: claude_workflow_rgb_image.jpg")
                    image_created = True
                    break
            
            if not image_created:
                print("❌ Could not determine image resolution")
                return False
                
        except Exception as e:
            print(f"❌ Error processing RGB data: {e}")
            return False
        
        return True
    
    def step_4_run_object_detection(self):
        """Step 4: Run YOLO object detection on processed images."""
        print("\n🎯 Step 4: Running Object Detection")
        print("=" * 40)
        
        # Load YOLO model
        model_path = "ycb_texture_training/ycb_texture_detector/weights/best.pt"
        if not os.path.exists(model_path):
            print(f"❌ YOLO model not found: {model_path}")
            return False
        
        print(f"✅ Loading YOLO model: {model_path}")
        model = YOLO(model_path)
        
        # Load processed image
        image_path = "claude_workflow_rgb_image.jpg"
        if not os.path.exists(image_path):
            print(f"❌ Processed image not found: {image_path}")
            return False
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Failed to load image: {image_path}")
            return False
        
        print(f"✅ Loaded image: {image.shape}")
        
        # Run detection with multiple confidence thresholds
        confidence_thresholds = [0.1, 0.05, 0.01, 0.005]
        detections_found = []
        
        for conf_threshold in confidence_thresholds:
            print(f"\n🎯 Testing confidence threshold: {conf_threshold}")
            
            results = model(image, conf=conf_threshold)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    print(f"✅ Found {len(boxes)} detections!")
                    
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        class_name = model.names[cls]
                        
                        detection = {
                            'class': class_name,
                            'confidence': float(conf),
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'threshold': conf_threshold
                        }
                        detections_found.append(detection)
                        
                        print(f"   {i+1}. {class_name} (conf: {conf:.4f})")
                        print(f"      Bbox: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
                    
                    # Save detection results
                    results_data = {
                        'image_path': image_path,
                        'image_size': image.shape,
                        'confidence_threshold': conf_threshold,
                        'detections': detections_found,
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    with open('claude_workflow_detections.json', 'w') as f:
                        json.dump(results_data, f, indent=2)
                    
                    print("💾 Detection results saved: claude_workflow_detections.json")
                    break
            else:
                print("   ❌ No detections found")
        
        if detections_found:
            self.results['detections'] = detections_found
            return True
        else:
            print("❌ No objects detected with any confidence threshold")
            return False
    
    def step_5_generate_report(self):
        """Step 5: Generate comprehensive workflow report."""
        print("\n📋 Step 5: Generating Workflow Report")
        print("=" * 40)
        
        report = f"""
# Claude AI 6D Pose Recognition Workflow Report

## Workflow Summary
- **Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}
- **Status**: {'✅ SUCCESS' if self.results.get('detections') else '❌ FAILED'}
- **Project**: AI 6D Pose Recognition with CoppeliaSim

## Steps Completed

### Step 1: Environment Analysis
- ✅ Required files verified
- ✅ Python packages checked
- ✅ YOLO model available

### Step 2: Lua Script Generation
- ✅ Generated optimized capture script
- ✅ Enhanced error handling
- ✅ Stable camera operations

### Step 3: Camera Data Processing
- ✅ RGB data captured and processed
- ✅ Image resolution determined
- ✅ Image file created

### Step 4: Object Detection
- ✅ YOLO model loaded
- ✅ Multiple confidence thresholds tested
- ✅ Detection results saved

## Results

### Detections Found
"""
        
        if self.results.get('detections'):
            for i, det in enumerate(self.results['detections']):
                report += f"""
**Detection {i+1}**:
- **Object**: {det['class']}
- **Confidence**: {det['confidence']:.4f}
- **Bounding Box**: [{det['bbox'][0]:.1f}, {det['bbox'][1]:.1f}, {det['bbox'][2]:.1f}, {det['bbox'][3]:.1f}]
- **Threshold Used**: {det['threshold']}
"""
        else:
            report += "\n❌ No objects detected\n"
        
        report += f"""
## Files Generated
- `claude_workflow_capture.lua` - Optimized Lua script for CoppeliaSim
- `claude_workflow_rgb_image.jpg` - Processed camera image
- `claude_workflow_detections.json` - Detection results
- `claude_workflow_report.md` - This report

## Next Steps
1. **Review detection results** in `claude_workflow_detections.json`
2. **Adjust camera position** if no detections found
3. **Try different objects** in the scene
4. **Use spherical camera** for higher resolution
5. **Train model** with more diverse data

## Technical Notes
- **Camera Type**: Kinect RGBD
- **Image Resolution**: 64x48 pixels
- **YOLO Model**: YCB texture detector
- **Detection Method**: Multi-threshold confidence testing

---
*Generated by Claude AI Workflow Runner*
"""
        
        with open("claude_workflow_report.md", "w") as f:
            f.write(report)
        
        print("✅ Generated: claude_workflow_report.md")
        return True
    
    def run_complete_workflow(self):
        """Run the complete Claude-powered workflow."""
        print("🚀 Starting Claude AI 6D Pose Recognition Workflow")
        print("=" * 60)
        
        steps = [
            ("Environment Analysis", self.step_1_analyze_environment),
            ("Generate Lua Script", self.step_2_generate_optimized_lua),
            ("Process Camera Data", self.step_3_process_camera_data),
            ("Object Detection", self.step_4_run_object_detection),
            ("Generate Report", self.step_5_generate_report)
        ]
        
        for step_name, step_func in steps:
            print(f"\n🔄 Running: {step_name}")
            success = step_func()
            
            if not success:
                print(f"❌ Workflow failed at: {step_name}")
                return False
            
            print(f"✅ Completed: {step_name}")
        
        print("\n🎉 Claude Workflow Completed Successfully!")
        print("📁 Generated Files:")
        print("   - claude_workflow_capture.lua")
        print("   - claude_workflow_rgb_image.jpg")
        print("   - claude_workflow_detections.json")
        print("   - claude_workflow_report.md")
        
        return True

def main():
    """Main function to run the Claude workflow."""
    runner = ClaudeWorkflowRunner()
    success = runner.run_complete_workflow()
    
    if success:
        print("\n🎯 Next Steps:")
        print("1. Copy claude_workflow_capture.lua into CoppeliaSim")
        print("2. Run the script in CoppeliaSim")
        print("3. Check claude_workflow_report.md for results")
        print("4. Adjust camera position if needed")
    else:
        print("\n❌ Workflow failed. Check the error messages above.")

if __name__ == "__main__":
    main()




