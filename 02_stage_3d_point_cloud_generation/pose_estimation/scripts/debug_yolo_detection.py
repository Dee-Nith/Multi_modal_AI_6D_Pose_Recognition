#!/usr/bin/env python3
"""
🔍 Debug YOLO Detection Issues
=============================
Check why YOLO is detecting all objects as master_chef_can.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os

def debug_yolo_detection():
    """Debug YOLO detection on image 22."""
    print("🔍 Debugging YOLO Detection...")
    
    # Load YOLO model
    model_path = '../../coppelia_sim_results/weights/best.pt'
    print(f"📦 Loading YOLO model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    model = YOLO(model_path)
    
    # Check model classes
    print(f"🎯 Model classes: {model.names}")
    print(f"📊 Number of classes: {len(model.names)}")
    
    # Load image 22
    rgb_file = "/tmp/auto_kinect_22_rgb.txt"
    if not os.path.exists(rgb_file):
        print(f"❌ Image not found: {rgb_file}")
        return
    
    # Load RGB image
    with open(rgb_file, 'rb') as f:
        rgb_data = np.frombuffer(f.read(), dtype=np.uint8)
    rgb_image = rgb_data.reshape(480, 640, 3)
    
    print(f"📸 Image shape: {rgb_image.shape}")
    
    # Run detection with verbose output
    print("\n🔍 Running YOLO detection...")
    results = model(rgb_image, verbose=True)
    
    print(f"\n📊 Detection Results:")
    for i, result in enumerate(results):
        print(f"  Result {i}:")
        if result.boxes is not None:
            boxes = result.boxes
            print(f"    Number of detections: {len(boxes)}")
            
            for j, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                name = result.names[cls]
                
                print(f"      Detection {j}:")
                print(f"        Class ID: {cls}")
                print(f"        Class Name: {name}")
                print(f"        Confidence: {conf:.3f}")
                print(f"        BBox: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
        else:
            print("    No detections")

def check_model_info():
    """Check detailed model information."""
    print("\n🔍 Checking Model Information...")
    
    model_path = '../../coppelia_sim_results/weights/best.pt'
    model = YOLO(model_path)
    
    # Print model info
    print(f"📦 Model path: {model_path}")
    print(f"🎯 Classes: {model.names}")
    
    # Check if model was trained correctly
    print(f"\n📊 Class Distribution:")
    for i, name in model.names.items():
        print(f"  Class {i}: {name}")

if __name__ == "__main__":
    check_model_info()
    debug_yolo_detection()




