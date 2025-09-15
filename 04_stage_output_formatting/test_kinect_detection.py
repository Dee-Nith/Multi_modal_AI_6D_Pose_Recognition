#!/usr/bin/env python3
"""
Test Object Detection on Kinect Data
Try detection with very low confidence thresholds
"""

import cv2
import numpy as np
import os
from ultralytics import YOLO

def test_kinect_detection():
    """Test object detection on Kinect data with low confidence."""
    print("🔍 Testing Kinect Object Detection")
    print("=" * 40)
    
    # Check for Kinect image
    kinect_rgb_path = "kinect_rgb_image.jpg"
    
    if not os.path.exists(kinect_rgb_path):
        print(f"❌ Kinect RGB image not found: {kinect_rgb_path}")
        return
    
    # Load the Kinect image
    image = cv2.imread(kinect_rgb_path)
    if image is None:
        print(f"❌ Failed to load Kinect image: {kinect_rgb_path}")
        return
    
    print(f"✅ Loaded Kinect image: {image.shape}")
    
    # Load YOLO model
    model_path = "ycb_texture_training/ycb_texture_detector/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"❌ YOLO model not found: {model_path}")
        return
    
    print(f"✅ Loading YOLO model: {model_path}")
    model = YOLO(model_path)
    
    # Test with different confidence thresholds
    confidence_thresholds = [0.1, 0.05, 0.01, 0.005, 0.001]
    
    for conf_threshold in confidence_thresholds:
        print(f"\n🎯 Testing with confidence threshold: {conf_threshold}")
        
        # Run detection
        results = model(image, conf=conf_threshold)
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    class_name = model.names[cls]
                    
                    detections.append({
                        'class': class_name,
                        'confidence': float(conf),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)]
                    })
        
        print(f"   Detections: {len(detections)}")
        
        if detections:
            for i, det in enumerate(detections):
                print(f"   {i+1}. {det['class']} (conf: {det['confidence']:.4f})")
                print(f"      Bbox: [{det['bbox'][0]:.1f}, {det['bbox'][1]:.1f}, {det['bbox'][2]:.1f}, {det['bbox'][3]:.1f}]")
            
            # Save detection results
            results_data = {
                'image_path': kinect_rgb_path,
                'image_size': image.shape,
                'confidence_threshold': conf_threshold,
                'detections': detections
            }
            
            output_file = f'kinect_detection_conf_{conf_threshold}.json'
            import json
            with open(output_file, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            print(f"   💾 Results saved as: {output_file}")
            break  # Stop if we found detections
        else:
            print("   ❌ No objects detected")
    
    # If no detections found, try image preprocessing
    if not detections:
        print("\n🔄 Trying image preprocessing...")
        
        # Try different preprocessing techniques
        preprocessing_methods = [
            ("Original", image),
            ("Resized 2x", cv2.resize(image, (image.shape[1]*2, image.shape[0]*2))),
            ("Resized 4x", cv2.resize(image, (image.shape[1]*4, image.shape[0]*4))),
            ("Brightened", cv2.convertScaleAbs(image, alpha=1.5, beta=30)),
            ("Contrast Enhanced", cv2.convertScaleAbs(image, alpha=1.3, beta=0)),
        ]
        
        for method_name, processed_image in preprocessing_methods:
            print(f"   Testing {method_name}...")
            
            results = model(processed_image, conf=0.01)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    print(f"   ✅ {method_name} worked! Found {len(boxes)} detections")
                    
                    # Save the processed image
                    processed_path = f"kinect_processed_{method_name.lower().replace(' ', '_')}.jpg"
                    cv2.imwrite(processed_path, processed_image)
                    print(f"   💾 Processed image saved as: {processed_path}")
                    break
            else:
                print(f"   ❌ {method_name} didn't help")
    
    print("\n✅ Kinect detection testing complete!")

if __name__ == "__main__":
    test_kinect_detection()




