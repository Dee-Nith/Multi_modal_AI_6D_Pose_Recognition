#!/usr/bin/env python3
"""
Test object detection with different confidence thresholds
"""

import cv2
import numpy as np
import json
import os
from ultralytics import YOLO

def test_detection_sensitivity():
    """Test detection with different confidence thresholds."""
    
    image_path = "http_camera_capture.jpg"
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return
    
    print(f"✅ Loaded image: {image.shape}")
    
    # Load YOLO model
    try:
        model_path = "ycb_texture_training/ycb_texture_detector/weights/best.pt"
        if os.path.exists(model_path):
            model = YOLO(model_path)
            print(f"✅ Trained YOLO model loaded: {model_path}")
        else:
            model = YOLO('yolov8n.pt')
            print("⚠️ Using pre-trained YOLO model")
    except Exception as e:
        print(f"❌ Error loading YOLO model: {e}")
        return
    
    # Test different confidence thresholds
    confidence_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    print(f"\n🔍 Testing detection sensitivity:")
    print(f"=" * 50)
    
    for conf_thresh in confidence_thresholds:
        # Run detection with custom confidence threshold
        results = model(image, conf=conf_thresh, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Get class name
                    if hasattr(result, 'names') and cls in result.names:
                        class_name = result.names[cls]
                    else:
                        class_name = f"Class_{cls}"
                    
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(conf),
                        'class': class_name,
                        'class_id': int(cls)
                    })
        
        print(f"   Confidence threshold {conf_thresh:.1f}: {len(detections)} detections")
        
        if detections:
            for det in detections:
                print(f"     - {det['class']} (conf: {det['confidence']:.3f})")
            
            # Save the first detection we find
            if conf_thresh == 0.1:  # Save results from lowest threshold
                detection_data = {
                    "confidence_threshold": conf_thresh,
                    "image_path": image_path,
                    "image_shape": image.shape,
                    "detections": detections,
                    "total_detections": len(detections)
                }
                
                with open("sensitive_detection_results.json", "w") as f:
                    json.dump(detection_data, f, indent=2)
                
                # Draw bounding boxes on image
                annotated_image = image.copy()
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(annotated_image, f"{det['class']} {det['confidence']:.2f}", 
                               (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                cv2.imwrite("sensitive_detection.jpg", annotated_image)
                print(f"   💾 Results saved as: sensitive_detection_results.json")
                print(f"   💾 Annotated image saved as: sensitive_detection.jpg")
                break
    
    if not any(detections for detections in [model(image, conf=conf, verbose=False) for conf in confidence_thresholds]):
        print(f"   ❌ No objects detected even with very low confidence threshold")
        print(f"   💡 This suggests the objects might not be visible or recognizable in this camera view")

if __name__ == "__main__":
    test_detection_sensitivity()




