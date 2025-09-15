#!/usr/bin/env python3
"""
🧪 Test Enhanced YOLO Model
==========================
Test the enhanced YOLO model on new captured images.
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

def test_enhanced_model():
    """Test the enhanced YOLO model on new images."""
    print("🧪 Testing Enhanced YOLO Model")
    print("=" * 40)
    
    # Load the enhanced model
    model_path = "coppelia_sim_dataset/enhanced_model2/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Enhanced model not found: {model_path}")
        return
    
    print(f"📦 Loading enhanced model: {model_path}")
    model = YOLO(model_path)
    
    # Test images
    test_image_ids = [22, 23, 24, 32]
    
    for image_id in test_image_ids:
        print(f"\n🧪 Testing image {image_id}...")
        
        # Load image from /tmp
        rgb_file = f"/tmp/auto_kinect_{image_id}_rgb.txt"
        
        if not os.path.exists(rgb_file):
            print(f"❌ Image not found: {rgb_file}")
            continue
        
        try:
            # Load raw RGB data
            with open(rgb_file, 'rb') as f:
                rgb_data = np.frombuffer(f.read(), dtype=np.uint8)
            
            # Reshape to image
            rgb_image = rgb_data.reshape(480, 640, 3)
            
            # Run detection
            results = model(rgb_image, verbose=False)
            
            # Display results
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes
                    print(f"  🎯 Detected {len(boxes)} objects:")
                    
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        name = result.names[cls]
                        
                        print(f"    • {name}: {conf:.2f} at [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
                else:
                    print(f"  ❌ No objects detected")
            
            # Create visualization
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Original image
            axes[0].imshow(rgb_image)
            axes[0].set_title(f'Image {image_id}: Original')
            axes[0].axis('off')
            
            # Detection results
            result_image = results[0].plot()
            axes[1].imshow(result_image)
            axes[1].set_title(f'Image {image_id}: Enhanced YOLO Detection')
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'enhanced_detection_image_{image_id}.png', dpi=150, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"❌ Error processing image {image_id}: {e}")

if __name__ == "__main__":
    test_enhanced_model()




