#!/usr/bin/env python3
"""
🧪 Comprehensive Enhanced YOLO Model Test
=========================================
Test the enhanced YOLO model with different confidence thresholds.
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

def comprehensive_model_test():
    """Comprehensive test of the enhanced YOLO model."""
    print("🧪 Comprehensive Enhanced YOLO Model Test")
    print("=" * 50)
    
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
            
            # Test with different confidence thresholds
            confidence_thresholds = [0.1, 0.3, 0.5]
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Original image
            axes[0, 0].imshow(rgb_image)
            axes[0, 0].set_title(f'Image {image_id}: Original')
            axes[0, 0].axis('off')
            
            for i, conf_thresh in enumerate(confidence_thresholds):
                print(f"  🔍 Testing with confidence threshold: {conf_thresh}")
                
                # Run detection with custom confidence threshold
                results = model(rgb_image, conf=conf_thresh, verbose=False)
                
                # Display results
                for result in results:
                    if result.boxes is not None:
                        boxes = result.boxes
                        print(f"    🎯 Detected {len(boxes)} objects:")
                        
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            cls = int(box.cls[0].cpu().numpy())
                            name = result.names[cls]
                            
                            print(f"      • {name}: {conf:.2f} at [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
                    else:
                        print(f"    ❌ No objects detected")
                
                # Create visualization
                result_image = results[0].plot()
                
                if i == 0:
                    axes[0, 1].imshow(result_image)
                    axes[0, 1].set_title(f'Confidence: {conf_thresh}')
                    axes[0, 1].axis('off')
                elif i == 1:
                    axes[1, 0].imshow(result_image)
                    axes[1, 0].set_title(f'Confidence: {conf_thresh}')
                    axes[1, 0].axis('off')
                elif i == 2:
                    axes[1, 1].imshow(result_image)
                    axes[1, 1].set_title(f'Confidence: {conf_thresh}')
                    axes[1, 1].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'comprehensive_detection_image_{image_id}.png', dpi=150, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"❌ Error processing image {image_id}: {e}")
    
    # Also test on some training images to verify model works
    print(f"\n🧪 Testing on training images...")
    
    # Test on a few training images
    train_images_dir = "coppelia_sim_dataset/train/images"
    if os.path.exists(train_images_dir):
        train_images = [f for f in os.listdir(train_images_dir) if f.endswith('.jpg')][:3]
        
        for train_image in train_images:
            print(f"\n🧪 Testing training image: {train_image}")
            
            image_path = os.path.join(train_images_dir, train_image)
            results = model(image_path, conf=0.3, verbose=False)
            
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

if __name__ == "__main__":
    comprehensive_model_test()




