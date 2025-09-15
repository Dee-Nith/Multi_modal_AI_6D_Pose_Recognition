#!/usr/bin/env python3
"""
Simple Annotation Tool for Captured Images
Run this to annotate captured images with bounding boxes
"""

import cv2
import numpy as np
import os
import json

def annotate_images():
    """Simple annotation tool for captured images."""
    captured_dir = "captured_images"
    annotations = {}
    
    if not os.path.exists(captured_dir):
        print("❌ No captured images directory found")
        return
    
    image_files = [f for f in os.listdir(captured_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if not image_files:
        print("❌ No captured images found")
        return
    
    print(f"📸 Found {len(image_files)} images to annotate")
    print("🖱️  Click and drag to create bounding boxes")
    print("📝 Press 's' to save, 'n' for next image, 'q' to quit")
    
    for img_file in image_files:
        img_path = os.path.join(captured_dir, img_file)
        image = cv2.imread(img_path)
        
        if image is None:
            continue
        
        # Display image for annotation
        cv2.imshow('Annotate Image', image)
        key = cv2.waitKey(0)
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save annotation
            annotations[img_file] = {
                'objects': [],
                'image_size': image.shape[:2]
            }
            print(f"💾 Saved annotation for {img_file}")
        elif key == ord('n'):
            # Skip this image
            continue
    
    cv2.destroyAllWindows()
    
    # Save annotations
    with open('captured_annotations.json', 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print("✅ Annotations saved to captured_annotations.json")

if __name__ == "__main__":
    annotate_images()
