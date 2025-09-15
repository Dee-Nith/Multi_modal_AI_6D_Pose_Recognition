
#!/usr/bin/env python3
# Claude-generated Python script

import cv2
import numpy as np
from ultralytics import YOLO

def claude_detection():
    print("🤖 Claude-generated detection running...")
    
    # Load model
    model = YOLO("ycb_texture_training/ycb_texture_detector/weights/best.pt")
    
    # Process image
    image = cv2.imread("claude_rgb_image.jpg")
    if image is not None:
        results = model(image, conf=0.1)
        print(f"✅ Detection complete: {len(results)} results")
    else:
        print("❌ Image not found")

if __name__ == "__main__":
    claude_detection()
