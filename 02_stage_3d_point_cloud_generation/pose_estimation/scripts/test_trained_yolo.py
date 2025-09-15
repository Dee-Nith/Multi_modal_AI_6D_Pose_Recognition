#!/usr/bin/env python3
"""
Test Trained YOLOv8 Model
Test the trained YOLOv8 model on YCB objects
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def test_trained_model():
    """Test the trained YOLOv8 model."""
    print("🧪 Testing Trained YOLOv8 Model")
    print("=" * 40)
    
    # Check if trained model exists
    model_path = Path("ycb_yolo_training/ycb_detector/weights/best.pt")
    if not model_path.exists():
        print("❌ Trained model not found!")
        print("💡 Training may still be in progress...")
        return False
    
    print(f"✅ Found trained model: {model_path}")
    
    # Load the trained model
    try:
        model = YOLO(str(model_path))
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Test on validation images
    val_images = list(Path("ycb_yolo_dataset/images/val").glob("*.jpg"))[:5]
    
    if not val_images:
        print("❌ No validation images found!")
        return False
    
    print(f"\n🔍 Testing on {len(val_images)} validation images...")
    
    total_detections = 0
    for i, img_path in enumerate(val_images):
        print(f"\n📸 Testing {img_path.name}...")
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print("  ❌ Could not load image")
            continue
        
        # Run inference
        try:
            results = model(image, verbose=False)
            
            # Process results
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes
                    detections = len(boxes)
                    total_detections += detections
                    
                    print(f"  📦 Detected {detections} objects:")
                    
                    for j, box in enumerate(boxes):
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        print(f"    {j+1}. {class_name} (confidence: {confidence:.2f}) at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
                        
                        # Draw bounding box on image
                        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(image, f"{class_name}: {confidence:.2f}", 
                                  (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    print("  ⚠️ No objects detected")
            
            # Save annotated image
            output_path = f"test_result_{i}.jpg"
            cv2.imwrite(output_path, image)
            print(f"  💾 Saved annotated image: {output_path}")
            
        except Exception as e:
            print(f"  ❌ Inference failed: {e}")
    
    print(f"\n📊 Test Summary:")
    print(f"  - Images tested: {len(val_images)}")
    print(f"  - Total detections: {total_detections}")
    print(f"  - Average detections per image: {total_detections/len(val_images):.1f}")
    
    return True

def check_training_completion():
    """Check if training has completed."""
    print("\n🔍 Checking Training Completion...")
    
    model_path = Path("ycb_yolo_training/ycb_detector/weights/best.pt")
    if model_path.exists():
        print("✅ Training completed! Best model available.")
        return True
    else:
        print("⏳ Training still in progress...")
        return False

def main():
    """Main function."""
    print("🤖 YOLOv8 Trained Model Testing")
    print("=" * 50)
    
    # Check if training is complete
    if check_training_completion():
        # Test the model
        success = test_trained_model()
        
        if success:
            print("\n🎉 Model testing completed successfully!")
            print("🚀 Ready to integrate with robotic grasping pipeline!")
        else:
            print("\n💥 Model testing failed!")
    else:
        print("\n⏳ Please wait for training to complete...")
        print("💡 Run this script again once training is done.")

if __name__ == "__main__":
    main()







