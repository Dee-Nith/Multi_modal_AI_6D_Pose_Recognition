#!/usr/bin/env python3
"""
Test YOLOv8s Instance Segmentation on New CoppeliaSim Capture
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt

def test_model_on_new_capture():
    """Test the trained model on new CoppeliaSim capture"""
    print("🧪 Testing YOLOv8s on New CoppeliaSim Capture")
    print("=" * 60)
    
    # Path to the new capture
    image_path = "/Users/nith/Desktop/AI_6D_Pose_recognition/semantic_segmentation_project/new_captures/all_captures/processed_images/capture_1_rgb.jpg"
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"📸 Testing image: {image_path}")
    
    # Load the trained model
    model_path = "training_results/yolov8s_instance_segmentation/weights/best.pt"
    
    if not Path(model_path).exists():
        print(f"❌ Trained model not found: {model_path}")
        return
    
    print(f"🤖 Loading model: {model_path}")
    
    try:
        # Load the model
        model = YOLO(model_path)
        print("✅ Model loaded successfully!")
        
        # Run inference
        print("\n🔍 Running inference...")
        results = model.predict(
            source=image_path,
            conf=0.25,  # Confidence threshold
            save=True,
            project="test_results",
            name="new_capture_detection"
        )
        
        print("✅ Inference completed!")
        
        # Display results
        print("\n📊 Detection Results:")
        for i, result in enumerate(results):
            print(f"\n--- Image {i+1} ---")
            
            if result.boxes is not None:
                boxes = result.boxes
                masks = result.masks
                
                print(f"  📦 Objects detected: {len(boxes)}")
                
                for j, (box, mask) in enumerate(zip(boxes, masks)):
                    # Get class info
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names[cls]
                    
                    # Get bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    print(f"    Object {j+1}:")
                    print(f"      Class: {class_name}")
                    print(f"      Confidence: {conf:.3f}")
                    print(f"      Bounding Box: ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f})")
                    
                    if mask is not None:
                        mask_area = np.sum(mask.data[0].cpu().numpy())
                        print(f"      Mask Area: {mask_area} pixels")
            
            else:
                print("  ❌ No objects detected")
        
        # Show the image with detections
        print("\n🖼️ Displaying results...")
        
        # Load and display the image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Plot results
        plt.figure(figsize=(12, 8))
        plt.imshow(img_rgb)
        plt.title("YOLOv8s Instance Segmentation - New CoppeliaSim Capture")
        plt.axis('off')
        
        # Save the plot
        plot_path = "test_results/new_capture_detection/new_capture_visualization.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📁 Visualization saved: {plot_path}")
        
        # Show results summary
        print(f"\n🎉 Test completed successfully!")
        print(f"📁 Results saved in: test_results/new_capture_detection/")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return None

def main():
    """Main function"""
    print("🎯 Testing Trained YOLOv8s on New CoppeliaSim Capture")
    print("=" * 70)
    
    # Test the model
    results = test_model_on_new_capture()
    
    if results:
        print("\n✅ Test completed successfully!")
        print("🔍 Check the test_results folder for detailed outputs")
    else:
        print("\n❌ Test failed")

if __name__ == "__main__":
    main()




