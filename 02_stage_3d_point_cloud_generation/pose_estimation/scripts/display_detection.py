#!/usr/bin/env python3
"""
Display YOLOv8s Detection Results on New CoppeliaSim Capture
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def display_detection_results():
    """Display detection results with bounding boxes and labels"""
    print("🖼️ Displaying Detection Results")
    print("=" * 50)
    
    # Path to the new capture
    image_path = "/Users/nith/Desktop/AI_6D_Pose_recognition/semantic_segmentation_project/new_captures/all_captures/processed_images/capture_1_rgb.jpg"
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    # Load the trained model
    model_path = "training_results/yolov8s_instance_segmentation/weights/best.pt"
    
    if not Path(model_path).exists():
        print(f"❌ Trained model not found: {model_path}")
        return
    
    try:
        # Load the model
        model = YOLO(model_path)
        print("✅ Model loaded successfully!")
        
        # Run inference
        print("🔍 Running inference...")
        results = model.predict(
            source=image_path,
            conf=0.25,  # Confidence threshold
            save=False  # Don't save, just get results
        )
        
        # Load the original image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Original image
        ax1.imshow(img_rgb)
        ax1.set_title("Original CoppeliaSim Capture", fontsize=16, fontweight='bold')
        ax1.axis('off')
        
        # Image with detections
        ax2.imshow(img_rgb)
        ax2.set_title("YOLOv8s Instance Segmentation Results", fontsize=16, fontweight='bold')
        ax2.axis('off')
        
        # Color map for different classes
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
        
        # Process detection results
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None:
                boxes = result.boxes
                masks = result.masks
                
                print(f"\n📊 Objects Detected: {len(boxes)}")
                
                for i, (box, mask) in enumerate(zip(boxes, masks)):
                    # Get class info
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names[cls]
                    
                    # Get bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Choose color
                    color = colors[i % len(colors)]
                    
                    # Draw bounding box
                    rect = patches.Rectangle(
                        (x1, y1), x2-x1, y2-y1,
                        linewidth=3, edgecolor=color, facecolor='none'
                    )
                    ax2.add_patch(rect)
                    
                    # Add label
                    label = f"{class_name} ({conf:.3f})"
                    ax2.text(x1, y1-10, label, fontsize=12, fontweight='bold',
                            color=color, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                    
                    # Print detection info
                    print(f"  {i+1}. {class_name}: {conf:.3f} at ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f})")
                    
                    # Draw mask if available
                    if mask is not None:
                        mask_data = mask.data[0].cpu().numpy()
                        mask_resized = cv2.resize(mask_data.astype(np.uint8), (img_rgb.shape[1], img_rgb.shape[0]))
                        
                        # Create colored mask overlay
                        mask_colored = np.zeros_like(img_rgb)
                        mask_colored[mask_resized == 1] = [255, 0, 0]  # Red mask
                        
                        # Overlay mask with transparency
                        ax2.imshow(mask_colored, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the comparison
        output_path = "detection_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n📁 Comparison saved: {output_path}")
        
        # Show the plot
        plt.show()
        
        print("\n🎉 Detection display completed!")
        
    except Exception as e:
        print(f"❌ Error during display: {e}")

def main():
    """Main function"""
    print("🎯 Displaying YOLOv8s Detection Results")
    print("=" * 60)
    
    # Display results
    display_detection_results()

if __name__ == "__main__":
    main()




