#!/usr/bin/env python3
"""
Improved YOLOv8s Detection Display - Maintains Original Brightness
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def improved_detection_display():
    """Display detection results with improved brightness and visibility"""
    print("🖼️ Improved Detection Display - Maintaining Brightness")
    print("=" * 60)
    
    # Path to the new capture
    image_path = "/Users/nith/Desktop/AI_6D_Pose_recognition/semantic_segmentation_project/new_captures/all_captures/processed_images/capture_23_rgb.jpg"
    
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
            conf=0.25,
            save=False
        )
        
        # Load the original image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create figure with better sizing
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
        
        # Original image - maintain brightness
        ax1.imshow(img_rgb, vmin=0, vmax=255)
        ax1.set_title("Original CoppeliaSim Capture", fontsize=18, fontweight='bold', pad=20)
        ax1.axis('off')
        
        # Image with detections - maintain original brightness
        ax2.imshow(img_rgb, vmin=0, vmax=255)  # Explicit brightness range
        ax2.set_title("YOLOv8s Instance Segmentation Results", fontsize=18, fontweight='bold', pad=20)
        ax2.axis('off')
        
        # Enhanced color map for better visibility
        colors = ['#FF0000', '#00FF00', '#0000FF', '#FFA500', '#800080', '#00FFFF', '#FF00FF', '#FFFF00']
        
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
                    
                    # Draw enhanced bounding box
                    rect = patches.Rectangle(
                        (x1, y1), x2-x1, y2-y1,
                        linewidth=4, edgecolor=color, facecolor='none'
                    )
                    ax2.add_patch(rect)
                    
                    # Enhanced label with better visibility
                    label = f"{class_name} ({conf:.3f})"
                    ax2.text(x1, y1-15, label, fontsize=14, fontweight='bold',
                            color=color, bbox=dict(
                                boxstyle="round,pad=0.5", 
                                facecolor='white', 
                                alpha=0.9,
                                edgecolor=color,
                                linewidth=2
                            ))
                    
                    # Print detection info
                    print(f"  {i+1}. {class_name}: {conf:.3f}")
                    
                    # Draw improved mask overlay
                    if mask is not None:
                        mask_data = mask.data[0].cpu().numpy()
                        mask_resized = cv2.resize(mask_data.astype(np.uint8), (img_rgb.shape[1], img_rgb.shape[0]))
                        
                        # Create colored mask with better visibility
                        mask_colored = np.zeros_like(img_rgb)
                        mask_colored[mask_resized == 1] = [255, 100, 100]  # Lighter red for better visibility
                        
                        # Overlay mask with reduced transparency
                        ax2.imshow(mask_colored, alpha=0.2)  # Reduced from 0.3 to 0.2
        
        # Adjust layout and spacing
        plt.tight_layout(pad=3.0)
        
        # Save the improved comparison
        output_path = "improved_detection_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\n📁 Improved visualization saved: {output_path}")
        
        # Show the plot
        plt.show()
        
        print("\n🎉 Improved detection display completed!")
        print("✨ Better brightness and visibility achieved!")
        
    except Exception as e:
        print(f"❌ Error during improved display: {e}")

def main():
    """Main function"""
    print("🎯 Improved YOLOv8s Detection Display")
    print("=" * 60)
    
    # Display improved results
    improved_detection_display()

if __name__ == "__main__":
    main()




