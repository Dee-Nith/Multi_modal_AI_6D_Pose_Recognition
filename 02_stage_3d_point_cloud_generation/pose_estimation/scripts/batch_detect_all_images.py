#!/usr/bin/env python3
"""
Batch YOLOv8s Detection on All Processed CoppeliaSim Images
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import time

def batch_detect_images():
    """Detect objects in all JPG images in the processed folder"""
    print("🔍 Batch Detection on All Processed Images")
    print("=" * 60)
    
    # Path to processed images folder
    images_folder = "/Users/nith/Desktop/AI_6D_Pose_recognition/semantic_segmentation_project/new_captures/all_captures/processed_images"
    
    # Check if folder exists
    if not Path(images_folder).exists():
        print(f"❌ Folder not found: {images_folder}")
        return
    
    # Get all JPG files
    jpg_files = list(Path(images_folder).glob("*.jpg"))
    
    if not jpg_files:
        print("❌ No JPG files found in the folder")
        return
    
    print(f"📁 Found {len(jpg_files)} JPG files")
    
    # Load the trained model
    model_path = "training_results/yolov8s_instance_segmentation/weights/best.pt"
    
    if not Path(model_path).exists():
        print(f"❌ Trained model not found: {model_path}")
        return
    
    try:
        # Load the model
        model = YOLO(model_path)
        print("✅ Model loaded successfully!")
        
        # Create output directory
        output_dir = Path("batch_detection_results")
        output_dir.mkdir(exist_ok=True)
        
        # Process each image
        total_objects = 0
        start_time = time.time()
        
        for i, image_path in enumerate(jpg_files):
            print(f"\n🔄 Processing image {i+1}/{len(jpg_files)}: {image_path.name}")
            
            # Run inference
            results = model.predict(
                source=str(image_path),
                conf=0.25,  # Confidence threshold
                save=False  # Don't save, just get results
            )
            
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None:
                    boxes = result.boxes
                    masks = result.masks
                    
                    num_objects = len(boxes)
                    total_objects += num_objects
                    
                    print(f"  📦 Objects detected: {num_objects}")
                    
                    # Load the original image
                    img = cv2.imread(str(image_path))
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Create figure for this image
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
                    
                    # Original image
                    ax1.imshow(img_rgb)
                    ax1.set_title(f"Original: {image_path.name}", fontsize=14, fontweight='bold')
                    ax1.axis('off')
                    
                    # Image with detections
                    ax2.imshow(img_rgb)
                    ax2.set_title(f"Detections: {num_objects} objects", fontsize=14, fontweight='bold')
                    ax2.axis('off')
                    
                    # Color map for different classes
                    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
                    
                    # Process detection results
                    for j, (box, mask) in enumerate(zip(boxes, masks)):
                        # Get class info
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = model.names[cls]
                        
                        # Get bounding box
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Choose color
                        color = colors[j % len(colors)]
                        
                        # Draw bounding box
                        rect = patches.Rectangle(
                            (x1, y1), x2-x1, y2-y1,
                            linewidth=3, edgecolor=color, facecolor='none'
                        )
                        ax2.add_patch(rect)
                        
                        # Add label
                        label = f"{class_name} ({conf:.3f})"
                        ax2.text(x1, y1-10, label, fontsize=10, fontweight='bold',
                                color=color, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                        
                        # Print detection info
                        print(f"    {j+1}. {class_name}: {conf:.3f}")
                        
                        # Draw mask if available
                        if mask is not None:
                            mask_data = mask.data[0].cpu().numpy()
                            mask_resized = cv2.resize(mask_data.astype(np.uint8), (img_rgb.shape[1], img_rgb.shape[0]))
                            
                            # Create colored mask overlay
                            mask_colored = np.zeros_like(img_rgb)
                            mask_colored[mask_resized == 1] = [255, 0, 0]  # Red mask
                            
                            # Overlay mask with transparency
                            ax2.imshow(mask_colored, alpha=0.3)
                else:
                    print("  ❌ No objects detected")
                    num_objects = 0
                    
                    # Load the original image
                    img = cv2.imread(str(image_path))
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Create figure for this image
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
                    
                    # Original image
                    ax1.imshow(img_rgb)
                    ax1.set_title(f"Original: {image_path.name}", fontsize=14, fontweight='bold')
                    ax1.axis('off')
                    
                    # Image with no detections
                    ax2.imshow(img_rgb)
                    ax2.set_title("No Objects Detected", fontsize=14, fontweight='bold', color='red')
                    ax2.axis('off')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save the comparison
            output_path = output_dir / f"detection_{image_path.stem}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()  # Close to free memory
            
            print(f"  💾 Saved: {output_path}")
        
        # Calculate statistics
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\n🎉 Batch Detection Completed!")
        print(f"📊 Statistics:")
        print(f"  📁 Images processed: {len(jpg_files)}")
        print(f"  📦 Total objects detected: {total_objects}")
        print(f"  ⏱️ Total processing time: {processing_time:.2f} seconds")
        print(f"  🚀 Average time per image: {processing_time/len(jpg_files):.2f} seconds")
        print(f"  📁 Results saved in: {output_dir}")
        
        # Create summary
        summary_path = output_dir / "detection_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("YOLOv8s Batch Detection Summary\n")
            f.write("=" * 40 + "\n")
            f.write(f"Images processed: {len(jpg_files)}\n")
            f.write(f"Total objects detected: {total_objects}\n")
            f.write(f"Processing time: {processing_time:.2f} seconds\n")
            f.write(f"Average time per image: {processing_time/len(jpg_files):.2f} seconds\n\n")
            
            f.write("Image Details:\n")
            for i, image_path in enumerate(jpg_files):
                f.write(f"{i+1}. {image_path.name}\n")
        
        print(f"📋 Summary saved: {summary_path}")
        
    except Exception as e:
        print(f"❌ Error during batch detection: {e}")

def main():
    """Main function"""
    print("🎯 Batch YOLOv8s Detection on All Processed Images")
    print("=" * 70)
    
    # Run batch detection
    batch_detect_images()

if __name__ == "__main__":
    main()




