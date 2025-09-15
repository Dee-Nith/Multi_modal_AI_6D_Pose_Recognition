#!/usr/bin/env python3
"""
🔄 Retrain YOLO with New Captured Data
=====================================
Convert new captured images to training format and retrain YOLO model.
"""

import os
import shutil
import cv2
import numpy as np
import json
import glob
from pathlib import Path
import yaml
from ultralytics import YOLO
import matplotlib.pyplot as plt

class YOLORetrainingPipeline:
    """Pipeline to retrain YOLO with new captured data."""
    
    def __init__(self):
        """Initialize the retraining pipeline."""
        print("🔄 Initializing YOLO Retraining Pipeline...")
        
        # Paths
        self.tmp_dir = "/tmp"
        self.dataset_dir = "coppelia_sim_dataset"
        self.new_images_dir = "new_captured_images"
        self.new_labels_dir = "new_captured_labels"
        
        # Create directories
        os.makedirs(self.new_images_dir, exist_ok=True)
        os.makedirs(self.new_labels_dir, exist_ok=True)
        
        # Class mapping
        self.classes = {
            'master_chef_can': 0,
            'banana': 1,
            'cracker_box': 2,
            'mug': 3,
            'mustard_bottle': 4
        }
        
        print("✅ YOLO Retraining Pipeline initialized!")
    
    def find_captured_images(self):
        """Find all captured images in /tmp directory."""
        print("🔍 Finding captured images...")
        
        rgb_files = glob.glob(os.path.join(self.tmp_dir, "auto_kinect_*_rgb.txt"))
        depth_files = glob.glob(os.path.join(self.tmp_dir, "auto_kinect_*_depth.txt"))
        
        print(f"📸 Found {len(rgb_files)} RGB files and {len(depth_files)} depth files")
        
        # Extract image IDs
        image_ids = []
        for rgb_file in rgb_files:
            filename = os.path.basename(rgb_file)
            if filename.startswith("auto_kinect_") and filename.endswith("_rgb.txt"):
                image_id = filename.replace("auto_kinect_", "").replace("_rgb.txt", "")
                try:
                    image_id = int(image_id)
                    image_ids.append(image_id)
                except ValueError:
                    continue
        
        image_ids.sort()
        print(f"🎯 Found {len(image_ids)} valid image IDs: {image_ids[:10]}...")
        
        return image_ids
    
    def convert_txt_to_jpg(self, image_id):
        """Convert .txt RGB file to .jpg image."""
        rgb_file = os.path.join(self.tmp_dir, f"auto_kinect_{image_id}_rgb.txt")
        
        if not os.path.exists(rgb_file):
            print(f"❌ RGB file not found: {rgb_file}")
            return None
        
        try:
            # Load raw RGB data
            with open(rgb_file, 'rb') as f:
                rgb_data = np.frombuffer(f.read(), dtype=np.uint8)
            
            # Reshape to image
            rgb_image = rgb_data.reshape(480, 640, 3)
            
            # Save as JPG
            output_path = os.path.join(self.new_images_dir, f"auto_kinect_{image_id}_rgb.jpg")
            cv2.imwrite(output_path, rgb_image)
            
            print(f"✅ Converted image {image_id}")
            return output_path
            
        except Exception as e:
            print(f"❌ Error converting image {image_id}: {e}")
            return None
    
    def create_annotations_for_image(self, image_id):
        """Create YOLO annotations for a specific image."""
        # Manual annotation based on visual analysis
        annotations = []
        
        if image_id == 22:
            # master_chef_can (blue can)
            annotations.append({
                'class': 'master_chef_can',
                'bbox': [164, 234, 258, 329],  # [x1, y1, x2, y2]
                'confidence': 0.95
            })
            # cracker_box (reddish-orange box)
            annotations.append({
                'class': 'cracker_box',
                'bbox': [239, 132, 371, 301],
                'confidence': 0.90
            })
            # mustard_bottle (yellow bottle)
            annotations.append({
                'class': 'mustard_bottle',
                'bbox': [420, 176, 479, 279],
                'confidence': 0.85
            })
        
        elif image_id == 23:
            # master_chef_can
            annotations.append({
                'class': 'master_chef_can',
                'bbox': [180, 200, 280, 320],
                'confidence': 0.90
            })
            # banana
            annotations.append({
                'class': 'banana',
                'bbox': [300, 250, 350, 300],
                'confidence': 0.85
            })
            # mustard_bottle
            annotations.append({
                'class': 'mustard_bottle',
                'bbox': [400, 180, 450, 280],
                'confidence': 0.80
            })
        
        elif image_id == 24:
            # master_chef_can
            annotations.append({
                'class': 'master_chef_can',
                'bbox': [200, 220, 280, 300],
                'confidence': 0.90
            })
            # banana
            annotations.append({
                'class': 'banana',
                'bbox': [320, 240, 370, 290],
                'confidence': 0.85
            })
            # second banana
            annotations.append({
                'class': 'banana',
                'bbox': [350, 230, 400, 280],
                'confidence': 0.80
            })
        
        elif image_id == 32:
            # master_chef_can
            annotations.append({
                'class': 'master_chef_can',
                'bbox': [220, 200, 300, 280],
                'confidence': 0.90
            })
            # banana
            annotations.append({
                'class': 'banana',
                'bbox': [320, 220, 370, 270],
                'confidence': 0.85
            })
            # second master_chef_can
            annotations.append({
                'class': 'master_chef_can',
                'bbox': [180, 180, 260, 260],
                'confidence': 0.80
            })
            # third master_chef_can
            annotations.append({
                'class': 'master_chef_can',
                'bbox': [400, 200, 480, 280],
                'confidence': 0.75
            })
            # mustard_bottle
            annotations.append({
                'class': 'mustard_bottle',
                'bbox': [350, 180, 400, 250],
                'confidence': 0.85
            })
        
        # Add more annotations for other images as needed
        # You can expand this based on your captured images
        
        return annotations
    
    def convert_bbox_to_yolo_format(self, bbox, image_width=640, image_height=480):
        """Convert [x1, y1, x2, y2] to YOLO format [x_center, y_center, width, height] (normalized)."""
        x1, y1, x2, y2 = bbox
        
        # Calculate center and dimensions
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        
        # Normalize
        x_center_norm = x_center / image_width
        y_center_norm = y_center / image_height
        width_norm = width / image_width
        height_norm = height / image_height
        
        return [x_center_norm, y_center_norm, width_norm, height_norm]
    
    def create_yolo_label_file(self, image_id, annotations):
        """Create YOLO format label file."""
        label_file = os.path.join(self.new_labels_dir, f"auto_kinect_{image_id}_rgb.txt")
        
        with open(label_file, 'w') as f:
            for annotation in annotations:
                class_name = annotation['class']
                if class_name in self.classes:
                    class_id = self.classes[class_name]
                    bbox = annotation['bbox']
                    yolo_bbox = self.convert_bbox_to_yolo_format(bbox)
                    
                    # Write in YOLO format: class_id x_center y_center width height
                    line = f"{class_id} {' '.join([str(x) for x in yolo_bbox])}\n"
                    f.write(line)
        
        print(f"✅ Created label file for image {image_id}")
        return label_file
    
    def process_new_images(self, image_ids):
        """Process new images and create annotations."""
        print("🔄 Processing new images...")
        
        processed_images = []
        
        for image_id in image_ids:
            print(f"\n📸 Processing image {image_id}...")
            
            # Convert image
            image_path = self.convert_txt_to_jpg(image_id)
            if image_path is None:
                continue
            
            # Create annotations
            annotations = self.create_annotations_for_image(image_id)
            if not annotations:
                print(f"⚠️ No annotations for image {image_id}")
                continue
            
            # Create label file
            label_path = self.create_yolo_label_file(image_id, annotations)
            
            processed_images.append({
                'image_id': image_id,
                'image_path': image_path,
                'label_path': label_path,
                'annotations': annotations
            })
        
        print(f"✅ Processed {len(processed_images)} images with annotations")
        return processed_images
    
    def merge_with_existing_dataset(self, processed_images):
        """Merge new images with existing dataset."""
        print("🔄 Merging with existing dataset...")
        
        # Copy new images to dataset
        train_images_dir = os.path.join(self.dataset_dir, "train", "images")
        train_labels_dir = os.path.join(self.dataset_dir, "train", "labels")
        
        os.makedirs(train_images_dir, exist_ok=True)
        os.makedirs(train_labels_dir, exist_ok=True)
        
        for item in processed_images:
            # Copy image
            image_filename = os.path.basename(item['image_path'])
            dst_image_path = os.path.join(train_images_dir, image_filename)
            shutil.copy2(item['image_path'], dst_image_path)
            
            # Copy label
            label_filename = os.path.basename(item['label_path'])
            dst_label_path = os.path.join(train_labels_dir, label_filename)
            shutil.copy2(item['label_path'], dst_label_path)
        
        print(f"✅ Merged {len(processed_images)} new images into training dataset")
    
    def update_dataset_yaml(self):
        """Update dataset.yaml with new class information."""
        print("🔄 Updating dataset configuration...")
        
        yaml_content = {
            'path': './coppelia_sim_dataset',
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'nc': len(self.classes),
            'names': list(self.classes.keys())
        }
        
        yaml_path = os.path.join(self.dataset_dir, "data.yaml")
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        print(f"✅ Updated dataset configuration: {yaml_path}")
        print(f"📊 Classes: {list(self.classes.keys())}")
    
    def retrain_yolo_model(self):
        """Retrain YOLO model with enhanced dataset."""
        print("🔄 Starting YOLO model retraining...")
        
        # Load existing model or start fresh
        model_path = os.path.join(self.dataset_dir, "runs", "detect", "train", "weights", "best.pt")
        
        if os.path.exists(model_path):
            print(f"📦 Loading existing model: {model_path}")
            model = YOLO(model_path)
        else:
            print("🆕 Starting with fresh YOLOv8n model")
            model = YOLO('yolov8n.pt')
        
        # Training configuration
        config = {
            'data': os.path.join(self.dataset_dir, "data.yaml"),
            'epochs': 100,
            'imgsz': 640,
            'batch': 16,
            'device': 'cpu',  # Change to '0' for GPU
            'project': self.dataset_dir,
            'name': 'retrained_model',
            'save': True,
            'save_period': 10,
            'patience': 20,
            'optimizer': 'Adam',
            'lr0': 0.001,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'pose': 12.0,
            'kobj': 1.0,
            'label_smoothing': 0.0,
            'nbs': 64,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'plots': True
        }
        
        print("🚀 Starting training...")
        print(f"📊 Training configuration:")
        for key, value in config.items():
            print(f"   {key}: {value}")
        
        # Start training
        results = model.train(**config)
        
        print("✅ Training completed!")
        return results
    
    def test_retrained_model(self, test_image_ids=[22, 23, 24, 32]):
        """Test the retrained model on new images."""
        print("🧪 Testing retrained model...")
        
        # Load the best model
        best_model_path = os.path.join(self.dataset_dir, "runs", "detect", "retrained_model", "weights", "best.pt")
        
        if not os.path.exists(best_model_path):
            print(f"❌ Best model not found: {best_model_path}")
            return
        
        model = YOLO(best_model_path)
        
        for image_id in test_image_ids:
            print(f"\n🧪 Testing image {image_id}...")
            
            # Load image
            image_path = os.path.join(self.new_images_dir, f"auto_kinect_{image_id}_rgb.jpg")
            if not os.path.exists(image_path):
                print(f"❌ Image not found: {image_path}")
                continue
            
            # Run detection
            results = model(image_path, verbose=False)
            
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
    
    def run_complete_pipeline(self):
        """Run the complete retraining pipeline."""
        print("🚀 Starting Complete YOLO Retraining Pipeline")
        print("=" * 60)
        
        # 1. Find captured images
        image_ids = self.find_captured_images()
        
        # 2. Process new images
        processed_images = self.process_new_images(image_ids)
        
        if not processed_images:
            print("❌ No images processed. Exiting.")
            return
        
        # 3. Merge with existing dataset
        self.merge_with_existing_dataset(processed_images)
        
        # 4. Update dataset configuration
        self.update_dataset_yaml()
        
        # 5. Retrain model
        results = self.retrain_yolo_model()
        
        # 6. Test retrained model
        self.test_retrained_model()
        
        print("\n🎉 Complete YOLO Retraining Pipeline finished!")
        print("📁 New model saved in: coppelia_sim_dataset/runs/detect/retrained_model/weights/")

def main():
    """Main function."""
    print("🔄 YOLO Retraining with New Captured Data")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = YOLORetrainingPipeline()
    
    # Run complete pipeline
    pipeline.run_complete_pipeline()

if __name__ == "__main__":
    main()




