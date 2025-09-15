#!/usr/bin/env python3
"""
🔄 Enhance Existing YOLO Dataset with New Captured Images
========================================================
Add new captured images to existing CoppeliaSim dataset and retrain.
"""

import os
import shutil
import cv2
import numpy as np
import glob
import yaml
from ultralytics import YOLO
import matplotlib.pyplot as plt

class EnhancedYOLODataset:
    """Enhance existing YOLO dataset with new captured images."""
    
    def __init__(self):
        """Initialize the enhancement pipeline."""
        print("🔄 Initializing Enhanced YOLO Dataset...")
        
        # Paths
        self.tmp_dir = "/tmp"
        self.dataset_dir = "coppelia_sim_dataset"
        self.new_images_dir = "new_captured_images"
        
        # Create directories
        os.makedirs(self.new_images_dir, exist_ok=True)
        
        # Class mapping (from existing dataset)
        self.classes = {
            'master_chef_can': 0,
            'banana': 1,
            'cracker_box': 2,
            'mug': 3,
            'mustard_bottle': 4
        }
        
        print("✅ Enhanced YOLO Dataset initialized!")
    
    def find_captured_images(self):
        """Find all captured images in /tmp directory."""
        print("🔍 Finding captured images...")
        
        rgb_files = glob.glob(os.path.join(self.tmp_dir, "auto_kinect_*_rgb.txt"))
        
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
    
    def find_existing_annotation(self, image_id):
        """Find existing Roboflow annotation for the image."""
        # Look for annotation in the labels directory
        labels_dir = os.path.join(self.dataset_dir, "labels")
        
        # Try different possible annotation file names
        possible_names = [
            f"auto_kinect_{image_id}_rgb_jpg.rf.*.txt",
            f"auto_kinect_{image_id}_rgb.txt",
            f"auto_kinect_{image_id}.txt"
        ]
        
        for pattern in possible_names:
            matches = glob.glob(os.path.join(labels_dir, pattern))
            if matches:
                return matches[0]
        
        return None
    
    def process_new_images(self, image_ids):
        """Process new images and find their annotations."""
        print("🔄 Processing new images...")
        
        processed_images = []
        
        for image_id in image_ids:
            print(f"\n📸 Processing image {image_id}...")
            
            # Convert image
            image_path = self.convert_txt_to_jpg(image_id)
            if image_path is None:
                continue
            
            # Find existing annotation
            annotation_path = self.find_existing_annotation(image_id)
            if annotation_path is None:
                print(f"⚠️ No annotation found for image {image_id}")
                continue
            
            print(f"✅ Found annotation: {os.path.basename(annotation_path)}")
            
            processed_images.append({
                'image_id': image_id,
                'image_path': image_path,
                'annotation_path': annotation_path
            })
        
        print(f"✅ Processed {len(processed_images)} images with annotations")
        return processed_images
    
    def add_to_existing_dataset(self, processed_images):
        """Add new images to existing dataset."""
        print("🔄 Adding to existing dataset...")
        
        # Paths for existing dataset
        train_images_dir = os.path.join(self.dataset_dir, "train", "images")
        train_labels_dir = os.path.join(self.dataset_dir, "train", "labels")
        valid_images_dir = os.path.join(self.dataset_dir, "valid", "images")
        valid_labels_dir = os.path.join(self.dataset_dir, "valid", "labels")
        
        # Ensure directories exist
        os.makedirs(train_images_dir, exist_ok=True)
        os.makedirs(train_labels_dir, exist_ok=True)
        os.makedirs(valid_images_dir, exist_ok=True)
        os.makedirs(valid_labels_dir, exist_ok=True)
        
        # Split into train and validation (80/20)
        train_count = int(len(processed_images) * 0.8)
        train_images = processed_images[:train_count]
        val_images = processed_images[train_count:]
        
        print(f"📊 Train: {len(train_images)} images, Validation: {len(val_images)} images")
        
        # Add to training set
        for item in train_images:
            # Copy image
            image_filename = f"auto_kinect_{item['image_id']}_rgb.jpg"
            dst_image_path = os.path.join(train_images_dir, image_filename)
            shutil.copy2(item['image_path'], dst_image_path)
            
            # Copy annotation
            annotation_filename = f"auto_kinect_{item['image_id']}_rgb.txt"
            dst_annotation_path = os.path.join(train_labels_dir, annotation_filename)
            shutil.copy2(item['annotation_path'], dst_annotation_path)
            
            print(f"✅ Added to train: {image_filename}")
        
        # Add to validation set
        for item in val_images:
            # Copy image
            image_filename = f"auto_kinect_{item['image_id']}_rgb.jpg"
            dst_image_path = os.path.join(valid_images_dir, image_filename)
            shutil.copy2(item['image_path'], dst_image_path)
            
            # Copy annotation
            annotation_filename = f"auto_kinect_{item['image_id']}_rgb.txt"
            dst_annotation_path = os.path.join(valid_labels_dir, annotation_filename)
            shutil.copy2(item['annotation_path'], dst_annotation_path)
            
            print(f"✅ Added to validation: {image_filename}")
        
        print(f"✅ Added {len(processed_images)} new images to existing dataset")
    
    def update_dataset_yaml(self):
        """Update dataset.yaml with enhanced information."""
        print("🔄 Updating dataset configuration...")
        
        yaml_content = {
            'path': './coppelia_sim_dataset',
            'train': 'train/images',
            'val': 'valid/images',
            'nc': len(self.classes),
            'names': list(self.classes.keys())
        }
        
        yaml_path = os.path.join(self.dataset_dir, "data.yaml")
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        print(f"✅ Updated dataset configuration: {yaml_path}")
        print(f"📊 Classes: {list(self.classes.keys())}")
    
    def retrain_enhanced_model(self):
        """Retrain YOLO model with enhanced dataset."""
        print("🔄 Starting enhanced YOLO model training...")
        
        # Load existing best model
        best_model_path = os.path.join(self.dataset_dir, "runs", "detect", "train", "weights", "best.pt")
        
        if os.path.exists(best_model_path):
            print(f"📦 Loading existing best model: {best_model_path}")
            model = YOLO(best_model_path)
        else:
            print("🆕 Starting with fresh YOLOv8n model")
            model = YOLO('yolov8n.pt')
        
        # Training configuration
        config = {
            'data': os.path.join(self.dataset_dir, "data.yaml"),
            'epochs': 100,
            'imgsz': 640,
            'batch': 32,  # Increased batch size for GPU
            'device': 'mps',  # Use Apple Silicon MPS for faster training
            'project': self.dataset_dir,
            'name': 'enhanced_model',
            'save': True,
            'save_period': 10,
            'patience': 20,
            'optimizer': 'Adam',
            'lr0': 0.0001,  # Lower learning rate for fine-tuning
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
        
        print("🚀 Starting enhanced training...")
        print(f"📊 Training configuration:")
        for key, value in config.items():
            print(f"   {key}: {value}")
        
        # Start training
        results = model.train(**config)
        
        print("✅ Enhanced training completed!")
        return results
    
    def test_enhanced_model(self, test_image_ids=[22, 23, 24, 32]):
        """Test the enhanced trained model on new images."""
        print("🧪 Testing enhanced trained model...")
        
        # Load the best model
        best_model_path = os.path.join(self.dataset_dir, "runs", "detect", "enhanced_model", "weights", "best.pt")
        
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
    
    def run_enhancement_pipeline(self):
        """Run the complete enhancement pipeline."""
        print("🚀 Starting Enhanced YOLO Dataset Pipeline")
        print("=" * 60)
        
        # 1. Find captured images
        image_ids = self.find_captured_images()
        
        # 2. Process new images
        processed_images = self.process_new_images(image_ids)
        
        if not processed_images:
            print("❌ No images processed. Exiting.")
            return
        
        # 3. Add to existing dataset
        self.add_to_existing_dataset(processed_images)
        
        # 4. Update dataset configuration
        self.update_dataset_yaml()
        
        # 5. Retrain enhanced model
        results = self.retrain_enhanced_model()
        
        # 6. Test enhanced model
        self.test_enhanced_model()
        
        print("\n🎉 Enhanced YOLO Dataset Pipeline finished!")
        print("📁 Enhanced model saved in: coppelia_sim_dataset/runs/detect/enhanced_model/weights/")

def main():
    """Main function."""
    print("🔄 Enhance Existing YOLO Dataset with New Captured Images")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = EnhancedYOLODataset()
    
    # Run enhancement pipeline
    pipeline.run_enhancement_pipeline()

if __name__ == "__main__":
    main()
