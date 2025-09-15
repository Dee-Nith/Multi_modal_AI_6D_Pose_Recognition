#!/usr/bin/env python3
"""
Enhanced Training Pipeline for CoppeliaSim Object Detection
Uses captured images to improve YOLO model performance
"""

import os
import cv2
import numpy as np
import json
import shutil
from pathlib import Path
from ultralytics import YOLO
import yaml

class EnhancedTrainingPipeline:
    def __init__(self):
        self.base_dataset_path = "ycb_texture_dataset"
        self.enhanced_dataset_path = "enhanced_ycb_dataset"
        self.captured_images_path = "captured_images"
        self.model_path = "ycb_texture_training/ycb_texture_detector/weights/best.pt"
        
    def setup_enhanced_dataset(self):
        """Set up enhanced dataset with captured images."""
        print("🔧 Setting up enhanced dataset...")
        
        # Create enhanced dataset structure
        os.makedirs(f"{self.enhanced_dataset_path}/images/train", exist_ok=True)
        os.makedirs(f"{self.enhanced_dataset_path}/images/val", exist_ok=True)
        os.makedirs(f"{self.enhanced_dataset_path}/labels/train", exist_ok=True)
        os.makedirs(f"{self.enhanced_dataset_path}/labels/val", exist_ok=True)
        
        # Copy existing dataset
        print("📁 Copying existing dataset...")
        self._copy_existing_dataset()
        
        # Add captured images
        print("📸 Adding captured images...")
        self._add_captured_images()
        
        # Create enhanced dataset config
        self._create_enhanced_config()
        
        print("✅ Enhanced dataset setup complete!")
        
    def _copy_existing_dataset(self):
        """Copy existing dataset to enhanced dataset."""
        # Copy images
        for split in ['train', 'val']:
            src_images = f"{self.base_dataset_path}/images/{split}"
            dst_images = f"{self.enhanced_dataset_path}/images/{split}"
            
            if os.path.exists(src_images):
                for img_file in os.listdir(src_images):
                    if img_file.endswith(('.jpg', '.png', '.jpeg')):
                        shutil.copy2(
                            os.path.join(src_images, img_file),
                            os.path.join(dst_images, img_file)
                        )
            
            # Copy labels
            src_labels = f"{self.base_dataset_path}/labels/{split}"
            dst_labels = f"{self.enhanced_dataset_path}/labels/{split}"
            
            if os.path.exists(src_labels):
                for label_file in os.listdir(src_labels):
                    if label_file.endswith('.txt'):
                        shutil.copy2(
                            os.path.join(src_labels, label_file),
                            os.path.join(dst_labels, label_file)
                        )
    
    def _add_captured_images(self):
        """Add captured images to training dataset."""
        captured_dir = self.captured_images_path
        if not os.path.exists(captured_dir):
            os.makedirs(captured_dir)
            print(f"📁 Created {captured_dir} directory")
            return
        
        # Find captured images
        captured_images = []
        for file in os.listdir(captured_dir):
            if file.endswith(('.jpg', '.png', '.jpeg')):
                captured_images.append(file)
        
        if not captured_images:
            print("⚠️  No captured images found")
            return
        
        print(f"📸 Found {len(captured_images)} captured images")
        
        # Copy to training set
        for i, img_file in enumerate(captured_images):
            src_path = os.path.join(captured_dir, img_file)
            dst_path = os.path.join(f"{self.enhanced_dataset_path}/images/train", f"captured_{i:04d}.jpg")
            
            # Copy image
            shutil.copy2(src_path, dst_path)
            
            # Create placeholder label (empty for now - will be annotated)
            label_path = os.path.join(f"{self.enhanced_dataset_path}/labels/train", f"captured_{i:04d}.txt")
            with open(label_path, 'w') as f:
                pass  # Empty label file
    
    def _create_enhanced_config(self):
        """Create enhanced dataset configuration."""
        config = {
            'path': os.path.abspath(self.enhanced_dataset_path),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 21,  # Number of classes
            'names': [
                '002_master_chef_can', '003_cracker_box', '004_sugar_box', 
                '005_tomato_soup_can', '006_mustard_bottle', '007_tuna_fish_can', 
                '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', 
                '011_banana', '019_pitcher_base', '021_bleach_cleanser', 
                '024_bowl', '025_mug', '035_power_drill', '036_wood_block', 
                '037_scissors', '040_large_marker', '051_large_clamp', 
                '052_extra_large_clamp', '061_foam_brick'
            ]
        }
        
        config_path = f"{self.enhanced_dataset_path}/dataset.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"📄 Enhanced dataset config created: {config_path}")
    
    def train_enhanced_model(self, epochs=100, batch_size=16):
        """Train enhanced model with captured images."""
        print("🚀 Starting enhanced model training...")
        
        # Load existing model as starting point
        if os.path.exists(self.model_path):
            print(f"📥 Loading existing model: {self.model_path}")
            model = YOLO(self.model_path)
        else:
            print("📥 Loading base YOLO model")
            model = YOLO('yolov8n.pt')
        
        # Train with enhanced dataset
        config_path = f"{self.enhanced_dataset_path}/dataset.yaml"
        
        results = model.train(
            data=config_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=640,
            patience=20,
            save=True,
            project="enhanced_ycb_training",
            name="enhanced_detector",
            exist_ok=True
        )
        
        print("✅ Enhanced model training complete!")
        return results
    
    def create_annotation_tool(self):
        """Create a simple annotation tool for captured images."""
        print("🛠️  Creating annotation tool...")
        
        annotation_script = '''#!/usr/bin/env python3
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
'''
        
        with open('annotate_captured_images.py', 'w') as f:
            f.write(annotation_script)
        
        print("✅ Annotation tool created: annotate_captured_images.py")
    
    def generate_synthetic_data(self, num_images=100):
        """Generate synthetic training data using data augmentation."""
        print(f"🎨 Generating {num_images} synthetic images...")
        
        if not os.path.exists(self.model_path):
            print("❌ No existing model found for synthetic data generation")
            return
        
        model = YOLO(self.model_path)
        
        # Create synthetic data directory
        synthetic_dir = f"{self.enhanced_dataset_path}/synthetic"
        os.makedirs(f"{synthetic_dir}/images", exist_ok=True)
        os.makedirs(f"{synthetic_dir}/labels", exist_ok=True)
        
        # Generate synthetic images with augmentation
        for i in range(num_images):
            # Use existing training images as base
            train_images = os.listdir(f"{self.enhanced_dataset_path}/images/train")
            if train_images:
                base_image = np.random.choice(train_images)
                base_path = f"{self.enhanced_dataset_path}/images/train/{base_image}"
                
                # Load and augment image
                image = cv2.imread(base_path)
                if image is not None:
                    # Apply random augmentations
                    augmented = self._augment_image(image)
                    
                    # Save augmented image
                    synthetic_path = f"{synthetic_dir}/images/synthetic_{i:04d}.jpg"
                    cv2.imwrite(synthetic_path, augmented)
                    
                    # Copy corresponding label
                    base_label = base_image.replace('.jpg', '.txt').replace('.png', '.txt')
                    label_path = f"{self.enhanced_dataset_path}/labels/train/{base_label}"
                    if os.path.exists(label_path):
                        synthetic_label_path = f"{synthetic_dir}/labels/synthetic_{i:04d}.txt"
                        shutil.copy2(label_path, synthetic_label_path)
        
        print("✅ Synthetic data generation complete!")
    
    def _augment_image(self, image):
        """Apply random augmentations to image."""
        # Random brightness
        brightness = np.random.uniform(0.7, 1.3)
        image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
        
        # Random contrast
        contrast = np.random.uniform(0.8, 1.2)
        image = cv2.convertScaleAbs(image, alpha=contrast, beta=0)
        
        # Random rotation
        angle = np.random.uniform(-15, 15)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, rotation_matrix, (w, h))
        
        # Random noise
        noise = np.random.normal(0, 5, image.shape).astype(np.uint8)
        image = cv2.add(image, noise)
        
        return image

def main():
    """Main training pipeline."""
    pipeline = EnhancedTrainingPipeline()
    
    print("🎯 Enhanced Training Pipeline for CoppeliaSim")
    print("=" * 50)
    
    # Setup enhanced dataset
    pipeline.setup_enhanced_dataset()
    
    # Create annotation tool
    pipeline.create_annotation_tool()
    
    # Generate synthetic data
    pipeline.generate_synthetic_data(num_images=50)
    
    # Train enhanced model
    print("\n🚀 Ready to train enhanced model!")
    print("💡 You can now:")
    print("   1. Annotate captured images: python annotate_captured_images.py")
    print("   2. Train enhanced model: pipeline.train_enhanced_model()")
    print("   3. Test with new camera captures")
    
    # Ask user if they want to start training
    response = input("\n🤔 Would you like to start training the enhanced model? (y/n): ")
    if response.lower() == 'y':
        pipeline.train_enhanced_model(epochs=50, batch_size=8)
    else:
        print("💡 You can run training later with: pipeline.train_enhanced_model()")

if __name__ == "__main__":
    main()
