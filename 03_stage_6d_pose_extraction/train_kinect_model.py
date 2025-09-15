#!/usr/bin/env python3
"""
Train Kinect-Specific YOLO Model
Train a new model on the actual objects in your scene
"""

import cv2
import numpy as np
import os
import json
import yaml
from pathlib import Path
from ultralytics import YOLO
import glob

class KinectModelTrainer:
    def __init__(self):
        """Initialize the Kinect model trainer."""
        self.objects = [
            "master_chef_can",
            "cracker_box", 
            "mug",
            "banana",
            "mustard_bottle"
        ]
        
        self.training_dir = "kinect_training_dataset"
        self.images_dir = f"{self.training_dir}/images"
        self.labels_dir = f"{self.training_dir}/labels"
        
        # Create directories
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
    
    def process_captured_images(self):
        """Process captured Kinect training images."""
        print("🔄 Processing captured Kinect images...")
        
        # Find all training data files
        rgb_files = glob.glob("training_kinect_*_rgb.txt")
        depth_files = glob.glob("training_kinect_*_depth.txt")
        
        print(f"📊 Found {len(rgb_files)} RGB files and {len(depth_files)} depth files")
        
        processed_count = 0
        
        for rgb_file in rgb_files:
            try:
                # Extract image number
                image_num = rgb_file.split('_')[2]
                
                # Read RGB data
                with open(rgb_file, 'rb') as f:
                    rgb_data = f.read()
                
                if len(rgb_data) == 921600:  # 640x480x3
                    # Convert to image
                    data_array = np.frombuffer(rgb_data, dtype=np.uint8)
                    data_array = data_array.reshape(480, 640, 3)
                    
                    # Convert BGR to RGB
                    rgb_image = cv2.cvtColor(data_array, cv2.COLOR_BGR2RGB)
                    
                    # Save image
                    image_filename = f"kinect_training_{image_num}.jpg"
                    image_path = os.path.join(self.images_dir, image_filename)
                    cv2.imwrite(image_path, rgb_image)
                    
                    print(f"✅ Processed: {image_filename}")
                    processed_count += 1
                    
                    # Create placeholder label file (you'll need to annotate these)
                    label_filename = f"kinect_training_{image_num}.txt"
                    label_path = os.path.join(self.labels_dir, label_filename)
                    
                    # Create empty label file for now
                    with open(label_path, 'w') as f:
                        pass  # Empty file - will be filled with annotations
                    
            except Exception as e:
                print(f"❌ Error processing {rgb_file}: {e}")
        
        print(f"📊 Processed {processed_count} training images")
        return processed_count
    
    def create_dataset_yaml(self):
        """Create dataset configuration file."""
        dataset_config = {
            'path': os.path.abspath(self.training_dir),
            'train': 'images',
            'val': 'images',
            'nc': len(self.objects),
            'names': self.objects
        }
        
        yaml_path = f"{self.training_dir}/dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"💾 Dataset config saved: {yaml_path}")
        return yaml_path
    
    def create_annotation_guide(self):
        """Create a guide for manual annotation."""
        guide_content = f"""
# Kinect Training Data Annotation Guide

## Objects to Annotate:
{chr(10).join([f"{i+1}. {obj}" for i, obj in enumerate(self.objects)])}

## Annotation Format (YOLO):
Each line in the label file should be: `class_id center_x center_y width height`

Where:
- class_id: 0-{len(self.objects)-1} (0=master_chef_can, 1=cracker_box, etc.)
- center_x, center_y: Normalized center coordinates (0-1)
- width, height: Normalized bounding box dimensions (0-1)

## Example:
```
0 0.5 0.5 0.2 0.3  # master_chef_can in center
1 0.8 0.3 0.15 0.25  # cracker_box in top-right
```

## Files to Annotate:
{chr(10).join([f"- {f}" for f in os.listdir(self.labels_dir) if f.endswith('.txt')])}

## Annotation Tool:
You can use tools like:
- LabelImg: https://github.com/tzutalin/labelImg
- CVAT: https://cvat.org/
- Roboflow: https://roboflow.com/
"""
        
        guide_path = f"{self.training_dir}/annotation_guide.md"
        with open(guide_path, 'w') as f:
            f.write(guide_content)
        
        print(f"📖 Annotation guide saved: {guide_path}")
    
    def train_model(self, dataset_yaml):
        """Train the YOLO model on Kinect data."""
        print("🚀 Starting YOLO model training...")
        
        # Initialize model
        model = YOLO('yolov8n.pt')  # Start with nano model
        
        # Training parameters
        training_params = {
            'data': dataset_yaml,
            'epochs': 100,
            'imgsz': 640,
            'batch': 16,
            'device': 'cpu',  # Change to 'cuda' if you have GPU
            'project': 'kinect_trained_model',
            'name': 'kinect_objects'
        }
        
        print("🎯 Training parameters:")
        for key, value in training_params.items():
            print(f"   {key}: {value}")
        
        # Start training
        try:
            results = model.train(**training_params)
            print("✅ Training completed successfully!")
            return model
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return None
    
    def test_trained_model(self, model_path):
        """Test the trained model on Kinect images."""
        print("🧪 Testing trained model...")
        
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return
        
        model = YOLO(model_path)
        
        # Test on a few images
        test_images = glob.glob(f"{self.images_dir}/*.jpg")[:3]
        
        for img_path in test_images:
            print(f"\n📸 Testing: {os.path.basename(img_path)}")
            
            # Run detection
            results = model(img_path, conf=0.1)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    print(f"✅ Found {len(boxes)} detections:")
                    
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        if cls < len(self.objects):
                            class_name = self.objects[cls]
                            print(f"   {i+1}. {class_name} (conf: {conf:.3f})")
                        else:
                            print(f"   {i+1}. Unknown class {cls} (conf: {conf:.3f})")
                else:
                    print("❌ No detections found")
    
    def run_training_pipeline(self):
        """Run the complete training pipeline."""
        print("🚀 Kinect Model Training Pipeline")
        print("=" * 50)
        
        # Step 1: Process captured images
        processed_count = self.process_captured_images()
        
        if processed_count == 0:
            print("❌ No training images found!")
            print("💡 Please run the capture script in CoppeliaSim first")
            return
        
        # Step 2: Create dataset configuration
        dataset_yaml = self.create_dataset_yaml()
        
        # Step 3: Create annotation guide
        self.create_annotation_guide()
        
        print(f"\n📊 Training Dataset Ready:")
        print(f"   - Images: {processed_count}")
        print(f"   - Objects: {len(self.objects)}")
        print(f"   - Classes: {', '.join(self.objects)}")
        
        print(f"\n📝 Next Steps:")
        print(f"   1. Annotate the images in {self.labels_dir}/")
        print(f"   2. Follow the guide in {self.training_dir}/annotation_guide.md")
        print(f"   3. Run training with: python train_kinect_model.py --train")
        
        # Check if user wants to train now
        if len(glob.glob(f"{self.labels_dir}/*.txt")) > 0:
            print(f"\n🎯 Found label files! Starting training...")
            model = self.train_model(dataset_yaml)
            
            if model:
                # Test the model
                model_path = "kinect_trained_model/kinect_objects/weights/best.pt"
                self.test_trained_model(model_path)
        else:
            print(f"\n⏳ Waiting for annotations...")
            print(f"   Please annotate the images and run again")

def main():
    """Main function."""
    trainer = KinectModelTrainer()
    trainer.run_training_pipeline()

if __name__ == "__main__":
    main()
