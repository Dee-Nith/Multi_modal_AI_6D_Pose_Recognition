#!/usr/bin/env python3
"""
Fresh YOLOv8s Instance Segmentation Training Script
Clean start with Roboflow dataset and CoppeliaSim captures
"""

import os
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch

def setup_training_environment():
    """Setup training environment and check GPU availability"""
    print("🚀 Setting up YOLOv8s training environment...")
    
    # Check CUDA/MPS availability
    if torch.cuda.is_available():
        device = "cuda"
        print(f"✅ CUDA GPU detected: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("✅ Apple Silicon MPS detected")
    else:
        device = "cpu"
        print("⚠️ Using CPU (slower training)")
    
    print(f"🎯 Training device: {device}")
    return device

def prepare_dataset_paths():
    """Prepare dataset paths for training"""
    print("\n📁 Preparing dataset paths...")
    
    # Get current working directory
    current_dir = Path.cwd()
    print(f"📍 Current directory: {current_dir}")
    
    # Base dataset path - use the clean structure
    base_path = current_dir / "Robot_Grasping_Dataset"
    print(f"📍 Dataset base path: {base_path}")
    
    # Check if dataset exists
    if not base_path.exists():
        print(f"❌ Dataset not found at: {base_path}")
        return None
    
    # Create absolute paths for data.yaml
    train_path = base_path / "train" / "images"
    valid_path = base_path / "valid" / "images"
    test_path = base_path / "test" / "images"
    
    # Verify paths exist
    paths = [train_path, valid_path, test_path]
    for path in paths:
        if not path.exists():
            print(f"❌ Path not found: {path}")
            return None
    
    print(f"✅ Train images: {train_path}")
    print(f"✅ Valid images: {valid_path}")
    print(f"✅ Test images: {test_path}")
    
    return base_path

def create_training_config(base_path):
    """Create training configuration"""
    print("\n⚙️ Creating training configuration...")
    
    # Use absolute paths to avoid duplication
    config = {
        'train': str(base_path.absolute() / "train" / "images"),
        'val': str(base_path.absolute() / "valid" / "images"),
        'test': str(base_path.absolute() / "test" / "images"),
        'nc': 6,  # Number of classes
        'names': ['banana', 'cracker_box', 'master_chef_can', 'mug', 'mustard_bottle', 'objects']
    }
    
    # Save config in current directory
    config_path = Path.cwd() / "data.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ Training config saved: {config_path}")
    
    # Print config contents for verification
    print("\n📋 Config contents:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Verify the file was created
    if config_path.exists():
        print(f"✅ Config file verified: {config_path}")
    else:
        print(f"❌ Config file not found: {config_path}")
    
    return config_path

def train_yolov8s_instance_segmentation(config_path, device):
    """Train YOLOv8s instance segmentation model"""
    print("\n🎯 Starting YOLOv8s training...")
    
    # Initialize YOLOv8s model
    model = YOLO('yolov8s-seg.pt')  # Instance segmentation model
    
    # Training parameters
    training_args = {
        'data': str(config_path),
        'epochs': 100,
        'imgsz': 640,
        'batch': 8,
        'device': device,
        'patience': 20,
        'save': True,
        'save_period': 10,
        'cache': False,
        'workers': 4,
        'project': 'training_results',
        'name': 'yolov8s_instance_segmentation',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'auto',
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 2.0,
        'label_smoothing': 0.0,
        'nbs': 64,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
        'plots': True
    }
    
    print("📊 Training parameters:")
    for key, value in training_args.items():
        if key in ['epochs', 'imgsz', 'batch', 'device', 'lr0']:
            print(f"  {key}: {value}")
    
    # Start training
    try:
        print("\n🚀 Starting training...")
        print(f"📁 Using config file: {config_path}")
        results = model.train(**training_args)
        
        print("\n✅ Training completed successfully!")
        print(f"📁 Results saved in: {results.save_dir}")
        
        return model, results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None, None

def validate_model(model, config_path):
    """Validate the trained model"""
    print("\n🔍 Validating trained model...")
    
    try:
        # Run validation
        results = model.val(data=str(config_path))
        
        print("\n📊 Validation Results:")
        print(f"  mAP@0.5: {results.box.map50:.3f}")
        print(f"  mAP@0.5:0.95: {results.box.map:.3f}")
        print(f"  Precision: {results.box.mp:.3f}")
        print(f"  Recall: {results.box.mr:.3f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return None

def main():
    """Main training function"""
    print("🎯 Fresh YOLOv8s Instance Segmentation Training")
    print("=" * 60)
    
    # Setup environment
    device = setup_training_environment()
    
    # Prepare dataset
    base_path = prepare_dataset_paths()
    if base_path is None:
        print("❌ Dataset preparation failed")
        return
    
    # Create config
    config_path = create_training_config(base_path)
    
    # Train model
    model, results = train_yolov8s_instance_segmentation(config_path, device)
    
    if model is not None:
        # Validate model
        validation_results = validate_model(model, config_path)
        
        print("\n🎉 Training pipeline completed!")
        print(f"📁 Model saved in: training_results/yolov8s_instance_segmentation/")
        print(f"📊 Training results: {results.save_dir if results else 'N/A'}")
        
        # Save model path for later use
        model_path = Path("training_results/yolov8s_instance_segmentation/weights/best.pt")
        if model_path.exists():
            print(f"🏆 Best model: {model_path}")
        else:
            print("⚠️ Best model not found")
    else:
        print("❌ Training failed")

if __name__ == "__main__":
    main()




