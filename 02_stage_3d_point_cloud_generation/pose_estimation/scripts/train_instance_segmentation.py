#!/usr/bin/env python3
"""
GPU-Accelerated YOLOv8 Instance Segmentation Training Script
Trains for 50 epochs with GPU acceleration
"""

import os
import sys
import torch
from pathlib import Path
from ultralytics import YOLO
import yaml

def check_gpu():
    """Check GPU availability and set device"""
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        print(f"🚀 GPU Available: {device}")
        print(f"🔢 CUDA Version: {torch.version.cuda}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return 'cuda:0'
    else:
        print("⚠️  No GPU detected, using CPU (training will be slower)")
        return 'cpu'

def setup_training_environment():
    """Setup training environment and paths"""
    print("🎯 Setting up training environment...")
    
    # Base paths
    base_dir = Path("/Users/nith/Desktop/AI_6D_Pose_recognition/semantic_segmentation_project")
    dataset_path = base_dir / "Robot_Grasping_Dataset"
    output_dir = Path.cwd()  # Current directory (object_detection_model)
    
    # Verify dataset exists
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    # Verify data.yaml exists
    data_yaml = dataset_path / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")
    
    print(f"📁 Dataset path: {dataset_path}")
    print(f"📁 Output directory: {output_dir}")
    
    return base_dir, dataset_path, output_dir

def load_dataset_info(data_yaml_path):
    """Load and display dataset information"""
    print("\n📊 Loading dataset information...")
    
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"   📸 Training images: {data_config.get('train', 'N/A')}")
    print(f"   📸 Validation images: {data_config.get('val', 'N/A')}")
    print(f"   📸 Test images: {data_config.get('test', 'N/A')}")
    print(f"   🏷️  Number of classes: {data_config.get('nc', 'N/A')}")
    print(f"   📝 Class names: {data_config.get('names', 'N/A')}")
    
    return data_config

def train_model(dataset_path, output_dir, device, epochs=50):
    """Train YOLOv8 instance segmentation model"""
    print(f"\n🚀 Starting YOLOv8s Instance Segmentation Training...")
    print(f"   🎯 Device: {device}")
    print(f"   🔄 Epochs: {epochs}")
    print(f"   📁 Dataset: {dataset_path}")
    
    # Initialize model
    model = YOLO('yolov8s-seg.pt')  # Start with pre-trained YOLOv8s-seg
    
    # Training configuration
    training_args = {
        'data': str(dataset_path / 'data.yaml'),
        'epochs': epochs,
        'imgsz': 640,
        'batch': 16,  # Adjust based on GPU memory
        'device': device,
        'workers': 8,
        'patience': 20,  # Early stopping patience
        'save': True,
        'save_period': 10,  # Save every 10 epochs
        'cache': True,  # Cache images for faster training
        'optimizer': 'AdamW',  # Use AdamW optimizer
        'lr0': 0.01,  # Initial learning rate
        'lrf': 0.01,  # Final learning rate factor
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,  # Box loss gain
        'cls': 0.5,  # Class loss gain
        'dfl': 1.5,  # DFL loss gain
        'pose': 12.0,  # Pose loss gain (for keypoints if any)
        'kobj': 2.0,  # Keypoint obj loss gain
        'label_smoothing': 0.0,
        'nbs': 64,  # Nominal batch size
        'overlap_mask': True,  # Masks should overlap during training
        'mask_ratio': 4,  # Mask downsample ratio
        'dropout': 0.0,  # Use dropout for regularization
        'val': True,  # Validate during training
        'plots': True,  # Generate training plots
        'project': str(output_dir / 'results'),
        'name': 'yolov8s_instance_segmentation',
        'exist_ok': True,
        'pretrained': True,
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'single_cls': False,
        'rect': False,
        'cos_lr': False,
        'close_mosaic': 10,
        'resume': False,
        'amp': True,  # Use automatic mixed precision
        'fraction': 1.0,
        'profile': False,
        'freeze': None,
        'lr_scheduler': 'cosine',
        'hsv_h': 0.015,  # HSV-Hue augmentation
        'hsv_s': 0.7,    # HSV-Saturation augmentation
        'hsv_v': 0.4,    # HSV-Value augmentation
        'degrees': 0.0,   # Image rotation
        'translate': 0.1, # Image translation
        'scale': 0.5,     # Image scaling
        'shear': 0.0,     # Image shear
        'perspective': 0.0, # Image perspective
        'flipud': 0.0,    # Image flip up-down
        'fliplr': 0.5,    # Image flip left-right
        'mosaic': 1.0,    # Image mosaic
        'mixup': 0.0,     # Image mixup
        'copy_paste': 0.0, # Copy-paste augmentation
    }
    
    print("\n⚙️  Training Configuration:")
    for key, value in training_args.items():
        if key in ['epochs', 'batch', 'device', 'imgsz', 'lr0', 'optimizer']:
            print(f"   {key}: {value}")
    
    # Start training
    try:
        print(f"\n🔥 Starting training...")
        results = model.train(**training_args)
        
        print(f"\n✅ Training completed successfully!")
        print(f"   📁 Results saved to: {output_dir / 'results'}")
        
        return results, model
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        raise

def validate_model(model, dataset_path, output_dir):
    """Validate the trained model"""
    print(f"\n🔍 Validating trained model...")
    
    try:
        # Run validation
        metrics = model.val(data=str(dataset_path / 'data.yaml'))
        
        print(f"✅ Validation completed!")
        print(f"   📊 mAP50: {metrics.box.map50:.4f}")
        print(f"   📊 mAP50-95: {metrics.box.map:.4f}")
        print(f"   📊 Precision: {metrics.box.mp:.4f}")
        print(f"   📊 Recall: {metrics.box.mr:.4f}")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Validation failed: {str(e)}")
        return None

def main():
    """Main training function"""
    print("🎯 YOLOv8 Instance Segmentation Training Script")
    print("=" * 60)
    
    try:
        # Check GPU
        device = check_gpu()
        
        # Setup environment
        base_dir, dataset_path, output_dir = setup_training_environment()
        
        # Load dataset info
        data_config = load_dataset_info(dataset_path / "data.yaml")
        
        # Train model
        results, model = train_model(dataset_path, output_dir, device, epochs=50)
        
        # Validate model
        metrics = validate_model(model, dataset_path, output_dir)
        
        print(f"\n🎉 Training Pipeline Completed Successfully!")
        print(f"   📁 Model saved to: {output_dir / 'results'}")
        print(f"   📊 Training metrics available in results folder")
        
        # Save model to models directory
        model_path = output_dir / 'models' / 'best.pt'
        model_path.parent.mkdir(exist_ok=True)
        
        # Copy best model
        best_model_path = Path(results.save_dir) / 'weights' / 'best.pt'
        if best_model_path.exists():
            import shutil
            shutil.copy2(best_model_path, model_path)
            print(f"   💾 Best model copied to: {model_path}")
        
    except Exception as e:
        print(f"\n❌ Training pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()




