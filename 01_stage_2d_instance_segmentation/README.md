# 🧠 Training Stage

This stage handles dataset preparation, model training, and output management for multiple deep learning models.

## 📁 Subfolders

### 📊 dataset_preparation
- YCB dataset processing
- Data format conversion
- Train/validation/test splits
- Annotation validation

### 🎯 yolo_training
- YOLOv8 training scripts
- YOLOv9 training scripts
- Multi-model ensemble training
- Hyperparameter optimization

### 🎁 model_outputs
- Trained model weights (.pt files)
- Training logs and metrics
- Model configurations
- Performance reports

### 🔄 data_augmentation
- Image augmentation pipelines
- Synthetic data generation
- Domain adaptation techniques
- Data balancing utilities

### 📦 ycb_dataset_processing
- YCB dataset download and setup
- Model mesh processing
- Texture mapping utilities
- Dataset validation tools

## 🚀 Usage

1. **Prepare Dataset**
   ```bash
   cd dataset_preparation
   python prepare_ycb_dataset.py
   ```

2. **Train Models**
   ```bash
   cd yolo_training
   python train_multi_model.py
   ```

3. **Augment Data**
   ```bash
   cd data_augmentation
   python augment_dataset.py
   ```

## 📊 Training Configuration

- **Models**: YOLOv8s, YOLOv8m, YOLOv9c
- **Input Size**: 640x640
- **Batch Size**: 16-32 (GPU dependent)
- **Epochs**: 100-200
- **Optimizer**: AdamW with cosine scheduling

## 📈 Expected Performance

- **mAP50**: >90%
- **mAP50-95**: >75%
- **Training Time**: 4-8 hours on RTX 3080
- **Model Size**: 50-200MB per model
