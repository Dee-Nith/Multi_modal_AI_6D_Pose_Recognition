# 🤖 AI 6D Pose Recognition System

A comprehensive computer vision system for 6D pose estimation using YOLOv8 instance segmentation, designed for robotic grasping applications with CoppeliaSim integration.

## 🎯 Project Overview

This project implements a complete pipeline for 6D pose recognition, combining:
- **YOLOv8 Instance Segmentation** for object detection
- **6D Pose Estimation** for spatial orientation
- **CoppeliaSim Integration** for robotic simulation
- **Real-time Processing** with GPU acceleration

## 🚀 Key Features

- ✅ **Custom YOLOv8 Training** with YCB dataset
- ✅ **6D Pose Estimation** using PnP algorithms
- ✅ **CoppeliaSim Integration** for robotic simulation
- ✅ **Real-time Camera Processing** with Kinect support
- ✅ **GPU Acceleration** for fast inference
- ✅ **Comprehensive Evaluation** and testing pipeline

## 📁 Project Structure

```
AI_6D_Pose_recognition/
├── 📁 object_detection_model/          # YOLOv8 training and models
├── 📁 pose_estimation/                 # 6D pose estimation algorithms
├── 📁 src/                            # Core source code
├── 📁 docs/                           # Documentation and reports
├── 📁 examples/                       # Example scripts and demos
├── 📄 main.py                         # Main pipeline entry point
├── 📄 requirements.txt                # Python dependencies
├── 📄 MSc_Report_AI_6D_Pose_Recognition.md  # Complete project report
└── 📄 README.md                       # This file
```

## 🛠️ Installation

### Prerequisites
- **Python**: 3.8+
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: 8GB+ (16GB recommended)
- **Storage**: 10GB+ free space

### Quick Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/AI_6D_Pose_recognition.git
cd AI_6D_Pose_recognition
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Setup GPU training environment** (optional)
```bash
cd object_detection_model
python setup_gpu_training.py
```

## 🎯 Usage

### 1. Object Detection Training

Train a custom YOLOv8 model on your dataset:

```bash
cd object_detection_model
python train_instance_segmentation.py
```

### 2. 6D Pose Estimation

Run the complete pipeline:

```bash
python main.py
```

### 3. Real-time Processing

For live camera processing:

```bash
python live_camera_with_trained_model.py
```

### 4. CoppeliaSim Integration

Connect with CoppeliaSim for robotic simulation:

```bash
python live_coppelia_connection.py
```

## 📊 Dataset

The project uses the **YCB (Yale-CMU-Berkeley) Object and Model Set** with:
- **Objects**: banana, cracker_box, master_chef_can, mug, mustard_bottle
- **Format**: YOLO format with instance segmentation masks
- **Augmentation**: HSV, geometric, mosaic, and mixup techniques

## 🔧 Configuration

### Model Configuration
- **Architecture**: YOLOv8s-seg (Instance Segmentation)
- **Input Size**: 640x640
- **Batch Size**: 16 (adjustable based on GPU memory)
- **Epochs**: 50
- **Optimizer**: AdamW with cosine scheduling

### Performance Metrics
- **mAP50**: >0.85 (85%+ accuracy)
- **mAP50-95**: >0.70 (70%+ across IoU thresholds)
- **Precision**: >0.80 (80%+ precision)
- **Recall**: >0.80 (80%+ recall)

## 🧪 Testing

Run comprehensive tests:

```bash
# Test object detection
python test_trained_yolo.py

# Test 6D pose estimation
python test_pipeline.py

# Test CoppeliaSim integration
python test_real_camera_integration.py
```

## 📈 Results

The system achieves:
- **Real-time Processing**: 30+ FPS on GPU
- **High Accuracy**: 85%+ mAP50 on YCB dataset
- **Robust Detection**: Works in various lighting conditions
- **Scalable**: Supports multiple object classes

## 🔬 Research & Development

This project was developed as part of an MSc research project focusing on:
- Computer vision for robotic applications
- 6D pose estimation methodologies
- Real-time object detection and tracking
- Integration with robotic simulation environments

## 📚 Documentation

- **Complete Report**: `MSc_Report_AI_6D_Pose_Recognition.md`
- **LaTeX Report**: `MSc_Report_LaTeX.tex`
- **Setup Guides**: `YCB_IMPORT_GUIDE.md`, `LaTeX_Setup_Guide.md`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **YCB Dataset**: Yale-CMU-Berkeley Object and Model Set
- **YOLOv8**: Ultralytics for the detection framework
- **CoppeliaSim**: Coppelia Robotics for simulation environment
- **OpenCV**: Computer vision library
- **PyTorch**: Deep learning framework

## 📞 Contact

**Author**: [Your Name]  
**Email**: [your.email@example.com]  
**Project Link**: [https://github.com/yourusername/AI_6D_Pose_recognition](https://github.com/yourusername/AI_6D_Pose_recognition)

---

**⭐ If you found this project helpful, please give it a star!**

## 🔗 Related Projects

- [YOLOv8](https://github.com/ultralytics/ultralytics)
- [CoppeliaSim](https://www.coppeliarobotics.com/)
- [YCB Dataset](https://www.ycbbenchmarks.com/)

---

*This project was completed as part of an MSc research program, demonstrating advanced computer vision techniques for robotic applications.*