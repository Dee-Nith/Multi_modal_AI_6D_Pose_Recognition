# 📷 Data Capture Stage

This stage handles all data collection activities for the multi-model pose recognition system.

## 📁 Subfolders

### 🎬 coppelia_scene_setup
- CoppeliaSim scene files (.ttt)
- YCB model configurations
- Camera and lighting setup
- Robot workspace configuration

### 📜 lua_capture_scripts
- Auto-capture Lua scripts
- Multi-angle capture utilities
- Real-time data streaming scripts
- Kinect integration scripts

### 📹 kinect_integration
- Kinect camera setup
- Depth sensor configuration
- RGB-D data synchronization
- Calibration utilities

### 💾 data_storage
- Captured image datasets
- RGB-D data files
- Metadata and annotations
- Data organization tools

### 📐 camera_calibration
- Intrinsic camera parameters
- Extrinsic calibration data
- Distortion correction
- Multi-camera calibration

## 🚀 Usage

1. **Setup CoppeliaSim Scene**
   ```bash
   cd coppelia_scene_setup
   # Load scene files in CoppeliaSim
   ```

2. **Run Capture Scripts**
   ```bash
   cd lua_capture_scripts
   # Execute auto_kinect_capture.lua
   ```

3. **Store and Organize Data**
   ```bash
   cd data_storage
   # Use organization utilities
   ```

## 📊 Output Data

- **RGB Images**: 640x480 resolution
- **Depth Maps**: 16-bit depth data
- **Metadata**: Camera poses, timestamps
- **Annotations**: Object labels and bounding boxes
