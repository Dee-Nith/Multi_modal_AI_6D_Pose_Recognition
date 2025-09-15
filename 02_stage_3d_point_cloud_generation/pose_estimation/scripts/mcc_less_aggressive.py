#!/usr/bin/env python3
"""
MCC-Less Aggressive 3D Reconstruction Pipeline
Preserves more points for better visualization
"""

import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from ultralytics import YOLO
import time
import json

class MCCLessAggressive:
    def __init__(self, model_path="training_results/yolov8s_instance_segmentation/weights/best.pt"):
        """Initialize the MCC-less aggressive 3D reconstruction pipeline"""
        print("🚀 Initializing MCC-Less Aggressive 3D Reconstruction Pipeline")
        
        # Load trained YOLO model
        self.model = YOLO(model_path)
        print(f"✅ YOLO model loaded: {model_path}")
        
        # 3D reconstruction parameters - LESS AGGRESSIVE
        self.voxel_size = 0.00001  # 0.01mm voxels (much smaller for more detail)
        self.depth_scale = 1000.0  # Depth scale factor
        self.min_points = 100  # Minimum points for reconstruction
        
        print("🎯 Pipeline ready for less aggressive 3D reconstruction!")
    
    def load_rgb_depth_data(self, rgb_path, depth_path):
        """Load RGB and depth data from CoppeliaSim captures"""
        print(f"\n📁 Loading data from:")
        print(f"   RGB: {rgb_path}")
        print(f"   Depth: {depth_path}")
        
        # Load RGB image
        rgb_img = cv2.imread(rgb_path)
        if rgb_img is None:
            raise ValueError(f"Could not load RGB image: {rgb_path}")
        
        # Load depth data (handle both .txt and .npy formats)
        if depth_path.endswith('.npy'):
            depth_data = np.load(depth_path)
        elif depth_path.endswith('.txt'):
            depth_data = np.loadtxt(depth_path, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported depth format: {depth_path}")
        
        # Reshape depth to match RGB dimensions
        height, width = rgb_img.shape[:2]
        if depth_data.size != height * width:
            try:
                depth_img = depth_data.reshape(height, width)
            except ValueError:
                print(f"⚠️ Depth reshape failed, using original shape")
                depth_img = depth_data
        else:
            depth_img = depth_data.reshape(height, width)
        
        print(f"✅ RGB: {rgb_img.shape}, Depth: {depth_img.shape}")
        print(f"📊 Depth range: {depth_img.min():.6f} to {depth_img.max():.6f}")
        return rgb_img, depth_img
    
    def detect_objects(self, rgb_img):
        """Detect objects using trained YOLOv8s model"""
        print("\n🔍 Running object detection...")
        
        # Run inference with lower confidence for more detections
        results = self.model(rgb_img, conf=0.25, verbose=False)
        
        if not results or len(results) == 0:
            print("⚠️ No objects detected")
            return []
        
        detections = []
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes
                masks = result.masks
                
                for i, (box, mask) in enumerate(zip(boxes, masks)):
                    # Get bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    class_name = result.names[cls]
                    
                    # Get mask
                    mask_data = mask.data[0].cpu().numpy()
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class': cls,
                        'class_name': class_name,
                        'mask': mask_data
                    })
                    
                    print(f"   📦 {class_name}: conf={conf:.3f}, bbox=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
        
        print(f"✅ Detected {len(detections)} objects")
        return detections
    
    def create_enhanced_point_cloud(self, rgb_img, depth_img, detections):
        """Create enhanced point cloud using MCC-inspired approach"""
        print("\n☁️ Creating enhanced 3D point cloud...")
        
        # Convert to Open3D format
        rgb_o3d = o3d.geometry.Image(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
        depth_o3d = o3d.geometry.Image(depth_img.astype(np.float32))
        
        # Create RGBD image
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d,
            depth_scale=self.depth_scale,
            depth_trunc=3.0,  # 3m max depth
            convert_rgb_to_intensity=False
        )
        
        # Camera intrinsic parameters (approximate for CoppeliaSim)
        width, height = rgb_img.shape[1], rgb_img.shape[0]
        fx = fy = max(width, height) * 0.8  # Approximate focal length
        cx, cy = width / 2, height / 2
        
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, fx, fy, cx, cy
        )
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        
        print(f"   📊 Initial point cloud: {len(pcd.points)} points")
        
        # Analyze coordinate system
        points = np.asarray(pcd.points)
        if len(points) > 0:
            print(f"   📊 Coordinate ranges:")
            print(f"      X: {points[:, 0].min():.6f} to {points[:, 0].max():.6f}")
            print(f"      Y: {points[:, 1].min():.6f} to {points[:, 1].max():.6f}")
            print(f"      Z: {points[:, 2].min():.6f} to {points[:, 2].max():.6f}")
        
        # Apply LESS AGGRESSIVE MCC-inspired enhancements
        pcd = self._apply_less_aggressive_enhancements(pcd, detections)
        
        print(f"✅ Enhanced point cloud: {len(pcd.points)} points")
        return pcd
    
    def _apply_less_aggressive_enhancements(self, pcd, detections):
        """Apply LESS AGGRESSIVE MCC-inspired enhancements to point cloud"""
        print("   🔧 Applying LESS aggressive MCC-inspired enhancements...")
        
        # 1. Remove statistical outliers (very gentle)
        if len(pcd.points) > 1000:
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=4.0)  # Much more lenient
            print(f"   📊 After outlier removal: {len(pcd.points)} points")
        
        # 2. Downsample for efficiency (much less aggressive)
        if len(pcd.points) > 10000:
            # Use much smaller voxel size for more detail
            pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
            print(f"   📊 After downsampling: {len(pcd.points)} points")
        
        # 3. Estimate normals for better surface reconstruction
        if len(pcd.points) > 10:
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.0005, max_nn=50  # Smaller radius, more neighbors
            ))
            print(f"   📊 Normals estimated")
        
        # 4. Apply MINIMAL object-aware filtering
        if detections:
            pcd = self._minimal_object_filtering(pcd, detections)
        
        # 5. Skip ground plane removal to preserve more points
        print("   ⏭️ Skipping ground plane removal to preserve points")
        
        print(f"   ✅ Enhanced point cloud: {len(pcd.points)} points")
        return pcd
    
    def _minimal_object_filtering(self, pcd, detections):
        """Minimal filtering that preserves most points"""
        print("   🧠 Applying minimal object filtering...")
        
        # Get points and colors
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        # Create mask for valid object regions - VERY GENEROUS
        valid_mask = np.zeros(len(points), dtype=bool)
        
        # Calculate image center for reference
        img_center_x = 320  # Approximate for 640x480
        img_center_y = 240
        
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Calculate bbox center
            bbox_center_x = (x1 + x2) / 2
            bbox_center_y = (y1 + y2) / 2
            
            # Calculate distance from image center
            distance_from_center = np.sqrt((bbox_center_x - img_center_x)**2 + (bbox_center_y - img_center_y)**2)
            
            # VERY generous filtering - keep most points
            scale_factor = 3.0  # Very generous scaling
            
            # Convert 2D bbox to 3D space with VERY generous scaling
            x_min = x1 * scale_factor / 1000000  # Scale down significantly
            x_max = x2 * scale_factor / 1000000
            y_min = y1 * scale_factor / 1000000
            y_max = y2 * scale_factor / 1000000
            z_min, z_max = 0.000001, 0.1  # Very wide depth range
            
            # Apply 3D bounding box filter
            valid_points = (
                (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
                (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
                (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
            )
            valid_mask |= valid_points
        
        # If filtering is too aggressive, keep ALL points
        if np.sum(valid_mask) < len(points) * 0.5:  # If less than 50% points remain
            print(f"   ⚠️ Filtering too aggressive, keeping ALL points")
            valid_mask = np.ones(len(points), dtype=bool)
        
        # Apply filter
        if np.any(valid_mask):
            pcd_filtered = o3d.geometry.PointCloud()
            pcd_filtered.points = o3d.utility.Vector3dVector(points[valid_mask])
            pcd_filtered.colors = o3d.utility.Vector3dVector(colors[valid_mask])
            print(f"   ✅ Minimal filtering: {len(points)} -> {len(pcd_filtered.points)} points")
            return pcd_filtered
        
        return pcd
    
    def run_full_pipeline(self, rgb_path, depth_path, output_path="mcc_less_aggressive_results"):
        """Run the complete MCC-less aggressive 3D reconstruction pipeline"""
        print("🚀 Starting MCC-Less Aggressive 3D Reconstruction Pipeline")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # 1. Load data
            rgb_img, depth_img = self.load_rgb_depth_data(rgb_path, depth_path)
            
            # 2. Detect objects
            detections = self.detect_objects(rgb_img)
            
            # 3. Create enhanced point cloud
            pcd = self.create_enhanced_point_cloud(rgb_img, depth_img, detections)
            
            # 4. Performance metrics
            total_time = time.time() - start_time
            print(f"\n⏱️ Pipeline completed in {total_time:.2f} seconds")
            print(f"📊 Performance: {len(pcd.points)/total_time:.0f} points/second")
            
            # 5. Save results
            output_dir = self._save_results(rgb_img, depth_img, pcd, output_path)
            
            return output_dir, pcd
            
        except Exception as e:
            print(f"❌ Pipeline failed: {str(e)}")
            raise
    
    def _save_results(self, rgb_img, depth_img, pcd, output_path):
        """Save all results"""
        print(f"\n💾 Saving results to: {output_path}")
        
        # Create output directory
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)
        
        # Save point cloud
        if len(pcd.points) > 0:
            o3d.io.write_point_cloud(str(output_dir / "enhanced_pointcloud.ply"), pcd)
            print(f"   ✅ Point cloud saved: enhanced_pointcloud.ply ({len(pcd.points)} points)")
        
        # Save images
        cv2.imwrite(str(output_dir / "rgb_input.jpg"), rgb_img)
        depth_viz = ((depth_img - depth_img.min()) / (depth_img.max() - depth_img.min()) * 255).astype(np.uint8)
        cv2.imwrite(str(output_dir / "depth_input.jpg"), depth_viz)
        print(f"   ✅ Images saved")
        
        return output_dir

def main():
    """Main function to demonstrate the pipeline"""
    print("🎯 MCC-Less Aggressive 3D Reconstruction Demo")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = MCCLessAggressive()
    
    # Example usage with your CoppeliaSim data
    rgb_path = "new_captures/all_captures/processed_images/capture_1_rgb.jpg"
    depth_path = "new_captures/all_captures/processed_images/capture_1_depth.npy"
    
    # Check if files exist
    if not Path(rgb_path).exists():
        print(f"❌ RGB file not found: {rgb_path}")
        print("Please provide valid paths to your CoppeliaSim captures")
        return
    
    if not Path(depth_path).exists():
        print(f"❌ Depth file not found: {depth_path}")
        print("Please provide valid paths to your CoppeliaSim captures")
        return
    
    # Run pipeline
    try:
        output_dir, pcd = pipeline.run_full_pipeline(rgb_path, depth_path)
        print(f"\n🎉 Success! Check results in: {output_dir}")
        
        # Print summary
        print(f"\n📊 Final Point Cloud Summary:")
        print(f"   Total Points: {len(pcd.points)}")
        if len(pcd.points) > 0:
            points = np.asarray(pcd.points)
            print(f"   X Range: {points[:, 0].min():.6f} to {points[:, 0].max():.6f}")
            print(f"   Y Range: {points[:, 1].min():.6f} to {points[:, 1].max():.6f}")
            print(f"   Z Range: {points[:, 2].min():.6f} to {points[:, 2].max():.6f}")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")

if __name__ == "__main__":
    main()




