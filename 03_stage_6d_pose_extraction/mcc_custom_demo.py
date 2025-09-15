#!/usr/bin/env python3
"""
Custom MCC Demo - Works without PyTorch3D
Uses MCC-inspired approach with our CoppeliaSim RGB-D data
"""

import numpy as np
import cv2
import torch
import json
import open3d as o3d
from pathlib import Path
import matplotlib.pyplot as plt
from ultralytics import YOLO
import time

class CustomMCCDemo:
    def __init__(self, model_path="training_results/yolov8s_instance_segmentation/weights/best.pt"):
        """Initialize Custom MCC Demo"""
        print("🚀 Initializing Custom MCC Demo (No PyTorch3D Required)")
        
        # Load YOLO model for object detection
        self.model = YOLO(model_path)
        
        # MCC-inspired parameters
        self.depth_scale = 1000.0
        self.occupancy_threshold = 0.5
        self.max_points = 20000  # Limit for visualization
        
        print("✅ Custom MCC Demo ready!")
    
    def load_rgbd_data(self, rgb_path, depth_path):
        """Load RGB-D data like original MCC demo"""
        print(f"\n📁 Loading RGB-D data:")
        print(f"   RGB: {rgb_path}")
        print(f"   Depth: {depth_path}")
        
        # Load RGB
        rgb = cv2.imread(rgb_path)
        if rgb is None:
            raise ValueError(f"Could not load RGB: {rgb_path}")
        
        # Load depth
        if depth_path.endswith('.npy'):
            depth = np.load(depth_path)
        elif depth_path.endswith('.txt'):
            depth = np.loadtxt(depth_path, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported depth format: {depth_path}")
        
        # Ensure depth matches RGB dimensions
        if depth.size != rgb.shape[0] * rgb.shape[1]:
            depth = depth.reshape(rgb.shape[:2])
        
        print(f"✅ Loaded RGB: {rgb.shape}, Depth: {depth.shape}")
        return rgb, depth
    
    def generate_segmentation(self, rgb):
        """Generate segmentation mask using YOLO (replaces the .seg file)"""
        print("\n🎯 Generating segmentation with YOLO...")
        
        results = self.model(rgb, conf=0.25, verbose=False)
        
        # Create combined mask
        height, width = rgb.shape[:2]
        mask = np.zeros((height, width), dtype=bool)
        
        detections = []
        if results and len(results) > 0:
            for result in results:
                if result.masks is not None:
                    for i, mask_data in enumerate(result.masks):
                        # Get mask
                        mask_array = mask_data.data[0].cpu().numpy()
                        mask_resized = cv2.resize(mask_array.astype(np.float32), (width, height))
                        mask = np.logical_or(mask, mask_resized > 0.5)
                        
                        # Store detection info
                        box = result.boxes[i]
                        cls = int(box.cls[0].cpu().numpy())
                        conf = float(box.conf[0].cpu().numpy())
                        class_name = result.names[cls]
                        
                        detections.append({
                            'class': class_name,
                            'confidence': conf,
                            'mask_area': np.sum(mask_resized > 0.5)
                        })
        
        print(f"✅ Generated segmentation: {np.sum(mask)} pixels, {len(detections)} objects")
        for det in detections:
            print(f"   📦 {det['class']}: conf={det['confidence']:.3f}, area={det['mask_area']}")
        
        return mask, detections
    
    def create_point_cloud_from_rgbd(self, rgb, depth, mask=None):
        """Create point cloud from RGB-D (mimics MCC's point cloud generation)"""
        print("\n☁️ Creating point cloud from RGB-D...")
        
        # Convert to Open3D
        rgb_o3d = o3d.geometry.Image(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        depth_o3d = o3d.geometry.Image(depth.astype(np.float32))
        
        # Create RGBD image
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d,
            depth_scale=self.depth_scale,
            depth_trunc=3.0,
            convert_rgb_to_intensity=False
        )
        
        # Camera intrinsics (approximate)
        height, width = rgb.shape[:2]
        fx = fy = max(width, height) * 0.8
        cx, cy = width / 2, height / 2
        
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, fx, fy, cx, cy
        )
        
        # Generate point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        
        # Apply mask if provided
        if mask is not None:
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            
            # Create 3D mask by projecting 2D mask
            if len(points) > 0:
                # Simple approach: use mask to filter points
                mask_3d = self._project_mask_to_3d(points, mask, width, height, intrinsic)
                
                if np.any(mask_3d):
                    pcd_filtered = o3d.geometry.PointCloud()
                    pcd_filtered.points = o3d.utility.Vector3dVector(points[mask_3d])
                    pcd_filtered.colors = o3d.utility.Vector3dVector(colors[mask_3d])
                    pcd = pcd_filtered
        
        print(f"✅ Point cloud created: {len(pcd.points)} points")
        return pcd
    
    def _project_mask_to_3d(self, points, mask_2d, width, height, intrinsic):
        """Project 2D mask to 3D points"""
        # Simple projection back to image coordinates
        mask_3d = np.zeros(len(points), dtype=bool)
        
        if len(points) == 0:
            return mask_3d
        
        # Project 3D points to 2D
        fx, fy = intrinsic.get_focal_length()
        cx, cy = intrinsic.get_principal_point()
        
        x_2d = (points[:, 0] * fx / points[:, 2] + cx).astype(int)
        y_2d = (points[:, 1] * fy / points[:, 2] + cy).astype(int)
        
        # Check bounds and apply mask
        valid = (x_2d >= 0) & (x_2d < width) & (y_2d >= 0) & (y_2d < height) & (points[:, 2] > 0)
        
        for i in range(len(points)):
            if valid[i] and mask_2d[y_2d[i], x_2d[i]]:
                mask_3d[i] = True
        
        return mask_3d
    
    def mcc_inspired_enhancement(self, pcd):
        """Apply MCC-inspired enhancements to point cloud"""
        print("\n🔧 Applying MCC-inspired enhancements...")
        
        if len(pcd.points) == 0:
            return pcd
        
        # 1. Outlier removal (gentle)
        if len(pcd.points) > 50:
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=3.0)
            print(f"   📊 After outlier removal: {len(pcd.points)} points")
        
        # 2. Estimate normals
        if len(pcd.points) > 10:
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.001, max_nn=30
            ))
            print(f"   📊 Normals estimated")
        
        # 3. Downsample if too many points (for visualization)
        if len(pcd.points) > self.max_points:
            # Calculate voxel size to get approximately max_points
            points = np.asarray(pcd.points)
            bbox_size = np.max(points, axis=0) - np.min(points, axis=0)
            volume = np.prod(bbox_size)
            target_voxel_volume = volume / self.max_points
            voxel_size = target_voxel_volume ** (1/3)
            
            pcd = pcd.voxel_down_sample(voxel_size=max(voxel_size, 0.00001))
            print(f"   📊 Downsampled to: {len(pcd.points)} points")
        
        return pcd
    
    def create_mcc_visualization(self, rgb, pcd, detections, output_dir):
        """Create MCC-style visualization"""
        print(f"\n🎨 Creating MCC-style visualization...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 1. Save original RGB
        cv2.imwrite(str(output_path / "input_rgb.jpg"), rgb)
        
        # 2. Save point cloud
        if len(pcd.points) > 0:
            o3d.io.write_point_cloud(str(output_path / "enhanced_pointcloud.ply"), pcd)
        
        # 3. Create summary visualization
        fig = plt.figure(figsize=(20, 12))
        
        # RGB image
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        plt.title('Input RGB Image', fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # Point cloud statistics
        plt.subplot(2, 3, 2)
        if len(pcd.points) > 0:
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            
            # Top-down view
            plt.scatter(points[:, 0], points[:, 1], c=colors, s=1, alpha=0.6)
            plt.title(f'Point Cloud Top View\n{len(points)} points', fontsize=14, fontweight='bold')
            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
            plt.axis('equal')
        else:
            plt.text(0.5, 0.5, 'No points generated', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Point Cloud Top View', fontsize=14, fontweight='bold')
        
        # Object detection summary
        plt.subplot(2, 3, 3)
        if detections:
            objects = [det['class'] for det in detections]
            confidences = [det['confidence'] for det in detections]
            
            plt.barh(range(len(objects)), confidences, color='skyblue')
            plt.yticks(range(len(objects)), objects)
            plt.xlabel('Confidence')
            plt.title(f'Detected Objects ({len(detections)})', fontsize=14, fontweight='bold')
            plt.xlim(0, 1)
        else:
            plt.text(0.5, 0.5, 'No objects detected', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Detected Objects', fontsize=14, fontweight='bold')
        
        # Coordinate ranges
        plt.subplot(2, 3, 4)
        if len(pcd.points) > 0:
            points = np.asarray(pcd.points)
            ranges = ['X', 'Y', 'Z']
            min_vals = [points[:, i].min() for i in range(3)]
            max_vals = [points[:, i].max() for i in range(3)]
            
            x = np.arange(len(ranges))
            width = 0.35
            
            plt.bar(x - width/2, min_vals, width, label='Min', alpha=0.7)
            plt.bar(x + width/2, max_vals, width, label='Max', alpha=0.7)
            
            plt.xlabel('Coordinate')
            plt.ylabel('Value (m)')
            plt.title('Coordinate Ranges', fontsize=14, fontweight='bold')
            plt.xticks(x, ranges)
            plt.legend()
        
        # Point density analysis
        plt.subplot(2, 3, 5)
        if len(pcd.points) > 0:
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            
            # Depth distribution
            depths = points[:, 2]
            plt.hist(depths, bins=50, alpha=0.7, color='green')
            plt.xlabel('Depth (m)')
            plt.ylabel('Point Count')
            plt.title('Depth Distribution', fontsize=14, fontweight='bold')
        
        # Project info
        plt.subplot(2, 3, 6)
        plt.text(0.1, 0.9, 'MCC-Inspired 3D Reconstruction', fontsize=16, fontweight='bold', transform=plt.gca().transAxes)
        plt.text(0.1, 0.8, f'Total Points: {len(pcd.points)}', fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.7, f'Objects Detected: {len(detections)}', fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.6, 'Technology Stack:', fontsize=12, fontweight='bold', transform=plt.gca().transAxes)
        plt.text(0.1, 0.5, '• YOLOv8s Instance Segmentation', fontsize=10, transform=plt.gca().transAxes)
        plt.text(0.1, 0.4, '• Open3D Point Cloud Processing', fontsize=10, transform=plt.gca().transAxes)
        plt.text(0.1, 0.3, '• MCC-Inspired Enhancement', fontsize=10, transform=plt.gca().transAxes)
        plt.text(0.1, 0.2, '• CoppeliaSim RGB-D Integration', fontsize=10, transform=plt.gca().transAxes)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path / "mcc_demo_results.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 4. Save summary JSON
        summary = {
            'total_points': int(len(pcd.points)),
            'objects_detected': int(len(detections)),
            'detections': [
                {
                    'class': det['class'],
                    'confidence': float(det['confidence']),
                    'mask_area': int(det['mask_area'])
                }
                for det in detections
            ],
            'coordinate_ranges': {}
        }
        
        if len(pcd.points) > 0:
            points = np.asarray(pcd.points)
            summary['coordinate_ranges'] = {
                'x_min': float(points[:, 0].min()),
                'x_max': float(points[:, 0].max()),
                'y_min': float(points[:, 1].min()),
                'y_max': float(points[:, 1].max()),
                'z_min': float(points[:, 2].min()),
                'z_max': float(points[:, 2].max())
            }
        
        try:
            with open(output_path / "mcc_demo_summary.json", 'w') as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save JSON summary: {e}")
            # Save a simple text summary instead
            with open(output_path / "mcc_demo_summary.txt", 'w') as f:
                f.write(f"Custom MCC Demo Results\n")
                f.write(f"========================\n")
                f.write(f"Total Points: {len(pcd.points)}\n")
                f.write(f"Objects Detected: {len(detections)}\n")
                for i, det in enumerate(detections):
                    f.write(f"  {i+1}. {det['class']}: {det['confidence']:.3f}\n")
        
        print(f"✅ Visualization saved to: {output_path}")
        return output_path
    
    def run_demo(self, rgb_path, depth_path, output_dir="mcc_custom_demo_results"):
        """Run the complete custom MCC demo"""
        print("🚀 Starting Custom MCC Demo")
        print("=" * 50)
        
        start_time = time.time()
        
        try:
            # 1. Load RGB-D data
            rgb, depth = self.load_rgbd_data(rgb_path, depth_path)
            
            # 2. Generate segmentation
            mask, detections = self.generate_segmentation(rgb)
            
            # 3. Create point cloud
            pcd = self.create_point_cloud_from_rgbd(rgb, depth, mask)
            
            # 4. Apply MCC-inspired enhancements
            pcd_enhanced = self.mcc_inspired_enhancement(pcd)
            
            # 5. Create visualization
            output_path = self.create_mcc_visualization(rgb, pcd_enhanced, detections, output_dir)
            
            # 6. Performance summary
            total_time = time.time() - start_time
            print(f"\n⏱️ Demo completed in {total_time:.2f} seconds")
            
            # 7. Open interactive viewer
            if len(pcd_enhanced.points) > 0:
                print(f"\n🎮 Opening interactive 3D viewer...")
                o3d.visualization.draw_geometries([pcd_enhanced], 
                                                window_name="Custom MCC Demo - 3D Point Cloud")
            
            print(f"\n🎉 Demo completed successfully!")
            print(f"📁 Results saved to: {output_path}")
            print(f"📄 Files created:")
            print(f"   - mcc_demo_results.png (Summary visualization)")
            print(f"   - enhanced_pointcloud.ply (3D point cloud)")
            print(f"   - mcc_demo_summary.json (Analysis results)")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Demo failed: {str(e)}")
            raise

def main():
    """Main function to run the demo"""
    print("🎯 Custom MCC Demo - No PyTorch3D Required")
    print("=" * 45)
    
    # Initialize demo
    demo = CustomMCCDemo()
    
    # Use your CoppeliaSim data
    rgb_path = "new_captures/all_captures/processed_images/capture_1_rgb.jpg"
    depth_path = "new_captures/all_captures/processed_images/capture_1_depth.npy"
    
    # Check files exist
    if not Path(rgb_path).exists():
        print(f"❌ RGB file not found: {rgb_path}")
        return
    
    if not Path(depth_path).exists():
        print(f"❌ Depth file not found: {depth_path}")
        return
    
    # Run demo
    try:
        output_dir = demo.run_demo(rgb_path, depth_path)
        print(f"\n✅ Success! Check the results and 3D viewer.")
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")

if __name__ == "__main__":
    main()
