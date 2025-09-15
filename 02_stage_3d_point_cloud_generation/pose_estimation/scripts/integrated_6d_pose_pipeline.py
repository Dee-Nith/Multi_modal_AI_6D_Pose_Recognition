#!/usr/bin/env python3
"""
Integrated 6D Pose Recognition Pipeline
Combines YOLOv8s instance segmentation, 6D pose estimation, and point cloud generation
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
from scipy.spatial.transform import Rotation
import glob

class Integrated6DPosePipeline:
    def __init__(self, model_path="training_results/yolov8s_instance_segmentation/weights/best.pt"):
        """Initialize the integrated 6D pose pipeline"""
        print("🚀 Initializing Integrated 6D Pose Recognition Pipeline")
        
        # Load trained YOLO model
        self.model = YOLO(model_path)
        
        # 6D pose estimation parameters
        self.depth_scale = 1000.0
        self.min_confidence = 0.25
        self.voxel_size = 0.001  # 1mm voxels
        
        # Camera intrinsic parameters (approximate for CoppeliaSim)
        self.camera_matrix = np.array([
            [500, 0, 320],    # fx, 0, cx
            [0, 500, 240],    # 0, fy, cy  
            [0, 0, 1]         # 0, 0, 1
        ])
        
        print("✅ Integrated 6D Pose Pipeline ready!")
    
    def load_rgb_depth_data(self, rgb_path, depth_path):
        """Load RGB and depth data"""
        print(f"\n📁 Loading RGB-D data:")
        print(f"   RGB: {rgb_path}")
        print(f"   Depth: {depth_path}")
        
        # Load RGB image
        rgb = cv2.imread(rgb_path)
        if rgb is None:
            raise ValueError(f"Could not load RGB: {rgb_path}")
        
        # Load depth data
        if depth_path.endswith('.npy'):
            depth = np.load(depth_path)
        elif depth_path.endswith('.txt'):
            depth = np.loadtxt(depth_path, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported depth format: {depth_path}")
        
        # Reshape depth to match RGB dimensions
        if depth.size != rgb.shape[0] * rgb.shape[1]:
            depth = depth.reshape(rgb.shape[:2])
        
        print(f"✅ RGB: {rgb.shape}, Depth: {depth.shape}")
        return rgb, depth
    
    def detect_objects_with_masks(self, rgb):
        """Detect objects and extract instance masks"""
        print("\n🎯 Running instance segmentation...")
        
        results = self.model(rgb, conf=self.min_confidence, verbose=False)
        
        detections = []
        if results and len(results) > 0:
            for result in results:
                if result.boxes is not None and result.masks is not None:
                    boxes = result.boxes
                    masks = result.masks
                    
                    for i, (box, mask) in enumerate(zip(boxes, masks)):
                        # Get bounding box
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())
                        class_name = result.names[cls]
                        
                        # Get mask
                        mask_data = mask.data[0].cpu().numpy()
                        
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'class': cls,
                            'class_name': class_name,
                            'mask': mask_data,
                            'mask_area': np.sum(mask_data > 0.5)
                        })
                        
                        print(f"   📦 {class_name}: conf={conf:.3f}, bbox=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}], area={detections[-1]['mask_area']}")
        
        print(f"✅ Detected {len(detections)} objects")
        return detections
    
    def estimate_6d_pose_from_mask(self, rgb, depth, detection):
        """Estimate 6D pose using mask and depth information"""
        print(f"\n📐 Estimating 6D pose for {detection['class_name']}...")
        
        # Extract mask region
        mask = detection['mask']
        bbox = detection['bbox']
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Resize mask to image dimensions
        height, width = rgb.shape[:2]
        mask_resized = cv2.resize(mask.astype(np.float32), (width, height))
        
        # Create 3D points from masked depth
        points_3d = []
        colors_3d = []
        
        for y in range(y1, y2):
            for x in range(x1, x2):
                if mask_resized[y, x] > 0.5:  # Mask threshold
                    depth_val = depth[y, x]
                    if depth_val > 0 and depth_val < 3.0:  # Valid depth range
                        # Back-project to 3D
                        z = depth_val / self.depth_scale
                        x_3d = (x - self.camera_matrix[0, 2]) * z / self.camera_matrix[0, 0]
                        y_3d = (y - self.camera_matrix[1, 2]) * z / self.camera_matrix[1, 1]
                        
                        points_3d.append([x_3d, y_3d, z])
                        colors_3d.append(rgb[y, x] / 255.0)  # Normalize colors
        
        if len(points_3d) < 10:
            print(f"   ⚠️ Insufficient points for pose estimation: {len(points_3d)}")
            return None
        
        points_3d = np.array(points_3d)
        colors_3d = np.array(colors_3d)
        
        # Estimate pose using PCA (Principal Component Analysis)
        pose = self._estimate_pose_pca(points_3d, colors_3d, detection)
        
        if pose:
            print(f"   ✅ 6D pose estimated successfully")
            print(f"      Position: [{pose['translation'][0]:.4f}, {pose['translation'][1]:.4f}, {pose['translation'][2]:.4f}]")
            print(f"      Rotation: [{pose['rotation_euler'][0]:.2f}°, {pose['rotation_euler'][1]:.2f}°, {pose['rotation_euler'][2]:.2f}°]")
        
        return pose
    
    def _estimate_pose_pca(self, points_3d, colors_3d, detection):
        """Estimate pose using PCA on 3D points"""
        try:
            # Center the points
            centroid = np.mean(points_3d, axis=0)
            centered_points = points_3d - centroid
            
            # Compute covariance matrix
            cov_matrix = np.cov(centered_points.T)
            
            # Eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            
            # Sort eigenvectors by eigenvalues (descending)
            sorted_indices = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, sorted_indices]
            
            # Extract rotation matrix
            rotation_matrix = eigenvectors
            
            # Ensure right-handed coordinate system
            if np.linalg.det(rotation_matrix) < 0:
                rotation_matrix[:, 2] *= -1
            
            # Convert to Euler angles
            rotation_euler = self._rotation_matrix_to_euler(rotation_matrix)
            
            # Estimate object dimensions
            dimensions = self._estimate_object_dimensions(points_3d, rotation_matrix)
            
            pose = {
                'translation': centroid.tolist(),
                'rotation_matrix': rotation_matrix.tolist(),
                'rotation_euler': rotation_euler.tolist(),
                'dimensions': dimensions.tolist(),
                'confidence': detection['confidence'],
                'class_name': detection['class_name'],
                'points_count': len(points_3d)
            }
            
            return pose
            
        except Exception as e:
            print(f"   ❌ PCA pose estimation failed: {e}")
            return None
    
    def _rotation_matrix_to_euler(self, rotation_matrix):
        """Convert rotation matrix to Euler angles (ZYX convention)"""
        try:
            # Extract Euler angles from rotation matrix
            sy = np.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
            
            if sy > 1e-6:
                roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                pitch = np.arctan2(-rotation_matrix[2, 0], sy)
                yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            else:
                roll = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                pitch = np.arctan2(-rotation_matrix[2, 0], sy)
                yaw = 0
            
            # Convert to degrees
            euler_angles = np.array([roll, pitch, yaw]) * 180 / np.pi
            return euler_angles
            
        except Exception as e:
            print(f"   ⚠️ Euler conversion failed: {e}")
            return np.array([0, 0, 0])
    
    def _estimate_object_dimensions(self, points_3d, rotation_matrix):
        """Estimate object dimensions along principal axes"""
        try:
            # Transform points to object coordinate system
            centered_points = points_3d - np.mean(points_3d, axis=0)
            transformed_points = centered_points @ rotation_matrix.T
            
            # Calculate dimensions along each axis
            dimensions = np.max(transformed_points, axis=0) - np.min(transformed_points, axis=0)
            return dimensions
            
        except Exception as e:
            print(f"   ⚠️ Dimension estimation failed: {e}")
            return np.array([0.1, 0.1, 0.1])
    
    def create_enhanced_point_cloud(self, rgb, depth, detections, poses):
        """Create enhanced point cloud with object-aware processing"""
        print("\n☁️ Creating enhanced point cloud with 6D pose information...")
        
        # Convert to Open3D format
        rgb_o3d = o3d.geometry.Image(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        depth_o3d = o3d.geometry.Image(depth.astype(np.float32))
        
        # Create RGBD image
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d,
            depth_scale=self.depth_scale,
            depth_trunc=3.0,
            convert_rgb_to_intensity=False
        )
        
        # Create point cloud
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            rgb.shape[1], rgb.shape[0],
            self.camera_matrix[0, 0], self.camera_matrix[1, 1],
            self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        )
        
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        print(f"   📊 Initial point cloud: {len(pcd.points)} points")
        
        # Apply object-aware filtering
        pcd_enhanced = self._apply_object_aware_filtering(pcd, detections, poses)
        
        return pcd_enhanced
    
    def _apply_object_aware_filtering(self, pcd, detections, poses):
        """Apply intelligent filtering based on detected objects and poses"""
        print("   🔧 Applying object-aware filtering...")
        
        if len(pcd.points) == 0:
            return pcd
        
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        # Create mask for valid object regions
        valid_mask = np.zeros(len(points), dtype=bool)
        
        for detection, pose in zip(detections, poses):
            if pose is None:
                continue
            
            # Get object bounding box in 3D
            bbox_3d = self._get_3d_bounding_box(pose, detection)
            
            # Filter points within 3D bounding box
            object_mask = (
                (points[:, 0] >= bbox_3d['x_min']) & (points[:, 0] <= bbox_3d['x_max']) &
                (points[:, 1] >= bbox_3d['y_min']) & (points[:, 1] <= bbox_3d['y_max']) &
                (points[:, 2] >= bbox_3d['z_min']) & (points[:, 2] <= bbox_3d['z_max'])
            )
            
            valid_mask |= object_mask
        
        # If no objects detected, keep all points
        if not np.any(valid_mask):
            print("   ⚠️ No valid objects, keeping all points")
            valid_mask = np.ones(len(points), dtype=bool)
        
        # Apply filter
        pcd_filtered = o3d.geometry.PointCloud()
        pcd_filtered.points = o3d.utility.Vector3dVector(points[valid_mask])
        pcd_filtered.colors = o3d.utility.Vector3dVector(colors[valid_mask])
        
        print(f"   ✅ Object-aware filtering: {len(points)} -> {len(pcd_filtered.points)} points")
        
        # Post-process
        if len(pcd_filtered.points) > 0:
            # Remove outliers
            pcd_filtered, _ = pcd_filtered.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            
            # Downsample for efficiency
            pcd_filtered = pcd_filtered.voxel_down_sample(voxel_size=self.voxel_size)
            
            # Estimate normals
            pcd_filtered.estimate_normals()
            
            print(f"   ✅ Post-processed: {len(pcd_filtered.points)} points")
        
        return pcd_filtered
    
    def _get_3d_bounding_box(self, pose, detection):
        """Get 3D bounding box for object based on pose and dimensions"""
        # Expand bounding box to account for object size
        expansion_factor = 1.5
        
        dimensions = np.array(pose['dimensions'])
        center = np.array(pose['translation'])
        
        half_dims = dimensions * expansion_factor / 2
        
        bbox_3d = {
            'x_min': center[0] - half_dims[0],
            'x_max': center[0] + half_dims[0],
            'y_min': center[1] - half_dims[1],
            'y_max': center[1] + half_dims[1],
            'z_min': center[2] - half_dims[2],
            'z_max': center[2] + half_dims[2]
        }
        
        return bbox_3d
    
    def create_comprehensive_visualization(self, rgb, depth, detections, poses, pcd_enhanced, output_dir):
        """Create comprehensive visualization of all pipeline outputs"""
        print(f"\n🎨 Creating comprehensive visualization...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save enhanced point cloud
        if len(pcd_enhanced.points) > 0:
            o3d.io.write_point_cloud(str(output_path / "enhanced_pointcloud_6d.ply"), pcd_enhanced)
            print(f"   ✅ Enhanced point cloud saved")
        
        # Create visualization
        fig = plt.figure(figsize=(20, 15))
        
        # RGB with bounding boxes and poses
        plt.subplot(3, 3, 1)
        rgb_display = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        for detection, pose in zip(detections, poses):
            bbox = detection['bbox']
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Draw bounding box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, edgecolor='red', linewidth=2)
            plt.gca().add_patch(rect)
            
            # Add pose information
            if pose:
                pos_text = f"{detection['class_name']}\nPos: [{pose['translation'][0]:.3f}, {pose['translation'][1]:.3f}, {pose['translation'][2]:.3f}]"
                plt.text(x1, y1-10, pos_text, color='red', fontsize=8, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white'))
        
        plt.imshow(rgb_display)
        plt.title('RGB with 6D Poses', fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # Depth visualization
        plt.subplot(3, 3, 2)
        depth_viz = (depth - depth.min()) / (depth.max() - depth.min()) * 255
        plt.imshow(depth_viz, cmap='viridis')
        plt.title('Depth Map', fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # Point cloud top view
        plt.subplot(3, 3, 3)
        if len(pcd_enhanced.points) > 0:
            points = np.asarray(pcd_enhanced.points)
            colors = np.asarray(pcd_enhanced.colors)
            plt.scatter(points[:, 0], points[:, 1], c=colors, s=1, alpha=0.6)
            plt.title(f'Enhanced Point Cloud\n{len(points)} points', fontsize=14, fontweight='bold')
            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
            plt.axis('equal')
        else:
            plt.text(0.5, 0.5, 'No points', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Enhanced Point Cloud', fontsize=14, fontweight='bold')
        
        # 6D pose summary
        plt.subplot(3, 3, 4)
        self._plot_6d_pose_summary(detections, poses)
        
        # Object dimensions
        plt.subplot(3, 3, 5)
        self._plot_object_dimensions(detections, poses)
        
        # Point cloud statistics
        plt.subplot(3, 3, 6)
        if len(pcd_enhanced.points) > 0:
            points = np.asarray(pcd_enhanced.points)
            plt.hist(points[:, 2], bins=30, alpha=0.7, color='green')
            plt.xlabel('Depth (m)')
            plt.ylabel('Point Count')
            plt.title('Depth Distribution', fontsize=14, fontweight='bold')
        
        # Pipeline performance
        plt.subplot(3, 3, 7)
        self._plot_pipeline_performance(detections, poses, pcd_enhanced)
        
        # 3D scene overview
        plt.subplot(3, 3, 8)
        if len(pcd_enhanced.points) > 0:
            points = np.asarray(pcd_enhanced.points)
            colors = np.asarray(pcd_enhanced.colors)
            plt.scatter(points[:, 0], points[:, 2], c=colors, s=1, alpha=0.6)
            plt.title('3D Scene Overview (X-Z)', fontsize=14, fontweight='bold')
            plt.xlabel('X (m)')
            plt.ylabel('Z (m)')
            plt.axis('equal')
        
        # Project summary
        plt.subplot(3, 3, 9)
        self._plot_project_summary(detections, poses, pcd_enhanced)
        
        plt.tight_layout()
        plt.savefig(output_path / "integrated_6d_pipeline_results.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save detailed results
        self._save_detailed_results(detections, poses, pcd_enhanced, output_path)
        
        print(f"✅ Comprehensive visualization saved to: {output_path}")
        return output_path
    
    def _plot_6d_pose_summary(self, detections, poses):
        """Plot 6D pose summary"""
        plt.title('6D Pose Summary', fontsize=14, fontweight='bold')
        
        if not poses or all(p is None for p in poses):
            plt.text(0.5, 0.5, 'No poses estimated', ha='center', va='center', transform=plt.gca().transAxes)
            return
        
        valid_poses = [p for p in poses if p is not None]
        if not valid_poses:
            plt.text(0.5, 0.5, 'No valid poses', ha='center', va='center', transform=plt.gca().transAxes)
            return
        
        # Plot translation vs confidence
        translations = np.array([p['translation'] for p in valid_poses])
        confidences = np.array([p['confidence'] for p in valid_poses])
        
        plt.scatter(translations[:, 0], translations[:, 2], c=confidences, cmap='viridis', s=100)
        plt.xlabel('X (m)')
        plt.ylabel('Z (m)')
        plt.colorbar(label='Confidence')
        plt.grid(True, alpha=0.3)
    
    def _plot_object_dimensions(self, detections, poses):
        """Plot object dimensions"""
        plt.title('Object Dimensions', fontsize=14, fontweight='bold')
        
        valid_poses = [p for p in poses if p is not None]
        if not valid_poses:
            plt.text(0.5, 0.5, 'No valid poses', ha='center', va='center', transform=plt.gca().transAxes)
            return
        
        # Extract dimensions
        dimensions = np.array([p['dimensions'] for p in valid_poses])
        class_names = [p['class_name'] for p in valid_poses]
        
        # Plot dimensions
        x = np.arange(len(class_names))
        width = 0.25
        
        plt.bar(x - width, dimensions[:, 0], width, label='Width', alpha=0.7)
        plt.bar(x, dimensions[:, 1], width, label='Height', alpha=0.7)
        plt.bar(x + width, dimensions[:, 2], width, label='Depth', alpha=0.7)
        
        plt.xlabel('Objects')
        plt.ylabel('Dimension (m)')
        plt.xticks(x, [name[:10] for name in class_names], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    def _plot_pipeline_performance(self, detections, poses, pcd_enhanced):
        """Plot pipeline performance metrics"""
        plt.title('Pipeline Performance', fontsize=14, fontweight='bold')
        
        metrics = {
            'Objects Detected': len(detections),
            'Poses Estimated': len([p for p in poses if p is not None]),
            'Point Cloud Size': len(pcd_enhanced.points) if pcd_enhanced else 0,
            'Avg Confidence': np.mean([d['confidence'] for d in detections]) if detections else 0
        }
        
        x = list(metrics.keys())
        y = list(metrics.values())
        
        bars = plt.bar(x, y, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, y):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{value:.1f}', ha='center', va='bottom')
    
    def _plot_project_summary(self, detections, poses, pcd_enhanced):
        """Plot project summary"""
        plt.title('Integrated 6D Pose Pipeline', fontsize=16, fontweight='bold')
        
        summary_text = f"""
        🎯 Pipeline Results
        
        📦 Objects Detected: {len(detections)}
        📐 Poses Estimated: {len([p for p in poses if p is not None])}
        ☁️ Point Cloud: {len(pcd_enhanced.points) if pcd_enhanced else 0} points
        
        🔧 Technologies Used:
        • YOLOv8s Instance Segmentation
        • PCA-based 6D Pose Estimation
        • Object-aware Point Cloud Processing
        • Multi-modal Data Fusion
        
        🎓 MSc Project: AI-driven 2D-to-3D Conversion
        """
        
        plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        plt.axis('off')
    
    def _save_detailed_results(self, detections, poses, pcd_enhanced, output_path):
        """Save detailed results to files"""
        # Save 6D pose information
        pose_data = []
        for detection, pose in zip(detections, poses):
            pose_info = {
                'class_name': detection['class_name'],
                'confidence': detection['confidence'],
                'bbox_2d': detection['bbox'],
                'mask_area': detection['mask_area']
            }
            
            if pose:
                pose_info.update({
                    'translation': pose['translation'],
                    'rotation_euler': pose['rotation_euler'],
                    'dimensions': pose['dimensions'],
                    'points_count': pose['points_count']
                })
            else:
                pose_info.update({
                    'translation': None,
                    'rotation_euler': None,
                    'dimensions': None,
                    'points_count': 0
                })
            
            pose_data.append(pose_info)
        
        # Save as JSON
        with open(output_path / "6d_pose_results.json", 'w') as f:
            json.dump(pose_data, f, indent=2)
        
        # Save as text summary
        with open(output_path / "pipeline_summary.txt", 'w') as f:
            f.write("Integrated 6D Pose Pipeline Results\n")
            f.write("===================================\n\n")
            f.write(f"Total Objects: {len(detections)}\n")
            f.write(f"Successful Poses: {len([p for p in poses if p is not None])}\n")
            f.write(f"Point Cloud Size: {len(pcd_enhanced.points) if pcd_enhanced else 0}\n\n")
            
            for i, (det, pose) in enumerate(zip(detections, poses)):
                f.write(f"Object {i+1}: {det['class_name']}\n")
                f.write(f"  Confidence: {det['confidence']:.3f}\n")
                f.write(f"  BBox: {det['bbox']}\n")
                f.write(f"  Mask Area: {det['mask_area']}\n")
                
                if pose:
                    f.write(f"  Position: {pose['translation']}\n")
                    f.write(f"  Rotation: {pose['rotation_euler']}\n")
                    f.write(f"  Dimensions: {pose['dimensions']}\n")
                    f.write(f"  3D Points: {pose['points_count']}\n")
                else:
                    f.write(f"  Pose: Failed to estimate\n")
                f.write("\n")
    
    def run_complete_pipeline(self, rgb_path, depth_path, output_dir="integrated_6d_pipeline_results"):
        """Run the complete integrated 6D pose pipeline"""
        print("🚀 Starting Integrated 6D Pose Recognition Pipeline")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # 1. Load RGB-D data
            rgb, depth = self.load_rgb_depth_data(rgb_path, depth_path)
            
            # 2. Instance segmentation
            detections = self.detect_objects_with_masks(rgb)
            
            # 3. 6D pose estimation
            poses = []
            for detection in detections:
                pose = self.estimate_6d_pose_from_mask(rgb, depth, detection)
                poses.append(pose)
            
            # 4. Enhanced point cloud generation
            pcd_enhanced = self.create_enhanced_point_cloud(rgb, depth, detections, poses)
            
            # 5. Comprehensive visualization
            output_path = self.create_comprehensive_visualization(
                rgb, depth, detections, poses, pcd_enhanced, output_dir
            )
            
            # 6. Performance summary
            total_time = time.time() - start_time
            print(f"\n⏱️ Pipeline completed in {total_time:.2f} seconds")
            
            # 7. Open interactive viewer
            if pcd_enhanced and len(pcd_enhanced.points) > 0:
                print(f"\n🎮 Opening interactive 3D viewer...")
                o3d.visualization.draw_geometries([pcd_enhanced], 
                                                window_name="Integrated 6D Pose Pipeline - Enhanced 3D Scene")
            
            print(f"\n🎉 Integrated 6D Pose Pipeline completed successfully!")
            print(f"📁 Results saved to: {output_path}")
            print(f"📄 Files created:")
            print(f"   - enhanced_pointcloud_6d.ply (Enhanced 3D scene)")
            print(f"   - integrated_6d_pipeline_results.png (Comprehensive visualization)")
            print(f"   - 6d_pose_results.json (Detailed pose data)")
            print(f"   - pipeline_summary.txt (Text summary)")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Pipeline failed: {str(e)}")
            raise

def main():
    """Main function to run the integrated pipeline"""
    print("🎯 Integrated 6D Pose Recognition Pipeline")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = Integrated6DPosePipeline()
    
    # Use your CoppeliaSim data
    rgb_path = "new_captures/all_captures/processed_images/capture_1_rgb.jpg"
    depth_path = "new_captures/all_captures/processed_images/capture_1_depth.npy"
    
    # Check if files exist
    if not Path(rgb_path).exists():
        print(f"❌ RGB file not found: {rgb_path}")
        return
    
    if not Path(depth_path).exists():
        print(f"❌ Depth file not found: {depth_path}")
        return
    
    # Run complete pipeline
    try:
        output_dir = pipeline.run_complete_pipeline(rgb_path, depth_path)
        print(f"\n✅ Success! Integrated 6D pose pipeline completed.")
        print(f"🎓 This demonstrates your complete MSc project pipeline!")
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")

if __name__ == "__main__":
    main()




