#!/usr/bin/env python3
"""
MCC-Inspired 3D Reconstruction Pipeline
Combines YOLOv8s detection with enhanced 3D reconstruction
"""

import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from ultralytics import YOLO
import time
from scipy.spatial.transform import Rotation
import json

class MCCInspired3DReconstruction:
    def __init__(self, model_path="training_results/yolov8s_instance_segmentation/weights/best.pt"):
        """Initialize the MCC-inspired 3D reconstruction pipeline"""
        print("🚀 Initializing MCC-Inspired 3D Reconstruction Pipeline")
        
        # Load trained YOLO model
        self.model = YOLO(model_path)
        print(f"✅ YOLO model loaded: {model_path}")
        
        # 3D reconstruction parameters
        self.voxel_size = 0.01  # 1cm voxels
        self.depth_scale = 1000.0  # Depth scale factor
        self.min_points = 100  # Minimum points for reconstruction
        
        print("🎯 Pipeline ready for 3D reconstruction!")
    
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
            # Try to reshape, if it fails, use original shape
            try:
                depth_img = depth_data.reshape(height, width)
            except ValueError:
                print(f"⚠️ Depth reshape failed, using original shape")
                depth_img = depth_data
        else:
            depth_img = depth_data.reshape(height, width)
        
        print(f"✅ RGB: {rgb_img.shape}, Depth: {depth_img.shape}")
        return rgb_img, depth_img
    
    def detect_objects(self, rgb_img):
        """Detect objects using trained YOLOv8s model"""
        print("\n🔍 Running object detection...")
        
        # Run inference
        results = self.model(rgb_img, conf=0.3, verbose=False)
        
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
        
        # Apply MCC-inspired enhancements
        pcd = self._apply_mcc_enhancements(pcd, detections)
        
        print(f"✅ Point cloud created with {len(pcd.points)} points")
        return pcd
    
    def _apply_mcc_enhancements(self, pcd, detections):
        """Apply MCC-inspired enhancements to point cloud"""
        print("   🔧 Applying MCC-inspired enhancements...")
        
        # 1. Remove statistical outliers
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        # 2. Downsample for efficiency
        pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        
        # 3. Estimate normals for better surface reconstruction
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30
        ))
        
        # 4. Apply object-aware filtering using detections
        if detections:
            pcd = self._filter_by_object_masks(pcd, detections)
        
        # 5. Remove ground plane (common in robotics scenes)
        pcd = self._remove_ground_plane(pcd)
        
        print(f"   ✅ Enhanced point cloud: {len(pcd.points)} points")
        return pcd
    
    def _filter_by_object_masks(self, pcd, detections):
        """Filter point cloud based on detected object masks"""
        print("   🎯 Applying object-aware filtering...")
        
        # Get points and colors
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        # Create mask for valid object regions
        valid_mask = np.zeros(len(points), dtype=bool)
        
        for detection in detections:
            mask = detection['mask']
            bbox = detection['bbox']
            
            # Resize mask to match image dimensions
            mask_resized = cv2.resize(mask.astype(np.uint8), 
                                    (int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])))
            
            # Apply mask to point cloud
            # This is a simplified approach - in practice, you'd project 3D points to 2D
            # For now, we'll use bounding box filtering
            x1, y1, x2, y2 = bbox
            
            # Simple 2D projection (approximate)
            # In a real implementation, you'd use proper camera projection
            valid_points = (
                (points[:, 0] >= x1/100) & (points[:, 0] <= x2/100) &
                (points[:, 1] >= y1/100) & (points[:, 1] <= y2/100)
            )
            valid_mask |= valid_points
        
        # Apply filter
        if np.any(valid_mask):
            pcd_filtered = o3d.geometry.PointCloud()
            pcd_filtered.points = o3d.utility.Vector3dVector(points[valid_mask])
            pcd_filtered.colors = o3d.utility.Vector3dVector(colors[valid_mask])
            return pcd_filtered
        
        return pcd
    
    def _remove_ground_plane(self, pcd):
        """Remove ground plane using RANSAC"""
        print("   🏠 Removing ground plane...")
        
        # Find ground plane
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.01, ransac_n=3, num_iterations=1000
        )
        
        if len(inliers) > len(pcd.points) * 0.3:  # If large plane found
            # Remove ground plane points
            pcd = pcd.select_by_index(inliers, invert=True)
            print(f"   ✅ Ground plane removed, remaining: {len(pcd.points)} points")
        
        return pcd
    
    def reconstruct_3d_objects(self, pcd, detections):
        """Reconstruct 3D objects from point cloud"""
        print("\n🏗️ Reconstructing 3D objects...")
        
        objects_3d = []
        
        for i, detection in enumerate(detections):
            print(f"   🔍 Processing {detection['class_name']}...")
            
            # Extract object region from point cloud
            # This is a simplified approach - in practice, you'd use proper segmentation
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Create bounding box filter (approximate 3D region)
            points = np.asarray(pcd.points)
            valid_points = (
                (points[:, 0] >= x1/100) & (points[:, 0] <= x2/100) &
                (points[:, 1] >= y1/100) & (points[:, 1] <= y2/100)
            )
            
            if np.any(valid_points):
                # Create object point cloud
                obj_pcd = o3d.geometry.PointCloud()
                obj_pcd.points = o3d.utility.Vector3dVector(points[valid_points])
                
                # Estimate object properties
                obj_info = {
                    'class_name': detection['class_name'],
                    'confidence': detection['confidence'],
                    'point_count': len(obj_pcd.points),
                    'bbox_3d': self._estimate_3d_bbox(obj_pcd),
                    'center': np.mean(points[valid_points], axis=0),
                    'point_cloud': obj_pcd
                }
                
                objects_3d.append(obj_info)
                print(f"      ✅ {detection['class_name']}: {len(obj_pcd.points)} points")
        
        print(f"✅ Reconstructed {len(objects_3d)} 3D objects")
        return objects_3d
    
    def _estimate_3d_bbox(self, pcd):
        """Estimate 3D bounding box for object"""
        if len(pcd.points) == 0:
            return None
        
        points = np.asarray(pcd.points)
        
        # Get min/max bounds
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)
        
        # Create bounding box
        bbox = o3d.geometry.OrientedBoundingBox.create_from_points(pcd.points)
        
        return {
            'min': min_bound.tolist(),
            'max': max_bound.tolist(),
            'center': bbox.center.tolist(),
            'extent': bbox.extent.tolist(),
            'orientation': bbox.R.tolist()
        }
    
    def visualize_results(self, rgb_img, depth_img, pcd, objects_3d, output_path="mcc_3d_reconstruction_results"):
        """Visualize all results"""
        print(f"\n🎨 Creating visualization: {output_path}")
        
        # Create output directory
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)
        
        # 1. Save RGB and Depth images
        cv2.imwrite(str(output_dir / "rgb_input.jpg"), rgb_img)
        
        # Normalize depth for visualization
        depth_viz = ((depth_img - depth_img.min()) / (depth_img.max() - depth_img.min()) * 255).astype(np.uint8)
        cv2.imwrite(str(output_dir / "depth_input.jpg"), depth_viz)
        
        # 2. Save point cloud
        o3d.io.write_point_cloud(str(output_dir / "enhanced_pointcloud.ply"), pcd)
        
        # 3. Create comprehensive visualization
        self._create_comprehensive_visualization(rgb_img, depth_img, pcd, objects_3d, output_dir)
        
        # 4. Save object information
        self._save_object_info(objects_3d, output_dir)
        
        print(f"✅ Results saved to: {output_dir}")
        return output_dir
    
    def _create_comprehensive_visualization(self, rgb_img, depth_img, pcd, objects_3d, output_dir):
        """Create comprehensive visualization of results"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # RGB image
        axes[0, 0].imshow(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("RGB Input", fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Depth image
        axes[0, 1].imshow(depth_img, cmap='plasma')
        axes[0, 1].set_title("Depth Input", fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Point cloud (top view)
        if len(pcd.points) > 0:
            points = np.asarray(pcd.points)
            axes[0, 2].scatter(points[:, 0], points[:, 1], c=points[:, 2], s=1, cmap='viridis')
            axes[0, 2].set_title("Point Cloud (Top View)", fontsize=14, fontweight='bold')
            axes[0, 2].set_xlabel("X")
            axes[0, 2].set_ylabel("Y")
        
        # Object detection overlay
        rgb_with_boxes = rgb_img.copy()
        for obj in objects_3d:
            # Draw 3D bounding box info
            center = obj['center']
            axes[1, 0].text(0.1, 0.9 - len(objects_3d) * 0.1, 
                           f"{obj['class_name']}: {obj['point_count']} pts", 
                           transform=axes[1, 0].transAxes, fontsize=12,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))
        
        axes[1, 0].imshow(cv2.cvtColor(rgb_with_boxes, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title("Object Detection Results", fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Point cloud statistics
        if len(pcd.points) > 0:
            points = np.asarray(pcd.points)
            axes[1, 1].hist(points[:, 2], bins=50, alpha=0.7, color='skyblue')
            axes[1, 1].set_title("Depth Distribution", fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel("Depth (m)")
            axes[1, 1].set_ylabel("Point Count")
        
        # 3D reconstruction summary
        axes[1, 2].text(0.1, 0.9, f"Total Points: {len(pcd.points)}", 
                        transform=axes[1, 2].transAxes, fontsize=14, fontweight='bold')
        axes[1, 2].text(0.1, 0.8, f"Objects Detected: {len(objects_3d)}", 
                        transform=axes[1, 2].transAxes, fontsize=14, fontweight='bold')
        axes[1, 2].text(0.1, 0.7, f"Voxel Size: {self.voxel_size}m", 
                        transform=axes[1, 2].transAxes, fontsize=14, fontweight='bold')
        axes[1, 2].set_title("3D Reconstruction Summary", fontsize=14, fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(str(output_dir / "comprehensive_visualization.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_object_info(self, objects_3d, output_dir):
        """Save detailed object information"""
        object_data = []
        
        for obj in objects_3d:
            obj_info = {
                'class_name': obj['class_name'],
                'confidence': obj['confidence'],
                'point_count': obj['point_count'],
                'center': obj['center'].tolist(),
                'bbox_3d': obj['bbox_3d']
            }
            object_data.append(obj_info)
        
        # Save as JSON
        with open(output_dir / "objects_3d_info.json", 'w') as f:
            json.dump(object_data, f, indent=2)
        
        # Save as text summary
        with open(output_dir / "objects_3d_summary.txt", 'w') as f:
            f.write("MCC-Inspired 3D Reconstruction Results\n")
            f.write("=" * 50 + "\n\n")
            
            for i, obj in enumerate(objects_3d):
                f.write(f"Object {i+1}:\n")
                f.write(f"  Class: {obj['class_name']}\n")
                f.write(f"  Confidence: {obj['confidence']:.3f}\n")
                f.write(f"  Point Count: {obj['point_count']}\n")
                f.write(f"  Center: [{obj['center'][0]:.3f}, {obj['center'][1]:.3f}, {obj['center'][2]:.3f}]\n")
                f.write(f"  3D BBox: {obj['bbox_3d']}\n\n")
    
    def run_full_pipeline(self, rgb_path, depth_path, output_path="mcc_3d_reconstruction_results"):
        """Run the complete MCC-inspired 3D reconstruction pipeline"""
        print("🚀 Starting MCC-Inspired 3D Reconstruction Pipeline")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # 1. Load data
            rgb_img, depth_img = self.load_rgb_depth_data(rgb_path, depth_path)
            
            # 2. Detect objects
            detections = self.detect_objects(rgb_img)
            
            # 3. Create enhanced point cloud
            pcd = self.create_enhanced_point_cloud(rgb_img, depth_img, detections)
            
            # 4. Reconstruct 3D objects
            objects_3d = self.reconstruct_3d_objects(pcd, detections)
            
            # 5. Visualize results
            output_dir = self.visualize_results(rgb_img, depth_img, pcd, objects_3d, output_path)
            
            # 6. Performance metrics
            total_time = time.time() - start_time
            print(f"\n⏱️ Pipeline completed in {total_time:.2f} seconds")
            print(f"📊 Performance: {len(pcd.points)/total_time:.0f} points/second")
            
            return output_dir, objects_3d, pcd
            
        except Exception as e:
            print(f"❌ Pipeline failed: {str(e)}")
            raise

def main():
    """Main function to demonstrate the pipeline"""
    print("🎯 MCC-Inspired 3D Reconstruction Demo")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = MCCInspired3DReconstruction()
    
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
        output_dir, objects_3d, pcd = pipeline.run_full_pipeline(rgb_path, depth_path)
        print(f"\n🎉 Success! Check results in: {output_dir}")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")

if __name__ == "__main__":
    main()
