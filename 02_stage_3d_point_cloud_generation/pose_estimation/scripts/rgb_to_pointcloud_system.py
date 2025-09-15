#!/usr/bin/env python3
"""
🚀 State-of-the-Art RGB to Point Cloud Conversion System
========================================================
Converts RGB images to 3D point clouds for advanced 3D scene understanding.
Integrates with existing 6D pose estimation pipeline.

Author: AI Assistant
Date: 2024
"""

import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from ultralytics import YOLO
import json
import os
import time
from pathlib import Path

class RGBToPointCloudSystem:
    """
    Advanced RGB to Point Cloud conversion system with 3D object analysis.
    """
    
    def __init__(self):
        """Initialize the RGB to Point Cloud system."""
        print("🚀 Initializing RGB to Point Cloud System...")
        
        # Load existing components
        self.model = YOLO('../../coppelia_sim_results/weights/best.pt')
        self.camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
        self.dist_coeffs = np.zeros(4)
        
        # Point cloud processing parameters
        self.voxel_size = 0.01  # 1cm voxel size for downsampling
        self.depth_scale = 1000.0  # Depth scaling factor
        self.max_depth = 2.0  # Maximum depth in meters
        
        print("✅ RGB to Point Cloud System initialized!")
    
    def rgb_depth_to_pointcloud(self, rgb_image, depth_image):
        """
        Convert RGB + Depth to colored point cloud using Open3D.
        
        Args:
            rgb_image: RGB image (H, W, 3)
            depth_image: Depth image (H, W) in meters
            
        Returns:
            open3d.geometry.PointCloud: Colored point cloud
        """
        print("🔄 Converting RGB + Depth to Point Cloud...")
        
        # Ensure depth is in correct format
        if depth_image.dtype != np.float32:
            depth_image = depth_image.astype(np.float32)
        
        # Create Open3D RGBD image
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb_image),
            o3d.geometry.Image(depth_image),
            depth_scale=1.0,  # Depth is already in meters
            depth_trunc=self.max_depth,
            convert_rgb_to_intensity=False
        )
        
        # Create point cloud from RGBD
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.intrinsic_matrix = self.camera_matrix
        
        pointcloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        
        # Remove invalid points
        pointcloud = pointcloud.remove_non_finite_points()
        
        print(f"✅ Point cloud created with {len(pointcloud.points)} points")
        return pointcloud
    
    def process_pointcloud(self, pointcloud):
        """
        Process point cloud for better quality and analysis.
        
        Args:
            pointcloud: Input point cloud
            
        Returns:
            open3d.geometry.PointCloud: Processed point cloud
        """
        print("🔄 Processing point cloud...")
        
        # Downsample for efficiency
        pointcloud = pointcloud.voxel_down_sample(voxel_size=self.voxel_size)
        
        # Remove statistical outliers
        pointcloud, _ = pointcloud.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0
        )
        
        # Estimate normals for better visualization
        pointcloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1, max_nn=30
            )
        )
        
        print(f"✅ Processed point cloud: {len(pointcloud.points)} points")
        return pointcloud
    
    def segment_objects_in_pointcloud(self, pointcloud, detections):
        """
        Segment objects in point cloud based on YOLO detections.
        
        Args:
            pointcloud: Input point cloud
            detections: YOLO detection results
            
        Returns:
            dict: Segmented point clouds for each object
        """
        print("🔄 Segmenting objects in point cloud...")
        
        segmented_objects = {}
        points = np.asarray(pointcloud.points)
        colors = np.asarray(pointcloud.colors)
        
        for detection in detections:
            object_name = detection['name']
            bbox = detection['bbox']  # [x1, y1, x2, y2]
            
            # Convert 2D bbox to 3D point cloud region
            x1, y1, x2, y2 = bbox
            x1, x2 = int(x1), int(x2)
            y1, y2 = int(y1), int(y2)
            
            # Create mask for this object
            mask = np.zeros((points.shape[0],), dtype=bool)
            
            # Find points within the bbox region
            for i, point in enumerate(points):
                # Project 3D point to 2D image coordinates
                if point[2] > 0:  # Valid depth
                    u = int(self.camera_matrix[0, 0] * point[0] / point[2] + self.camera_matrix[0, 2])
                    v = int(self.camera_matrix[1, 1] * point[1] / point[2] + self.camera_matrix[1, 2])
                    
                    if x1 <= u <= x2 and y1 <= v <= y2:
                        mask[i] = True
            
            # Extract object point cloud
            if np.any(mask):
                object_points = points[mask]
                object_colors = colors[mask]
                
                object_pc = o3d.geometry.PointCloud()
                object_pc.points = o3d.utility.Vector3dVector(object_points)
                object_pc.colors = o3d.utility.Vector3dVector(object_colors)
                
                segmented_objects[object_name] = object_pc
                print(f"  📦 {object_name}: {len(object_points)} points")
        
        return segmented_objects
    
    def extract_3d_bounding_boxes(self, segmented_objects):
        """
        Extract 3D bounding boxes for segmented objects.
        
        Args:
            segmented_objects: Dict of segmented point clouds
            
        Returns:
            dict: 3D bounding boxes for each object
        """
        print("🔄 Extracting 3D bounding boxes...")
        
        bboxes_3d = {}
        
        for object_name, pointcloud in segmented_objects.items():
            if len(pointcloud.points) > 10:  # Minimum points for bbox
                # Get bounding box
                bbox = pointcloud.get_axis_aligned_bounding_box()
                
                # Get center and extent
                center = bbox.get_center()
                extent = bbox.get_extent()
                
                bboxes_3d[object_name] = {
                    'center': center,
                    'extent': extent,
                    'bbox': bbox
                }
                
                print(f"  📦 {object_name}: center={center}, size={extent}")
        
        return bboxes_3d
    
    def visualize_pointcloud_analysis(self, pointcloud, segmented_objects, bboxes_3d, save_path):
        """
        Create comprehensive visualization of point cloud analysis.
        
        Args:
            pointcloud: Full point cloud
            segmented_objects: Segmented object point clouds
            bboxes_3d: 3D bounding boxes
            save_path: Path to save visualization
        """
        print("🔄 Creating point cloud visualization...")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Full point cloud (top view)
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        points = np.asarray(pointcloud.points)
        colors = np.asarray(pointcloud.colors)
        
        # Sample points for visualization (too many points slow down plotting)
        if len(points) > 5000:
            indices = np.random.choice(len(points), 5000, replace=False)
            points_viz = points[indices]
            colors_viz = colors[indices]
        else:
            points_viz = points
            colors_viz = colors
        
        ax1.scatter(points_viz[:, 0], points_viz[:, 1], points_viz[:, 2], 
                   c=colors_viz, s=1, alpha=0.6)
        ax1.set_title('Full Point Cloud (Top View)')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        
        # 2. Individual object point clouds
        for i, (object_name, object_pc) in enumerate(segmented_objects.items()):
            if i >= 4:  # Max 4 objects for visualization
                break
                
            ax = fig.add_subplot(2, 3, i + 2, projection='3d')
            obj_points = np.asarray(object_pc.points)
            obj_colors = np.asarray(object_pc.colors)
            
            if len(obj_points) > 1000:
                indices = np.random.choice(len(obj_points), 1000, replace=False)
                obj_points_viz = obj_points[indices]
                obj_colors_viz = obj_colors[indices]
            else:
                obj_points_viz = obj_points
                obj_colors_viz = obj_colors
            
            ax.scatter(obj_points_viz[:, 0], obj_points_viz[:, 1], obj_points_viz[:, 2], 
                      c=obj_colors_viz, s=2, alpha=0.8)
            
            # Add 3D bounding box if available
            if object_name in bboxes_3d:
                bbox = bboxes_3d[object_name]['bbox']
                bbox_points = np.asarray(bbox.get_box_points())
                bbox_lines = [[0, 1], [1, 2], [2, 3], [3, 0],
                             [4, 5], [5, 6], [6, 7], [7, 4],
                             [0, 4], [1, 5], [2, 6], [3, 7]]
                
                for line in bbox_lines:
                    ax.plot([bbox_points[line[0], 0], bbox_points[line[1], 0]],
                           [bbox_points[line[0], 1], bbox_points[line[1], 1]],
                           [bbox_points[line[0], 2], bbox_points[line[1], 2]], 'r-', linewidth=2)
            
            ax.set_title(f'{object_name.replace("_", " ").title()}')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
        
        # 6. Summary statistics
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('off')
        
        stats_text = f"""
Point Cloud Analysis Summary:
============================
Total Points: {len(pointcloud.points):,}
Objects Detected: {len(segmented_objects)}
Voxel Size: {self.voxel_size*1000:.1f}mm
Max Depth: {self.max_depth}m

Object Statistics:
"""
        
        for object_name, object_pc in segmented_objects.items():
            stats_text += f"• {object_name}: {len(object_pc.points):,} points\n"
        
        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Point cloud visualization saved to: {save_path}")
        
        return fig
    
    def process_image_pair(self, rgb_file, depth_file, output_dir="../results"):
        """
        Complete pipeline: RGB + Depth → Point Cloud → 3D Analysis.
        
        Args:
            rgb_file: Path to RGB image file
            depth_file: Path to depth image file
            output_dir: Output directory for results
        """
        print(f"\n🚀 Processing RGB→Point Cloud: {os.path.basename(rgb_file)}")
        
        # Load images
        rgb_image = self.load_rgb_image(rgb_file)
        depth_image = self.load_depth_image(depth_file)
        
        if rgb_image is None or depth_image is None:
            print("❌ Failed to load images")
            return None
        
        # Convert to point cloud
        pointcloud = self.rgb_depth_to_pointcloud(rgb_image, depth_image)
        pointcloud = self.process_pointcloud(pointcloud)
        
        # Detect objects in RGB image
        detections = self.detect_objects(rgb_image)
        
        # Segment objects in point cloud
        segmented_objects = self.segment_objects_in_pointcloud(pointcloud, detections)
        
        # Extract 3D bounding boxes
        bboxes_3d = self.extract_3d_bounding_boxes(segmented_objects)
        
        # Create visualization
        timestamp = int(time.time())
        save_path = f"{output_dir}/pointcloud_analysis_{timestamp}.jpg"
        self.visualize_pointcloud_analysis(pointcloud, segmented_objects, bboxes_3d, save_path)
        
        # Save point cloud data
        pcd_save_path = f"{output_dir}/pointcloud_{timestamp}.ply"
        o3d.io.write_point_cloud(pcd_save_path, pointcloud)
        print(f"✅ Point cloud saved to: {pcd_save_path}")
        
        return {
            'pointcloud': pointcloud,
            'segmented_objects': segmented_objects,
            'bboxes_3d': bboxes_3d,
            'detections': detections,
            'visualization_path': save_path,
            'pointcloud_path': pcd_save_path
        }
    
    def load_rgb_image(self, rgb_file):
        """Load RGB image from file."""
        try:
            if rgb_file.endswith('.txt'):
                # Load from text file (your existing format)
                with open(rgb_file, 'rb') as f:
                    rgb_data = np.frombuffer(f.read(), dtype=np.uint8)
                rgb_data = rgb_data.reshape(480, 640, 3)
                return rgb_data
            else:
                # Load from image file
                return cv2.imread(rgb_file)
        except Exception as e:
            print(f"❌ Error loading RGB image: {e}")
            return None
    
    def load_depth_image(self, depth_file):
        """Load depth image from file."""
        try:
            if depth_file.endswith('.txt'):
                # Load from text file (your existing format)
                with open(depth_file, 'r') as f:
                    content = f.read().strip()
                # Parse comma-separated values
                depth_values = [float(x) for x in content.split(',') if x.strip()]
                depth_data = np.array(depth_values, dtype=np.float32)
                depth_data = depth_data.reshape(480, 640)
                return depth_data
            else:
                # Load from image file
                depth_data = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
                return depth_data.astype(np.float32) / 1000.0  # Convert to meters
        except Exception as e:
            print(f"❌ Error loading depth image: {e}")
            return None
    
    def detect_objects(self, rgb_image):
        """Detect objects using YOLO model."""
        results = self.model(rgb_image, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    name = result.names[cls]
                    
                    detections.append({
                        'name': name,
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class_id': cls
                    })
        
        return detections

def main():
    """Main function to test the RGB to Point Cloud system."""
    print("🚀 RGB to Point Cloud System - Test Mode")
    print("=" * 50)
    
    # Initialize system
    system = RGBToPointCloudSystem()
    
    # Test with latest captured images
    print("\n🎯 Test Options:")
    print("1. Test with latest captured images")
    print("2. Test with specific image pair")
    print("3. Batch process all recent images")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        # Find latest images
        rgb_files = sorted(glob.glob("/tmp/auto_kinect_*_rgb.txt"))
        depth_files = sorted(glob.glob("/tmp/auto_kinect_*_depth.txt"))
        
        if rgb_files and depth_files:
            latest_rgb = rgb_files[-1]
            latest_depth = depth_files[-1]
            print(f"📸 Found latest images:")
            print(f"  RGB: {latest_rgb}")
            print(f"  Depth: {latest_depth}")
            
            result = system.process_image_pair(latest_rgb, latest_depth)
            if result:
                print("✅ RGB→Point Cloud conversion completed successfully!")
        else:
            print("❌ No captured images found in /tmp/")
    
    elif choice == "2":
        rgb_file = input("Enter RGB file path: ").strip()
        depth_file = input("Enter depth file path: ").strip()
        
        result = system.process_image_pair(rgb_file, depth_file)
        if result:
            print("✅ RGB→Point Cloud conversion completed successfully!")
    
    elif choice == "3":
        # Process all recent images
        rgb_files = sorted(glob.glob("/tmp/auto_kinect_*_rgb.txt"))
        depth_files = sorted(glob.glob("/tmp/auto_kinect_*_depth.txt"))
        
        print(f"🔄 Processing {len(rgb_files)} image pairs...")
        
        for i, (rgb_file, depth_file) in enumerate(zip(rgb_files, depth_files)):
            print(f"\n📸 Processing pair {i+1}/{len(rgb_files)}: {os.path.basename(rgb_file)}")
            result = system.process_image_pair(rgb_file, depth_file)
            
            if result:
                print(f"✅ Pair {i+1} completed successfully!")
            else:
                print(f"❌ Pair {i+1} failed!")
        
        print(f"\n🎉 Batch processing completed! {len(rgb_files)} pairs processed.")
    
    else:
        print("❌ Invalid choice!")

if __name__ == "__main__":
    import glob
    main()
