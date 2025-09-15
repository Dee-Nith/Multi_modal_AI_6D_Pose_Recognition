#!/usr/bin/env python3
"""
🚀 Enhanced RGB to Point Cloud System
====================================
Generates dense, high-quality colored point clouds with maximum detail.
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

class EnhancedPointCloudSystem:
    """
    Enhanced RGB to Point Cloud system for dense, high-quality 3D reconstruction.
    """
    
    def __init__(self):
        """Initialize the enhanced point cloud system."""
        print("🚀 Initializing Enhanced Point Cloud System...")
        
        # Load existing components
        self.model = YOLO('../../coppelia_sim_results/weights/best.pt')
        self.camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
        self.dist_coeffs = np.zeros(4)
        
        # Enhanced point cloud parameters for maximum density
        self.voxel_size = 0.005  # 5mm voxel size (much smaller for more detail)
        self.depth_scale = 1000.0
        self.max_depth = 3.0  # Increased depth range
        self.min_depth = 0.01  # Minimum depth threshold
        
        # Enhanced processing parameters
        self.downsample_factor = 1  # No downsampling for maximum density
        self.outlier_removal = False  # Keep all points initially
        
        print("✅ Enhanced Point Cloud System initialized!")
    
    def load_rgb_image(self, rgb_file):
        """Load RGB image from file."""
        try:
            if rgb_file.endswith('.txt'):
                with open(rgb_file, 'rb') as f:
                    rgb_data = np.frombuffer(f.read(), dtype=np.uint8)
                rgb_data = rgb_data.reshape(480, 640, 3)
                return rgb_data
            else:
                return cv2.imread(rgb_file)
        except Exception as e:
            print(f"❌ Error loading RGB image: {e}")
            return None
    
    def load_depth_image(self, depth_file):
        """Load depth image from file."""
        try:
            if depth_file.endswith('.txt'):
                with open(depth_file, 'r') as f:
                    content = f.read().strip()
                depth_values = [float(x) for x in content.split(',') if x.strip()]
                depth_data = np.array(depth_values, dtype=np.float32)
                depth_data = depth_data.reshape(480, 640)
                return depth_data
            else:
                depth_data = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
                return depth_data.astype(np.float32) / 1000.0
        except Exception as e:
            print(f"❌ Error loading depth image: {e}")
            return None
    
    def enhance_depth_data(self, depth_image):
        """
        Enhance depth data for better point cloud generation.
        """
        print("🔄 Enhancing depth data...")
        
        # Filter out invalid depths
        valid_mask = (depth_image > self.min_depth) & (depth_image < self.max_depth)
        
        # Apply bilateral filter to reduce noise while preserving edges
        depth_filtered = cv2.bilateralFilter(
            depth_image.astype(np.float32), 
            d=15, sigmaColor=0.1, sigmaSpace=15
        )
        
        # Apply median filter to remove outliers
        depth_filtered = cv2.medianBlur(depth_filtered.astype(np.uint16), 5).astype(np.float32)
        
        # Apply morphological operations to fill small holes
        kernel = np.ones((3,3), np.uint8)
        depth_filtered = cv2.morphologyEx(depth_filtered, cv2.MORPH_CLOSE, kernel)
        
        # Combine with original valid mask
        depth_enhanced = np.where(valid_mask, depth_filtered, depth_image)
        
        print(f"✅ Depth enhancement completed")
        return depth_enhanced
    
    def create_dense_pointcloud(self, rgb_image, depth_image):
        """
        Create dense, high-quality point cloud with maximum detail.
        """
        print("🔄 Creating dense point cloud...")
        
        # Enhance depth data
        depth_enhanced = self.enhance_depth_data(depth_image)
        
        # Create dense point cloud manually for maximum control
        height, width = depth_enhanced.shape
        points = []
        colors = []
        
        # Generate points for every pixel (no downsampling)
        for v in range(height):
            for u in range(width):
                depth = depth_enhanced[v, u]
                
                # Skip invalid depths
                if depth <= self.min_depth or depth >= self.max_depth:
                    continue
                
                # Convert to 3D coordinates
                x = (u - self.camera_matrix[0, 2]) * depth / self.camera_matrix[0, 0]
                y = (v - self.camera_matrix[1, 2]) * depth / self.camera_matrix[1, 1]
                z = depth
                
                # Get color
                color = rgb_image[v, u] / 255.0  # Normalize to [0,1]
                
                points.append([x, y, z])
                colors.append(color)
        
        # Convert to numpy arrays
        points = np.array(points, dtype=np.float32)
        colors = np.array(colors, dtype=np.float32)
        
        print(f"✅ Generated {len(points):,} dense points")
        
        # Create Open3D point cloud
        pointcloud = o3d.geometry.PointCloud()
        pointcloud.points = o3d.utility.Vector3dVector(points)
        pointcloud.colors = o3d.utility.Vector3dVector(colors)
        
        return pointcloud
    
    def process_dense_pointcloud(self, pointcloud):
        """
        Process dense point cloud while preserving maximum detail.
        """
        print("🔄 Processing dense point cloud...")
        
        # Remove non-finite points
        pointcloud = pointcloud.remove_non_finite_points()
        
        # Very light downsampling only if too many points
        if len(pointcloud.points) > 100000:
            print(f"📊 Downsampling from {len(pointcloud.points):,} to ~50,000 points for performance")
            pointcloud = pointcloud.voxel_down_sample(voxel_size=self.voxel_size)
        
        # Estimate normals for better visualization
        pointcloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.05, max_nn=30
            )
        )
        
        print(f"✅ Processed point cloud: {len(pointcloud.points):,} points")
        return pointcloud
    
    def segment_objects_dense(self, pointcloud, detections):
        """
        Segment objects in dense point cloud with high precision.
        """
        print("🔄 Segmenting objects in dense point cloud...")
        
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
            
            # Find points within the bbox region with higher precision
            for i, point in enumerate(points):
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
                print(f"  📦 {object_name}: {len(object_points):,} points")
        
        return segmented_objects
    
    def create_enhanced_visualization(self, pointcloud, segmented_objects, save_path):
        """
        Create enhanced visualization with maximum detail.
        """
        print("🔄 Creating enhanced visualization...")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(24, 16))
        
        # 1. Full dense point cloud (top view)
        ax1 = fig.add_subplot(2, 4, 1, projection='3d')
        points = np.asarray(pointcloud.points)
        colors = np.asarray(pointcloud.colors)
        
        # Sample more points for visualization
        if len(points) > 20000:
            indices = np.random.choice(len(points), 20000, replace=False)
            points_viz = points[indices]
            colors_viz = colors[indices]
        else:
            points_viz = points
            colors_viz = colors
        
        ax1.scatter(points_viz[:, 0], points_viz[:, 1], points_viz[:, 2], 
                   c=colors_viz, s=0.5, alpha=0.8)
        ax1.set_title(f'Dense Point Cloud ({len(pointcloud.points):,} points)', fontsize=12)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        
        # 2. Individual object point clouds (up to 6 objects)
        for i, (object_name, object_pc) in enumerate(segmented_objects.items()):
            if i >= 6:  # Max 6 objects for visualization
                break
                
            ax = fig.add_subplot(2, 4, i + 2, projection='3d')
            obj_points = np.asarray(object_pc.points)
            obj_colors = np.asarray(object_pc.colors)
            
            # Show more points for each object
            if len(obj_points) > 5000:
                indices = np.random.choice(len(obj_points), 5000, replace=False)
                obj_points_viz = obj_points[indices]
                obj_colors_viz = obj_colors[indices]
            else:
                obj_points_viz = obj_points
                obj_colors_viz = obj_colors
            
            ax.scatter(obj_points_viz[:, 0], obj_points_viz[:, 1], obj_points_viz[:, 2], 
                      c=obj_colors_viz, s=1, alpha=0.9)
            
            ax.set_title(f'{object_name.replace("_", " ").title()}\n({len(obj_points):,} points)', fontsize=10)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
        
        # 8. Summary statistics
        ax8 = fig.add_subplot(2, 4, 8)
        ax8.axis('off')
        
        stats_text = f"""
Enhanced Point Cloud Analysis:
=============================
Total Points: {len(pointcloud.points):,}
Objects Detected: {len(segmented_objects)}
Voxel Size: {self.voxel_size*1000:.1f}mm
Depth Range: {self.min_depth:.3f}m - {self.max_depth:.1f}m

Object Statistics:
"""
        
        for object_name, object_pc in segmented_objects.items():
            stats_text += f"• {object_name}: {len(object_pc.points):,} points\n"
        
        ax8.text(0.1, 0.9, stats_text, transform=ax8.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Enhanced visualization saved to: {save_path}")
        
        return fig
    
    def process_image_pair_enhanced(self, rgb_file, depth_file, output_dir="../results"):
        """
        Enhanced pipeline: RGB + Depth → Dense Point Cloud → 3D Analysis.
        """
        print(f"\n🚀 Enhanced RGB→Point Cloud: {os.path.basename(rgb_file)}")
        
        # Load images
        rgb_image = self.load_rgb_image(rgb_file)
        depth_image = self.load_depth_image(depth_file)
        
        if rgb_image is None or depth_image is None:
            print("❌ Failed to load images")
            return None
        
        # Create dense point cloud
        pointcloud = self.create_dense_pointcloud(rgb_image, depth_image)
        pointcloud = self.process_dense_pointcloud(pointcloud)
        
        # Detect objects in RGB image
        detections = self.detect_objects(rgb_image)
        
        # Segment objects in dense point cloud
        segmented_objects = self.segment_objects_dense(pointcloud, detections)
        
        # Create enhanced visualization
        timestamp = int(time.time())
        save_path = f"{output_dir}/enhanced_pointcloud_{timestamp}.jpg"
        self.create_enhanced_visualization(pointcloud, segmented_objects, save_path)
        
        # Save dense point cloud data
        pcd_save_path = f"{output_dir}/enhanced_pointcloud_{timestamp}.ply"
        o3d.io.write_point_cloud(pcd_save_path, pointcloud)
        print(f"✅ Enhanced point cloud saved to: {pcd_save_path}")
        
        return {
            'pointcloud': pointcloud,
            'segmented_objects': segmented_objects,
            'detections': detections,
            'visualization_path': save_path,
            'pointcloud_path': pcd_save_path
        }
    
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
    """Main function to test the enhanced point cloud system."""
    print("🚀 Enhanced Point Cloud System - Test Mode")
    print("=" * 50)
    
    # Initialize system
    system = EnhancedPointCloudSystem()
    
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
            
            result = system.process_image_pair_enhanced(latest_rgb, latest_depth)
            if result:
                print("✅ Enhanced RGB→Point Cloud conversion completed successfully!")
        else:
            print("❌ No captured images found in /tmp/")
    
    elif choice == "2":
        rgb_file = input("Enter RGB file path: ").strip()
        depth_file = input("Enter depth file path: ").strip()
        
        result = system.process_image_pair_enhanced(rgb_file, depth_file)
        if result:
            print("✅ Enhanced RGB→Point Cloud conversion completed successfully!")
    
    elif choice == "3":
        # Process all recent images
        rgb_files = sorted(glob.glob("/tmp/auto_kinect_*_rgb.txt"))
        depth_files = sorted(glob.glob("/tmp/auto_kinect_*_depth.txt"))
        
        print(f"🔄 Processing {len(rgb_files)} image pairs with enhanced system...")
        
        for i, (rgb_file, depth_file) in enumerate(zip(rgb_files, depth_files)):
            print(f"\n📸 Processing pair {i+1}/{len(rgb_files)}: {os.path.basename(rgb_file)}")
            result = system.process_image_pair_enhanced(rgb_file, depth_file)
            
            if result:
                print(f"✅ Pair {i+1} completed successfully!")
            else:
                print(f"❌ Pair {i+1} failed!")
        
        print(f"\n🎉 Enhanced batch processing completed! {len(rgb_files)} pairs processed.")
    
    else:
        print("❌ Invalid choice!")

if __name__ == "__main__":
    import glob
    main()




