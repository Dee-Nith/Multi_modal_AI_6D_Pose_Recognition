#!/usr/bin/env python3
"""
🎯 Simple Manual Alignment System
================================
Manually align images 12, 13, 14 using coordinate transformations.
"""

import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import time

class SimpleManualAlignment:
    """Simple manual alignment for images 12, 13, 14."""
    
    def __init__(self):
        """Initialize the simple alignment system."""
        print("🎯 Initializing Simple Manual Alignment System...")
        
        # Load YOLO model
        self.model = YOLO('../../coppelia_sim_results/weights/best.pt')
        self.camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
        
        print("✅ Simple Manual Alignment System initialized!")
    
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
    
    def create_pointcloud(self, rgb_image, depth_image, image_id):
        """Create point cloud from RGB + Depth data."""
        print(f"  📸 Creating point cloud for image {image_id}...")
        
        # Enhance depth data
        valid_mask = (depth_image > 0.01) & (depth_image < 3.0)
        depth_filtered = cv2.bilateralFilter(
            depth_image.astype(np.float32), 
            d=15, sigmaColor=0.1, sigmaSpace=15
        )
        depth_enhanced = np.where(valid_mask, depth_filtered, depth_image)
        
        # Detect objects
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
        
        print(f"    🎯 Detected {len(detections)} objects")
        
        # Create object masks
        object_masks = []
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            x1, x2 = int(x1), int(x2)
            y1, y2 = int(y1), int(y2)
            
            # Expand bbox
            margin = 60
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(rgb_image.shape[1], x2 + margin)
            y2 = min(rgb_image.shape[0], y2 + margin)
            
            object_masks.append((x1, y1, x2, y2))
        
        # Create point cloud
        height, width = depth_enhanced.shape
        points = []
        colors = []
        
        for v in range(0, height, 2):
            for u in range(0, width, 2):
                depth = depth_enhanced[v, u]
                
                if depth <= 0.01 or depth >= 3.0:
                    continue
                
                # Convert to 3D coordinates
                x = (u - self.camera_matrix[0, 2]) * depth / self.camera_matrix[0, 0]
                y = (v - self.camera_matrix[1, 2]) * depth / self.camera_matrix[1, 1]
                z = depth
                
                # Smart filtering
                is_object_point = False
                for x1, y1, x2, y2 in object_masks:
                    if x1 <= u <= x2 and y1 <= v <= y2:
                        is_object_point = True
                        break
                
                # Check if near objects
                is_near_object = self.is_near_object(u, v, object_masks, max_distance=100)
                
                # Check if above ground
                is_above_ground = z > 0.05
                
                # Check color intensity
                color = rgb_image[v, u] / 255.0
                color_intensity = np.mean(color)
                has_reasonable_color = color_intensity > 0.1
                
                # Keep good points
                if is_object_point or (is_near_object and is_above_ground and has_reasonable_color):
                    points.append([x, y, z])
                    colors.append(color)
        
        if len(points) == 0:
            return None, detections
        
        # Create Open3D point cloud
        pointcloud = o3d.geometry.PointCloud()
        pointcloud.points = o3d.utility.Vector3dVector(np.array(points))
        pointcloud.colors = o3d.utility.Vector3dVector(np.array(colors))
        
        return pointcloud, detections
    
    def is_near_object(self, u, v, object_masks, max_distance=100):
        """Check if point is near any detected object."""
        for x1, y1, x2, y2 in object_masks:
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            distance = np.sqrt((u - center_x)**2 + (v - center_y)**2)
            
            if distance <= max_distance:
                return True
        return False
    
    def apply_manual_transformation(self, pointcloud, image_id):
        """Apply manual transformation based on image ID."""
        # Create transformation matrix
        transformation = np.eye(4)
        
        if image_id == 12:
            # Front view - no transformation needed (reference)
            print("    📍 Image 12: Front view (reference)")
            return pointcloud
        
        elif image_id == 13:
            # Right view - rotate around Y-axis
            print("    🔄 Image 13: Right view - applying rotation")
            # Rotate 90 degrees around Y-axis
            angle = np.radians(90)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            transformation[:3, :3] = np.array([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ])
            # Add translation
            transformation[:3, 3] = [0.5, 0, 0]  # Move right
        
        elif image_id == 14:
            # Back view - rotate around Y-axis
            print("    🔄 Image 14: Back view - applying rotation")
            # Rotate 180 degrees around Y-axis
            angle = np.radians(180)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            transformation[:3, :3] = np.array([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ])
            # Add translation
            transformation[:3, 3] = [0, 0, 0.5]  # Move back
        
        # Apply transformation
        transformed_pcd = pointcloud.transform(transformation)
        return transformed_pcd
    
    def process_images_12_13_14(self, output_dir="../results"):
        """Process images 12, 13, 14 with manual alignment."""
        print("🚀 Processing images 12, 13, 14 with manual alignment...")
        
        image_ids = [12, 13, 14]
        pointclouds = []
        image_results = {}
        
        for image_id in image_ids:
            print(f"\n📸 Processing image {image_id}...")
            
            # Load images
            rgb_file = f"/tmp/auto_kinect_{image_id}_rgb.txt"
            depth_file = f"/tmp/auto_kinect_{image_id}_depth.txt"
            
            if not os.path.exists(rgb_file) or not os.path.exists(depth_file):
                print(f"❌ Missing files for image {image_id}")
                continue
            
            rgb_image = self.load_rgb_image(rgb_file)
            depth_image = self.load_depth_image(depth_file)
            
            if rgb_image is None or depth_image is None:
                print(f"❌ Failed to load images for {image_id}")
                continue
            
            # Create point cloud
            result = self.create_pointcloud(rgb_image, depth_image, image_id)
            
            if result is None:
                print(f"❌ No valid data for image {image_id}")
                continue
            
            pointcloud, detections = result
            
            if pointcloud is not None:
                print(f"  ✅ Generated {len(pointcloud.points):,} points")
                
                # Apply manual transformation
                aligned_pcd = self.apply_manual_transformation(pointcloud, image_id)
                pointclouds.append(aligned_pcd)
                
                image_results[image_id] = {
                    'pointcloud': aligned_pcd,
                    'detections': detections,
                    'point_count': len(aligned_pcd.points)
                }
            else:
                print(f"❌ No valid points for image {image_id}")
        
        if len(pointclouds) < 2:
            print("❌ Need at least 2 point clouds!")
            return None
        
        # Merge point clouds
        print(f"\n🔄 Merging {len(pointclouds)} point clouds...")
        merged_pointcloud = pointclouds[0]
        
        for i, pcd in enumerate(pointclouds[1:], 1):
            merged_pointcloud += pcd
            print(f"  📊 After merge {i}: {len(merged_pointcloud.points):,} points")
        
        # Clean up merged point cloud
        print("🔄 Cleaning merged point cloud...")
        merged_pointcloud = merged_pointcloud.remove_duplicated_points()
        merged_pointcloud, _ = merged_pointcloud.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0
        )
        
        # Estimate normals
        merged_pointcloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1, max_nn=30
            )
        )
        
        final_point_count = len(merged_pointcloud.points)
        print(f"✅ Final merged point cloud: {final_point_count:,} points")
        
        # Save results
        timestamp = int(time.time())
        
        # Save point cloud
        pcd_path = f"{output_dir}/manual_aligned_12_13_14_{timestamp}.ply"
        o3d.io.write_point_cloud(pcd_path, merged_pointcloud)
        print(f"✅ Manual aligned point cloud saved to: {pcd_path}")
        
        # Create visualization
        viz_path = f"{output_dir}/manual_aligned_12_13_14_{timestamp}.jpg"
        self.create_visualization(merged_pointcloud, image_results, viz_path)
        
        return {
            'merged_pointcloud': merged_pointcloud,
            'image_results': image_results,
            'pointcloud_path': pcd_path,
            'visualization_path': viz_path,
            'total_images': len(pointclouds),
            'final_points': final_point_count
        }
    
    def create_visualization(self, merged_pointcloud, image_results, save_path):
        """Create visualization of results."""
        print("🔄 Creating visualization...")
        
        fig = plt.figure(figsize=(20, 8))
        
        # Main merged point cloud
        ax1 = fig.add_subplot(1, 4, 1, projection='3d')
        points = np.asarray(merged_pointcloud.points)
        colors = np.asarray(merged_pointcloud.colors)
        
        if len(points) > 20000:
            indices = np.random.choice(len(points), 20000, replace=False)
            points_viz = points[indices]
            colors_viz = colors[indices]
        else:
            points_viz = points
            colors_viz = colors
        
        ax1.scatter(points_viz[:, 0], points_viz[:, 1], points_viz[:, 2], 
                   c=colors_viz, s=1, alpha=0.8)
        ax1.set_title(f'Manual Aligned (12,13,14)\n({len(merged_pointcloud.points):,} points)', 
                     fontsize=12, fontweight='bold')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        
        # Individual images
        for i, image_id in enumerate([12, 13, 14]):
            if image_id in image_results:
                ax = fig.add_subplot(1, 4, i + 2, projection='3d')
                
                pcd = image_results[image_id]['pointcloud']
                points = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors)
                
                if len(points) > 5000:
                    indices = np.random.choice(len(points), 5000, replace=False)
                    points_viz = points[indices]
                    colors_viz = colors[indices]
                else:
                    points_viz = points
                    colors_viz = colors
                
                ax.scatter(points_viz[:, 0], points_viz[:, 1], points_viz[:, 2], 
                          c=colors_viz, s=1, alpha=0.9)
                
                view_names = {12: "Front", 13: "Right", 14: "Back"}
                ax.set_title(f'{view_names[image_id]} View ({image_id})\n({image_results[image_id]["point_count"]:,} points)', fontsize=10)
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_zlabel('Z (m)')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualization saved to: {save_path}")
        
        return fig

def main():
    """Main function."""
    print("🎯 Simple Manual Alignment System")
    print("=" * 40)
    
    # Initialize system
    system = SimpleManualAlignment()
    
    # Process images 12, 13, 14
    result = system.process_images_12_13_14()
    
    if result:
        print("\n🎉 MANUAL ALIGNMENT COMPLETED!")
        print("=" * 40)
        print(f"📊 Total Images: {result['total_images']}")
        print(f"📈 Final Points: {result['final_points']:,}")
        print(f"📁 Point Cloud: {result['pointcloud_path']}")
        print(f"🖼️ Visualization: {result['visualization_path']}")
        
        # View result
        choice = input("\nView the manually aligned point cloud? (y/n): ").strip().lower()
        if choice == 'y':
            pcd = result['merged_pointcloud']
            print(f"🖼️ Opening manually aligned point cloud with {len(pcd.points):,} points...")
            o3d.visualization.draw_geometries([pcd])
    else:
        print("❌ Failed to process images")

if __name__ == "__main__":
    main()




