#!/usr/bin/env python3
"""
🚀 Real Multi-Angle Point Cloud System
======================================
Creates point clouds from real multi-angle RGB + Depth captures.
"""

import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import json
import os
import time
import glob
from pathlib import Path

class RealMultiAnglePointCloud:
    """
    Point cloud system using real multi-angle RGB + Depth captures.
    """
    
    def __init__(self):
        """Initialize the real multi-angle system."""
        print("🚀 Initializing Real Multi-Angle Point Cloud System...")
        
        # Load existing components
        self.model = YOLO('../../coppelia_sim_results/weights/best.pt')
        self.camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
        self.dist_coeffs = np.zeros(4)
        
        # Expected capture angles
        self.capture_angles = [0, 45, 90, 135, 180, 225, 270, 315]
        
        print("✅ Real Multi-Angle Point Cloud System initialized!")
        print(f"📸 Expected angles: {self.capture_angles}")
    
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
    
    def detect_objects_in_view(self, rgb_image):
        """Detect objects in the RGB image."""
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
    
    def calculate_camera_pose(self, angle):
        """Calculate camera pose for given angle."""
        # Center point where camera looks at
        center_point = np.array([-0.625, 0.075, 0.750])
        capture_distance = 1.0
        capture_height = 0.8
        
        # Calculate camera position
        angle_rad = np.radians(angle)
        camera_x = center_point[0] + capture_distance * np.cos(angle_rad)
        camera_y = center_point[1] + capture_distance * np.sin(angle_rad)
        camera_z = center_point[2] + capture_height
        
        camera_pos = np.array([camera_x, camera_y, camera_z])
        
        # Calculate camera orientation (looking at center)
        direction = center_point - camera_pos
        direction = direction / np.linalg.norm(direction)
        
        # Create transformation matrix
        # Z-axis points toward target
        z_axis = direction
        # X-axis is perpendicular to z and world up
        x_axis = np.cross(np.array([0, 0, 1]), z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        # Y-axis completes the right-handed system
        y_axis = np.cross(z_axis, x_axis)
        
        rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
        
        # Create 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = camera_pos
        
        return {
            'angle': angle,
            'position': camera_pos,
            'transform': transform,
            'rotation_matrix': rotation_matrix
        }
    
    def create_filtered_pointcloud(self, rgb_image, depth_image, pose, detections):
        """Create filtered point cloud from real capture."""
        # Enhance depth data
        valid_mask = (depth_image > 0.01) & (depth_image < 3.0)
        depth_filtered = cv2.bilateralFilter(
            depth_image.astype(np.float32), 
            d=15, sigmaColor=0.1, sigmaSpace=15
        )
        depth_enhanced = np.where(valid_mask, depth_filtered, depth_image)
        
        # Create object masks
        object_masks = []
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            x1, x2 = int(x1), int(x2)
            y1, y2 = int(y1), int(y2)
            
            # Expand bbox to include more object area
            margin = 40
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(rgb_image.shape[1], x2 + margin)
            y2 = min(rgb_image.shape[0], y2 + margin)
            
            object_masks.append((x1, y1, x2, y2))
        
        # Create point cloud with smart filtering
        height, width = depth_enhanced.shape
        points = []
        colors = []
        
        for v in range(0, height, 2):  # Sample every 2nd pixel for efficiency
            for u in range(0, width, 2):
                depth = depth_enhanced[v, u]
                
                if depth <= 0.01 or depth >= 3.0:
                    continue
                
                # Convert to 3D coordinates
                x = (u - self.camera_matrix[0, 2]) * depth / self.camera_matrix[0, 0]
                y = (v - self.camera_matrix[1, 2]) * depth / self.camera_matrix[1, 1]
                z = depth
                
                # Apply camera pose transformation
                point_camera = np.array([x, y, z, 1])
                point_world = pose['transform'] @ point_camera
                
                # Smart filtering: keep main objects, remove background
                is_object_point = False
                for x1, y1, x2, y2 in object_masks:
                    if x1 <= u <= x2 and y1 <= v <= y2:
                        is_object_point = True
                        break
                
                # Check if point is near objects
                is_near_object = self.is_near_object(u, v, object_masks, max_distance=60)
                
                # Check if point is above ground level
                is_above_ground = point_world[2] > 0.05
                
                # Check if point has reasonable color
                color = rgb_image[v, u] / 255.0
                color_intensity = np.mean(color)
                has_reasonable_color = color_intensity > 0.15
                
                # Check if point is not part of background surface
                is_not_background = self.is_not_background_surface(u, v, depth_enhanced, rgb_image)
                
                # Keep points that meet criteria
                if is_object_point or (is_near_object and is_above_ground and is_not_background and has_reasonable_color):
                    points.append(point_world[:3])
                    colors.append(color)
        
        if len(points) == 0:
            return None
        
        # Create Open3D point cloud
        pointcloud = o3d.geometry.PointCloud()
        pointcloud.points = o3d.utility.Vector3dVector(np.array(points))
        pointcloud.colors = o3d.utility.Vector3dVector(np.array(colors))
        
        return pointcloud
    
    def is_near_object(self, u, v, object_masks, max_distance=50):
        """Check if a point is near any detected object."""
        for x1, y1, x2, y2 in object_masks:
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            distance = np.sqrt((u - center_x)**2 + (v - center_y)**2)
            
            if distance <= max_distance:
                return True
        return False
    
    def is_not_background_surface(self, u, v, depth_image, rgb_image):
        """Check if a point is NOT part of a large flat background surface."""
        window_size = 8
        height, width = depth_image.shape
        
        u_start = max(0, u - window_size)
        u_end = min(width, u + window_size)
        v_start = max(0, v - window_size)
        v_end = min(height, v + window_size)
        
        depth_region = depth_image[v_start:v_end, u_start:u_end]
        color_region = rgb_image[v_start:v_end, u_start:u_end]
        
        depth_std = np.std(depth_region)
        color_std = np.std(color_region)
        
        is_uniform_surface = depth_std < 0.02 and color_std < 15
        
        # Check for large flat areas
        center_depth = depth_image[v, u]
        similar_depth_count = 0
        
        for dv in range(-4, 5):
            for du in range(-4, 5):
                check_v = v + dv
                check_u = u + du
                if 0 <= check_v < height and 0 <= check_u < width:
                    if abs(depth_image[check_v, check_u] - center_depth) < 0.015:
                        similar_depth_count += 1
        
        is_large_flat_area = similar_depth_count > 15
        
        return not (is_uniform_surface or is_large_flat_area)
    
    def find_multi_angle_captures(self):
        """Find all available multi-angle captures."""
        captures = {}
        
        for angle in self.capture_angles:
            rgb_file = f"/tmp/multi_angle_{angle}_rgb.txt"
            depth_file = f"/tmp/multi_angle_{angle}_depth.txt"
            
            if os.path.exists(rgb_file) and os.path.exists(depth_file):
                captures[angle] = {
                    'rgb': rgb_file,
                    'depth': depth_file
                }
        
        return captures
    
    def process_real_multi_angle_data(self, output_dir="../results"):
        """Process real multi-angle captures into point clouds."""
        print("🔍 Looking for multi-angle captures...")
        
        captures = self.find_multi_angle_captures()
        
        if not captures:
            print("❌ No multi-angle captures found!")
            print("📝 Please run the multi_angle_kinect_capture.lua script in CoppeliaSim first")
            return None
        
        print(f"✅ Found {len(captures)} angle captures: {list(captures.keys())}°")
        
        # Process each angle
        all_pointclouds = []
        angle_results = {}
        
        for angle in sorted(captures.keys()):
            print(f"\n📸 Processing angle {angle}°...")
            
            # Load images
            rgb_image = self.load_rgb_image(captures[angle]['rgb'])
            depth_image = self.load_depth_image(captures[angle]['depth'])
            
            if rgb_image is None or depth_image is None:
                print(f"❌ Failed to load images for angle {angle}°")
                continue
            
            # Calculate camera pose for this angle
            pose = self.calculate_camera_pose(angle)
            
            # Detect objects
            detections = self.detect_objects_in_view(rgb_image)
            print(f"  🎯 Detected {len(detections)} objects")
            
            # Create filtered point cloud
            pointcloud = self.create_filtered_pointcloud(rgb_image, depth_image, pose, detections)
            
            if pointcloud is not None:
                print(f"  ✅ Generated {len(pointcloud.points):,} points")
                all_pointclouds.append(pointcloud)
                angle_results[angle] = {
                    'pointcloud': pointcloud,
                    'detections': detections,
                    'point_count': len(pointcloud.points)
                }
            else:
                print(f"  ❌ No valid points generated for angle {angle}°")
        
        if not all_pointclouds:
            print("❌ No valid point clouds generated!")
            return None
        
        # Merge all point clouds
        print(f"\n🔄 Merging {len(all_pointclouds)} point clouds...")
        merged_pointcloud = all_pointclouds[0]
        
        for i, pcd in enumerate(all_pointclouds[1:], 1):
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
        
        print(f"✅ Final merged point cloud: {len(merged_pointcloud.points):,} points")
        
        # Save results
        timestamp = int(time.time())
        
        # Save point cloud
        pcd_path = f"{output_dir}/real_multiangle_pointcloud_{timestamp}.ply"
        o3d.io.write_point_cloud(pcd_path, merged_pointcloud)
        print(f"✅ Point cloud saved to: {pcd_path}")
        
        # Create visualization
        viz_path = f"{output_dir}/real_multiangle_analysis_{timestamp}.jpg"
        self.create_real_multiangle_visualization(merged_pointcloud, angle_results, viz_path)
        
        return {
            'merged_pointcloud': merged_pointcloud,
            'angle_results': angle_results,
            'pointcloud_path': pcd_path,
            'visualization_path': viz_path
        }
    
    def create_real_multiangle_visualization(self, merged_pointcloud, angle_results, save_path):
        """Create visualization of real multi-angle results."""
        print("🔄 Creating real multi-angle visualization...")
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Final merged point cloud
        ax1 = fig.add_subplot(2, 4, 1, projection='3d')
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
        ax1.set_title(f'Real Multi-Angle Reconstruction\n({len(merged_pointcloud.points):,} points)', fontsize=12)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        
        # 2-8. Individual angle results
        for i, (angle, result) in enumerate(sorted(angle_results.items())[:7]):
            ax = fig.add_subplot(2, 4, i + 2, projection='3d')
            
            pcd = result['pointcloud']
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
            
            ax.set_title(f'Real Angle {angle}°\n({result["point_count"]:,} points)', fontsize=10)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
        
        # Summary in last subplot if needed
        if len(angle_results) < 7:
            ax_summary = fig.add_subplot(2, 4, 8)
            ax_summary.axis('off')
            
            stats_text = f"""
Real Multi-Angle Capture Results:
================================
Total Angles: {len(angle_results)}
Final Points: {len(merged_pointcloud.points):,}

Angle Summary:
"""
            
            for angle, result in sorted(angle_results.items()):
                stats_text += f"• {angle}°: {result['point_count']:,} points\n"
            
            ax_summary.text(0.1, 0.9, stats_text, transform=ax_summary.transAxes, 
                           fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Real multi-angle visualization saved to: {save_path}")
        
        return fig

def main():
    """Main function to process real multi-angle captures."""
    print("🚀 Real Multi-Angle Point Cloud System")
    print("=" * 50)
    
    # Initialize system
    system = RealMultiAnglePointCloud()
    
    # Process real multi-angle data
    result = system.process_real_multi_angle_data()
    
    if result:
        print("\n🎉 Real multi-angle processing completed successfully!")
        print(f"📁 Point cloud: {result['pointcloud_path']}")
        print(f"🖼️ Visualization: {result['visualization_path']}")
        
        # View the result
        choice = input("\nView the final point cloud? (y/n): ").strip().lower()
        if choice == 'y':
            pcd = result['merged_pointcloud']
            print(f"🖼️ Opening point cloud with {len(pcd.points):,} points...")
            o3d.visualization.draw_geometries([pcd])
    else:
        print("❌ Failed to process multi-angle data")
        print("📝 Make sure to run multi_angle_kinect_capture.lua in CoppeliaSim first!")

if __name__ == "__main__":
    main()




