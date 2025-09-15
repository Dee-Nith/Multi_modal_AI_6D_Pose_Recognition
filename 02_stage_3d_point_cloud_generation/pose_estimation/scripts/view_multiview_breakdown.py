#!/usr/bin/env python3
"""
🎯 Multi-View Point Cloud Breakdown Viewer
=========================================
View all individual point clouds used in multi-view reconstruction.
"""

import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import time

class MultiViewBreakdownViewer:
    """
    Viewer to display individual point clouds from multi-view reconstruction.
    """
    
    def __init__(self):
        """Initialize the viewer."""
        print("🎯 Initializing Multi-View Breakdown Viewer...")
        
        # Load components
        self.model = YOLO('../../coppelia_sim_results/weights/best.pt')
        self.camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
        
        # Multi-view parameters (same as reconstruction)
        self.view_angles = [0, 45, 90, 135, 180, 225, 270, 315]
        self.camera_distance = 1.0
        self.camera_height = 0.5
        
        print("✅ Multi-View Breakdown Viewer initialized!")
    
    def generate_camera_poses(self):
        """Generate camera poses for multi-view capture."""
        poses = []
        
        for angle in self.view_angles:
            angle_rad = np.radians(angle)
            x = self.camera_distance * np.cos(angle_rad)
            y = self.camera_distance * np.sin(angle_rad)
            z = self.camera_height
            
            camera_pos = np.array([x, y, z])
            target_pos = np.array([0, 0, 0])
            
            z_axis = (target_pos - camera_pos) / np.linalg.norm(target_pos - camera_pos)
            x_axis = np.cross(np.array([0, 0, 1]), z_axis)
            x_axis = x_axis / np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)
            
            rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
            
            transform = np.eye(4)
            transform[:3, :3] = rotation_matrix
            transform[:3, 3] = camera_pos
            
            poses.append({
                'angle': angle,
                'position': camera_pos,
                'transform': transform,
                'rotation_matrix': rotation_matrix
            })
        
        return poses
    
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
    
    def create_synthetic_view(self, rgb_image, depth_image, angle):
        """Create synthetic view for a specific angle."""
        # Apply rotation to simulate different view
        height, width = rgb_image.shape[:2]
        center = (width // 2, height // 2)
        
        # Create rotation matrix for image
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation to RGB and depth
        rgb_rotated = cv2.warpAffine(rgb_image, rotation_matrix, (width, height))
        depth_rotated = cv2.warpAffine(depth_image, rotation_matrix, (width, height))
        
        # Add perspective change for diagonal views
        if angle in [45, 135, 225, 315]:
            pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
            pts2 = np.float32([[50, 50], [width-50, 30], [30, height-50], [width-30, height-30]])
            
            perspective_matrix = cv2.getPerspectiveTransform(pts1, pts2)
            rgb_rotated = cv2.warpPerspective(rgb_rotated, perspective_matrix, (width, height))
            depth_rotated = cv2.warpPerspective(depth_rotated, perspective_matrix, (width, height))
        
        return rgb_rotated, depth_rotated
    
    def create_pointcloud_from_view(self, rgb_image, depth_image, pose):
        """Create point cloud from a single view with pose transformation."""
        # Enhance depth data
        valid_mask = (depth_image > 0.01) & (depth_image < 3.0)
        depth_filtered = cv2.bilateralFilter(
            depth_image.astype(np.float32), 
            d=15, sigmaColor=0.1, sigmaSpace=15
        )
        depth_enhanced = np.where(valid_mask, depth_filtered, depth_image)
        
        # Create point cloud
        height, width = depth_enhanced.shape
        points = []
        colors = []
        
        for v in range(0, height, 2):  # Sample every 2nd pixel
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
                
                # Get color
                color = rgb_image[v, u] / 255.0
                
                points.append(point_world[:3])
                colors.append(color)
        
        if len(points) == 0:
            return None
        
        # Create Open3D point cloud
        pointcloud = o3d.geometry.PointCloud()
        pointcloud.points = o3d.utility.Vector3dVector(np.array(points))
        pointcloud.colors = o3d.utility.Vector3dVector(np.array(colors))
        
        return pointcloud
    
    def view_all_individual_pointclouds(self, rgb_file, depth_file):
        """
        View all individual point clouds from multi-view reconstruction.
        """
        print(f"\n🎯 Viewing Individual Point Clouds from: {os.path.basename(rgb_file)}")
        
        # Load original images
        rgb_image = self.load_rgb_image(rgb_file)
        depth_image = self.load_depth_image(depth_file)
        
        if rgb_image is None or depth_image is None:
            print("❌ Failed to load images")
            return
        
        # Generate camera poses
        camera_poses = self.generate_camera_poses()
        
        # Create and view each individual point cloud
        for i, pose in enumerate(camera_poses):
            angle = pose['angle']
            print(f"\n📸 Viewing point cloud {i+1}/8: {angle}°")
            
            # Create synthetic view
            rgb_view, depth_view = self.create_synthetic_view(rgb_image, depth_image, angle)
            
            # Create point cloud for this view
            pointcloud = self.create_pointcloud_from_view(rgb_view, depth_view, pose)
            
            if pointcloud is not None and len(pointcloud.points) > 0:
                print(f"  ✅ Generated {len(pointcloud.points):,} points")
                print(f"  🖼️ Opening viewer for {angle}° view...")
                
                # View this individual point cloud
                o3d.visualization.draw_geometries([pointcloud])
                
                # Ask user if they want to continue
                if i < len(camera_poses) - 1:
                    choice = input(f"\nPress Enter to view next ({camera_poses[i+1]['angle']}°) or 'q' to quit: ")
                    if choice.lower() == 'q':
                        print("👋 Stopping point cloud viewing.")
                        break
            else:
                print(f"  ❌ No valid points generated for {angle}° view")
        
        print("\n🎉 Finished viewing all individual point clouds!")
    
    def create_comparison_visualization(self, rgb_file, depth_file, save_path):
        """
        Create a comparison visualization showing all individual point clouds.
        """
        print(f"\n🔄 Creating comparison visualization...")
        
        # Load original images
        rgb_image = self.load_rgb_image(rgb_file)
        depth_image = self.load_depth_image(depth_file)
        
        if rgb_image is None or depth_image is None:
            print("❌ Failed to load images")
            return
        
        # Generate camera poses
        camera_poses = self.generate_camera_poses()
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # Create point clouds for each view
        for i, pose in enumerate(camera_poses):
            angle = pose['angle']
            
            # Create synthetic view
            rgb_view, depth_view = self.create_synthetic_view(rgb_image, depth_image, angle)
            
            # Create point cloud for this view
            pointcloud = self.create_pointcloud_from_view(rgb_view, depth_view, pose)
            
            # Create subplot
            ax = fig.add_subplot(2, 4, i + 1, projection='3d')
            
            if pointcloud is not None and len(pointcloud.points) > 0:
                points = np.asarray(pointcloud.points)
                colors = np.asarray(pointcloud.colors)
                
                # Sample points for visualization
                if len(points) > 5000:
                    indices = np.random.choice(len(points), 5000, replace=False)
                    points_viz = points[indices]
                    colors_viz = colors[indices]
                else:
                    points_viz = points
                    colors_viz = colors
                
                ax.scatter(points_viz[:, 0], points_viz[:, 1], points_viz[:, 2], 
                          c=colors_viz, s=1, alpha=0.8)
                
                ax.set_title(f'View {angle}°\n({len(pointcloud.points):,} points)', fontsize=10)
            else:
                ax.set_title(f'View {angle}°\n(No points)', fontsize=10)
            
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Comparison visualization saved to: {save_path}")
        
        return fig

def main():
    """Main function to view multi-view breakdown."""
    print("🎯 Multi-View Point Cloud Breakdown Viewer")
    print("=" * 50)
    
    # Initialize viewer
    viewer = MultiViewBreakdownViewer()
    
    # Find latest images
    rgb_files = sorted(glob.glob("/tmp/auto_kinect_*_rgb.txt"))
    depth_files = sorted(glob.glob("/tmp/auto_kinect_*_depth.txt"))
    
    if not rgb_files or not depth_files:
        print("❌ No captured images found in /tmp/")
        return
    
    latest_rgb = rgb_files[-1]
    latest_depth = depth_files[-1]
    
    print(f"📸 Found latest images:")
    print(f"  RGB: {latest_rgb}")
    print(f"  Depth: {latest_depth}")
    
    print("\n🎯 Viewing Options:")
    print("1. View individual point clouds one by one (interactive)")
    print("2. Create comparison visualization (all views at once)")
    
    choice = input("Enter choice (1-2): ").strip()
    
    if choice == "1":
        # View individual point clouds
        viewer.view_all_individual_pointclouds(latest_rgb, latest_depth)
    
    elif choice == "2":
        # Create comparison visualization
        timestamp = int(time.time())
        save_path = f"../results/multiview_breakdown_{timestamp}.jpg"
        viewer.create_comparison_visualization(latest_rgb, latest_depth, save_path)
        print(f"✅ Comparison visualization created!")
    
    else:
        print("❌ Invalid choice!")

if __name__ == "__main__":
    import glob
    main()




