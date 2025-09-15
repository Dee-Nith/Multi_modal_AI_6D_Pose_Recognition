#!/usr/bin/env python3
"""
🎯 Filtered Point Cloud Viewer
=============================
Remove background surfaces and keep only main objects.
"""

import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import time

class FilteredPointCloudViewer:
    """
    Viewer that filters out background surfaces and keeps only main objects.
    """
    
    def __init__(self):
        """Initialize the filtered viewer."""
        print("🎯 Initializing Filtered Point Cloud Viewer...")
        
        # Load components
        self.model = YOLO('../../coppelia_sim_results/weights/best.pt')
        self.camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
        
        # Multi-view parameters
        self.view_angles = [0, 45, 90, 135, 180, 225, 270, 315]
        self.camera_distance = 1.0
        self.camera_height = 0.5
        
        print("✅ Filtered Point Cloud Viewer initialized!")
    
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
        height, width = rgb_image.shape[:2]
        center = (width // 2, height // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rgb_rotated = cv2.warpAffine(rgb_image, rotation_matrix, (width, height))
        depth_rotated = cv2.warpAffine(depth_image, rotation_matrix, (width, height))
        
        if angle in [45, 135, 225, 315]:
            pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
            pts2 = np.float32([[50, 50], [width-50, 30], [30, height-50], [width-30, height-30]])
            
            perspective_matrix = cv2.getPerspectiveTransform(pts1, pts2)
            rgb_rotated = cv2.warpPerspective(rgb_rotated, perspective_matrix, (width, height))
            depth_rotated = cv2.warpPerspective(depth_rotated, perspective_matrix, (width, height))
        
        return rgb_rotated, depth_rotated
    
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
    
    def create_filtered_pointcloud(self, rgb_image, depth_image, pose, detections):
        """
        Create filtered point cloud keeping only main objects, removing ALL background surfaces.
        """
        # Enhance depth data
        valid_mask = (depth_image > 0.01) & (depth_image < 3.0)
        depth_filtered = cv2.bilateralFilter(
            depth_image.astype(np.float32), 
            d=15, sigmaColor=0.1, sigmaSpace=15
        )
        depth_enhanced = np.where(valid_mask, depth_filtered, depth_image)
        
        # Create object masks with larger margins
        object_masks = []
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            x1, x2 = int(x1), int(x2)
            y1, y2 = int(y1), int(y2)
            
            # Much larger margin to include more object area
            margin = 50
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(rgb_image.shape[1], x2 + margin)
            y2 = min(rgb_image.shape[0], y2 + margin)
            
            object_masks.append((x1, y1, x2, y2))
        
        # Create point cloud with aggressive filtering
        height, width = depth_enhanced.shape
        points = []
        colors = []
        
        for v in range(0, height, 2):
            for u in range(0, width, 2):
                depth = depth_enhanced[v, u]
                
                if depth <= 0.01 or depth >= 3.0:
                    continue
                
                # Convert to 3D coordinates first
                x = (u - self.camera_matrix[0, 2]) * depth / self.camera_matrix[0, 0]
                y = (v - self.camera_matrix[1, 2]) * depth / self.camera_matrix[1, 1]
                z = depth
                
                # Apply camera pose transformation
                point_camera = np.array([x, y, z, 1])
                point_world = pose['transform'] @ point_camera
                
                # SMART FILTERING: Keep main objects, remove only background surfaces
                
                # Check if point is within object regions
                is_object_point = False
                for x1, y1, x2, y2 in object_masks:
                    if x1 <= u <= x2 and y1 <= v <= y2:
                        is_object_point = True
                        break
                
                # Check if point is near objects (but with larger tolerance)
                is_near_object = self.is_near_object(u, v, object_masks, max_distance=80)
                
                # Check if point is above ground level (remove floor/wall points)
                is_above_ground = point_world[2] > 0.05  # Lower threshold to keep more object points
                
                # Check if point has reasonable color (not pure black/gray)
                color = rgb_image[v, u] / 255.0
                color_intensity = np.mean(color)
                has_reasonable_color = color_intensity > 0.1  # Lower threshold to keep more objects
                
                # Check if point is not part of large flat surfaces (background walls/floors)
                # This is the key: remove points that form large flat areas
                is_not_background_surface = self.is_not_background_surface(u, v, depth_enhanced, rgb_image)
                
                # Keep points that are either:
                # 1. Definitely object points, OR
                # 2. Near objects AND above ground AND not background surface
                if is_object_point or (is_near_object and is_above_ground and is_not_background_surface and has_reasonable_color):
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
            # Calculate distance to object center
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            distance = np.sqrt((u - center_x)**2 + (v - center_y)**2)
            
            if distance <= max_distance:
                return True
        return False
    
    def is_not_background_surface(self, u, v, depth_image, rgb_image):
        """
        Check if a point is NOT part of a large flat background surface.
        Returns True if the point should be kept (not background).
        """
        # Check a small region around the point
        window_size = 10
        height, width = depth_image.shape
        
        # Get region bounds
        u_start = max(0, u - window_size)
        u_end = min(width, u + window_size)
        v_start = max(0, v - window_size)
        v_end = min(height, v + window_size)
        
        # Extract depth and color in this region
        depth_region = depth_image[v_start:v_end, u_start:u_end]
        color_region = rgb_image[v_start:v_end, u_start:u_end]
        
        # Check if this region is too uniform (indicating flat surface)
        depth_std = np.std(depth_region)
        color_std = np.std(color_region)
        
        # If depth and color are very uniform, it's likely a flat background surface
        is_uniform_surface = depth_std < 0.02 and color_std < 20
        
        # Check if the point is in a large continuous area of similar depth
        # This helps identify walls, floors, and other large flat surfaces
        center_depth = depth_image[v, u]
        similar_depth_count = 0
        
        for dv in range(-5, 6):
            for du in range(-5, 6):
                check_v = v + dv
                check_u = u + du
                if 0 <= check_v < height and 0 <= check_u < width:
                    if abs(depth_image[check_v, check_u] - center_depth) < 0.01:
                        similar_depth_count += 1
        
        # If too many nearby points have similar depth, it's likely a flat surface
        is_large_flat_area = similar_depth_count > 20
        
        # Return True if NOT a background surface
        return not (is_uniform_surface or is_large_flat_area)
    
    def view_filtered_pointclouds(self, rgb_file, depth_file):
        """
        View filtered point clouds (objects only, no background).
        """
        print(f"\n🎯 Viewing Filtered Point Clouds from: {os.path.basename(rgb_file)}")
        
        # Load original images
        rgb_image = self.load_rgb_image(rgb_file)
        depth_image = self.load_depth_image(depth_file)
        
        if rgb_image is None or depth_image is None:
            print("❌ Failed to load images")
            return
        
        # Generate camera poses
        camera_poses = self.generate_camera_poses()
        
        # Create and view each filtered point cloud
        for i, pose in enumerate(camera_poses):
            angle = pose['angle']
            print(f"\n📸 Creating filtered point cloud {i+1}/8: {angle}°")
            
            # Create synthetic view
            rgb_view, depth_view = self.create_synthetic_view(rgb_image, depth_image, angle)
            
            # Detect objects in this view
            detections = self.detect_objects_in_view(rgb_view)
            print(f"  🎯 Detected {len(detections)} objects")
            
            # Create filtered point cloud
            pointcloud = self.create_filtered_pointcloud(rgb_view, depth_view, pose, detections)
            
            if pointcloud is not None and len(pointcloud.points) > 0:
                print(f"  ✅ Generated {len(pointcloud.points):,} filtered points")
                print(f"  🖼️ Opening filtered viewer for {angle}° view...")
                
                # View this filtered point cloud
                o3d.visualization.draw_geometries([pointcloud])
                
                # Ask user if they want to continue
                if i < len(camera_poses) - 1:
                    choice = input(f"\nPress Enter to view next ({camera_poses[i+1]['angle']}°) or 'q' to quit: ")
                    if choice.lower() == 'q':
                        print("👋 Stopping filtered point cloud viewing.")
                        break
            else:
                print(f"  ❌ No valid filtered points generated for {angle}° view")
        
        print("\n🎉 Finished viewing all filtered point clouds!")
    
    def create_filtered_comparison(self, rgb_file, depth_file, save_path):
        """
        Create comparison visualization of filtered point clouds.
        """
        print(f"\n🔄 Creating filtered comparison visualization...")
        
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
        
        # Create filtered point clouds for each view
        for i, pose in enumerate(camera_poses):
            angle = pose['angle']
            
            # Create synthetic view
            rgb_view, depth_view = self.create_synthetic_view(rgb_image, depth_image, angle)
            
            # Detect objects in this view
            detections = self.detect_objects_in_view(rgb_view)
            
            # Create filtered point cloud
            pointcloud = self.create_filtered_pointcloud(rgb_view, depth_view, pose, detections)
            
            # Create subplot
            ax = fig.add_subplot(2, 4, i + 1, projection='3d')
            
            if pointcloud is not None and len(pointcloud.points) > 0:
                points = np.asarray(pointcloud.points)
                colors = np.asarray(pointcloud.colors)
                
                # Sample points for visualization
                if len(points) > 3000:
                    indices = np.random.choice(len(points), 3000, replace=False)
                    points_viz = points[indices]
                    colors_viz = colors[indices]
                else:
                    points_viz = points
                    colors_viz = colors
                
                ax.scatter(points_viz[:, 0], points_viz[:, 1], points_viz[:, 2], 
                          c=colors_viz, s=2, alpha=0.9)
                
                ax.set_title(f'Filtered View {angle}°\n({len(pointcloud.points):,} points)', fontsize=10)
            else:
                ax.set_title(f'Filtered View {angle}°\n(No points)', fontsize=10)
            
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Filtered comparison visualization saved to: {save_path}")
        
        return fig

def main():
    """Main function to view filtered point clouds."""
    print("🎯 Filtered Point Cloud Viewer")
    print("=" * 40)
    
    # Initialize viewer
    viewer = FilteredPointCloudViewer()
    
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
    
    print("\n🎯 Filtered Viewing Options:")
    print("1. View filtered point clouds one by one (objects only)")
    print("2. Create filtered comparison visualization")
    
    choice = input("Enter choice (1-2): ").strip()
    
    if choice == "1":
        # View filtered point clouds
        viewer.view_filtered_pointclouds(latest_rgb, latest_depth)
    
    elif choice == "2":
        # Create filtered comparison visualization
        timestamp = int(time.time())
        save_path = f"../results/filtered_pointclouds_{timestamp}.jpg"
        viewer.create_filtered_comparison(latest_rgb, latest_depth, save_path)
        print(f"✅ Filtered comparison visualization created!")
    
    else:
        print("❌ Invalid choice!")

if __name__ == "__main__":
    import glob
    main()
