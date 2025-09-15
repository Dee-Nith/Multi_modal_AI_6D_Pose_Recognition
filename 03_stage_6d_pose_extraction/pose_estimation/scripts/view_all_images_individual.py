#!/usr/bin/env python3
"""
👁️ Individual Image Viewer
==========================
View all images from 12 onwards individually, one after the other.
"""

import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import time

class IndividualImageViewer:
    """View all images individually."""
    
    def __init__(self):
        """Initialize the individual image viewer."""
        print("👁️ Initializing Individual Image Viewer...")
        
        # Load YOLO model
        self.model = YOLO('../../coppelia_sim_results/weights/best.pt')
        self.camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
        
        print("✅ Individual Image Viewer initialized!")
    
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
        
        for v in range(0, height, 2):  # Sample every 2nd pixel
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
    
    def find_all_images_from_12(self):
        """Find all images from 12 onwards."""
        images = []
        for i in range(12, 100):  # Check up to 99
            rgb_file = f"/tmp/auto_kinect_{i}_rgb.txt"
            depth_file = f"/tmp/auto_kinect_{i}_depth.txt"
            
            if os.path.exists(rgb_file) and os.path.exists(depth_file):
                images.append(i)
            elif i > 32:  # Stop if we've gone past the last image
                break
        
        return sorted(images)
    
    def view_all_images_individually(self):
        """View all images from 12 onwards individually."""
        print("🔍 Finding all images from 12 onwards...")
        
        all_image_ids = self.find_all_images_from_12()
        
        if not all_image_ids:
            print("❌ No images found from 12 onwards!")
            return
        
        print(f"✅ Found {len(all_image_ids)} images: {all_image_ids}")
        print(f"\n👁️ Will view {len(all_image_ids)} images individually...")
        print("Press 'q' in the viewer to close and move to next image")
        print("Press 'Ctrl+C' to stop viewing all images")
        
        for i, image_id in enumerate(all_image_ids):
            try:
                print(f"\n📸 [{i+1}/{len(all_image_ids)}] Processing image {image_id}...")
                
                # Load images
                rgb_file = f"/tmp/auto_kinect_{image_id}_rgb.txt"
                depth_file = f"/tmp/auto_kinect_{image_id}_depth.txt"
                
                rgb_image = self.load_rgb_image(rgb_file)
                depth_image = self.load_depth_image(depth_file)
                
                if rgb_image is None or depth_image is None:
                    print(f"  ❌ Failed to load images for {image_id}")
                    continue
                
                # Create point cloud
                result = self.create_pointcloud(rgb_image, depth_image, image_id)
                
                if result is None:
                    print(f"  ❌ No valid data for image {image_id}")
                    continue
                
                pointcloud, detections = result
                
                if pointcloud is None:
                    print(f"  ❌ No valid points for image {image_id}")
                    continue
                
                # Clean up point cloud
                pointcloud = pointcloud.remove_duplicated_points()
                pointcloud, _ = pointcloud.remove_statistical_outlier(
                    nb_neighbors=20, std_ratio=2.0
                )
                
                # Estimate normals
                pointcloud.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=0.1, max_nn=30
                    )
                )
                
                final_point_count = len(pointcloud.points)
                print(f"  ✅ Generated {final_point_count:,} points")
                print(f"  🎯 Detected {len(detections)} objects")
                
                # Show object details
                for detection in detections:
                    print(f"    • {detection['name']}: {detection['confidence']:.2f}")
                
                print(f"\n🖼️ Opening point cloud viewer for image {image_id}...")
                print(f"📊 Points: {final_point_count:,} | Objects: {len(detections)}")
                print("Press 'q' to close and continue to next image...")
                
                # View point cloud
                o3d.visualization.draw_geometries([pointcloud])
                
                print(f"✅ Completed viewing image {image_id}")
                
                # Ask if user wants to continue
                if i < len(all_image_ids) - 1:
                    choice = input(f"\nContinue to next image ({all_image_ids[i+1]})? (y/n): ").strip().lower()
                    if choice != 'y':
                        print("🛑 Stopping image viewing...")
                        break
                
            except KeyboardInterrupt:
                print("\n🛑 User interrupted. Stopping image viewing...")
                break
            except Exception as e:
                print(f"❌ Error processing image {image_id}: {e}")
                continue
        
        print(f"\n🎉 Completed viewing {len(all_image_ids)} images!")

def main():
    """Main function."""
    print("👁️ Individual Image Viewer")
    print("=" * 40)
    
    # Initialize viewer
    viewer = IndividualImageViewer()
    
    # View all images individually
    viewer.view_all_images_individually()

if __name__ == "__main__":
    main()




