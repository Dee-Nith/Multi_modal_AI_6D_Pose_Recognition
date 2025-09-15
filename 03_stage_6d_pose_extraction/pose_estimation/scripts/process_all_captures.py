#!/usr/bin/env python3
"""
🚀 All Captures Processor
========================
Process ALL manually captured images from 12 onwards for maximum point cloud density.
"""

import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import time
import glob

class AllCapturesProcessor:
    """Process ALL manually captured images for maximum point cloud density."""
    
    def __init__(self):
        """Initialize the processor."""
        print("🚀 Initializing All Captures Processor...")
        
        # Load YOLO model
        self.model = YOLO('../../coppelia_sim_results/weights/best.pt')
        self.camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
        
        print("✅ All Captures Processor initialized!")
    
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
    
    def detect_objects(self, rgb_image):
        """Detect objects in RGB image."""
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
        detections = self.detect_objects(rgb_image)
        
        # Create object masks
        object_masks = []
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            x1, x2 = int(x1), int(x2)
            y1, y2 = int(y1), int(y2)
            
            # Expand bbox
            margin = 50
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
                is_near_object = self.is_near_object(u, v, object_masks, max_distance=80)
                
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
    
    def is_near_object(self, u, v, object_masks, max_distance=80):
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
    
    def process_all_images(self, output_dir="../results"):
        """Process ALL images from 12 onwards."""
        print("🔍 Finding all images from 12 onwards...")
        
        all_image_ids = self.find_all_images_from_12()
        
        if not all_image_ids:
            print("❌ No images found from 12 onwards!")
            return None
        
        print(f"✅ Found {len(all_image_ids)} images: {all_image_ids}")
        
        all_pointclouds = []
        image_results = {}
        total_points = 0
        
        print(f"\n🚀 Processing {len(all_image_ids)} images...")
        
        for i, image_id in enumerate(all_image_ids):
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
            pointcloud, detections = self.create_pointcloud(rgb_image, depth_image, image_id)
            
            if pointcloud is not None:
                point_count = len(pointcloud.points)
                total_points += point_count
                print(f"  🎯 Detected {len(detections)} objects")
                print(f"  ✅ Generated {point_count:,} points")
                print(f"  📊 Running total: {total_points:,} points")
                
                all_pointclouds.append(pointcloud)
                image_results[image_id] = {
                    'pointcloud': pointcloud,
                    'detections': detections,
                    'point_count': point_count
                }
            else:
                print(f"  ❌ No valid points for image {image_id}")
        
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
        
        final_point_count = len(merged_pointcloud.points)
        print(f"✅ Final merged point cloud: {final_point_count:,} points")
        print(f"📈 Average points per image: {final_point_count // len(all_pointclouds):,}")
        
        # Save results
        timestamp = int(time.time())
        
        # Save point cloud
        pcd_path = f"{output_dir}/all_captures_pointcloud_{timestamp}.ply"
        o3d.io.write_point_cloud(pcd_path, merged_pointcloud)
        print(f"✅ Point cloud saved to: {pcd_path}")
        
        # Create comprehensive visualization
        viz_path = f"{output_dir}/all_captures_analysis_{timestamp}.jpg"
        self.create_comprehensive_visualization(merged_pointcloud, image_results, viz_path)
        
        # Save summary
        summary_path = f"{output_dir}/all_captures_summary_{timestamp}.txt"
        self.save_summary(image_results, final_point_count, summary_path)
        
        return {
            'merged_pointcloud': merged_pointcloud,
            'image_results': image_results,
            'pointcloud_path': pcd_path,
            'visualization_path': viz_path,
            'summary_path': summary_path,
            'total_images': len(all_pointclouds),
            'final_points': final_point_count
        }
    
    def create_comprehensive_visualization(self, merged_pointcloud, image_results, save_path):
        """Create comprehensive visualization of all results."""
        print("🔄 Creating comprehensive visualization...")
        
        num_images = len(image_results)
        
        # Create a large figure with multiple views
        fig = plt.figure(figsize=(24, 16))
        
        # 1. Final merged point cloud (large)
        ax1 = fig.add_subplot(2, 4, (1, 3), projection='3d')
        points = np.asarray(merged_pointcloud.points)
        colors = np.asarray(merged_pointcloud.colors)
        
        if len(points) > 30000:
            indices = np.random.choice(len(points), 30000, replace=False)
            points_viz = points[indices]
            colors_viz = colors[indices]
        else:
            points_viz = points
            colors_viz = colors
        
        ax1.scatter(points_viz[:, 0], points_viz[:, 1], points_viz[:, 2], 
                   c=colors_viz, s=1, alpha=0.8)
        ax1.set_title(f'Complete Multi-Angle Reconstruction\n({len(merged_pointcloud.points):,} points from {len(image_results)} images)', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        
        # 2. Summary statistics
        ax2 = fig.add_subplot(2, 4, 4)
        ax2.axis('off')
        
        stats_text = f"""
Complete Multi-Angle Capture Results:
====================================
Total Images Processed: {len(image_results)}
Final Point Cloud: {len(merged_pointcloud.points):,} points
Average per Image: {len(merged_pointcloud.points) // len(image_results):,} points

Image Range: {min(image_results.keys())} - {max(image_results.keys())}

Object Detection Summary:
"""
        
        total_detections = sum(len(result['detections']) for result in image_results.values())
        avg_detections = total_detections / len(image_results)
        stats_text += f"Total Objects Detected: {total_detections}\n"
        stats_text += f"Average per Image: {avg_detections:.1f}\n"
        
        ax2.text(0.1, 0.9, stats_text, transform=ax2.transAxes, 
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 3-6. Sample individual images (4 views)
        sample_images = sorted(image_results.keys())[:4]
        for i, image_id in enumerate(sample_images):
            ax = fig.add_subplot(2, 4, 5 + i, projection='3d')
            
            pcd = image_results[image_id]['pointcloud']
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            
            if len(points) > 3000:
                indices = np.random.choice(len(points), 3000, replace=False)
                points_viz = points[indices]
                colors_viz = colors[indices]
            else:
                points_viz = points
                colors_viz = colors
            
            ax.scatter(points_viz[:, 0], points_viz[:, 1], points_viz[:, 2], 
                      c=colors_viz, s=1, alpha=0.9)
            
            ax.set_title(f'Image {image_id}\n({image_results[image_id]["point_count"]:,} points)', fontsize=10)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Comprehensive visualization saved to: {save_path}")
        
        return fig
    
    def save_summary(self, image_results, final_points, summary_path):
        """Save detailed summary of all captures."""
        print("📝 Saving detailed summary...")
        
        with open(summary_path, 'w') as f:
            f.write("Complete Multi-Angle Capture Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total Images Processed: {len(image_results)}\n")
            f.write(f"Final Point Cloud Size: {final_points:,} points\n")
            f.write(f"Average Points per Image: {final_points // len(image_results):,}\n\n")
            
            f.write("Individual Image Results:\n")
            f.write("-" * 30 + "\n")
            
            total_detections = 0
            for image_id in sorted(image_results.keys()):
                result = image_results[image_id]
                f.write(f"Image {image_id:2d}: {result['point_count']:6,} points, "
                       f"{len(result['detections']):2d} objects\n")
                total_detections += len(result['detections'])
            
            f.write(f"\nTotal Objects Detected: {total_detections}\n")
            f.write(f"Average Objects per Image: {total_detections / len(image_results):.1f}\n")
        
        print(f"✅ Summary saved to: {summary_path}")

def main():
    """Main function."""
    print("🚀 All Captures Processor")
    print("=" * 50)
    
    # Initialize processor
    processor = AllCapturesProcessor()
    
    # Process all images
    result = processor.process_all_images()
    
    if result:
        print("\n🎉 ALL CAPTURES PROCESSING COMPLETED!")
        print("=" * 50)
        print(f"📊 Total Images: {result['total_images']}")
        print(f"📈 Final Points: {result['final_points']:,}")
        print(f"📁 Point Cloud: {result['pointcloud_path']}")
        print(f"🖼️ Visualization: {result['visualization_path']}")
        print(f"📝 Summary: {result['summary_path']}")
        
        # View result
        choice = input("\nView the final point cloud? (y/n): ").strip().lower()
        if choice == 'y':
            pcd = result['merged_pointcloud']
            print(f"🖼️ Opening point cloud with {len(pcd.points):,} points...")
            o3d.visualization.draw_geometries([pcd])
    else:
        print("❌ Failed to process all images")

if __name__ == "__main__":
    main()




