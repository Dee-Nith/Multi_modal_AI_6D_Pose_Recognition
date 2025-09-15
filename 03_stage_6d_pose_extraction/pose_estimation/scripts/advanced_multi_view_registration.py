#!/usr/bin/env python3
"""
🚀 Advanced Multi-View Registration System
=========================================
Properly align and merge point clouds from different camera angles using:
- Feature matching and correspondence
- Camera pose estimation
- Coordinate transformation
- Iterative closest point (ICP) alignment
"""

import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import time
from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN
import json

class AdvancedMultiViewRegistration:
    """Advanced multi-view point cloud registration and alignment."""
    
    def __init__(self):
        """Initialize the advanced registration system."""
        print("🚀 Initializing Advanced Multi-View Registration System...")
        
        # Load YOLO model
        self.model = YOLO('../../coppelia_sim_results/weights/best.pt')
        self.camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
        self.dist_coeffs = np.zeros(4)
        
        # Registration parameters
        self.voxel_size = 0.02  # 2cm voxel size for downsampling
        self.icp_threshold = 0.05  # 5cm ICP threshold
        self.feature_radius = 0.1  # 10cm feature radius
        
        print("✅ Advanced Multi-View Registration System initialized!")
    
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
    
    def detect_objects_and_features(self, rgb_image, depth_image, image_id):
        """Detect objects and extract features for registration."""
        print(f"  🔍 Detecting objects and features for image {image_id}...")
        
        # Detect objects with YOLO
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
        
        # Create filtered point cloud
        pointcloud = self.create_filtered_pointcloud(rgb_image, depth_image, detections)
        
        if pointcloud is None or len(pointcloud.points) == 0:
            return None
        
        # Extract features for registration
        features = self.extract_registration_features(pointcloud, detections)
        
        return pointcloud, detections, features
    
    def create_filtered_pointcloud(self, rgb_image, depth_image, detections):
        """Create filtered point cloud with object focus."""
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
            return None
        
        # Create Open3D point cloud
        pointcloud = o3d.geometry.PointCloud()
        pointcloud.points = o3d.utility.Vector3dVector(np.array(points))
        pointcloud.colors = o3d.utility.Vector3dVector(np.array(colors))
        
        return pointcloud
    
    def is_near_object(self, u, v, object_masks, max_distance=100):
        """Check if point is near any detected object."""
        for x1, y1, x2, y2 in object_masks:
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            distance = np.sqrt((u - center_x)**2 + (v - center_y)**2)
            
            if distance <= max_distance:
                return True
        return False
    
    def extract_registration_features(self, pointcloud, detections):
        """Extract features for registration."""
        # Downsample for feature extraction
        downsampled = pointcloud.voxel_down_sample(voxel_size=self.voxel_size)
        
        # Estimate normals
        downsampled.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.feature_radius, max_nn=30
            )
        )
        
        # Extract FPFH features
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            downsampled,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.feature_radius * 2, max_nn=100
            )
        )
        
        # Find object centers for reference points
        object_centers = []
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            center_u = (x1 + x2) / 2
            center_v = (y1 + y2) / 2
            
            # Convert to 3D (approximate)
            depth = 1.0  # Approximate depth for center
            center_x = (center_u - self.camera_matrix[0, 2]) * depth / self.camera_matrix[0, 0]
            center_y = (center_v - self.camera_matrix[1, 2]) * depth / self.camera_matrix[1, 1]
            center_z = depth
            
            object_centers.append({
                'name': detection['name'],
                'center_2d': [center_u, center_v],
                'center_3d': [center_x, center_y, center_z],
                'bbox': bbox
            })
        
        return {
            'downsampled': downsampled,
            'fpfh': fpfh,
            'object_centers': object_centers,
            'detections': detections
        }
    
    def find_correspondences(self, features1, features2):
        """Find correspondences between two feature sets."""
        # Use FPFH features for matching
        fpfh1 = features1['fpfh']
        fpfh2 = features2['fpfh']
        
        # Find correspondences using feature matching
        correspondences = []
        
        # Simple feature matching (can be improved with RANSAC)
        for i in range(min(len(fpfh1.data.T), 100)):  # Limit to 100 features
            feature1 = fpfh1.data.T[i]
            
            # Find best match
            best_dist = float('inf')
            best_j = -1
            
            for j in range(min(len(fpfh2.data.T), 100)):
                feature2 = fpfh2.data.T[j]
                dist = np.linalg.norm(feature1 - feature2)
                
                if dist < best_dist:
                    best_dist = dist
                    best_j = j
            
            if best_j != -1 and best_dist < 0.5:  # Threshold
                correspondences.append([i, best_j])
        
        return correspondences
    
    def estimate_transformation(self, source_features, target_features, correspondences):
        """Estimate transformation between two point clouds."""
        if len(correspondences) < 3:
            return None
        
        # Get corresponding points
        source_points = np.asarray(source_features['downsampled'].points)
        target_points = np.asarray(target_features['downsampled'].points)
        
        source_corr = []
        target_corr = []
        
        for corr in correspondences:
            if corr[0] < len(source_points) and corr[1] < len(target_points):
                source_corr.append(source_points[corr[0]])
                target_corr.append(target_points[corr[1]])
        
        if len(source_corr) < 3:
            return None
        
        source_corr = np.array(source_corr)
        target_corr = np.array(target_corr)
        
        # Estimate transformation using SVD
        # Center the points
        source_centered = source_corr - np.mean(source_corr, axis=0)
        target_centered = target_corr - np.mean(target_corr, axis=0)
        
        # Compute covariance matrix
        H = source_centered.T @ target_centered
        
        # SVD decomposition
        U, S, Vt = np.linalg.svd(H)
        
        # Rotation matrix
        R = Vt.T @ U.T
        
        # Ensure proper rotation matrix
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Translation
        t = np.mean(target_corr, axis=0) - R @ np.mean(source_corr, axis=0)
        
        # Create transformation matrix
        transformation = np.eye(4)
        transformation[:3, :3] = R
        transformation[:3, 3] = t
        
        return transformation
    
    def align_pointclouds(self, pointclouds, features_list):
        """Align multiple point clouds using pairwise registration."""
        print("🔄 Aligning point clouds...")
        
        if len(pointclouds) < 2:
            return pointclouds
        
        # Use first point cloud as reference
        reference_idx = 0
        aligned_pointclouds = [pointclouds[reference_idx]]
        transformations = [np.eye(4)]  # Identity for reference
        
        print(f"  📍 Using point cloud {reference_idx} as reference")
        
        # Align each point cloud to the reference
        for i in range(1, len(pointclouds)):
            print(f"  🔄 Aligning point cloud {i} to reference...")
            
            # Find correspondences
            correspondences = self.find_correspondences(
                features_list[i], features_list[reference_idx]
            )
            
            print(f"    📊 Found {len(correspondences)} correspondences")
            
            # Estimate transformation
            transformation = self.estimate_transformation(
                features_list[i], features_list[reference_idx], correspondences
            )
            
            if transformation is not None:
                # Apply transformation
                aligned_pcd = pointclouds[i].transform(transformation)
                aligned_pointclouds.append(aligned_pcd)
                transformations.append(transformation)
                
                print(f"    ✅ Successfully aligned point cloud {i}")
            else:
                # If transformation fails, use original
                aligned_pointclouds.append(pointclouds[i])
                transformations.append(np.eye(4))
                print(f"    ⚠️ Using original point cloud {i} (no transformation)")
        
        return aligned_pointclouds, transformations
    
    def refine_alignment_with_icp(self, aligned_pointclouds):
        """Refine alignment using Iterative Closest Point."""
        print("🔄 Refining alignment with ICP...")
        
        if len(aligned_pointclouds) < 2:
            return aligned_pointclouds
        
        refined_pointclouds = [aligned_pointclouds[0]]
        
        for i in range(1, len(aligned_pointclouds)):
            print(f"  🔄 ICP refinement for point cloud {i}...")
            
            # Downsample for ICP
            source_down = aligned_pointclouds[i].voxel_down_sample(voxel_size=self.voxel_size)
            target_down = aligned_pointclouds[0].voxel_down_sample(voxel_size=self.voxel_size)
            
            # Estimate normals
            source_down.estimate_normals()
            target_down.estimate_normals()
            
            # ICP registration
            icp_result = o3d.pipelines.registration.registration_icp(
                source_down, target_down, self.icp_threshold, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            
            # Apply ICP transformation
            refined_pcd = aligned_pointclouds[i].transform(icp_result.transformation)
            refined_pointclouds.append(refined_pcd)
            
            print(f"    ✅ ICP fitness: {icp_result.fitness:.3f}")
        
        return refined_pointclouds
    
    def process_all_images_with_registration(self, output_dir="../results"):
        """Process all images with advanced registration."""
        print("🚀 Processing all images with advanced registration...")
        
        # Find all images from 12 onwards
        all_image_ids = []
        for i in range(12, 100):
            rgb_file = f"/tmp/auto_kinect_{i}_rgb.txt"
            depth_file = f"/tmp/auto_kinect_{i}_depth.txt"
            
            if os.path.exists(rgb_file) and os.path.exists(depth_file):
                all_image_ids.append(i)
            elif i > 32:
                break
        
        if not all_image_ids:
            print("❌ No images found from 12 onwards!")
            return None
        
        print(f"✅ Found {len(all_image_ids)} images: {all_image_ids}")
        
        # Process each image
        pointclouds = []
        features_list = []
        image_results = {}
        
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
            
            # Detect objects and extract features
            result = self.detect_objects_and_features(rgb_image, depth_image, image_id)
            
            if result is None:
                print(f"  ❌ No valid data for image {image_id}")
                continue
            
            pointcloud, detections, features = result
            
            if pointcloud is not None:
                print(f"  🎯 Detected {len(detections)} objects")
                print(f"  ✅ Generated {len(pointcloud.points):,} points")
                
                pointclouds.append(pointcloud)
                features_list.append(features)
                image_results[image_id] = {
                    'pointcloud': pointcloud,
                    'detections': detections,
                    'features': features,
                    'point_count': len(pointcloud.points)
                }
            else:
                print(f"  ❌ No valid points for image {image_id}")
        
        if len(pointclouds) < 2:
            print("❌ Need at least 2 point clouds for registration!")
            return None
        
        # Align point clouds
        print(f"\n🔄 Aligning {len(pointclouds)} point clouds...")
        aligned_pointclouds, transformations = self.align_pointclouds(pointclouds, features_list)
        
        # Refine with ICP
        refined_pointclouds = self.refine_alignment_with_icp(aligned_pointclouds)
        
        # Merge aligned point clouds
        print("🔄 Merging aligned point clouds...")
        merged_pointcloud = refined_pointclouds[0]
        
        for i, pcd in enumerate(refined_pointclouds[1:], 1):
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
        print(f"✅ Final aligned point cloud: {final_point_count:,} points")
        
        # Save results
        timestamp = int(time.time())
        
        # Save point cloud
        pcd_path = f"{output_dir}/aligned_pointcloud_{timestamp}.ply"
        o3d.io.write_point_cloud(pcd_path, merged_pointcloud)
        print(f"✅ Aligned point cloud saved to: {pcd_path}")
        
        # Save transformations
        transform_path = f"{output_dir}/transformations_{timestamp}.json"
        self.save_transformations(transformations, all_image_ids, transform_path)
        
        # Create visualization
        viz_path = f"{output_dir}/aligned_analysis_{timestamp}.jpg"
        self.create_aligned_visualization(merged_pointcloud, image_results, viz_path)
        
        return {
            'merged_pointcloud': merged_pointcloud,
            'image_results': image_results,
            'transformations': transformations,
            'pointcloud_path': pcd_path,
            'transform_path': transform_path,
            'visualization_path': viz_path,
            'total_images': len(pointclouds),
            'final_points': final_point_count
        }
    
    def save_transformations(self, transformations, image_ids, save_path):
        """Save transformation matrices."""
        print("📝 Saving transformation matrices...")
        
        data = {
            'transformations': [t.tolist() for t in transformations],
            'image_ids': image_ids[:len(transformations)],
            'timestamp': time.time()
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Transformations saved to: {save_path}")
    
    def create_aligned_visualization(self, merged_pointcloud, image_results, save_path):
        """Create visualization of aligned results."""
        print("🔄 Creating aligned visualization...")
        
        fig = plt.figure(figsize=(20, 12))
        
        # Main aligned point cloud
        ax1 = fig.add_subplot(2, 3, (1, 2), projection='3d')
        points = np.asarray(merged_pointcloud.points)
        colors = np.asarray(merged_pointcloud.colors)
        
        if len(points) > 25000:
            indices = np.random.choice(len(points), 25000, replace=False)
            points_viz = points[indices]
            colors_viz = colors[indices]
        else:
            points_viz = points
            colors_viz = colors
        
        ax1.scatter(points_viz[:, 0], points_viz[:, 1], points_viz[:, 2], 
                   c=colors_viz, s=1, alpha=0.8)
        ax1.set_title(f'Aligned Multi-View Reconstruction\n({len(merged_pointcloud.points):,} points)', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        
        # Summary statistics
        ax2 = fig.add_subplot(2, 3, 3)
        ax2.axis('off')
        
        stats_text = f"""
Advanced Multi-View Registration Results:
========================================
Total Images: {len(image_results)}
Final Points: {len(merged_pointcloud.points):,}
Registration Method: Feature-based + ICP

Quality Metrics:
- Voxel Size: {self.voxel_size}m
- ICP Threshold: {self.icp_threshold}m
- Feature Radius: {self.feature_radius}m
"""
        
        ax2.text(0.1, 0.9, stats_text, transform=ax2.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # Sample individual images
        sample_images = sorted(image_results.keys())[:3]
        for i, image_id in enumerate(sample_images):
            ax = fig.add_subplot(2, 3, 4 + i, projection='3d')
            
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
        print(f"✅ Aligned visualization saved to: {save_path}")
        
        return fig

def main():
    """Main function."""
    print("🚀 Advanced Multi-View Registration System")
    print("=" * 50)
    
    # Initialize system
    system = AdvancedMultiViewRegistration()
    
    # Process all images with registration
    result = system.process_all_images_with_registration()
    
    if result:
        print("\n🎉 ADVANCED REGISTRATION COMPLETED!")
        print("=" * 50)
        print(f"📊 Total Images: {result['total_images']}")
        print(f"📈 Final Points: {result['final_points']:,}")
        print(f"📁 Point Cloud: {result['pointcloud_path']}")
        print(f"🔄 Transformations: {result['transform_path']}")
        print(f"🖼️ Visualization: {result['visualization_path']}")
        
        # View result
        choice = input("\nView the aligned point cloud? (y/n): ").strip().lower()
        if choice == 'y':
            pcd = result['merged_pointcloud']
            print(f"🖼️ Opening aligned point cloud with {len(pcd.points):,} points...")
            o3d.visualization.draw_geometries([pcd])
    else:
        print("❌ Failed to process images with registration")

if __name__ == "__main__":
    main()
