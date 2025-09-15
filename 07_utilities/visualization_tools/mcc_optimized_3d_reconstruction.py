#!/usr/bin/env python3
"""
MCC-Optimized 3D Reconstruction Pipeline
Enhanced version with better point cloud filtering and processing
"""

import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from ultralytics import YOLO
import time
import json

class MCCOptimized3DReconstruction:
    def __init__(self, model_path="training_results/yolov8s_instance_segmentation/weights/best.pt"):
        """Initialize the MCC-optimized 3D reconstruction pipeline"""
        print("🚀 Initializing MCC-Optimized 3D Reconstruction Pipeline")
        
        # Load trained YOLO model
        self.model = YOLO(model_path)
        print(f"✅ YOLO model loaded: {model_path}")
        
        # 3D reconstruction parameters
        self.voxel_size = 0.005  # 5mm voxels for higher detail
        self.depth_scale = 1000.0  # Depth scale factor
        self.min_points = 50  # Minimum points for reconstruction
        
        print("🎯 Pipeline ready for optimized 3D reconstruction!")
    
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
        
        # Run inference with lower confidence for more detections
        results = self.model(rgb_img, conf=0.25, verbose=False)
        
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
        
        print(f"   📊 Initial point cloud: {len(pcd.points)} points")
        
        # Apply MCC-inspired enhancements
        pcd = self._apply_mcc_enhancements(pcd, detections)
        
        print(f"✅ Enhanced point cloud: {len(pcd.points)} points")
        return pcd
    
    def _apply_mcc_enhancements(self, pcd, detections):
        """Apply MCC-inspired enhancements to point cloud"""
        print("   🔧 Applying MCC-inspired enhancements...")
        
        # 1. Remove statistical outliers (less aggressive)
        if len(pcd.points) > 20:
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=15, std_ratio=2.5)
            print(f"   📊 After outlier removal: {len(pcd.points)} points")
        
        # 2. Downsample for efficiency (less aggressive)
        if len(pcd.points) > 1000:
            pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
            print(f"   📊 After downsampling: {len(pcd.points)} points")
        
        # 3. Estimate normals for better surface reconstruction
        if len(pcd.points) > 10:
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.05, max_nn=20
            ))
            print(f"   📊 Normals estimated")
        
        # 4. Apply intelligent object-aware filtering
        if detections:
            pcd = self._intelligent_object_filtering(pcd, detections)
        
        # 5. Remove ground plane (if enough points)
        if len(pcd.points) > 10:
            pcd = self._remove_ground_plane(pcd)
        
        print(f"   ✅ Enhanced point cloud: {len(pcd.points)} points")
        return pcd
    
    def _intelligent_object_filtering(self, pcd, detections):
        """Intelligent filtering that preserves more relevant points"""
        print("   🧠 Applying intelligent object filtering...")
        
        # Get points and colors
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        # Create mask for valid object regions
        valid_mask = np.zeros(len(points), dtype=bool)
        
        # Calculate image center for reference
        img_center_x = 320  # Approximate for 640x480
        img_center_y = 240
        
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Calculate bbox center
            bbox_center_x = (x1 + x2) / 2
            bbox_center_y = (y1 + y2) / 2
            
            # Calculate distance from image center
            distance_from_center = np.sqrt((bbox_center_x - img_center_x)**2 + (bbox_center_y - img_center_y)**2)
            
            # Adaptive filtering based on distance from center
            if distance_from_center < 100:  # Close to center
                scale_factor = 0.8  # More aggressive filtering
            elif distance_from_center < 200:  # Medium distance
                scale_factor = 0.6
            else:  # Far from center
                scale_factor = 0.4  # Less aggressive filtering
            
            # Convert 2D bbox to 3D space with adaptive scaling
            x_min = x1 * scale_factor / 1000
            x_max = x2 * scale_factor / 1000
            y_min = y1 * scale_factor / 1000
            y_max = y2 * scale_factor / 1000
            z_min, z_max = 0.05, 2.5  # Wider depth range
            
            # Apply 3D bounding box filter
            valid_points = (
                (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
                (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
                (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
            )
            valid_mask |= valid_points
        
        # Adaptive filtering: if too few points, expand the region
        if np.sum(valid_mask) < len(points) * 0.15:  # If less than 15% points remain
            print(f"   ⚠️ Filtering too aggressive, expanding region")
            # Expand all bboxes by 50%
            for detection in detections:
                bbox = detection['bbox']
                x1, y1, x2, y2 = bbox
                
                # Expand bbox
                width = x2 - x1
                height = y2 - y1
                x1_expanded = max(0, x1 - width * 0.25)
                x2_expanded = min(640, x2 + width * 0.25)
                y1_expanded = max(0, y1 - height * 0.25)
                y2_expanded = min(480, y2 + height * 0.25)
                
                # Apply expanded filter
                x_min = x1_expanded / 1000
                x_max = x2_expanded / 1000
                y_min = y1_expanded / 1000
                y_max = y2_expanded / 1000
                z_min, z_max = 0.05, 3.0
                
                valid_points = (
                    (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
                    (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
                    (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
                )
                valid_mask |= valid_points
        
        # If still too aggressive, keep more points
        if np.sum(valid_mask) < len(points) * 0.1:
            print(f"   ⚠️ Still too aggressive, keeping 50% of points")
            # Keep points in a larger region around detected objects
            valid_mask = np.zeros(len(points), dtype=bool)
            for detection in detections:
                bbox = detection['bbox']
                x1, y1, x2, y2 = bbox
                
                # Very generous filtering
                x_min = max(0, x1 - 100) / 1000
                x_max = min(640, x2 + 100) / 1000
                y_min = max(0, y1 - 100) / 1000
                y_max = min(480, y2 + 100) / 1000
                z_min, z_max = 0.01, 5.0
                
                valid_points = (
                    (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
                    (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
                    (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
                )
                valid_mask |= valid_points
        
        # Apply filter
        if np.any(valid_mask):
            pcd_filtered = o3d.geometry.PointCloud()
            pcd_filtered.points = o3d.utility.Vector3dVector(points[valid_mask])
            pcd_filtered.colors = o3d.utility.Vector3dVector(colors[valid_mask])
            print(f"   ✅ Intelligent filtering: {len(points)} -> {len(pcd_filtered.points)} points")
            return pcd_filtered
        
        return pcd
    
    def _remove_ground_plane(self, pcd):
        """Remove ground plane using RANSAC"""
        print("   🏠 Removing ground plane...")
        
        # Check if we have enough points for RANSAC
        if len(pcd.points) < 10:
            print("   ⚠️ Not enough points for ground plane removal")
            return pcd
        
        try:
            # Find ground plane with more lenient parameters
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=0.02, ransac_n=5, num_iterations=500
            )
            
            if len(inliers) > len(pcd.points) * 0.4:  # If large plane found
                # Remove ground plane points
                pcd = pcd.select_by_index(inliers, invert=True)
                print(f"   ✅ Ground plane removed, remaining: {len(pcd.points)} points")
            else:
                print("   ℹ️ No significant ground plane detected")
                
        except Exception as e:
            print(f"   ⚠️ Ground plane removal failed: {str(e)}")
        
        return pcd
    
    def reconstruct_3d_objects(self, pcd, detections):
        """Reconstruct 3D objects from point cloud"""
        print("\n🏗️ Reconstructing 3D objects...")
        
        objects_3d = []
        
        for i, detection in enumerate(detections):
            print(f"   🔍 Processing {detection['class_name']}...")
            
            # Extract object region from point cloud
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Create bounding box filter with generous margins
            points = np.asarray(pcd.points)
            
            # Use generous filtering for object extraction
            x_min = max(0, x1 - 50) / 1000
            x_max = min(640, x2 + 50) / 1000
            y_min = max(0, y1 - 50) / 1000
            y_max = min(480, y2 + 50) / 1000
            z_min, z_max = 0.01, 3.0
            
            valid_points = (
                (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
                (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
                (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
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
            else:
                print(f"      ⚠️ {detection['class_name']}: No points found")
        
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
        
        try:
            # Create oriented bounding box
            bbox = o3d.geometry.OrientedBoundingBox.create_from_points(pcd.points)
            
            return {
                'min': min_bound.tolist(),
                'max': max_bound.tolist(),
                'center': bbox.center.tolist(),
                'extent': bbox.extent.tolist(),
                'orientation': bbox.R.tolist()
            }
        except Exception as e:
            print(f"      ⚠️ BBox estimation failed: {str(e)}")
            return {
                'min': min_bound.tolist(),
                'max': max_bound.tolist(),
                'center': ((min_bound + max_bound) / 2).tolist(),
                'extent': (max_bound - min_bound).tolist(),
                'orientation': np.eye(3).tolist()
            }
    
    def run_full_pipeline(self, rgb_path, depth_path, output_path="mcc_optimized_3d_reconstruction_results"):
        """Run the complete MCC-optimized 3D reconstruction pipeline"""
        print("🚀 Starting MCC-Optimized 3D Reconstruction Pipeline")
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
            
            # 5. Performance metrics
            total_time = time.time() - start_time
            print(f"\n⏱️ Pipeline completed in {total_time:.2f} seconds")
            print(f"📊 Performance: {len(pcd.points)/total_time:.0f} points/second")
            
            # 6. Save results
            output_dir = self._save_results(rgb_img, depth_img, pcd, objects_3d, output_path)
            
            return output_dir, objects_3d, pcd
            
        except Exception as e:
            print(f"❌ Pipeline failed: {str(e)}")
            raise
    
    def _save_results(self, rgb_img, depth_img, pcd, objects_3d, output_path):
        """Save all results"""
        print(f"\n💾 Saving results to: {output_path}")
        
        # Create output directory
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)
        
        # Save point cloud
        if len(pcd.points) > 0:
            o3d.io.write_point_cloud(str(output_dir / "enhanced_pointcloud.ply"), pcd)
            print(f"   ✅ Point cloud saved: enhanced_pointcloud.ply")
        
        # Save object information
        if objects_3d:
            self._save_object_info(objects_3d, output_dir)
            print(f"   ✅ Object info saved")
        
        # Save images
        cv2.imwrite(str(output_dir / "rgb_input.jpg"), rgb_img)
        depth_viz = ((depth_img - depth_img.min()) / (depth_img.max() - depth_img.min()) * 255).astype(np.uint8)
        cv2.imwrite(str(output_dir / "depth_input.jpg"), depth_viz)
        print(f"   ✅ Images saved")
        
        return output_dir
    
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
            f.write("MCC-Optimized 3D Reconstruction Results\n")
            f.write("=" * 50 + "\n\n")
            
            for i, obj in enumerate(objects_3d):
                f.write(f"Object {i+1}:\n")
                f.write(f"  Class: {obj['class_name']}\n")
                f.write(f"  Confidence: {obj['confidence']:.3f}\n")
                f.write(f"  Point Count: {obj['point_count']}\n")
                f.write(f"  Center: [{obj['center'][0]:.3f}, {obj['center'][1]:.3f}, {obj['center'][2]:.3f}]\n")
                if obj['bbox_3d']:
                    f.write(f"  3D BBox: {obj['bbox_3d']}\n")
                f.write("\n")

def main():
    """Main function to demonstrate the pipeline"""
    print("🎯 MCC-Optimized 3D Reconstruction Demo")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = MCCOptimized3DReconstruction()
    
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
        
        # Print summary
        if objects_3d:
            print(f"\n📊 Reconstruction Summary:")
            for obj in objects_3d:
                print(f"   {obj['class_name']}: {obj['point_count']} points")
        else:
            print(f"\n⚠️ No 3D objects reconstructed")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")

if __name__ == "__main__":
    main()




