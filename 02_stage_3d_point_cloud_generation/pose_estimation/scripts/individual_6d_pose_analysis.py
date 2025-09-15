#!/usr/bin/env python3
"""
Individual 6D Pose Analysis for All Captures
Runs object detection and 6D pose estimation on each image individually,
storing results in separate folders for each capture.
"""

import cv2
import numpy as np
import json
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
import os

class Individual6DPoseAnalyzer:
    def __init__(self):
        """Initialize the 6D pose analyzer"""
        print("🎯 Initializing Individual 6D Pose Analyzer...")
        
        # Load trained YOLOv8s model
        model_path = "training_results/yolov8s_instance_segmentation/weights/best.pt"
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = YOLO(model_path)
        print(f"✅ Loaded YOLOv8s model: {model_path}")
        
        # Camera parameters (from CoppeliaSim Kinect)
        self.camera_matrix = np.array([
            [525.0, 0, 320.0],    # fx, 0, cx
            [0, 525.0, 240.0],    # 0, fy, cy
            [0, 0, 1]             # 0, 0, 1
        ])
        
        self.depth_scale = 1000.0  # Depth scale factor
        self.min_confidence = 0.5
        
        # Data directory
        self.data_dir = Path("/Users/nith/Desktop/AI_6D_Pose_recognition/semantic_segmentation_project/new_captures/all_captures/processed_images")
        
        # Output directory
        self.output_dir = Path("individual_6d_pose_results")
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"📁 Data directory: {self.data_dir}")
        print(f"📁 Output directory: {self.output_dir}")
    
    def find_capture_files(self):
        """Find all available capture files"""
        print("\n🔍 Finding capture files...")
        
        # Find all RGB and depth files
        rgb_files = list(self.data_dir.glob("capture_*_rgb.jpg"))
        depth_files = list(self.data_dir.glob("capture_*_depth.npy"))
        
        # Extract capture numbers
        captures = {}
        for rgb_file in rgb_files:
            # Extract capture number from filename (e.g., "capture_1_rgb.jpg" -> 1)
            capture_num = int(rgb_file.stem.split('_')[1])
            depth_file = self.data_dir / f"capture_{capture_num}_depth.npy"
            
            if depth_file.exists():
                captures[capture_num] = {
                    'rgb': rgb_file,
                    'depth': depth_file
                }
        
        # Sort by capture number
        sorted_captures = dict(sorted(captures.items()))
        
        print(f"✅ Found {len(sorted_captures)} complete capture pairs:")
        for capture_num in sorted_captures.keys():
            print(f"   📸 Capture {capture_num}")
        
        return sorted_captures
    
    def load_rgb_depth_data(self, rgb_path, depth_path):
        """Load RGB and depth data for a single capture"""
        # Load RGB image
        rgb = cv2.imread(str(rgb_path))
        if rgb is None:
            raise ValueError(f"Could not load RGB: {rgb_path}")
        
        # Load depth data
        depth = np.load(depth_path)
        
        # Reshape depth to match RGB dimensions
        if depth.size != rgb.shape[0] * rgb.shape[1]:
            depth = depth.reshape(rgb.shape[:2])
        
        return rgb, depth
    
    def detect_objects_with_masks(self, rgb):
        """Detect objects and extract instance masks"""
        results = self.model(rgb, conf=self.min_confidence, verbose=False)
        
        detections = []
        if results and len(results) > 0:
            for result in results:
                if result.boxes is not None and result.masks is not None:
                    boxes = result.boxes
                    masks = result.masks
                    
                    for i, (box, mask) in enumerate(zip(boxes, masks)):
                        # Get bounding box
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())
                        class_name = result.names[cls]
                        
                        # Get mask
                        mask_data = mask.data[0].cpu().numpy()
                        
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'class': cls,
                            'class_name': class_name,
                            'mask': mask_data,
                            'mask_area': np.sum(mask_data > 0.5)
                        })
        
        return detections
    
    def estimate_6d_pose_from_mask(self, rgb, depth, detection):
        """Estimate 6D pose using mask and depth information"""
        # Extract mask region
        mask = detection['mask']
        bbox = detection['bbox']
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Resize mask to image dimensions
        height, width = rgb.shape[:2]
        mask_resized = cv2.resize(mask.astype(np.float32), (width, height))
        
        # Create 3D points from masked depth
        points_3d = []
        colors_3d = []
        
        for y in range(y1, y2):
            for x in range(x1, x2):
                if mask_resized[y, x] > 0.5:  # Mask threshold
                    depth_val = depth[y, x]
                    if depth_val > 0 and depth_val < 3.0:  # Valid depth range
                        # Back-project to 3D
                        z = depth_val / self.depth_scale
                        x_3d = (x - self.camera_matrix[0, 2]) * z / self.camera_matrix[0, 0]
                        y_3d = (y - self.camera_matrix[1, 2]) * z / self.camera_matrix[1, 1]
                        
                        points_3d.append([x_3d, y_3d, z])
                        colors_3d.append(rgb[y, x] / 255.0)
        
        if len(points_3d) < 10:
            return None
        
        points_3d = np.array(points_3d)
        colors_3d = np.array(colors_3d)
        
        # Estimate pose using PCA
        pose = self._estimate_pose_pca(points_3d, colors_3d, detection)
        return pose
    
    def _estimate_pose_pca(self, points_3d, colors_3d, detection):
        """Estimate pose using PCA on 3D points"""
        try:
            # Center the points
            centroid = np.mean(points_3d, axis=0)
            centered_points = points_3d - centroid
            
            # Compute covariance matrix
            cov_matrix = np.cov(centered_points.T)
            
            # Eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            
            # Sort eigenvectors by eigenvalues (descending)
            sorted_indices = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, sorted_indices]
            
            # Extract rotation matrix
            rotation_matrix = eigenvectors
            
            # Ensure right-handed coordinate system
            if np.linalg.det(rotation_matrix) < 0:
                rotation_matrix[:, 2] *= -1
            
            # Convert to Euler angles
            rotation_euler = self._rotation_matrix_to_euler(rotation_matrix)
            
            # Estimate object dimensions
            dimensions = self._estimate_object_dimensions(points_3d, rotation_matrix)
            
            pose = {
                'translation': centroid.tolist(),
                'rotation_matrix': rotation_matrix.tolist(),
                'rotation_euler': rotation_euler.tolist(),
                'dimensions': dimensions.tolist(),
                'confidence': detection['confidence'],
                'class_name': detection['class_name'],
                'points_count': len(points_3d)
            }
            
            return pose
            
        except Exception as e:
            return None
    
    def _rotation_matrix_to_euler(self, rotation_matrix):
        """Convert rotation matrix to Euler angles (ZYX convention)"""
        try:
            sy = np.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
            
            if sy > 1e-6:
                roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                pitch = np.arctan2(-rotation_matrix[2, 0], sy)
                yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            else:
                roll = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                pitch = np.arctan2(-rotation_matrix[2, 0], sy)
                yaw = 0
            
            # Convert to degrees
            euler_angles = np.array([roll, pitch, yaw]) * 180 / np.pi
            return euler_angles
            
        except Exception as e:
            return np.array([0, 0, 0])
    
    def _estimate_object_dimensions(self, points_3d, rotation_matrix):
        """Estimate object dimensions along principal axes"""
        try:
            # Transform points to object coordinate system
            centered_points = points_3d - np.mean(points_3d, axis=0)
            transformed_points = np.dot(centered_points, rotation_matrix)
            
            # Calculate dimensions along each axis
            dimensions = np.max(transformed_points, axis=0) - np.min(transformed_points, axis=0)
            return dimensions
            
        except Exception as e:
            return np.array([0.001, 0.001, 0.001])
    
    def create_visualization(self, rgb, detections, poses, capture_num):
        """Create visualization for a single capture with proper mask display"""
        # Convert BGR to RGB for matplotlib
        rgb_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Left: Original image with detections and masks
        ax1.imshow(rgb_rgb)
        ax1.set_title(f'Capture {capture_num}: Object Detection & Instance Segmentation', fontsize=16, fontweight='bold')
        ax1.axis('off')
        
        # Draw masks and bounding boxes
        for i, (detection, pose) in enumerate(zip(detections, poses)):
            bbox = detection['bbox']
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Get mask and resize to image dimensions
            mask = detection['mask']
            height, width = rgb.shape[:2]
            mask_resized = cv2.resize(mask.astype(np.float32), (width, height))
            
            # Create mask overlay
            mask_overlay = np.zeros_like(rgb_rgb)
            mask_overlay[mask_resized > 0.5] = [0, 255, 0]  # Green mask
            
            # Apply mask overlay with transparency
            ax1.imshow(mask_overlay, alpha=0.3)
            
            # Draw bounding box
            rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                           linewidth=3, edgecolor='red', facecolor='none')
            ax1.add_patch(rect)
            
            # Add object label with better formatting
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Color code based on confidence
            if confidence > 0.8:
                color = 'lime'
                confidence_text = f"High ({confidence:.3f})"
            elif confidence > 0.6:
                color = 'yellow'
                confidence_text = f"Medium ({confidence:.3f})"
            else:
                color = 'red'
                confidence_text = f"Low ({confidence:.3f})"
            
            label = f"{class_name}\n{confidence_text}"
            ax1.text(x1, y1-10, label, color=color, fontsize=11, fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.8, edgecolor=color, pad=2))
        
        # Right: Detailed 6D pose information
        ax2.axis('off')
        ax2.set_facecolor('black')
        
        # Create detailed pose information display
        pose_info = f"🎯 Capture {capture_num} - 6D Pose Results\n"
        pose_info += "=" * 50 + "\n\n"
        
        successful_poses = 0
        for i, (detection, pose) in enumerate(zip(detections, poses)):
            pose_info += f"📦 Object {i+1}: {detection['class_name'].upper()}\n"
            pose_info += f"   🎯 Confidence: {detection['confidence']:.3f}\n"
            pose_info += f"   📐 BBox: [{detection['bbox'][0]:.1f}, {detection['bbox'][1]:.1f}, {detection['bbox'][2]:.1f}, {detection['bbox'][3]:.1f}]\n"
            pose_info += f"   🎭 Mask Area: {detection['mask_area']} pixels\n"
            
            if pose:
                successful_poses += 1
                pose_info += f"   ✅ 6D POSE ESTIMATED:\n"
                pose_info += f"      📍 Position (X, Y, Z): [{pose['translation'][0]:.6f}, {pose['translation'][1]:.6f}, {pose['translation'][2]:.6f}] m\n"
                pose_info += f"      🔄 Rotation (Roll, Pitch, Yaw): [{pose['rotation_euler'][0]:.2f}°, {pose['rotation_euler'][1]:.2f}°, {pose['rotation_euler'][2]:.2f}°]\n"
                pose_info += f"      📏 Dimensions (W, H, D): [{pose['dimensions'][0]:.6f}, {pose['dimensions'][1]:.6f}, {pose['dimensions'][2]:.6f}] m\n"
                pose_info += f"      ☁️ 3D Points: {pose['points_count']}\n"
            else:
                pose_info += f"   ❌ 6D POSE: FAILED\n"
            
            pose_info += "\n"
        
        # Add summary
        pose_info += f"\n📊 SUMMARY:\n"
        pose_info += f"   📸 Total Objects: {len(detections)}\n"
        pose_info += f"   ✅ Successful 6D Poses: {successful_poses}\n"
        pose_info += f"   📈 Success Rate: {(successful_poses/len(detections)*100):.1f}%\n"
        
        # Add detection quality assessment
        high_conf = sum(1 for d in detections if d['confidence'] > 0.8)
        medium_conf = sum(1 for d in detections if 0.6 < d['confidence'] <= 0.8)
        low_conf = sum(1 for d in detections if d['confidence'] <= 0.6)
        
        pose_info += f"\n🔍 DETECTION QUALITY:\n"
        pose_info += f"   🟢 High Confidence (>0.8): {high_conf}\n"
        pose_info += f"   🟡 Medium Confidence (0.6-0.8): {medium_conf}\n"
        pose_info += f"   🔴 Low Confidence (≤0.6): {low_conf}\n"
        
        ax2.text(0.05, 0.95, pose_info, transform=ax2.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                color='white', bbox=dict(boxstyle="round,pad=0.5", facecolor='black', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def process_single_capture(self, capture_num, rgb_path, depth_path):
        """Process a single capture and save results"""
        print(f"\n🎯 Processing Capture {capture_num}...")
        
        # Create capture-specific output directory
        capture_output_dir = self.output_dir / f"capture_{capture_num}"
        capture_output_dir.mkdir(exist_ok=True)
        
        try:
            # Load data
            rgb, depth = self.load_rgb_depth_data(rgb_path, depth_path)
            print(f"   ✅ Loaded RGB: {rgb.shape}, Depth: {depth.shape}")
            
            # Detect objects
            detections = self.detect_objects_with_masks(rgb)
            print(f"   🎯 Detected {len(detections)} objects")
            
            if len(detections) == 0:
                print(f"   ⚠️ No objects detected in capture {capture_num}")
                return None
            
            # Estimate 6D poses
            poses = []
            for detection in detections:
                pose = self.estimate_6d_pose_from_mask(rgb, depth, detection)
                poses.append(pose)
                
                if pose:
                    print(f"   ✅ {detection['class_name']}: pos=[{pose['translation'][0]:.6f}, {pose['translation'][1]:.6f}, {pose['translation'][2]:.6f}], rot=[{pose['rotation_euler'][0]:.1f}°, {pose['rotation_euler'][1]:.1f}°, {pose['rotation_euler'][2]:.1f}°]")
                else:
                    print(f"   ❌ {detection['class_name']}: pose estimation failed")
            
            # Save results
            self.save_capture_results(capture_num, detections, poses, capture_output_dir)
            
            # Create and save visualization
            fig = self.create_visualization(rgb, detections, poses, capture_num)
            viz_path = capture_output_dir / f"capture_{capture_num}_6d_pose_visualization.png"
            fig.savefig(viz_path, dpi=150, bbox_inches='tight', facecolor='black')
            plt.close(fig)
            print(f"   🖼️ Visualization saved: {viz_path}")
            
            return {
                'capture_num': capture_num,
                'detections': detections,
                'poses': poses,
                'output_dir': capture_output_dir
            }
            
        except Exception as e:
            print(f"   ❌ Error processing capture {capture_num}: {e}")
            return None
    
    def save_capture_results(self, capture_num, detections, poses, output_dir):
        """Save detailed results for a single capture"""
        # Save JSON results
        results = {
            'capture_number': capture_num,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_objects': len(detections),
            'successful_poses': sum(1 for pose in poses if pose is not None),
            'detections': [],
            'poses': []
        }
        
        for i, (detection, pose) in enumerate(zip(detections, poses)):
            detection_info = {
                'object_id': i + 1,
                'class_name': detection['class_name'],
                'confidence': detection['confidence'],
                'bbox': detection['bbox'],
                'mask_area': detection['mask_area']
            }
            results['detections'].append(detection_info)
            
            if pose:
                pose_info = {
                    'object_id': i + 1,
                    'translation': pose['translation'],
                    'rotation_euler': pose['rotation_euler'],
                    'dimensions': pose['dimensions'],
                    'points_count': pose['points_count']
                }
                results['poses'].append(pose_info)
        
        # Save JSON
        json_path = output_dir / f"capture_{capture_num}_6d_pose_results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save text summary
        txt_path = output_dir / f"capture_{capture_num}_6d_pose_summary.txt"
        with open(txt_path, 'w') as f:
            f.write(f"Capture {capture_num} - 6D Pose Analysis Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {results['timestamp']}\n")
            f.write(f"Total Objects: {results['total_objects']}\n")
            f.write(f"Successful 6D Poses: {results['successful_poses']}\n")
            f.write(f"Success Rate: {(results['successful_poses']/results['total_objects']*100):.1f}%\n\n")
            
            for i, (detection, pose) in enumerate(zip(detections, poses)):
                f.write(f"Object {i+1}: {detection['class_name']}\n")
                f.write(f"  Confidence: {detection['confidence']:.3f}\n")
                f.write(f"  BBox: {detection['bbox']}\n")
                f.write(f"  Mask Area: {detection['mask_area']} pixels\n")
                
                if pose:
                    f.write(f"  ✅ 6D Pose:\n")
                    f.write(f"    Position (X, Y, Z): {pose['translation']}\n")
                    f.write(f"    Rotation (Roll, Pitch, Yaw): {pose['rotation_euler']}\n")
                    f.write(f"    Dimensions (W, H, D): {pose['dimensions']}\n")
                    f.write(f"    3D Points: {pose['points_count']}\n")
                else:
                    f.write(f"  ❌ 6D Pose: FAILED\n")
                f.write("\n")
        
        print(f"   💾 Results saved to: {output_dir}")
    
    def run_all_captures(self):
        """Run 6D pose estimation on all available captures"""
        print("🚀 Starting Individual 6D Pose Analysis for All Captures")
        print("=" * 70)
        
        # Find all capture files
        captures = self.find_capture_files()
        if not captures:
            print("❌ No capture files found!")
            return []
        
        all_results = []
        start_time = time.time()
        
        # Process each capture
        for capture_num, file_paths in captures.items():
            result = self.process_single_capture(capture_num, file_paths['rgb'], file_paths['depth'])
            if result:
                all_results.append(result)
            
            # Small delay to prevent overwhelming output
            time.sleep(0.1)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Create overall summary
        self.create_overall_summary(all_results, processing_time)
        
        print(f"\n🎉 Individual 6D pose analysis completed!")
        print(f"⏱️ Total processing time: {processing_time:.2f} seconds")
        print(f"📁 Results saved in: {self.output_dir}")
        
        return all_results
    
    def create_overall_summary(self, all_results, processing_time):
        """Create overall summary report"""
        if not all_results:
            return
        
        summary_path = self.output_dir / "overall_analysis_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("Individual 6D Pose Analysis - Overall Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Processing Time: {processing_time:.2f} seconds\n")
            f.write(f"Total Captures Processed: {len(all_results)}\n\n")
            
            total_objects = 0
            total_poses = 0
            
            for result in all_results:
                capture_num = result['capture_num']
                detections = result['detections']
                poses = result['poses']
                
                successful_poses = sum(1 for pose in poses if pose is not None)
                
                f.write(f"Capture {capture_num}:\n")
                f.write(f"  Objects Detected: {len(detections)}\n")
                f.write(f"  6D Poses Estimated: {successful_poses}\n")
                f.write(f"  Success Rate: {(successful_poses/len(detections)*100):.1f}%\n")
                f.write(f"  Output Directory: capture_{capture_num}/\n\n")
                
                total_objects += len(detections)
                total_poses += successful_poses
            
            f.write(f"OVERALL SUMMARY:\n")
            f.write(f"  Total Objects: {total_objects}\n")
            f.write(f"  Total 6D Poses: {total_poses}\n")
            f.write(f"  Overall Success Rate: {(total_poses/total_objects*100):.1f}%\n")
            f.write(f"  Results Directory: {self.output_dir}\n")
        
        print(f"📄 Overall summary saved: {summary_path}")

def main():
    """Main function"""
    try:
        analyzer = Individual6DPoseAnalyzer()
        results = analyzer.run_all_captures()
        
        print(f"\n🎯 Individual analysis completed for {len(results)} captures!")
        print(f"📁 Check the '{analyzer.output_dir}' folder for detailed results")
        print(f"📂 Each capture has its own subfolder with complete results")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
