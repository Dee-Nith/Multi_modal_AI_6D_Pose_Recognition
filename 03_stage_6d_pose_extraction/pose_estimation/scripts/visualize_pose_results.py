#!/usr/bin/env python3
"""
🎨 Visualize 6D Pose Estimation Results
======================================
Display detected objects with bounding boxes and 3D coordinates.
"""

import cv2
import numpy as np
import json
import trimesh
from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PoseVisualization:
    """Visualize 6D pose estimation results."""
    
    def __init__(self):
        """Initialize the visualization system."""
        print("🎨 Initializing Pose Visualization System...")
        
        # Load YOLO model
        self.model = YOLO('../../coppelia_sim_results/weights/best.pt')
        
        # Load camera calibration
        with open('../calibration/coppelia_camera_calibration.json', 'r') as f:
            calibration = json.load(f)
        
        self.camera_matrix = np.array(calibration['camera_matrix'])
        self.dist_coeffs = np.array(calibration['dist_coeffs'])
        
        # Load 3D models
        self.models = {}
        model_paths = {
            'master_chef_can': '../models/master_chef_can.obj',
            'cracker_box': '../models/cracker_box.obj',
            'mug': '../models/mug.obj',
            'banana': '../models/banana.obj',
            'mustard_bottle': '../models/mustard_bottle.obj'
        }
        
        for name, path in model_paths.items():
            if os.path.exists(path):
                mesh = trimesh.load(path)
                self.models[name] = mesh
                print(f"  📦 {name}: {len(mesh.vertices)} vertices")
            else:
                print(f"  ❌ Missing model: {path}")
        
        print("✅ Pose Visualization System initialized!")
    
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
    
    def estimate_simple_pose(self, rgb_image, depth_image, detection):
        """Estimate simplified 6D pose for a single object."""
        name = detection['name']
        bbox = detection['bbox']
        
        if name not in self.models:
            return None
        
        # Get 2D bounding box
        x1, y1, x2, y2 = bbox
        x1, x2 = int(x1), int(x2)
        y1, y2 = int(y1), int(y2)
        
        # Calculate center of bounding box
        center_u = (x1 + x2) / 2
        center_v = (y1 + y2) / 2
        
        # Get depth at center
        if 0 <= int(center_v) < depth_image.shape[0] and 0 <= int(center_u) < depth_image.shape[1]:
            center_depth = depth_image[int(center_v), int(center_u)]
        else:
            # Use average depth in bbox
            depths = []
            for v in range(y1, y2, 5):
                for u in range(x1, x2, 5):
                    if 0 <= v < depth_image.shape[0] and 0 <= u < depth_image.shape[1]:
                        depth = depth_image[v, u]
                        if depth > 0.01 and depth < 3.0:
                            depths.append(depth)
            
            if len(depths) == 0:
                return None
            
            center_depth = np.mean(depths)
        
        # Convert to 3D coordinates
        center_x = (center_u - self.camera_matrix[0, 2]) * center_depth / self.camera_matrix[0, 0]
        center_y = (center_v - self.camera_matrix[1, 2]) * center_depth / self.camera_matrix[1, 1]
        center_z = center_depth
        
        return {
            'name': name,
            'translation': [center_x, center_y, center_z],
            'confidence': detection['confidence'],
            'bbox': bbox,
            'depth': center_depth
        }
    
    def visualize_image_with_poses(self, image_id):
        """Visualize a single image with pose estimation results."""
        print(f"\n🎨 Visualizing image {image_id}...")
        
        # Load images
        rgb_file = f"/tmp/auto_kinect_{image_id}_rgb.txt"
        depth_file = f"/tmp/auto_kinect_{image_id}_depth.txt"
        
        if not os.path.exists(rgb_file) or not os.path.exists(depth_file):
            print(f"❌ Missing files for image {image_id}")
            return None
        
        rgb_image = self.load_rgb_image(rgb_file)
        depth_image = self.load_depth_image(depth_file)
        
        if rgb_image is None or depth_image is None:
            print(f"❌ Failed to load images for {image_id}")
            return None
        
        # Detect objects
        detections = self.detect_objects(rgb_image)
        print(f"🎯 Detected {len(detections)} objects")
        
        # Estimate poses
        pose_results = []
        for detection in detections:
            pose_result = self.estimate_simple_pose(rgb_image, depth_image, detection)
            if pose_result:
                pose_results.append(pose_result)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. RGB Image with Bounding Boxes
        rgb_display = rgb_image.copy()
        for i, pose_result in enumerate(pose_results):
            bbox = pose_result['bbox']
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            # Draw bounding box
            color = (0, 255, 0) if pose_result['confidence'] > 0.5 else (0, 165, 255)
            cv2.rectangle(rgb_display, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f"{pose_result['name']} ({pose_result['confidence']:.2f})"
            cv2.putText(rgb_display, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add 3D coordinates
            trans = pose_result['translation']
            coord_text = f"({trans[0]:.2f}, {trans[1]:.2f}, {trans[2]:.2f})"
            cv2.putText(rgb_display, coord_text, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        axes[0].imshow(cv2.cvtColor(rgb_display, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f'Image {image_id}: RGB with 6D Poses')
        axes[0].axis('off')
        
        # 2. Depth Image
        depth_display = depth_image.copy()
        depth_display[depth_display > 2.0] = 2.0  # Clip for visualization
        depth_display = (depth_display / 2.0 * 255).astype(np.uint8)
        axes[1].imshow(depth_display, cmap='viridis')
        axes[1].set_title(f'Image {image_id}: Depth Map')
        axes[1].axis('off')
        
        # 3. 3D Scatter Plot
        ax3d = axes[2]
        ax3d = fig.add_subplot(1, 3, 3, projection='3d')
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for i, pose_result in enumerate(pose_results):
            trans = pose_result['translation']
            color = colors[i % len(colors)]
            
            ax3d.scatter(trans[0], trans[1], trans[2], c=color, s=100, alpha=0.7)
            ax3d.text(trans[0], trans[1], trans[2], f"{pose_result['name']}\n{pose_result['confidence']:.2f}", 
                     fontsize=8)
        
        ax3d.set_xlabel('X (m)')
        ax3d.set_ylabel('Y (m)')
        ax3d.set_zlabel('Z (m)')
        ax3d.set_title(f'Image {image_id}: 3D Object Positions')
        
        # Set equal aspect ratio
        max_range = np.array([trans[0] for trans in [p['translation'] for p in pose_results]] + 
                           [trans[1] for trans in [p['translation'] for p in pose_results]] + 
                           [trans[2] for trans in [p['translation'] for p in pose_results]]).max()
        mid_x = np.mean([p['translation'][0] for p in pose_results])
        mid_y = np.mean([p['translation'][1] for p in pose_results])
        mid_z = np.mean([p['translation'][2] for p in pose_results])
        ax3d.set_xlim(mid_x - max_range, mid_x + max_range)
        ax3d.set_ylim(mid_y - max_range, mid_y + max_range)
        ax3d.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        
        # Save and show
        output_path = f"pose_visualization_image_{image_id}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved visualization: {output_path}")
        
        plt.show()
        
        return pose_results
    
    def visualize_all_images(self, image_ids=[22, 23, 24, 32]):
        """Visualize all specified images."""
        print("🎨 Visualizing 6D Pose Estimation Results...")
        
        all_results = {}
        
        for image_id in image_ids:
            results = self.visualize_image_with_poses(image_id)
            if results:
                all_results[image_id] = results
        
        return all_results

def main():
    """Main function."""
    print("🎨 6D Pose Estimation Visualization")
    print("=" * 40)
    
    # Initialize system
    visualizer = PoseVisualization()
    
    # Visualize all images
    results = visualizer.visualize_all_images()
    
    print(f"\n🎉 Completed visualization of {len(results)} images!")

if __name__ == "__main__":
    main()




