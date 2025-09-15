#!/usr/bin/env python3
"""
CoppeliaSim to MCC Data Converter
Converts CoppeliaSim RGB-D data to MCC-compatible format
"""

import cv2
import numpy as np
import open3d as o3d
from pathlib import Path
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import trimesh

class CoppeliaToMCCConverter:
    def __init__(self, model_path="training_results/yolov8s_instance_segmentation/weights/best.pt"):
        """Initialize the converter"""
        print("🔄 Initializing CoppeliaSim to MCC Converter")
        
        # Load trained YOLO model for segmentation
        self.model = YOLO(model_path)
        print(f"✅ YOLO model loaded: {model_path}")
        
        # MCC parameters
        self.depth_scale = 1000.0
        self.output_size = (800, 800)  # MCC expects 800x800 images
        
    def load_coppelia_data(self, rgb_path, depth_path):
        """Load CoppeliaSim RGB and depth data"""
        print(f"\n📁 Loading CoppeliaSim data:")
        print(f"   RGB: {rgb_path}")
        print(f"   Depth: {depth_path}")
        
        # Load RGB image
        rgb_img = cv2.imread(rgb_path)
        if rgb_img is None:
            raise ValueError(f"Could not load RGB image: {rgb_path}")
        
        # Load depth data
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
    
    def create_point_cloud(self, rgb_img, depth_img):
        """Create point cloud from RGB-D data"""
        print("\n☁️ Creating point cloud from RGB-D data...")
        
        # Convert to Open3D format
        rgb_o3d = o3d.geometry.Image(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
        depth_o3d = o3d.geometry.Image(depth_img.astype(np.float32))
        
        # Create RGBD image
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d,
            depth_scale=self.depth_scale,
            depth_trunc=3.0,
            convert_rgb_to_intensity=False
        )
        
        # Camera intrinsic parameters (approximate for CoppeliaSim)
        width, height = rgb_img.shape[1], rgb_img.shape[0]
        fx = fy = max(width, height) * 0.8
        cx, cy = width / 2, height / 2
        
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, fx, fy, cx, cy
        )
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        
        print(f"✅ Point cloud created: {len(pcd.points)} points")
        return pcd
    
    def generate_segmentation_mask(self, rgb_img):
        """Generate segmentation mask using YOLOv8s"""
        print("\n🎯 Generating segmentation mask...")
        
        # Run YOLO detection
        results = self.model(rgb_img, conf=0.25, verbose=False)
        
        # Create blank mask
        height, width = rgb_img.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        if results and len(results) > 0:
            for result in results:
                if result.masks is not None:
                    for mask_data in result.masks:
                        # Get mask data
                        mask_array = mask_data.data[0].cpu().numpy()
                        
                        # Resize mask to image size
                        mask_resized = cv2.resize(mask_array.astype(np.float32), (width, height))
                        
                        # Add to combined mask
                        mask = np.maximum(mask, (mask_resized > 0.5).astype(np.uint8) * 255)
        
        print(f"✅ Segmentation mask generated: {mask.shape}")
        return mask
    
    def save_mcc_format(self, rgb_img, pcd, mask, output_dir):
        """Save data in MCC-compatible format"""
        print(f"\n💾 Saving MCC-compatible data to: {output_dir}")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 1. Save RGB image (resize to 800x800)
        rgb_resized = cv2.resize(rgb_img, self.output_size)
        rgb_path = output_path / "input_rgb.jpg"
        cv2.imwrite(str(rgb_path), rgb_resized)
        print(f"   ✅ RGB image saved: {rgb_path}")
        
        # 2. Save point cloud as OBJ file
        obj_path = output_path / "input_pointcloud.obj"
        self._save_point_cloud_as_obj(pcd, obj_path)
        print(f"   ✅ Point cloud saved: {obj_path}")
        
        # 3. Save segmentation mask
        mask_resized = cv2.resize(mask, self.output_size)
        mask_path = output_path / "input_segmentation.png"
        cv2.imwrite(str(mask_path), mask_resized)
        print(f"   ✅ Segmentation mask saved: {mask_path}")
        
        # 4. Create MCC demo script
        demo_script = self._create_mcc_demo_script(output_path)
        demo_path = output_path / "run_mcc_demo.py"
        with open(demo_path, 'w') as f:
            f.write(demo_script)
        print(f"   ✅ MCC demo script saved: {demo_path}")
        
        return output_path
    
    def _save_point_cloud_as_obj(self, pcd, obj_path):
        """Save point cloud as OBJ file for MCC"""
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        if len(points) == 0:
            print("⚠️ No points to save")
            return
        
        # Create mesh from point cloud (simple approach)
        # For MCC, we need a mesh, so we'll create a simple triangulation
        try:
            # Estimate normals if not present
            if not pcd.has_normals():
                pcd.estimate_normals()
            
            # Create mesh using Poisson reconstruction
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
            
            # If Poisson fails, create a simple mesh
            if len(mesh.vertices) == 0:
                print("⚠️ Poisson reconstruction failed, creating simple mesh")
                mesh = self._create_simple_mesh(points, colors)
            
        except Exception as e:
            print(f"⚠️ Mesh creation failed: {e}, creating simple mesh")
            mesh = self._create_simple_mesh(points, colors)
        
        # Save as OBJ
        o3d.io.write_triangle_mesh(str(obj_path), mesh)
        print(f"   📊 Mesh saved with {len(mesh.vertices)} vertices")
    
    def _create_simple_mesh(self, points, colors):
        """Create a simple mesh from points"""
        # Create a simple triangulated mesh
        mesh = o3d.geometry.TriangleMesh()
        
        # Add vertices
        mesh.vertices = o3d.utility.Vector3dVector(points)
        
        # Create simple triangles (this is a basic approach)
        # For better results, you might want to use Delaunay triangulation
        triangles = []
        for i in range(0, len(points) - 2, 3):
            if i + 2 < len(points):
                triangles.append([i, i+1, i+2])
        
        if triangles:
            mesh.triangles = o3d.utility.Vector3iVector(triangles)
        
        return mesh
    
    def _create_mcc_demo_script(self, output_path):
        """Create a script to run MCC demo"""
        script = f'''#!/usr/bin/env python3
"""
MCC Demo Script for CoppeliaSim Data
Generated automatically by CoppeliaToMCCConverter
"""

import sys
import os

# Add MCC to path
mcc_path = os.path.join(os.path.dirname(__file__), "..", "mcc")
sys.path.insert(0, mcc_path)

# Import MCC modules
import demo

# Set up arguments
class Args:
    def __init__(self):
        self.image = "{output_path}/input_rgb.jpg"
        self.point_cloud = "{output_path}/input_pointcloud.obj"
        self.seg = "{output_path}/input_segmentation.png"
        self.output = "{output_path}/mcc_output"
        self.granularity = 0.05
        self.score_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.temperature = 0.1
        self.checkpoint = "co3dv2_all_categories.pth"
        self.eval = True
        self.resume = "co3dv2_all_categories.pth"
        self.viz_granularity = 0.05

if __name__ == "__main__":
    print("🚀 Running MCC Demo on CoppeliaSim Data")
    print("=" * 50)
    
    args = Args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {{args.checkpoint}}")
        print("Please download the MCC checkpoint from:")
        print("https://github.com/facebookresearch/MCC")
        sys.exit(1)
    
    # Run MCC demo
    try:
        demo.main(args)
        print(f"✅ MCC demo completed! Check: {{args.output}}.html")
    except Exception as e:
        print(f"❌ MCC demo failed: {{e}}")
'''
        return script
    
    def convert_and_save(self, rgb_path, depth_path, output_dir="mcc_input_data"):
        """Convert CoppeliaSim data to MCC format"""
        print("🔄 Starting CoppeliaSim to MCC conversion")
        print("=" * 50)
        
        try:
            # 1. Load CoppeliaSim data
            rgb_img, depth_img = self.load_coppelia_data(rgb_path, depth_path)
            
            # 2. Create point cloud
            pcd = self.create_point_cloud(rgb_img, depth_img)
            
            # 3. Generate segmentation mask
            mask = self.generate_segmentation_mask(rgb_img)
            
            # 4. Save in MCC format
            output_path = self.save_mcc_format(rgb_img, pcd, mask, output_dir)
            
            print(f"\n🎉 Conversion completed!")
            print(f"📁 Output directory: {output_path}")
            print(f"📄 Files created:")
            print(f"   - input_rgb.jpg (RGB image)")
            print(f"   - input_pointcloud.obj (Point cloud)")
            print(f"   - input_segmentation.png (Segmentation mask)")
            print(f"   - run_mcc_demo.py (Demo script)")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Conversion failed: {str(e)}")
            raise

def main():
    """Main function to demonstrate the converter"""
    print("🎯 CoppeliaSim to MCC Converter Demo")
    print("=" * 40)
    
    # Initialize converter
    converter = CoppeliaToMCCConverter()
    
    # Example usage with your CoppeliaSim data
    rgb_path = "new_captures/all_captures/processed_images/capture_1_rgb.jpg"
    depth_path = "new_captures/all_captures/processed_images/capture_1_depth.npy"
    
    # Check if files exist
    if not Path(rgb_path).exists():
        print(f"❌ RGB file not found: {rgb_path}")
        return
    
    if not Path(depth_path).exists():
        print(f"❌ Depth file not found: {depth_path}")
        return
    
    # Convert data
    try:
        output_dir = converter.convert_and_save(rgb_path, depth_path)
        print(f"\n✅ Success! MCC-compatible data saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")

if __name__ == "__main__":
    main()




