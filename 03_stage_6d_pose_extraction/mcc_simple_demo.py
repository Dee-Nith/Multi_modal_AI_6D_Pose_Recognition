#!/usr/bin/env python3
"""
Simple MCC Demo - Works without PyTorch3D
Direct visualization of RGB-D with object detection
"""

import numpy as np
import cv2
import torch
import open3d as o3d
from pathlib import Path
import matplotlib.pyplot as plt
from ultralytics import YOLO
import time

def main():
    print("🎯 Simple MCC Demo - No Dependencies Required")
    print("=" * 45)
    
    # Load YOLO model
    print("🔄 Loading YOLO model...")
    model = YOLO("training_results/yolov8s_instance_segmentation/weights/best.pt")
    print("✅ YOLO model loaded")
    
    # Load RGB-D data
    rgb_path = "new_captures/all_captures/processed_images/capture_1_rgb.jpg"
    depth_path = "new_captures/all_captures/processed_images/capture_1_depth.npy"
    
    print(f"\n📁 Loading data:")
    print(f"   RGB: {rgb_path}")
    print(f"   Depth: {depth_path}")
    
    rgb = cv2.imread(rgb_path)
    depth = np.load(depth_path).reshape(rgb.shape[:2])
    
    print(f"✅ RGB: {rgb.shape}, Depth: {depth.shape}")
    
    # Run object detection
    print("\n🎯 Running object detection...")
    results = model(rgb, conf=0.25, verbose=False)
    
    detections = []
    if results and len(results) > 0:
        for result in results:
            if result.boxes is not None:
                for i, box in enumerate(result.boxes):
                    cls = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    class_name = result.names[cls]
                    detections.append(f"{class_name}: {conf:.3f}")
    
    print(f"✅ Detected {len(detections)} objects:")
    for det in detections:
        print(f"   📦 {det}")
    
    # Create point cloud
    print("\n☁️ Creating point cloud...")
    rgb_o3d = o3d.geometry.Image(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    depth_o3d = o3d.geometry.Image(depth.astype(np.float32))
    
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d, depth_o3d, depth_scale=1000.0, depth_trunc=3.0
    )
    
    height, width = rgb.shape[:2]
    fx = fy = max(width, height) * 0.8
    cx, cy = width / 2, height / 2
    
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    
    print(f"✅ Point cloud created: {len(pcd.points)} points")
    
    # Create visualization
    print("\n🎨 Creating visualization...")
    output_dir = Path("mcc_simple_demo_results")
    output_dir.mkdir(exist_ok=True)
    
    # Save point cloud
    if len(pcd.points) > 0:
        o3d.io.write_point_cloud(str(output_dir / "pointcloud.ply"), pcd)
        print(f"✅ Point cloud saved: {output_dir}/pointcloud.ply")
    
    # Create summary plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # RGB image
    axes[0, 0].imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Input RGB Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Depth image
    depth_viz = (depth - depth.min()) / (depth.max() - depth.min()) * 255
    axes[0, 1].imshow(depth_viz, cmap='viridis')
    axes[0, 1].set_title('Depth Image', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Point cloud top view
    if len(pcd.points) > 0:
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        axes[1, 0].scatter(points[:, 0], points[:, 1], c=colors, s=1, alpha=0.6)
        axes[1, 0].set_title(f'Point Cloud Top View ({len(points)} points)', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('X (m)')
        axes[1, 0].set_ylabel('Y (m)')
        axes[1, 0].axis('equal')
    
    # Detection summary
    axes[1, 1].text(0.1, 0.9, 'MCC-Style Demo Results', fontsize=16, fontweight='bold', transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.8, f'Total Points: {len(pcd.points)}', fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.7, f'Objects Detected: {len(detections)}', fontsize=12, transform=axes[1, 1].transAxes)
    
    y_pos = 0.6
    for det in detections:
        axes[1, 1].text(0.1, y_pos, f'• {det}', fontsize=10, transform=axes[1, 1].transAxes)
        y_pos -= 0.05
    
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / "mcc_demo_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Summary plot saved: {output_dir}/mcc_demo_summary.png")
    
    # Open 3D viewer
    if len(pcd.points) > 0:
        print(f"\n🎮 Opening interactive 3D viewer...")
        o3d.visualization.draw_geometries([pcd], window_name="MCC Demo - Interactive 3D Viewer")
    
    print(f"\n🎉 Demo completed!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📄 Files created:")
    print(f"   - pointcloud.ply (3D point cloud)")
    print(f"   - mcc_demo_summary.png (Summary visualization)")

if __name__ == "__main__":
    main()




