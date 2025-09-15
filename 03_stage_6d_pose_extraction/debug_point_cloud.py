#!/usr/bin/env python3
"""
Debug Point Cloud Script
To understand coordinate system and fix downsampling issues
"""

import cv2
import numpy as np
import open3d as o3d
from pathlib import Path

def debug_point_cloud():
    """Debug point cloud creation and coordinate system"""
    print("🔍 Debugging Point Cloud Creation")
    print("=" * 40)
    
    # Load data
    rgb_path = "new_captures/all_captures/processed_images/capture_1_rgb.jpg"
    depth_path = "new_captures/all_captures/processed_images/capture_1_depth.npy"
    
    # Load RGB image
    rgb_img = cv2.imread(rgb_path)
    depth_data = np.load(depth_path)
    depth_img = depth_data.reshape(rgb_img.shape[:2])
    
    print(f"📊 RGB: {rgb_img.shape}")
    print(f"📊 Depth: {depth_img.shape}")
    print(f"📊 Depth range: {depth_img.min():.6f} to {depth_img.max():.6f}")
    
    # Convert to Open3D format
    rgb_o3d = o3d.geometry.Image(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
    depth_o3d = o3d.geometry.Image(depth_img.astype(np.float32))
    
    # Create RGBD image
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d, depth_o3d,
        depth_scale=1000.0,
        depth_trunc=3.0,
        convert_rgb_to_intensity=False
    )
    
    # Camera intrinsic parameters
    width, height = rgb_img.shape[1], rgb_img.shape[0]
    fx = fy = max(width, height) * 0.8
    cx, cy = width / 2, height / 2
    
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width, height, fx, fy, cx, cy
    )
    
    print(f"📊 Camera intrinsic: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    
    print(f"📊 Initial point cloud: {len(pcd.points)} points")
    
    # Analyze point cloud coordinates
    points = np.asarray(pcd.points)
    if len(points) > 0:
        print(f"📊 X range: {points[:, 0].min():.6f} to {points[:, 0].max():.6f}")
        print(f"📊 Y range: {points[:, 1].min():.6f} to {points[:, 1].max():.6f}")
        print(f"📊 Z range: {points[:, 2].min():.6f} to {points[:, 2].max():.6f}")
        
        # Check if coordinates are reasonable
        x_range = points[:, 0].max() - points[:, 0].min()
        y_range = points[:, 1].max() - points[:, 1].min()
        z_range = points[:, 2].max() - points[:, 2].min()
        
        print(f"📊 Coordinate ranges: X={x_range:.6f}, Y={y_range:.6f}, Z={z_range:.6f}")
        
        # Try different voxel sizes
        print("\n🔧 Testing different voxel sizes:")
        voxel_sizes = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0]
        
        for voxel_size in voxel_sizes:
            pcd_test = pcd.voxel_down_sample(voxel_size=voxel_size)
            print(f"   Voxel {voxel_size}m: {len(pcd_test.points)} points")
        
        # Try without downsampling
        print("\n🔧 Testing without downsampling:")
        pcd_no_downsample = pcd.copy()
        
        # Just remove outliers
        pcd_clean, _ = pcd_no_downsample.remove_statistical_outlier(nb_neighbors=20, std_ratio=3.0)
        print(f"   After outlier removal: {len(pcd_clean.points)} points")
        
        # Save debug point cloud
        o3d.io.write_point_cloud("debug_pointcloud.ply", pcd_clean)
        print(f"✅ Debug point cloud saved: debug_pointcloud.ply")
        
        return pcd_clean
    else:
        print("❌ No points in point cloud!")
        return None

if __name__ == "__main__":
    debug_point_cloud()




