#!/usr/bin/env python3
"""
🎯 Point Cloud Viewer
====================
Simple viewer to display generated point cloud files.
"""

import open3d as o3d
import numpy as np
import os
import glob

def view_pointcloud(pcd_file):
    """View a point cloud file."""
    print(f"🔄 Loading point cloud: {pcd_file}")
    
    # Load point cloud
    pcd = o3d.io.read_point_cloud(pcd_file)
    
    if len(pcd.points) == 0:
        print("❌ No points found in point cloud!")
        return
    
    print(f"✅ Loaded {len(pcd.points)} points")
    print(f"📊 Point cloud bounds:")
    print(f"   X: {np.min(np.asarray(pcd.points)[:, 0]):.3f} to {np.max(np.asarray(pcd.points)[:, 0]):.3f}")
    print(f"   Y: {np.min(np.asarray(pcd.points)[:, 1]):.3f} to {np.max(np.asarray(pcd.points)[:, 1]):.3f}")
    print(f"   Z: {np.min(np.asarray(pcd.points)[:, 2]):.3f} to {np.max(np.asarray(pcd.points)[:, 2]):.3f}")
    
    # Visualize
    print("🖼️ Opening point cloud viewer...")
    o3d.visualization.draw_geometries([pcd])

def list_pointcloud_files():
    """List all available point cloud files."""
    pcd_files = glob.glob("../results/pointcloud_*.ply")
    
    if not pcd_files:
        print("❌ No point cloud files found!")
        return []
    
    print("📁 Available Point Cloud Files:")
    print("=" * 50)
    
    for i, file in enumerate(sorted(pcd_files)):
        # Get file size
        size = os.path.getsize(file)
        size_mb = size / (1024 * 1024)
        
        # Get point count (approximate)
        pcd = o3d.io.read_point_cloud(file)
        point_count = len(pcd.points)
        
        print(f"{i+1:2d}. {os.path.basename(file)}")
        print(f"    Size: {size_mb:.2f} MB, Points: {point_count:,}")
        print()
    
    return pcd_files

def main():
    """Main function."""
    print("🎯 Point Cloud Viewer")
    print("=" * 30)
    
    # List available files
    pcd_files = list_pointcloud_files()
    
    if not pcd_files:
        return
    
    # Get user choice
    try:
        choice = int(input("Enter file number to view (or 0 to exit): "))
        
        if choice == 0:
            print("👋 Goodbye!")
            return
        
        if 1 <= choice <= len(pcd_files):
            selected_file = pcd_files[choice - 1]
            view_pointcloud(selected_file)
        else:
            print("❌ Invalid choice!")
    
    except ValueError:
        print("❌ Please enter a valid number!")
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()




