#!/usr/bin/env python3
"""
Convert PLY point cloud to PNG image for report
"""

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def convert_ply_to_image():
    print("🔄 Converting PLY point cloud to PNG image...")
    
    # Load the PLY file
    ply_path = "/Users/nith/Desktop/AI_6D_Pose_recognition/semantic_segmentation_project/mcc_simple_demo_results/pointcloud.ply"
    
    try:
        # Load point cloud
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        print(f"✅ Loaded PLY file with {len(points)} points")
        print(f"   📊 Point cloud bounds:")
        print(f"      X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
        print(f"      Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
        print(f"      Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
        
        # Create figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot points
        if colors.size > 0:
            # Use colors if available
            scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                               c=colors, s=0.5, alpha=0.8)
        else:
            # Use grayscale if no colors
            scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                               c='gray', s=0.5, alpha=0.8)
        
        # Set labels and title
        ax.set_xlabel('X (meters)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y (meters)', fontsize=12, fontweight='bold')
        ax.set_zlabel('Z (meters)', fontsize=12, fontweight='bold')
        ax.set_title('MCC Point Cloud Reconstruction - Raw Input Data', fontsize=14, fontweight='bold')
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        # Set initial view angle
        ax.view_init(elev=20, azim=45)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add point count info
        info_text = f'Total Points: {len(points):,} | MCC Reconstruction Input'
        fig.text(0.5, 0.02, info_text, ha='center', va='bottom', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
        
        # Save high-quality image
        output_path = 'mcc_input_visualization.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        print(f"💾 Image saved as: {output_path}")
        print(f"📊 Image size: 12x10 inches, 300 DPI")
        print(f"🎨 Ready for use in LaTeX report!")
        
        # Show the plot
        plt.show()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Please check if the PLY file exists and is readable")

if __name__ == "__main__":
    convert_ply_to_image()




