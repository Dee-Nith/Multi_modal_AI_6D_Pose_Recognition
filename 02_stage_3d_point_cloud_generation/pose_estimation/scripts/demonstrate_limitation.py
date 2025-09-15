#!/usr/bin/env python3
"""
Demonstrate the fundamental limitation of single-view RGB-D
vs complete 3D object models
"""

import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from pathlib import Path

def load_ycb_object_model(object_name="master_chef_can"):
    """Load complete YCB 3D object model"""
    print(f"🎯 Looking for complete 3D model of {object_name}...")
    
    # Check if we have YCB models
    possible_paths = [
        f"ycb_models/{object_name}/google_16k/textured.obj",
        f"ycb_models/{object_name}/textured.obj",
        f"{object_name}.obj"
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            try:
                mesh = o3d.io.read_triangle_mesh(path)
                if len(mesh.vertices) > 0:
                    print(f"✅ Loaded complete model: {len(mesh.vertices)} vertices")
                    return mesh
            except:
                continue
    
    print("⚠️ Complete 3D model not found, creating synthetic model")
    return create_synthetic_object()

def create_synthetic_object():
    """Create a synthetic 3D object to demonstrate the concept"""
    # Create a cylinder (like a can)
    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=0.03, height=0.1)
    mesh.paint_uniform_color([0.8, 0.2, 0.2])  # Red color
    mesh.compute_vertex_normals()
    print(f"✅ Created synthetic cylinder: {len(mesh.vertices)} vertices")
    return mesh

def simulate_single_view_capture(mesh, view_angle=0):
    """Simulate what a single RGB-D camera would see"""
    print(f"\n📷 Simulating single RGB-D view at {view_angle}° angle...")
    
    # Create camera pose
    camera_distance = 0.3
    angle_rad = np.radians(view_angle)
    camera_pos = [camera_distance * np.sin(angle_rad), 0, camera_distance * np.cos(angle_rad)]
    
    # Convert mesh to point cloud
    full_pcd = mesh.sample_points_uniformly(number_of_points=10000)
    
    # Simulate camera field of view and occlusion
    points = np.asarray(full_pcd.points)
    colors = np.asarray(full_pcd.colors)
    
    # Camera intrinsic simulation
    camera_matrix = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ])
    
    # Transform points to camera frame
    camera_transform = np.eye(4)
    camera_transform[:3, 3] = camera_pos
    
    # Simple visibility check (front-facing points only)
    visible_mask = points[:, 2] > 0.1  # Points in front of camera
    
    # Simulate limited field of view
    projected_points = points @ camera_matrix.T
    x_in_view = np.abs(projected_points[:, 0] / projected_points[:, 2]) < 320
    y_in_view = np.abs(projected_points[:, 1] / projected_points[:, 2]) < 240
    in_view_mask = x_in_view & y_in_view & (projected_points[:, 2] > 0)
    
    # Combine visibility constraints
    final_mask = visible_mask & in_view_mask
    
    # Create single-view point cloud
    visible_pcd = o3d.geometry.PointCloud()
    visible_pcd.points = o3d.utility.Vector3dVector(points[final_mask])
    visible_pcd.colors = o3d.utility.Vector3dVector(colors[final_mask])
    
    print(f"   📊 Complete object: {len(full_pcd.points)} points")
    print(f"   📊 Single view sees: {len(visible_pcd.points)} points")
    print(f"   📊 Visibility: {len(visible_pcd.points)/len(full_pcd.points)*100:.1f}%")
    
    return full_pcd, visible_pcd

def demonstrate_rgbd_limitation():
    """Demonstrate the fundamental limitation of RGB-D vs complete models"""
    print("🎯 Demonstrating RGB-D Limitation vs Complete 3D Models")
    print("=" * 60)
    
    # Load or create complete 3D object
    complete_mesh = load_ycb_object_model()
    
    # Simulate multiple single views
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    single_views = []
    
    for angle in angles:
        full_pcd, visible_pcd = simulate_single_view_capture(complete_mesh, angle)
        single_views.append(visible_pcd)
    
    # Create visualization
    fig = plt.figure(figsize=(20, 15))
    
    # Complete 3D model
    plt.subplot(3, 3, 1)
    full_points = np.asarray(full_pcd.points)
    full_colors = np.asarray(full_pcd.colors)
    plt.scatter(full_points[:, 0], full_points[:, 1], c=full_colors, s=1, alpha=0.6)
    plt.title('Complete 3D Object Model\n(What we want to achieve)', fontsize=12, fontweight='bold')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.axis('equal')
    
    # Single views
    for i, (angle, view_pcd) in enumerate(zip(angles, single_views)):
        plt.subplot(3, 3, i + 2)
        if len(view_pcd.points) > 0:
            view_points = np.asarray(view_pcd.points)
            view_colors = np.asarray(view_pcd.colors)
            plt.scatter(view_points[:, 0], view_points[:, 1], c=view_colors, s=2, alpha=0.8)
        plt.title(f'Single RGB-D View\n{angle}° ({len(view_pcd.points)} pts)', fontsize=10)
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig('rgbd_limitation_demo.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save comparison point clouds
    o3d.io.write_point_cloud("complete_object.ply", full_pcd)
    o3d.io.write_point_cloud("single_view_example.ply", single_views[0])
    
    print(f"\n📊 Key Insights:")
    print(f"   🔹 Complete model: {len(full_pcd.points)} points (100%)")
    print(f"   🔹 Single view: {len(single_views[0].points)} points ({len(single_views[0].points)/len(full_pcd.points)*100:.1f}%)")
    print(f"   🔹 Missing data: {100 - len(single_views[0].points)/len(full_pcd.points)*100:.1f}%")
    
    print(f"\n💡 Why AI like MCC is Revolutionary:")
    print(f"   ✨ Predicts HIDDEN surfaces using learned patterns")
    print(f"   ✨ Infers complete geometry from partial observations")
    print(f"   ✨ Uses multi-view consistency and shape priors")
    print(f"   ✨ Goes beyond physical sensor limitations")
    
    return full_pcd, single_views

def main():
    """Main demonstration"""
    full_pcd, single_views = demonstrate_rgbd_limitation()
    
    print(f"\n🎮 Opening interactive comparison...")
    print(f"   Blue: Complete 3D object model")
    print(f"   Red: What single RGB-D view captures")
    
    # Color code for comparison
    full_pcd.paint_uniform_color([0.2, 0.4, 0.8])  # Blue
    single_views[0].paint_uniform_color([0.8, 0.2, 0.2])  # Red
    
    # Show side by side
    o3d.visualization.draw_geometries([full_pcd, single_views[0]], 
                                    window_name="RGB-D Limitation: Blue=Complete, Red=Single View")

if __name__ == "__main__":
    main()




