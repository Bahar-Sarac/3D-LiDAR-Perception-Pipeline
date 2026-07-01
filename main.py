import open3d as o3d
import numpy as np

# ==========================================
# STEP 1: LOAD RAW SYNTHETIC KITTI DATA
# ==========================================
raw_data = np.fromfile(r"C:\PycharmProjects\lidar_pipeline\data\sample_scene.bin", dtype=np.float32)
kitti_points = raw_data.reshape(-1, 4)
xyz_points = kitti_points[:, :3]

print(f"Original point cloud count: {xyz_points.shape[0]}")

# ==========================================
# STEP 2: ROI (REGION OF INTEREST) FILTERING
# ==========================================
# Define the 3D driving corridor boundaries for the autonomous vehicle
# X: Left/Right (-5m to +5m), Y: Height (-1.5m to +2m), Z: Forward (0m to 20m)
x_min, x_max = -5.0, 5.0    # 5 meters left, 5 meters right (10m wide corridor)
y_min, y_max = -1.5, 2.0    # From road surface up to 2 meters high
z_min, z_max = 0.0, 20.0    # From vehicle bumper up to 20 meters ahead

# Create a boolean mask using NumPy logical operations
roi_mask = (xyz_points[:, 0] >= x_min) & (xyz_points[:, 0] <= x_max) & \
           (xyz_points[:, 1] >= y_min) & (xyz_points[:, 1] <= y_max) & \
           (xyz_points[:, 2] >= z_min) & (xyz_points[:, 2] <= z_max)

# Apply the mask to slice out only the points within our driving corridor
filtered_xyz = xyz_points[roi_mask]
print(f"Point count after ROI filtering: {filtered_xyz.shape[0]}")

# ==========================================
# STEP 3: VOXEL DOWNSAMPLING (PERFORMANCE OPTIMIZATION)
# ==========================================
# Convert back to Open3D object to use its professional geometric functions
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(filtered_xyz)

# Downsample the point cloud with a voxel size of 10cm (0.10 meters)
# It creates 3D cubes, aggregates dense points, and replaces them with a single centroid point
voxel_size = 0.10
pcd_downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
print(f"Point count after Voxel Downsampling: {len(pcd_downsampled.points)}")

# ==========================================
# STEP 4: VISUALIZATION
# ==========================================
pcd_downsampled.paint_uniform_color([1.0, 1.0, 1.0]) # Professional white points

vis = o3d.visualization.Visualizer()
vis.create_window(window_name="ROI Filtered & Downsampled Scene", width=1024, height=768)
vis.add_geometry(pcd_downsampled)

# Set professional dark theme
render_option = vis.get_render_option()
render_option.background_color = np.asarray([0.0, 0.0, 0.0])

print("Opening filtered 3D scene... Notice the reduced density and cropped boundaries.")
vis.run()
vis.destroy_window()