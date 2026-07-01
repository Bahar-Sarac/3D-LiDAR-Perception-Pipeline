import open3d as o3d
import numpy as np

# ==========================================
# STEP 1: LOAD RAW SYNTHETIC KITTI DATA
# ==========================================
raw_data = np.fromfile(r"C:\PycharmProjects\lidar_pipeline\data\sample_scene.bin", dtype=np.float32)
kitti_points = raw_data.reshape(-1, 4)
xyz_points = kitti_points[:, :3]

print(f"[INFO] Original point cloud count: {xyz_points.shape[0]}")

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
print(f"[INFO] Point count after ROI filtering: {filtered_xyz.shape[0]}")

# ==========================================
# STEP 3: VOXEL DOWNSAMPLING (PERFORMANCE OPTIMIZATION)
# ==========================================
# Convert back to Open3D object to use its professional geometric functions
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(filtered_xyz)

# Downsample the point cloud with a voxel size of 10cm (0.10 meters)
voxel_size = 0.10
pcd_downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
print(f"[INFO] Point count after Voxel Downsampling: {len(pcd_downsampled.points)}")

# ==========================================
# STEP 4: RANSAC GROUND PLANE SEGMENTATION
# ==========================================
# Segment the dominant road plane using mathematical iterative consensus
# distance_threshold: Max distance (in meters) a point can be from the model plane
# ransac_n: Minimum randomly sampled points required to initialize the geometric plane
# num_iterations: Total estimation trials to converge on the absolute optimal consensus
plane_model, inliers = pcd_downsampled.segment_plane(distance_threshold=0.08,
                                                     ransac_n=3,
                                                     num_iterations=200)

# Extract mathematical coefficients for the plane equation: Ax + By + Cz + D = 0
[A, B, C, D] = plane_model
print(f"[MATHEMATICS] Estimated Road Plane Equation: {A:.2f}x + {B:.2f}y + {C:.2f}z + {D:.2f} = 0")

# Segregate the scene into drivable road surface (inliers) and above-ground obstacles (outliers)
road_pcd = pcd_downsampled.select_by_index(inliers)
obstacle_pcd = pcd_downsampled.select_by_index(inliers, invert=True)

print(f"[PERCEPTION] Isolated Drivable Road Points: {len(road_pcd.points)}")
print(f"[PERCEPTION] Isolated Spatial Obstacle Points: {len(obstacle_pcd.points)}")

# ==========================================
# STEP 5: VISUALIZATION
# ==========================================
# Assign high-contrast semantic colors (Road: Red [RGB: 1,0,0] | Obstacles: White [RGB: 1,1,1])
road_pcd.paint_uniform_color([1.0, 0.0, 0.0])
obstacle_pcd.paint_uniform_color([1.0, 1.0, 1.0])

vis = o3d.visualization.Visualizer()
vis.create_window(window_name="LiDAR Perception Pipeline - RANSAC Ground Segmentation", width=1024, height=768)

# Load isolated point geometries into the render buffer
vis.add_geometry(road_pcd)
vis.add_geometry(obstacle_pcd)

# Set professional dark theme
render_option = vis.get_render_option()
render_option.background_color = np.asarray([0.0, 0.0, 0.0])

print("[RENDER] Launching 3D viewport... Displaying segmented road plane and isolated obstacle arrays.")
vis.run()
vis.destroy_window()