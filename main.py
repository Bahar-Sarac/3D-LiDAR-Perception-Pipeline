import open3d as o3d
import numpy as np

# STEP 1: Load raw synthetic KITTI data using absolute path
raw_data = np.fromfile(r"C:\PycharmProjects\lidar_pipeline\data\sample_scene.bin", dtype=np.float32)
print(f"Total number of float values loaded: {len(raw_data)}")

# STEP 2: Reshape the flat array into an (N, 4) matrix
# -1 automatically calculates row count (N), 4 represents [X, Y, Z, Intensity]
kitti_points = raw_data.reshape(-1, 4)
print(f"Point cloud matrix shape (Points, Features): {kitti_points.shape}")

# STEP 3: Slice the matrix to extract only XYZ spatial coordinates
# Open3D only needs geometric positions (first 3 columns: 0, 1, 2)
xyz_points = kitti_points[:, :3]

# STEP 4: Convert NumPy array to Open3D PointCloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz_points)

# STEP 5: Paint all points pure white for a high-contrast dark theme
pcd.paint_uniform_color([1.0, 1.0, 1.0])

# STEP 6: Create and configure the professional 3D visualizer
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="KITTI Synthetic Scene", width=1024, height=768)
vis.add_geometry(pcd)

# Set background to pure black
render_option = vis.get_render_option()
render_option.background_color = np.asarray([0.0, 0.0, 0.0])

# STEP 7: Run visualizer (Press 'Q' on your keyboard to close the window)
print("Opening 3D scene... Use your mouse to rotate, pan, and zoom.")
vis.run()
vis.destroy_window()