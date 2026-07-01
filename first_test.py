import open3d as o3d
import numpy as np

# Step 1: Generate 1000 random points in 3D space (X, Y, Z coordinates)
np_points = np.random.rand(1000, 3) * 2 - 1

# Step 2: Convert the raw NumPy matrix into an Open3D "PointCloud" object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np_points)

# Step 3: Paint all points pure white [1, 1, 1]
pcd.paint_uniform_color([1.0, 1.0, 1.0])

# Step 4: Create a Visualizer object to customize background color
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Black Background - White Points", width=800, height=600)

# Step 5: Add the point cloud to the visualizer
vis.add_geometry(pcd)

# Step 6: Get render options and set the background color to pure black [0, 0, 0]
render_option = vis.get_render_option()
render_option.background_color = np.asarray([0.0, 0.0, 0.0])

# Step 7: Run the visualizer and destroy the window when closed (Press 'Q')
print("Opening the customized 3D window... Press 'Q' to close.")
vis.run()
vis.destroy_window()