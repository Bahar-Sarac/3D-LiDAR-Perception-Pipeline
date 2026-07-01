import os
import numpy as np

# Ensure the "data" directory exists
# This directory is ignored by Git thanks to our .gitignore configuration
os.makedirs("data", exist_ok=True)
print("Generating advanced synthetic KITTI data (Car + Cylinder Container)...")

# STEP 1: GENERATE ROAD PLANE
x_road = np.linspace(-10, 10, 250)
z_road = np.linspace(2, 25, 250)
X_road, Z_road = np.meshgrid(x_road, z_road)
Y_road = np.zeros_like(X_road) - 1.0  # Road surface at Y = -1.0 meter
Y_road += np.random.normal(0, 0.015, Y_road.shape)  # Realistic sensor noise

road_points = np.vstack([X_road.ravel(), Y_road.ravel(), Z_road.ravel()]).T

# STEP 2: GENERATE CAR (3D Box Cluster)
# Positioned on the left side: X: -3 to -1, Z: 10 to 13
x_car = np.linspace(-3, -1, 25)
y_car = np.linspace(-1, 0.4, 15)
z_car = np.linspace(10, 13, 25)
X_c, Y_c, Z_c = np.meshgrid(x_car, y_car, z_car)
car_points = np.vstack([X_c.ravel(), Y_c.ravel(), Z_c.ravel()]).T

# STEP 3: GENERATE CYLINDRICAL CONTAINER (Cylinder Cluster)
# Positioned on the right side: Center at X = 3, Z = 15
# Radius = 0.6 meters, Height = from Y: -1 to 0.8 (1.8 meters tall)
num_slices = 30   # Angular resolution
num_stacks = 20   # Height resolution
num_rings = 10    # Interior/lid density resolution

cylinder_points = []

# Generate the outer wall of the cylinder using polar coordinates
for h in np.linspace(-1.0, 0.8, num_stacks):
    for theta in np.linspace(0, 2*np.pi, num_slices):
        r = 0.6
        x = 3.0 + r * np.cos(theta)
        z = 15.0 + r * np.sin(theta)
        cylinder_points.append([x, h, z])

# Fill the top lid to make it look like a solid industrial container
for r_fill in np.linspace(0, 0.6, num_rings):
    for theta in np.linspace(0, 2*np.pi, num_slices):
        x = 3.0 + r_fill * np.cos(theta)
        z = 15.0 + r_fill * np.sin(theta)
        cylinder_points.append([x, 0.8, z]) # Top lid height

container_points = np.array(cylinder_points)

# STEP 4: MERGE AND SAVE IN KITTI FORMAT
all_xyz = np.vstack([road_points, car_points, container_points])

# Add 4th dimension (Intensity) required by KITTI format
intensity = np.ones((all_xyz.shape[0], 1), dtype=np.float32) * 0.5
kitti_format_data = np.hstack([all_xyz, intensity]).astype(np.float32)

# Save to binary file locally
output_path = "data/sample_scene.bin"
kitti_format_data.tofile(output_path)

print(f"Success! Advanced scene saved to: {output_path}")
print(f"Total points generated: {kitti_format_data.shape[0]}")