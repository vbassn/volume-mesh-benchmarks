# Test case for volume mesh

import dtcc
from dtcc_core.builder import build_volume_mesh
from dtcc_core.builder.model_conversion import volume_mesh_to_builder_volume_mesh
#from build_volume_mesh import build_volume_mesh

# Set parameters (Helsingborg)
parameters = {}
dx = 200
dy = 200
xmin = 319891
ymin = 6399790
xmax = 319891 + dx
ymax = 6399790 + dy

# Define bounds
bounds = dtcc.Bounds(xmin, ymin, xmax, ymax)

# Download pointcloud and building footprints
pointcloud = dtcc.download_pointcloud(bounds=bounds)
buildings = dtcc.download_footprints(bounds=bounds)


pointcloud = pointcloud.remove_global_outliers(3)

# Build terrain raster
terrain = dtcc.build_terrain_raster(
    pointcloud, cell_size=2, radius=3, ground_only=True
)

    # Extract roof points
footprints = dtcc.extract_roof_points(
    buildings, pointcloud, statistical_outlier_remover=True
)

# Compute building heights
footprints = dtcc.compute_building_heights(footprints, terrain, overwrite=True)


city = dtcc.City()
city.add_buildings(footprints)
city.add_terrain(terrain)

print("City type:", type(city.buildings[0]))

# # Call build_volume_mesh
volume_mesh = build_volume_mesh(city, 100.0)

# volume_mesh = build_volume_mesh(
#     pointcloud=pointcloud,
#     buildings=buildings,
#     domain_height=100.0,
#     max_mesh_size=10.0,
# )

print("Volume mesh created successfully with the following properties:")
print(f"Number of cells: {volume_mesh.cells.shape[0]}")
print(f"Number of points: {volume_mesh.vertices.shape[0]}")
print(f"Number of markers: {volume_mesh.markers.shape[0]}")

max_top_height = 0.0
min_top_height = float('inf')
for i, marker in enumerate(volume_mesh.markers):
    if marker == -2 or marker == -1:
        top_height = volume_mesh.vertices[i, 2]
        max_top_height = max(max_top_height, top_height)
        min_top_height = min(min_top_height, top_height)
print(f"Max top height: {max_top_height}")
print(f"Min top height: {min_top_height}")
        
# Save mesh to file
volume_mesh.save("volume_mesh.xdmf")
