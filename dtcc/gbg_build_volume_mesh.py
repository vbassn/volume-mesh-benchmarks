# Test case for volume mesh

import dtcc
from dtcc_core.builder import build_city_volume_mesh

# Poseidon (57.6971779, 11.9795910)
x0 = 319995.962899
y0 = 6399009.716755
L = 500.0
H = 100.0
h = 10.0

# Used for Poisson, Helmholtz
# L = 500.0
# H = 75.0
# h = 4.0

# Used for wave equation
# L = 500.0
# H = 75.0
# h = 8.0

# Used for NS and advection-diffusion
# L = 500.0
# H = 100.0
# h = 4.0

# Define bounds
bounds = dtcc.Bounds(x0 - 0.5 * L, y0 - 0.5 * L, x0 + 0.5 * L, y0 + 0.5 * L)

# Download pointcloud and building footprints
pointcloud = dtcc.download_pointcloud(bounds=bounds)
buildings = dtcc.download_footprints(bounds=bounds)

pointcloud = pointcloud.remove_global_outliers(3)

# Build terrain raster
terrain = dtcc.build_terrain_raster(pointcloud, cell_size=2, radius=3, ground_only=True)

# Extract roof points
footprints = dtcc.extract_roof_points(
    buildings, pointcloud, statistical_outlier_remover=True
)

# Compute building heights
footprints = dtcc.compute_building_heights(footprints, terrain, overwrite=True)

city = dtcc.City()
city.add_buildings(footprints)
city.add_terrain(terrain)

# Build volume mesh
volume_mesh = build_city_volume_mesh(city=city, domain_height=H, max_mesh_size=h)

# Offset to origin
volume_mesh.offset_to_origin()

# Save mesh to file
# volume_mesh.save("gbg_volume_mesh.pb")
# volume_mesh.save("gbg_volume_mesh.xdmf")

volume_mesh.save("dtcc_volume_mesh.vtu")  # For Paraview
