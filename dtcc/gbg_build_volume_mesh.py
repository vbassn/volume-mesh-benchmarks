# Test case for volume mesh

import dtcc
from dtcc_core.builder import build_volume_mesh


# Poseidon (57.6971779, 11.9795910)
x0 = 319995.962899
y0 = 6399009.716755
L = 100.0
H = 50.0
h = 2.0

# Define bounds
bounds = dtcc.Bounds(x0 - 0.5 * L, y0 - 0.5 * L, x0 + 0.5 * L, y0 + 0.5 * L)

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

# Build volume mesh
volume_mesh = build_volume_mesh(city=city, 
                                domain_height=H, 
                                max_mesh_size=h)

# Extract and save boundary meshes
# extract_meshes_from_boundary_markers(volume_mesh, volume_mesh.boundary_markers)

# Save mesh to file
volume_mesh.save("gbg_volume_mesh.vtu")
volume_mesh.save("gbg_volume_mesh.xdmf")

