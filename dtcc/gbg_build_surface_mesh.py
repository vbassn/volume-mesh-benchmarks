import dtcc

# Poseidon (57.6971779, 11.9795910)
x0 = 319995.962899
y0 = 6399009.716755
L = 500.0

# Define bounds
bounds = dtcc.Bounds(x0 - 0.5 * L, y0 - 0.5 * L, x0 + 0.5 * L, y0 + 0.5 * L)

# Download pointcloud and building footprints
pointcloud = dtcc.download_pointcloud(bounds=bounds)
buildings = dtcc.download_footprints(bounds=bounds)

# Remove global outliers
pointcloud = pointcloud.remove_global_outliers(3.0)

# Build terrain raster
raster = dtcc.build_terrain_raster(pointcloud, cell_size=2, radius=3, ground_only=True)

# Extract roof points and compute building heights
buildings = dtcc.extract_roof_points(buildings, pointcloud)
buildings = dtcc.compute_building_heights(buildings, raster, overwrite=True)

# Create city and add geometries
city = dtcc.City()
city.add_terrain(raster)
city.add_buildings(buildings, remove_outside_terrain=True)

# Build surface mesh
mesh = dtcc.build_surface_mesh(city, lod=dtcc.GeometryType.LOD0)

# Offset to origin
mesh.offset_to_origin()

# Save mesh to file
mesh.save("gbg_surface_mesh.pb")
mesh.save("gbg_surface_mesh.xdmf")
