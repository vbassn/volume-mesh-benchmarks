import dtcc

# Poseidon (57.6971779, 11.9795910)
x0 = 319995.962899
y0 = 6399009.716755
L = 500.0

# Define bounds
bounds = dtcc.Bounds(x0 - 0.5 * L, y0 - 0.5 * L, x0 + 0.5 * L, y0 + 0.5 * L)

# Download pointcloud and building footprints
city = dtcc.City()
city.download_pointcloud(bounds=bounds, filter_on_z_bounds=True)
city.download_footprints(bounds=bounds)

# Compute building heights
city.building_heights_from_pointcloud()

# Build mesh
mesh = city.build_surface_mesh()

# Offset to origin
mesh.offset_to_origin()

# Save mesh to file
mesh.save("gbg_surface_mesh.pb")
mesh.save("gbg_surface_mesh.xdmf")
