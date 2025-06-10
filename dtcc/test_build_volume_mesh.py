# Test case for volume mesh

import dtcc
from build_volume_mesh import build_volume_mesh

# Set parameters (Helsingborg)
parameters = {}
xmin = 319891
ymin = 6399790
xmax = 319891 + 400.0
ymax = 6399790 + 400.0

# Define bounds
bounds = dtcc.Bounds(xmin, ymin, xmax, ymax)

# Download pointcloud and building footprints
pointcloud = dtcc.download_pointcloud(bounds=bounds)
buildings = dtcc.download_footprints(bounds=bounds)

# Call build_volume_mesh
volume_mesh = build_volume_mesh(pointcloud, buildings, 100.0)

# Save mesh to file
volume_mesh.save("volume_mesh.vtu")
