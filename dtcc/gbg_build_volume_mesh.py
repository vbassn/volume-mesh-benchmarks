# Test case for volume mesh

import dtcc
from build_volume_mesh import build_volume_mesh

# Poseidon (57.6971779, 11.9795910)
x0 = 319995.962899
y0 = 6399009.716755
L = 200.0
H = 100.0

# Define bounds
bounds = dtcc.Bounds(x0 - 0.5 * L, y0 - 0.5 * L, x0 + 0.5 * L, y0 + 0.5 * L)

# Download pointcloud and building footprints
pointcloud = dtcc.download_pointcloud(bounds=bounds)
buildings = dtcc.download_footprints(bounds=bounds)

# Call build_volume_mesh
volume_mesh = build_volume_mesh(pointcloud, buildings, H)

# Save mesh to file
volume_mesh.save("gbg_volume_mesh.vtu")
volume_mesh.save("gbg_volume_mesh.xdmf")
