# Test case for volume mesh

import dtcc
from build_volume_mesh import build_volume_mesh
from utils import extract_meshes_from_boundary_markers,save_mesh_with_boundary_markers

# Poseidon (57.6971779, 11.9795910)
x0 = 319995.962899
y0 = 6399009.716755
L = 100.0
H = 50.0
h = 5.0

# Define bounds
bounds = dtcc.Bounds(x0 - 0.5 * L, y0 - 0.5 * L, x0 + 0.5 * L, y0 + 0.5 * L)

# Download pointcloud and building footprints
pointcloud = dtcc.download_pointcloud(bounds=bounds)
buildings = dtcc.download_footprints(bounds=bounds)

# Build volume mesh
volume_mesh = build_volume_mesh(pointcloud, buildings, H, h)

# Extract and save boundary meshes
# extract_meshes_from_boundary_markers(volume_mesh, volume_mesh.boundary_markers)

# Save mesh to file
volume_mesh.save("gbg_volume_mesh.vtu")
volume_mesh.save("gbg_volume_mesh.xdmf")

