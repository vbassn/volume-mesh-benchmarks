import numpy as np
import os
import h5py
from typing import Optional, Sequence
from collections import defaultdict


import dtcc
import dtcc_core.builder.register


@dtcc_core.builder.register.register_model_method
def boundary_markers_info(volume_mesh: dtcc.VolumeMesh):
    print("Volume Mesh Info:")
    print(f"Volume Mesh with {len(volume_mesh.vertices)} vertices and {len(volume_mesh.cells)} cells")

def _extract_surface_mesh(
    volume_mesh: dtcc.VolumeMesh,
    boundary_faces: Sequence[Sequence[int]],
    boundary_markers: Optional[Sequence] = None
) -> dtcc.Mesh:
    """
    Extract a surface mesh from a tetrahedral volume mesh.

    Parameters
    ----------
    volume_mesh : VolumeMesh
        The input tetrahedral mesh, whose `vertices` array will be sampled.
    boundary_faces : Sequence of (3,) int sequences
        Each entry is a triplet of indices into volume_mesh.vertices, defining one triangular face.
    boundary_markers : optional Sequence
        If provided, must have the same length as boundary_faces; used to populate Mesh.markers.
        Otherwise markers default to zero.

    Returns
    -------
    Mesh
        A triangular surface mesh: only the vertices actually used by boundary_faces are kept,
        and face indices are re‐numbered to index into the new, smaller vertex array.
    """
    # (M,3) array of face‐vertex indices
    faces = np.asarray(boundary_faces, dtype=np.int64).reshape(-1, 3)

    # 1) Find all unique vertex‐indices used by these faces,
    #    and build a mapping old_idx -> new_idx
    unique_vids, inverse = np.unique(faces.flatten(), return_inverse=True)
    # unique_vids is shape (U,), inverse is length 3*M
    print(unique_vids.shape, inverse.shape)
    # 2) Extract only those vertices from the volume mesh
    volume_vertices = np.asarray(volume_mesh.vertices)
    surface_vertices = volume_vertices[unique_vids]
    # surface_vertices = np.array([(v.x, v.y, v.z) for v in surface_vertices])
    # 3) Re‐shape inverse to (M,3) to get the new faces
    surface_faces = inverse.reshape(-1, 3)

    # 4) Handle markers
    if boundary_markers is not None:
        markers = np.asarray(boundary_markers)
        if markers.shape[0] != surface_faces.shape[0]:
            raise ValueError("boundary_markers must have same length as boundary_faces")
    else:
        # default to zeros
        markers = np.zeros(surface_faces.shape[0], dtype=np.int64
                            if hasattr(volume_mesh, "markers") else np.int64)

    # 5) Build and return Mesh
    return dtcc.Mesh(vertices=surface_vertices,
                faces=surface_faces,
                markers=markers)

@dtcc_core.builder.register.register_model_method
def extract_meshes_from_boundary_markers(volume_mesh: dtcc.VolumeMesh, write_meshes: bool = False) -> dict[str, dtcc.Mesh]:

    if not isinstance(volume_mesh, dtcc.VolumeMesh):
        raise TypeError("Expected a dtcc.VolumeMesh instance.")
    
    if not hasattr(volume_mesh, "boundary_markers"):
        print("No boundary markers found in the volume mesh.")
        return

    unique_markers = set(volume_mesh.boundary_markers.values())
    num_buildings = (max(unique_markers) +1)/2

    # Mapping of fixed marker values to names
    marker_map = {
        -1: 'ground',
        -2: 'top',
        -3: 'north',
        -4: 'east',
        -5: 'south',
        -6: 'west'
    }

    groups = defaultdict(list)
    for face, m in volume_mesh.boundary_markers.items():
        if m in marker_map:
            groups[marker_map[m]].append((face, m))
        elif m >= 0:
            idx = m % num_buildings
            groups[f'building_{idx}'].append((face, m))
    
    extracted_meshes = {}
    for name, pairs in groups.items():
        faces, vals = zip(*pairs)
        kwargs = {'boundary_markers': list(vals)} if name in marker_map.values() else {}
        extracted_meshes[name] = _extract_surface_mesh(volume_mesh, list(faces), **kwargs)

    if write_meshes:
        for name, mesh in extracted_meshes.items():
            mesh.save(f"boundary_mesh_{name}.vtu")

    return extracted_meshes


@dtcc_core.builder.register.register_model_method
def save_boundary_markers(volume_mesh: dtcc.VolumeMesh, filename: str | None = None):
   
    if not hasattr(volume_mesh, "boundary_markers"):
        raise ValueError("Volume mesh does not contain boundary markers.")
    
    if filename is None:
        filename = "boundary_markers.h5"

    base, ext = os.path.splitext(filename)
    if ext.lower() != ".h5":
        filename = base + ".h5"
    faces = np.array(list(volume_mesh.boundary_markers.keys()), dtype=np.int64)
    markers = np.array(list(volume_mesh.boundary_markers.values()), dtype=np.int64)
    
    with h5py.File(filename, 'w') as f:
        # Store mesh sizes as file attributes
        f.attrs['num_vertices'] = volume_mesh.vertices.shape[0]
        f.attrs['num_cells'] = volume_mesh.cells.shape[0]
        # Create group and datasets
        grp = f.create_group('boundary_faces')
        grp.create_dataset('faces', data=faces)
        grp.create_dataset('markers', data=markers)

    dtcc.info("Saving boundary markers to: " + filename)

@dtcc_core.builder.register.register_model_method
def load_boundary_markers(volume_mesh: dtcc.VolumeMesh, filename: str):
    # Ensure .h5 suffix
    base, ext = os.path.splitext(filename)
    if ext.lower() != ".h5":
        filename = base + ".h5"

    dtcc.info("Loading boundary markers from: " + filename)
    # Load HDF5
    with h5py.File(filename, 'r') as f:
        # Read and check mesh sizes
        file_num_vertices = f.attrs.get('num_vertices')
        file_num_cells = f.attrs.get('num_cells')
        mesh_num_vertices = volume_mesh.vertices.shape[0]
        mesh_num_cells = volume_mesh.cells.shape[0]

        if file_num_vertices != mesh_num_vertices:
            raise ValueError(
                f"Vertex count mismatch: file has {file_num_vertices}, mesh has {mesh_num_vertices}"
            )
        if file_num_cells != mesh_num_cells:
            raise ValueError(
                f"Cell count mismatch: file has {file_num_cells}, mesh has {mesh_num_cells}"
            )

        # Read boundary faces and markers
        grp = f['boundary_faces']
        faces = np.array(grp['faces'], dtype=np.int64)
        markers = np.array(grp['markers'], dtype=str)

    # Reconstruct boundary_markers dict
    boundary_markers = {tuple(face): marker for face, marker in zip(faces, markers)}
    volume_mesh.boundary_markers = boundary_markers