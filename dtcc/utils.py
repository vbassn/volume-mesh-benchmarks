from dtcc import Mesh, VolumeMesh
import numpy as np
from typing import Optional, Sequence
import meshio
import re

_MESH_PATTERN = re.compile(
    # r"^MESH_REPORT"
    r"\s+step=(?P<step>\d+)"
    r"\s+vertices=(?P<vertices>\d+)"
    r"\s+cells=(?P<cells>\d+)"
    r"\s+time_s=(?P<time>[\d.]+)"
    r"\s+min_ar=(?P<min_ar>[\d.]+)"
    r"\s+median_ar=(?P<median_ar>[\d.]+)"
    r"\s+max_ar=(?P<max_ar>[\d.]+)"
)

def parse_mesh_reports(text: str) -> dict[int, dict[str, float | int]]:
    results: dict[int, dict[str, float | int]] = {}
    for line in text.splitlines():
        m = _MESH_PATTERN.match(line)
        if not m:
            continue
        gd = m.groupdict()
        step = int(gd.pop("step"))
        results[step] = {
            "vertices": int(gd["vertices"]),
            "cells":    int(gd["cells"]),
            "time_s":   float(gd["time"]),
            "min_ar":   float(gd["min_ar"]),
            "median_ar":float(gd["median_ar"]),
            "max_ar":   float(gd["max_ar"]),
        }
    return results


def parse_mesh_report_txt(filename):
    """
    Parse a mesh_report.txt file and return a dictionary of metrics keyed by metric_step.
    For example, 'vertices_2': 45405, 'time_s_3': 1.319, etc.
    """
    data = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # split into key=value tokens
            tokens = line.split()
            step = None
            # first pass: find the step value
            for tok in tokens:
                if tok.startswith('step='):
                    _, step = tok.split('=', 1)
                    break
            if step is None:
                continue
            # second pass: parse other metrics
            for tok in tokens:
                key, val = tok.split('=', 1)
                if key == 'step':
                    continue
                # build dict key with step suffix
                dict_key = f"{key}_{step}"
                # convert to int or float
                try:
                    if key in ('vertices', 'cells'):
                        data[dict_key] = int(val)
                    else:
                        data[dict_key] = float(val)
                except ValueError:
                    data[dict_key] = val
    return data



def extract_surface_mesh(
    volume_mesh: VolumeMesh,
    face_list: Sequence[Sequence[int]],
    face_markers: Optional[Sequence] = None
) -> Mesh:
    """
    Extract a surface mesh from a tetrahedral volume mesh.

    Parameters
    ----------
    volume_mesh : VolumeMesh
        The input tetrahedral mesh, whose `vertices` array will be sampled.
    face_list : Sequence of (3,) int sequences
        Each entry is a triplet of indices into volume_mesh.vertices, defining one triangular face.
    face_markers : optional Sequence
        If provided, must have the same length as face_list; used to populate Mesh.markers.
        Otherwise markers default to zero.

    Returns
    -------
    Mesh
        A triangular surface mesh: only the vertices actually used by face_list are kept,
        and face indices are re‐numbered to index into the new, smaller vertex array.
    """
    # (M,3) array of face‐vertex indices
    faces = np.asarray(face_list, dtype=np.int64).reshape(-1, 3)

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
    if face_markers is not None:
        markers = np.asarray(face_markers)
        if markers.shape[0] != surface_faces.shape[0]:
            raise ValueError("face_markers must have same length as face_list")
    else:
        # default to zeros
        markers = np.zeros(surface_faces.shape[0], dtype=np.int64
                            if hasattr(volume_mesh, "markers") else np.int64)

    # 5) Build and return Mesh
    return Mesh(vertices=surface_vertices,
                faces=surface_faces,
                markers=markers)



def extract_meshes_from_boundary_markers(volume_mesh, boundary_markers, save_extracted_meshes=False):

    unique_markers = set(boundary_markers.values())
    print(f"Unique boundary markers: {unique_markers}")
    num_buildings = (np.max(list(unique_markers)) +1)/2
    ground_faces = []
    top_faces = []
    building_faces = {key : [] for key in range(int(num_buildings))}
    north_faces = []
    south_faces = []
    east_faces = []
    west_faces = []
    for face,marker in boundary_markers.items():
        if marker == -1:
            ground_faces.append(face)
        elif marker == -2:
            top_faces.append(face)
        elif marker == -3:
            north_faces.append(face)
        elif marker == -4:
            east_faces.append(face)
        elif marker == -5:
            south_faces.append(face)
        elif marker == -6:
            west_faces.append(face)
        elif marker >= 0:
            building_faces[marker % num_buildings].append(face)
    
    extracted_meshes = {}
    if ground_faces:
        extracted_meshes["ground"] = extract_surface_mesh(volume_mesh, ground_faces, face_markers=[-1]*len(ground_faces))
    if top_faces:
        extracted_meshes["top"] = extract_surface_mesh(volume_mesh, top_faces, face_markers=[-2]*len(top_faces))
    if north_faces:
        extracted_meshes["north"] = extract_surface_mesh(volume_mesh, north_faces, face_markers=[-3]*len(north_faces))
    if south_faces:
        extracted_meshes["south"] = extract_surface_mesh(volume_mesh, south_faces, face_markers=[-4]*len(south_faces))
    if east_faces:
        extracted_meshes["east"] = extract_surface_mesh(volume_mesh, east_faces, face_markers=[-5]*len(east_faces))
    if west_faces:
        extracted_meshes["west"] = extract_surface_mesh(volume_mesh, west_faces, face_markers=[-6]*len(west_faces))
    if building_faces:
        for building, faces in building_faces.items():
            if faces:
                extracted_meshes[f"building_{building}"] = extract_surface_mesh(volume_mesh, faces,)


    if save_extracted_meshes:
        for name, mesh in extracted_meshes.items():
            mesh.save(f"boundary_mesh_{name}.vtu")
   


def save_mesh_with_boundary_markers(volume_mesh, filename):
    # Save the mesh to a file
    boundary_faces = np.array(
        list(volume_mesh.boundary_markers.keys()),
        dtype=int
    )
    # and the corresponding marker values to an (F,) int array
    boundary_markers = np.array(
        list(volume_mesh.boundary_markers.values()),
        dtype=int
    )

    # --- Ensure the tetra cells are an (M,4) int array ---
    tetra_cells = np.asarray(volume_mesh.cells, dtype=int)

    # --- Create a dummy marker array for the tetra block ---
    tetra_markers = np.full(len(tetra_cells), np.nan, dtype=int)

    # --- Build the meshio.Mesh object ---
    mesh = meshio.Mesh(
        points=np.asarray(volume_mesh.vertices, dtype=float),
        cells=[
            ("tetra", tetra_cells),
            ("triangle", boundary_faces),
        ],
        cell_data={
            # one array per block, in the same order as `cells`
            "mesh_tags": [
                tetra_markers,
                boundary_markers,
            ]
        }
    )

    meshio.write(filename, mesh)


# # -----------------------------------------------------------------
# # assume you already have
# # points        : (n_points, gdim) float64 array
# # cells_tet     : (n_tets, 4)      int   array      – the volume mesh
# # facet_tris    : (n_facets, 3)    int   array      – boundary facets
# # facet_marker  : (n_facets,)      int   array      – your markers
# # -----------------------------------------------------------------

# # 1.  write the 3-D volume grid (tetrahedra) exactly as you do now
# volume_mesh = meshio.Mesh(points, [("tetra", cells_tet)])
# meshio.write("mesh.xdmf", volume_mesh)          # creates mesh.xdmf + mesh.h5

# # 2.  append one more Grid that holds the facets + markers
# facet_mesh = meshio.Mesh(
#     points,
#     [("triangle", facet_tris)],                 # <— correct cell-type!
#     cell_data={"values": [facet_marker.astype(np.int32)]},
# )

# # meshio's XDMF helper lets us append extra grids to the same file
# with meshio.xdmf.TimeSeriesWriter("mesh.xdmf") as writer:
#     # we don't want a time series – but this helper is the easiest
#     writer.write_points_cells(
#         points,
#         [("triangle", facet_tris)],
#         cell_data={"values": [facet_marker.astype(np.int32)]},
#         grid_name="facet_markers",              # <— becomes the Grid Name
#     )