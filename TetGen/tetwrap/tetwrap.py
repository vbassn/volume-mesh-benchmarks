from ._tetwrap import build_volume_mesh as _build_volume_mesh
import dtcc
from . import switches
import numpy as np
from .mesh_boundary_facets import compute_boundary_facets



def build_volume_mesh(mesh: dtcc.Mesh, build_top_sidewalls: bool=True, top_height: float=100.0, switches_params=None, switches_overrides=None) -> dtcc.VolumeMesh:
    """
    Build a volume mesh from a surface mesh using TetGen via tetwrap.

    Parameters
    ----------
    mesh : dtcc.VolumeMesh
        The input surface mesh.
    switches_params : dict, optional
        Parameters for TetGen switches.
    switches_overrides : dict, optional
        Overrides for TetGen switches.

    Returns
    -------
    vertices : ndarray
        The vertices of the volume mesh.
    cells : ndarray
        The cells (tetrahedra) of the volume mesh.
    """
    if not isinstance(mesh, dtcc.Mesh):
        raise TypeError("Input must be a dtcc.Mesh instance.")

    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError("Input mesh must have faces defined.")
    
    if build_top_sidewalls:
        new_vertices, boundary_facets = compute_boundary_facets(mesh, top_height=top_height )
        mesh = dtcc.Mesh(vertices=new_vertices, faces=mesh.faces)
        b_facets = []
        for key, facet in boundary_facets.items():
            b_facets.append(facet)
            # print(f"Facet {key}: {facet}")

    # Prepare TetGen switches
    tetgen_switches = switches.build_tetgen_switches(params=switches_params, overrides=switches_overrides)
    
    # Call tetwrap to build the volume mesh
    vertices, cells = _build_volume_mesh(mesh.vertices, mesh.faces,b_facets, tetgen_switches)

    return dtcc.VolumeMesh(vertices=vertices, cells=cells)