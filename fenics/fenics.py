"Simple FEniCS style FEniCSx wrapper"

__all__ = []

# --- MPI imports --
from mpi4py import MPI

__all__.extend(["MPI"])

# --- Numpy imports
import numpy as np

__all__.extend(["np"])

# --- UFL imports ---
from ufl import TrialFunction, TestFunction, SpatialCoordinate
from ufl import dx, ds, inner, grad
from ufl import exp, sin, cos

__all__.extend(
    [
        "TrialFunction",
        "TestFunction",
        "SpatialCoordinate",
        "dx",
        "ds",
        "inner",
        "grad",
        "exp",
        "sin",
        "cos",
    ]
)

# --- DOLFINx io imports ---
from dolfinx.io import XDMFFile

__all__.extend(["XDMFFile"])

# --- DOLFINx fem imports ---
from dolfinx.fem import Constant, Function, functionspace
from dolfinx.fem.petsc import LinearProblem

FunctionSpace = lambda mesh, family, degree: functionspace(mesh, (family, degree))

__all__.extend(["Constant", "Function", "FunctionSpace", "LinearProblem"])

# --- DOLFINx log imports ---
from dolfinx.log import log, set_log_level, LogLevel

DEBUG = LogLevel.DEBUG
INFO = LogLevel.INFO
WARNING = LogLevel.WARNING
ERROR = LogLevel.ERROR

info = lambda message: log(INFO, message)

__all__.extend(
    ["info", "set_log_level", "LogLevel", "DEBUG", "INFO", "WARNING", "ERROR"]
)

# --- DOLFINx boundary conditions ---
from dolfinx.mesh import locate_entities_boundary, meshtags
from dolfinx.fem import locate_dofs_topological, dirichletbc
from ufl import Measure
from petsc4py.PETSc import ScalarType


def DirichletBC(V, value, marker):
    """Create a Dirichlet boundary condition."""

    # Get mesh and topological dimension
    mesh = V.mesh
    tdim = V.mesh.topology.dim

    # Extract facets and dofs
    facets = locate_entities_boundary(mesh, dim=tdim - 1, marker=marker)
    dofs = locate_dofs_topological(V=V, entity_dim=tdim - 1, entities=facets)

    # Create boundary condition
    bc = dirichletbc(value=ScalarType(value), dofs=dofs, V=V)

    return bc


def NeumannBC(mesh, marker, tag=None, facet_tags=None):
    """
    Mark facets selected by *marker* and return a UFL surface
    measure restricted to them.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
    marker : callable  f(x) -> bool array
        Geometric predicate evaluated on facet mid-points exactly like
        in locate_entities_boundary.
    tag : int, optional
        Integer tag id to use for the new facets.  If None we pick
        max(existing)+1 (or 1 if no previous tags).
    facet_tags : dolfinx.mesh.MeshTags | None
        Existing facet MeshTags to extend.  If you pass in a tags
        object, the new facets are appended so you can mix several
        boundary types.

    Returns
    -------
    ds : UFL measure
    """

    # Get topological dimension
    tdim = mesh.topology.dim

    # Extract  facets
    facets = locate_entities_boundary(mesh, dim=tdim - 1, marker=marker)

    # Choose a tag id
    if tag is None:
        tag = 1
        if facet_tags is not None and facet_tags.values.size > 0:
            tag = int(facet_tags.values.max()) + 1

    # Build / extend MeshTags
    new_vals = np.full(facets.size, tag, dtype=np.int32)
    if facet_tags is None:
        facet_tags = meshtags(mesh, tdim - 1, facets, new_vals)
    else:
        all_facets = np.hstack([facet_tags.indices, facets])
        all_vals = np.hstack([facet_tags.values, new_vals])
        facet_tags = meshtags(mesh, tdim - 1, all_facets, all_vals)

    # Create surface measure
    ds = Measure("ds", domain=mesh, subdomain_data=facet_tags)
    ds = ds(tag)

    return ds


__all__.extend(["DirichletBC", "NeumannBC"])

# --- Numpy imports ---

from numpy import isclose as near

__all__.extend(["near"])

# --- Plotting ---
import pyvista
import dolfinx


def plot(u, show=True):

    V = u.function_space
    cells, types, x = dolfinx.plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(cells, types, x)
    grid.point_data["u"] = u.x.array.real
    grid.set_active_scalars("u")
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    warped = grid.warp_by_scalar()
    plotter.add_mesh(warped)

    if show:
        plotter.show()

    return plotter


__all__.extend(["plot"])


# --- Saving and loading ---


def load_mesh(filename):
    """Load a mesh from an XDMF file."""
    if not filename.endswith(".xdmf"):
        raise ValueError("Filename must end with .xdmf")

    comm = MPI.COMM_WORLD
    with XDMFFile(comm, filename, "r") as xdmf_file:
        mesh = xdmf_file.read_mesh(name="Grid")

    return mesh


__all__.extend(["load_mesh"])


def _save_mesh(mesh, filename):
    """Save dolfinx Mesh to an XDMF file.

    Args:
        mesh (dolfinx.mesh.Mesh): The mesh to save.
        filename (str): Output XDMF filename (should end with .xdmf).
    """

    info(f"Saving mesh to file {filename}")

    if not filename.endswith(".xdmf"):
        raise ValueError("Filename must end with .xdmf")

    with XDMFFile(mesh.comm, filename, "w") as xdmf_file:
        xdmf_file.write_mesh(mesh)


def _save_function(self, filename, t=None):
    """Save Function to an XDMF file (supports time series).

    Args:
        filename (str): Output XDMF filename (should end with .xdmf).
        t (float, optional): Current simulation time. If given, writes a time series.
                             Mesh is written only once when t == 0.
    """

    info(f"Saving function to file {filename}")

    if not filename.endswith(".xdmf"):
        raise ValueError("Filename must end with .xdmf")

    mesh = self.function_space.mesh
    mode = "a" if (t is not None and t > 0) else "w"

    with XDMFFile(mesh.comm, filename, mode) as xdmf_file:
        if t is None or t == 0:
            xdmf_file.write_mesh(mesh)
        if t is None:
            xdmf_file.write_function(self)
        else:
            xdmf_file.write_function(self, t)


dolfinx.mesh.Mesh.save = _save_mesh
dolfinx.fem.Function.save = _save_function


# --- Meshes ---


def BoxMesh(xmin, ymin, zmin, xmax, ymax, zmax, nx, ny, nz):
    """
    Create a box mesh."
    """

    domain = [(xmin, ymin, zmin), (xmax, ymax, zmax)]
    mesh = dolfinx.mesh.create_box(
        comm=MPI.COMM_WORLD,
        points=domain,
        n=(nx, ny, nz),
        cell_type=dolfinx.mesh.CellType.tetrahedron,
    )

    return mesh


__all__.extend(["BoxMesh"])


# --- Mesh bounds and shifting ---


def bounds(mesh):
    """
    Compute global mesh bounds in parallel (3D).

    Returns:
        xmin, ymin, zmin, xmax, ymax, zmax
    """
    comm = mesh.comm
    coords = mesh.geometry.x

    local_min = coords.min(axis=0)
    local_max = coords.max(axis=0)

    global_min = np.zeros(3)
    global_max = np.zeros(3)

    comm.Allreduce(local_min, global_min, op=MPI.MIN)
    comm.Allreduce(local_max, global_max, op=MPI.MAX)

    xmin, ymin, zmin = global_min
    xmax, ymax, zmax = global_max

    return xmin, ymin, zmin, xmax, ymax, zmax


def shift_to_origin(mesh):
    """
    Shift mesh coordinates so the global minimum (xmin,ymin,zmin)
    moves to the origin (0,0,0).

    Returns:
        xmin, ymin, zmin, xmax, ymax, zmax for the shifted mesh
    """

    info(f"Shifting mesh to origin")

    # Get bounds
    xmin, ymin, zmin, xmax, ymax, zmax = bounds(mesh)

    # Print old bounds
    info(f"Original bounds: [{xmin}, {xmax}] x [{ymin}, {ymax}] x [{zmin}, {zmax}]")

    # Shift the mesh coordinates
    global_min = np.array([xmin, ymin, zmin])
    mesh.geometry.x[:] -= global_min

    # Print new bounds
    xmin, ymin, zmin, xmax, ymax, zmax = bounds(mesh)
    info(f"New bounds: [{xmin}, {xmax}] x [{ymin}, {ymax}] x [{zmin}, {zmax}]")

    return xmin, ymin, zmin, xmax, ymax, zmax


def _hmin(self):
    "Compute global minimum cell diameter (mesh size)."

    # Get local cell indices
    cells = np.arange(
        self.topology.index_map(self.topology.dim).size_local, dtype=np.int32
    )

    # Compute cell diameters
    cell_diameters = self.h(self.topology.dim, cells)

    # Local minimum
    local_min = np.min(cell_diameters)

    # Global minimum reduction across MPI ranks
    global_min = self.comm.allreduce(local_min, op=MPI.MIN)

    return global_min


dolfinx.mesh.Mesh.hmin = _hmin

__all__.extend(["bounds", "shift_to_origin"])
