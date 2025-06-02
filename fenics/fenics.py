"Simple FEniCS style FEniCSx wrapper"

__all__ = []

# --- MPI imports --
from mpi4py import MPI

__all__.extend(["MPI"])

# --- UFL imports ---
from ufl import TrialFunction, TestFunction, SpatialCoordinate
from ufl import dx, ds, inner, grad

__all__.extend(
    ["TrialFunction", "TestFunction", "SpatialCoordinate", "dx", "ds", "inner", "grad"]
)

# --- DOLFINx io imports ---
from dolfinx.io import XDMFFile

__all__.extend(["XDMFFile"])

# --- DOLFINx fem imports ---
from dolfinx.fem import Constant, functionspace
from dolfinx.fem.petsc import LinearProblem

FunctionSpace = lambda mesh, family, degree: functionspace(mesh, (family, degree))

__all__.extend(["Constant", "FunctionSpace", "LinearProblem"])

# --- DOLFINx log imports ---
from dolfinx.log import set_log_level, LogLevel

DEBUG = LogLevel.DEBUG
INFO = LogLevel.INFO
WARNING = LogLevel.WARNING
ERROR = LogLevel.ERROR

__all__.extend(["set_log_level", "LogLevel", "DEBUG", "INFO", "WARNING", "ERROR"])

# --- DOLFINx boundary conditions ---
from dolfinx.mesh import locate_entities_boundary
from dolfinx.fem import locate_dofs_topological, dirichletbc
from petsc4py.PETSc import ScalarType


def DirichletBC(V, value, marker):
    """Create a Dirichlet boundary condition."""

    msh = V.mesh
    tdim = V.mesh.topology.dim

    facets = locate_entities_boundary(msh, dim=tdim - 1, marker=marker)
    dofs = locate_dofs_topological(V=V, entity_dim=tdim - 1, entities=facets)
    bc = dirichletbc(value=ScalarType(value), dofs=dofs, V=V)

    return bc


__all__.extend(["DirichletBC"])

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


def _save(self, filename):
    """Decorator to save a dolfinx.fem.Function or dolfinx.mesh.Mesh to an XDMF file."""

    if not filename.endswith(".xdmf"):
        raise ValueError("Filename must end with .xdmf")

    # Get the mesh from the function or mesh object
    if isinstance(self, dolfinx.fem.Function):
        mesh = self.function_space.mesh
    elif isinstance(self, dolfinx.mesh.Mesh):
        mesh = self
    else:
        raise TypeError("Object must be a dolfinx.fem.Function or dolfinx.mesh.Mesh")

    # Write mesh and function to XDMF file
    with XDMFFile(mesh.comm, filename, "w") as xdmf_file:
        xdmf_file.write_mesh(mesh)
        if isinstance(self, dolfinx.fem.Function):
            xdmf_file.write_function(self)


dolfinx.fem.Function.save = _save
dolfinx.mesh.Mesh.save = _save
