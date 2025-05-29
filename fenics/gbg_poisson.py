from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import numpy as np
import pyvista

import dolfinx, ufl
from dolfinx import fem, io, mesh, plot
from dolfinx.io import XDMFFile
from dolfinx.fem.petsc import LinearProblem
from ufl import ds, dx, grad, inner

# Set log level to INFO
dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

# Load volume mesh
with XDMFFile(MPI.COMM_WORLD, "../dtcc/gbg_volume_mesh.xdmf", "r") as xdmf:
    msh = xdmf.read_mesh(name="Grid")

# Create function space
V = fem.functionspace(msh, ("Lagrange", 1))


# Marker for boundary conditions
def marker(x):
    # return True
    return np.isclose(x[0], 0.0) | np.isclose(x[0], 2.0) | True


# Define boundary conditions
facets = mesh.locate_entities_boundary(msh, dim=2, marker=marker)
dofs = fem.locate_dofs_topological(V=V, entity_dim=2, entities=facets)
bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)

# Define variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(msh)
f = fem.Constant(msh, 100.0)
a = inner(grad(u), grad(v)) * dx
L = inner(f, v) * dx

# Solve linear paroblem
petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}
problem = LinearProblem(a, L, bcs=[bc], petsc_options=petsc_options)
uh = problem.solve()

# Save solution to file
with io.XDMFFile(msh.comm, "gbg_poisson_output/solution.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(uh)

# View solution with pyvista
cells, types, x = plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(cells, types, x)
grid.point_data["u"] = uh.x.array.real
grid.set_active_scalars("u")
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
warped = grid.warp_by_scalar()
plotter.add_mesh(warped)
plotter.show()
