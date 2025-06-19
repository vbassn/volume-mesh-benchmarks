from dolfinx import mesh, fem, io
from dolfinx.fem import Function, FunctionSpace, dirichletbc, locate_dofs_topological
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import numpy as np
import ufl
import dolfinx

# Print DOLFINx version for verification
print(f"DOLFINx version: {dolfinx.__version__}")

# Step 1: Create a 3D tetrahedral mesh
comm = MPI.COMM_WORLD
domain = mesh.create_box(
    comm,
    [np.array([0, 0, 0]), np.array([1, 1, 1])],
    [4, 4, 4],
    mesh.CellType.tetrahedron,
)

# Step 2: Define facet markers
fdim = domain.topology.dim - 1  # Facet dimension (2 for 3D)


# Define boundary conditions for x=0 and x=1
def left_boundary(x):
    return np.isclose(x[0], 0.0)


def right_boundary(x):
    return np.isclose(x[0], 1.0)


# Ensure facet-cell connectivity is created
domain.topology.create_connectivity(fdim, domain.topology.dim)

# Locate facets on the boundaries
left_facets = mesh.locate_entities_boundary(domain, fdim, left_boundary)
right_facets = mesh.locate_entities_boundary(domain, fdim, right_boundary)

# Combine facet indices and assign markers (1 for x=0, 2 for x=1)
facet_indices = np.hstack([left_facets, right_facets])
facet_markers = np.hstack(
    [np.full_like(left_facets, 1), np.full_like(right_facets, 2)]
).astype(np.int32)

# Sort indices to ensure consistency (required for MeshTags)
sorted_indices = np.argsort(facet_indices)
facet_indices = facet_indices[sorted_indices]
facet_markers = facet_markers[sorted_indices]

# Create MeshTags for facets using meshtags (DOLFINx 0.9.0)
facet_tags = mesh.meshtags(domain, fdim, facet_indices, facet_markers)
facet_tags.name = "boundary_markers"

# Step 3: Write mesh and facet tags to XDMF file
with io.XDMFFile(
    comm, "test_facet_markers_output/mesh_with_facets.xdmf", "w"
) as xdmf_file:
    xdmf_file.write_mesh(domain)
    xdmf_file.write_meshtags(facet_tags, domain.geometry)

# Step 4: Read mesh and facet tags from XDMF file
with io.XDMFFile(
    comm, "test_facet_markers_output/mesh_with_facets.xdmf", "r"
) as xdmf_file:
    domain = xdmf_file.read_mesh(name="mesh")
    # Create facet connectivity before reading meshtags
    domain.topology.create_connectivity(fdim, fdim)
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    facet_tags = xdmf_file.read_meshtags(domain, name="boundary_markers")

# Step 5: Set up and solve a Poisson PDE
# Define function space
V = dolfinx.fem.functionspace(domain, ("Lagrange", 1))

# Define trial and test functions
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Define source term (f = 1.0 for simplicity)
f = fem.Constant(domain, 1.0)

# Define variational form: -div(grad(u)) = f
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx

# Apply Dirichlet boundary conditions using facet tags
u_left = fem.Constant(domain, 0.0)  # u = 0 on x=0
u_right = fem.Constant(domain, 1.0)  # u = 1 on x=1

# Locate dofs for boundaries
left_dofs = fem.locate_dofs_topological(V, fdim, facet_tags.find(1))
right_dofs = fem.locate_dofs_topological(V, fdim, facet_tags.find(2))

# Create boundary conditions
bc_left = dirichletbc(u_left, left_dofs, V)
bc_right = dirichletbc(u_right, right_dofs, V)
bcs = [bc_left, bc_right]

# Assemble and solve the linear system
problem = LinearProblem(
    a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
)
uh = problem.solve()

# Step 6: Save the solution to an XDMF file
with io.XDMFFile(comm, "test_facet_markers_output/solution.xdmf", "w") as xdmf_file:
    uh.name = "u"
    xdmf_file.write_mesh(domain)
    xdmf_file.write_function(uh)

# Optional: Compute and print the L2 norm of the solution for verification
norm_L2 = fem.assemble_scalar(fem.form(ufl.inner(uh, uh) * ufl.dx)) ** 0.5
if comm.rank == 0:
    print(f"L2 norm of solution: {norm_L2:.6f}")
