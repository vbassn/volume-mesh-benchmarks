from fenics import *


# Set log level to INFO
set_log_level(INFO)

# Load volume mesh
xdmf = XDMFFile(MPI.COMM_WORLD, "../dtcc/gbg_volume_mesh.xdmf", "r")
mesh = xdmf.read_mesh(name="Grid")

# Create function space
V = FunctionSpace(mesh, "Lagrange", 1)


# Define boundary marker
def marker(x):
    return near(x[0], 0.0) | near(x[0], 2.0) | True


# Define boundary condition
bc = DirichletBC(V, 0.0, marker)


# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
x = SpatialCoordinate(mesh)
f = Constant(mesh, 100.0)
a = inner(grad(u), grad(v)) * dx
L = inner(f, v) * dx

# Solve linear problem
petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}
problem = LinearProblem(a, L, bcs=[bc], petsc_options=petsc_options)
uh = problem.solve()

# Save solution to file
uh.save("gbg_poission_out/solution.xdmf")

# Plot solution
plot(uh)
