from fenics import *

# More verbose output
set_log_level(INFO)

# Load mesh and create function space
mesh = load_mesh("../dtcc/gbg_volume_mesh.xdmf")
V = FunctionSpace(mesh, "Lagrange", 1)


# Define boundary condition
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
uh.save("gbg_poission_output/solution.xdmf")

# Plot solution
plot(uh)
