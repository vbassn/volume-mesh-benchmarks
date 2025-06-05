from fenics import *

# More verbose output
set_log_level(INFO)

# Problem parameters
c = 1.0
T = 1.0

# Load mesh and shift to origin
mesh = load_mesh("../dtcc/gbg_volume_mesh.xdmf")
mesh = BoxMesh(0, 0, 0, 1, 1, 1, 64, 64, 64)
xmin, ymin, zmin, xmax, ymax, zmax = shift_to_origin(mesh)


# Set time step size based on CLF condition
h = mesh.hmin()
info(f"hmin = {h :.3g}")
dt = 0.25 * h / c
num_steps = round(T / dt + 0.5)
dt = T / num_steps
info(f"Using dt = {dt :.3g} and {num_steps} time steps based on CFL condition")


# Define source term
_xmin = np.array((xmin, ymin, zmin))
_xmax = np.array((xmax, ymax, zmax))
x0 = 0.5 * (_xmin + _xmax)
t0 = 0.1
sigma = 0.05
tau = 0.02
A = 100.0
_x = SpatialCoordinate(mesh)
_t = Constant(mesh, 0.0)
r2 = sum((_x[i] - x0[i]) ** 2 for i in range(3))
t2 = (_t - t0) ** 2
f = A * exp(-r2 / (2 * sigma**2)) * exp(-t2 / (2 * tau**2))


# Create function space
V = FunctionSpace(mesh, "Lagrange", 1)


# Define boundary condition
def boundary_marker(x):
    return near(x[0], 0.0) | near(x[0], 2.0) | True


bc = DirichletBC(V, 0.0, boundary_marker)


# Functions for solution at three time levels
u_0 = Function(V)  # u at time step n-1
u_1 = Function(V)  # u at time step n
u_2 = Function(V)  # u at time step n+1 (unknown to solve for)

# Initial conditions (u=0, velocity=0)
u_0.x.array[:] = 0.0
u_1.x.array[:] = 0.0


# Define variational problem (leapfrog-Galerkin method)
u = TrialFunction(V)
v = TestFunction(V)
k = Constant(mesh, dt**2 * c**2 / 6)
a = u * v * dx + k * inner(grad(u), grad(v)) * dx
L = (2 * u_1 - u_0) * v * dx - k * inner(grad(4 * u_1 + u_0), grad(v)) * dx
L += Constant(mesh, dt**2) * f * v * dx

# Simple formulation (standard leapfrog)
# a = u * v * dx
# L = (
#    (2 * u_1 - u_0) * v * dx
#    - dt**2 * c**2 * inner(grad(u_1), grad(v)) * dx
#    + dt**2 * f * v * dx
# )

# Define linear problem
direct = {"ksp_type": "preonly", "pc_type": "lu"}
iterative = {
    "ksp_type": "cg",
    "pc_type": "hypre",
    "pc_hypre_type": "boomeramg",
    "ksp_rtol": 1e-8,
    "ksp_max_it": 1000,
    "ksp_monitor": None,
}
problem = LinearProblem(a, L, u=u_2, bcs=[bc], petsc_options=iterative)

# Save initial solution
u_2.save("gbg_wave_output/solution.xdmf", t=0.0)

# Time-stepping loop
t = 0.0
for n in range(num_steps):
    t += dt
    print(f"t = {t}")

    # Update time for right-hand side expression
    _t.value = t

    # Solve for u_2
    problem.solve()

    # Shift solutions for next time step
    u_0.x.array[:] = u_1.x.array
    u_1.x.array[:] = u_2.x.array

    # Save solution
    u_2.save("gbg_wave_output/solution.xdmf", t=t)

# Save final solution
u_2.save("gbg_wave_output/final_solution.xdmf")

# Plot final solution
# plot(u_2)
