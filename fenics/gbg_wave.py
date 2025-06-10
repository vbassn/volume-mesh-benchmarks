from fenics import *

# More verbose output
set_log_level(INFO)

# Problem parameters
c = 323.0
T = 1.0
skip = 5

# Load mesh and shift to origin
mesh = load_mesh("../dtcc/gbg_volume_mesh.xdmf")
# mesh = BoxMesh(0, 0, 0, 200, 200, 100, 32, 32, 32)  # for testing
xmin, ymin, zmin, xmax, ymax, zmax = shift_to_origin(mesh)

# Shift and save surface mesh for visualization
surface_mesh = load_mesh("../dtcc/gbg_surface_mesh.xdmf")
shift_to_origin(surface_mesh)
surface_mesh.save("gbg_wave_output/surface_mesh.xdmf")

# Set time step size based on CLF condition
h = mesh.hmin()
info(f"hmin = {h :.3g}")
dt = 0.1 * h / c  # 0.25 and above blows up
num_steps = round(T / dt + 0.5)
dt = T / num_steps
info(f"Using dt = {dt :.3g} and {num_steps} time steps based on CFL condition")


# Define source term (a car driving around Poseidon)
A = 100.0  # amplitude
R = 10.0  # radius of circle
tau = 0.5  # time to drive one lap (fast!)
sigma = 1.0  # extent of source
_xmin = np.array((xmin, ymin, zmin))
_xmax = np.array((xmax, ymax, zmax))
x0 = np.array((0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.9 * zmin + 0.1 * zmax))
t0 = 0.1
_x = SpatialCoordinate(mesh)
_t = Constant(mesh, 0.0)
omega = 2 * np.pi / tau
xc = [x0[0] + R * cos(omega * _t), x0[1] + R * sin(omega * _t), x0[2]]
r2 = sum((_x[i] - xc[i]) ** 2 for i in range(3))
f = A * exp(-r2 / (2 * sigma**2))


# Create function space
V = FunctionSpace(mesh, "Lagrange", 1)


# Define boundary conditions:
#
# We use rigid boundary conditions on the ground and buildings
# which is implemented as homogeneous Neumann = do nothing.
#
# We use absorbing boundary conditions on the domain bounding box
# which is implemented as ∂_n u = - (1/c) ∂_t u (Sommerfeld).


def boundary_marker(x):
    atol = 1e-3
    return (
        near(x[0], xmin, atol=atol)
        | near(x[0], xmax, atol=atol)
        | near(x[1], ymin, atol=atol)
        | near(x[1], ymax, atol=atol)
        | near(x[2], zmax, atol=atol)
    )


ds = NeumannBC(mesh, boundary_marker)


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
alpha = Constant(mesh, c**2 * dt**2 / 6)
beta = Constant(mesh, c * dt)
gamma = Constant(mesh, dt**2)
a = u * v * dx + alpha * inner(grad(u), grad(v)) * dx + beta * u * v * ds
L = (2 * u_1 - u_0) * v * dx - alpha * inner(grad(4 * u_1 + u_0), grad(v)) * dx
L += gamma * f * v * dx + beta * u_1 * v * ds


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
problem = LinearProblem(a, L, u=u_2, bcs=[], petsc_options=iterative)

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
    if n % skip == 0:
        u_2.save("gbg_wave_output/solution.xdmf", t=t)

# Save final solution
u_2.save("gbg_wave_output/final_solution.xdmf")

# Plot final solution
# plot(u_2)
