from fenics import *

# More verbose output
set_log_level(INFO)

# ------------------------------------------------------------
# Problem parameters
# ------------------------------------------------------------
kappa = 1.0  # molecular diffusivity (m²/s)
T = 60.0  # final time (s)
skip = 1  # skip time steps when saving
supg = False  # toggle streamline‑upwind term

# ------------------------------------------------------------
# Geometry
# ------------------------------------------------------------
mesh, markers = load_mesh_with_markers("../dtcc/gbg_volume_mesh.xdmf")
xmin, ymin, zmin, xmax, ymax, zmax = bounds(mesh)

# -----------------------------------------------------------
# Get the number of buildings from the boundary markers
# -----------------------------------------------------------
max_marker = int(markers.values.max())
num_buildings = (max_marker + 1) // 2
info(f"Number of buildings: {num_buildings}")

# ------------------------------------------------------------
# Velocity field (placeholder, load from file later)
# ------------------------------------------------------------
_x = SpatialCoordinate(mesh)
L_x = xmax - xmin
L_y = ymax - ymin
L_z = zmax - zmin
B = 10.0
u = B * as_vector(
    (
        sin(np.pi * (_x[0] / L_x))
        * cos(np.pi * _x[1] / L_y)
        * sin(np.pi * _x[2] / L_z),
        -cos(np.pi * (_x[0] / L_x + 0.5))
        * sin(np.pi * _x[1] / L_y)
        * sin(np.pi * _x[2] / L_z),
        Constant(mesh, 0.0),
    )
)
U_max = B

# ------------------------------------------------------------
# Set time step (CFL‑like heuristic)
# ------------------------------------------------------------
h = mesh.hmin()
info(f"hmin = {h :.3g}")
dt = 0.9 * h / (U_max + kappa / h)
num_steps = round(T / dt + 0.5)
dt = T / num_steps
info(f"Using dt = {dt :.3g} and {num_steps} time steps")

# ------------------------------------------------------------
# Function space
# ------------------------------------------------------------
V = FunctionSpace(mesh, "Lagrange", 1)

# ------------------------------------------------------------
# Source term
# ------------------------------------------------------------
A = 100.0  # amplitude
sigma = 10.0  # spatial extent (m)
x0 = np.array((0.5 * (xmin + xmax), 0.25 * ymin + 0.25 * ymax, 0.9 * zmin + 0.1 * zmax))
t0 = 0.1
_t = Constant(mesh, 0.0)
r2 = sum((_x[i] - x0[i]) ** 2 for i in range(3))
f = A * exp(-r2 / (2 * sigma**2))

# ------------------------------------------------------------
# Boundary conditions
#
# Buildings & ground  → homogeneous Dirichlet (perfect sink)
# Everything else     → default Neumann (do nothing)
# ------------------------------------------------------------
c_D = 0.0
bcs = []
for i in range(num_buildings):
    bc_wall = DirichletBC(V, c_D, markers=markers, marker_value=i)
    bc_roof = DirichletBC(V, c_D, markers=markers, marker_value=2 * i)
    bcs.append(bc_wall)
    bcs.append(bc_roof)

# ------------------------------------------------------------
# Initial conditions
# ------------------------------------------------------------

# Create functions for solution at two time levels
c_0 = Function(V)  # c at time step n-1
c_1 = Function(V)  # c at time step n

# Initial condition
c_0.x.array[:] = 0.0
c_1.x.array[:] = 0.0

# ------------------------------------------------------------
# Variational problem
# ------------------------------------------------------------
c = TrialFunction(V)
v = TestFunction(V)

a = c / dt * v * dx + dot(u, grad(c)) * v * dx + kappa * inner(grad(c), grad(v)) * dx
L = c_0 / dt * v * dx + f * v * dx

# ------------------------------------------------------------
# SUPG stabilization
# ------------------------------------------------------------
if supg:

    # FIXME: Takes a long time to assemble, move outside of loop

    h_K = CellDiameter(mesh)
    u_norm = sqrt(dot(u, u) + 1.0e-12)
    tau = 1.0 / (2 * u_norm / h_K + 4 * kappa / h_K**2)

    a += tau * dot(u, grad(v)) * (c / dt + dot(u, grad(c)) - kappa * div(grad(c))) * dx
    L += tau * dot(u, grad(v)) * (c_0 / dt + f) * dx

# ------------------------------------------------------------
# Linear solver
# ------------------------------------------------------------
opts = {
    "ksp_monitor_short": None,
    "ksp_converged_reason": None,
    "ksp_type": "gmres",
    "ksp_rtol": 1.0e-6,
    "pc_type": "hypre",
    "pc_hypre_type": "boomeramg",
}

problem = LinearProblem(a, L, u=c_1, bcs=bcs, petsc_options=opts)

# -------------------------------------------------------------
# Time‑stepping loop
# -------------------------------------------------------------
c_1.save("gbg_advdiff_output/solution.xdmf", t=0.0)

t = 0.0
for n in range(num_steps):
    t += dt
    info(f"t = {t :.3f}: ||c|| = {np.linalg.norm(c_1.x.array[:]):.3e}")

    # Update time for right-hand side expression
    _t.value = t

    # Solve for c_1
    problem.solve()

    # Shift solution for next time step
    c_0.x.array[:] = c_1.x.array

    # Save
    if n % skip == 0:
        c_1.save("gbg_advdiff_output/solution.xdmf", t=t)

# ------------------------------------------------------------
# Post‑processing & output
# ------------------------------------------------------------

# Save final solution
c_1.save("gbg_advdiff_output/final_solution.xdmf")

# Save velocity field
W = FunctionSpace(mesh, "Lagrange", 1, dim=3)
u = project(u, W)
u.save("gbg_advdiff_output/velocity.xdmf")
