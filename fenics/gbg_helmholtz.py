from fenics import *

# More verbose output
set_log_level(INFO)

# ------------------------------------------------------------
# Problem parameters
# ------------------------------------------------------------
c = 343.0  # speed of sound (m/s)
f = 2.0  # frequency (Hz)
k = 2.0 * np.pi * f / c

# ------------------------------------------------------------
# Geometry
# ------------------------------------------------------------
mesh = load_mesh("../dtcc/gbg_volume_mesh.xdmf")
# mesh = BoxMesh(0, 0, 0, 200, 200, 100, 64, 64, 32)  # for testing
xmin, ymin, zmin, xmax, ymax, zmax = shift_to_origin(mesh)

# Shift and save surface mesh for visualization (optional)
surface_mesh = load_mesh("../dtcc/gbg_surface_mesh.xdmf")
shift_to_origin(surface_mesh)
surface_mesh.save("gbg_helmholtz_output/surface_mesh.xdmf")

# ------------------------------------------------------------
# Check if we resolve the wavelength
# ------------------------------------------------------------
h = mesh.hmin()
info(f"h = {h :.3g}")
info(f"kh = {k * h :.3g}")
if k * h > 0.9:
    error(f"Mesh too coarse for {f:.0f} Hz with P1 elements (k h = {k * h:.2f} > 0.9).")
    exit(1)

# ------------------------------------------------------------
# Source term (real-valued)
# ------------------------------------------------------------
A = 1.0  # amplitude
sigma = 5.0  # spatial extent (m)
x0 = np.array((0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.9 * zmin + 0.1 * zmax))
_x = SpatialCoordinate(mesh)
r2 = sum((_x[i] - x0[i]) ** 2 for i in range(3))
s = A * exp(-r2 / (2 * sigma**2))


# ------------------------------------------------------------
# Boundary conditions
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Function space: mixed (Re, Im) → R²
# -----------------------------------------------------------
W = FunctionSpace(mesh, (("Lagrange", 1), ("Lagrange", 1)))

# ------------------------------------------------------------
# Variational problem
# ------------------------------------------------------------
(p_re, p_im) = TrialFunctions(W)
(q_re, q_im) = TestFunctions(W)

a = (
    inner(grad(p_re), grad(q_re)) * dx
    + inner(grad(p_im), grad(q_im)) * dx
    - k**2 * (p_re * q_re + p_im * q_im) * dx
    + k * (p_im * q_re - p_re * q_im) * ds
)

L = -s * q_re * dx

# ------------------------------------------------------------
# Shifted form for preconditioning
# ------------------------------------------------------------
alpha = 0.2
beta = 0.65

diag_shift = (1.0 - alpha**2) * k**2 * (p_re * q_re + p_im * q_im) * dx
rot_shift = beta * k**2 * (p_im * q_re - p_re * q_im) * dx
a_pc = a + diag_shift + rot_shift

# ------------------------------------------------------------
# Linear solver
# ------------------------------------------------------------
direct = {
    "ksp_monitor_short": None,
    "ksp_converged_reason": None,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

opts = {
    "ksp_monitor_short": None,
    "ksp_converged_reason": None,
    "ksp_type": "fgmres",
    "ksp_rtol": 1.0e-6,
    # "ksp_max_it": 100,
    "ksp_gmres_restart": 100,
    "pc_type": "hypre",
    "pc_hypre_type": "boomeramg",
    "pc_hypre_boomeramg_cycle_type": "W",
    "pc_hypre_boomeramg_max_iter": 4,
    "pc_hypre_boomeramg_coarsen_type": "HMIS",
    "pc_hypre_boomeramg_interp_type": "ext+i",
    "pc_hypre_boomeramg_strong_threshold": 0.25,
    "pc_hypre_boomeramg_agg_nl": 1,
}

# Set up linear problem
problem = LinearProblem(a, L, bcs=[], petsc_options=opts)

# Add preconditioner
A_pc = assemble_matrix(a_pc)
A_pc.assemble()
problem.solver.setOperators(problem.A, A_pc)

# Solve linear problem
p = problem.solve()

# ------------------------------------------------------------
# Post-processing & output
# ------------------------------------------------------------

# Interpolate magnitude of complex solution
V = FunctionSpace(mesh, "Lagrange", 1)
expr = Expression(sqrt(p[0] ** 2 + p[1] ** 2), V.element.interpolation_points())
p_abs = interpolate(expr, V)

# Save solution
p_abs.save("gbg_helmholtz_output/solution.xdmf")
