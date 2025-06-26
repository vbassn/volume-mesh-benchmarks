from fenics import *

# More verbose output
set_log_level(INFO)

# ------------------------------------------------------------
# Geometry
# ------------------------------------------------------------
mesh, markers = load_mesh_with_markers("../dtcc/gbg_volume_mesh.xdmf")

# -----------------------------------------------------------
# Get the number of buildings from the boundary markers
# -----------------------------------------------------------
max_marker = int(markers.values.max())
num_buildings = (max_marker + 1) // 2
info(f"Number of buildings: {num_buildings}")

# ------------------------------------------------------------
# Function space
# -----------------------------------------------------------
V = FunctionSpace(mesh, "Lagrange", 1)

# ------------------------------------------------------------
# Boundary conditions
# ------------------------------------------------------------
bcs = []

# Boundary conditions on the buildings
for i in range(num_buildings):
    bc_wall = DirichletBC(V, 1.0, markers=markers, marker_value=i)
    bc_roof = DirichletBC(V, 1.0, markers=markers, marker_value=2 * i)
    bcs.append(bc_wall)
    bcs.append(bc_roof)

# Boundary condition on the top (-2) and four walls (-3, -4, -5, -6)
for i in (-2, -3, -4, -5, -6):
    bc = DirichletBC(V, 0.0, markers=markers, marker_value=i)
    bcs.append(bc)

# ------------------------------------------------------------
# Variational problem
# ------------------------------------------------------------
u = TrialFunction(V)
v = TestFunction(V)
x = SpatialCoordinate(mesh)
f = Constant(mesh, 0.0)
a = inner(grad(u), grad(v)) * dx
L = inner(f, v) * dx

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
    "ksp_type": "cg",
    "ksp_rtol": 1.0e-6,
    "pc_type": "hypre",
    "pc_hypre_type": "boomeramg",
}

problem = LinearProblem(a, L, bcs=bcs, petsc_options=opts)
u = problem.solve()

# ------------------------------------------------------------
# Post-processing & output
# ------------------------------------------------------------
u.save("gbg_poisson_output/solution.xdmf")
