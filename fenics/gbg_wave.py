from fenics import *
import numpy as np

set_log_level(INFO)

# Problem parameters
c = 1.0  # Wave speed
T = 2.0  # Final time
num_steps = 200  # Number of time steps
dt = T / num_steps  # Time step size

# Mesh and function space
mesh = load_mesh("../dtcc/gbg_volume_mesh.xdmf")
V = FunctionSpace(mesh, "Lagrange", 1)

# Define trial and test functions


# Define functions for solution at three time levels
u_0 = Function(V)  # u at time step n-1
u_1 = Function(V)  # u at time step n
u_2 = Function(V)  # u at time step n+1 (unknown to solve for)

# Initial conditions (u=0, velocity=0)
u_0.x.array[:] = 0.0
u_1.x.array[:] = 0.0


# Define boundary condition
def boundary_marker(x):
    return near(x[0], 0.0) | near(x[0], 2.0) | True


bc = DirichletBC(V, 0.0, boundary_marker)
bcs = [bc]

# Define variational problem (leapfrog Galerkin method)
u = TrialFunction(V)
v = TestFunction(V)
a = u * v * dx + (dt**2 * c**2 / 6) * inner(grad(u), grad(v)) * dx
L = (2 * u_1 - u_0) * v * dx - (dt**2 * c**2 / 6) * inner(
    grad(4 * u_1 + u_0), grad(v)
) * dx

# Define linear problem
problem = LinearProblem(
    a, L, u=u_2, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
)

# Time-stepping loop
t = 0.0
for n in range(num_steps):
    t += dt

    # Solve for u_2
    problem.solve()

    # Shift solutions for next time step
    u_0.assign(u_1)
    u_1.assign(u_2)

    # Save solutions at intervals
    if n % 20 == 0:
        print(f"Step {n}, Time {t:.3f}")
        u_2.save(f"wave_solution_{n:04d}.xdmf")

# Save final solution
u_2.save("gbg_wave_output/solution.xdmf")

# Plot final solution
plot(u_2)
