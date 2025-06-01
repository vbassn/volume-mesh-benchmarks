#!/usr/bin/env python3
import numpy as np
from mpi4py import MPI
import ufl

from dolfinx import fem, mesh, io
from dolfinx.fem.petsc import LinearProblem # Corrected import

# Common physical and numerical parameters
K_PARAM = 0.1  # Permeability
S_PARAM = 1.0  # Storativity (for transient problems)
PI = np.pi
POLY_DEGREE = 2  # Polynomial degree for Lagrange elements
N_ELEMENTS_1D = 16 # Number of elements in each spatial direction

# --- Common Helper Functions ---

def compute_L2_error_and_norm(domain_mesh, p_h_solution, exact_solution_expr, current_time=None):
    """
    Computes the L2 error ||p_h - p_exact||, the L2 norm ||p_exact||,
    and the relative L2 error.
    """
    V_space = p_h_solution.function_space
    p_exact_in_V = fem.Function(V_space)

    if current_time is not None:
        p_exact_in_V.interpolate(lambda x_coords: exact_solution_expr(x_coords, t=current_time))
    else:
        p_exact_in_V.interpolate(lambda x_coords: exact_solution_expr(x_coords))

    error_squared_form = fem.form((p_h_solution - p_exact_in_V)**2 * ufl.dx)
    error_local = fem.assemble_scalar(error_squared_form)
    total_error_squared = domain_mesh.comm.allreduce(error_local, op=MPI.SUM)
    L2_error_val = np.sqrt(total_error_squared)

    norm_exact_squared_form = fem.form(p_exact_in_V**2 * ufl.dx)
    norm_exact_local = fem.assemble_scalar(norm_exact_squared_form)
    total_norm_exact_squared = domain_mesh.comm.allreduce(norm_exact_local, op=MPI.SUM)
    norm_p_exact_val = np.sqrt(total_norm_exact_squared)

    if norm_p_exact_val < 1e-14:
        relative_L2_error_val = np.nan if L2_error_val > 1e-14 else 0.0
    else:
        relative_L2_error_val = L2_error_val / norm_p_exact_val
        
    return L2_error_val, relative_L2_error_val, norm_p_exact_val

def get_all_boundary_dofs(domain_mesh, V_space):
    """
    Locates degrees of freedom on all exterior facets of the domain_mesh.
    """
    tdim = domain_mesh.topology.dim
    fdim = tdim - 1
    
    domain_mesh.topology.create_entities(fdim)
    # Ensure facet-to-cell connectivity is built
    domain_mesh.topology.create_connectivity(fdim, tdim) 
    
    exterior_facets = mesh.exterior_facet_indices(domain_mesh.topology)
    boundary_dofs = fem.locate_dofs_topological(V_space, fdim, exterior_facets)
    return boundary_dofs

# --- 1. Steady-State 2D Darcy Flow ---
def solve_steady_2d_darcy(comm):
    if comm.rank == 0:
        print(f"\n--- Running Steady-State 2D Darcy Flow (N={N_ELEMENTS_1D}, P{POLY_DEGREE}) ---")

    def p_analytical_2d_steady(x_coords):
        return np.cos(PI * x_coords[0]) * np.cos(PI * x_coords[1])

    def source_f_2d_steady(x_coords):
        return K_PARAM * 2 * PI**2 * np.cos(PI * x_coords[0]) * np.cos(PI * x_coords[1])

    domain = mesh.create_unit_square(comm, N_ELEMENTS_1D, N_ELEMENTS_1D, mesh.CellType.quadrilateral)
    V = fem.functionspace(domain, ("Lagrange", POLY_DEGREE))

    f_h = fem.Function(V)
    f_h.interpolate(source_f_2d_steady)
    p_D_h = fem.Function(V)
    p_D_h.interpolate(p_analytical_2d_steady)
    
    boundary_dofs = get_all_boundary_dofs(domain, V)
    bc = fem.dirichletbc(p_D_h, boundary_dofs)

    p = ufl.TrialFunction(V)
    q = ufl.TestFunction(V)
    K_const = fem.Constant(domain, K_PARAM)
    a = K_const * ufl.dot(ufl.grad(p), ufl.grad(q)) * ufl.dx
    L = f_h * q * ufl.dx

    problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    p_h = problem.solve()
    p_h.name = "Pressure_Steady_2D"

    L2_err, rel_L2_err, norm_p_exact = compute_L2_error_and_norm(domain, p_h, p_analytical_2d_steady)
    if comm.rank == 0:
        print(f"  L2 Error: {L2_err:.4e}")
        print(f"  Relative L2 Error: {rel_L2_err:.4e}")
        print(f"  L2 Norm of Exact Solution: {norm_p_exact:.4e}")

    # Output to XDMF (interpolating to P1 for output)
    V_out = fem.functionspace(domain, ("Lagrange", 1))
    p_h_out = fem.Function(V_out)
    p_h_out.name = p_h.name
    p_h_out.interpolate(p_h)

    p_exact_V_orig = fem.Function(V)
    p_exact_V_orig.interpolate(p_analytical_2d_steady)
    p_exact_out_for_xdmf = fem.Function(V_out)
    p_exact_out_for_xdmf.name = "Pressure_Exact_Steady_2D"
    p_exact_out_for_xdmf.interpolate(p_exact_V_orig)

    with io.XDMFFile(domain.comm, "steady_2d_darcy.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(p_h_out)
        xdmf.write_function(p_exact_out_for_xdmf)
    if comm.rank == 0:
        print("  Solution (interpolated to P1) saved to steady_2d_darcy.xdmf")

# --- 2. Transient 2D Darcy Flow ---
def solve_transient_2d_darcy(comm, T_final=0.5, dt_val=0.01):
    if comm.rank == 0:
        print(f"\n--- Running Transient 2D Darcy Flow (N={N_ELEMENTS_1D}, P{POLY_DEGREE}, T_final={T_final}, dt={dt_val}) ---")
    num_steps = int(T_final / dt_val)

    D_coeff = K_PARAM / S_PARAM
    def p_analytical_2d_transient(x_coords, t):
        return np.exp(-D_coeff * 2 * PI**2 * t) * np.cos(PI * x_coords[0]) * np.cos(PI * x_coords[1])

    def source_f_2d_transient(x_coords, t):
        return np.zeros(x_coords.shape[1])

    domain = mesh.create_unit_square(comm, N_ELEMENTS_1D, N_ELEMENTS_1D, mesh.CellType.quadrilateral)
    V = fem.functionspace(domain, ("Lagrange", POLY_DEGREE))
    V_out = fem.functionspace(domain, ("Lagrange", 1)) # For XDMF output

    f_h = fem.Function(V) 
    p_D_h = fem.Function(V)
    p_n = fem.Function(V)
    p_n.name = "Pressure_Transient_2D_Initial"
    p_n.interpolate(lambda x: p_analytical_2d_transient(x, t=0.0))
    p_h = fem.Function(V)
    p_h.name = "Pressure_Transient_2D"
    p_h.x.array[:] = p_n.x.array

    boundary_dofs = get_all_boundary_dofs(domain, V)
    bc = fem.dirichletbc(p_D_h, boundary_dofs)

    _p = ufl.TrialFunction(V)
    q = ufl.TestFunction(V)
    S_const = fem.Constant(domain, S_PARAM)
    K_const = fem.Constant(domain, K_PARAM)
    dt_const = fem.Constant(domain, dt_val)
    a = (S_const/dt_const * _p * q + K_const * ufl.dot(ufl.grad(_p), ufl.grad(q))) * ufl.dx
    L = (S_const/dt_const * p_n * q + f_h * q) * ufl.dx

    problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    
    p_out_for_xdmf = fem.Function(V_out) # Reusable for XDMF
    with io.XDMFFile(domain.comm, "transient_2d_darcy.xdmf", "w") as xdmf_file:
        xdmf_file.write_mesh(domain)
        p_out_for_xdmf.name = p_n.name 
        p_out_for_xdmf.interpolate(p_n)
        xdmf_file.write_function(p_out_for_xdmf, 0.0)

        current_t = 0.0
        for i in range(num_steps):
            current_t += dt_val
            f_h.interpolate(lambda x: source_f_2d_transient(x, t=current_t))
            p_D_h.interpolate(lambda x: p_analytical_2d_transient(x, t=current_t))
            
            p_h_solved = problem.solve()
            p_h.x.array[:] = p_h_solved.x.array
            p_n.x.array[:] = p_h.x.array

            p_out_for_xdmf.name = p_h.name
            p_out_for_xdmf.interpolate(p_h)
            xdmf_file.write_function(p_out_for_xdmf, current_t)
            
            if comm.rank == 0 and (i + 1) % (num_steps // 10 if num_steps >=10 else 1) == 0 :
                 print(f"  Time step {i+1}/{num_steps}, Time: {current_t:.3f}")

    if comm.rank == 0:
        print(f"  Finished time stepping. Final time: {current_t:.3f}")
        print("  Solution (interpolated to P1) saved to transient_2d_darcy.xdmf")

    L2_err, rel_L2_err, norm_p_exact = compute_L2_error_and_norm(domain, p_h, p_analytical_2d_transient, current_time=T_final)
    if comm.rank == 0:
        print(f"  Verification at T_final = {T_final:.3f}:")
        print(f"    L2 Error: {L2_err:.4e}")
        print(f"    Relative L2 Error: {rel_L2_err:.4e}")
        print(f"    L2 Norm of Exact Solution: {norm_p_exact:.4e}")

# --- 3. Steady-State 3D Darcy Flow ---
def solve_steady_3d_darcy(comm):
    if comm.rank == 0:
        print(f"\n--- Running Steady-State 3D Darcy Flow (N={N_ELEMENTS_1D}, P{POLY_DEGREE}) ---")

    def p_analytical_3d_steady(x_coords):
        return np.cos(PI * x_coords[0]) * np.cos(PI * x_coords[1]) * np.cos(PI * x_coords[2])

    def source_f_3d_steady(x_coords):
        return K_PARAM * 3 * PI**2 * np.cos(PI * x_coords[0]) * np.cos(PI * x_coords[1]) * np.cos(PI * x_coords[2])

    domain = mesh.create_unit_cube(comm, N_ELEMENTS_1D, N_ELEMENTS_1D, N_ELEMENTS_1D, mesh.CellType.hexahedron)
    V = fem.functionspace(domain, ("Lagrange", POLY_DEGREE))

    f_h = fem.Function(V)
    f_h.interpolate(source_f_3d_steady)
    p_D_h = fem.Function(V)
    p_D_h.interpolate(p_analytical_3d_steady)
    
    boundary_dofs = get_all_boundary_dofs(domain, V)
    bc = fem.dirichletbc(p_D_h, boundary_dofs)

    p = ufl.TrialFunction(V)
    q = ufl.TestFunction(V)
    K_const = fem.Constant(domain, K_PARAM)
    a = K_const * ufl.dot(ufl.grad(p), ufl.grad(q)) * ufl.dx
    L = f_h * q * ufl.dx

    problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    p_h = problem.solve()
    p_h.name = "Pressure_Steady_3D"

    L2_err, rel_L2_err, norm_p_exact = compute_L2_error_and_norm(domain, p_h, p_analytical_3d_steady)
    if comm.rank == 0:
        print(f"  L2 Error: {L2_err:.4e}")
        print(f"  Relative L2 Error: {rel_L2_err:.4e}")
        print(f"  L2 Norm of Exact Solution: {norm_p_exact:.4e}")

    V_out = fem.functionspace(domain, ("Lagrange", 1))
    p_h_out = fem.Function(V_out)
    p_h_out.name = p_h.name
    p_h_out.interpolate(p_h)

    p_exact_V_orig = fem.Function(V)
    p_exact_V_orig.interpolate(p_analytical_3d_steady)
    p_exact_out_for_xdmf = fem.Function(V_out)
    p_exact_out_for_xdmf.name = "Pressure_Exact_Steady_3D"
    p_exact_out_for_xdmf.interpolate(p_exact_V_orig)

    with io.XDMFFile(domain.comm, "steady_3d_darcy.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(p_h_out)
        xdmf.write_function(p_exact_out_for_xdmf)
    if comm.rank == 0:
        print("  Solution (interpolated to P1) saved to steady_3d_darcy.xdmf")

# --- 4. Transient 3D Darcy Flow ---
def solve_transient_3d_darcy(comm, T_final=0.5, dt_val=0.01):
    if comm.rank == 0:
        print(f"\n--- Running Transient 3D Darcy Flow (N={N_ELEMENTS_1D}, P{POLY_DEGREE}, T_final={T_final}, dt={dt_val}) ---")
    num_steps = int(T_final / dt_val)

    D_coeff = K_PARAM / S_PARAM
    def p_analytical_3d_transient(x_coords, t):
        return np.exp(-D_coeff * 3 * PI**2 * t) * \
               np.cos(PI * x_coords[0]) * np.cos(PI * x_coords[1]) * np.cos(PI * x_coords[2])

    def source_f_3d_transient(x_coords, t):
        return np.zeros(x_coords.shape[1])

    domain = mesh.create_unit_cube(comm, N_ELEMENTS_1D, N_ELEMENTS_1D, N_ELEMENTS_1D, mesh.CellType.hexahedron)
    V = fem.functionspace(domain, ("Lagrange", POLY_DEGREE))
    V_out = fem.functionspace(domain, ("Lagrange", 1)) # For XDMF output

    f_h = fem.Function(V)
    p_D_h = fem.Function(V)
    p_n = fem.Function(V)
    p_n.name = "Pressure_Transient_3D_Initial"
    p_n.interpolate(lambda x: p_analytical_3d_transient(x, t=0.0))
    p_h = fem.Function(V)
    p_h.name = "Pressure_Transient_3D"
    p_h.x.array[:] = p_n.x.array

    boundary_dofs = get_all_boundary_dofs(domain, V)
    bc = fem.dirichletbc(p_D_h, boundary_dofs)

    _p = ufl.TrialFunction(V)
    q = ufl.TestFunction(V)
    S_const = fem.Constant(domain, S_PARAM)
    K_const = fem.Constant(domain, K_PARAM)
    dt_const = fem.Constant(domain, dt_val)
    a = (S_const/dt_const * _p * q + K_const * ufl.dot(ufl.grad(_p), ufl.grad(q))) * ufl.dx
    L = (S_const/dt_const * p_n * q + f_h * q) * ufl.dx

    problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    
    p_out_for_xdmf = fem.Function(V_out) # Reusable for XDMF
    with io.XDMFFile(domain.comm, "transient_3d_darcy.xdmf", "w") as xdmf_file:
        xdmf_file.write_mesh(domain)
        p_out_for_xdmf.name = p_n.name
        p_out_for_xdmf.interpolate(p_n)
        xdmf_file.write_function(p_out_for_xdmf, 0.0)

        current_t = 0.0
        for i in range(num_steps):
            current_t += dt_val
            f_h.interpolate(lambda x: source_f_3d_transient(x, t=current_t))
            p_D_h.interpolate(lambda x: p_analytical_3d_transient(x, t=current_t))
            
            p_h_solved = problem.solve()
            p_h.x.array[:] = p_h_solved.x.array
            p_n.x.array[:] = p_h.x.array

            p_out_for_xdmf.name = p_h.name
            p_out_for_xdmf.interpolate(p_h)
            xdmf_file.write_function(p_out_for_xdmf, current_t)
            
            if comm.rank == 0 and (i + 1) % (num_steps // 10 if num_steps >=10 else 1) == 0 :
                print(f"  Time step {i+1}/{num_steps}, Time: {current_t:.3f}")
            
    if comm.rank == 0:
        print(f"  Finished time stepping. Final time: {current_t:.3f}")
        print("  Solution (interpolated to P1) saved to transient_3d_darcy.xdmf")

    L2_err, rel_L2_err, norm_p_exact = compute_L2_error_and_norm(domain, p_h, p_analytical_3d_transient, current_time=T_final)
    if comm.rank == 0:
        print(f"  Verification at T_final = {T_final:.3f}:")
        print(f"    L2 Error: {L2_err:.4e}")
        print(f"    Relative L2 Error: {rel_L2_err:.4e}")
        print(f"    L2 Norm of Exact Solution: {norm_p_exact:.4e}")

# --- Main execution block ---
if __name__ == "__main__":
    comm = MPI.COMM_WORLD

    # You can choose to run one or more of these:
    solve_steady_2d_darcy(comm)
    solve_transient_2d_darcy(comm, T_final=0.1, dt_val=0.005)

    solve_steady_3d_darcy(comm)
    # Adjust N_ELEMENTS_1D, T_final, dt_val for 3D transient as it can be slow
    solve_transient_3d_darcy(comm, T_final=0.05, dt_val=0.0025) 

    if comm.rank == 0:
        print("\nAll selected Darcy flow simulations complete.")

