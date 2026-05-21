import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Import existing robust functions from the main module
from car_robot_time_optimal_casadi_and_MPC import (
    solve_time_optimal_problem,
    get_refined_polygon_vertices,
    animate_2way_hodograph,
    get_polygonal_constraints,
    simple_car_robot_dynamics
)

def solve_offline_poly_problem(S_start, S_target, N, L, X_ref, T_ref_guess, U_ref_guess, max_faces):
    import casadi as ca
    opti = ca.Opti()
    
    T = opti.variable()
    X = opti.variable(5, N+1)
    U = opti.variable(2, N)
    
    opti.minimize(T)
    opti.subject_to(T >= 0.1)
    
    opti.subject_to(X[:, 0] == S_start)
    opti.subject_to(X[0, N] == S_target[0])
    opti.subject_to(X[1, N] == S_target[1])
    opti.subject_to(X[2:, N] == S_target[2:])
    
    dt = T / N
    
    # Physics constants
    m, g, mu = 20.0, 9.81, 1.0
    rho = L / ca.sqrt(3) 
    c = (rho / L)**2     
    m_w, r = 1.0, 0.1
    k_rear = (2 * m_w / m) * ((r/2) / r)**2 
    f_max = mu * 0.5 * m * g
    
    # Polygon Angles and Inscribed Radius
    angles = [i * 2 * np.pi / max_faces for i in range(max_faces)]
    cos_pi_N = np.cos(np.pi / max_faces)
    
    for k in range(N):
        # RK4 Dynamics
        k1 = simple_car_robot_dynamics(X[:, k], U[:, k], L)
        k2 = simple_car_robot_dynamics(X[:, k] + dt*k1/2, U[:, k], L)
        k3 = simple_car_robot_dynamics(X[:, k] + dt*k2/2, U[:, k], L)
        k4 = simple_car_robot_dynamics(X[:, k] + dt*k3, U[:, k], L)
        opti.subject_to(X[:, k+1] == X[:, k] + dt*(k1 + 2*k2 + 2*k3 + k4)/6)
        
        v = X[4, k]
        phi = X[3, k]
        u1 = U[0, k]
        u2 = U[1, k]
        
        # Symbolic Forces
        F_N1 = (m/4) * ((1+c)*(u1*ca.sin(phi) + v*u2*ca.cos(phi)) + (1/L)*v**2 * ca.sin(phi)*ca.cos(phi))
        F_N2 = (m/4) * ((1-c)*(u1*ca.sin(phi) + v*u2*ca.cos(phi)) + (1/L)*v**2 * ca.sin(phi)*ca.cos(phi))
        F_T1 = m * (u1*ca.cos(phi) - v*u2*ca.sin(phi) - (1/(4*L))*v**2 * ca.sin(phi)**2)
        F_T2 = -m * k_rear * (u1*ca.cos(phi) - v*u2*ca.sin(phi))
        
        # Symbolic Force-Domain Polygon Constraints (Approximating the circle)
        for angle in angles:
            opti.subject_to( F_T1 * np.cos(angle) + F_N1 * np.sin(angle) <= f_max * cos_pi_N )
            opti.subject_to( F_T2 * np.cos(angle) + F_N2 * np.sin(angle) <= f_max * cos_pi_N )
        
        # Control & State limits
        opti.subject_to(opti.bounded(-5.0, u1, 5.0))
        opti.subject_to(opti.bounded(-3*np.pi/2, u2, 3*np.pi/2))
        opti.subject_to(opti.bounded(-np.pi/2, X[3, k], np.pi/2))
        opti.subject_to(opti.bounded(-20.0, X[4, k], 20.0))
        
    # Warm Start with Exact Solution
    opti.set_initial(T, T_ref_guess)
    opti.set_initial(X, X_ref)
    opti.set_initial(U, U_ref_guess)
    
    opti.solver('ipopt', {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'})
    sol = opti.solve()
    
    T_opt = sol.value(T)
    X_opt = sol.value(X)
    U_opt = sol.value(U)
    t_grid = np.linspace(0, T_opt, N+1)
    
    return T_opt, X_opt, U_opt, t_grid

def animate_reference_polygon(t_grid, X_ref, U_ref, L=1.0, max_faces=16, filename="reference_polygon_check.gif"):
    print(f"Generating Sanity Check Animation: {filename}...")
    m, g, mu = 20.0, 9.81, 1.0
    rho = L / np.sqrt(3)
    c = (rho / L)**2
    f_max = mu * 0.5 * m * g
    m_w, r = 1.0, 0.1
    k_rear = (2 * m_w / m) * ((r/2) / r)**2
    a_max, b_max = 5.0, 3 * np.pi / 2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    u1_grid = np.linspace(-6, 6, 200)
    u2_grid = np.linspace(-6, 6, 200)
    U1, U2 = np.meshgrid(u1_grid, u2_grid)

    N_sim = U_ref.shape[1]
    X_com_ref_full = X_ref[0, :]
    Y_com_ref_full = X_ref[1, :]
    
    x_min, x_max = min(X_com_ref_full) - 1.0, max(X_com_ref_full) + 1.0
    y_min, y_max = min(Y_com_ref_full) - 1.0, max(Y_com_ref_full) + 1.0
    
    def update(k):
        ax1.clear()
        ax2.clear()
        
        t_val = t_grid[k]
        v_val = X_ref[4, k]
        phi_val = X_ref[3, k]
        u_ref = U_ref[:, k]
        
        # --- AX1: Physical Trajectory ---
        ax1.plot(X_com_ref_full, Y_com_ref_full, 'b-', alpha=0.3, label='Exact Path')
        
        x_c, y_c, theta_c = X_ref[0, k], X_ref[1, k], X_ref[2, k]
        x_rear = x_c - 0.5 * L * np.cos(theta_c)
        y_rear = y_c - 0.5 * L * np.sin(theta_c)
        x_front = x_c + 0.5 * L * np.cos(theta_c)
        y_front = y_c + 0.5 * L * np.sin(theta_c)
        
        ax1.plot([x_rear, x_front], [y_rear, y_front], 'k-', linewidth=4, label='Car Body')
        ax1.plot(x_c, y_c, 'ko', markersize=6)
        ax1.set_xlim(x_min, x_max); ax1.set_ylim(y_min, y_max)
        ax1.set_aspect('equal')
        ax1.set_title(f"Reference Trajectory (t = {t_val:.2f} s)")
        ax1.grid(True)
        
        # --- AX2: Hodograph ---
        # 1. Exact Nonlinear Contours
        F_N1 = (m/4) * ((1+c)*(U1*np.sin(phi_val) + v_val*U2*np.cos(phi_val)) + (1/L)*v_val**2 * np.sin(phi_val)*np.cos(phi_val))
        F_T1 = m * (U1*np.cos(phi_val) - v_val*U2*np.sin(phi_val) - (1/(4*L))*v_val**2 * np.sin(phi_val)**2)
        F_N2 = (m/4) * ((1-c)*(U1*np.sin(phi_val) + v_val*U2*np.cos(phi_val)) + (1/L)*v_val**2 * np.sin(phi_val)*np.cos(phi_val))
        F_T2 = -m * k_rear * (U1*np.cos(phi_val) - v_val*U2*np.sin(phi_val))
        
        ax2.contour(U1, U2, F_N1**2 + F_T1**2 - f_max**2, levels=[0], colors='blue', linewidths=2, alpha=0.5)
        ax2.contour(U1, U2, F_N2**2 + F_T2**2 - f_max**2, levels=[0], colors='red', linewidths=2, alpha=0.5)
        ax2.plot([-a_max, a_max, a_max, -a_max, -a_max], [-b_max, -b_max, b_max, b_max, -b_max], 'k-', linewidth=2)
        
        # 2. Get the specific approximation points (vertices)
        refined_verts = get_refined_polygon_vertices(v_val, phi_val, L, max_faces=max_faces)
        
        if len(refined_verts) > 0:
            # Draw the polygon boundary
            poly_patch = plt.Polygon(refined_verts, closed=True, facecolor='cyan', edgecolor='blue', alpha=0.2, linewidth=2)
            ax2.add_patch(poly_patch)
            
            # BOLDLY mark the approximation points
            ax2.plot(refined_verts[:, 0], refined_verts[:, 1], 'mo', markersize=8, markeredgecolor='black', label=f'Approximation Points (N={max_faces})')
            
        # 3. Mark the Exact Control
        ax2.plot(u_ref[0], u_ref[1], 'b*', markersize=14, label='Exact Ref Control')

        ax2.set_xlim(-6, 6)
        ax2.set_ylim(-6, 6)
        ax2.set_title(f"Hodograph Check (v = {v_val:.2f} m/s | phi = {phi_val:.2f} rad)")
        ax2.set_xlabel('$u_1$ (Linear Accel) [m/s²]')
        ax2.set_ylabel('$u_2$ (Steering Rate) [rad/s]')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True)

    ani = animation.FuncAnimation(fig, update, frames=N_sim, blit=False)
    ani.save(filename, writer='pillow', fps=15)
    print(f"Animation successfully saved to {filename}")

def plot_offline_comparison(t_ref, X_ref, U_ref, t_poly, X_poly, U_poly, L=1.0, save_path=None, show_plot=False):
    """
    Generates an 8-panel plot comparing the exact non-linear trajectory 
    with the offline polygonal-constrained trajectory.
    """
    m, g, mu = 20.0, 9.81, 1.0
    rho = L / np.sqrt(3)
    c = (rho / L)**2
    f_max = mu * 0.5 * m * g
    m_w, r = 1.0, 0.1
    k_rear = (2 * m_w / m) * ((r/2) / r)**2
    
    plt.figure(figsize=(18, 8))
    plt.suptitle("Offline Trajectory Comparison: Exact vs. Polygonal")
    
    # 1. Spatial Trajectory
    X_rear_ref  = X_ref[0, :]  - L * np.cos(X_ref[2, :])
    Y_rear_ref  = X_ref[1, :]  - L * np.sin(X_ref[2, :])
    X_rear_poly = X_poly[0, :] - L * np.cos(X_poly[2, :])
    Y_rear_poly = X_poly[1, :] - L * np.sin(X_poly[2, :])

    plt.subplot(2, 4, 1)
    plt.plot(X_ref[0, :], X_ref[1, :], 'b-', linewidth=2, label='Exact CoM')
    plt.plot(X_poly[0, :], X_poly[1, :], 'g--', linewidth=2, label='Poly CoM')
    plt.plot(X_rear_ref,  Y_rear_ref,  'b-',  linewidth=1, alpha=0.4, label='Exact Rear Axle')
    plt.plot(X_rear_poly, Y_rear_poly, 'g--', linewidth=1, alpha=0.4, label='Poly Rear Axle')
    plt.plot(X_ref[0, 0], X_ref[1, 0], 'go', label='Start')
    plt.plot(X_ref[0, -1], X_ref[1, -1], 'rx', label='Target')
    plt.title('Spatial Trajectory Comparison')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    
    # 2. Velocity
    plt.subplot(2, 4, 2)
    plt.plot(t_ref, X_ref[4, :], 'b-', label='Exact v')
    plt.plot(t_poly, X_poly[4, :], 'g--', label='Poly v')
    plt.title('Velocity vs Time')
    plt.xlabel('Time [s]')
    plt.ylabel('v [m/s]')
    plt.legend()
    plt.grid(True)
    
    # 3. Heading (Theta)
    plt.subplot(2, 4, 3)
    plt.plot(t_ref, X_ref[2, :], 'b-', label='Exact Theta')
    plt.plot(t_poly, X_poly[2, :], 'g--', label='Poly Theta')
    plt.title('Heading (Theta) vs Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Theta [rad]')
    plt.legend()
    plt.grid(True)
    
    # 4. Steering Angle (Phi)
    plt.subplot(2, 4, 4)
    plt.plot(t_ref, X_ref[3, :], 'b-', label='Exact Phi')
    plt.plot(t_poly, X_poly[3, :], 'g--', label='Poly Phi')
    plt.title('Steering Angle (Phi) vs Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Phi [rad]')
    plt.legend()
    plt.grid(True)
    
    # 5. Controls
    plt.subplot(2, 4, 5)
    plt.step(t_ref[:-1], U_ref[0, :], 'b-', label='Exact u1', where='post', alpha=0.7)
    plt.step(t_poly[:-1], U_poly[0, :], 'c--', label='Poly u1', where='post', alpha=0.7)
    plt.step(t_ref[:-1], U_ref[1, :], 'r-', label='Exact u2', where='post', alpha=0.7)
    plt.step(t_poly[:-1], U_poly[1, :], 'm--', label='Poly u2', where='post', alpha=0.7)
    plt.title('Controls vs Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Controls')
    plt.legend()
    plt.grid(True)
    
    # Force Calculations for Subplots 6 & 7
    def calc_forces(X, U):
        v, phi = X[4, :-1], X[3, :-1]
        u1, u2 = U[0, :], U[1, :]
        FN1 = (m/4) * ((1+c)*(u1*np.sin(phi) + v*u2*np.cos(phi)) + (1/L)*v**2 * np.sin(phi)*np.cos(phi))
        FT1 = m * (u1*np.cos(phi) - v*u2*np.sin(phi) - (1/(4*L))*v**2 * np.sin(phi)**2)
        FN2 = (m/4) * ((1-c)*(u1*np.sin(phi) + v*u2*np.cos(phi)) + (1/L)*v**2 * np.sin(phi)*np.cos(phi))
        FT2 = -m * k_rear * (u1*np.cos(phi) - v*u2*np.sin(phi))
        return FN1, FT1, FN2, FT2
        
    FN1_ref, FT1_ref, FN2_ref, FT2_ref = calc_forces(X_ref, U_ref)
    FN1_poly, FT1_poly, FN2_poly, FT2_poly = calc_forces(X_poly, U_poly)
    
    # 6. Front Forces
    plt.subplot(2, 4, 6)
    plt.plot(t_ref[:-1], FN1_ref, 'b-', label='Exact $F_{N1}$', alpha=0.7)
    plt.plot(t_poly[:-1], FN1_poly, 'c--', label='Poly $F_{N1}$', alpha=0.7)
    plt.plot(t_ref[:-1], FT1_ref, 'g-', label='Exact $F_{T1}$', alpha=0.7)
    plt.plot(t_poly[:-1], FT1_poly, 'y--', label='Poly $F_{T1}$', alpha=0.7)
    plt.title('Front Forces vs Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Force [N]')
    plt.legend()
    plt.grid(True)
    
    # 7. Slip Constraints (Total Front Force)
    plt.subplot(2, 4, 7)
    plt.plot(t_ref[:-1], np.sqrt(FN1_ref**2 + FT1_ref**2), 'b-', label='Exact FW Force')
    plt.plot(t_poly[:-1], np.sqrt(FN1_poly**2 + FT1_poly**2), 'g--', label='Poly FW Force')
    plt.axhline(y=f_max, color='k', linestyle=':', label='Limit ($f_{max}$)')
    plt.title('FW Slip Constraints')
    plt.xlabel('Time [s]')
    plt.ylabel('Total Force [N]')
    plt.legend()
    plt.grid(True)
    
    # 8. State Error (Poly vs Exact)
    plt.subplot(2, 4, 8)
    # Interpolate exact states to poly time grid for accurate error
    x_ref_interp = np.interp(t_poly, t_ref, X_ref[0, :])
    y_ref_interp = np.interp(t_poly, t_ref, X_ref[1, :])
    th_ref_interp = np.interp(t_poly, t_ref, X_ref[2, :])
    plt.plot(t_poly, np.abs(x_ref_interp - X_poly[0, :]), 'r-', alpha=0.6, label='|X error|')
    plt.plot(t_poly, np.abs(y_ref_interp - X_poly[1, :]), 'g--', alpha=0.6, label='|Y error|')
    plt.plot(t_poly, np.abs(th_ref_interp - X_poly[2, :]), 'b:', alpha=0.6, label='|Theta error|')
    plt.title('Absolute Tracking Error (Poly vs Exact)')
    plt.xlabel('Time [s]')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    # --- Configuration ---
    L_val = 1.0
    num_approx_points = 8  # Change this to test 8, 12, 16, 24 points
    N_total = 500
    is_save_data = True
    
    #u turn 
    # S_start = [L_val, 0, 0, 0, 10]
    # S_target = [L_val, 3, np.pi, 0, 10] 
    # parallel parking
    S_start = [L_val, 0, 0, 0, 0]
    S_target = [L_val, 1, 0, 0, 0] 
    #intrestind trail 007
    # S_start = [0, 0, 0, 0, 7]
    # S_target = [1.5, -9.0, 3.0, 0, 2]
    # 50% error in poly
    # S_start = [0, 0, 0, 0, 9.4]
    # S_target = [10, 5.36, 0.622,0,2.8]
    
    print("--- Solving Exact Offline Reference Problem ---")
    T_ref, X_ref, U_ref, t_grid, solver_time = solve_time_optimal_problem(
        S_start, S_target, N=N_total, L=L_val
    )
    
    if T_ref is None:
        print("Offline solver failed to find a trajectory.")
        exit()
        
    print(f"Exact trajectory found in {solver_time:.3f} seconds.")
    
    # Generate the sanity check animation
    # animate_reference_polygon(
    #     t_grid, 
    #     X_ref, 
    #     U_ref, 
    #     L=L_val, 
    #     max_faces=num_approx_points,
    #     filename=f"sanity_check_polygon_N{num_approx_points}.gif"
    # )

    print(f"\n--- Solving Offline Polygonal Problem ({num_approx_points} points) ---")
    try:
        T_poly, X_poly, U_poly, t_grid_poly = solve_offline_poly_problem(
            S_start, S_target, N_total, L_val, X_ref, T_ref, U_ref, num_approx_points
        )
        
        if is_save_data:
            os.makedirs("data", exist_ok=True)
            save_file = os.path.join("data", f"offline_poly_data_N{num_approx_points}.npz")
            np.savez(save_file,
                     T_ref=T_ref, t_ref=t_grid, X_ref=X_ref, U_ref=U_ref,
                     T_poly=T_poly, t_poly=t_grid_poly, X_poly=X_poly, U_poly=U_poly)
            print(f"Data saved to {save_file}")

        time_diff = T_poly - T_ref
        print("\n" + "="*50)
        print(f"RESULTS: GEOMETRIC DEFICIT ANALYSIS (N={num_approx_points})")
        print(f"Exact Non-Linear Time:  {T_ref:.4f} s")
        print(f"Polygonal Bounded Time: {T_poly:.4f} s")
        print(f"Time Penalty (Deficit): {time_diff:.4f} s ({(time_diff/T_ref)*100:.3f}%)")
        print("="*50 + "\n")
        
        # Plot Detailed 8-Panel Comparison
        os.makedirs(os.path.join("results", "figures"), exist_ok=True)
        plot_offline_comparison(
            t_grid, X_ref, U_ref,
            t_grid_poly, X_poly, U_poly,
            L=L_val,
            save_path=os.path.join("results", "figures", f"offline_poly_comparison_N{num_approx_points}.png"),
            show_plot=True
        )

        print("--- Generating Comparison Animation ---")
        animate_2way_hodograph(
            t_grid_poly,
            X_ref,
            X_poly,
            U_ref,
            U_poly,
            L=L_val,
            max_faces=num_approx_points,
            filename=os.path.join("results", "figures", f"offline_poly_analysis_N{num_approx_points}.gif")
        )
    except Exception as e:
        print(f"Offline Poly Solver failed: {e}")