import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

import matplotlib.animation as animation


import casadi as ca

# Global Physics Parameters
m = 20.0
g = 9.81
mu = 1.0
m_w = 1.0
r = 0.1

def simple_car_robot_dynamics(S, u, L=1.0):
    """
    Calculates the state derivative dS/dt based on the Book Chapter (Eq 2.29).
    state: [x, y, theta, phi, v]  (Note: x, y track the Center of Mass)
    u: [u1, u2] (tangential acceleration, steering rate)
    """
    # Extract states
    x = S[0]
    y = S[1]
    theta = S[2]
    phi = S[3]
    v = S[4]

    # Extract controls
    u1 = u[0]
    u2 = u[1]

    # Kinematics tracking the Center of Mass (Eq 2.29)
    dx = v * (ca.cos(theta) * ca.cos(phi) - 0.5 * ca.sin(theta) * ca.sin(phi))
    dy = v * (ca.sin(theta) * ca.cos(phi) + 0.5 * ca.cos(theta) * ca.sin(phi))
    dtheta = (v * ca.sin(phi)) / (2 * L)
    dphi = u2
    dv = u1

    return ca.vertcat(dx, dy, dtheta, dphi, dv)

def solve_time_optimal_problem(S_start, S_target, N=500, L=1.0, phi_max=np.pi/2, a_max=5, b_max=3*np.pi/2, v_max=20.0):
    T_guess = 2.0
    
    rho = L / ca.sqrt(3) # Book's assumption for radius of gyration
    c = (rho / L)**2     # Equals 1/3
    
    # Rear wheel tangential factor (from book example)
    rho_w = r / 2 
    k_rear = (2 * m_w / m) * (rho_w / r)**2 # (Eq 2.32 coefficient)

    opti = ca.Opti()
    T = opti.variable()
    X = opti.variable(5, N+1)
    U = opti.variable(2, N)

    opti.minimize(T)
    opti.subject_to(T >= 0.1)

    # Boundary Constraints (Using Center of Mass!)
    opti.subject_to(X[:,0] == S_start)
    opti.subject_to(X[0,N] == S_target[0])
    opti.subject_to(X[1,N] == S_target[1])
    opti.subject_to(X[2:,N] == S_target[2:])

    dt = T/N
    for k in range(N):
        # 1. Kinematics (Eq 2.29 - Center of Mass)
        k1 = simple_car_robot_dynamics(X[:,k], U[:,k], L)
        k2 = simple_car_robot_dynamics(X[:,k] + dt*k1/2, U[:,k], L)
        k3 = simple_car_robot_dynamics(X[:,k] + dt*k2/2, U[:,k], L)
        k4 = simple_car_robot_dynamics(X[:,k] + dt*k3, U[:,k], L)
        x_next = X[:,k] + dt*(k1 + 2*k2 + 2*k3 + k4)/6
        opti.subject_to(X[:,k+1] == x_next)

        v = X[4,k]
        phi = X[3,k]
        u1 = U[0,k]
        u2 = U[1,k]

        # 2. Book Normal Forces (Eq 2.31)
        F_N1 = (m/4) * ((1+c)*(u1*ca.sin(phi) + v*u2*ca.cos(phi)) + (1/L)*v**2 * ca.sin(phi)*ca.cos(phi))
        F_N2 = (m/4) * ((1-c)*(u1*ca.sin(phi) + v*u2*ca.cos(phi)) + (1/L)*v**2 * ca.sin(phi)*ca.cos(phi))
        
        # 3. Book Tangential Forces (Eq 2.32)
        F_T1 = m * (u1*ca.cos(phi) - v*u2*ca.sin(phi) - (1/(4*L))*v**2 * ca.sin(phi)**2)
        F_T2 = -m * k_rear * (u1*ca.cos(phi) - v*u2*ca.sin(phi))

        # 4. Book Friction Constraints (Eq 2.33 - Scaled for IPOPT)
        f_max = mu * 0.5 * m * g
        opti.subject_to( (F_T1 / f_max)**2 + (F_N1 / f_max)**2 <= 1 )
        opti.subject_to( (F_T2 / f_max)**2 + (F_N2 / f_max)**2 <= 1 )

        # Control & State limits
        opti.subject_to(opti.bounded(-a_max, u1, a_max))
        opti.subject_to(opti.bounded(-b_max, u2, b_max))
        opti.subject_to(opti.bounded(-phi_max, phi, phi_max))
        opti.subject_to(opti.bounded(-v_max, v, v_max))

    # --- INJECT WARM START ---
    opti.set_initial(T, T_guess)
    mid = N // 2
    
    # Mild 1.0m bulge forward instead of 1.5m
    X_guess = np.concatenate([np.linspace(S_start[0], S_start[0]+1.0, mid), 
                              np.linspace(S_start[0]+1.0, S_target[0], N+1-mid)])
    Y_guess = np.linspace(S_start[1], S_target[1], N+1)
    
    # Milder steering swing
    theta_guess = np.concatenate([np.linspace(0, np.pi/7, mid), 
                                  np.linspace(np.pi/7, 0, N+1-mid)])
                                  
    # Start with a lower peak velocity guess (2.0 instead of 3.0)
    v_guess = np.concatenate([np.linspace(2.0, 0, mid), 
                              np.linspace(0, -2.0, N+1-mid)])
    
    opti.set_initial(X[0, :], X_guess)
    opti.set_initial(X[1, :], Y_guess)
    opti.set_initial(X[2, :], theta_guess)
    opti.set_initial(X[3, :], np.zeros(N+1)) 
    opti.set_initial(X[4, :], v_guess)

    opti.solver('ipopt')
    sol = opti.solve()
    solver_time = sol.stats()['t_wall_total']
    # Extract solution
    T_opt = sol.value(T)
    X_opt = sol.value(X)
    U_opt = sol.value(U)
    t_grid = np.linspace(0, T_opt, N+1)

    print(f"Optimal time T: {T_opt}")
    print(f"Solver time: {solver_time}")
    return T_opt, X_opt, U_opt, t_grid, solver_time


def calculate_FW_u2_roots(v, phi, u1, L=1.0):
    """Algebraically solves for u2 on the FW friction boundary given u1."""
    rho = L / np.sqrt(3)
    c = (rho / L)**2
    f_max = mu * 0.5 * m * g
    
    if abs(v) < 1e-4: return [] # Avoid division by zero at rest
    
    # Affine coefficients for FN1 and FT1
    K_N = (m/4) * (1+c)
    B_N = (m/(4*L)) * v**2 * np.sin(phi) * np.cos(phi)
    K_T = m
    B_T = (m/(4*L)) * v**2 * np.sin(phi)**2
    
    c1 = K_N * v * np.cos(phi)
    d1 = K_N * u1 * np.sin(phi) + B_N
    c2 = -K_T * v * np.sin(phi)
    d2 = K_T * u1 * np.cos(phi) - B_T
    
    # Quadratic coefficients: A*(u2)^2 + B*(u2) + C = 0
    A = c1**2 + c2**2
    B = 2 * (c1*d1 + c2*d2)
    C = d1**2 + d2**2 - f_max**2
    
    disc = B**2 - 4*A*C
    if disc < 0: return []
    
    return [(-B + np.sqrt(disc))/(2*A), (-B - np.sqrt(disc))/(2*A)]


def calculate_FW_u1_roots(v, phi, u2, L=1.0):
    """Algebraically solves for u1 on the FW friction boundary given a fixed u2 limit."""
    rho = L / np.sqrt(3)
    c = (rho / L)**2
    f_max = mu * 0.5 * m * g
    
    # Affine coefficients for FN1 and FT1
    K_N = (m/4) * (1+c)
    B_N = (m/(4*L)) * v**2 * np.sin(phi) * np.cos(phi)
    K_T = m
    B_T = (m/(4*L)) * v**2 * np.sin(phi)**2
    
    c1 = K_N * np.sin(phi)
    d1 = K_N * v * u2 * np.cos(phi) + B_N
    c2 = K_T * np.cos(phi)
    d2 = -K_T * v * u2 * np.sin(phi) - B_T
    
    # Quadratic coefficients: A*(u1)^2 + B*(u1) + C = 0
    A = c1**2 + c2**2
    B = 2 * (c1*d1 + c2*d2)
    C = d1**2 + d2**2 - f_max**2
    
    # Avoid division by zero if A is incredibly small
    if abs(A) < 1e-8: return []
    
    disc = B**2 - 4*A*C
    if disc < 0: return []
    
    return [(-B + np.sqrt(disc))/(2*A), (-B - np.sqrt(disc))/(2*A)]


def is_admissible(u1_test, u2_test, v, phi, L=1.0):
    """Checks if a control pair sits inside the valid friction limits for BOTH wheels."""
    rho = L / np.sqrt(3)
    c = (rho / L)**2
    f_max = mu * 0.5 * m * g
    k_rear = (2 * m_w / m) * ((r/2) / r)**2
    
    # Check Front Wheel
    F_N1 = (m/4) * ((1+c)*(u1_test*np.sin(phi) + v*u2_test*np.cos(phi)) + (1/L)*v**2 * np.sin(phi)*np.cos(phi))
    F_T1 = m * (u1_test*np.cos(phi) - v*u2_test*np.sin(phi) - (1/(4*L))*v**2 * np.sin(phi)**2)
    if F_N1**2 + F_T1**2 > f_max**2 + 1.0: return False # 1.0 is a numerical tolerance
    
    # Check Rear Wheel
    F_N2 = (m/4) * ((1-c)*(u1_test*np.sin(phi) + v*u2_test*np.cos(phi)) + (1/L)*v**2 * np.sin(phi)*np.cos(phi))
    F_T2 = -m * k_rear * (u1_test*np.cos(phi) - v*u2_test*np.sin(phi))
    if F_N2**2 + F_T2**2 > f_max**2 + 1.0: return False
    
    return True


def calculate_RW_u2_roots(v, phi, u1, L=1.0):
    """Algebraically solves for u2 on the RW friction boundary given u1."""
    rho = L / np.sqrt(3)
    c = (rho / L)**2
    f_max = mu * 0.5 * m * g
    k_rear = (2 * m_w / m) * ((r/2) / r)**2
    
    if abs(v) < 1e-4: return []
    
    K_N2 = (m/4) * (1-c)
    B_N = (m/(4*L)) * v**2 * np.sin(phi) * np.cos(phi)
    K_T2 = m * k_rear
    
    c1 = K_N2 * v * np.cos(phi)
    d1 = K_N2 * u1 * np.sin(phi) + B_N
    c2 = K_T2 * v * np.sin(phi)
    d2 = -K_T2 * u1 * np.cos(phi)
    
    A = c1**2 + c2**2
    B = 2 * (c1*d1 + c2*d2)
    C = d1**2 + d2**2 - f_max**2
    
    disc = B**2 - 4*A*C
    if disc < 0: return []
    return [(-B + np.sqrt(disc))/(2*A), (-B - np.sqrt(disc))/(2*A)]


def calculate_RW_u1_roots(v, phi, u2, L=1.0):
    """Algebraically solves for u1 on the RW friction boundary given u2."""
    rho = L / np.sqrt(3)
    c = (rho / L)**2
    f_max = mu * 0.5 * m * g
    k_rear = (2 * m_w / m) * ((r/2) / r)**2
    
    K_N2 = (m/4) * (1-c)
    B_N = (m/(4*L)) * v**2 * np.sin(phi) * np.cos(phi)
    K_T2 = m * k_rear
    
    c1 = K_N2 * np.sin(phi)
    d1 = K_N2 * v * u2 * np.cos(phi) + B_N
    c2 = -K_T2 * np.cos(phi)
    d2 = K_T2 * v * u2 * np.sin(phi)
    
    A = c1**2 + c2**2
    B = 2 * (c1*d1 + c2*d2)
    C = d1**2 + d2**2 - f_max**2
    
    if abs(A) < 1e-8: return []
    disc = B**2 - 4*A*C
    if disc < 0: return []
    return [(-B + np.sqrt(disc))/(2*A), (-B - np.sqrt(disc))/(2*A)]


def plot_solution(t_grid, X_opt, U_opt, T_opt, L=1.0):
    X_rear = X_opt[0, :] - L * np.cos(X_opt[2, :])
    Y_rear = X_opt[1, :] - L * np.sin(X_opt[2, :])

    plt.figure(figsize=(18, 8))
    plt.suptitle(f"Optimal Trajectory, T = {T_opt:.2f} s")

    # 1. Trajectory
    plt.subplot(2, 4, 1)
    plt.plot(X_opt[0, :], X_opt[1, :], label='Trajectory')
    plt.plot(X_rear, Y_rear, 'k--', label='Rear Wheels Trajectory')
    plt.plot(X_opt[0, 0], X_opt[1, 0], 'go', label='Start')
    plt.plot(X_opt[0, -1], X_opt[1, -1], 'rx', label='Target')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('Trajectory (Y vs X)')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)

    # 2. Velocity
    plt.subplot(2, 4, 2)
    plt.plot(t_grid, X_opt[4, :])
    plt.xlabel('Time [s]')
    plt.ylabel('v [m/s]')
    plt.title('Velocity vs Time')
    plt.grid(True)

    # 3. Heading (Theta)
    plt.subplot(2, 4, 3)
    plt.plot(t_grid, X_opt[2, :])
    plt.xlabel('Time [s]')
    plt.ylabel('Theta [rad]')
    plt.title('Heading (Theta) vs Time')
    plt.grid(True)

    # 4. Steering Angle (Phi)
    plt.subplot(2, 4, 4)
    plt.plot(t_grid, X_opt[3, :])
    plt.xlabel('Time [s]')
    plt.ylabel('Phi [rad]')
    plt.title('Steering Angle (Phi) vs Time')
    plt.grid(True)

    # 5. Controls
    plt.subplot(2, 4, 5)
    plt.step(t_grid[:-1], U_opt[0, :], label='u1 (accel)', where='post')
    plt.step(t_grid[:-1], U_opt[1, :], label='u2 (steer rate)', where='post')
    plt.xlabel('Time [s]')
    plt.ylabel('Controls')
    plt.title('Controls vs Time')
    plt.legend()
    plt.grid(True)

    # 6. Forces
    rho = L / np.sqrt(3)
    c = (rho / L)**2
    rho_w = r / 2 
    k_rear = (2 * m_w / m) * (rho_w / r)**2

    v_arr = X_opt[4, :-1]
    phi_arr = X_opt[3, :-1]
    u1_arr = U_opt[0, :]
    u2_arr = U_opt[1, :]

    F_N1 = (m/4) * ((1+c)*(u1_arr*np.sin(phi_arr) + v_arr*u2_arr*np.cos(phi_arr)) + (1/L)*v_arr**2 * np.sin(phi_arr)*np.cos(phi_arr))
    F_N2 = (m/4) * ((1-c)*(u1_arr*np.sin(phi_arr) + v_arr*u2_arr*np.cos(phi_arr)) + (1/L)*v_arr**2 * np.sin(phi_arr)*np.cos(phi_arr))
    F_T1 = m * (u1_arr*np.cos(phi_arr) - v_arr*u2_arr*np.sin(phi_arr) - (1/(4*L))*v_arr**2 * np.sin(phi_arr)**2)
    F_T2 = -m * k_rear * (u1_arr*np.cos(phi_arr) - v_arr*u2_arr*np.sin(phi_arr))

    plt.subplot(2, 4, 6)
    plt.plot(t_grid[:-1], F_N1, label='$F_{N1}$')
    plt.plot(t_grid[:-1], F_N2, label='$F_{N2}$')
    plt.plot(t_grid[:-1], F_T1, label='$F_{T1}$')
    plt.plot(t_grid[:-1], F_T2, label='$F_{T2}$')
    plt.xlabel('Time [s]')
    plt.ylabel('Force [N]')
    plt.title('Ground Forces vs Time')
    plt.legend()
    plt.grid(True)

    # 7. Slip Constraints (Total Force vs Limit)
    plt.subplot(2, 4, 7)
    f_max = mu * 0.5 * m * g
    
    F_tot1 = np.sqrt(F_N1**2 + F_T1**2)
    F_tot2 = np.sqrt(F_N2**2 + F_T2**2)
    
    plt.plot(t_grid[:-1], F_tot1, label='Front Total Force', color='blue')
    plt.plot(t_grid[:-1], F_tot2, label='Rear Total Force', color='orange')
    plt.axhline(y=f_max, color='r', linestyle='--', label='Friction Limit ($f_{max}$)')
    
    plt.xlabel('Time [s]')
    plt.ylabel('Force [N]')
    plt.title('Slip Constraints')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('plot_solution.png')
    plt.close()


def plot_hodograph(v_val, phi_val, u1_opt, u2_opt, L=1.0):
    """
    Plots the control Hodograph for a specific state (v, phi) to validate
    the analytical friction bounds against the CasADi optimal controls.
    """
    # Physics parameters (matching the book formulation)
    rho = L / np.sqrt(3)
    c = (rho / L)**2
    f_max = mu * 0.5 * m * g
    
    # Control limits
    a_max = 5.0
    b_max = 3 * np.pi / 2

    # Create a dense grid of u1 and u2 for plotting the ellipse
    u1_grid = np.linspace(-6, 6, 400)
    u2_grid = np.linspace(-6, 6, 400)
    U1, U2 = np.meshgrid(u1_grid, u2_grid)

    # Calculate Front Wheel Forces over the grid
    F_N1 = (m/4) * ((1+c)*(U1*np.sin(phi_val) + v_val*U2*np.cos(phi_val)) + (1/L)*v_val**2 * np.sin(phi_val)*np.cos(phi_val))
    F_T1 = m * (U1*np.cos(phi_val) - v_val*U2*np.sin(phi_val) - (1/(4*L))*v_val**2 * np.sin(phi_val)**2)

    # The constraint boundary (C1 = 0)
    C1 = F_N1**2 + F_T1**2 - f_max**2

    plt.figure(figsize=(8, 6))
    plt.title(f"Hodograph at v={v_val:.2f} m/s, phi={phi_val:.2f} rad")
    
    # 1. Plot the Front Wheel Friction Ellipse
    plt.contour(U1, U2, C1, levels=[0], colors='blue', linewidths=2)
    # Create a dummy line for the legend
    plt.plot([], [], 'b-', linewidth=2, label='FW Non-Sliding Constraint')

    # 2. Plot the Control Bounding Box
    plt.plot([-a_max, a_max, a_max, -a_max, -a_max], 
             [-b_max, -b_max, b_max, b_max, -b_max], 'k-', linewidth=2, label='Control Limits')

    # 3. Plot the CasADi Optimal Control Point
    plt.plot(u1_opt, u2_opt, 'r*', markersize=12, label='CasADi Optimal $u^*$')

    plt.xlabel('$u_1$ (Linear Accel) [m/s²]')
    plt.ylabel('$u_2$ (Steering Rate) [rad/s]')
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.legend()
    plt.grid(True)
    plt.savefig('plot_hodograph.png')
    plt.close()


def get_discriminant_extreme_points(v, phi, L=1.0):
    """Finds the extreme points of the ellipses by setting the discriminant to 0."""
    rho = L / np.sqrt(3)
    c = (rho / L)**2
    f_max = mu * 0.5 * m * g
    k_rear = (2 * m_w / m) * ((r/2) / r)**2
    
    if abs(v) < 1e-4: return []
    
    points = []
    
    # Function to solve Delta = B^2 - 4AC = 0
    def solve_delta_zero(c1, c2, p1, q1, p2, q2):
        A_quad = c1**2 + c2**2
        E = c1*p1 + c2*p2
        F = c1*q1 + c2*q2
        G = p1**2 + p2**2
        H = p1*q1 + p2*q2
        I = q1**2 + q2**2 - f_max**2
        
        a = E**2 - A_quad * G
        b = 2 * (E*F - A_quad * H)
        c_coeff = F**2 - A_quad * I
        
        disc = b**2 - 4*a*c_coeff
        if disc < 0 or abs(a) < 1e-8: return []
        return [(-b + np.sqrt(disc))/(2*a), (-b - np.sqrt(disc))/(2*a)], A_quad, E, F

    # --- Front Wheel Extremes ---
    K_N = (m/4) * (1+c)
    B_N = (m/(4*L)) * v**2 * np.sin(phi) * np.cos(phi)
    K_T = m
    B_T = (m/(4*L)) * v**2 * np.sin(phi)**2
    
    # FW u1 extremes (vertical tangents)
    res = solve_delta_zero(K_N*v*np.cos(phi), -K_T*v*np.sin(phi), K_N*np.sin(phi), B_N, K_T*np.cos(phi), -B_T)
    if res:
        for u1 in res[0]: points.append((u1, -(res[2]*u1 + res[3])/res[1]))
        
    # FW u2 extremes (horizontal tangents)
    res = solve_delta_zero(K_N*np.sin(phi), K_T*np.cos(phi), K_N*v*np.cos(phi), B_N, -K_T*v*np.sin(phi), -B_T)
    if res:
        for u2 in res[0]: points.append((-(res[2]*u2 + res[3])/res[1], u2))

    # --- Rear Wheel Extremes ---
    K_N2 = (m/4) * (1-c)
    K_T2 = m * k_rear
    
    # RW u1 extremes
    res = solve_delta_zero(K_N2*v*np.cos(phi), K_T2*v*np.sin(phi), K_N2*np.sin(phi), B_N, -K_T2*np.cos(phi), 0.0)
    if res:
        for u1 in res[0]: points.append((u1, -(res[2]*u1 + res[3])/res[1]))
        
    # RW u2 extremes
    res = solve_delta_zero(K_N2*np.sin(phi), -K_T2*np.cos(phi), K_N2*v*np.cos(phi), B_N, K_T2*v*np.sin(phi), 0.0)
    if res:
        for u2 in res[0]: points.append((-(res[2]*u2 + res[3])/res[1], u2))

    return points

def get_ellipse_intersections(v, phi, L=1.0):
    """Finds exact intersections of FW and RW ellipses using 1D root finding."""
    rho = L / np.sqrt(3)
    c = (rho / L)**2
    f_max = mu * 0.5 * m * g
    k_rear = (2 * m_w / m) * ((r/2) / r)**2

    if abs(v) < 1e-4: return []

    K_N1 = (m/4) * (1+c)
    B_N1 = (m/(4*L)) * v**2 * np.sin(phi) * np.cos(phi)
    K_T1 = m
    B_T1 = (m/(4*L)) * v**2 * np.sin(phi)**2

    K_N2 = (m/4) * (1-c)
    K_T2 = m * k_rear

    A_RW = np.array([[K_N2 * np.sin(phi), K_N2 * v * np.cos(phi)],
                     [-K_T2 * np.cos(phi), K_T2 * v * np.sin(phi)]])

    try:
        A_RW_inv = np.linalg.inv(A_RW)
    except np.linalg.LinAlgError:
        return []

    def fw_boundary_resid(alpha):
        # 1. Point on RW ellipse from angle alpha
        F_N2_val = f_max * np.cos(alpha)
        F_T2_val = f_max * np.sin(alpha)
        
        # 2. Convert to u1, u2
        rhs = np.array([F_N2_val - B_N1, F_T2_val])
        u = A_RW_inv @ rhs
        u1, u2 = u[0], u[1]
        
        # 3. Evaluate FW ellipse equation
        F_N1_val = K_N1 * (u1 * np.sin(phi) + v * u2 * np.cos(phi)) + B_N1
        F_T1_val = K_T1 * (u1 * np.cos(phi) - v * u2 * np.sin(phi)) - B_T1
        
        return F_N1_val**2 + F_T1_val**2 - f_max**2

    # Sample alpha to find sign changes
    alphas = np.linspace(0, 2*np.pi, 60)
    resids = [fw_boundary_resid(a) for a in alphas]
    
    intersections = []
    for i in range(len(alphas) - 1):
        if resids[i] * resids[i+1] < 0:
            # Sign change detected! Find exact root.
            alpha_root = opt.brentq(fw_boundary_resid, alphas[i], alphas[i+1])
            
            # Convert back to u1, u2
            F_N2_val = f_max * np.cos(alpha_root)
            F_T2_val = f_max * np.sin(alpha_root)
            rhs = np.array([F_N2_val - B_N1, F_T2_val])
            u = A_RW_inv @ rhs
            intersections.append((u[0], u[1]))
            
    return intersections

def get_polygon_vertices(v, phi, L=1.0):
    """Gathers exact analytical points and returns the ordered Convex Hull vertices."""
    a_max, b_max = 5.0, 3 * np.pi / 2
    
    wall_points = []
    for u1_val in [-a_max, a_max]:
        for r2 in calculate_FW_u2_roots(v, phi, u1_val, L) + calculate_RW_u2_roots(v, phi, u1_val, L):
            wall_points.append((u1_val, r2))
    for u2_val in [-b_max, b_max]:
        for r1 in calculate_FW_u1_roots(v, phi, u2_val, L) + calculate_RW_u1_roots(v, phi, u2_val, L):
            wall_points.append((r1, u2_val))
            
    valid_walls = [p for p in wall_points if abs(p[0]) <= a_max+1e-4 and abs(p[1]) <= b_max+1e-4 and is_admissible(p[0], p[1], v, phi, L)]
    
    extreme_points = get_discriminant_extreme_points(v, phi, L)
    valid_extremes = [p for p in extreme_points if abs(p[0]) <= a_max+1e-4 and abs(p[1]) <= b_max+1e-4 and is_admissible(p[0], p[1], v, phi, L)]
    
    ellipse_ints = get_ellipse_intersections(v, phi, L)
    valid_ints = [p for p in ellipse_ints if abs(p[0]) <= a_max+1e-4 and abs(p[1]) <= b_max+1e-4 and is_admissible(p[0], p[1], v, phi, L)]
    
    corners = [(-a_max, -b_max), (a_max, -b_max), (a_max, b_max), (-a_max, b_max)]
    valid_corners = [p for p in corners if is_admissible(p[0], p[1], v, phi, L)]

    all_points = valid_walls + valid_extremes + valid_ints + valid_corners
    
    if len(all_points) >= 3:
        unique_points = np.unique(np.array(all_points).round(decimals=5), axis=0)
        if len(unique_points) >= 3:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(unique_points)
            return unique_points[hull.vertices]
    return np.array([])

def get_refined_polygon_vertices(v, phi, L=1.0, max_faces=64):
    """Adds intermediate boundary points adaptively based on Euclidean chord length."""
    base_points = get_polygon_vertices(v, phi, L)
    N_base = len(base_points)
    if N_base < 3: return base_points
    
    N_extra = max(0, max_faces - N_base)
    if N_extra == 0: return base_points
    
    # 1. Calculate edge lengths
    edge_lengths = []
    for i in range(N_base):
        p1 = base_points[i]
        p2 = base_points[(i + 1) % N_base]
        edge_lengths.append(np.linalg.norm(p2 - p1))
        
    total_length = sum(edge_lengths)
    if total_length < 1e-5: return base_points
    
    # 2. Allocate points proportionally based on edge length
    fractions = [N_extra * (l / total_length) for l in edge_lengths]
    allocations = [int(f) for f in fractions]
    remainders = [(f - int(f), i) for i, f in enumerate(fractions)]
    
    # Distribute leftover points to edges with highest remainders
    points_left = N_extra - sum(allocations)
    remainders.sort(reverse=True, key=lambda x: x[0])
    for k in range(points_left):
        idx = remainders[k][1]
        allocations[idx] += 1
        
    # 3. Generate intermediate points and Ray-Cast
    centroid = np.mean(base_points, axis=0)
    refined_points = list(base_points)
    
    for i in range(N_base):
        n_pts = allocations[i]
        if n_pts == 0: continue
        
        p1 = base_points[i]
        p2 = base_points[(i + 1) % N_base]
        
        # Divide the edge into equal sub-chords
        for j in range(1, n_pts + 1):
            m = p1 + (j / (n_pts + 1.0)) * (p2 - p1)
            
            # Ray outward from centroid through the sub-chord
            d = m - centroid
            norm_d = np.linalg.norm(d)
            if norm_d < 1e-5: continue
            d = d / norm_d
            
            # Bisection search to find the true boundary
            s_low, s_high = 0.0, 1.0
            def check_adm(s_val):
                pt = m + s_val * d
                if abs(pt[0]) > 5.0 + 1e-4 or abs(pt[1]) > 3 * np.pi / 2 + 1e-4: return False
                return is_admissible(pt[0], pt[1], v, phi, L)
            
            while check_adm(s_high) and s_high < 10.0:
                s_high *= 2.0
                
            for _ in range(15):
                s_mid = (s_low + s_high) / 2.0
                if check_adm(s_mid): s_low = s_mid
                else: s_high = s_mid
                
            refined_points.append(m + s_low * d)
            
    # Build the final refined convex hull
    unique_pts = np.unique(np.array(refined_points).round(decimals=5), axis=0)
    if len(unique_pts) >= 3:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(unique_pts)
        return unique_pts[hull.vertices]
    return np.array([])

def animate_hodograph(t_grid, X_opt, U_opt, L=1.0, max_faces=64, filename="parallel_park_combined.gif"):
    print(f"Generating dual-plot animation: {filename} (This will take a moment...)")
    
    rho = L / np.sqrt(3)
    c = (rho / L)**2
    f_max = mu * 0.5 * m * g
    
    k_rear = (2 * m_w / m) * ((r/2) / r)**2
    a_max, b_max = 5.0, 3 * np.pi / 2

    # Create a 1x2 grid of subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    u1_grid = np.linspace(-6, 6, 200)
    u2_grid = np.linspace(-6, 6, 200)
    U1, U2 = np.meshgrid(u1_grid, u2_grid)

    # Pre-calculate full trajectories for the static background of ax1
    X_com_full = X_opt[0, :]
    Y_com_full = X_opt[1, :]
    X_rear_full = X_com_full - L * np.cos(X_opt[2, :])
    Y_rear_full = Y_com_full - L * np.sin(X_opt[2, :])
    
    # Set fixed limits for the trajectory plot so it doesn't jump around
    x_min, x_max = min(X_rear_full) - 0.5, max(X_com_full) + 0.5
    y_min, y_max = min(Y_rear_full) - 0.5, max(Y_com_full) + 0.5

    def update(frame_idx):
        ax1.clear()
        ax2.clear()
        
        t_val = t_grid[frame_idx]
        v_val = X_opt[4, frame_idx]
        phi_val = X_opt[3, frame_idx]
        u1_opt = U_opt[0, frame_idx]
        u2_opt = U_opt[1, frame_idx]
        
        # ==========================================
        # Subplot 1: Physical Trajectory
        # ==========================================
        ax1.plot(X_com_full, Y_com_full, 'b-', alpha=0.3, label='CoM Path')
        ax1.plot(X_rear_full, Y_rear_full, 'k--', alpha=0.3, label='Rear Axle Path')
        ax1.plot(X_com_full[0], Y_com_full[0], 'go', label='Start')
        ax1.plot(X_com_full[-1], Y_com_full[-1], 'rx', label='Target')
        
        # Current car geometry
        x_c = X_opt[0, frame_idx]
        y_c = X_opt[1, frame_idx]
        theta_c = X_opt[2, frame_idx]
        
        x_rear = x_c - L * np.cos(theta_c)
        y_rear = y_c - L * np.sin(theta_c)
        x_front = x_c + L * np.cos(theta_c)
        y_front = y_c + L * np.sin(theta_c)
        
        # Draw the car as a rigid line segment
        ax1.plot([x_rear, x_front], [y_rear, y_front], 'r-', linewidth=4, label='Car Body')
        ax1.plot(x_c, y_c, 'bo', markersize=6) # Center of Mass
        ax1.plot(x_rear, y_rear, 'ko', markersize=6) # Rear Axle
        
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
        ax1.set_aspect('equal')
        ax1.set_title(f"Vehicle Trajectory (t = {t_val:.2f} s)")
        ax1.set_xlabel("X [m]")
        ax1.set_ylabel("Y [m]")
        ax1.grid(True)
        
        # ==========================================
        # Subplot 2: Control Hodograph
        # ==========================================
        F_N1 = (m/4) * ((1+c)*(U1*np.sin(phi_val) + v_val*U2*np.cos(phi_val)) + (1/L)*v_val**2 * np.sin(phi_val)*np.cos(phi_val))
        F_T1 = m * (U1*np.cos(phi_val) - v_val*U2*np.sin(phi_val) - (1/(4*L))*v_val**2 * np.sin(phi_val)**2)
        C1 = F_N1**2 + F_T1**2 - f_max**2
        
        F_N2 = (m/4) * ((1-c)*(U1*np.sin(phi_val) + v_val*U2*np.cos(phi_val)) + (1/L)*v_val**2 * np.sin(phi_val)*np.cos(phi_val))
        F_T2 = -m * k_rear * (U1*np.cos(phi_val) - v_val*U2*np.sin(phi_val))
        C2 = F_N2**2 + F_T2**2 - f_max**2

        ax2.plot([-a_max, a_max, a_max, -a_max, -a_max], [-b_max, -b_max, b_max, b_max, -b_max], 'k-', linewidth=2)
        ax2.contour(U1, U2, C1, levels=[0], colors='blue', linewidths=2)
        ax2.contour(U1, U2, C2, levels=[0], colors='red', linewidths=2)
        
        ax2.plot([0, u1_opt], [0, u2_opt], 'g-', linewidth=2, label='Optimal $u^*$')
        
        # --- 1. Wall Intersections (Blue Crosses) ---
        wall_points = []
        for u1_val in [-a_max, a_max]:
            for r2 in calculate_FW_u2_roots(v_val, phi_val, u1_val, L) + calculate_RW_u2_roots(v_val, phi_val, u1_val, L):
                wall_points.append((u1_val, r2))
        for u2_val in [-b_max, b_max]:
            for r1 in calculate_FW_u1_roots(v_val, phi_val, u2_val, L) + calculate_RW_u1_roots(v_val, phi_val, u2_val, L):
                wall_points.append((r1, u2_val))
        
        valid_walls = [p for p in wall_points if abs(p[0]) <= a_max+1e-4 and abs(p[1]) <= b_max+1e-4 and is_admissible(p[0], p[1], v_val, phi_val, L)]
        for p in valid_walls:
            ax2.plot(p[0], p[1], 'bx', markersize=12, markeredgewidth=3) # Blue Crosses
            
        # --- 2. Discriminant Extremes (Yellow Highlights) ---
        extreme_points = get_discriminant_extreme_points(v_val, phi_val, L)
        valid_extremes = [p for p in extreme_points if abs(p[0]) <= a_max+1e-4 and abs(p[1]) <= b_max+1e-4 and is_admissible(p[0], p[1], v_val, phi_val, L)]
        for p in valid_extremes:
            ax2.plot(p[0], p[1], 'y*', markersize=16, markeredgecolor='orange') # Yellow Highlights
            
        # --- 2.5 Ellipse Intersections (Magenta Dots) ---
        ellipse_ints = get_ellipse_intersections(v_val, phi_val, L)
        valid_ints = [p for p in ellipse_ints if abs(p[0]) <= a_max+1e-4 and abs(p[1]) <= b_max+1e-4 and is_admissible(p[0], p[1], v_val, phi_val, L)]
        for p in valid_ints:
            ax2.plot(p[0], p[1], 'mo', markersize=10) # Magenta Dots
            
        # --- 3. Draw The Resulting Polygons ---
        # Base Polygon (Lime)
        poly_verts = get_polygon_vertices(v_val, phi_val, L)
        if len(poly_verts) > 0:
            poly_patch = plt.Polygon(poly_verts, closed=True, facecolor='lime', edgecolor='green', alpha=0.2, linewidth=2)
            ax2.add_patch(poly_patch)
            
        # Refined Polygon (Cyan)
        refined_verts = get_refined_polygon_vertices(v_val, phi_val, L, max_faces=max_faces)
        if len(refined_verts) > 0:
            ref_patch = plt.Polygon(refined_verts, closed=True, facecolor='cyan', edgecolor='blue', alpha=0.3, linewidth=2)
            ax2.add_patch(ref_patch)
            ax2.plot(refined_verts[:, 0], refined_verts[:, 1], 'c.', markersize=8)

        # Legend entries
        ax2.plot([], [], 'bx', markersize=10, markeredgewidth=2, label='Wall Intersections')
        ax2.plot([], [], 'y*', markersize=12, label='Discriminant Extremes ($\Delta=0$)')
        ax2.plot([], [], 'mo', markersize=10, label='Ellipse Intersections')
        ax2.plot([], [], color='lime', alpha=0.2, label='Base Polygon')
        ax2.plot([], [], color='cyan', alpha=0.3, label='Refined Polygon')

        ax2.set_xlim(-6, 6)
        ax2.set_ylim(-6, 6)
        ax2.set_title(f"Hodograph (v = {v_val:.2f} m/s | phi = {phi_val:.2f} rad)")
        ax2.set_xlabel('$u_1$ (Linear Accel) [m/s²]')
        ax2.set_ylabel('$u_2$ (Steering Rate) [rad/s]')
        ax2.grid(True)

    num_controls = U_opt.shape[1]
    frames = range(num_controls)
    
    ani = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    ani.save(filename, writer='pillow', fps=10)
    print(f"Animation saved to {filename}")

def plot_controls_with_approximation(t_grid, X_opt, U_opt, L=1.0):
    rho = L / np.sqrt(3)
    c = (rho / L)**2
    f_max = mu * 0.5 * m * g
    
    N = U_opt.shape[1]
    t_steps = t_grid[:-1]
    
    u1_opt = U_opt[0, :]
    u2_opt = U_opt[1, :]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Subplot 1: u1 vs time
    ax1.plot(t_steps, u1_opt, 'b-', label='CasADi $u_1^*$')
    ax1.set_ylabel('$u_1$ [m/s²]')
    ax1.set_title('Control 1: Acceleration')
    ax1.legend()
    ax1.grid(True)
    
    # Subplot 2: u2 vs time
    ax2.plot(t_steps, u2_opt, 'b-', label='CasADi $u_2^*$')
    
    # Analytical roots logic
    analytical_t = []
    analytical_u2 = []
    analytical_t1 = []
    analytical_u1 = []
    
    for k in range(N):
        v = X_opt[4, k]
        phi = X_opt[3, k]
        u1 = u1_opt[k]
        u2_casadi = u2_opt[k]
        
        # Recalculate Front Wheel Forces to check activity
        F_N1 = (m/4) * ((1+c)*(u1*np.sin(phi) + v*u2_casadi*np.cos(phi)) + (1/L)*v**2 * np.sin(phi)*np.cos(phi))
        F_T1 = m * (u1*np.cos(phi) - v*u2_casadi*np.sin(phi) - (1/(4*L))*v**2 * np.sin(phi)**2)
        F_tot1 = np.sqrt(F_N1**2 + F_T1**2)
        
        if F_tot1 > 0.98 * f_max:
            # Case 1: Given u1, solve for u2
            roots_u2 = calculate_FW_u2_roots(v, phi, u1, L)
            if roots_u2:
                closest_u2 = min(roots_u2, key=lambda r: abs(r - u2_casadi))
                analytical_t.append(t_steps[k])
                analytical_u2.append(closest_u2)
                
            # Case 2: Given u2, solve for u1
            roots_u1 = calculate_FW_u1_roots(v, phi, u2_casadi, L)
            if roots_u1:
                closest_u1 = min(roots_u1, key=lambda r: abs(r - u1))
                analytical_t1.append(t_steps[k])
                analytical_u1.append(closest_u1)
                
    ax1.scatter(analytical_t1, analytical_u1, color='magenta', zorder=5, label='Analytical FW Boundary $u_1$')
    ax1.legend()

    ax2.scatter(analytical_t, analytical_u2, color='red', zorder=5, label='Analytical FW Boundary $u_2$')
    ax2.set_ylabel('$u_2$ [rad/s]')
    ax2.set_xlabel('Time [s]')
    ax2.set_title('Control 2: Steering Rate')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('plot_controls_with_approximation.png')
    plt.close()


def verify_open_loop_approximation(t_grid, X_opt, U_opt, L=1.0):
    """
    Verifies the open-loop trajectory by applying the closest analytical 
    corner point constraints and simulating forward via RK4.
    """
    print("Running Open-Loop Approximation Verification...")
    rho = L / np.sqrt(3)
    c = (rho / L)**2
    f_max = mu * 0.5 * m * g
    a_max, b_max = 5.0, 3 * np.pi / 2
    rho_w = r / 2 
    k_rear = (2 * m_w / m) * (rho_w / r)**2
    
    N = U_opt.shape[1]
    U_approx = np.copy(U_opt)
    
    # 1. Construct the Approximated Control Signal
    for k in range(N):
        v = X_opt[4, k]
        phi = X_opt[3, k]
        u1_casadi = U_opt[0, k]
        u2_casadi = U_opt[1, k]
        
        # Check both Front and Rear Forces
        F_N1 = (m/4) * ((1+c)*(u1_casadi*np.sin(phi) + v*u2_casadi*np.cos(phi)) + (1/L)*v**2 * np.sin(phi)*np.cos(phi))
        F_T1 = m * (u1_casadi*np.cos(phi) - v*u2_casadi*np.sin(phi) - (1/(4*L))*v**2 * np.sin(phi)**2)
        F_tot1 = np.sqrt(F_N1**2 + F_T1**2)
        
        F_N2 = (m/4) * ((1-c)*(u1_casadi*np.sin(phi) + v*u2_casadi*np.cos(phi)) + (1/L)*v**2 * np.sin(phi)*np.cos(phi))
        F_T2 = -m * k_rear * (u1_casadi*np.cos(phi) - v*u2_casadi*np.sin(phi))
        F_tot2 = np.sqrt(F_N2**2 + F_T2**2)
        
        if F_tot1 > 0.98 * f_max or F_tot2 > 0.98 * f_max:
            valid_pairs = []
            
            # Evaluate FW limits
            if F_tot1 > 0.98 * f_max:
                for u1_val in [-a_max, a_max]:
                    for r2 in calculate_FW_u2_roots(v, phi, u1_val, L):
                        if abs(r2) <= b_max + 1e-4 and is_admissible(u1_val, r2, v, phi, L): 
                            valid_pairs.append((u1_val, r2))
                for u2_val in [-b_max, b_max]:
                    for r1 in calculate_FW_u1_roots(v, phi, u2_val, L):
                        if abs(r1) <= a_max + 1e-4 and is_admissible(r1, u2_val, v, phi, L): 
                            valid_pairs.append((r1, u2_val))
                        
            # Evaluate RW limits
            if F_tot2 > 0.98 * f_max:
                for u1_val in [-a_max, a_max]:
                    for r2 in calculate_RW_u2_roots(v, phi, u1_val, L):
                        if abs(r2) <= b_max + 1e-4 and is_admissible(u1_val, r2, v, phi, L): 
                            valid_pairs.append((u1_val, r2))
                for u2_val in [-b_max, b_max]:
                    for r1 in calculate_RW_u1_roots(v, phi, u2_val, L):
                        if abs(r1) <= a_max + 1e-4 and is_admissible(r1, u2_val, v, phi, L): 
                            valid_pairs.append((r1, u2_val))
            
            # Snap to the valid root closest to the optimal solver
            if valid_pairs:
                best_pair = min(valid_pairs, key=lambda p: (p[0] - u1_casadi)**2 + (p[1] - u2_casadi)**2)
                U_approx[0, k] = best_pair[0]
                U_approx[1, k] = best_pair[1]
                
    # 2. Forward Integrate the ODE using RK4
    X_approx = np.zeros_like(X_opt)
    X_approx[:, 0] = X_opt[:, 0]
    
    for k in range(N):
        dt = t_grid[k+1] - t_grid[k]
        state_k = X_approx[:, k]
        ctrl_k = U_approx[:, k]
        
        def dyn(s, u):
            return np.array(simple_car_robot_dynamics(s, u, L)).flatten()
            
        k1 = dyn(state_k, ctrl_k)
        k2 = dyn(state_k + dt * k1 / 2, ctrl_k)
        k3 = dyn(state_k + dt * k2 / 2, ctrl_k)
        k4 = dyn(state_k + dt * k3, ctrl_k)
        
        X_approx[:, k+1] = state_k + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

    # 3. Plot the Results
    plt.figure(figsize=(18, 8))
    plt.suptitle("Open-Loop Approximation Verification")
    
    # 1. Trajectory
    plt.subplot(2, 4, 1)
    plt.plot(X_opt[0, :], X_opt[1, :], 'b-', label='Exact CoM')
    plt.plot(X_approx[0, :], X_approx[1, :], 'r--', label='Approx CoM')
    plt.plot(X_opt[0, :] - L*np.cos(X_opt[2, :]), X_opt[1, :] - L*np.sin(X_opt[2, :]), 'b-', alpha=0.3, label='Exact Rear')
    plt.plot(X_approx[0, :] - L*np.cos(X_approx[2, :]), X_approx[1, :] - L*np.sin(X_approx[2, :]), 'r--', alpha=0.3, label='Approx Rear')
    plt.plot(X_opt[0, 0], X_opt[1, 0], 'go', label='Start')
    plt.plot(X_opt[0, -1], X_opt[1, -1], 'rx', label='Target')
    plt.title('Spatial Trajectory Comparison')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    
    # 2. Velocity
    plt.subplot(2, 4, 2)
    plt.plot(t_grid, X_opt[4, :], 'b-', label='Exact v')
    plt.plot(t_grid, X_approx[4, :], 'r--', label='Approx v')
    plt.title('Velocity vs Time')
    plt.xlabel('Time [s]')
    plt.ylabel('v [m/s]')
    plt.legend()
    plt.grid(True)
    
    # 3. Heading (Theta)
    plt.subplot(2, 4, 3)
    plt.plot(t_grid, X_opt[2, :], 'b-', label='Exact Theta')
    plt.plot(t_grid, X_approx[2, :], 'r--', label='Approx Theta')
    plt.title('Heading (Theta) vs Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Theta [rad]')
    plt.legend()
    plt.grid(True)
    
    # 4. Steering Angle (Phi)
    plt.subplot(2, 4, 4)
    plt.plot(t_grid, X_opt[3, :], 'b-', label='Exact Phi')
    plt.plot(t_grid, X_approx[3, :], 'r--', label='Approx Phi')
    plt.title('Steering Angle (Phi) vs Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Phi [rad]')
    plt.legend()
    plt.grid(True)
    
    # 5. Controls
    plt.subplot(2, 4, 5)
    plt.step(t_grid[:-1], U_opt[0, :], 'b-', label='Exact u1', where='post')
    plt.step(t_grid[:-1], U_approx[0, :], 'c--', label='Approx u1', where='post')
    plt.step(t_grid[:-1], U_opt[1, :], 'g-', label='Exact u2', where='post')
    plt.step(t_grid[:-1], U_approx[1, :], 'y--', label='Approx u2', where='post')
    plt.title('Controls vs Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Controls')
    plt.legend()
    plt.grid(True)
    
    # Force Calculations
    v_arr, phi_arr = X_opt[4, :-1], X_opt[3, :-1]
    u1_arr, u2_arr = U_opt[0, :], U_opt[1, :]
    F_N1_opt = (m/4) * ((1+c)*(u1_arr*np.sin(phi_arr) + v_arr*u2_arr*np.cos(phi_arr)) + (1/L)*v_arr**2 * np.sin(phi_arr)*np.cos(phi_arr))
    F_T1_opt = m * (u1_arr*np.cos(phi_arr) - v_arr*u2_arr*np.sin(phi_arr) - (1/(4*L))*v_arr**2 * np.sin(phi_arr)**2)
    
    v_app, phi_app = X_approx[4, :-1], X_approx[3, :-1]
    u1_app, u2_app = U_approx[0, :], U_approx[1, :]
    F_N1_app = (m/4) * ((1+c)*(u1_app*np.sin(phi_app) + v_app*u2_app*np.cos(phi_app)) + (1/L)*v_app**2 * np.sin(phi_app)*np.cos(phi_app))
    F_T1_app = m * (u1_app*np.cos(phi_app) - v_app*u2_app*np.sin(phi_app) - (1/(4*L))*v_app**2 * np.sin(phi_app)**2)
    
    # 6. Forces (Front Wheels Only for Clarity)
    plt.subplot(2, 4, 6)
    plt.plot(t_grid[:-1], F_N1_opt, 'b-', label='Exact $F_{N1}$')
    plt.plot(t_grid[:-1], F_N1_app, 'c--', label='Approx $F_{N1}$')
    plt.plot(t_grid[:-1], F_T1_opt, 'g-', label='Exact $F_{T1}$')
    plt.plot(t_grid[:-1], F_T1_app, 'y--', label='Approx $F_{T1}$')
    plt.title('Front Forces vs Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Force [N]')
    plt.legend()
    plt.grid(True)
    
    # 7. Slip Constraints
    plt.subplot(2, 4, 7)
    plt.plot(t_grid[:-1], np.sqrt(F_N1_opt**2 + F_T1_opt**2), 'b-', label='Exact FW Force')
    plt.plot(t_grid[:-1], np.sqrt(F_N1_app**2 + F_T1_app**2), 'r--', label='Approx FW Force')
    plt.axhline(y=f_max, color='k', linestyle=':', label='Limit ($f_{max}$)')
    plt.title('FW Slip Constraints')
    plt.xlabel('Time [s]')
    plt.ylabel('Total Force [N]')
    plt.legend()
    plt.grid(True)
    
    # 8. State Error
    plt.subplot(2, 4, 8)
    plt.plot(t_grid, np.abs(X_opt[0, :] - X_approx[0, :]), label='|X error|')
    plt.plot(t_grid, np.abs(X_opt[1, :] - X_approx[1, :]), label='|Y error|')
    plt.plot(t_grid, np.abs(X_opt[2, :] - X_approx[2, :]), label='|Theta error|')
    plt.title('Absolute State Error')
    plt.xlabel('Time [s]')
    plt.ylabel('Absolute Error')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('verify_open_loop_approximation.pdf', format='pdf', bbox_inches='tight')
    plt.show()


def plot_area_ratio(t_grid, X_opt, L=1.0, max_faces=64):
    print("Calculating Area Ratios for Base and Refined Polygons...")
    rho = L / np.sqrt(3)
    c = (rho / L)**2
    f_max = mu * 0.5 * m * g
    k_rear = (2 * m_w / m) * ((r/2) / r)**2
    a_max, b_max = 5.0, 3 * np.pi / 2
    
    N = X_opt.shape[1]
    base_ratios = np.zeros(N-1)
    refined_ratios = np.zeros(N-1)
    
    grid_res = 300
    u1_vals = np.linspace(-a_max, a_max, grid_res)
    u2_vals = np.linspace(-b_max, b_max, grid_res)
    dU = (u1_vals[1] - u1_vals[0]) * (u2_vals[1] - u2_vals[0])
    U1, U2 = np.meshgrid(u1_vals, u2_vals)
    
    for k in range(N-1):
        v = X_opt[4, k]
        phi = X_opt[3, k]
        
        # 1. Exact Area (Numerical Integration)
        F_N1 = (m/4) * ((1+c)*(U1*np.sin(phi) + v*U2*np.cos(phi)) + (1/L)*v**2 * np.sin(phi)*np.cos(phi))
        F_T1 = m * (U1*np.cos(phi) - v*U2*np.sin(phi) - (1/(4*L))*v**2 * np.sin(phi)**2)
        F_N2 = (m/4) * ((1-c)*(U1*np.sin(phi) + v*U2*np.cos(phi)) + (1/L)*v**2 * np.sin(phi)*np.cos(phi))
        F_T2 = -m * k_rear * (U1*np.cos(phi) - v*U2*np.sin(phi))
        
        valid_mask = (F_N1**2 + F_T1**2 <= f_max**2) & (F_N2**2 + F_T2**2 <= f_max**2)
        exact_area = np.sum(valid_mask) * dU
        
        # 2. Base Polygon Area
        base_verts = get_polygon_vertices(v, phi, L)
        base_area = 0.0
        if len(base_verts) >= 3:
            from scipy.spatial import ConvexHull
            base_area = ConvexHull(base_verts).volume
            
        # 3. Refined Polygon Area
        refined_verts = get_refined_polygon_vertices(v, phi, L, max_faces=max_faces)
        refined_area = 0.0
        if len(refined_verts) >= 3:
            from scipy.spatial import ConvexHull
            refined_area = ConvexHull(refined_verts).volume
            
        # 4. Calculate Ratios
        if exact_area > 0:
            base_ratios[k] = base_area / exact_area
            refined_ratios[k] = refined_area / exact_area
        else:
            base_ratios[k] = 1.0
            refined_ratios[k] = 1.0
            
    # --- Plotting (Ratios Only) ---
    plt.figure(figsize=(10, 6))
    plt.plot(t_grid[:-1], base_ratios * 100.0, 'r-', linewidth=2, label='Base Polygon Ratio')
    plt.plot(t_grid[:-1], refined_ratios * 100.0, 'c--', linewidth=2, label='Refined Polygon Ratio')
    
    plt.axhline(100, color='gray', linestyle=':')
    plt.xlabel('Time [s]')
    plt.ylabel('Coverage Ratio (Polygon / Exact) [%]')
    plt.title('Polygonal Approximation Coverage over Time')
    plt.ylim(0, 105)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()  
    plt.savefig('area_ratio_analysis.png')
    plt.show()

def plot_area_ratio_comparison(t_grid, X_opt, L=1.0, budgets=[4, 8, 12, 16, 20, 24], filename='plot_area_ratio_comparison.png'):
    print(f"Calculating Area Ratios for Budgets: {budgets} (This may take a minute)...")
    rho = L / np.sqrt(3)
    c = (rho / L)**2
    f_max = mu * 0.5 * m * g
    k_rear = (2 * m_w / m) * ((r/2) / r)**2
    a_max, b_max = 5.0, 3 * np.pi / 2
    
    N = X_opt.shape[1]
    ratios_dict = {b: np.zeros(N-1) for b in budgets}
    
    # Grid for Exact Area Numerical Integration
    grid_res = 300
    u1_vals = np.linspace(-a_max, a_max, grid_res)
    u2_vals = np.linspace(-b_max, b_max, grid_res)
    dU = (u1_vals[1] - u1_vals[0]) * (u2_vals[1] - u2_vals[0])
    U1, U2 = np.meshgrid(u1_vals, u2_vals)
    
    for k in range(N-1):
        v = X_opt[4, k]
        phi = X_opt[3, k]
        
        # 1. Exact Area (Calculated once per time step)
        F_N1 = (m/4) * ((1+c)*(U1*np.sin(phi) + v*U2*np.cos(phi)) + (1/L)*v**2 * np.sin(phi)*np.cos(phi))
        F_T1 = m * (U1*np.cos(phi) - v*U2*np.sin(phi) - (1/(4*L))*v**2 * np.sin(phi)**2)
        F_N2 = (m/4) * ((1-c)*(U1*np.sin(phi) + v*U2*np.cos(phi)) + (1/L)*v**2 * np.sin(phi)*np.cos(phi))
        F_T2 = -m * k_rear * (U1*np.cos(phi) - v*U2*np.sin(phi))
        
        valid_mask = (F_N1**2 + F_T1**2 <= f_max**2) & (F_N2**2 + F_T2**2 <= f_max**2)
        exact_area = np.sum(valid_mask) * dU
        
        # 2. Polygon Area for each budget
        for b in budgets:
            refined_verts = get_refined_polygon_vertices(v, phi, L, max_faces=b)
            refined_area = 0.0
            if len(refined_verts) >= 3:
                from scipy.spatial import ConvexHull
                refined_area = ConvexHull(refined_verts).volume
                
            if exact_area > 0:
                ratios_dict[b][k] = refined_area / exact_area
            else:
                ratios_dict[b][k] = 1.0
                
    # --- Plotting ---
    plt.figure(figsize=(12, 7))
    
    # Use a colormap to differentiate the lines cleanly
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(budgets)))
    
    for idx, b in enumerate(budgets):
        plt.plot(t_grid[:-1], ratios_dict[b] * 100.0, color=colors[idx], linewidth=2, label=f'{b} points')
        
    plt.axhline(100, color='gray', linestyle=':')
    plt.xlabel('Time [s]')
    plt.ylabel('Coverage Ratio (Polygon / Exact) [%]')
    plt.title('Polygonal Approximation Coverage vs. Point Budget')
    plt.ylim(0, 105)
    
    # Place legend outside or in a clear spot
    plt.legend(loc='lower right', title="Number of Points")
    plt.grid(True)
    plt.tight_layout()  
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()

if __name__ == "__main__":
    L_val = 1.0
    num_approx_points = 12  # Adaptive Meshing Strategy budget
    # Center of mass starts at X=1 (rear wheels at X=0)
    S_start = [L_val, 0, 0, 0, 0]
    # Center of mass ends at X=1 (rear wheels at X=0), parked at Y=1
    S_target = [L_val, 1, 0, 0, 0] 
    # S_start = [L_val, 0, 0, 0, 10]
    # S_target = [L_val, 3, np.pi, 0, 10] 
    
    T_opt, X_opt, U_opt, t_grid, solver_time = solve_time_optimal_problem(S_start, S_target, L=L_val)
    # plot_solution(t_grid, X_opt, U_opt, T_opt, L=L_val)

    idx = int(0.15 * U_opt.shape[1])
    print(f"index time : {t_grid[idx]:.3f} [s]")

    v_test = X_opt[4, idx]
    phi_test = X_opt[3, idx]
    u1_test = U_opt[0, idx]
    u2_test = U_opt[1, idx]
    
    plot_hodograph(v_test, phi_test, u1_test, u2_test, L=L_val)

    animate_hodograph(t_grid, X_opt, U_opt, L=L_val, max_faces=num_approx_points, filename="parallel_park_combined.gif")
    
    plot_controls_with_approximation(t_grid, X_opt, U_opt, L=L_val)
    
    verify_open_loop_approximation(t_grid, X_opt, U_opt, L=L_val)
    plot_area_ratio_comparison(t_grid, X_opt, L=L_val, budgets=[4, 8, 12, 16], filename='area_ratio_comparison.png')
