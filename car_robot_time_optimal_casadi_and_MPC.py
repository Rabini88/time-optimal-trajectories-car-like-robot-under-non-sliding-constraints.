import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial import ConvexHull
import scipy.optimize as opt

# Global Physics Parameters
m = 20.0
g = 9.81
mu = 1.0
mu_poly = 1.0
m_w = 1.0
r = 0.1

def simple_car_robot_dynamics(S, u, L=1.0):
    """
    Calculates the state derivative dS/dt based on the Book Chapter (Eq 2.29).
    state: [x, y, theta, phi, v]  (Note: x, y track the Center of Mass)
    u: [u1, u2] (tangential acceleration, steering rate)
    """
    x = S[0]
    y = S[1]
    theta = S[2]
    phi = S[3]
    v = S[4]

    u1 = u[0]
    u2 = u[1]

    # Kinematics tracking the Center of Mass (Eq 2.29)
    dx = v * (ca.cos(theta) * ca.cos(phi) - 0.5 * ca.sin(theta) * ca.sin(phi))
    dy = v * (ca.sin(theta) * ca.cos(phi) + 0.5 * ca.cos(theta) * ca.sin(phi))
    dtheta = (v * ca.sin(phi)) / (2 * L)
    dphi = u2
    dv = u1

    return ca.vertcat(dx, dy, dtheta, dphi, dv)

def get_rk4_function(L=1.0):
    """Creates a CasADi Function for RK4 integration to use in MPC and Plant."""
    S = ca.MX.sym('S', 5)
    U = ca.MX.sym('U', 2)
    dt = ca.MX.sym('dt')
    
    k1 = simple_car_robot_dynamics(S, U, L)
    k2 = simple_car_robot_dynamics(S + dt*k1/2, U, L)
    k3 = simple_car_robot_dynamics(S + dt*k2/2, U, L)
    k4 = simple_car_robot_dynamics(S + dt*k3, U, L)
    S_next = S + dt*(k1 + 2*k2 + 2*k3 + k4)/6
    
    return ca.Function('F_rk4', [S, U, dt], [S_next])


def solve_time_optimal_problem(S_start, S_target, N=500, L=1.0, phi_max=np.pi/2, a_max=5, b_max=3*np.pi/2, v_max=20.0):
    T_guess = 2.0
    
    rho = L / ca.sqrt(3) 
    c = (rho / L)**2     
    
    rho_w = r / 2 
    k_rear = (2 * m_w / m) * (rho_w / r)**2 

    opti = ca.Opti()
    T = opti.variable()
    X = opti.variable(5, N+1)
    U = opti.variable(2, N)

    opti.minimize(T)
    opti.subject_to(T >= 0.1)

    # Boundary Constraints
    opti.subject_to(X[:,0] == S_start)
    opti.subject_to(X[0,N] == S_target[0])
    opti.subject_to(X[1,N] == S_target[1])
    opti.subject_to(X[2:,N] == S_target[2:])

    dt = T/N
    for k in range(N):
        # Kinematics (RK4)
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

        # Normal Forces
        F_N1 = (m/4) * ((1+c)*(u1*ca.sin(phi) + v*u2*ca.cos(phi)) + (1/L)*v**2 * ca.sin(phi)*ca.cos(phi))
        F_N2 = (m/4) * ((1-c)*(u1*ca.sin(phi) + v*u2*ca.cos(phi)) + (1/L)*v**2 * ca.sin(phi)*ca.cos(phi))
        
        # Tangential Forces
        F_T1 = m * (u1*ca.cos(phi) - v*u2*ca.sin(phi) - (1/(4*L))*v**2 * ca.sin(phi)**2)
        F_T2 = -m * k_rear * (u1*ca.cos(phi) - v*u2*ca.sin(phi))

        # Friction Constraints
        f_max = mu * 0.5 * m * g
        opti.subject_to( (F_T1 / f_max)**2 + (F_N1 / f_max)**2 <= 1 )
        opti.subject_to( (F_T2 / f_max)**2 + (F_N2 / f_max)**2 <= 1 )

        # Control & State limits
        opti.subject_to(opti.bounded(-a_max, u1, a_max))
        opti.subject_to(opti.bounded(-b_max, u2, b_max))
        opti.subject_to(opti.bounded(-phi_max, phi, phi_max))
        opti.subject_to(opti.bounded(-v_max, v, v_max))

    # Warm Start (matching dynamic_car_time_optimal_casadi.py)
    opti.set_initial(T, T_guess)
    mid = N // 2

    X_guess = np.concatenate([np.linspace(S_start[0], S_start[0]+1.0, mid),
                               np.linspace(S_start[0]+1.0, S_target[0], N+1-mid)])
    Y_guess = np.linspace(S_start[1], S_target[1], N+1)
    theta_guess = np.concatenate([np.linspace(0, np.pi/7, mid),
                                   np.linspace(np.pi/7, 0, N+1-mid)])
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
    
    T_opt = sol.value(T)
    X_opt = sol.value(X)
    U_opt = sol.value(U)
    t_grid = np.linspace(0, T_opt, N+1)

    print(f"Optimal time T: {T_opt}")
    print(f"Solver time: {solver_time}")
    return T_opt, X_opt, U_opt, t_grid, solver_time

def setup_mpc(dt_val, N_mpc=15, L=1.0):
    opti = ca.Opti()
    
    # MPC Parameters
    x_curr = opti.parameter(5)
    x_targ = opti.parameter(5)
    
    # MPC Variables
    X = opti.variable(5, N_mpc+1)
    U = opti.variable(2, N_mpc)
    
    # Cost: Heavily penalize spatial error, slightly penalize control effort to stop chattering
    state_error = X[:, -1] - x_targ
    control_effort = ca.sumsqr(U[0, :]) + 10.0 * ca.sumsqr(U[1, :]) # Higher penalty on steering
    
    cost = 1e4 * ca.sumsqr(state_error) + 1e-4 * control_effort 
    opti.minimize(cost)
    
    opti.subject_to(X[:, 0] == x_curr)
    
    F_rk4 = get_rk4_function(L)
    m, g, mu, a_max, b_max = 20.0, 9.81, 1.0, 5.0, 3*np.pi/2
    rho = L / ca.sqrt(3)
    c = (rho / L)**2     
    m_w, r = 1.0, 0.1
    k_rear = (2 * m_w / m) * ((r/2) / r)**2 
    f_max = mu * 0.5 * m * g

    for k in range(N_mpc):
        opti.subject_to(X[:,k+1] == F_rk4(X[:,k], U[:,k], dt_val)) # Use fixed dt_val
        v, phi, u1, u2 = X[4,k], X[3,k], U[0,k], U[1,k]

        # Exact Nonlinear Friction Limits inside the MPC
        F_N1 = (m/4) * ((1+c)*(u1*ca.sin(phi) + v*u2*ca.cos(phi)) + (1/L)*v**2 * ca.sin(phi)*ca.cos(phi))
        F_N2 = (m/4) * ((1-c)*(u1*ca.sin(phi) + v*u2*ca.cos(phi)) + (1/L)*v**2 * ca.sin(phi)*np.cos(phi))
        F_T1 = m * (u1*ca.cos(phi) - v*u2*ca.sin(phi) - (1/(4*L))*v**2 * np.sin(phi)**2)
        F_T2 = -m * k_rear * (u1*ca.cos(phi) - v*u2*ca.sin(phi))

        opti.subject_to( F_T1**2 + F_N1**2 <= f_max**2 )
        opti.subject_to( F_T2**2 + F_N2**2 <= f_max**2 )

        opti.subject_to(opti.bounded(-a_max, u1, a_max))
        opti.subject_to(opti.bounded(-b_max, u2, b_max))
        opti.subject_to(opti.bounded(-np.pi/2, phi, np.pi/2))
        opti.subject_to(opti.bounded(-20.0, v, 20.0))

    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
    opti.solver('ipopt', opts)
    
    return opti, x_curr, x_targ, X, U

def calculate_FW_u2_roots(v, phi, u1, L=1.0, mu_val=None):
    if mu_val is None: mu_val = mu_poly
    rho = L / np.sqrt(3)
    c = (rho / L)**2
    f_max = mu_val * 0.5 * m * g
    if abs(v) < 1e-4: return []
    K_N = (m/4) * (1+c)
    B_N = (m/(4*L)) * v**2 * np.sin(phi) * np.cos(phi)
    K_T = m
    B_T = (m/(4*L)) * v**2 * np.sin(phi)**2
    c1 = K_N * v * np.cos(phi)
    d1 = K_N * u1 * np.sin(phi) + B_N
    c2 = -K_T * v * np.sin(phi)
    d2 = K_T * u1 * np.cos(phi) - B_T
    A = c1**2 + c2**2
    B = 2 * (c1*d1 + c2*d2)
    C = d1**2 + d2**2 - f_max**2
    disc = B**2 - 4*A*C
    if disc < 0: return []
    return [(-B + np.sqrt(disc))/(2*A), (-B - np.sqrt(disc))/(2*A)]

def calculate_FW_u1_roots(v, phi, u2, L=1.0, mu_val=None):
    if mu_val is None: mu_val = mu_poly
    rho = L / np.sqrt(3)
    c = (rho / L)**2
    f_max = mu_val * 0.5 * m * g
    K_N = (m/4) * (1+c)
    B_N = (m/(4*L)) * v**2 * np.sin(phi) * np.cos(phi)
    K_T = m
    B_T = (m/(4*L)) * v**2 * np.sin(phi)**2
    c1 = K_N * np.sin(phi)
    d1 = K_N * v * u2 * np.cos(phi) + B_N
    c2 = K_T * np.cos(phi)
    d2 = -K_T * v * u2 * np.sin(phi) - B_T
    A = c1**2 + c2**2
    B = 2 * (c1*d1 + c2*d2)
    C = d1**2 + d2**2 - f_max**2
    if abs(A) < 1e-8: return []
    disc = B**2 - 4*A*C
    if disc < 0: return []
    return [(-B + np.sqrt(disc))/(2*A), (-B - np.sqrt(disc))/(2*A)]

def calculate_RW_u2_roots(v, phi, u1, L=1.0, mu_val=None):
    if mu_val is None: mu_val = mu_poly
    rho = L / np.sqrt(3)
    c = (rho / L)**2
    f_max = mu_val * 0.5 * m * g
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

def calculate_RW_u1_roots(v, phi, u2, L=1.0, mu_val=None):
    if mu_val is None: mu_val = mu_poly
    rho = L / np.sqrt(3)
    c = (rho / L)**2
    f_max = mu_val * 0.5 * m * g
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

def is_admissible(u1_test, u2_test, v, phi, L=1.0, mu_val=None):
    if mu_val is None: mu_val = mu_poly
    rho = L / np.sqrt(3)
    c = (rho / L)**2
    f_max = mu_val * 0.5 * m * g
    k_rear = (2 * m_w / m) * ((r/2) / r)**2
    
    F_N1 = (m/4) * ((1+c)*(u1_test*np.sin(phi) + v*u2_test*np.cos(phi)) + (1/L)*v**2 * np.sin(phi)*np.cos(phi))
    F_T1 = m * (u1_test*np.cos(phi) - v*u2_test*np.sin(phi) - (1/(4*L))*v**2 * np.sin(phi)**2)
    if F_N1**2 + F_T1**2 > f_max**2 + 1e-2: return False 
    
    F_N2 = (m/4) * ((1-c)*(u1_test*np.sin(phi) + v*u2_test*np.cos(phi)) + (1/L)*v**2 * np.sin(phi)*np.cos(phi))
    F_T2 = -m * k_rear * (u1_test*np.cos(phi) - v*u2_test*np.sin(phi))
    if F_N2**2 + F_T2**2 > f_max**2 + 1e-2: return False
    
    return True

def get_discriminant_extreme_points(v, phi, L=1.0, mu_val=None):
    """Finds the extreme points of the ellipses by setting the discriminant to 0."""
    if mu_val is None: mu_val = mu_poly
    rho = L / np.sqrt(3)
    c = (rho / L)**2
    f_max = mu_val * 0.5 * m * g
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

def get_ellipse_intersections(v, phi, L=1.0, mu_val=None):
    """Finds exact intersections of FW and RW ellipses using 1D root finding."""
    if mu_val is None: mu_val = mu_poly
    rho = L / np.sqrt(3)
    c = (rho / L)**2
    f_max = mu_val * 0.5 * m * g
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

def get_polygonal_constraints(v, phi, L=1.0, max_faces=24, mu_val=None, u_ref_k=None):
    if mu_val is None: mu_val = mu_poly
    refined_verts = get_refined_polygon_vertices(v, phi, L, max_faces, mu_val)
    
    A = None
    b_vec = None
    
    # 1. Try to build the nominal convex hull
    if len(refined_verts) >= 3:
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(refined_verts)
            A, b_vec = hull.equations[:, :-1], -hull.equations[:, -1]
        except Exception:
            pass # Catch Qhull collinear errors
            
    # 2. Reference Control Fallback
    if A is None or b_vec is None:
        if u_ref_k is not None:
            # Center the fallback box exactly on the offline optimal control
            center = u_ref_k
        elif len(refined_verts) > 0:
            center = np.mean(refined_verts, axis=0)
        else:
            center = np.array([0.0, 0.0])
            
        # Build a small box around the reference control (0.1 padding gives the QP room to breathe)
        eps = 0.1
        A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b_vec = np.array([center[0] + eps, -center[0] + eps, 
                          center[1] + eps, -center[1] + eps])
    
    # 3. Pad the arrays for CasADi fixed-size parameters
    num_faces = len(b_vec)
    A_padded, b_padded = np.zeros((max_faces, 2)), 1000.0 * np.ones(max_faces)
    A_padded[:min(num_faces, max_faces)] = A[:max_faces]
    b_padded[:min(num_faces, max_faces)] = b_vec[:max_faces]
    
    return A_padded, b_padded

def get_polygon_vertices(v, phi, L=1.0, mu_val=None):
    """Gathers exact analytical points and returns the ordered Convex Hull vertices."""
    if mu_val is None: mu_val = mu_poly
    a_max, b_max = 5.0, 3 * np.pi / 2
    
    wall_points = []
    for u1_val in [-a_max, a_max]:
        for r2 in calculate_FW_u2_roots(v, phi, u1_val, L, mu_val) + calculate_RW_u2_roots(v, phi, u1_val, L, mu_val):
            wall_points.append((u1_val, r2))
    for u2_val in [-b_max, b_max]:
        for r1 in calculate_FW_u1_roots(v, phi, u2_val, L, mu_val) + calculate_RW_u1_roots(v, phi, u2_val, L, mu_val):
            wall_points.append((r1, u2_val))
            
    valid_walls = [p for p in wall_points if abs(p[0]) <= a_max+1e-4 and abs(p[1]) <= b_max+1e-4 and is_admissible(p[0], p[1], v, phi, L, mu_val)]
    
    extreme_points = get_discriminant_extreme_points(v, phi, L, mu_val)
    valid_extremes = [p for p in extreme_points if abs(p[0]) <= a_max+1e-4 and abs(p[1]) <= b_max+1e-4 and is_admissible(p[0], p[1], v, phi, L, mu_val)]
    
    ellipse_ints = get_ellipse_intersections(v, phi, L, mu_val)
    valid_ints = [p for p in ellipse_ints if abs(p[0]) <= a_max+1e-4 and abs(p[1]) <= b_max+1e-4 and is_admissible(p[0], p[1], v, phi, L, mu_val)]
    
    corners = [(-a_max, -b_max), (a_max, -b_max), (a_max, b_max), (-a_max, b_max)]
    valid_corners = [p for p in corners if is_admissible(p[0], p[1], v, phi, L, mu_val)]

    all_points = valid_walls + valid_extremes + valid_ints + valid_corners
    
    if len(all_points) >= 3:
        unique_points = np.unique(np.array(all_points).round(decimals=5), axis=0)
        if len(unique_points) >= 3:
            hull = ConvexHull(unique_points)
            return unique_points[hull.vertices]
    return np.array([])

def get_refined_polygon_vertices(v, phi, L=1.0, max_faces=24, mu_val=None):
    """Adds intermediate boundary points adaptively based on Euclidean chord length."""
    if mu_val is None: mu_val = mu_poly
    base_points = get_polygon_vertices(v, phi, L, mu_val)
    N_base = len(base_points)
    if N_base < 3: return base_points
    
    N_extra = max(0, max_faces - N_base)
    if N_extra == 0: return base_points
    
    edge_lengths = [np.linalg.norm(base_points[(i + 1) % N_base] - base_points[i]) for i in range(N_base)]
    total_length = sum(edge_lengths)
    if total_length < 1e-5: return base_points
    
    fractions = [N_extra * (l / total_length) for l in edge_lengths]
    allocations = [int(f) for f in fractions]
    remainders = [(f - int(f), i) for i, f in enumerate(fractions)]
    
    points_left = N_extra - sum(allocations)
    remainders.sort(reverse=True, key=lambda x: x[0])
    for k in range(points_left):
        allocations[remainders[k][1]] += 1
        
    centroid = np.mean(base_points, axis=0)
    refined_points = list(base_points)
    
    for i in range(N_base):
        n_pts = allocations[i]
        if n_pts == 0: continue
        p1, p2 = base_points[i], base_points[(i + 1) % N_base]
        
        for j in range(1, n_pts + 1):
            m = p1 + (j / (n_pts + 1.0)) * (p2 - p1)
            d = m - centroid
            norm_d = np.linalg.norm(d)
            if norm_d < 1e-5: continue
            d = d / norm_d
            
            s_low, s_high = 0.0, 1.0
            def check_adm(s_val):
                pt = m + s_val * d
                if abs(pt[0]) > 5.0 + 1e-4 or abs(pt[1]) > 3 * np.pi / 2 + 1e-4: return False
                return is_admissible(pt[0], pt[1], v, phi, L, mu_val)
            
            while check_adm(s_high) and s_high < 10.0: s_high *= 2.0
            for _ in range(15):
                s_mid = (s_low + s_high) / 2.0
                if check_adm(s_mid): s_low = s_mid
                else: s_high = s_mid
            refined_points.append(m + s_low * d)
            
    unique_pts = np.unique(np.array(refined_points).round(decimals=5), axis=0)
    if len(unique_pts) >= 3:
        from scipy.spatial import ConvexHull
        return unique_pts[ConvexHull(unique_pts).vertices]
    return np.array([])

def setup_mpc_polygonal(dt_val, N_mpc=15, L=1.0, max_faces=24):
    opti = ca.Opti()
    
    # MPC Parameters
    x_curr = opti.parameter(5)
    x_targ = opti.parameter(5)
    A_poly = opti.parameter(max_faces, 2)
    b_poly = opti.parameter(max_faces)
    
    # MPC Variables
    X = opti.variable(5, N_mpc+1)
    U = opti.variable(2, N_mpc)
    
    state_error = X[:, -1] - x_targ
    control_effort = ca.sumsqr(U[0, :]) + 10.0 * ca.sumsqr(U[1, :])
    
    cost = 1e4 * ca.sumsqr(state_error) + 1e-4 * control_effort 
    opti.minimize(cost)
    opti.subject_to(X[:, 0] == x_curr)
    
    F_rk4 = get_rk4_function(L)
    a_max, b_max = 5.0, 3*np.pi/2

    for k in range(N_mpc):
        opti.subject_to(X[:,k+1] == F_rk4(X[:,k], U[:,k], dt_val))
        v, phi, u1, u2 = X[4,k], X[3,k], U[0,k], U[1,k]

        # Apply the frozen local polygon to the ENTIRE prediction horizon
        opti.subject_to(A_poly @ U[:, k] <= b_poly)

        opti.subject_to(opti.bounded(-a_max, u1, a_max))
        opti.subject_to(opti.bounded(-b_max, u2, b_max))
        opti.subject_to(opti.bounded(-np.pi/2, phi, np.pi/2))
        opti.subject_to(opti.bounded(-20.0, v, 20.0))

    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
    opti.solver('ipopt', opts)
    
    return opti, x_curr, x_targ, A_poly, b_poly, X, U

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
    plt.savefig('plot_solution_exact.png')
    plt.show()

def plot_hodograph(v_val, phi_val, u1_opt, u2_opt, L=1.0):
    rho = L / np.sqrt(3)
    c = (rho / L)**2
    f_max = mu * 0.5 * m * g
    a_max, b_max = 5.0, 3 * np.pi / 2

    u1_grid = np.linspace(-6, 6, 400)
    u2_grid = np.linspace(-6, 6, 400)
    U1, U2 = np.meshgrid(u1_grid, u2_grid)

    F_N1 = (m/4) * ((1+c)*(U1*np.sin(phi_val) + v_val*U2*np.cos(phi_val)) + (1/L)*v_val**2 * np.sin(phi_val)*np.cos(phi_val))
    F_T1 = m * (U1*np.cos(phi_val) - v_val*U2*np.sin(phi_val) - (1/(4*L))*v_val**2 * np.sin(phi_val)**2)
    C1 = F_N1**2 + F_T1**2 - f_max**2

    plt.figure(figsize=(8, 6))
    plt.title(f"Hodograph at v={v_val:.2f} m/s, phi={phi_val:.2f} rad")
    
    plt.contour(U1, U2, C1, levels=[0], colors='blue', linewidths=2)
    plt.plot([], [], 'b-', linewidth=2, label='FW Non-Sliding Constraint')
    plt.plot([-a_max, a_max, a_max, -a_max, -a_max], 
             [-b_max, -b_max, b_max, b_max, -b_max], 'k-', linewidth=2, label='Control Limits')
    plt.plot(u1_opt, u2_opt, 'r*', markersize=12, label='CasADi Optimal $u^*$')

    plt.xlabel('$u_1$ (Linear Accel) [m/s²]')
    plt.ylabel('$u_2$ (Steering Rate) [rad/s]')
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.legend()
    plt.grid(True)
    plt.savefig('plot_hodograph_exact.png')
    plt.close()

def animate_hodograph(t_grid, X_opt, U_opt, L=1.0, max_faces=24, filename="exact_trajectory.gif"):
    print(f"Generating dual-plot animation: {filename} (This will take a moment...)")
    
    rho = L / np.sqrt(3)
    c = (rho / L)**2
    f_max = mu * 0.5 * m * g
    k_rear = (2 * m_w / m) * ((r/2) / r)**2
    a_max, b_max = 5.0, 3 * np.pi / 2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    u1_grid = np.linspace(-6, 6, 200)
    u2_grid = np.linspace(-6, 6, 200)
    U1, U2 = np.meshgrid(u1_grid, u2_grid)

    X_com_full = X_opt[0, :]
    Y_com_full = X_opt[1, :]
    X_rear_full = X_com_full - L * np.cos(X_opt[2, :])
    Y_rear_full = Y_com_full - L * np.sin(X_opt[2, :])
    
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
        
        # Subplot 1: Physical Trajectory
        ax1.plot(X_com_full, Y_com_full, 'b-', alpha=0.3, label='CoM Path')
        ax1.plot(X_rear_full, Y_rear_full, 'k--', alpha=0.3, label='Rear Axle Path')
        ax1.plot(X_com_full[0], Y_com_full[0], 'go', label='Start')
        ax1.plot(X_com_full[-1], Y_com_full[-1], 'rx', label='Target')
        
        x_c = X_opt[0, frame_idx]
        y_c = X_opt[1, frame_idx]
        theta_c = X_opt[2, frame_idx]
        
        x_rear = x_c - L * np.cos(theta_c)
        y_rear = y_c - L * np.sin(theta_c)
        x_front = x_c + L * np.cos(theta_c)
        y_front = y_c + L * np.sin(theta_c)
        
        ax1.plot([x_rear, x_front], [y_rear, y_front], 'r-', linewidth=4, label='Car Body')
        ax1.plot(x_c, y_c, 'bo', markersize=6) 
        ax1.plot(x_rear, y_rear, 'ko', markersize=6) 
        
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
        ax1.set_aspect('equal')
        ax1.set_title(f"Vehicle Trajectory (t = {t_val:.2f} s)")
        ax1.set_xlabel("X [m]")
        ax1.set_ylabel("Y [m]")
        ax1.grid(True)
        
        # Subplot 2: Control Hodograph
        F_N1 = (m/4) * ((1+c)*(U1*np.sin(phi_val) + v_val*U2*np.cos(phi_val)) + (1/L)*v_val**2 * np.sin(phi_val)*np.cos(phi_val))
        F_T1 = m * (U1*np.cos(phi_val) - v_val*U2*np.sin(phi_val) - (1/(4*L))*v_val**2 * np.sin(phi_val)**2)
        C1 = F_N1**2 + F_T1**2 - f_max**2
        
        F_N2 = (m/4) * ((1-c)*(U1*np.sin(phi_val) + v_val*U2*np.cos(phi_val)) + (1/L)*v_val**2 * np.sin(phi_val)*np.cos(phi_val))
        F_T2 = -m * k_rear * (U1*np.cos(phi_val) - v_val*U2*np.sin(phi_val))
        C2 = F_N2**2 + F_T2**2 - f_max**2

        ax2.plot([-a_max, a_max, a_max, -a_max, -a_max], [-b_max, -b_max, b_max, b_max, -b_max], 'k-', linewidth=2)
        ax2.contour(U1, U2, C1, levels=[0], colors='blue', linewidths=2)
        ax2.contour(U1, U2, C2, levels=[0], colors='red', linewidths=2)
        
        # 3. The Refined Convex Hull Polygon
        refined_verts = get_refined_polygon_vertices(v_val, phi_val, L, max_faces=max_faces)
        if len(refined_verts) > 0:
            poly_patch = plt.Polygon(refined_verts, closed=True, facecolor='cyan', edgecolor='blue', alpha=0.3, linewidth=2, label='Refined Polygon Admissible Zone')
            ax2.add_patch(poly_patch)

        ax2.plot([0, u1_opt], [0, u2_opt], 'g-', linewidth=2, label='Optimal $u^*$')
        ax2.plot(u1_opt, u2_opt, 'g*', markersize=12)

        ax2.set_xlim(-6, 6)
        ax2.set_ylim(-6, 6)
        ax2.set_title(f"Hodograph (v = {v_val:.2f} m/s | phi = {phi_val:.2f} rad)")
        ax2.set_xlabel('$u_1$ (Linear Accel) [m/s²]')
        ax2.set_ylabel('$u_2$ (Steering Rate) [rad/s]')
        ax2.grid(True)

    num_controls = U_opt.shape[1]
    
    # Pass the exact integer to avoid the 100-frame cutoff
    ani = animation.FuncAnimation(fig, update, frames=num_controls, blit=False)
    ani.save(filename, writer='pillow', fps=10)
    print(f"Animation saved to {filename}")

def plot_mpc_vs_exact(t_ref, X_ref, U_ref, t_sim, X_sim, U_sim, t_sim_poly, X_sim_poly, U_sim_poly, L=1.0, save_path=None, show_plot=False):
    m, g, mu = 20.0, 9.81, 1.0
    rho = L / np.sqrt(3)
    c = (rho / L)**2
    f_max = mu * 0.5 * m * g
    m_w, r = 1.0, 0.1
    k_rear = (2 * m_w / m) * ((r/2) / r)**2
    
    # For plotting, we use the full reference trajectory
    X_ref_cut = X_ref
    U_ref_cut = U_ref
    t_ref_cut = t_ref

    plt.figure(figsize=(18, 8))
    plt.suptitle("MPC Closed-Loop Verification vs Exact Reference")
    
    # 1. Spatial Trajectory
    plt.subplot(2, 4, 1)
    # Plot full reference path
    plt.plot(X_ref[0, :], X_ref[1, :], 'b-', linewidth=2, label='Exact CoM')
    plt.plot(X_sim[0, :], X_sim[1, :], 'r--', linewidth=2, label='MPC CoM')
    plt.plot(X_sim_poly[0, :], X_sim_poly[1, :], 'g:', linewidth=2, label='Poly MPC CoM')
    
    # Plot rear axle paths
    plt.plot(X_ref[0, :] - L*np.cos(X_ref[2, :]), X_ref[1, :] - L*np.sin(X_ref[2, :]), 'b-', alpha=0.3, label='Exact Rear')
    plt.plot(X_sim[0, :] - L*np.cos(X_sim[2, :]), X_sim[1, :] - L*np.sin(X_sim[2, :]), 'r--', alpha=0.3, label='MPC Rear')
    plt.plot(X_sim_poly[0, :] - L*np.cos(X_sim_poly[2, :]), X_sim_poly[1, :] - L*np.sin(X_sim_poly[2, :]), 'g:', alpha=0.3, label='Poly MPC Rear')
    
    plt.plot(X_ref[0, 0], X_ref[1, 0], 'go', label='Start')
    plt.title('Spatial Trajectory Comparison')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    
    # 2. Velocity
    plt.subplot(2, 4, 2)
    plt.plot(t_ref_cut, X_ref_cut[4, :], 'b-', label='Exact v')
    plt.plot(t_sim, X_sim[4, :], 'r--', label='MPC v')
    plt.plot(t_sim_poly, X_sim_poly[4, :], 'g:', label='Poly MPC v')
    plt.title('Velocity vs Time')
    plt.xlabel('Time [s]')
    plt.ylabel('v [m/s]')
    plt.legend()
    plt.grid(True)
    
    # 3. Heading (Theta)
    plt.subplot(2, 4, 3)
    plt.plot(t_ref_cut, X_ref_cut[2, :], 'b-', label='Exact Theta')
    plt.plot(t_sim, X_sim[2, :], 'r--', label='MPC Theta')
    plt.plot(t_sim_poly, X_sim_poly[2, :], 'g:', label='Poly MPC Theta')
    plt.title('Heading (Theta) vs Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Theta [rad]')
    plt.legend()
    plt.grid(True)
    
    # 4. Steering Angle (Phi)
    plt.subplot(2, 4, 4)
    plt.plot(t_ref_cut, X_ref_cut[3, :], 'b-', label='Exact Phi')
    plt.plot(t_sim, X_sim[3, :], 'r--', label='MPC Phi')
    plt.plot(t_sim_poly, X_sim_poly[3, :], 'g:', label='Poly MPC Phi')
    plt.title('Steering Angle (Phi) vs Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Phi [rad]')
    plt.legend()
    plt.grid(True)
    
    # 5. Controls
    plt.subplot(2, 4, 5)
    plt.step(t_ref_cut[:-1], U_ref_cut[0, :], 'b-', label='Exact u1', where='post')
    plt.step(t_sim[:-1], U_sim[0, :], 'c--', label='MPC u1', where='post')
    plt.step(t_ref_cut[:-1], U_ref_cut[1, :], 'g-', label='Exact u2', where='post')
    plt.step(t_sim[:-1], U_sim[1, :], 'y--', label='MPC u2', where='post')
    plt.step(t_sim_poly[:-1], U_sim_poly[0, :], 'g:', label='Poly MPC u1', where='post')
    plt.step(t_sim_poly[:-1], U_sim_poly[1, :], color='lime', linestyle=':', label='Poly MPC u2', where='post')
    plt.title('Controls vs Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Controls')
    plt.legend()
    plt.grid(True)
    
    # Force Calculations for Subplots 6 & 7
    def calc_forces(X, U, t):
        v, phi = X[4, :-1], X[3, :-1]
        u1, u2 = U[0, :], U[1, :]
        FN1 = (m/4) * ((1+c)*(u1*np.sin(phi) + v*u2*np.cos(phi)) + (1/L)*v**2 * np.sin(phi)*np.cos(phi))
        FT1 = m * (u1*np.cos(phi) - v*u2*np.sin(phi) - (1/(4*L))*v**2 * np.sin(phi)**2)
        FN2 = (m/4) * ((1-c)*(u1*np.sin(phi) + v*u2*np.cos(phi)) + (1/L)*v**2 * np.sin(phi)*np.cos(phi))
        FT2 = -m * k_rear * (u1*np.cos(phi) - v*u2*np.sin(phi))
        return FN1, FT1, FN2, FT2
        
    FN1_ref, FT1_ref, FN2_ref, FT2_ref = calc_forces(X_ref_cut, U_ref_cut, t_ref_cut)
    FN1_sim, FT1_sim, FN2_sim, FT2_sim = calc_forces(X_sim, U_sim, t_sim)
    FN1_poly, FT1_poly, FN2_poly, FT2_poly = calc_forces(X_sim_poly, U_sim_poly, t_sim_poly)
    
    # 6. Front Forces
    plt.subplot(2, 4, 6)
    plt.plot(t_ref_cut[:-1], FN1_ref, 'b-', label='Exact $F_{N1}$')
    plt.plot(t_sim[:-1], FN1_sim, 'c--', label='MPC $F_{N1}$')
    plt.plot(t_sim_poly[:-1], FN1_poly, 'g:', label='Poly MPC $F_{N1}$')
    plt.plot(t_ref_cut[:-1], FT1_ref, 'g-', label='Exact $F_{T1}$')
    plt.plot(t_sim[:-1], FT1_sim, 'y--', label='MPC $F_{T1}$')
    plt.plot(t_sim_poly[:-1], FT1_poly, color='lime', linestyle=':', label='Poly MPC $F_{T1}$')
    plt.title('Front Forces vs Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Force [N]')
    plt.legend()
    plt.grid(True)
    
    # 7. Slip Constraints (Total Front Force)
    plt.subplot(2, 4, 7)
    plt.plot(t_ref_cut[:-1], np.sqrt(FN1_ref**2 + FT1_ref**2), 'b-', label='Exact FW Force')
    plt.plot(t_sim[:-1], np.sqrt(FN1_sim**2 + FT1_sim**2), 'r--', label='MPC FW Force')
    plt.plot(t_sim_poly[:-1], np.sqrt(FN1_poly**2 + FT1_poly**2), 'g:', label='Poly MPC FW Force')
    plt.axhline(y=f_max, color='k', linestyle=':', label='Limit ($f_{max}$)')
    plt.title('FW Slip Constraints')
    plt.xlabel('Time [s]')
    plt.ylabel('Total Force [N]')
    plt.legend()
    plt.grid(True)
    
    # 8. State Error
    plt.subplot(2, 4, 8)
    # Interpolate reference states to simulated time grid for accurate error
    
    # Error for Exact MPC
    x_ref_interp_e = np.interp(t_sim, t_ref, X_ref[0, :])
    y_ref_interp_e = np.interp(t_sim, t_ref, X_ref[1, :])
    th_ref_interp_e = np.interp(t_sim, t_ref, X_ref[2, :])
    plt.plot(t_sim, np.abs(x_ref_interp_e - X_sim[0, :]), 'r-', alpha=0.6, label='|X error|')
    plt.plot(t_sim, np.abs(y_ref_interp_e - X_sim[1, :]), 'r--', alpha=0.6, label='|Y error|')
    plt.plot(t_sim, np.abs(th_ref_interp_e - X_sim[2, :]), 'r:', alpha=0.6, label='|Theta error|')

    # Error for Polygonal MPC
    x_ref_interp_p = np.interp(t_sim_poly, t_ref, X_ref[0, :])
    y_ref_interp_p = np.interp(t_sim_poly, t_ref, X_ref[1, :])
    th_ref_interp_p = np.interp(t_sim_poly, t_ref, X_ref[2, :])
    plt.plot(t_sim_poly, np.abs(x_ref_interp_p - X_sim_poly[0, :]), 'g-', label='Poly |X| error')
    plt.plot(t_sim_poly, np.abs(y_ref_interp_p - X_sim_poly[1, :]), 'g--', label='Poly |Y| error')
    plt.plot(t_sim_poly, np.abs(th_ref_interp_p - X_sim_poly[2, :]), 'g:', label='Poly |Theta| error')
    plt.title('Absolute Tracking Error')
    plt.xlabel('Time [s]')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight')
        if not show_plot:
            plt.close() # Close silently so it doesn't block batch runs
    if save_path is None or show_plot:
        plt.show()

def animate_2way_hodograph(t_sim, X_ref, X_sim_poly, U_ref, U_mpc_poly, L=1.0, max_faces=24, filename="compare_2way_hodograph.gif"):
    print(f"Generating 2-Way Hodograph Animation: {filename}...")
    m, g, mu = 20.0, 9.81, 1.0
    rho = L / np.sqrt(3); c = (rho / L)**2
    f_max = mu * 0.5 * m * g
    k_rear = (2 * m_w / m) * ((r/2) / r)**2
    a_max, b_max = 5.0, 3 * np.pi / 2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    u1_grid = np.linspace(-6, 6, 200)
    u2_grid = np.linspace(-6, 6, 200)
    U1, U2 = np.meshgrid(u1_grid, u2_grid)

    # Find the maximum length to ensure we wait for both cars to finish
    N_sim = U_mpc_poly.shape[1]
    
    # Background paths for ax1
    X_com_poly_full = X_sim_poly[0, :N_sim+1]
    Y_com_poly_full = X_sim_poly[1, :N_sim+1]
    X_com_ref_full = X_ref[0, :N_sim+1]
    Y_com_ref_full = X_ref[1, :N_sim+1]

    # Dynamic axis limits to fit all trajectories
    x_min = min(np.min(X_com_poly_full), np.min(X_com_ref_full)) - 1.0
    x_max = max(np.max(X_com_poly_full), np.max(X_com_ref_full)) + 1.0
    y_min = min(np.min(Y_com_poly_full), np.min(Y_com_ref_full)) - 1.0
    y_max = max(np.max(Y_com_poly_full), np.max(Y_com_ref_full)) + 1.0
    
    num_frames = N_sim // 2
    
    def update(frame_idx):
        k = frame_idx * 2  # Multiply by 2 to achieve the frame skipping
        ax1.clear()
        ax2.clear()
        
        # Safely clamp indices so cars "park" if they finish early
        k_poly = min(k, X_sim_poly.shape[1] - 1)
        
        t_val = t_sim[k_poly]
        # We evaluate constraints around the Poly MPC state
        v_val = X_sim_poly[4, k_poly]
        phi_val = X_sim_poly[3, k_poly]
        
        # Clamp index to avoid out-of-bounds errors if arrays have different lengths
        u_ref = U_ref[:, min(k, U_ref.shape[1] - 1)]
        u_poly = U_mpc_poly[:, min(k, U_mpc_poly.shape[1] - 1)]
        
        # --- AX1: Physical Trajectory ---
        ax1.plot(X_com_ref_full, Y_com_ref_full, 'b-', alpha=0.2, label='Ref. Path')
        ax1.plot(X_com_poly_full, Y_com_poly_full, 'g:', alpha=0.3, label='Poly MPC Path')
        
        # Draw car body for the Polygonal MPC state
        x_c, y_c, theta_c = X_sim_poly[0, k_poly], X_sim_poly[1, k_poly], X_sim_poly[2, k_poly]
        x_rear = x_c - 0.5 * L * np.cos(theta_c)
        y_rear = y_c - 0.5 * L * np.sin(theta_c)
        x_front = x_c + 0.5 * L * np.cos(theta_c)
        y_front = y_c + 0.5 * L * np.sin(theta_c)
        
        ax1.plot([x_rear, x_front], [y_rear, y_front], 'k-', linewidth=4, label='Car Body')
        ax1.plot(x_c, y_c, 'ko', markersize=6)
        ax1.set_xlim(x_min, x_max); ax1.set_ylim(y_min, y_max)
        ax1.set_aspect('equal')
        ax1.set_title(f"Vehicle Trajectory (t = {t_val:.2f} s)")
        ax1.grid(True)
        
        # --- AX2: Hodograph ---
        # 1. Exact Nonlinear Contours
        F_N1 = (m/4) * ((1+c)*(U1*np.sin(phi_val) + v_val*U2*np.cos(phi_val)) + (1/L)*v_val**2 * np.sin(phi_val)*np.cos(phi_val))
        F_T1 = m * (U1*np.cos(phi_val) - v_val*U2*np.sin(phi_val) - (1/(4*L))*v_val**2 * np.sin(phi_val)**2)
        F_N2 = (m/4) * ((1-c)*(U1*np.sin(phi_val) + v_val*U2*np.cos(phi_val)) + (1/L)*v_val**2 * np.sin(phi_val)*np.cos(phi_val))
        F_T2 = -m * k_rear * (U1*np.cos(phi_val) - v_val*U2*np.sin(phi_val))
        
        ax2.contour(U1, U2, F_N1**2 + F_T1**2 - f_max**2, levels=[0], colors='blue', linewidths=2, alpha=0.5)
        ax2.contour(U1, U2, F_N2**2 + F_T2**2 - f_max**2, levels=[0], colors='red', linewidths=2, alpha=0.5)
        
        # 2. Bounding Box
        ax2.plot([-a_max, a_max, a_max, -a_max, -a_max], [-b_max, -b_max, b_max, b_max, -b_max], 'k-', linewidth=2)
        
        # 3. The Refined Convex Hull Polygon
        refined_verts = get_refined_polygon_vertices(v_val, phi_val, L, max_faces=max_faces)
        if len(refined_verts) > 0:
            poly_patch = plt.Polygon(refined_verts, closed=True, facecolor='cyan', edgecolor='blue', alpha=0.3, linewidth=2, label='Refined Polygon Admissible Zone')
            ax2.add_patch(poly_patch)
            
        # 4. The Controls
        ax2.plot(u_ref[0], u_ref[1], 'b*', markersize=14, label='Offline Exact')
        ax2.plot(u_poly[0], u_poly[1], 'gd', markersize=10, label='MPC Poly')

        ax2.set_xlim(-6, 6)
        ax2.set_ylim(-6, 6)
        ax2.set_title(f"Hodograph (v = {v_val:.2f} m/s | phi = {phi_val:.2f} rad)")
        ax2.set_xlabel('$u_1$ (Linear Accel) [m/s²]')
        ax2.set_ylabel('$u_2$ (Steering Rate) [rad/s]')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True)

    ani = animation.FuncAnimation(fig, update, frames=num_frames, blit=False)
    ani.save(filename, writer='pillow', fps=15)
    print(f"Animation saved to {filename}")
if __name__ == "__main__":
    num_approx_points = 12  # User specified budget
    L_val = 1.0
    N_total = 500
    #u turn 
    # S_start = [L_val, 0, 0, 0, 10]
    # S_target = [L_val, 3, np.pi, 0, 10] 
    # parallel parking
    S_start = [L_val, 0, 0, 0, 0]
    S_target = [L_val, 1, 0, 0, 0] 
    #intrestind trail 007
    # S_start = [0, 0, 0, 0, 7]
    # S_target = [6.5, -9.0, 3.0, 0, 2]
    # 50% error in poly
    # S_start = [0, 0, 0, 0, 9.4]
    # S_target = [10, 5.36, 0.622,0,2.8]
    
    # 1. Offline Reference
    T_ref, X_ref, U_ref, t_grid, solver_time = solve_time_optimal_problem(S_start, S_target, N=N_total, L=L_val)
    
    # 2. Setup MPC
    print("--- Starting Closed-Loop MPC Simulation ---")
    N_mpc = 15
    dt_ref = T_ref / N_total
    opti_mpc, x_curr_p, x_targ_p, X_mpc, U_mpc = setup_mpc(dt_val=dt_ref, N_mpc=N_mpc, L=L_val)
    F_rk4 = get_rk4_function(L_val)
    
    # --- Extended Simulation ---
    max_sim_steps = N_total + 200 # Allow for extra time to converge
    
    X_sim = np.zeros((5, max_sim_steps + 1))
    X_sim[:, 0] = X_ref[:, 0]
    U_sim = np.zeros((2, max_sim_steps))
    t_sim = np.zeros(max_sim_steps + 1)
    current_state = X_ref[:, 0]
    
    # 3. Execution Loop with Convergence Check
    k = 0
    while k < max_sim_steps:
        # Check for convergence via Finish Line crossing
        dx = current_state[0] - S_target[0]
        dy = current_state[1] - S_target[1]
        theta_t = S_target[2]
        
        cross_proj = dx * np.cos(theta_t) + dy * np.sin(theta_t)
        dist_to_target = np.linalg.norm([dx, dy])
        
        if cross_proj >= 0 and dist_to_target < 2.0 and k > N_total * 0.8:
            print(f"Exact MPC crossed the finish line at step {k} (distance from target center: {dist_to_target:.3f}m)")
            break

        # Determine target state for MPC
        if k + N_mpc <= N_total:
            target_state = X_ref[:, k + N_mpc]
        else:
            target_state = S_target # Use final target when reference runs out
        
        opti_mpc.set_value(x_curr_p, current_state)
        opti_mpc.set_value(x_targ_p, target_state)
        
        # Warm start only when reference is available
        if k + N_mpc <= N_total:
            opti_mpc.set_initial(X_mpc, X_ref[:, k:k+N_mpc+1])
            opti_mpc.set_initial(U_mpc, U_ref[:, k:k+N_mpc])
        
        try:
            sol = opti_mpc.solve()
            u_opt = sol.value(U_mpc[:, 0]) 
        except RuntimeError:
            print(f"MPC Failed at step {k}, using fallback controls.")
            u_opt = U_ref[:, k] if k < U_ref.shape[1] else np.zeros(2)
            
        U_sim[:, k] = u_opt
        t_sim[k+1] = t_sim[k] + dt_ref
        current_state = np.array(F_rk4(current_state, u_opt, dt_ref)).flatten()
        X_sim[:, k+1] = current_state
        
        if k % 50 == 0:
            print(f"Simulated step {k}/{max_sim_steps}")
        
        k += 1
    
    # Trim arrays to the actual number of steps taken
    sim_len_exact = k
    X_sim = X_sim[:, :sim_len_exact+1]
    U_sim = U_sim[:, :sim_len_exact]
    t_sim = t_sim[:sim_len_exact+1]

    # 4. Setup Polygonal MPC
    
    print("--- Starting Closed-Loop MPC Simulation (Polygonal) ---")
    opti_poly, x_curr_p, x_targ_p, A_p, b_p, X_p, U_p = setup_mpc_polygonal(dt_val=dt_ref, N_mpc=N_mpc, L=L_val, max_faces=num_approx_points)
    
    X_sim_poly = np.zeros((5, max_sim_steps + 1))
    X_sim_poly[:, 0] = X_ref[:, 0]
    U_sim_poly = np.zeros((2, max_sim_steps))
    t_sim_poly = np.zeros(max_sim_steps + 1)
    current_state_poly = X_ref[:, 0]
    
    k = 0
    while k < max_sim_steps:
        # Check for convergence via Finish Line crossing
        dx_poly = current_state_poly[0] - S_target[0]
        dy_poly = current_state_poly[1] - S_target[1]
        theta_t = S_target[2]
        
        cross_proj_poly = dx_poly * np.cos(theta_t) + dy_poly * np.sin(theta_t)
        dist_to_target_poly = np.linalg.norm([dx_poly, dy_poly])
        
        if cross_proj_poly >= 0 and dist_to_target_poly < 2.0 and k > N_total * 0.8:
            print(f"Polygonal MPC crossed the finish line at step {k} (distance from target center: {dist_to_target_poly:.3f}m)")
            break

        # Determine target state and exact reference control
        if k + N_mpc <= N_total:
            target_state = X_ref[:, k + N_mpc]
            u_ref_current = U_ref[:, k]
        else:
            target_state = S_target
            u_ref_current = np.array([0.0, 0.0]) # Parked
        
        opti_poly.set_value(x_curr_p, current_state_poly)
        opti_poly.set_value(x_targ_p, target_state)
        
        # Generate constraints, injecting the reference control in case of singularity
        A_mat, b_vec = get_polygonal_constraints(
            current_state_poly[4], 
            current_state_poly[3], 
            L_val, 
            max_faces=num_approx_points, 
            u_ref_k=u_ref_current
        )
        opti_poly.set_value(A_p, A_mat)
        opti_poly.set_value(b_p, b_vec)
        
        # Warm start
        if k + N_mpc <= N_total:
            opti_poly.set_initial(X_p, X_ref[:, k:k+N_mpc+1])
            opti_poly.set_initial(U_p, U_ref[:, k:k+N_mpc])
        
        try:
            sol = opti_poly.solve()
            u_opt = sol.value(U_p[:, 0]) 
        except RuntimeError:
            print(f"Poly MPC Failed at step {k}, using fallback controls.")
            u_opt = U_ref[:, k] if k < U_ref.shape[1] else np.zeros(2)
            
        U_sim_poly[:, k] = u_opt
        t_sim_poly[k+1] = t_sim_poly[k] + dt_ref
        current_state_poly = np.array(F_rk4(current_state_poly, u_opt, dt_ref)).flatten()
        X_sim_poly[:, k+1] = current_state_poly
        
        if k % 50 == 0:
            print(f"Simulated Poly step {k}/{max_sim_steps}")
        
        k += 1

    # Trim arrays
    sim_len_poly = k
    X_sim_poly = X_sim_poly[:, :sim_len_poly+1]
    U_sim_poly = U_sim_poly[:, :sim_len_poly]
    t_sim_poly = t_sim_poly[:sim_len_poly+1]

    # 5. Plot Detailed Comparison
    plot_mpc_vs_exact(
        t_grid, X_ref, U_ref, t_sim, X_sim, U_sim, t_sim_poly, X_sim_poly, U_sim_poly, 
        L=L_val, 
        save_path=f"mpc_vs_exact_8_subplots_N{num_approx_points}.png", 
        show_plot=True
    )

    # 6. Generate Final 2-Way Animation
    # Pass the full arrays, let the animation function handle the max length and clamping
    animate_2way_hodograph(
        t_sim_poly, # Pass Poly time array as the master clock
        X_ref, 
        X_sim_poly, 
        U_ref, 
        U_sim_poly, 
        L=L_val, 
        max_faces=num_approx_points,
        filename=f"compare_2way_hodograph_N{num_approx_points}.gif"
    )