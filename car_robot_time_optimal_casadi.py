import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

def simple_car_robot_dynamics(S, u,L = 1):
    """
    Calculates the state derivative dS/dt.
    state: [x, y, theta, phi, v]
    u: [u1, u2] (along-track accel, steer rate)
    """
    #uses CasADi vars
    x = S[0]
    y = S[1]
    theta = S[2]
    phi = S[3]
    v = S[4]

    u1 = u[0]
    u2 = u[1]

    dx = v * ca.cos(phi) * ca.cos(theta)
    dy = v * ca.cos(phi) * ca.sin(theta)
    dtheta = (v * ca.sin(phi)) / (2 * L)
    dphi = u2
    dv = u1

    return ca.vertcat(dx, dy, dtheta, dphi, dv)

def solve_time_optimal_problem(S_start,S_target,N = 500,phi_max = np.pi/2,a_max = 5,b_max = 3*np.pi/2, free_final_x=False, free_final_y=False):
    T_guess = 10.0
    opti = ca.Opti()
    T = opti.variable()
    X = opti.variable(5,N+1)
    U = opti.variable(2,N)

    opti.minimize(T)
    opti.subject_to(T >= 0.1)

    #constraints
    opti.subject_to(X[:,0] == S_start)
    if not free_final_x:
        opti.subject_to(X[0,N] == S_target[0])
    if not free_final_y:
        opti.subject_to(X[1,N] == S_target[1])
    opti.subject_to(X[2:,N] == S_target[2:])

    #dynamics constraints
    dt = T/N
    for k in range(N):
        k1 = simple_car_robot_dynamics(X[:,k],U[:,k])
        k2 = simple_car_robot_dynamics(X[:,k]+dt*k1/2,U[:,k])
        k3 = simple_car_robot_dynamics(X[:,k]+dt*k2/2,U[:,k])
        k4 = simple_car_robot_dynamics(X[:,k]+dt*k3,U[:,k])
        x_next = X[:,k] + dt*(k1+2*k2+2*k3+k4)/6
        opti.subject_to(X[:,k+1] == x_next)

        #control limit
        opti.subject_to(opti.bounded(-a_max,U[0,k],a_max))
        opti.subject_to(opti.bounded(-b_max,U[1,k],b_max))
        #steer limit
        opti.subject_to(opti.bounded(-phi_max,X[3,k],phi_max))
        #velocity limit
        opti.subject_to(opti.bounded(-20.0, X[4,k], 20.0))

    opti.set_initial(T, 3.0)
    for i in range(5):
        opti.set_initial(X[i, :], np.linspace(S_start[i], S_target[i], N+1))

    # --- INJECT WARM START INITIAL GUESS ---
    opti.set_initial(T, T_guess)
    
    # Create a topologically correct guess for parallel parking
    mid = N // 2
    
    # X bulges forward, then returns to 0
    X_guess = np.concatenate([np.linspace(0, 1.5, mid), np.linspace(1.5, 0, N+1-mid)])
    
    # Y transitions smoothly from 0 to 1
    Y_guess = np.linspace(S_start[1], S_target[1], N+1)
    
    # Theta swings out (e.g., 45 degrees) and comes back
    theta_guess = np.concatenate([np.linspace(0, np.pi/4, mid), np.linspace(np.pi/4, 0, N+1-mid)])
    
    # Velocity is positive (forward) then negative (reverse)
    v_guess = np.concatenate([np.linspace(3, 0, mid), np.linspace(0, -3, N+1-mid)])
    
    opti.set_initial(X[0, :], X_guess)
    opti.set_initial(X[1, :], Y_guess)
    opti.set_initial(X[2, :], theta_guess)
    opti.set_initial(X[3, :], np.zeros(N+1)) # phi can safely start at 0
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


def plot_solution(t_grid, X_opt, U_opt, T_opt):
    plt.figure(figsize=(14, 8))
    plt.suptitle(f"Optimal Trajectory, T = {T_opt:.2f} s")

    # 1. Trajectory
    plt.subplot(2, 3, 1)
    plt.plot(X_opt[0, :], X_opt[1, :], label='Trajectory')
    plt.plot(X_opt[0, 0], X_opt[1, 0], 'go', label='Start')
    plt.plot(X_opt[0, -1], X_opt[1, -1], 'rx', label='Target')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('Trajectory (Y vs X)')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)

    # 2. Velocity
    plt.subplot(2, 3, 2)
    plt.plot(t_grid, X_opt[4, :])
    plt.xlabel('Time [s]')
    plt.ylabel('v [m/s]')
    plt.title('Velocity vs Time')
    plt.grid(True)

    # 3. Heading (Theta)
    plt.subplot(2, 3, 3)
    plt.plot(t_grid, X_opt[2, :])
    plt.xlabel('Time [s]')
    plt.ylabel('Theta [rad]')
    plt.title('Heading (Theta) vs Time')
    plt.grid(True)

    # 4. Steering Angle (Phi)
    plt.subplot(2, 3, 4)
    plt.plot(t_grid, X_opt[3, :])
    plt.xlabel('Time [s]')
    plt.ylabel('Phi [rad]')
    plt.title('Steering Angle (Phi) vs Time')
    plt.grid(True)

    # 5. Controls
    plt.subplot(2, 3, 5)
    plt.step(t_grid[:-1], U_opt[0, :], label='u1 (accel)', where='post')
    plt.step(t_grid[:-1], U_opt[1, :], label='u2 (steer rate)', where='post')
    plt.xlabel('Time [s]')
    plt.ylabel('Controls')
    plt.title('Controls vs Time')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('plot_solution.png')
    plt.show()

if __name__ == "__main__":
        # Example 
    isFree_x = False
    S_start = [1, 0, 0, 0, 0] # x y theta phi v
    S_target = [1, 0.75, 0, 0, 0]
    T_opt, X_opt, U_opt, t_grid, solver_time = solve_time_optimal_problem(S_start, S_target, free_final_x=isFree_x)
    plot_solution(t_grid, X_opt, U_opt, T_opt)
