import os
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
import csv
from matplotlib.ticker import PercentFormatter

from car_robot_time_optimal_casadi_and_MPC import solve_time_optimal_problem
from offline_poly_analysis import (
    solve_offline_poly_problem, 
    plot_offline_comparison
)

def generate_random_scenario():
    """Generates random starting and target vehicle states."""
    S_start = [0.0, 0.0, 0.0, 0.0, np.random.uniform(0, 10)]
    
    S_target = [
        np.random.uniform(10, 10),
        np.random.uniform(-10, 10),
        np.random.uniform(-np.pi, np.pi),
        0.0,
        np.random.uniform(0, 10)
    ]
    return S_start, S_target

def run_single_scenario(S_start, S_target, L_val=1.0, trial_idx=0, run_dir=".", num_approx_points=12):
    """Runs an exact and poly offline optimization for a single start-target coordinate pair."""
    N_total = 200 # Fixed horizon steps for the batch reference trajectory
    
    # 1. Exact Offline Trajectory
    try:
        T_ref, X_ref, U_ref, t_grid, solver_time = solve_time_optimal_problem(
            S_start, S_target, N=N_total, L=L_val
        )
    except RuntimeError:
        # Solver completely failed
        return None

    if T_ref is None:
        return None

    # 2. Polygonal Offline Trajectory
    try:
        T_poly, X_poly, U_poly, t_grid_poly = solve_offline_poly_problem(
            S_start, S_target, N_total, L_val, X_ref, T_ref, U_ref, num_approx_points
        )
    except Exception as e:
        print(f"Offline Poly Solver failed: {e}")
        return None
        
    # Compute Maximum Spatial Error using interpolation onto the poly time grid
    x_ref_interp = np.interp(t_grid_poly, t_grid, X_ref[0, :])
    y_ref_interp = np.interp(t_grid_poly, t_grid, X_ref[1, :])
    
    err_x = x_ref_interp - X_poly[0, :]
    err_y = y_ref_interp - X_poly[1, :]
    spatial_errors = np.sqrt(err_x**2 + err_y**2)
    max_poly_err = np.max(spatial_errors)

    # Save the individual trial plot
    plot_filename = os.path.join(run_dir, f"trial_{trial_idx:03d}.png")
    plot_offline_comparison(
        t_grid, X_ref, U_ref,
        t_grid_poly, X_poly, U_poly,
        L=L_val, save_path=plot_filename, show_plot=False
    )

    return {
        'exact_time': T_ref,
        'poly_time': T_poly,
        'poly_err': max_poly_err,
        'S_start': S_start,
        'S_target': S_target,
        'final_poly_state': X_poly[:, -1]
    }

def run_batch_experiments(num_trials=100, num_approx_points=12):
    results = []
    
    # Create timestamped directory
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join("results", "batch", f"batch_results_{num_trials}runs_{num_approx_points}pts_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"Starting Batch Experiments for {num_trials} trials. Saving plots to ./{run_dir}/")
    
    for i in range(num_trials):
        S_start, S_target = generate_random_scenario()
        res = run_single_scenario(S_start, S_target, trial_idx=i, run_dir=run_dir, num_approx_points=num_approx_points)
        if res is not None:
            results.append(res)
            print(f"Trial {i+1}/{num_trials} successful.")
        else:
            print(f"Trial {i+1}/{num_trials} failed (Offline Solver Error). Skipping.")
            
    if not results:
        print("No successful trials. Exiting.")
        return
        
    # Save trial data to CSV
    csv_filename = os.path.join(run_dir, "trial_data.csv")
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'start_x', 'start_y', 'start_theta', 'start_phi', 'start_v', 
            'target_x', 'target_y', 'target_theta', 'target_phi', 'target_v', 
            'T_exact', 'T_poly', 'max_poly_error', 
            'final_poly_x', 'final_poly_y', 'final_poly_theta', 'final_poly_phi', 'final_poly_v'
        ])
        for r in results:
            row = list(r['S_start']) + list(r['S_target']) + [
                r['exact_time'], r['poly_time'], r['poly_err']
            ] + list(r['final_poly_state'])
            writer.writerow(row)
            
    # Create Side-By-Side Subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Subplot 1: Tracking Error Histogram (Poly MPC vs Exact Offline)
    poly_errs = [r['poly_err'] for r in results]
    
    weights1 = np.ones(len(poly_errs)) / len(poly_errs)
    ax1.hist(poly_errs, weights=weights1, alpha=0.7, color='green', label='Poly Offline Error', bins=10)
    ax1.yaxis.set_major_formatter(PercentFormatter(1))
    
    mean_poly, std_poly = np.mean(poly_errs), np.std(poly_errs)
    
    ax1.axvline(mean_poly, color='darkgreen', linestyle='-', linewidth=2, label=f'Mean: {mean_poly:.2f}m')
    ax1.axvline(mean_poly + std_poly, color='darkgreen', linestyle='--', linewidth=2, label=f'+1 SD: {mean_poly+std_poly:.2f}m')
    
    ax1.set_title("Polygonal Offline Maximum Spatial Tracking Error")
    ax1.set_xlabel("Tracking Error from Exact Reference [m]")
    ax1.set_ylabel("Percentage of Scenarios")
    ax1.legend()
    ax1.grid(True, alpha=0.7)
    
    # Subplot 2: Relative Time Error Histogram
    t_exacts = np.array([r['exact_time'] for r in results])
    t_polys = np.array([r['poly_time'] for r in results])
    
    # Calculate absolute relative error in completion time
    time_rel_errors = np.abs(t_exacts - t_polys) / t_exacts
    
    weights2 = np.ones(len(time_rel_errors)) / len(time_rel_errors)
    ax2.hist(time_rel_errors, weights=weights2, alpha=0.7, color='purple', label='Relative Time Error', bins=10)
    ax2.yaxis.set_major_formatter(PercentFormatter(1))
    
    mean_time_err = np.mean(time_rel_errors)
    std_time_err = np.std(time_rel_errors)
    
    # Format the labels as percentages
    ax2.axvline(mean_time_err, color='indigo', linestyle='-', linewidth=2, label=f'Mean: {mean_time_err:.2%}')
    ax2.axvline(mean_time_err + std_time_err, color='indigo', linestyle='--', linewidth=2, label=f'+1 SD: {mean_time_err+std_time_err:.2%}')
    
    ax2.set_title("Relative Error in Completion Time (Offline Poly)")
    ax2.set_xlabel("Absolute Relative Error |$T_{exact}$ - $T_{poly}$| / $T_{exact}$")
    ax2.set_ylabel("Percentage of Scenarios")
    
    # Convert x-axis ticks to percentages for readability
    vals = ax2.get_xticks()
    ax2.set_xticks(vals)
    ax2.set_xticklabels(['{:,.1%}'.format(x) for x in vals])
    
    ax2.legend()
    ax2.grid(True, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'summary_plots.png'))
    plt.show()

if __name__ == '__main__':
    run_batch_experiments(num_trials=1000, num_approx_points=8)

    # run_batch_experiments(num_trials=200, num_approx_points=4)