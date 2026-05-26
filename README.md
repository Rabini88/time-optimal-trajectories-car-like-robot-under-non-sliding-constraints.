# Time-Optimal Trajectory Planning for a Car-Like Mobile Robot

Code accompanying the paper: *[Paper title to be added]*.

## Overview

This repository implements time-optimal trajectory planning for a car-like mobile robot subject to non-sliding (tire friction) constraints. The work addresses a fundamental challenge in autonomous vehicle motion planning: the admissible control set at each instant — defined by the nonlinear friction ellipse constraints on both front and rear wheels — has a complex, state-dependent shape that is expensive to enforce inside a real-time controller.

The key contribution is an **analytical polygonal approximation** of this admissible control set. At each state (velocity `v`, steering angle `φ`), the exact boundary of the friction-constrained hodograph (the set of feasible accelerations in control space) is computed analytically and inscribed by a convex polygon with a user-specified number of faces. This polygon is then used as a linear constraint inside a Model Predictive Controller (MPC), replacing the nonlinear friction circles with a set of linear inequalities that can be solved efficiently by standard QP solvers.

### System Model

The car is modeled as a rigid body with:
- **State**: `[x, y, θ, φ, v]` — position of the center of mass, heading angle, steering angle, and longitudinal speed
- **Controls**: `[u₁, u₂]` — tangential acceleration and steering rate
- **Dynamics**: kinematic bicycle model tracking the center of mass (Eq. 2.29 of the paper)
- **Constraints**: non-sliding conditions on both front and rear wheels, expressed as elliptic friction bounds in the force domain, which map to a non-convex feasible region in control space `(u₁, u₂)`.

**Physical parameters** (fixed throughout):
- Vehicle mass: `m = 20 kg`, wheelbase: `L = 1 m`
- Friction coefficient: `μ = 1.0`
- Wheel mass/radius: `m_w = 1 kg`, `r = 0.1 m`

### Pipeline

1. **Offline exact solver** (`car_robot_time_optimal_casadi_and_MPC.py`): solves the time-optimal control problem using IPOPT via CasADi, with the full nonlinear friction constraints. Produces a reference trajectory `(X_ref, U_ref, T_ref)`.
2. **Polygon construction** (`get_refined_polygon_vertices`): for each state along the reference, analytically computes the convex hull of the admissible control set and refines it to a target face budget using adaptive edge subdivision.
3. **Offline poly solver** (`offline_poly_analysis.py`): re-solves the time-optimal problem using the polygon approximation in place of the nonlinear constraints, and computes the geometric time deficit relative to the exact solution.
4. **MPC tracking** (`car_robot_time_optimal_casadi_and_MPC.py`): a receding-horizon MPC uses the pre-computed polygon constraints (updated at each step from the current state) to track the offline reference trajectory in closed loop.
5. **Batch evaluation** (`batch_experiments.py`): runs the pipeline over many random scenarios to characterize the statistical distribution of the time deficit and spatial tracking error.

### Pseudo-code: $\mathcal{FW}$ Approximation Algorithm

**Input:** Current speed $\nu$ and steering angle $\phi$, physical parameters $m,L,f_{max}$, control limits $a_{max},b_{max}$, number of approximation vertices $N$.

*/* Identify candidate boundary points */*

1. Let $P_{\text{candidate}} = \emptyset$

*/* Compute boundary intersections */*

2. Solve for $u_2$ roots on $\mathcal{FW}$ boundaries at $u_1 = \pm a_{max}$
3. Add all real root pairs $(u_1, u_2)$ from these limits to $P_{\text{candidate}}$

*/* Compute extrema and ellipse intersections */*

4. Set discriminant $\Delta = 0$ for $F_{N1}^2+F_{T1}^2 = f_{max}^2$, add ellipse extrema to $P_{\text{candidate}}$
5. Add control $(u_1,u_2) = (\pm a_{max},\pm b_{max})$ to $P_{\text{candidate}}$

*/* Filter for admissibility */*

6. Let $P_{\text{valid}}=\emptyset$
7. **For** $p \in P_{\text{candidate}}$:
    * Compute exact forces $F_{N1},F_{T1}$ using $\nu,\phi$ and $(u_1,u_2)$
    * **If** $F_{N1}^2+F_{T1}^2 \leq f_{max}^2$:
        * $P_{\text{valid}} \leftarrow p$

*/* Construct base convex hull */*

8. Compute base polygon $V_{base}$ using convex hull of $P_{\text{valid}}$
9. Set $N_{add} = N-|V_{base}|$
10. Set $V_{\text{refine}} = V_{base}$

*/* Adaptive edge refinement */*

11. **While** $N_{add}>0$:
    * Compute distances between adjacent vertices of $V_{base}$
    * Find the longest distance $E = (p_1,p_2)$ of $V_{base}$
    * Compute centroid $c$ of $V_{base}$
    * Compute edge intermediate point $m = p_1 + \frac{1}{2}(p_2-p_1)$
    * Define normalized ray direction from $c$ through $m$
    * Find intersection point $p_{\text{new}}$ with $\mathcal{FW}$ ellipse
    * $V_{\text{refine}}\leftarrow p_{\text{new}}$
    * $N_{add} = N-|V_{\text{refine}}|$

12. Compute final convex hull $V_{\text{refine}}$

**Output:** Linear constraint matrix $A$ and vector $b$ s.t. $Au\leq b\subseteq \mathcal{FW}$.

---

## Repository Structure

```
.
├── car_robot_time_optimal_casadi.py         # Kinematic model + time-optimal solver (baseline, no MPC)
├── car_robot_time_optimal_casadi_and_MPC.py # Main module: full solver, polygon construction, MPC
├── dynamic_car_time_optimal_casadi.py       # Dynamic model with explicit tire force computation
├── offline_poly_analysis.py                 # Offline polygon-constrained trajectory solver + comparison plots
├── batch_experiments.py                     # Random-scenario batch runner; saves CSV + per-trial plots
├── plot_batch_results.py                    # Interactive: loads batch CSVs, plots time-error histograms
├── plot_poly_data.py                        # Loads saved .npz trajectory data, produces publication figures
│
├── results/
│   ├── exact_solutions/                     # Example exact trajectories for 3 scenarios (GIFs, PNGs)
│   ├── parallel_park/                       # Parallel parking scenario animations
│   ├── polygon_approximation/               # Polygon approximation results across N values
│   ├── u_turn/                              # U-turn scenario results (μ = 1)
│   ├── batch/                               # Batch experiment output (CSV + summary plots)
│   └── figures/                             # All other generated figures and animations
│
├── data/                                    # Saved trajectory data (.npz) — gitignored
├── references/                              # Reference PDFs — gitignored
├── requirements.txt
└── README.md
```

---

## Dependencies

Python 3.x with:

| Package | Version | Purpose |
|---|---|---|
| [casadi](https://web.casadi.org/) | 3.7.2 | Nonlinear optimization (IPOPT interface) |
| numpy | 2.3.5 | Numerical arrays |
| scipy | 1.16.3 | Convex hull, root finding |
| matplotlib | 3.10.8 | Plotting and animation |
| pillow | 12.0.0 | GIF export for animations |
| pandas | 3.0.0 | Batch result CSV loading |
| seaborn | 0.13.2 | Statistical histograms |

Install:
```bash
pip install -r requirements.txt
```

---

## Usage

### Solve a single time-optimal trajectory and run MPC
```bash
python car_robot_time_optimal_casadi_and_MPC.py
```
Runs the parallel parking scenario by default. Edit `S_start` / `S_target` at the bottom of the file to change the scenario.

### Run offline polygon approximation analysis
```bash
python offline_poly_analysis.py
```
Solves the exact reference and the polygon-constrained problem side by side, prints the time deficit, and saves a comparison plot and animation to `results/figures/`.

### Run batch statistical experiments
```bash
python batch_experiments.py
```
Runs 1000 random scenarios by default (edit `num_trials` / `num_approx_points` at the bottom). Results are saved to a timestamped subdirectory under `results/batch/`.

### Plot results from a saved batch run
```bash
python plot_batch_results.py
```
Interactive: lists available batch runs in `results/batch/`, prompts for selection, and saves a time-error histogram PDF to `results/figures/`.

### Plot detailed trajectory figures from saved data
```bash
python plot_poly_data.py data/offline_poly_data_N8.npz
```
Loads a saved `.npz` trajectory file and produces individual publication-quality PDF figures (spatial trajectory, velocity, forces, slip constraints, tracking error) in `results/figures/`.
