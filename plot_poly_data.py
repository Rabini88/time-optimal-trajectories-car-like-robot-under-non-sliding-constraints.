import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

# --- Physics constants (must match the solvers) ---
L    = 1.0
m, g, mu = 20.0, 9.81, 1.0
rho  = L / np.sqrt(3)
c    = (rho / L)**2
m_w, r = 1.0, 0.1
k_rear = (2 * m_w / m) * ((r / 2) / r)**2
f_max  = mu * 0.5 * m * g

def calc_forces(X, U):
    v, phi = X[4, :-1], X[3, :-1]
    u1, u2 = U[0, :], U[1, :]
    FN1 = (m/4) * ((1+c)*(u1*np.sin(phi) + v*u2*np.cos(phi)) + (1/L)*v**2 * np.sin(phi)*np.cos(phi))
    FT1 =  m    * (u1*np.cos(phi) - v*u2*np.sin(phi) - (1/(4*L))*v**2 * np.sin(phi)**2)
    FN2 = (m/4) * ((1-c)*(u1*np.sin(phi) + v*u2*np.cos(phi)) + (1/L)*v**2 * np.sin(phi)*np.cos(phi))
    FT2 = -m * k_rear * (u1*np.cos(phi) - v*u2*np.sin(phi))
    return FN1, FT1, FN2, FT2


def load_data(path):
    d = np.load(path)
    return (float(d["T_ref"]), d["t_ref"], d["X_ref"], d["U_ref"],
            float(d["T_poly"]), d["t_poly"], d["X_poly"], d["U_poly"])


def pick_file():
    files = sorted(glob.glob(os.path.join("data", "offline_poly_data_N*.npz")))
    if not files:
        raise FileNotFoundError("No offline_poly_data_N*.npz files found.")
    if len(files) == 1:
        print(f"Loading {files[0]}")
        return files[0]
    print("Available data files:")
    for i, f in enumerate(files):
        print(f"  [{i}] {f}")
    idx = int(input("Select file index: ").strip())
    return files[idx]


def save_fig(fig, name, prefix):
    path = f"{prefix}_{name}.pdf"
    fig.savefig(path, format="pdf", bbox_inches="tight")
    print(f"Saved {path}")


def flip_exact(X_ref, U_ref):
    """Reflect the exact trajectory about y=0.5 to get the other symmetric branch."""
    X_flip = X_ref.copy()
    U_flip = U_ref.copy()
    X_flip[1, :] = 1.0 - X_ref[1, :]  # reflect y about y=0.5
    X_flip[2, :] = -X_ref[2, :]        # negate theta
    X_flip[3, :] = -X_ref[3, :]        # negate phi
    U_flip[1, :] = -U_ref[1, :]        # negate steering rate u2
    return X_flip, U_flip


def plot_all(data_path,
             flip_exact_to_match_poly=False,
             plot_spatial=True,
             plot_velocity=False,
             plot_theta=False,
             plot_phi=False,
             plot_controls=False,
             plot_front_forces=False,
             plot_slip=False,
             plot_tracking_error=False,
             plot_rear_axle=False):
    T_ref, t_ref, X_ref, U_ref, T_poly, t_poly, X_poly, U_poly = load_data(data_path)
    os.makedirs(os.path.join("results", "figures"), exist_ok=True)
    prefix = os.path.join("results", "figures", os.path.basename(data_path).replace(".npz", ""))

    if flip_exact_to_match_poly:
        mid = X_ref.shape[1] // 2
        if np.sign(X_ref[2, mid]) != np.sign(X_poly[2, mid]):
            print("Detected different symmetric branches — flipping exact solution.")
            X_ref, U_ref = flip_exact(X_ref, U_ref)
        else:
            print("Both solutions on the same branch — no flip applied.")

    FN1_r, FT1_r, FN2_r, FT2_r = calc_forces(X_ref,  U_ref)
    FN1_p, FT1_p, FN2_p, FT2_p = calc_forces(X_poly, U_poly)

    # Interpolate exact states onto poly time grid for error computation
    x_interp  = np.interp(t_poly, t_ref, X_ref[0, :])
    y_interp  = np.interp(t_poly, t_ref, X_ref[1, :])
    th_interp = np.interp(t_poly, t_ref, X_ref[2, :])

    blue, green = "#1f77b4", "#2ca02c"
    lw = 2

    if plot_spatial:
        x_rear_r  = X_ref[0, :]  - L * np.cos(X_ref[2, :])
        y_rear_r  = X_ref[1, :]  - L * np.sin(X_ref[2, :])
        x_rear_p  = X_poly[0, :] - L * np.cos(X_poly[2, :])
        y_rear_p  = X_poly[1, :] - L * np.sin(X_poly[2, :])

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(X_ref[0, :],  X_ref[1, :],  color=blue,  linewidth=lw,   label=f"Exact CoM ($T$={T_ref:.3f} s)")
        ax.plot(X_poly[0, :], X_poly[1, :], color=green, linewidth=lw,   linestyle="--", label=f"Poly CoM ($T$={T_poly:.3f} s)")
        ax.plot(x_rear_r,     y_rear_r,     color=blue,  linewidth=lw-1, linestyle="-",  alpha=0.4, label="Exact Rear Axle")
        ax.plot(x_rear_p,     y_rear_p,     color=green, linewidth=lw-1, linestyle="--", alpha=0.4, label="Poly Rear Axle")
        ax.plot(X_ref[0, 0],  X_ref[1, 0],  "go", markersize=8, label="Start")
        ax.plot(X_ref[0, -1], X_ref[1, -1], "rx", markersize=10, markeredgewidth=2, label="Target")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_title("Spatial Trajectory")
        ax.set_aspect("equal")
        ax.legend()
        plt.tight_layout()
        save_fig(fig, "1_spatial", prefix)

    if plot_velocity:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(t_ref,  X_ref[4, :],  color=blue,  linewidth=lw, label="Exact")
        ax.plot(t_poly, X_poly[4, :], color=green, linewidth=lw, linestyle="--", label="Poly")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("v [m/s]")
        ax.set_title("Velocity")
        ax.legend()
        plt.tight_layout()
        save_fig(fig, "2_velocity", prefix)

    if plot_theta:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(t_ref,  X_ref[2, :],  color=blue,  linewidth=lw, label="Exact")
        ax.plot(t_poly, X_poly[2, :], color=green, linewidth=lw, linestyle="--", label="Poly")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("$\\theta$ [rad]")
        ax.set_title("Heading Angle")
        ax.legend()
        plt.tight_layout()
        save_fig(fig, "3_theta", prefix)

    if plot_phi:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(t_ref,  X_ref[3, :],  color=blue,  linewidth=lw, label="Exact")
        ax.plot(t_poly, X_poly[3, :], color=green, linewidth=lw, linestyle="--", label="Poly")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("$\\phi$ [rad]")
        ax.set_title("Steering Angle")
        ax.legend()
        plt.tight_layout()
        save_fig(fig, "4_phi", prefix)

    if plot_controls:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.step(t_ref[:-1],  U_ref[0, :],  color=blue,      linewidth=lw, label="Exact $u_1$",  where="post")
        ax.step(t_poly[:-1], U_poly[0, :], color=green,     linewidth=lw, label="Poly $u_1$",   where="post", linestyle="--")
        ax.step(t_ref[:-1],  U_ref[1, :],  color="#d62728", linewidth=lw, label="Exact $u_2$",  where="post")
        ax.step(t_poly[:-1], U_poly[1, :], color="#ff7f0e", linewidth=lw, label="Poly $u_2$",   where="post", linestyle="--")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Control input")
        ax.set_title("Controls")
        ax.legend()
        plt.tight_layout()
        save_fig(fig, "5_controls", prefix)

    if plot_front_forces:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(t_ref[:-1],  FN1_r, color=blue,  linewidth=lw, label="Exact $F_{N1}$")
        ax.plot(t_poly[:-1], FN1_p, color=blue,  linewidth=lw, linestyle="--", label="Poly $F_{N1}$",  alpha=0.7)
        ax.plot(t_ref[:-1],  FT1_r, color=green, linewidth=lw, label="Exact $F_{T1}$")
        ax.plot(t_poly[:-1], FT1_p, color=green, linewidth=lw, linestyle="--", label="Poly $F_{T1}$",  alpha=0.7)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Force [N]")
        ax.set_title("Front Axle Forces")
        ax.legend()
        plt.tight_layout()
        save_fig(fig, "6_front_forces", prefix)

    if plot_slip:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(t_ref[:-1],  np.sqrt(FN1_r**2 + FT1_r**2), color=blue,  linewidth=lw, label="Exact")
        ax.plot(t_poly[:-1], np.sqrt(FN1_p**2 + FT1_p**2), color=green, linewidth=lw, linestyle="--", label="Poly")
        ax.axhline(f_max, color="black", linestyle=":", linewidth=lw, label=f"$f_{{max}}$ = {f_max:.1f} N")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("$\\|F_{W1}\\|$ [N]")
        ax.set_title("Front Wheel Slip Constraint")
        ax.legend()
        plt.tight_layout()
        save_fig(fig, "7_slip", prefix)

    if plot_tracking_error:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(t_poly, np.abs(x_interp  - X_poly[0, :]), color="#d62728", linewidth=lw, label="|X error|")
        ax.plot(t_poly, np.abs(y_interp  - X_poly[1, :]), color=green,     linewidth=lw, linestyle="--", label="|Y error|")
        ax.plot(t_poly, np.abs(th_interp - X_poly[2, :]), color=blue,      linewidth=lw, linestyle=":",  label="|$\\theta$ error|")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Absolute error")
        ax.set_title("Tracking Error (Poly vs Exact)")
        ax.legend()
        plt.tight_layout()
        save_fig(fig, "8_tracking_error", prefix)

    if plot_rear_axle:
        x_rear_r = X_ref[0, :]  - L * np.cos(X_ref[2, :])
        y_rear_r = X_ref[1, :]  - L * np.sin(X_ref[2, :])
        x_rear_p = X_poly[0, :] - L * np.cos(X_poly[2, :])
        y_rear_p = X_poly[1, :] - L * np.sin(X_poly[2, :])
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(x_rear_r, y_rear_r, color=blue,  linewidth=lw, label=f"exact ($T$={T_ref:.3f} sec)")
        ax.plot(x_rear_p, y_rear_p, color=green, linewidth=lw, linestyle="--", label=f"approximation ($T$={T_poly:.3f} sec)")
        ax.plot(x_rear_r[-1], y_rear_r[-1], "go", markersize=8)
        ax.plot(x_rear_r[0],  y_rear_r[0],  "rx", markersize=10, markeredgewidth=2)
        ax.text(x_rear_r[-1], y_rear_r[-1] - 0.05, "start",  ha="center", va="top", fontsize=13)
        ax.text(x_rear_r[0],  y_rear_r[0]  - 0.05, "target", ha="center", va="top", fontsize=13)
        arrow_len = 0.15
        for idx, color in [(-1, "g"), (0, "r")]:
            ax.annotate("",
                xy=(x_rear_r[idx] + arrow_len * np.cos(X_ref[2, idx]),
                    y_rear_r[idx] + arrow_len * np.sin(X_ref[2, idx])),
                xytext=(x_rear_r[idx], y_rear_r[idx]),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5))
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        # ax.set_title("Rear Axle Midpoint Trajectory")
        ax.set_aspect("equal")
        ax.legend()
        plt.tight_layout()
        save_fig(fig, "9_rear_axle", prefix)

    plt.show()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else pick_file()
    flip_to_match_poly = True
    plot_all(path, flip_exact_to_match_poly=flip_to_match_poly, plot_rear_axle=True)
