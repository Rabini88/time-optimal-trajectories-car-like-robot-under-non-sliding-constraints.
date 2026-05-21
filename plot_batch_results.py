import os
import re
import glob
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def discover_result_dirs():
    return sorted(glob.glob(os.path.join("results", "batch", "batch_results_*")))


def load_results(result_dirs):
    all_dfs = []
    for d in result_dirs:
        csv_path = os.path.join(d, "trial_data.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: no trial_data.csv found in {d}, skipping.")
            continue
        df = pd.read_csv(csv_path)
        df["source"] = d
        all_dfs.append(df)
        print(f"Loaded {len(df)} trials from {d}")
    if not all_dfs:
        raise FileNotFoundError("No CSV files found in the specified directories.")
    return pd.concat(all_dfs, ignore_index=True)


def plot_histograms(df):
    t_exacts = df["T_exact"].values
    t_polys = df["T_poly"].values
    time_rel_errors = np.abs(t_exacts - t_polys) / t_exacts

    mean_time_err = np.mean(time_rel_errors)
    std_time_err = np.std(time_rel_errors)

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.histplot(
        time_rel_errors * 100,
        binwidth=1,
        stat="percent",
        color="purple",
        alpha=1.0,
        ax=ax,
    )

    ax.axvline(mean_time_err * 100, color='indigo', linestyle='-', linewidth=2,
               label=f'Mean: {mean_time_err:.2%}')
    ax.axvline((mean_time_err + std_time_err) * 100, color='indigo', linestyle='--', linewidth=2,
               label=f'+1 SD: {mean_time_err + std_time_err:.2%}')

    ax.set_xlim([0, 20])
    # ax.set_title("Relative Error in Completion Time (Offline Poly)")
    ax.set_xlabel("Absolute Relative Error |$T_{exact}$ - $T_{poly}$| / $T_{exact}$ [%]")
    ax.set_ylabel("Percentage of Scenarios [%]")
    ax.legend()
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.7)

    plt.tight_layout()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.join("results", "figures"), exist_ok=True)
    pdf_path = os.path.join("results", "figures", f"time_error_histogram_{timestamp}.pdf")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    print(f"Saved figure to {pdf_path}")

    plt.show()


def parse_dir_name(d):
    m = re.match(r"batch_results_(\d+)runs_(\d+)pts_(\d{8})_(\d{6})", d)
    if not m:
        return d
    runs, pts, date_str, time_str = m.groups()
    dt = datetime.datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
    return f"{dt.day}.{dt.month}.{dt.year}  {dt.hour:02d}:{dt.minute:02d}  |  {runs} runs  |  {pts} points"


def select_dirs_interactively():
    available = discover_result_dirs()
    if not available:
        raise FileNotFoundError("No batch_results_* directories found.")

    print("\nAvailable result directories:")
    for i, d in enumerate(available):
        print(f"  [{i}] {parse_dir_name(d)}")

    print("\nEnter the numbers of the directories to load (e.g. 0 2 3), or press Enter to load all:")
    raw = input("> ").strip()

    if raw == "":
        return available

    indices = [int(x) for x in raw.split()]
    return [available[i] for i in indices]


if __name__ == "__main__":
    dirs = select_dirs_interactively()
    print(f"\nLoading from: {dirs}")
    df = load_results(dirs)
    print(f"Total trials loaded: {len(df)}")
    plot_histograms(df)
