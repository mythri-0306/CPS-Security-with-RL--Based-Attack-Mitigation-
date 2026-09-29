"""
Power Grid Frequency Control CPS Testbed - plots.py

Visualization utility for simulation runs:
- Plots true frequency (solid) and controller received frequency (dashed)
- Displays configurable shaded safe operating frequency band (e.g. 49.8 - 50.2 Hz)
- Displays power dynamics (P_gen vs P_load)
- Highlights active attack windows (attack_active == True)
- Saves output PNG alongside the CSV log
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless backend for WSL / server environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_simulation_run(df_or_csv, run_label=None, safe_band=(49.8, 50.2), save_path=None):
    """Generate and save dual-panel time-series visualization for a simulation run.

    :param df_or_csv: pandas DataFrame or string path to CSV log file
    :param str run_label: Optional title/label for the simulation run
    :param tuple safe_band: (low_freq, high_freq) tuple for normal frequency band
    :param str save_path: Optional explicit output path for the PNG
    :returns str: Path to saved PNG file
    """
    # 1. Load Data
    if isinstance(df_or_csv, str):
        csv_path = df_or_csv
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        df = pd.read_csv(csv_path)
        if run_label is None:
            base = os.path.splitext(os.path.basename(csv_path))[0]
            run_label = base
    elif isinstance(df_or_csv, pd.DataFrame):
        df = df_or_csv
        if run_label is None:
            run_label = "Simulation Run"
        csv_path = None
    else:
        raise TypeError("df_or_csv must be a pandas DataFrame or file path string.")

    # Parse boolean attack_active column
    if "attack_active" in df.columns:
        df["attack_active"] = df["attack_active"].astype(str).str.lower().isin(["true", "1"])
    else:
        df["attack_active"] = False

    time_col = "sim_time_sec" if "sim_time_sec" in df.columns else "timestamp"
    t = df[time_col]

    # Determine save path
    if save_path is None:
        if csv_path:
            save_path = os.path.splitext(csv_path)[0] + ".png"
        else:
            save_path = f"logs/{run_label}.png"

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    # 2. Create Figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1.2]})

    # -------------------------------------------------------------
    # Panel 1: Frequency Tracking & Modbus Reception
    # -------------------------------------------------------------
    low_bound, high_bound = safe_band

    # Shaded safe operating band
    ax1.axhspan(low_bound, high_bound, color='green', alpha=0.12,
                label=f"Safe Band ({low_bound:.1f} - {high_bound:.1f} Hz)")
    ax1.axhline(50.00, color='gray', linestyle=':', linewidth=1.0, alpha=0.7, label="Nominal (50.0 Hz)")
    ax1.axhline(low_bound, color='green', linestyle='--', linewidth=0.8, alpha=0.6)
    ax1.axhline(high_bound, color='green', linestyle='--', linewidth=0.8, alpha=0.6)

    # True frequency (Ground Truth)
    ax1.plot(t, df["true_frequency"], color="#1f77b4", linewidth=2.0,
             label="True Frequency (Ground Truth)")

    # Controller received frequency (Dashed overlay)
    if "controller_reading_received" in df.columns:
        ax1.plot(t, df["controller_reading_received"], color="#ff7f0e",
                 linestyle="--", linewidth=1.5, alpha=0.85,
                 label="Controller Reading (Received)")

    ax1.set_ylabel("Frequency (Hz)", fontsize=11, fontweight='bold')
    ax1.set_title(f"Power Grid CPS Frequency Control — {run_label}",
                  fontsize=13, fontweight='bold', pad=10)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=9)

    # -------------------------------------------------------------
    # Panel 2: Power Dynamics (Generation vs Load)
    # -------------------------------------------------------------
    if "generation_command" in df.columns:
        ax2.plot(t, df["generation_command"], color="#2ca02c", linewidth=2.0,
                 label="Generation Setpoint $P_{gen}$ (MW)")
    if "load_demand_mw" in df.columns:
        ax2.plot(t, df["load_demand_mw"], color="#d62728", linestyle="-.", linewidth=1.8,
                 label="Load Demand $P_{load}$ (MW)")

    ax2.set_xlabel("Simulated Time (seconds)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Power (MW)", fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=9)

    # -------------------------------------------------------------
    # Highlight Attack Active Windows (for Phase 3 compatibility)
    # -------------------------------------------------------------
    if df["attack_active"].any():
        attack_diff = df["attack_active"].astype(int).diff().fillna(0)
        attack_starts = df.loc[attack_diff == 1, time_col].tolist()
        attack_ends = df.loc[attack_diff == -1, time_col].tolist()

        if df["attack_active"].iloc[0]:
            attack_starts.insert(0, df[time_col].iloc[0])
        if df["attack_active"].iloc[-1]:
            attack_ends.append(df[time_col].iloc[-1])

        for start, end in zip(attack_starts, attack_ends):
            ax1.axvspan(start, end, color='red', alpha=0.22, hatch='//', label="Attack Active Window")
            ax2.axvspan(start, end, color='red', alpha=0.22, hatch='//')

        # De-duplicate legend entries if attack windows exist
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys(), loc="upper right", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"INFO [Plots] Plot successfully saved to: {save_path}")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot simulation run time-series")
    parser.add_argument("csv_path", type=str, help="Path to the simulation CSV file")
    parser.add_argument("--run-label", type=str, default=None, help="Title label for the plot")
    parser.add_argument("--output", type=str, default=None, help="Output PNG path")
    args = parser.parse_args()

    plot_simulation_run(
        df_or_csv=args.csv_path,
        run_label=args.run_label,
        save_path=args.output
    )
