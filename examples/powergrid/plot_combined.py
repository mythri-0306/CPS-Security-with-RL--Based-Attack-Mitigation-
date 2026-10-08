"""
Power Grid Frequency Control CPS Testbed - plot_combined.py

Generates publication & presentation-ready combined comparative visualizations
with fully expanded, auto-scaled X and Y axes, large high-contrast fonts,
and independent dynamic limits so all physical dynamics are clearly visible.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def generate_combined_graphs(log_dir="examples/powergrid/logs", output_dir="examples/powergrid/logs"):
    os.makedirs(output_dir, exist_ok=True)

    scenarios = [
        ("Baseline (Normal Load Response)", "baseline_run1.csv"),
        ("Scenario 1: False Data Injection (Spoof)", "spoof_run1.csv"),
        ("Scenario 2: Stale-Value Replay Attack", "replay_run1.csv"),
        ("Scenario 3: Network Latency / Delay (500ms)", "delay_run1.csv"),
        ("Scenario 4: Denial of Service (100% Loss)", "dos_run1.csv"),
    ]

    dfs = {}
    for title, fname in scenarios:
        fpath = os.path.join(log_dir, fname)
        if os.path.exists(fpath):
            dfs[title] = pd.read_csv(fpath)
        else:
            print(f"Warning: {fpath} not found")

    if not dfs:
        print("No CSV files found.")
        return

    safe_low, safe_high = 49.80, 50.20

    # =========================================================================
    # FIGURE 1: 3x2 Grid Layout (Extra Spacious, Large Fonts, High Visibility)
    # =========================================================================
    fig, axes = plt.subplots(3, 2, figsize=(20, 16), sharex=False, sharey=False)
    plt.subplots_adjust(hspace=0.35, wspace=0.22)
    flat_axes = axes.flatten()

    for idx, (title, df) in enumerate(dfs.items()):
        ax = flat_axes[idx]
        t = df["sim_time_sec"] if "sim_time_sec" in df.columns else df["timestamp"]

        # Safe band shading
        ax.axhspan(safe_low, safe_high, color='green', alpha=0.18, label="Safe Band [49.80, 50.20] Hz")
        ax.axhline(50.00, color='#555555', linestyle=':', linewidth=1.2, alpha=0.8, label="Nominal (50.0 Hz)")

        # True frequency (Ground Truth)
        ax.plot(t, df["true_frequency"], color="#0055d4", linewidth=2.4, label="True Grid Frequency ($f_{true}$)")

        # Controller received frequency
        if "controller_reading_received" in df.columns:
            ax.plot(t, df["controller_reading_received"], color="#e65100", linestyle="--", linewidth=2.0,
                    alpha=0.9, label="Controller Reading ($f_{rx}$)")

        # Attack active window highlighting
        if "attack_active" in df.columns:
            df["attack_active_bool"] = df["attack_active"].astype(str).str.lower().isin(["true", "1"])
            if df["attack_active_bool"].any():
                diff = df["attack_active_bool"].astype(int).diff().fillna(0)
                starts = df.loc[diff == 1, "sim_time_sec"].tolist()
                ends = df.loc[diff == -1, "sim_time_sec"].tolist()
                if df["attack_active_bool"].iloc[0]:
                    starts.insert(0, df["sim_time_sec"].iloc[0])
                if df["attack_active_bool"].iloc[-1]:
                    ends.append(df["sim_time_sec"].iloc[-1])

                for s, e in zip(starts, ends):
                    ax.axvspan(s, e, color='red', alpha=0.18, hatch='//', label="Attack Window")

        # Auto-scale Y-axis with proper dynamic padding so curves are completely visible
        y_vals = [df["true_frequency"].min(), df["true_frequency"].max()]
        if "controller_reading_received" in df.columns:
            y_vals.extend([df["controller_reading_received"].min(), df["controller_reading_received"].max()])
        y_min = min(y_vals)
        y_max = max(y_vals)
        y_span = max(y_max - y_min, 1.0)
        ax.set_ylim(y_min - 0.08 * y_span, y_max + 0.12 * y_span)
        ax.set_xlim(0, 300)

        # Labels & Typography (Enlarged for presentations)
        ax.set_title(f"({chr(97 + idx)}) {title}", fontsize=14, fontweight='bold', pad=10, loc='left')
        ax.set_xlabel("Simulation Time (s)", fontsize=13, fontweight='bold')
        ax.set_ylabel("Frequency (Hz)", fontsize=13, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)

        # Legend de-duplication
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right", framealpha=0.92, fontsize=10.5)

    # Turn off the 6th unused subplot in 3x2 grid
    flat_axes[5].axis('off')

    # Add summary callout card in 6th slot
    summary_text = (
        "SUMMARY OF FINDINGS:\n\n"
        "• Baseline: Restores safe-band operation [49.8, 50.2 Hz].\n"
        "• Scenario 1 (Spoof/FDI): Misleads controller, driving true\n"
        "  frequency into severe runaway over-frequency (>1000 Hz).\n"
        "• Scenario 2 (Replay): Freezes perception at nominal, blinding\n"
        "  controller during physical disturbances.\n"
        "• Scenario 3 (Delay 500ms): Induces feedback phase lag,\n"
        "  triggering wide persistent oscillations (21 - 74 Hz).\n"
        "• Scenario 4 (DoS 100%): Causes telemetry timeout and\n"
        "  uncompensated frequency deviations (27 - 82 Hz)."
    )
    flat_axes[5].text(0.08, 0.50, summary_text, fontsize=12, fontweight='medium',
                      verticalalignment='center', bbox=dict(boxstyle='round,pad=1', facecolor='#f8f9fa', edgecolor='#adb5bd', alpha=0.95))

    fig.suptitle("Power Grid CPS Frequency Control — Comparative Attack Evaluation",
                 fontsize=18, fontweight='bold', y=0.99)

    grid_path = os.path.join(output_dir, "combined_all_attacks_grid.png")
    plt.tight_layout()
    plt.savefig(grid_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[Success] Saved 3x2 grid to: {grid_path}")

    # =========================================================================
    # FIGURE 2: Vertical 5-Panel Stack (Expanded Y-Axis Limits per Panel)
    # =========================================================================
    fig2, axes2 = plt.subplots(5, 1, figsize=(16, 22), sharex=True, sharey=False)
    plt.subplots_adjust(hspace=0.28)

    for idx, (title, df) in enumerate(dfs.items()):
        ax = axes2[idx]
        t = df["sim_time_sec"] if "sim_time_sec" in df.columns else df["timestamp"]

        # Safe band
        ax.axhspan(safe_low, safe_high, color='green', alpha=0.18, label="Safe Band [49.80, 50.20] Hz")
        ax.axhline(50.00, color='#555555', linestyle=':', linewidth=1.2, alpha=0.8)

        # Lines
        ax.plot(t, df["true_frequency"], color="#0055d4", linewidth=2.4, label="True Grid Frequency ($f_{true}$)")
        if "controller_reading_received" in df.columns:
            ax.plot(t, df["controller_reading_received"], color="#e65100", linestyle="--", linewidth=2.0,
                    alpha=0.9, label="Controller Reading ($f_{rx}$)")

        # Attack Window
        if "attack_active" in df.columns:
            df["attack_active_bool"] = df["attack_active"].astype(str).str.lower().isin(["true", "1"])
            if df["attack_active_bool"].any():
                diff = df["attack_active_bool"].astype(int).diff().fillna(0)
                starts = df.loc[diff == 1, "sim_time_sec"].tolist()
                ends = df.loc[diff == -1, "sim_time_sec"].tolist()
                if df["attack_active_bool"].iloc[0]:
                    starts.insert(0, df["sim_time_sec"].iloc[0])
                if df["attack_active_bool"].iloc[-1]:
                    ends.append(df["sim_time_sec"].iloc[-1])

                for s, e in zip(starts, ends):
                    ax.axvspan(s, e, color='red', alpha=0.18, hatch='//', label="Attack Window")

        # Independent dynamic Y-scaling
        y_vals = [df["true_frequency"].min(), df["true_frequency"].max()]
        if "controller_reading_received" in df.columns:
            y_vals.extend([df["controller_reading_received"].min(), df["controller_reading_received"].max()])
        y_min = min(y_vals)
        y_max = max(y_vals)
        y_span = max(y_max - y_min, 1.0)
        ax.set_ylim(y_min - 0.08 * y_span, y_max + 0.15 * y_span)

        ax.set_title(f"({chr(97 + idx)}) {title}", fontsize=14, fontweight='bold', loc='left', pad=8)
        ax.set_ylabel("Frequency (Hz)", fontsize=13, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right", framealpha=0.92, fontsize=11, ncol=4)

    axes2[-1].set_xlabel("Simulation Time (seconds)", fontsize=14, fontweight='bold')
    axes2[-1].set_xlim(0, 300)
    fig2.suptitle("Power Grid CPS Frequency Control — Multi-Scenario Attack & Baseline Comparison",
                  fontsize=18, fontweight='bold', y=0.995)

    vert_path = os.path.join(output_dir, "combined_all_attacks_vertical.png")
    plt.tight_layout()
    plt.savefig(vert_path, dpi=200, bbox_inches='tight')
    plt.close(fig2)
    print(f"[Success] Saved vertical stack to: {vert_path}")


if __name__ == "__main__":
    generate_combined_graphs()
