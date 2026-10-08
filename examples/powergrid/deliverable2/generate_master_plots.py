"""
Deliverable 2: Master Publication-Quality Plot Generator - generate_master_plots.py

Generates:
1. combined_all_attacks_defense_grid.png: 5-row master comparative figure of all attacks.
2. deliverable2_metrics_barchart.png: 4-panel visual summary of metrics across all defense modes.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_PG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_PG_DIR, "..", ".."))
for p in [_PG_DIR, _REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from deliverable2.config import DEFAULT_CONFIG


def generate_master_grid_plot(results_dir: str):
    """Generate 5-row comparative figure showing physical mitigation for all 5 attacks."""
    attacks = ["spoof", "slow_drift", "replay", "delay", "dos"]
    titles = {
        "spoof": "1. Value Spoofing Attack (Modbus FDI)",
        "slow_drift": "2. Stealthy Slow-Drift Attack (0.05 Hz/s Ramp)",
        "replay": "3. Stale-Value Replay Attack (Past Nominal State)",
        "delay": "4. Link Latency Delay Attack (800ms Netem Delay)",
        "dos": "5. Packet-Loss DoS Attack (Sensor Blackout)"
    }
    modes = ["none", "rule_based", "kalman_only", "multi_signal"]
    colors = {
        "none": "#d62728",         # Red: Unmitigated
        "rule_based": "#ff7f0e",   # Orange: Rule-Based
        "kalman_only": "#9467bd",  # Purple: Kalman-Only
        "multi_signal": "#2ca02c"  # Green: Multi-Signal (Proposed)
    }
    labels = {
        "none": "No Defense (Unmitigated)",
        "rule_based": "Rule-Based Defense",
        "kalman_only": "Kalman-Only Defense",
        "multi_signal": "Multi-Signal Defense (Proposed)"
    }
    styles = {
        "none": "-",
        "rule_based": "--",
        "kalman_only": "-.",
        "multi_signal": "-"
    }
    widths = {
        "none": 1.8,
        "rule_based": 1.6,
        "kalman_only": 1.6,
        "multi_signal": 2.2
    }

    fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)

    for i, sc in enumerate(attacks):
        ax = axes[i]
        ax.axhspan(49.80, 50.20, color='green', alpha=0.10, label="Safe Band (49.8 - 50.2 Hz)")
        ax.axhline(50.00, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
        ax.axvspan(20.0, 40.0, color='red', alpha=0.12, hatch='//', label="Attack Window (20s - 40s)")

        for mode in modes:
            csv_path = os.path.join(results_dir, f"eval_{sc}_{mode}.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                t = df["sim_time_sec"]
                ax.plot(t, df["true_frequency"], color=colors[mode], linestyle=styles[mode],
                        linewidth=widths[mode], label=labels[mode])

        ax.set_ylabel("Freq (Hz)", fontsize=10, fontweight='bold')
        ax.set_title(titles[sc], fontsize=11, fontweight='bold', pad=5)
        ax.grid(True, linestyle=':', alpha=0.6)
        if i == 0:
            ax.legend(loc="upper right", framealpha=0.9, fontsize=8.5, ncol=3)

    axes[-1].set_xlabel("Simulation Time (seconds)", fontsize=11, fontweight='bold')
    plt.suptitle("Deliverable 2 Master Comparison: CPS Attack Mitigation Across All Scenarios",
                 fontsize=14, fontweight='bold', y=0.995)

    save_path = os.path.join(results_dir, "master_all_attacks_mitigation_grid.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"INFO [Plots] Saved Master Grid Plot to: {save_path}")
    return save_path


def generate_metrics_barchart(results_dir: str):
    """Generate 4-panel analytical bar chart of all Deliverable 2 metrics."""
    master_csv = os.path.join(results_dir, "master_evaluation_metrics.csv")
    if not os.path.exists(master_csv):
        print("Master CSV not found.")
        return

    df = pd.read_csv(master_csv)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # -------------------------------------------------------------
    # Panel (0,0): Detection Precision & Recall by Attack (Multi-Signal)
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    ms_df = df[(df["defense_mode"] == "multi_signal") & (df["scenario"] != "NORMAL")]
    scenarios = ms_df["scenario"].tolist()
    prec = ms_df["precision"].to_numpy() * 100
    rec = ms_df["recall"].to_numpy() * 100

    x = np.arange(len(scenarios))
    w = 0.35
    ax1.bar(x - w/2, prec, w, label="Precision (%)", color="#1f77b4", edgecolor="black")
    ax1.bar(x + w/2, rec, w, label="Recall (%)", color="#2ca02c", edgecolor="black")
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, fontweight='bold')
    ax1.set_ylabel("Percentage (%)", fontweight='bold')
    ax1.set_title("Multi-Signal Detection Precision & Recall", fontweight='bold', fontsize=11)
    ax1.set_ylim(0, 115)
    ax1.grid(True, linestyle=':', alpha=0.6, axis='y')
    ax1.legend(loc="lower right")

    # -------------------------------------------------------------
    # Panel (0,1): False Alarm Rate (FAR) Across Scenarios
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    all_ms = df[df["defense_mode"] == "multi_signal"]
    far_vals = all_ms["far"].to_numpy() * 100
    sc_all = all_ms["scenario"].tolist()
    ax2.bar(sc_all, far_vals, color="#ff7f0e", edgecolor="black", width=0.5)
    ax2.axhline(1.0, color='red', linestyle='--', linewidth=1.2, label="1.0% Target Ceiling")
    ax2.set_ylabel("False Alarm Rate (%)", fontweight='bold')
    ax2.set_title("False Alarm Rate (FAR) Across All Scenarios", fontweight='bold', fontsize=11)
    ax2.set_ylim(0, max(2.5, max(far_vals) + 0.5))
    ax2.grid(True, linestyle=':', alpha=0.6, axis='y')
    ax2.legend(loc="upper right")

    # -------------------------------------------------------------
    # Panel (1,0): Detection Delay by Attack Scenario
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    delay_vals = ms_df["detection_delay_sec"].to_numpy()
    bars = ax3.bar(scenarios, delay_vals, color="#9467bd", edgecolor="black", width=0.5)
    for b, val in zip(bars, delay_vals):
        ax3.text(b.get_x() + b.get_width()/2, val + 0.15, f"{val:.2f}s", ha="center", fontweight='bold', fontsize=9)
    ax3.set_ylabel("Delay (seconds)", fontweight='bold')
    ax3.set_title("Detection Delay by Attack Type", fontweight='bold', fontsize=11)
    ax3.set_ylim(0, max(delay_vals) + 1.5)
    ax3.grid(True, linestyle=':', alpha=0.6, axis='y')

    # -------------------------------------------------------------
    # Panel (1,1): Maximum Frequency Deviation Comparison
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    spoof_comp = df[df["scenario"] == "SPOOF"]
    def_modes = spoof_comp["defense_mode"].tolist()
    max_devs = spoof_comp["max_freq_dev_hz"].tolist()
    bar_colors = ["#d62728", "#ff7f0e", "#9467bd", "#2ca02c"]
    labels_clean = ["No Defense", "Rule-Based", "Kalman-Only", "Multi-Signal"]
    bars4 = ax4.bar(labels_clean, max_devs, color=bar_colors, edgecolor="black", width=0.55)
    for b, val in zip(bars4, max_devs):
        ax4.text(b.get_x() + b.get_width()/2, val + 0.08, f"{val:.2f} Hz", ha="center", fontweight='bold', fontsize=9)
    ax4.axhline(0.20, color='red', linestyle='--', linewidth=1.2, label="Safe Band Limit (+0.20 Hz)")
    ax4.set_ylabel("Max Frequency Deviation (Hz)", fontweight='bold')
    ax4.set_title("Physical Deviation Under Spoofing (Defense Comparison)", fontweight='bold', fontsize=11)
    ax4.set_ylim(0, max(max_devs) + 0.8)
    ax4.grid(True, linestyle=':', alpha=0.6, axis='y')
    ax4.legend(loc="upper right")

    plt.suptitle("Deliverable 2 Evaluation Metrics & Security Benchmark", fontsize=13, fontweight='bold', y=0.995)
    save_path = os.path.join(results_dir, "deliverable2_metrics_barchart.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"INFO [Plots] Saved Metrics Bar Chart to: {save_path}")
    return save_path


if __name__ == "__main__":
    rdir = DEFAULT_CONFIG.results_dir
    generate_master_grid_plot(rdir)
    generate_metrics_barchart(rdir)
