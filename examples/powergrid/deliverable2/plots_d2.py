"""
Deliverable 2 Plotting Utilities - plots_d2.py

Generates publication-quality diagnostic plots for Deliverable 2:
1. State Estimation Tracking (Actual vs Kalman Estimate vs Measured)
2. Kalman Innovation Residual & 3-Sigma Confidence Bounds
3. Residual Distribution & Normalcy Check
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_kalman_performance(df: pd.DataFrame, save_path: str, title: str = "Kalman State Estimator Performance (Normal Operation)"):
    """Generate 3-panel figure: Tracking, Residual, and Error Distribution."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=False,
                             gridspec_kw={'height_ratios': [2, 1.3, 1.1]})

    t = df["sim_time_sec"]

    # -------------------------------------------------------------
    # Panel 1: Frequency Tracking
    # -------------------------------------------------------------
    ax1 = axes[0]
    ax1.axhspan(49.80, 50.20, color='green', alpha=0.10, label="Safe Band (49.8 - 50.2 Hz)")
    ax1.axhline(50.00, color='gray', linestyle=':', linewidth=1.0, alpha=0.7)

    if "true_frequency" in df.columns:
        ax1.plot(t, df["true_frequency"], label="True Grid Frequency (Ground Truth)",
                 color="#1f77b4", linewidth=2.2)

    if "measurement_y" in df.columns:
        ax1.plot(t, df["measurement_y"], label="Sensor Measurement $y_k$",
                 color="#ff7f0e", linestyle="--", linewidth=1.2, alpha=0.75)

    if "estimated_freq" in df.columns:
        ax1.plot(t, df["estimated_freq"], label=r"Kalman Estimate $\hat{f}_{k|k}$",
                 color="#2ca02c", linestyle="-.", linewidth=2.0)

    ax1.set_ylabel("Frequency (Hz)", fontsize=11, fontweight='bold')
    ax1.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=9)

    # -------------------------------------------------------------
    # Panel 2: Innovation Residual r_k with 3-Sigma Bounds
    # -------------------------------------------------------------
    ax2 = axes[1]
    residuals = df["residual"].to_numpy()
    sigma = np.sqrt(df["residual_cov_S"].to_numpy()) if "residual_cov_S" in df.columns else np.std(residuals)

    ax2.plot(t, residuals, label=r"Innovation Residual $r_k = y_k - \hat{f}_{k|k-1}$",
             color="#d62728", linewidth=1.6)
    ax2.fill_between(t, -3 * sigma, 3 * sigma, color="#d62728", alpha=0.12,
                     label=r"$\pm 3\sigma$ Innovation Bounds")
    ax2.axhline(0.0, color='black', linestyle='-', linewidth=0.8, alpha=0.7)

    ax2.set_ylabel("Residual (Hz)", fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=9)

    # -------------------------------------------------------------
    # Panel 3: Estimation Error (True - Estimate) & Power Profile
    # -------------------------------------------------------------
    ax3 = axes[2]
    if "true_frequency" in df.columns and "estimated_freq" in df.columns:
        err = df["true_frequency"] - df["estimated_freq"]
        ax3.plot(t, err, label=r"Estimation Error $(f_{true} - \hat{f})$",
                 color="#9467bd", linewidth=1.5)
        ax3.axhline(0.0, color='black', linestyle=':', linewidth=0.8)
        ax3.set_ylabel("Error (Hz)", fontsize=11, fontweight='bold')
        ax3.set_xlabel("Simulation Time (seconds)", fontsize=11, fontweight='bold')
        ax3.grid(True, linestyle=':', alpha=0.6)
        ax3.legend(loc="upper right", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"INFO [Plots] Saved Kalman performance plot to: {save_path}")
    return save_path


def plot_detector_performance(df: pd.DataFrame, save_path: str, title: str = "Multi-Signal Attack Detector Evaluation"):
    """Generate 3-panel figure: Telemetry, 4 Sub-Scores, and Combined Score with Threshold."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True,
                             gridspec_kw={'height_ratios': [1.8, 1.4, 1.4]})

    t = df["sim_time_sec"]

    # -------------------------------------------------------------
    # Panel 1: Physical Telemetry & Attack Window
    # -------------------------------------------------------------
    ax1 = axes[0]
    ax1.axhspan(49.80, 50.20, color='green', alpha=0.10, label="Safe Band (49.8 - 50.2 Hz)")
    ax1.axhline(50.00, color='gray', linestyle=':', linewidth=1.0, alpha=0.7)

    ax1.plot(t, df["true_frequency"], label="True Frequency (Ground Truth)",
             color="#1f77b4", linewidth=2.2)
    ax1.plot(t, df["controller_reading_received"], label="Controller Received Reading",
             color="#ff7f0e", linestyle="--", linewidth=1.8, alpha=0.85)

    # Highlight ground-truth attack window
    if df["attack_active"].any():
        attack_mask = df["attack_active"].astype(bool)
        ax1.fill_between(t, 47.0, 53.0, where=attack_mask, color='red', alpha=0.15,
                         hatch='//', label="Ground-Truth Attack Active")
        ax1.set_ylim(min(48.0, df["true_frequency"].min() - 0.2),
                     max(52.0, df["true_frequency"].max() + 0.2))

    ax1.set_ylabel("Frequency (Hz)", fontsize=11, fontweight='bold')
    ax1.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=9)

    # -------------------------------------------------------------
    # Panel 2: 4 Sub-Score Channels
    # -------------------------------------------------------------
    ax2 = axes[1]
    ax2.plot(t, df["score_residual"], label=r"Signal 1: Residual Anomaly $s_{res}$ (w=0.40)",
             color="#d62728", linewidth=1.8)
    ax2.plot(t, df["score_sudden"], label=r"Signal 2: Sudden Change $s_{sud}$ (w=0.25)",
             color="#9467bd", linestyle="--", linewidth=1.6)
    ax2.plot(t, df["score_timing"], label=r"Signal 3: Timing Anomaly $s_{tim}$ (w=0.20)",
             color="#8c564b", linestyle="-.", linewidth=1.6)
    ax2.plot(t, df["score_replay"], label=r"Signal 4: Replay Pattern $s_{rep}$ (w=0.15)",
             color="#e377c2", linestyle=":", linewidth=2.0)

    ax2.set_ylabel("Sub-Score [0-1]", fontsize=11, fontweight='bold')
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=8.5)

    # -------------------------------------------------------------
    # Panel 3: Combined Attack Score & Binary Decision
    # -------------------------------------------------------------
    ax3 = axes[2]
    threshold = 0.50
    ax3.plot(t, df["combined_attack_score"], label="Combined Attack Score $S_{attack}$",
             color="#17becf", linewidth=2.2)
    ax3.axhline(threshold, color='red', linestyle='--', linewidth=1.5,
                label=f"Decision Threshold ($\tau$ = {threshold:.2f})")

    # Fill detected attack region
    if "is_attack_detected" in df.columns:
        det_mask = df["is_attack_detected"].astype(bool)
        ax3.fill_between(t, 0.0, 1.0, where=det_mask, color='orange', alpha=0.25,
                         label="Detector Output = ATTACK ALERT")

    ax3.set_ylabel("Combined Score", fontsize=11, fontweight='bold')
    ax3.set_xlabel("Simulation Time (seconds)", fontsize=11, fontweight='bold')
    ax3.set_ylim(-0.05, 1.05)
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(loc="upper right", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"INFO [Plots] Saved detector performance plot to: {save_path}")
    return save_path


def plot_defense_comparison(df_dict: dict, save_path: str, attack_title: str = "Attack Scenario"):
    """Generate 2-panel comparative figure showing physical mitigation across all defense baselines."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1.2]})

    colors = {
        "none": "#d62728",         # Red: Unmitigated Attack
        "rule_based": "#ff7f0e",   # Orange: Rule-Based Baseline
        "kalman_only": "#9467bd",  # Purple: Kalman-Only Baseline
        "multi_signal": "#2ca02c"  # Green: Proposed Multi-Signal Defense
    }
    labels = {
        "none": "1. No Defense (Unmitigated Attack)",
        "rule_based": "2. Rule-Based Defense",
        "kalman_only": "3. Kalman-Only Defense",
        "multi_signal": "4. Multi-Signal Defense (Proposed)"
    }
    styles = {
        "none": "-",
        "rule_based": "--",
        "kalman_only": "-.",
        "multi_signal": "-"
    }
    widths = {
        "none": 2.0,
        "rule_based": 1.8,
        "kalman_only": 1.8,
        "multi_signal": 2.4
    }

    # Reference data for time axis and attack window
    ref_df = next(iter(df_dict.values()))
    t = ref_df["sim_time_sec"]

    # -------------------------------------------------------------
    # Panel 1: Physical Frequency Mitigation Comparison
    # -------------------------------------------------------------
    ax1.axhspan(49.80, 50.20, color='green', alpha=0.10, label="Safe Operating Band (49.80 - 50.20 Hz)")
    ax1.axhline(50.00, color='gray', linestyle=':', linewidth=1.0, alpha=0.7)

    # Highlight attack window
    if ref_df["attack_active"].any():
        attack_mask = ref_df["attack_active"].astype(bool)
        ax1.fill_between(t, 45.0, 55.0, where=attack_mask, color='red', alpha=0.12,
                         hatch='//', label="Active Attack Window")

    for mode, df in df_dict.items():
        if mode in colors:
            ax1.plot(t, df["true_frequency"], color=colors[mode], linestyle=styles[mode],
                     linewidth=widths[mode], label=labels[mode])

    ax1.set_ylabel("True Frequency (Hz)", fontsize=11, fontweight='bold')
    ax1.set_title(f"CPS Attack Mitigation Comparison — {attack_title}",
                  fontsize=13, fontweight='bold', pad=10)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=9)

    # -------------------------------------------------------------
    # Panel 2: Generator Setpoint Control Response
    # -------------------------------------------------------------
    for mode, df in df_dict.items():
        if mode in colors:
            ax2.plot(t, df["generation_command"], color=colors[mode], linestyle=styles[mode],
                     linewidth=widths[mode], label=f"$P_{{gen}}$ ({mode})")

    ax2.set_ylabel("Generation $P_{gen}$ (MW)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Simulation Time (seconds)", fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=8.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"INFO [Plots] Saved defense comparison plot to: {save_path}")
    return save_path
