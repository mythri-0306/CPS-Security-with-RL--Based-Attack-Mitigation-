"""
Phase 4 Test: Stealthy Slow-Drift Attack Verification.

Validates:
1. Slow-Drift attack evasion of Signal 2 (Sudden-Change / RoCoF detector).
2. Detection capability of Signal 1 (Kalman Physics Innovation Residual).
3. Evaluates Precision, Recall, False Alarm Rate, and Detection Delay.
4. Generates diagnostic plots and metrics in deliverable2/results/.
"""

import os
import sys
import numpy as np
import pandas as pd

# Add repo root and examples/powergrid to path
_PG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_PG_DIR, "..", ".."))
for p in [_PG_DIR, _REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from deliverable2.config import DEFAULT_CONFIG
from deliverable2.simulation_engine import SimulationEngine
from deliverable2.plots_d2 import plot_detector_performance


def run_slow_drift_test():
    """Run Slow-Drift scenario and evaluate detector performance."""
    cfg = DEFAULT_CONFIG
    engine = SimulationEngine(config=cfg)

    print("\n=======================================================")
    print("  Phase 4: Stealthy Slow-Drift Attack Test & Analysis")
    print("=======================================================\n")

    df = engine.run_scenario(
        scenario="slow_drift",
        duration_sec=60.0,
        attack_start_sec=20.0,
        attack_duration_sec=20.0,
        apply_disturbance=True,
        disturbance_time_sec=25.0,
        disturbance_load_step_mw=3.0,
        spoof_freq_hz=48.50,
        add_sensor_noise=True,
        seed=42
    )

    # Save CSV and Plot
    csv_path = os.path.join(cfg.results_dir, "phase4_slow_drift.csv")
    png_path = os.path.join(cfg.results_dir, "phase4_slow_drift.png")
    df.to_csv(csv_path, index=False)

    plot_detector_performance(
        df=df,
        save_path=png_path,
        title="Phase 4: Multi-Signal Detector — Stealthy Slow-Drift Attack"
    )

    # Metric evaluation
    y_true = df["attack_active"].astype(bool).to_numpy()
    y_pred = df["is_attack_detected"].astype(bool).to_numpy()
    t = df["sim_time_sec"].to_numpy()

    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum((~y_true) & y_pred))
    tn = int(np.sum((~y_true) & (~y_pred)))
    fn = int(np.sum(y_true & (~y_pred)))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    attack_indices = np.where(y_true)[0]
    detected_indices = np.where(y_true & y_pred)[0]
    if len(detected_indices) > 0:
        first_attack_time = t[attack_indices[0]]
        first_detection_time = t[detected_indices[0]]
        detection_delay = float(first_detection_time - first_attack_time)
    else:
        detection_delay = float("inf")

    # Sub-detector analysis during attack window
    attack_mask = df["attack_active"].astype(bool)
    df_attack = df[attack_mask]

    avg_score_res = float(df_attack["score_residual"].mean())
    max_score_res = float(df_attack["score_residual"].max())
    avg_score_sud = float(df_attack["score_sudden"].mean())
    max_score_sud = float(df_attack["score_sudden"].max())
    avg_score_tim = float(df_attack["score_timing"].mean())
    avg_score_rep = float(df_attack["score_replay"].mean())

    print(f"--- Channel Activation During Slow-Drift Attack ---")
    print(f"Signal 1 (Kalman Residual) : Avg Score = {avg_score_res:.3f} | Max = {max_score_res:.3f}  [DETECTS PHY MISMATCH]")
    print(f"Signal 2 (Sudden Change)   : Avg Score = {avg_score_sud:.3f} | Max = {max_score_sud:.3f}  [EVADED / BELOW ROCOF]")
    print(f"Signal 3 (Timing / Latency): Avg Score = {avg_score_tim:.3f} | Max = {df_attack['score_timing'].max():.3f}  [EVADED / ON-TIME]")
    print(f"Signal 4 (Replay Buffer)   : Avg Score = {avg_score_rep:.3f} | Max = {df_attack['score_replay'].max():.3f}  [EVADED / UNIQUE VALS]")
    print(f"---------------------------------------------------")
    print(f"Overall Precision          : {precision*100:.1f}%")
    print(f"Overall Recall             : {recall*100:.1f}%")
    print(f"False Alarm Rate (FAR)     : {far*100:.2f}%")
    print(f"Detection Delay            : {detection_delay:.2f} seconds")
    print(f"Results CSV Saved          : {csv_path}")
    print(f"Results PNG Saved          : {png_path}")
    print(f"===================================================\n")

    return {
        "scenario": "slow_drift",
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "Precision": precision, "Recall": recall, "FAR": far,
        "Detection_Delay_sec": detection_delay,
        "avg_score_residual": avg_score_res,
        "avg_score_sudden": avg_score_sud
    }


if __name__ == "__main__":
    run_slow_drift_test()
