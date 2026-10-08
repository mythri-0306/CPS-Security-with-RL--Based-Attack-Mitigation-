"""
Phase 3 Test: Multi-Signal Attack Detector Verification.

Tests the Multi-Signal Detector across:
1. Normal Operation (Measure False Alarm Rate)
2. Value Spoofing Attack (Modbus HR 0 FDI)
3. Stale-Value Replay Attack (Nominal History Playback)
4. Link Latency Delay Attack (800ms Network Jitter)
5. Packet-Loss DoS Attack (Sensor Blackout)

Saves all plots and metrics to deliverable2/results/.
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


def evaluate_detection_metrics(df: pd.DataFrame, scenario_name: str) -> dict:
    """Compute per-timestep confusion matrix, Precision, Recall, FAR, and Detection Delay."""
    y_true = df["attack_active"].astype(bool).to_numpy()
    y_pred = df["is_attack_detected"].astype(bool).to_numpy()
    t = df["sim_time_sec"].to_numpy()

    # Confusion matrix elements
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum((~y_true) & y_pred))
    tn = int(np.sum((~y_true) & (~y_pred)))
    fn = int(np.sum(y_true & (~y_pred)))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else (1.0 if not np.any(y_true) else 0.0)
    far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # Detection Delay (time from first attack active to first detection)
    attack_indices = np.where(y_true)[0]
    if len(attack_indices) > 0:
        first_attack_time = t[attack_indices[0]]
        detected_indices = np.where(y_true & y_pred)[0]
        if len(detected_indices) > 0:
            first_detection_time = t[detected_indices[0]]
            detection_delay = float(max(0.0, first_detection_time - first_attack_time))
        else:
            detection_delay = float("inf")  # Never detected
    else:
        detection_delay = 0.0  # Normal operation (no attack)

    return {
        "scenario": scenario_name,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "FAR": far,
        "Detection_Delay_sec": detection_delay
    }


def run_phase3_tests():
    """Run full Phase 3 test suite across Normal and all 4 attacks."""
    cfg = DEFAULT_CONFIG
    engine = SimulationEngine(config=cfg)

    scenarios = ["normal", "spoof", "replay", "delay", "dos"]
    metrics_summary = []

    print("\n=======================================================")
    print("  Phase 3: Multi-Signal Detector Validation Test Suite")
    print("=======================================================\n")

    for sc in scenarios:
        print(f"[Phase 3] Running Scenario: '{sc.upper()}' ...")
        df = engine.run_scenario(
            scenario=sc,
            duration_sec=60.0,
            attack_start_sec=20.0,
            attack_duration_sec=20.0,
            apply_disturbance=True,
            disturbance_time_sec=25.0,
            disturbance_load_step_mw=3.0,
            spoof_freq_hz=48.50,
            delay_ms=800.0,
            dos_loss_pct=90.0,
            add_sensor_noise=True,
            seed=42
        )

        # Save CSV
        csv_path = os.path.join(cfg.results_dir, f"phase3_detector_{sc}.csv")
        df.to_csv(csv_path, index=False)

        # Save Plot
        png_path = os.path.join(cfg.results_dir, f"phase3_detector_{sc}.png")
        plot_detector_performance(
            df=df,
            save_path=png_path,
            title=f"Phase 3: Multi-Signal Detector — {sc.upper()} Scenario"
        )

        metrics = evaluate_detection_metrics(df, scenario_name=sc)
        metrics_summary.append(metrics)

        print(f"  --> Precision: {metrics['Precision']*100:.1f}% | Recall: {metrics['Recall']*100:.1f}% | FAR: {metrics['FAR']*100:.2f}% | Delay: {metrics['Detection_Delay_sec']:.2f}s")

    summary_df = pd.DataFrame(metrics_summary)
    summary_csv = os.path.join(cfg.results_dir, "phase3_detection_metrics_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    print("\n=======================================================")
    print("  Phase 3 Multi-Signal Detection Evaluation Summary")
    print("=======================================================")
    print(summary_df.to_string(index=False))
    print(f"\nSummary metrics saved to: {summary_csv}")
    print("=======================================================\n")

    return summary_df


if __name__ == "__main__":
    run_phase3_tests()
