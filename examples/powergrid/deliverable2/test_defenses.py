"""
Phase 5 Test: Defense Baseline Verification & Mitigation Comparison.

Evaluates all defense modes across the attack scenarios:
1. No Defense (Unmitigated Attack)
2. Rule-Based Defense (Baseline A)
3. Kalman-Only Defense (Baseline B)
4. Multi-Signal Defense (Proposed Solution)

Calculates:
- Maximum Frequency Deviation |f - 50.0| (Hz)
- Time Outside Safe Frequency Range [49.80, 50.20] (seconds)
- Attack Success Flag (Unsafe excursion > 3.0s without mitigation)
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
from deliverable2.plots_d2 import plot_defense_comparison


def evaluate_physical_impact(df: pd.DataFrame) -> dict:
    """Calculate physical grid metrics: Max Deviation, Time Outside Safe Range, and ASR."""
    f = df["true_frequency"].to_numpy()
    dt = df["sim_time_sec"].iloc[1] - df["sim_time_sec"].iloc[0] if len(df) > 1 else 0.1

    # 1. Maximum Deviation from Nominal 50.00 Hz
    max_dev = float(np.max(np.abs(f - 50.00)))

    # 2. Cumulative Time Outside Safe Operating Band [49.80, 50.20]
    unsafe_mask = (f < 49.80) | (f > 50.20)
    time_outside_safe_sec = float(np.sum(unsafe_mask) * dt)

    # 3. Attack Success: True if unsafe excursion exceeds 3.0 seconds
    is_attack_success = bool(time_outside_safe_sec > 3.0)

    return {
        "max_freq_dev_hz": max_dev,
        "time_outside_safe_sec": time_outside_safe_sec,
        "attack_success": is_attack_success
    }


def run_defense_evaluations():
    """Run comparative defense evaluation across Spoof, Slow-Drift, Replay, Delay, DoS."""
    cfg = DEFAULT_CONFIG
    engine = SimulationEngine(config=cfg)

    scenarios = ["spoof", "slow_drift", "replay", "delay", "dos"]
    defense_modes = ["none", "rule_based", "kalman_only", "multi_signal"]

    records = []

    print("\n=======================================================")
    print("  Phase 5: CPS Defense Baseline Comparative Evaluation")
    print("=======================================================\n")

    for sc in scenarios:
        print(f"\n[Phase 5] Evaluating Defenses on Scenario: '{sc.upper()}' ...")
        scenario_dfs = {}

        for def_mode in defense_modes:
            df = engine.run_scenario(
                scenario=sc,
                defense_mode=def_mode,
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
            scenario_dfs[def_mode] = df

            impact = evaluate_physical_impact(df)
            records.append({
                "Scenario": sc.upper(),
                "Defense_Mode": def_mode,
                "Max_Deviation_Hz": impact["max_freq_dev_hz"],
                "Time_Outside_Safe_sec": impact["time_outside_safe_sec"],
                "Attack_Success": impact["attack_success"]
            })

            print(f"  --> Defense '{def_mode:<12}': Max Dev = {impact['max_freq_dev_hz']:.3f} Hz | Unsafe Time = {impact['time_outside_safe_sec']:5.1f}s | Success: {impact['attack_success']}")

        # Save comparative plot for this scenario
        plot_path = os.path.join(cfg.results_dir, f"phase5_defense_comparison_{sc}.png")
        plot_defense_comparison(
            df_dict=scenario_dfs,
            save_path=plot_path,
            attack_title=f"{sc.upper()} Attack Mitigation"
        )

    summary_df = pd.DataFrame(records)
    summary_csv = os.path.join(cfg.results_dir, "phase5_defense_comparison_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    print("\n=======================================================")
    print("  Phase 5 Defense Evaluation Summary Table")
    print("=======================================================")
    print(summary_df.to_string(index=False))
    print(f"\nSummary table saved to: {summary_csv}")
    print("=======================================================\n")

    return summary_df


if __name__ == "__main__":
    run_defense_evaluations()
