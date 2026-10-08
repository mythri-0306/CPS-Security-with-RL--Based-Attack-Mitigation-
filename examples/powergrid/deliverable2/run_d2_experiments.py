"""
Deliverable 2: Master Experiment Runner - run_d2_experiments.py

Executes the complete comprehensive evaluation matrix across:
- 5 Attack Scenarios: Spoofing, Slow-Drift, Replay, Delay, DoS
- 5 Evaluation Modes per scenario:
  1. Normal Operation (Clean Baseline)
  2. Attack, No Defense (Unmitigated Vulnerability)
  3. Rule-Based Defense (Baseline A)
  4. Kalman-Only Defense (Baseline B)
  5. Multi-Signal Detector + Defense (Proposed Solution)

Outputs:
- master_evaluation_metrics.csv (Full numerical matrix)
- master_evaluation_summary.md (Markdown summary report)
- Multi-panel comparative plots for every attack scenario in deliverable2/results/
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
from deliverable2.metrics import compute_metrics, ScenarioMetrics
from deliverable2.plots_d2 import plot_defense_comparison, plot_detector_performance


def run_master_experiments():
    """Run full evaluation matrix and generate master tables and plots."""
    cfg = DEFAULT_CONFIG
    engine = SimulationEngine(config=cfg)

    attacks = ["spoof", "slow_drift", "replay", "delay", "dos"]
    defense_modes = ["none", "rule_based", "kalman_only", "multi_signal"]

    all_metric_records = []

    print("\n==========================================================================")
    print("  DELIVERABLE 2: MASTER CYBER-PHYSICAL SECURITY EVALUATION EXPERIMENTS")
    print("==========================================================================\n")

    # --------------------------------------------------------------------------
    # 1. Baseline Normal Operation Evaluation
    # --------------------------------------------------------------------------
    print("[1/6] Running Normal Operation Baseline (No Attack) ...")
    df_normal = engine.run_scenario(
        scenario="normal",
        defense_mode="none",
        duration_sec=60.0,
        apply_disturbance=True,
        disturbance_time_sec=25.0,
        disturbance_load_step_mw=3.0,
        add_sensor_noise=True,
        seed=42
    )
    df_normal.to_csv(os.path.join(cfg.results_dir, "eval_normal_baseline.csv"), index=False)
    m_norm = compute_metrics(df_normal, scenario_name="NORMAL", defense_mode="none")
    all_metric_records.append(m_norm)
    print(f"      --> FAR: {m_norm.far*100:.2f}% | Max Dev: {m_norm.max_freq_dev_hz:.3f} Hz | Unsafe Time: {m_norm.time_outside_safe_sec:.1f}s")

    # --------------------------------------------------------------------------
    # 2. Evaluation Across Each Attack Scenario
    # --------------------------------------------------------------------------
    for idx, sc in enumerate(attacks, start=2):
        print(f"\n[{idx}/6] Running Attack Scenario: '{sc.upper()}' across All Defense Modes ...")
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

            # Save CSV log
            csv_file = os.path.join(cfg.results_dir, f"eval_{sc}_{def_mode}.csv")
            df.to_csv(csv_file, index=False)

            m = compute_metrics(df, scenario_name=sc, defense_mode=def_mode)
            all_metric_records.append(m)

            delay_str = f"{m.detection_delay_sec:.2f}s" if m.detection_delay_sec != float("inf") else "Never"
            print(f"      [{def_mode:<12}] Precision: {m.precision*100:5.1f}% | Recall: {m.recall*100:5.1f}% | Delay: {delay_str:<5} | Max Dev: {m.max_freq_dev_hz:.3f} Hz | Unsafe: {m.time_outside_safe_sec:4.1f}s | ASR: {m.attack_success_rate*100:.0f}%")

        # Generate comparative mitigation plot
        plot_path = os.path.join(cfg.results_dir, f"eval_defense_comparison_{sc}.png")
        plot_defense_comparison(
            df_dict=scenario_dfs,
            save_path=plot_path,
            attack_title=f"{sc.upper()} Attack Mitigation Comparison"
        )

    # --------------------------------------------------------------------------
    # 3. Compile Master Results Table
    # --------------------------------------------------------------------------
    results_df = pd.DataFrame([vars(m) for m in all_metric_records])

    # Save master CSV
    master_csv = os.path.join(cfg.results_dir, "master_evaluation_metrics.csv")
    results_df.to_csv(master_csv, index=False)

    # Save Markdown Summary Table
    md_path = os.path.join(cfg.results_dir, "master_evaluation_summary.md")
    with open(md_path, "w") as f_md:
        f_md.write("# Master Evaluation Summary — Deliverable 2\n\n")
        f_md.write("### Cyber Detection & Physical Resilience Matrix\n\n")
        
        # Format table header
        cols = ["scenario", "defense_mode", "precision", "recall", "far",
                "detection_delay_sec", "max_freq_dev_hz", "time_outside_safe_sec", "attack_success_rate"]
        f_md.write("| " + " | ".join(cols) + " |\n")
        f_md.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for _, row in results_df.iterrows():
            row_vals = [
                f"{row['scenario']}",
                f"{row['defense_mode']}",
                f"{row['precision']*100:.1f}%",
                f"{row['recall']*100:.1f}%",
                f"{row['far']*100:.2f}%",
                f"{row['detection_delay_sec']:.2f}s" if row['detection_delay_sec'] != float("inf") else "Never",
                f"{row['max_freq_dev_hz']:.3f} Hz",
                f"{row['time_outside_safe_sec']:.1f} s",
                f"{row['attack_success_rate']*100:.0f}%"
            ]
            f_md.write("| " + " | ".join(row_vals) + " |\n")
        f_md.write("\n\n")

    print("\n==========================================================================")
    print("  MASTER EVALUATION SUMMARY TABLE")
    print("==========================================================================")
    print(results_df[[
        "scenario", "defense_mode", "precision", "recall", "far",
        "detection_delay_sec", "max_freq_dev_hz", "time_outside_safe_sec", "attack_success_rate"
    ]].to_string(index=False))

    print(f"\n[Master] Results CSV saved to : {master_csv}")
    print(f"[Master] Markdown report saved to: {md_path}")
    print("==========================================================================\n")

    return results_df


if __name__ == "__main__":
    run_master_experiments()
