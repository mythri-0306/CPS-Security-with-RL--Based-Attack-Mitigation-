"""
Phase 4 Experiment Runner - run_variants.py

Orchestrates Phase 4 reward variant sweeps across multiple random seeds:
- Trains DQN (and optional PPO) on variants R1_baseline, R2_costly, R3_degraded, R4_switch.
- Evaluates trained models, Causal Oracle, and baselines on disjoint evaluation seeds.
- Exports results/phase4_variants.csv and results/phase4_summary.md.
"""

import os
import sys
import argparse
import subprocess
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

# Ensure repository root and examples/powergrid are in sys.path
_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

_PG_DIR = os.path.join(_ROOT_DIR, "examples", "powergrid")
if os.path.exists(_PG_DIR) and _PG_DIR not in sys.path:
    sys.path.insert(0, _PG_DIR)

from stable_baselines3 import DQN, PPO
from grid_env import GridDefenseEnv
from estimator_interface import KalmanEstimator
from config import ATTACK_TYPES, REWARD_VARIANTS
from sanity_check import evaluate_single_episode, summarize_episodes, load_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run Phase 4 Reward Variant Experiments")
    parser.add_argument("--timesteps", type=int, default=200000,
                        help="Training timesteps per run (default: 200000, use 5000 for smoke test)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="List of random training seeds (default: 0 1 2)")
    parser.add_argument("--variants", type=str, nargs="+", default=["R1_baseline", "R2_costly", "R3_degraded", "R4_switch"],
                        help="Reward variants to evaluate (default: R1..R4)")
    parser.add_argument("--with-ppo", action="store_true",
                        help="Run PPO on best variant after DQN sweep")
    parser.add_argument("--eval-episodes", type=int, default=50,
                        help="Evaluation episodes per attack type (default: 50)")
    parser.add_argument("--eval-base-seed", type=int, default=20000,
                        help="Base seed for evaluation (disjoint from training seeds; default: 20000)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training if checkpoints already exist")
    return parser.parse_args()


def train_model(algo: str, variant: str, seed: int, timesteps: int, estimator: str = "kalman"):
    run_name = f"{algo}_{variant}_s{seed}"
    model_path = os.path.join(_ROOT_DIR, "models", run_name, "best_model", "best_model.zip")
    if os.path.exists(model_path):
        print(f"[{run_name}] Checkpoint exists at {model_path}. Skipping training.")
        return run_name, model_path

    cmd = [
        sys.executable,
        os.path.join(_ROOT_DIR, "train_dqn.py"),
        "--algo", algo,
        "--reward-variant", variant,
        "--seed", str(seed),
        "--timesteps", str(timesteps),
        "--estimator", estimator,
        "--run-name", run_name,
        "--obs-extended",
    ]
    print(f"\n[Training] Running command: {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=_ROOT_DIR)
    return run_name, model_path


def evaluate_policy(env: GridDefenseEnv, policy_type: str, model=None, episodes_per_type: int = 50, base_seed: int = 20000):
    by_attack = {at: [] for at in ATTACK_TYPES}
    seed_counter = base_seed
    for attack_type in ATTACK_TYPES:
        for _ in range(episodes_per_type):
            res = evaluate_single_episode(
                env=env,
                policy_type=policy_type,
                model=model,
                seed=seed_counter,
                attack_type=attack_type,
            )
            by_attack[attack_type].append(res)
            seed_counter += 1

    summaries = {}
    for at in ATTACK_TYPES:
        summaries[at] = summarize_episodes(by_attack[at])
    return summaries


def main():
    args = parse_args()
    os.makedirs(os.path.join(_ROOT_DIR, "results"), exist_ok=True)
    os.makedirs(os.path.join(_ROOT_DIR, "logs"), exist_ok=True)

    print("=" * 80)
    print("PHASE 4 EXPERIMENT RUNNER")
    print(f"Variants     : {args.variants}")
    print(f"Seeds        : {args.seeds}")
    print(f"Timesteps    : {args.timesteps}")
    print(f"Eval Episodes: {args.eval_episodes} per attack type (base seed: {args.eval_base_seed})")
    print("=" * 80)

    # 1. Training Phase
    runs = []
    for var in args.variants:
        for s in args.seeds:
            if not args.skip_train:
                run_name, model_path = train_model("dqn", var, s, args.timesteps)
            else:
                run_name = f"dqn_{var}_s{s}"
                model_path = os.path.join(_ROOT_DIR, "models", run_name, "best_model", "best_model.zip")
            runs.append(("dqn", var, s, run_name, model_path))

    # Optional PPO run
    if args.with_ppo:
        # Default best variant to R3_degraded or R4_switch
        ppo_variant = "R3_degraded" if "R3_degraded" in args.variants else args.variants[0]
        if not args.skip_train:
            run_name, model_path = train_model("ppo", ppo_variant, 0, args.timesteps)
        else:
            run_name = f"ppo_{ppo_variant}_s0"
            model_path = os.path.join(_ROOT_DIR, "models", run_name, "best_model", "best_model.zip")
        runs.append(("ppo", ppo_variant, 0, run_name, model_path))

    # 2. Evaluation Phase
    eval_records = []
    eval_env = GridDefenseEnv(estimator=KalmanEstimator(), obs_extended=True)

    # Evaluate Baselines first (shared across variants)
    print("\n" + "=" * 80)
    print("Evaluating Baselines: Always-Trust, Always-Distrust, and Oracle (Causal)...")
    print("=" * 80)
    baselines = [
        ("baseline", "Always-Trust", "always_trust"),
        ("baseline", "Always-Distrust", "always_distrust"),
        ("oracle", "Oracle (Causal)", "oracle"),
    ]
    baseline_summaries = {}
    for b_algo, b_var, b_type in baselines:
        print(f"Evaluating {b_var} ...")
        sums = evaluate_policy(
            env=eval_env,
            policy_type=b_type,
            model=None,
            episodes_per_type=args.eval_episodes,
            base_seed=args.eval_base_seed,
        )
        baseline_summaries[b_var] = sums
        for at in ATTACK_TYPES:
            s = sums[at]
            eval_records.append({
                "algo": b_algo,
                "variant": b_var,
                "seed": 0,
                "attack_type": at,
                "mean_oob": s["mean_oob"],
                "clean_distrust_rate": s["mean_clean_distrust"] * 100.0,
                "detection_delay": s["mean_delay"] if s["mean_delay"] is not None else -1.0,
                "missed_count": s["missed_count"],
            })

    # Evaluate Trained Runs
    print("\n" + "=" * 80)
    print("Evaluating Trained Models...")
    print("=" * 80)
    for algo, var, s, run_name, model_path in runs:
        print(f"Evaluating run: {run_name} ...")
        model, _ = load_model(model_path)
        sums = evaluate_policy(
            env=eval_env,
            policy_type="trained",
            model=model,
            episodes_per_type=args.eval_episodes,
            base_seed=args.eval_base_seed,
        )
        for at in ATTACK_TYPES:
            st = sums[at]
            eval_records.append({
                "algo": algo,
                "variant": var,
                "seed": s,
                "attack_type": at,
                "mean_oob": st["mean_oob"],
                "clean_distrust_rate": st["mean_clean_distrust"] * 100.0,
                "detection_delay": st["mean_delay"] if st["mean_delay"] is not None else -1.0,
                "missed_count": st["missed_count"],
            })

    # 3. Export CSV
    df = pd.DataFrame(eval_records)
    csv_path = os.path.join(_ROOT_DIR, "results", "phase4_variants.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[Saved CSV] {csv_path}")

    # 4. Generate Markdown Summary Report
    summary_md_path = os.path.join(_ROOT_DIR, "results", "phase4_summary.md")
    lines = []
    lines.append("# Phase 4 Multi-Reward Benchmark Summary\n")
    lines.append(f"Evaluated with **Kalman State Estimator** and **Extended Observation Features** across {args.eval_episodes} episodes per attack type.\n")
    lines.append("## Reward Variants Tested\n")
    lines.append("| Variant | W_BAND | W_DISTRUST | W_DISTRUST_ATTACK | W_SWITCH | Notes |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :--- |")
    for v_name, v_cfg in REWARD_VARIANTS.items():
        lines.append(f"| **{v_name}** | {v_cfg['w_band']} | {v_cfg['w_distrust']} | {v_cfg['w_distrust_attack']} | {v_cfg['w_switch']} | {'Baseline' if '1' in v_name else 'Costly Distrust' if '2' in v_name else 'Degraded Fallback' if '3' in v_name else 'Anti-Flapping'} |")
    lines.append("\n---\n")

    lines.append("## Aggregated Performance Table (Reward-Independent Metrics)\n")
    lines.append("| Variant / Policy | Attack Type | Mean OOB (steps) | Clean Distrust Rate (%) | Detection Delay (steps) | Missed Attacks |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: |")

    # Baselines rows
    for b_label in ["Always-Trust", "Always-Distrust", "Oracle (Causal)"]:
        b_df = df[df["variant"] == b_label]
        for at in ATTACK_TYPES:
            row = b_df[b_df["attack_type"] == at].iloc[0]
            delay_str = f"{row['detection_delay']:.1f}" if row['detection_delay'] >= 0 else "N/A"
            lines.append(f"| {b_label} | `{at}` | {row['mean_oob']:.1f} | {row['clean_distrust_rate']:.1f}% | {delay_str} | {int(row['missed_count'])} |")

    # Variant aggregated rows
    for var in args.variants:
        v_df = df[(df["algo"] == "dqn") & (df["variant"] == var)]
        for at in ATTACK_TYPES:
            at_df = v_df[v_df["attack_type"] == at]
            mean_oob = at_df["mean_oob"].mean()
            std_oob = at_df["mean_oob"].std() if len(at_df) > 1 else 0.0
            mean_dis = at_df["clean_distrust_rate"].mean()
            std_dis = at_df["clean_distrust_rate"].std() if len(at_df) > 1 else 0.0

            valid_delays = at_df[at_df["detection_delay"] >= 0]["detection_delay"]
            mean_delay_str = f"{valid_delays.mean():.1f}" if len(valid_delays) > 0 else "N/A"
            total_miss = int(at_df["missed_count"].sum())

            lines.append(f"| **DQN {var}** | `{at}` | {mean_oob:.1f} ± {std_oob:.1f} | {mean_dis:.1f}% ± {std_dis:.1f}% | {mean_delay_str} | {total_miss} |")

    if args.with_ppo:
        p_df = df[df["algo"] == "ppo"]
        p_var = p_df["variant"].iloc[0]
        for at in ATTACK_TYPES:
            row = p_df[p_df["attack_type"] == at].iloc[0]
            delay_str = f"{row['detection_delay']:.1f}" if row['detection_delay'] >= 0 else "N/A"
            lines.append(f"| **PPO {p_var}** | `{at}` | {row['mean_oob']:.1f} | {row['clean_distrust_rate']:.1f}% | {delay_str} | {int(row['missed_count'])} |")

    summary_text = "\n".join(lines) + "\n"
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"[Saved Summary Markdown] {summary_md_path}")

    # 5. Plot training curves
    try:
        plot_cmd = [sys.executable, os.path.join(_ROOT_DIR, "plot_training_curves.py")]
        subprocess.call(plot_cmd, cwd=_ROOT_DIR)
    except Exception as e:
        print(f"Note: Could not run plot_training_curves.py: {e}")

    print("\nPhase 4 Experiment Sweep Complete.")


if __name__ == "__main__":
    main()
