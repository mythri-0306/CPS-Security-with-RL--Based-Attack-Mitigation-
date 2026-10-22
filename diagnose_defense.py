"""
Phase 3 Defense Diagnosis Script - diagnose_defense.py

Evaluates and compares 5 policies under the real Kalman Estimator across 50 episodes
per attack type:
1. Always-Trust (action 0)
2. Always-Distrust (action 1)
3. Random Policy
4. Causal Oracle (action 1 iff attack_active in previous step)
5. Trained DQN Agent

Produces:
- Benchmark performance tables saved to results/phase3_diagnosis.txt
- Single-episode telemetry plots for "spoof" and "stealthy_drift" under Always-Distrust
  saved to results/
"""

import os
import sys
import argparse
from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure root and examples/powergrid are in sys.path
_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

_PG_DIR = os.path.join(_ROOT_DIR, "examples", "powergrid")
if os.path.exists(_PG_DIR) and _PG_DIR not in sys.path:
    sys.path.insert(0, _PG_DIR)

from stable_baselines3 import DQN
from grid_env import GridDefenseEnv
from estimator_interface import KalmanEstimator
from config import (
    ATTACK_TYPES, DT, SAFE_BAND_LOW, SAFE_BAND_HIGH, NOMINAL_FREQ
)


def resolve_model_path(path_arg: str) -> str:
    """Resolve model path handling directories, extensions, and best_model naming."""
    if os.path.isdir(path_arg):
        candidate = os.path.join(path_arg, "best_model.zip")
        if os.path.exists(candidate):
            return candidate
        candidate2 = os.path.join(path_arg, "best_model", "best_model.zip")
        if os.path.exists(candidate2):
            return candidate2
        candidate3 = os.path.join(path_arg, "final_model.zip")
        if os.path.exists(candidate3):
            return candidate3
        zips = [f for f in os.listdir(path_arg) if f.endswith(".zip")]
        if zips:
            return os.path.join(path_arg, zips[0])
        return candidate
    elif os.path.exists(path_arg):
        return path_arg
    elif os.path.exists(path_arg + ".zip"):
        return path_arg + ".zip"
    return path_arg if path_arg.endswith(".zip") else path_arg + ".zip"


def evaluate_single_episode(
    env: GridDefenseEnv,
    policy_type: str,
    model=None,
    seed: int = 0,
    attack_type: Optional[str] = None,
    record_trajectory: bool = False,
):
    """Run a single episode and return trajectory metrics and optional time series data."""
    options = {}
    if attack_type is not None:
        options["attack_type"] = attack_type

    obs, info = env.reset(seed=seed, options=options)
    done = False
    total_reward = 0.0
    oob_steps = 0
    clean_steps = 0
    clean_distrust_steps = 0

    # Causal oracle state
    prev_attack_active = info["attack_active"]

    trajectory = {
        "time": [],
        "true_freq": [],
        "measured_freq": [],
        "kalman_pred": [],
        "action": [],
        "attack_active": [],
        "attack_start_time": env.attack_start_time,
        "attack_type": env.attack_type,
    }

    while not done:
        # Determine action
        if policy_type == "trained":
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        elif policy_type == "random":
            action = int(env.action_space.sample())
        elif policy_type == "always_trust":
            action = 0
        elif policy_type == "always_distrust":
            action = 1
        elif policy_type == "oracle":
            # Causal oracle: uses previous step's attack_active flag
            action = 1 if prev_attack_active else 0
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")

        current_time = env.step_count * env.dt

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        if record_trajectory:
            trajectory["time"].append(current_time)
            trajectory["true_freq"].append(info["true_freq"])
            trajectory["measured_freq"].append(info["measured_freq"])
            trajectory["kalman_pred"].append(info["kalman_pred"])
            trajectory["action"].append(action)
            trajectory["attack_active"].append(info["attack_active"])

        if info["out_of_band"]:
            oob_steps += 1

        if not info["attack_active"]:
            clean_steps += 1
            if action == 1:
                clean_distrust_steps += 1

        prev_attack_active = info["attack_active"]
        done = terminated or truncated

    clean_distrust_rate = (clean_distrust_steps / clean_steps) if clean_steps > 0 else 0.0

    result = {
        "reward": total_reward,
        "oob_steps": oob_steps,
        "clean_distrust_rate": clean_distrust_rate,
    }
    if record_trajectory:
        result["trajectory"] = trajectory

    return result


def plot_episode(trajectory: Dict[str, Any], save_path: str, title: str):
    """Plot single episode time series showing true_freq, measured_freq, kalman_pred, and safe band."""
    t = np.array(trajectory["time"])
    f_true = np.array(trajectory["true_freq"])
    f_meas = np.array(trajectory["measured_freq"])
    f_pred = np.array(trajectory["kalman_pred"])
    attack_start = trajectory["attack_start_time"]
    att_type = trajectory["attack_type"]

    plt.figure(figsize=(12, 6))
    plt.plot(t, f_meas, label="Measured Freq (Sensor/Attack)", color="#d95f02", alpha=0.7, linewidth=1.5)
    plt.plot(t, f_pred, label="Kalman Prior Prediction", color="#7570b3", linestyle="--", linewidth=2.0)
    plt.plot(t, f_true, label="True Physical Freq", color="#1b9e77", linewidth=2.5)

    # Safe operating band
    plt.axhline(SAFE_BAND_HIGH, color="red", linestyle=":", label=f"Safe Band Upper ({SAFE_BAND_HIGH} Hz)")
    plt.axhline(SAFE_BAND_LOW, color="red", linestyle=":", label=f"Safe Band Lower ({SAFE_BAND_LOW} Hz)")
    plt.axhline(NOMINAL_FREQ, color="gray", linestyle="-.", alpha=0.5, label=f"Nominal ({NOMINAL_FREQ} Hz)")

    # Attack start marker
    plt.axvline(attack_start, color="black", linestyle="-", linewidth=1.5, label=f"Attack Start (t={attack_start:.1f}s)")

    plt.title(f"{title} (Attack Type: {att_type})", fontsize=14, fontweight="bold")
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Frequency (Hz)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[Plot Saved] {save_path}")


def run_diagnosis(
    model_path: str,
    episodes_per_type: int = 50,
    base_seed: int = 20000,
):
    resolved_model_path = resolve_model_path(model_path)
    print(f"Loading trained model from: {resolved_model_path}")
    model = DQN.load(resolved_model_path)

    env = GridDefenseEnv(estimator=KalmanEstimator())

    policies = [
        ("Always-Trust", "always_trust"),
        ("Always-Distrust", "always_distrust"),
        ("Random", "random"),
        ("Oracle (Causal)", "oracle"),
        ("Trained Agent", "trained"),
    ]

    print("\n" + "=" * 80)
    print(f"PHASE 3 DEFENSE DIAGNOSIS: KALMAN ESTIMATOR BENCHMARK")
    print(f"Episodes per attack type: {episodes_per_type} | Base seed: {base_seed}")
    print("=" * 80)

    # Data structure: metrics[policy_name][attack_type] = list of episode metrics
    metrics: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        p_name: {att: [] for att in ATTACK_TYPES} for p_name, _ in policies
    }

    # Generate single episode plots for Always-Distrust
    os.makedirs(os.path.join(_ROOT_DIR, "results"), exist_ok=True)
    plot_spoof_res = evaluate_single_episode(
        env=env,
        policy_type="always_distrust",
        seed=base_seed + 101,
        attack_type="spoof",
        record_trajectory=True,
    )
    plot_episode(
        plot_spoof_res["trajectory"],
        os.path.join(_ROOT_DIR, "results", "always_distrust_spoof.png"),
        "Always-Distrust Telemetry - Spoof Attack",
    )

    plot_drift_res = evaluate_single_episode(
        env=env,
        policy_type="always_distrust",
        seed=base_seed + 102,
        attack_type="stealthy_drift",
        record_trajectory=True,
    )
    plot_episode(
        plot_drift_res["trajectory"],
        os.path.join(_ROOT_DIR, "results", "always_distrust_stealthy_drift.png"),
        "Always-Distrust Telemetry - Stealthy Drift Attack",
    )

    # Run full evaluation
    for p_name, p_type in policies:
        print(f"Evaluating policy: {p_name} ...")
        for att_idx, att in enumerate(ATTACK_TYPES):
            for ep_idx in range(episodes_per_type):
                ep_seed = base_seed + (att_idx * 1000) + ep_idx
                res = evaluate_single_episode(
                    env=env,
                    policy_type=p_type,
                    model=model,
                    seed=ep_seed,
                    attack_type=att,
                    record_trajectory=False,
                )
                metrics[p_name][att].append(res)

    # Format Output Tables
    lines = []
    lines.append("=" * 95)
    lines.append("PHASE 3 DEFENSE DIAGNOSIS: OVERALL POLICY SUMMARY")
    lines.append("=" * 95)
    header = f"{'Policy':<20} | {'Mean Reward':<16} | {'Std Reward':<12} | {'Mean OOB Steps':<16} | {'Clean Distrust Rate':<20}"
    lines.append(header)
    lines.append("-" * 95)

    for p_name, _ in policies:
        all_rewards = []
        all_oob = []
        all_distrust = []
        for att in ATTACK_TYPES:
            for ep in metrics[p_name][att]:
                all_rewards.append(ep["reward"])
                all_oob.append(ep["oob_steps"])
                all_distrust.append(ep["clean_distrust_rate"])

        mean_r = np.mean(all_rewards)
        std_r = np.std(all_rewards)
        mean_oob = np.mean(all_oob)
        mean_distrust = np.mean(all_distrust) * 100.0

        row = f"{p_name:<20} | {mean_r:>14.2f}  | {std_r:>10.2f}  | {mean_oob:>14.2f}  | {mean_distrust:>18.2f}%"
        lines.append(row)
    lines.append("=" * 95)

    # Per-Attack Breakdown Tables
    lines.append("\n" + "=" * 95)
    lines.append("PER-ATTACK BREAKDOWN (Mean Reward / Mean OOB Steps / Clean Distrust %)")
    lines.append("=" * 95)
    header2 = f"{'Attack Type':<16} | {'Policy':<18} | {'Mean Reward':<14} | {'Mean OOB Steps':<14} | {'Clean Distrust %':<16}"
    lines.append(header2)
    lines.append("-" * 95)

    for att in ATTACK_TYPES:
        for p_name, _ in policies:
            att_eps = metrics[p_name][att]
            mean_r = np.mean([ep["reward"] for ep in att_eps])
            mean_oob = np.mean([ep["oob_steps"] for ep in att_eps])
            mean_distrust = np.mean([ep["clean_distrust_rate"] for ep in att_eps]) * 100.0
            row2 = f"{att:<16} | {p_name:<18} | {mean_r:>12.2f} | {mean_oob:>12.2f} | {mean_distrust:>14.2f}%"
            lines.append(row2)
        lines.append("-" * 95)

    output_text = "\n".join(lines)
    print("\n" + output_text)

    # Save to file
    out_path = os.path.join(_ROOT_DIR, "results", "phase3_diagnosis.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output_text + "\n")
    print(f"\n[Diagnosis Saved] {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", nargs="?", default="models/dqn_kalman_v1/best_model",
                        help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of evaluation episodes per attack type (default: 50)")
    parser.add_argument("--base-seed", type=int, default=20000,
                        help="Base seed for evaluation (default: 20000)")
    args = parser.parse_args()
    run_diagnosis(args.model_path, episodes_per_type=args.episodes, base_seed=args.base_seed)
