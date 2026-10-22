"""
Phase 4 Policy Sanity Check Script - sanity_check.py

Evaluates and benchmarks policies on GridDefenseEnv over randomized episodes:
1. Trained RL Agent (DQN or PPO)
2. Causal Oracle (action 1 iff attack_active in previous step)
3. Always-Trust Baseline (Action 0)
4. Always-Distrust Baseline (Action 1)
5. Random Policy

Reward-Independent Comparison Metrics:
- Mean Out-of-Band Steps (safety)
- Clean-Step Distrust Rate (%) (false alarms)
- Detection Delay & Missed Attacks (timeliness)
"""

import os
import sys
import argparse
from typing import Dict, Any, List, Optional
import numpy as np

# Ensure root and examples/powergrid are in sys.path
_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

_PG_DIR = os.path.join(_ROOT_DIR, "examples", "powergrid")
if os.path.exists(_PG_DIR) and _PG_DIR not in sys.path:
    sys.path.insert(0, _PG_DIR)

from stable_baselines3 import DQN, PPO
from grid_env import GridDefenseEnv
from estimator_interface import KalmanEstimator, PlaceholderEstimator
from config import ATTACK_TYPES, DT, OBS_EXTENDED


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate and Benchmark Power Grid Defense Policies")
    parser.add_argument("model_path", type=str, help="Path to trained model (.zip or directory containing best_model.zip)")
    parser.add_argument("--estimator", type=str, choices=["placeholder", "kalman"], default="kalman",
                        help="Estimator type: 'kalman' (default) or 'placeholder'")
    parser.add_argument("--episodes-per-type", type=int, default=50,
                        help="Number of evaluation episodes per attack type (default: 50)")
    parser.add_argument("--base-seed", type=int, default=20000,
                        help="Base seed for evaluation (strictly distinct from training seeds; default: 20000)")
    parser.add_argument("--obs-extended", dest="obs_extended", action="store_true", default=None,
                        help="Force extended 8-dimensional observation vector")
    parser.add_argument("--no-obs-extended", dest="obs_extended", action="store_false",
                        help="Force legacy 5-dimensional observation vector")
    return parser.parse_args()


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
    elif os.path.exists(os.path.join(path_arg, "best_model.zip")):
        return os.path.join(path_arg, "best_model.zip")
    return path_arg if path_arg.endswith(".zip") else path_arg + ".zip"


def load_model(model_path: str):
    """Load model supporting either DQN or PPO."""
    try:
        model = DQN.load(model_path)
        algo = "DQN"
    except Exception:
        model = PPO.load(model_path)
        algo = "PPO"
    return model, algo


def evaluate_single_episode(env: GridDefenseEnv, policy_type: str, model, seed: int, attack_type: Optional[str] = None):
    """Run a single episode and return trajectory metrics."""
    options = {}
    if attack_type is not None:
        options["attack_type"] = attack_type

    obs, info = env.reset(seed=seed, options=options)
    done = False
    total_reward = 0.0
    oob_steps = 0
    clean_steps = 0
    clean_distrust_steps = 0

    attack_onset_step = None
    first_detection_step = None
    prev_attack_active = info["attack_active"]

    while not done:
        # Action selection
        if policy_type == "trained":
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        elif policy_type == "oracle":
            # Causal oracle: uses previous step's attack_active flag
            action = 1 if prev_attack_active else 0
        elif policy_type == "random":
            action = int(env.action_space.sample())
        elif policy_type == "always_trust":
            action = 0
        elif policy_type == "always_distrust":
            action = 1
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")

        current_step = info["step"]
        if info["attack_active"] and attack_onset_step is None:
            attack_onset_step = current_step

        # Check detection during active attack
        if info["attack_active"] and action == 1 and first_detection_step is None:
            first_detection_step = current_step

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        if info["out_of_band"]:
            oob_steps += 1

        if not info["attack_active"]:
            clean_steps += 1
            if action == 1:
                clean_distrust_steps += 1

        prev_attack_active = info["attack_active"]
        done = terminated or truncated

    # Compute detection delay (steps from attack onset to first action 1)
    detected = False
    detection_delay = None
    if attack_onset_step is not None:
        if first_detection_step is not None:
            detection_delay = max(0, first_detection_step - attack_onset_step)
            detected = True
        else:
            detected = False

    clean_distrust_rate = (clean_distrust_steps / clean_steps) if clean_steps > 0 else 0.0

    return {
        "reward": total_reward,
        "oob_steps": oob_steps,
        "clean_distrust_rate": clean_distrust_rate,
        "has_attack": (attack_onset_step is not None),
        "detected": detected,
        "detection_delay": detection_delay,
        "attack_type": info["attack_type"],
    }


def run_evaluation(env: GridDefenseEnv, policy_type: str, model, episodes_per_type: int, base_seed: int):
    """Run evaluation for a policy across all attack types with controlled seeds."""
    all_results = []
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
            all_results.append(res)
            by_attack[attack_type].append(res)
            seed_counter += 1

    return all_results, by_attack


def summarize_episodes(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate summary statistics over an episode list."""
    rewards = [e["reward"] for e in episodes]
    oob = [e["oob_steps"] for e in episodes]
    distrust = [e["clean_distrust_rate"] for e in episodes]

    delays = [e["detection_delay"] for e in episodes if e["detection_delay"] is not None]
    attack_episodes = [e for e in episodes if e["has_attack"]]
    missed_count = sum(1 for e in attack_episodes if not e["detected"])
    total_attacks = len(attack_episodes)

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_oob": float(np.mean(oob)),
        "std_oob": float(np.std(oob)),
        "mean_clean_distrust": float(np.mean(distrust)),
        "mean_delay": float(np.mean(delays)) if delays else None,
        "missed_count": missed_count,
        "total_attacks": total_attacks,
    }


def main():
    args = parse_args()
    model_path = resolve_model_path(args.model_path)

    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at '{model_path}'")
        sys.exit(1)

    print(f"Loading trained model from '{model_path}'...")
    model, algo = load_model(model_path)
    print(f"Model ({algo}) loaded successfully.\n")

    # Determine obs_extended from model observation space shape if not explicitly overridden
    model_obs_dim = model.observation_space.shape[0]
    if args.obs_extended is None:
        obs_extended = bool(model_obs_dim == 8)
    else:
        obs_extended = args.obs_extended

    # Instantiate chosen estimator
    if args.estimator == "kalman":
        estimator = KalmanEstimator()
    else:
        estimator = PlaceholderEstimator()

    env = GridDefenseEnv(estimator=estimator, obs_extended=obs_extended)

    policies = [
        ("Trained Agent", "trained"),
        ("Oracle (Causal)", "oracle"),
        ("Always-Trust (Action 0)", "always_trust"),
        ("Always-Distrust (Action 1)", "always_distrust"),
        ("Random Policy", "random"),
    ]

    print("================================================================================")
    print(f"Running Policy Benchmark Sanity Check (Phase 4)")
    print(f"  - Algorithm           : {algo}")
    print(f"  - Estimator           : {args.estimator.upper()}")
    print(f"  - Obs Dimension       : {env.observation_space.shape[0]} (extended={obs_extended})")
    print(f"  - Model Path          : {model_path}")
    print(f"  - Episodes per Type   : {args.episodes_per_type} ({len(ATTACK_TYPES) * args.episodes_per_type} episodes per policy)")
    print(f"  - Evaluation Seed Base: {args.base_seed} (distinct from training)")
    print("================================================================================\n")

    overall_summaries = {}
    per_attack_summaries = {p[0]: {} for p in policies}

    for label, ptype in policies:
        print(f"Evaluating {label}...")
        all_res, by_att = run_evaluation(
            env=env,
            policy_type=ptype,
            model=model,
            episodes_per_type=args.episodes_per_type,
            base_seed=args.base_seed,
        )
        overall_summaries[label] = summarize_episodes(all_res)
        for at in ATTACK_TYPES:
            per_attack_summaries[label][at] = summarize_episodes(by_att[at])

    # 1. Overall Policy Benchmark Table (Reward-Independent focus)
    print("\n" + "=" * 95)
    print("                     POLICY BENCHMARK SUMMARY TABLE (Phase 4)")
    print("=" * 95)
    header = f"{'Policy':<28} | {'Mean OOB (steps)':>18} | {'Clean Distrust':>14} | {'Mean Return':>16}"
    print(header)
    print("-" * len(header))

    for label, _ in policies:
        s = overall_summaries[label]
        ret_str = f"{s['mean_reward']:.2f} +/- {s['std_reward']:.1f}"
        oob_str = f"{s['mean_oob']:.1f} +/- {s['std_oob']:.1f}"
        dis_str = f"{s['mean_clean_distrust'] * 100:.1f}%"
        print(f"{label:<28} | {oob_str:>18} | {dis_str:>14} | {ret_str:>16}")

    print("=" * 95 + "\n")

    # 2. Per-Attack-Type Detailed Breakdown for All Policies
    print("=" * 115)
    print("                          DETAILED BREAKDOWN PER ATTACK TYPE")
    print("=" * 115)
    sub_header = f"{'Attack Type':<16} | {'Policy':<26} | {'Mean OOB':>10} | {'Clean Distrust':>14} | {'Detect Delay':>18} | {'Mean Return':>12}"
    print(sub_header)
    print("-" * len(sub_header))

    for at in ATTACK_TYPES:
        for label, _ in policies:
            s = per_attack_summaries[label][at]
            ret_str = f"{s['mean_reward']:.1f}"
            oob_str = f"{s['mean_oob']:.1f}"
            dis_str = f"{s['mean_clean_distrust'] * 100:.1f}%"

            if at == "none":
                delay_str = "N/A"
            else:
                if s["mean_delay"] is not None and s["missed_count"] == 0:
                    delay_str = f"{s['mean_delay']:.1f} steps"
                elif s["mean_delay"] is not None:
                    delay_str = f"{s['mean_delay']:.1f} ({s['missed_count']}/{s['total_attacks']} miss)"
                else:
                    delay_str = f"MISSED ({s['missed_count']}/{s['total_attacks']})"

            print(f"{at:<16} | {label:<26} | {oob_str:>10} | {dis_str:>14} | {delay_str:>18} | {ret_str:>12}")
        print("-" * len(sub_header))

    print("=" * 115 + "\n")


if __name__ == "__main__":
    main()
