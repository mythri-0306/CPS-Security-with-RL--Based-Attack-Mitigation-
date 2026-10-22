"""
Fair Baseline: Detector-Threshold Policy (detector_threshold_policy.py)

Implements a non-learning heuristic defense that observes the EXACT same 8-dim
observation vector as RL, applying threshold gating:
    distrust if |residual_norm| > T_RES
              or packet_age_norm > T_AGE
              or duplicate_flag >= 0.5
              or rolling_residual_var_norm > T_VAR

Includes latching: once triggered, holds distrust for K steps (DETECTOR_HOLD_STEPS)
to avoid chattering / flapping.

Thresholds and hold steps are tuned STRICTLY on SEED_SETS["CKPT_EVAL"] (never on FINAL_EVAL)
minimizing the Phase 5 composite score:
    score = mean_OOB_steps + 20 * clean_distrust_fraction
"""

import os
import sys
import numpy as np

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
_PG_DIR = os.path.join(_ROOT_DIR, "examples", "powergrid")
if os.path.exists(_PG_DIR) and _PG_DIR not in sys.path:
    sys.path.append(_PG_DIR)

from grid_env import GridDefenseEnv
from estimator_interface import KalmanEstimator
from config import SEED_SETS, ATTACK_TYPES, DT


class DetectorThresholdFullPolicy:
    """4-term threshold policy using residual, age, duplicate, and rolling variance."""
    def __init__(self, t_res: float = 2.0, t_age: float = 0.1, t_var: float = 2.0, hold_steps: int = 0):
        self.t_res = float(t_res)
        self.t_age = float(t_age)
        self.t_var = float(t_var)
        self.hold_steps = int(hold_steps)
        self.hold_counter = 0

    def reset(self, seed: int = None):
        self.hold_counter = 0

    def get_action(self, obs: np.ndarray, info: dict = None) -> int:
        res_norm = abs(float(obs[2]))
        age_norm = float(obs[5])
        dup_flag = float(obs[6])
        var_norm = float(obs[7])

        triggered = (
            (res_norm > self.t_res)
            or (age_norm > self.t_age)
            or (dup_flag >= 0.5)
            or (var_norm > self.t_var)
        )

        if triggered:
            self.hold_counter = self.hold_steps
            return 1
        elif self.hold_counter > 0:
            self.hold_counter -= 1
            return 1
        else:
            return 0

    def predict(self, obs: np.ndarray, state=None, deterministic: bool = True):
        return self.get_action(obs)


# Backwards compatibility alias
DetectorThresholdPolicy = DetectorThresholdFullPolicy


class DetectorThresholdTimingPolicy:
    """Uses only stale-telemetry (packet_age, duplicate_flag) and innovation residual (|residual_norm| > T_RES).
    NO rolling-variance term."""
    def __init__(self, t_res: float = 2.0, t_age: float = 0.1, hold_steps: int = 0):
        self.t_res = float(t_res)
        self.t_age = float(t_age)
        self.hold_steps = int(hold_steps)
        self.hold_counter = 0

    def reset(self, seed: int = None):
        self.hold_counter = 0

    def get_action(self, obs: np.ndarray, info: dict = None) -> int:
        res_norm = abs(float(obs[2]))
        age_norm = float(obs[5])
        dup_flag = float(obs[6])

        triggered = (
            (res_norm > self.t_res)
            or (age_norm > self.t_age)
            or (dup_flag >= 0.5)
        )

        if triggered:
            self.hold_counter = self.hold_steps
            return 1
        elif self.hold_counter > 0:
            self.hold_counter -= 1
            return 1
        else:
            return 0

    def predict(self, obs: np.ndarray, state=None, deterministic: bool = True):
        return self.get_action(obs)


def evaluate_policy_on_seeds(policy, seeds: list, episodes_per_type: int = 5):
    """Evaluates policy on given seeds across all attack types."""
    env = GridDefenseEnv(estimator=KalmanEstimator(), obs_extended=True)
    oob_all = []
    distrust_all = []

    seed_idx = 0
    for at in ATTACK_TYPES:
        for _ in range(episodes_per_type):
            seed = seeds[seed_idx % len(seeds)]
            seed_idx += 1
            policy.reset(seed=seed)
            obs, info = env.reset(seed=seed, options={"attack_type": at})
            done = False
            ep_oob = 0
            clean_steps, clean_distrust = 0, 0

            while not done:
                action = policy.get_action(obs, info)
                obs, reward, terminated, truncated, info = env.step(action)
                if info["out_of_band"]:
                    ep_oob += 1
                if not info["attack_active"]:
                    clean_steps += 1
                    if action == 1:
                        clean_distrust += 1
                done = terminated or truncated

            oob_all.append(ep_oob)
            dr = (clean_distrust / clean_steps) if clean_steps > 0 else 0.0
            distrust_all.append(dr)

    env.close()
    mean_oob = float(np.mean(oob_all))
    mean_dr = float(np.mean(distrust_all))
    score = mean_oob + 20.0 * mean_dr
    return score, mean_oob, mean_dr


def tune_timing_detector():
    """Tunes Detector-Threshold-Timing and Timing-Hold strictly on CKPT_EVAL seeds."""
    ckpt_seeds = SEED_SETS["CKPT_EVAL"]
    assert not (set(ckpt_seeds) & set(SEED_SETS["FINAL_EVAL"])), "CKPT_EVAL seeds overlap with FINAL_EVAL!"
    assert not (set(ckpt_seeds) & set(SEED_SETS["TRAIN"])), "CKPT_EVAL seeds overlap with TRAIN!"

    print(f"Tuning Detector-Threshold-Timing strictly on CKPT_EVAL seeds ({len(ckpt_seeds)} seeds)...")

    grid_res = [1.0, 1.5, 2.0, 2.5]
    grid_age = [0.05, 0.10, 0.20]
    grid_hold = [0, 3, 5, 10]

    # 1. Timing without hold (hold_steps=0)
    best_score_no_hold = float("inf")
    best_params_no_hold = None
    best_oob_no_hold = None
    best_dr_no_hold = None

    # 2. Timing with hold
    best_score_hold = float("inf")
    best_params_hold = None
    best_oob_hold = None
    best_dr_hold = None

    for r in grid_res:
        for a in grid_age:
            # First hold=0
            pol0 = DetectorThresholdTimingPolicy(t_res=r, t_age=a, hold_steps=0)
            score0, oob0, dr0 = evaluate_policy_on_seeds(pol0, ckpt_seeds, episodes_per_type=5)
            if score0 < best_score_no_hold:
                best_score_no_hold = score0
                best_params_no_hold = (r, a, 0)
                best_oob_no_hold = oob0
                best_dr_no_hold = dr0

            for h in grid_hold:
                pol = DetectorThresholdTimingPolicy(t_res=r, t_age=a, hold_steps=h)
                score, oob, dr = evaluate_policy_on_seeds(pol, ckpt_seeds, episodes_per_type=5)
                if score < best_score_hold:
                    best_score_hold = score
                    best_params_hold = (r, a, h)
                    best_oob_hold = oob
                    best_dr_hold = dr

    print("\n======================================================================")
    print("CKPT_EVAL Tuning Results:")
    print(f"  [Timing (No Hold)] Score: {best_score_no_hold:.3f} | OOB: {best_oob_no_hold:.2f}s | Clean Distrust: {best_dr_no_hold*100:.2f}% | Params: T_RES={best_params_no_hold[0]}, T_AGE={best_params_no_hold[1]}")
    print(f"  [Timing-Hold]      Score: {best_score_hold:.3f} | OOB: {best_oob_hold:.2f}s | Clean Distrust: {best_dr_hold*100:.2f}% | Params: T_RES={best_params_hold[0]}, T_AGE={best_params_hold[1]}, K={best_params_hold[2]}")
    print("======================================================================\n")

    return best_params_no_hold, best_params_hold


if __name__ == "__main__":
    p_timing, p_hold = tune_timing_detector()

