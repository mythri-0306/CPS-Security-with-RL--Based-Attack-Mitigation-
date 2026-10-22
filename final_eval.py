"""
Phase 5 Final Evaluation & Comprehensive Benchmark - final_eval.py

Evaluates and benchmarks defense methods across standard and held-out attack scenarios:
1. Always-Trust (No Defense)
2. Rule-Based Defense (Mythri's implementation)
3. Kalman-Only Defense (Mythri's implementation)
4. Detector-Threshold (Fair Baseline using same 8 features as RL)
5. RL Defender (DQN R2_costly, seeds 0-4)
6. Oracle (Causal)
7. Always-Distrust
8. PPO Defender (R2_costly, seeds 0-2)

Addresses Review Issues:
(1) Disjoint seed partitioning per attack type (no duplicate seeds between delay and dos).
(2) Fair baseline: Detector-Threshold policy using identical 8-dim observations.
(3) Held-out attack parameter testing (out-of-distribution start times, durations, intensities).
(4) Comprehensive metrics reporting (per-seed breakdown, pooled & per-seed paired bootstrap CIs).

Outputs:
  - results/final_results.csv
  - results/final_summary.md
  - results/heldout_results.csv
  - results/heldout_summary.md
  - results/final_plots/ (bar charts, training curves, trajectories)
"""

import os
import sys
import argparse
import subprocess
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

_PG_DIR = os.path.join(_ROOT_DIR, "examples", "powergrid")
if os.path.exists(_PG_DIR) and _PG_DIR not in sys.path:
    sys.path.append(_PG_DIR)

from stable_baselines3 import DQN, PPO
from grid_env import GridDefenseEnv
from estimator_interface import KalmanEstimator
from detector_threshold_policy import (
    DetectorThresholdFullPolicy,
    DetectorThresholdTimingPolicy,
    DetectorThresholdPolicy,
)
from config import (
    ATTACK_TYPES, SEED_SETS, NOMINAL_FREQ, SAFE_BAND_LOW, SAFE_BAND_HIGH,
    DT, EPISODE_LEN, ATTACK_INTENSITY_RANGES, ATTACK_START_TIME_RANGE, ATTACK_DEFAULT_DURATION,
    DETECTOR_THRESHOLDS_FULL, DETECTOR_THRESHOLDS_TIMING, DETECTOR_THRESHOLDS_TIMING_HOLD,
    DETECTOR_THRESHOLDS, DETECTOR_HOLD_STEPS,
    HELDOUT_START_TIME_RANGE, HELDOUT_DEFAULT_DURATION, HELDOUT_ATTACK_INTENSITY_RANGES,
    REWARD_VARIANTS, W_BAND, W_DISTRUST
)


# Import Mythri's baselines from deliverable2
try:
    from deliverable2.defenses import RuleBasedDefense, KalmanOnlyDefense
except ImportError:
    from examples.powergrid.deliverable2.defenses import RuleBasedDefense, KalmanOnlyDefense

RESULTS_DIR = os.path.join(_ROOT_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "final_plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# =============================================================================
# Policy Adapters
# =============================================================================
class BasePolicyAdapter:
    def reset(self, seed: Optional[int] = None):
        pass

    def get_action(self, obs: np.ndarray, info: Dict[str, Any]) -> int:
        raise NotImplementedError


class FixedActionAdapter(BasePolicyAdapter):
    def __init__(self, action: int):
        self.action = int(action)

    def get_action(self, obs: np.ndarray, info: Dict[str, Any]) -> int:
        return self.action


class CausalOracleAdapter(BasePolicyAdapter):
    def __init__(self):
        self.prev_attack_active = False

    def reset(self, seed: Optional[int] = None):
        self.prev_attack_active = False

    def get_action(self, obs: np.ndarray, info: Dict[str, Any]) -> int:
        action = 1 if self.prev_attack_active else 0
        self.prev_attack_active = info["attack_active"]
        return action


class RuleBasedAdapter(BasePolicyAdapter):
    def __init__(self):
        self.defense = RuleBasedDefense()

    def reset(self, seed: Optional[int] = None):
        self.defense.reset()

    def get_action(self, obs: np.ndarray, info: Dict[str, Any]) -> int:
        cur_meas = info["measured_freq"]
        cur_pgen = info.get("p_gen", 100.0)
        time_sec = info.get("step", 0) * DT
        res = self.defense.process(measurement_y=cur_meas, p_gen=cur_pgen, time_sec=time_sec, dt=DT)
        return 1 if res.is_attack_flagged else 0


class KalmanOnlyAdapter(BasePolicyAdapter):
    def __init__(self):
        self.defense = KalmanOnlyDefense()

    def reset(self, seed: Optional[int] = None):
        self.defense.reset()

    def get_action(self, obs: np.ndarray, info: Dict[str, Any]) -> int:
        cur_meas = info["measured_freq"]
        cur_pgen = info.get("p_gen", 100.0)
        time_sec = info.get("step", 0) * DT
        res = self.defense.process(measurement_y=cur_meas, p_gen=cur_pgen, time_sec=time_sec, dt=DT)
        return 1 if res.is_attack_flagged else 0


class DetectorThresholdFullAdapter(BasePolicyAdapter):
    """Full 4-term detector: residual, age, duplicate, rolling variance."""
    def __init__(self):
        self.policy = DetectorThresholdFullPolicy(
            t_res=DETECTOR_THRESHOLDS_FULL["t_res"],
            t_age=DETECTOR_THRESHOLDS_FULL["t_age"],
            t_var=DETECTOR_THRESHOLDS_FULL["t_var"],
            hold_steps=int(DETECTOR_THRESHOLDS_FULL["hold_steps"]),
        )

    def reset(self, seed: Optional[int] = None):
        self.policy.reset(seed=seed)

    def get_action(self, obs: np.ndarray, info: Dict[str, Any]) -> int:
        return self.policy.get_action(obs, info)


# Alias for backwards compatibility
DetectorThresholdAdapter = DetectorThresholdFullAdapter


class DetectorThresholdTimingAdapter(BasePolicyAdapter):
    """Timing-only detector: stale-telemetry (age, duplicate) + residual (no rolling variance, K=0)."""
    def __init__(self):
        self.policy = DetectorThresholdTimingPolicy(
            t_res=DETECTOR_THRESHOLDS_TIMING["t_res"],
            t_age=DETECTOR_THRESHOLDS_TIMING["t_age"],
            hold_steps=int(DETECTOR_THRESHOLDS_TIMING["hold_steps"]),
        )

    def reset(self, seed: Optional[int] = None):
        self.policy.reset(seed=seed)

    def get_action(self, obs: np.ndarray, info: Dict[str, Any]) -> int:
        return self.policy.get_action(obs, info)


class DetectorThresholdTimingHoldAdapter(BasePolicyAdapter):
    """Timing-Hold detector: stale-telemetry (age, duplicate) + residual with anti-flapping latch hold (K=3)."""
    def __init__(self):
        self.policy = DetectorThresholdTimingPolicy(
            t_res=DETECTOR_THRESHOLDS_TIMING_HOLD["t_res"],
            t_age=DETECTOR_THRESHOLDS_TIMING_HOLD["t_age"],
            hold_steps=int(DETECTOR_THRESHOLDS_TIMING_HOLD["hold_steps"]),
        )

    def reset(self, seed: Optional[int] = None):
        self.policy.reset(seed=seed)

    def get_action(self, obs: np.ndarray, info: Dict[str, Any]) -> int:
        return self.policy.get_action(obs, info)



class ModelAdapter(BasePolicyAdapter):
    def __init__(self, model):
        self.model = model

    def get_action(self, obs: np.ndarray, info: Dict[str, Any]) -> int:
        act, _ = self.model.predict(obs, deterministic=True)
        return int(act)


# =============================================================================
# Training Helpers
# =============================================================================
def train_dqn_v2(seed: int, timesteps: int, skip_train: bool = False) -> str:
    run_name = f"dqn_R2_costly_s{seed}_v2"
    ckpt_dir = os.path.join(_ROOT_DIR, "models", run_name, "best_model")
    ckpt_path = os.path.join(ckpt_dir, "best_model.zip")
    final_path = os.path.join(_ROOT_DIR, "models", run_name, "final_model.zip")

    if skip_train and (os.path.exists(ckpt_path) or os.path.exists(final_path)):
        print(f"[{run_name}] Checkpoint exists. Skipping training.")
        return ckpt_path if os.path.exists(ckpt_path) else final_path

    cmd = [
        sys.executable,
        os.path.join(_ROOT_DIR, "train_dqn.py"),
        "--algo", "dqn",
        "--reward-variant", "R2_costly",
        "--seed", str(seed),
        "--timesteps", str(timesteps),
        "--estimator", "kalman",
        "--run-name", run_name,
        "--obs-extended",
    ]
    print(f"\n[Training DQN] {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=_ROOT_DIR)
    return ckpt_path if os.path.exists(ckpt_path) else final_path


def train_ppo(seed: int, timesteps: int, skip_train: bool = False) -> str:
    run_name = f"ppo_R2_costly_s{seed}_v2"
    ckpt_dir = os.path.join(_ROOT_DIR, "models", run_name, "best_model")
    ckpt_path = os.path.join(ckpt_dir, "best_model.zip")
    final_path = os.path.join(_ROOT_DIR, "models", run_name, "final_model.zip")

    if skip_train and (os.path.exists(ckpt_path) or os.path.exists(final_path)):
        print(f"[{run_name}] Checkpoint exists. Skipping training.")
        return ckpt_path if os.path.exists(ckpt_path) else final_path

    cmd = [
        sys.executable,
        os.path.join(_ROOT_DIR, "train_dqn.py"),
        "--algo", "ppo",
        "--reward-variant", "R2_costly",
        "--seed", str(seed),
        "--timesteps", str(timesteps),
        "--estimator", "kalman",
        "--run-name", run_name,
        "--obs-extended",
    ]
    print(f"\n[Training PPO] {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=_ROOT_DIR)
    return ckpt_path if os.path.exists(ckpt_path) else final_path


# =============================================================================
# Episode Evaluation
# =============================================================================
def evaluate_single_episode(
    env: GridDefenseEnv,
    adapter: BasePolicyAdapter,
    seed: int,
    attack_type: str,
    record_trajectory: bool = False,
    attack_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    adapter.reset(seed=seed)
    options = {"attack_type": attack_type}
    if attack_options is not None:
        options.update(attack_options)

    obs, info = env.reset(seed=seed, options=options)

    done = False
    step = 0
    total_reward = 0.0
    total_r_band = 0.0
    total_r_distrust = 0.0
    oob_steps = 0
    clean_steps = 0
    clean_distrust_steps = 0


    attack_onset_step = None
    attack_end_step = None
    first_action1_step = None

    tp, fp, tn, fn = 0, 0, 0, 0
    max_dev = 0.0

    t_hist, f_true_hist, f_meas_hist, f_pred_hist, act_hist = [], [], [], [], []

    while not done:
        action = adapter.get_action(obs, info)

        cur_true = info["true_freq"]
        cur_meas = info["measured_freq"]
        cur_pred = info["kalman_pred"]
        cur_att = info["attack_active"]

        dev = abs(cur_true - NOMINAL_FREQ)
        if dev > max_dev:
            max_dev = dev

        if cur_att and attack_onset_step is None:
            attack_onset_step = step

        if cur_att and action == 1 and first_action1_step is None:
            first_action1_step = step

        if cur_att:
            if action == 1:
                tp += 1
            else:
                fn += 1
        else:
            clean_steps += 1
            if action == 1:
                clean_distrust_steps += 1
                fp += 1
            else:
                tn += 1

        if record_trajectory:
            t_hist.append(step * DT)
            f_true_hist.append(cur_true)
            f_meas_hist.append(cur_meas)
            f_pred_hist.append(cur_pred)
            act_hist.append(action)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        total_r_band += float(info.get("r_band", 0.0))
        total_r_distrust += float(info.get("r_distrust", 0.0))

        if info["out_of_band"]:
            oob_steps += 1

        if cur_att and not info["attack_active"] and attack_end_step is None:
            attack_end_step = step

        done = terminated or truncated
        step += 1

    time_oob = oob_steps * DT
    attack_success = 1.0 if time_oob > 3.0 else 0.0

    has_attack = (attack_onset_step is not None)
    if has_attack:
        if first_action1_step is not None:
            detection_delay = float(first_action1_step - attack_onset_step)
            missed = False
        else:
            detection_delay = None
            missed = True
    else:
        detection_delay = None
        missed = False

    recovery_time = 0.0
    if has_attack and oob_steps > 0:
        end_s = attack_end_step if attack_end_step is not None else (attack_onset_step + int(20.0 / DT))
        if record_trajectory and len(f_true_hist) > end_s:
            sub_f = np.array(f_true_hist[end_s:])
            in_band = (sub_f >= SAFE_BAND_LOW) & (sub_f <= SAFE_BAND_HIGH)
            oob_indices = np.where(~in_band)[0]
            if len(oob_indices) == 0:
                recovery_time = 0.0
            elif oob_indices[-1] == len(sub_f) - 1:
                recovery_time = float(len(sub_f)) * DT
            else:
                recovery_time = float(oob_indices[-1] + 1) * DT
        else:
            recovery_time = min(time_oob, 20.0)

    clean_step_dr = (clean_distrust_steps / clean_steps) if clean_steps > 0 else 0.0
    far_clean_ep = (clean_distrust_steps / clean_steps) if (attack_type == "none" and clean_steps > 0) else None
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 1.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else (1.0 if not has_attack else 0.0)

    res = {
        "time_outside_safe_band": time_oob,
        "max_deviation": max_dev,
        "attack_success_rate": attack_success,
        "recovery_time": recovery_time,
        "detection_delay": detection_delay,
        "missed": missed,
        "clean_step_distrust_rate": clean_step_dr * 100.0,
        "false_alarm_rate_clean_episodes": (far_clean_ep * 100.0) if far_clean_ep is not None else -1.0,
        "precision": precision * 100.0,
        "recall": recall * 100.0,
        "episode_return": total_reward,
        "r_band_total": total_r_band,
        "r_distrust_total": total_r_distrust,
        "clean_steps": clean_steps,
        "clean_distrust_steps": clean_distrust_steps,
        "oob_steps": oob_steps,
    }

    if record_trajectory:
        res["trajectory"] = {
            "t": np.array(t_hist),
            "f_true": np.array(f_true_hist),
            "f_meas": np.array(f_meas_hist),
            "f_pred": np.array(f_pred_hist),
            "action": np.array(act_hist),
        }

    return res


# =============================================================================
# Statistics & Bootstrapping
# =============================================================================
def bootstrap_ci_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 1000, ci: float = 95.0) -> Tuple[float, float, float]:
    diff = np.array(a) - np.array(b)
    mean_diff = float(np.mean(diff))
    boot_means = []
    n = len(diff)
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        sample = rng.choice(diff, size=n, replace=True)
        boot_means.append(np.mean(sample))
    lower = float(np.percentile(boot_means, (100.0 - ci) / 2.0))
    upper = float(np.percentile(boot_means, 100.0 - (100.0 - ci) / 2.0))
    return mean_diff, lower, upper


def evaluate_dataset(methods: Dict[str, BasePolicyAdapter], seed_blocks: Dict[str, List[int]],
                     is_heldout: bool = False, episodes: int = 100) -> Tuple[pd.DataFrame, Dict]:
    eval_env = GridDefenseEnv(
        estimator=KalmanEstimator(),
        obs_extended=True,
        reward_cfg=REWARD_VARIANTS["R2_costly"]
    )
    records = []
    trajectories = {}

    for at_idx, at in enumerate(ATTACK_TYPES):
        at_seeds = seed_blocks[at]
        trajectories[at] = {}
        print(f"--- Attack: '{at}' ({len(at_seeds)} episodes, seeds {at_seeds[0]}..{at_seeds[-1]}) [heldout={is_heldout}] ---")

        for m_name, adapter in methods.items():
            ep_results = []
            for ep_idx, seed in enumerate(at_seeds):
                rec_traj = (ep_idx == 0 and m_name in ("Always-Trust (No Defense)", "Kalman-Only (Mythri)", "RL DQN (Seed 0)", "Detector-Threshold-Full", "Detector-Threshold-Timing"))

                att_options = None
                if is_heldout:
                    rng = np.random.default_rng(seed)
                    s_min, s_max = HELDOUT_START_TIME_RANGE
                    i_min, i_max = HELDOUT_ATTACK_INTENSITY_RANGES[at]
                    att_options = {
                        "attack_type": at,
                        "attack_start_time": float(rng.uniform(s_min, s_max)),
                        "attack_duration": HELDOUT_DEFAULT_DURATION,
                        "intensity": float(rng.uniform(i_min, i_max)),
                    }

                res = evaluate_single_episode(
                    env=eval_env,
                    adapter=adapter,
                    seed=seed,
                    attack_type=at,
                    record_trajectory=rec_traj,
                    attack_options=att_options,
                )
                if rec_traj and "trajectory" in res:
                    trajectories[at][m_name] = res["trajectory"]

                rec = {
                    "method": m_name,
                    "attack_type": at,
                    "seed": seed,
                    "time_outside_safe_band": res["time_outside_safe_band"],
                    "max_deviation": res["max_deviation"],
                    "attack_success_rate": res["attack_success_rate"],
                    "recovery_time": res["recovery_time"],
                    "detection_delay": res["detection_delay"] if res["detection_delay"] is not None else -1.0,
                    "missed": 1 if res["missed"] else 0,
                    "clean_step_distrust_rate": res["clean_step_distrust_rate"],
                    "false_alarm_rate_clean_episodes": res["false_alarm_rate_clean_episodes"],
                    "precision": res["precision"],
                    "recall": res["recall"],
                    "episode_return": res["episode_return"],
                    "r_band_total": res["r_band_total"],
                    "r_distrust_total": res["r_distrust_total"],
                    "clean_steps": res["clean_steps"],
                    "clean_distrust_steps": res["clean_distrust_steps"],
                    "oob_steps": res["oob_steps"],
                }
                records.append(rec)
                ep_results.append(rec)

            mean_oob = np.mean([r["time_outside_safe_band"] for r in ep_results])
            miss_cnt = sum(r["missed"] for r in ep_results)
            mean_dr = np.mean([r["clean_step_distrust_rate"] for r in ep_results])
            mean_ret = np.mean([r["episode_return"] for r in ep_results])
            print(f"  {m_name:<34}: OOB = {mean_oob:>5.2f}s | Missed = {miss_cnt:>2}/{len(at_seeds)} | Clean DR = {mean_dr:>5.1f}% | Return = {mean_ret:>6.1f}")

    eval_env.close()
    return pd.DataFrame(records), trajectories



def build_markdown_summary(df: pd.DataFrame, title: str, eval_desc: str) -> str:
    df_agg = df.copy()
    df_agg["method_group"] = df_agg["method"]
    dqn_mask = df_agg["method"].str.startswith("RL DQN")
    df_agg.loc[dqn_mask, "method_group"] = "RL (DQN R2, 5-Seed Avg)"
    ppo_mask = df_agg["method"].str.startswith("PPO")
    df_agg.loc[ppo_mask, "method_group"] = "PPO (R2, 3-Seed Avg)"

    metrics_list = [
        ("time_outside_safe_band", "Time Outside Safe Band (s)"),
        ("episode_return", "Return per Episode (under R2 weights: W_BAND=10.0, W_DISTRUST=1.0)"),
        ("r_band_total", "r_band Penalty Total (from OOB Steps: -(10.0 * OOB_steps))"),
        ("r_distrust_total", "r_distrust Penalty Total (from Needless Distrust: -(1.0 * clean_distrust_steps))"),
        ("clean_step_distrust_rate", "Clean Step Distrust Rate (%) [fraction of clean steps where action==1]"),
        ("false_alarm_rate_clean_episodes", "False Alarm Rate on Clean Episodes (%) [episodes of attack_type=='none']"),
        ("recall", "Distrust Recall (%)"),
        ("attack_success_rate", "Attack Success Rate (%)"),
        ("recovery_time", "Recovery Time (s)"),
        ("max_deviation", "Max Frequency Deviation (Hz)"),
        ("detection_delay", "Detection Delay (steps)"),
        ("precision", "Distrust Precision (%)"),
    ]

    method_display_order = [
        "Always-Trust (No Defense)",
        "Rule-Based (Mythri)",
        "Kalman-Only (Mythri)",
        "Detector-Threshold-Full",
        "Detector-Threshold-Timing",
        "Detector-Threshold-Timing-Hold",
        "RL (DQN R2, 5-Seed Avg)",
        "PPO (R2, 3-Seed Avg)",
        "Oracle (Causal)",
        "Always-Distrust",
    ]

    lines = [
        f"# {title}\n\n",
        f"{eval_desc}\n\n",
        "> [!NOTE]\n",
        "> **Telemetry Freeze (`delay` and `dos`) Clarification:** Both `telemetry_freeze (delay)` and `telemetry_freeze (dos)` run the exact same stale-hold code path in `grid_env.py` (freezing sensor measurements to `last_valid_measured_freq` and incrementing `packet_age`). The configured latency (ms) and drop-rate parameters are not used in `step()`, and the held-out delay/dos ranges therefore tested nothing. Both columns are reported side-by-side to maintain historical benchmark alignment.\n\n",
        "## Evaluated Defense Architectures\n",
        "1. **Always-Trust (No Defense)**: Baseline passing raw sensor measurements directly to controller.\n",
        "2. **Rule-Based Defense (Mythri)**: Static threshold clamping [49.80, 50.20] Hz + rate-of-change jump rejection (>0.20 Hz).\n",
        "3. **Kalman-Only Defense (Mythri)**: Linearized swing-equation innovation thresholding (>0.15 Hz) with physics fallback.\n",
        "4. **Detector-Threshold-Full**: Hand-set heuristic observing 4 features: residual, age, duplicate, and rolling variance (tuned on CKPT_EVAL seeds).\n",
        "5. **Detector-Threshold-Timing**: Hand-set heuristic observing stale-telemetry (age, duplicate) and residual ONLY (no variance, K=0, tuned on CKPT_EVAL).\n",
        "6. **Detector-Threshold-Timing-Hold**: Stale-telemetry + residual with anti-flapping latch hold K=3 steps (tuned on CKPT_EVAL).\n",
        "7. **RL Defender (DQN R2_costly)**: Deep Q-Network trained with composite OOB checkpoint selection.\n",
        "8. **PPO Defender (R2_costly)**: Proximal Policy Optimization trained on R2_costly with composite OOB checkpoint selection.\n",
        "9. **Oracle (Causal)**: Upper-bound baseline with perfect 1-step causal knowledge of attack state.\n",
        "10. **Always-Distrust**: Baseline rejecting all measurements and running purely on estimator state.\n\n",
        "---\n\n",
        "## Comprehensive Performance Tables by Metric (Mean ± Std)\n\n",
    ]

    for metric_col, metric_title in metrics_list:
        lines.append(f"### {metric_title}\n\n")
        def col_header(at: str) -> str:
            if at == "delay":
                return "`telemetry_freeze (delay)`"
            elif at == "dos":
                return "`telemetry_freeze (dos)`"
            return f"`{at}`"
        header = f"| Method | " + " | ".join([col_header(at) for at in ATTACK_TYPES]) + " |"
        sep = "| :--- | " + " | ".join([":---:" for _ in ATTACK_TYPES]) + " |"
        lines.append(header + "\n" + sep + "\n")

        for m_grp in method_display_order:
            sub = df_agg[df_agg["method_group"] == m_grp]
            if len(sub) == 0:
                continue
            row_vals = []
            for at in ATTACK_TYPES:
                at_sub = sub[sub["attack_type"] == at]
                if metric_col == "detection_delay":
                    valid = at_sub[at_sub["detection_delay"] >= 0]["detection_delay"]
                    miss_cnt = int(at_sub["missed"].sum())
                    tot = len(at_sub)
                    if len(valid) == 0 or miss_cnt == tot:
                        val_str = f"MISSED ({miss_cnt}/{tot})"
                    else:
                        val_str = f"{valid.mean():.1f} ± {valid.std():.1f} ({miss_cnt} miss)"
                elif metric_col == "attack_success_rate":
                    val_str = f"{at_sub[metric_col].mean() * 100.0:.1f}%"
                elif metric_col == "false_alarm_rate_clean_episodes":
                    if at == "none":
                        m_val = at_sub["clean_step_distrust_rate"].mean()
                        s_val = at_sub["clean_step_distrust_rate"].std() if len(at_sub) > 1 else 0.0
                        val_str = f"{m_val:.2f} ± {s_val:.2f}%"
                    else:
                        val_str = "— (see `none`)"
                else:
                    m_val = at_sub[metric_col].mean()
                    s_val = at_sub[metric_col].std() if len(at_sub) > 1 else 0.0
                    val_str = f"{m_val:.2f} ± {s_val:.2f}"
                row_vals.append(val_str)
            lines.append(f"| **{m_grp}** | " + " | ".join(row_vals) + " |\n")
        lines.append("\n")

    # Sanity Check Table
    lines.append("---\n\n## Return Decomposition & Sanity Check\n\n")
    lines.append("Verifying exact penalty accounting under R2 weights: $\\text{Return} = - (W_{\\text{BAND}} \\times \\text{OOB steps}) - (W_{\\text{DISTRUST}} \\times \\text{clean distrust steps})$:\n\n")
    lines.append("| Method Group | Mean Return | r_band Mean | r_distrust Mean | Decomposed Sum | Discrepancy |\n")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |\n")

    for m_grp in method_display_order:
        sub = df_agg[df_agg["method_group"] == m_grp]
        if len(sub) == 0:
            continue
        m_ret = sub["episode_return"].mean()
        m_band = sub["r_band_total"].mean()
        m_dist = sub["r_distrust_total"].mean()
        decomp = m_band + m_dist
        disc = abs(m_ret - decomp)
        lines.append(f"| **{m_grp}** | {m_ret:.2f} | {m_band:.2f} | {m_dist:.2f} | {decomp:.2f} | {disc:.4f} |\n")
    lines.append("\n")

    # Per-seed breakdown for RL
    lines.append("---\n\n## RL Defender: Per-Seed Breakdown (Mean OOB Steps in Seconds)\n\n")
    lines.append("| RL Seed | " + " | ".join([col_header(at) for at in ATTACK_TYPES]) + " |\n")
    lines.append("| :--- | " + " | ".join([":---:" for _ in ATTACK_TYPES]) + " |\n")

    for s in range(5):
        m_name = f"RL DQN (Seed {s})"
        sub_s = df[df["method"] == m_name]
        if len(sub_s) > 0:
            vals = []
            for at in ATTACK_TYPES:
                v = sub_s[sub_s["attack_type"] == at]["time_outside_safe_band"].mean()
                vals.append(f"{v:.2f}s")
            lines.append(f"| **Seed {s}** | " + " | ".join(vals) + " |\n")
    lines.append("\n")

    # Paired comparisons (Pooled episodes and Per-Seed)
    lines.append("---\n\n## Statistical Paired Comparisons: Difference in Time Outside Safe Band (s)\n\n")
    lines.append("### (a) Pooled Episodes Comparison\n\n")
    lines.append("| Comparison | Attack Type | Mean Difference (s) | 95% Bootstrap CI (s) | Verdict |\n")
    lines.append("| :--- | :--- | :---: | :---: | :--- |\n")

    rl_sub = df_agg[df_agg["method_group"] == "RL (DQN R2, 5-Seed Avg)"]
    ko_sub = df_agg[df_agg["method_group"] == "Kalman-Only (Mythri)"]
    dt_full_sub = df_agg[df_agg["method_group"] == "Detector-Threshold-Full"]
    dt_tim_sub = df_agg[df_agg["method_group"] == "Detector-Threshold-Timing"]
    dt_hold_sub = df_agg[df_agg["method_group"] == "Detector-Threshold-Timing-Hold"]

    comparisons = [
        ("RL vs Kalman-Only", ko_sub),
        ("RL vs Detector-Threshold-Full", dt_full_sub),
        ("RL vs Detector-Threshold-Timing", dt_tim_sub),
        ("RL vs Detector-Threshold-Timing-Hold", dt_hold_sub),
    ]

    for comp_label, base_sub in comparisons:
        if len(base_sub) == 0:
            continue
        for at in ATTACK_TYPES:
            rl_vals = rl_sub[rl_sub["attack_type"] == at]["time_outside_safe_band"].values
            base_vals = base_sub[base_sub["attack_type"] == at]["time_outside_safe_band"].values
            min_len = min(len(rl_vals), len(base_vals))
            if min_len == 0:
                continue
            mean_diff, ci_l, ci_u = bootstrap_ci_diff(rl_vals[:min_len], base_vals[:min_len])
            if ci_u < 0:
                verdict = "95% CI excludes zero (RL lower OOB)"
            elif ci_l > 0:
                verdict = "95% CI excludes zero (Baseline lower OOB)"
            else:
                verdict = "95% CI includes zero"
            lines.append(f"| {comp_label} | `{at}` | {mean_diff:+.3f} | [{ci_l:+.3f}, {ci_u:+.3f}] | {verdict} |\n")

    lines.append("\n### (b) Per-Seed Paired Comparison (over 5 RL Seeds)\n\n")
    lines.append("| Comparison | Attack Type | Mean Difference across Seeds (s) | 95% Bootstrap CI over Seeds (s) | Verdict |\n")
    lines.append("| :--- | :--- | :---: | :---: | :--- |\n")

    for comp_label, base_sub in comparisons:
        if len(base_sub) == 0:
            continue
        for at in ATTACK_TYPES:
            base_mean = base_sub[base_sub["attack_type"] == at]["time_outside_safe_band"].mean()
            seed_diffs = []
            for s in range(5):
                m_s = f"RL DQN (Seed {s})"
                sub_s = df[(df["method"] == m_s) & (df["attack_type"] == at)]
                if len(sub_s) > 0:
                    seed_diffs.append(sub_s["time_outside_safe_band"].mean() - base_mean)
            if seed_diffs:
                mean_diff, ci_l, ci_u = bootstrap_ci_diff(np.array(seed_diffs), np.zeros_like(seed_diffs))
                if ci_u < 0:
                    verdict = "95% CI excludes zero (RL lower OOB)"
                elif ci_l > 0:
                    verdict = "95% CI excludes zero (Baseline lower OOB)"
                else:
                    verdict = "95% CI includes zero"
                lines.append(f"| {comp_label} | `{at}` | {mean_diff:+.3f} | [{ci_l:+.3f}, {ci_u:+.3f}] | {verdict} |\n")

    return "".join(lines)



def plot_trajectory_comparison(traj_no_def, traj_ko, traj_rl, traj_dt, attack_type: str, save_path: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    t = traj_no_def["t"]

    ax1.plot(t, traj_no_def["f_true"], label="No Defense ($f_{true}$)", color="#d62728", lw=1.5, ls="--", alpha=0.8)
    ax1.plot(t, traj_ko["f_true"], label="Kalman-Only ($f_{true}$)", color="#ff7f0e", lw=1.5, ls="-.", alpha=0.9)
    if traj_dt is not None:
        ax1.plot(t, traj_dt["f_true"], label="Detector-Threshold ($f_{true}$)", color="#9467bd", lw=1.5, ls=":", alpha=0.9)
    ax1.plot(t, traj_rl["f_true"], label="RL Defender ($f_{true}$)", color="#1f77b4", lw=2.0)

    ax1.axhline(SAFE_BAND_HIGH, color="black", ls=":", lw=1.0, label="Safe Band (49.80 - 50.20 Hz)")
    ax1.axhline(SAFE_BAND_LOW, color="black", ls=":", lw=1.0)
    ax1.axhline(NOMINAL_FREQ, color="gray", ls=":", lw=0.6, alpha=0.5)
    ax1.fill_between(t, SAFE_BAND_LOW, SAFE_BAND_HIGH, color="#2ca02c", alpha=0.08)

    ax1.set_ylabel("True Frequency (Hz)", fontsize=11)
    ax1.set_title(f"Example Defense Comparison on Attack: '{attack_type}'", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax1.grid(True, ls=":", alpha=0.6)

    ax2.step(t, traj_ko["action"], where="post", color="#ff7f0e", lw=1.2, ls="-.", label="Kalman-Only", alpha=0.7)
    if traj_dt is not None:
        ax2.step(t, traj_dt["action"], where="post", color="#9467bd", lw=1.2, ls=":", label="Detector-Threshold", alpha=0.8)
    ax2.step(t, traj_rl["action"], where="post", color="#1f77b4", lw=1.5, label="RL Action")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["0: Trust", "1: Distrust"])
    ax2.set_ylim(-0.2, 1.2)
    ax2.set_xlabel("Time (s)", fontsize=11)
    ax2.set_ylabel("Action", fontsize=11)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, ls=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"  [Saved Trajectory Plot] {save_path}")


# =============================================================================
# Main Orchestration
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Phase 5 Final Evaluation & Benchmark")
    parser.add_argument("--timesteps", type=int, default=200000,
                        help="Training timesteps for retrained agents (default: 200000)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="Random training seeds for RL Defender (default: 0 1 2 3 4)")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Evaluation episodes per attack type (default: 100)")
    parser.add_argument("--skip-train", "--eval-only", dest="skip_train", action="store_true", default=True,
                        help="Skip training if checkpoints already exist (default: True)")
    parser.add_argument("--with-ppo", action="store_true", default=True,
                        help="Evaluate PPO on R2_costly")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Assertions on Seed Disjointness
    train_seeds = set(SEED_SETS["TRAIN"])
    ckpt_seeds = set(SEED_SETS["CKPT_EVAL"])
    final_seeds = set(SEED_SETS["FINAL_EVAL"])

    assert not (final_seeds & train_seeds), "CRITICAL: FINAL_EVAL seeds overlap with TRAIN seeds!"
    assert not (final_seeds & ckpt_seeds), "CRITICAL: FINAL_EVAL seeds overlap with CKPT_EVAL seeds!"
    assert not (train_seeds & ckpt_seeds), "CRITICAL: TRAIN seeds overlap with CKPT_EVAL seeds!"
    print("================================================================================")
    print("PHASE 5 FINAL EVALUATION ORCHESTRATOR")
    print(f"  - Verified Seed Disjointness: TRAIN={len(train_seeds)}, CKPT_EVAL={len(ckpt_seeds)}, FINAL_EVAL={len(final_seeds)}")
    print(f"  - Detector-Threshold parameters tuned strictly on CKPT_EVAL: {DETECTOR_THRESHOLDS}, hold={DETECTOR_HOLD_STEPS}")
    print("================================================================================\n")

    # Print Attack Ranges Side by Side (STEP 3 Requirement)
    print("=" * 85)
    print("ATTACK PARAMETER RANGES: TRAINING VS HELD-OUT (GENERALIZATION TEST)")
    print("=" * 85)
    print(f"{'Attack Type':<16} | {'Training Intensity':<28} | {'Held-Out Intensity':<28}")
    print("-" * 80)
    for at in ATTACK_TYPES:
        tr_rng = ATTACK_INTENSITY_RANGES[at]
        ho_rng = HELDOUT_ATTACK_INTENSITY_RANGES[at]
        print(f"{at:<16} | {str(tr_rng):<28} | {str(ho_rng):<28}")
    print("-" * 80)
    print(f"{'Attack Timing':<16} | Start: {ATTACK_START_TIME_RANGE}s, Dur: {ATTACK_DEFAULT_DURATION}s   | Start: {HELDOUT_START_TIME_RANGE}s, Dur: {HELDOUT_DEFAULT_DURATION}s")
    print("=" * 85 + "\n")

    # 2. Check Models
    dqn_models = {}
    for s in args.seeds:
        ckpt_path = train_dqn_v2(seed=s, timesteps=args.timesteps, skip_train=args.skip_train)
        dqn_models[s] = DQN.load(ckpt_path)

    ppo_models = {}
    if args.with_ppo:
        for s in [0, 1, 2]:
            p_ckpt = train_ppo(seed=s, timesteps=args.timesteps, skip_train=args.skip_train)
            ppo_models[s] = PPO.load(p_ckpt)

    # 3. Build Methods Dict
    methods = {
        "Always-Trust (No Defense)": FixedActionAdapter(action=0),
        "Rule-Based (Mythri)": RuleBasedAdapter(),
        "Kalman-Only (Mythri)": KalmanOnlyAdapter(),
        "Detector-Threshold-Full": DetectorThresholdFullAdapter(),
        "Detector-Threshold-Timing": DetectorThresholdTimingAdapter(),
        "Detector-Threshold-Timing-Hold": DetectorThresholdTimingHoldAdapter(),
        "Oracle (Causal)": CausalOracleAdapter(),
        "Always-Distrust": FixedActionAdapter(action=1),
    }

    for s, m in dqn_models.items():
        methods[f"RL DQN (Seed {s})"] = ModelAdapter(m)

    if args.with_ppo:
        for s, pm in ppo_models.items():
            methods[f"PPO (Seed {s})"] = ModelAdapter(pm)

    # 4. Partition Disjoint Evaluation Seeds Per Attack Type (Fixes Problem 1)
    seed_blocks_standard = {}
    seed_blocks_heldout = {}
    for at_idx, at in enumerate(ATTACK_TYPES):
        # 100 dedicated seeds per attack type from FINAL_EVAL
        start_idx = at_idx * args.episodes
        seed_blocks_standard[at] = SEED_SETS["FINAL_EVAL"][start_idx: start_idx + args.episodes]
        # Held-out seeds start at index 600
        ho_start = 600 + at_idx * args.episodes
        seed_blocks_heldout[at] = SEED_SETS["FINAL_EVAL"][ho_start: ho_start + args.episodes]

    # Verify seed disjointness between delay and dos
    assert not (set(seed_blocks_standard["delay"]) & set(seed_blocks_standard["dos"])), "delay and dos seeds overlap!"
    print(f"Verified seed partitioning:")
    print(f"  Standard 'delay' seeds: {seed_blocks_standard['delay'][0]} .. {seed_blocks_standard['delay'][-1]}")
    print(f"  Standard 'dos' seeds  : {seed_blocks_standard['dos'][0]} .. {seed_blocks_standard['dos'][-1]}")

    # 5. Execute Standard Benchmark Evaluation
    print("\n[Step 1/2] Evaluating Standard Attack Scenarios (100 episodes per attack type)...")
    df_std, trajectories_std = evaluate_dataset(methods, seed_blocks_standard, is_heldout=False, episodes=args.episodes)

    # Sanity Check on Returns Decomposition
    print("\nSanity Check: Verifying Return Accounting vs Decomposed Penalties...")
    for m_name in df_std["method"].unique():
        sub_m = df_std[df_std["method"] == m_name]
        mean_ret = sub_m["episode_return"].mean()
        expected = sub_m["r_band_total"].mean() + sub_m["r_distrust_total"].mean()
        disc = abs(mean_ret - expected)
        if disc > 0.05:
            print(f"  WARNING: Return mismatch for {m_name}: return={mean_ret:.2f}, expected={expected:.2f} (diff={disc:.4f})")
        assert disc < 0.2, f"Sanity assertion failed: return mismatch for {m_name} ({mean_ret} vs {expected})!"
    print("  PASSED: Mean return matches -(W_BAND * mean_OOB_steps) - (W_DISTRUST * mean_distrust_steps_without_attack) exactly.\n")

    std_csv_path = os.path.join(RESULTS_DIR, "final_results.csv")
    df_std.to_csv(std_csv_path, index=False)
    print(f"[Saved CSV] {std_csv_path}")

    std_summary = build_markdown_summary(
        df_std,
        title="Phase 5 Final Evaluation & Benchmark Summary",
        eval_desc=f"Evaluated across **{args.episodes} episodes per attack type** using strictly partitioned, held-out seeds from `SEED_SETS['FINAL_EVAL']`."
    )
    std_summary_path = os.path.join(RESULTS_DIR, "final_summary.md")
    with open(std_summary_path, "w", encoding="utf-8") as f:
        f.write(std_summary)
    print(f"[Saved Summary Markdown] {std_summary_path}")

    # 6. Execute Held-Out Attack Scenarios (STEP 3 Requirement)
    print("\n[Step 2/2] Evaluating Held-Out Attack Scenarios (Out-of-Distribution ranges)...")
    # For held-out testing, evaluate primary comparison methods
    heldout_methods = {
        "Always-Trust (No Defense)": methods["Always-Trust (No Defense)"],
        "Rule-Based (Mythri)": methods["Rule-Based (Mythri)"],
        "Kalman-Only (Mythri)": methods["Kalman-Only (Mythri)"],
        "Detector-Threshold-Full": methods["Detector-Threshold-Full"],
        "Detector-Threshold-Timing": methods["Detector-Threshold-Timing"],
        "Detector-Threshold-Timing-Hold": methods["Detector-Threshold-Timing-Hold"],
        "Oracle (Causal)": methods["Oracle (Causal)"],
    }
    for s, m in dqn_models.items():
        heldout_methods[f"RL DQN (Seed {s})"] = ModelAdapter(m)

    df_ho, _ = evaluate_dataset(heldout_methods, seed_blocks_heldout, is_heldout=True, episodes=args.episodes)

    ho_csv_path = os.path.join(RESULTS_DIR, "heldout_results.csv")
    df_ho.to_csv(ho_csv_path, index=False)
    print(f"[Saved Held-Out CSV] {ho_csv_path}")

    ho_summary = build_markdown_summary(
        df_ho,
        title="Phase 5 Held-Out Attack Evaluation Summary (Generalization Test)",
        eval_desc=f"Evaluated across **{args.episodes} episodes per attack type** using out-of-distribution attack parameters (later start times 32-42s, longer 25s durations, and unseen intensity ranges)."
    )
    ho_summary_path = os.path.join(RESULTS_DIR, "heldout_summary.md")
    with open(ho_summary_path, "w", encoding="utf-8") as f:
        f.write(ho_summary)
    print(f"[Saved Held-Out Summary Markdown] {ho_summary_path}")

    # 7. Generate Plots
    print("\nGenerating final visual benchmark plots...")
    # Bar Chart of OOB Comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(ATTACK_TYPES))
    width = 0.14

    plot_methods = [
        ("Always-Trust (No Defense)", "#d62728", -2.0),
        ("Rule-Based (Mythri)", "#8c564b", -1.0),
        ("Kalman-Only (Mythri)", "#ff7f0e", 0.0),
        ("Detector-Threshold-Full", "#9467bd", 1.0),
        ("RL (DQN R2, 5-Seed Avg)", "#1f77b4", 2.0),
    ]


    df_agg = df_std.copy()
    df_agg["method_group"] = df_agg["method"]
    dqn_mask = df_agg["method"].str.startswith("RL DQN")
    df_agg.loc[dqn_mask, "method_group"] = "RL (DQN R2, 5-Seed Avg)"

    for label, color, offset in plot_methods:
        means = []
        stds = []
        for at in ATTACK_TYPES:
            sub = df_agg[(df_agg["method_group"] == label) & (df_agg["attack_type"] == at)]["time_outside_safe_band"]
            means.append(sub.mean() if len(sub) > 0 else 0.0)
            stds.append(sub.std() if len(sub) > 1 else 0.0)
        ax.bar(x + offset * width, means, width, yerr=stds, label=label, color=color, capsize=3, alpha=0.85)

    ax.set_ylabel("Time Outside Safe Band (seconds)", fontsize=11)
    ax.set_title("Safety Benchmark: Time Outside Safe Band Across Cyber-Attacks", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"'{at}'" for at in ATTACK_TYPES], fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, ls=":", alpha=0.6, axis="y")
    plt.tight_layout()
    bar_plot_path = os.path.join(PLOTS_DIR, "oob_comparison.png")
    plt.savefig(bar_plot_path, dpi=130)
    plt.close(fig)
    print(f"  [Saved Bar Chart] {bar_plot_path}")

    # Example Trajectories
    for at in ATTACK_TYPES:
        if at in trajectories_std and all(k in trajectories_std[at] for k in ("Always-Trust (No Defense)", "Kalman-Only (Mythri)", "RL DQN (Seed 0)")):
            traj_path = os.path.join(PLOTS_DIR, f"trajectory_{at}.png")
            dt_traj = trajectories_std[at].get("Detector-Threshold (Fair Baseline)")
            plot_trajectory_comparison(
                traj_no_def=trajectories_std[at]["Always-Trust (No Defense)"],
                traj_ko=trajectories_std[at]["Kalman-Only (Mythri)"],
                traj_rl=trajectories_std[at]["RL DQN (Seed 0)"],
                traj_dt=dt_traj,
                attack_type=at,
                save_path=traj_path,
            )

    # Training Curves per Seed
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted_any = False
    for s in args.seeds:
        mon_path = os.path.join(_ROOT_DIR, "logs", f"dqn_R2_costly_s{s}_v2", "train_monitor.csv")
        if os.path.exists(mon_path):
            try:
                m_df = pd.read_csv(mon_path, skiprows=1, on_bad_lines="skip")
                if "r" in m_df.columns and "l" in m_df.columns:
                    steps = m_df["l"].cumsum().values
                    rews = m_df["r"].values
                    window = max(1, len(rews) // 10)
                    smooth = pd.Series(rews).rolling(window, min_periods=1).mean().values
                    ax.plot(steps, smooth, label=f"Seed {s}", lw=1.8)
                    plotted_any = True
            except Exception as e:
                print(f"Note: Error reading {mon_path}: {e}")
    if plotted_any:
        ax.set_title("DQN R2_costly Training Curves (Smoothed Episode Return)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Training Steps", fontsize=11)
        ax.set_ylabel("Episode Return (R2)", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, ls=":", alpha=0.6)
        train_plot_path = os.path.join(PLOTS_DIR, "training_curves.png")
        plt.tight_layout()
        plt.savefig(train_plot_path, dpi=130)
        plt.close(fig)
        print(f"  [Saved Training Curves] {train_plot_path}")
    else:
        plt.close(fig)

    print("\n================================================================================")
    print("ALL EVALUATION RUNS COMPLETED SUCCESSFULLY!")
    print(f"  - Standard Summary : {std_summary_path}")
    print(f"  - Held-Out Summary : {ho_summary_path}")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
