"""
Phase 5 Training Script - train_dqn.py

Trains Reinforcement Learning defense agents (DQN or PPO) with MlpPolicy on
GridDefenseEnv, supporting Phase 4+ named reward variants, extended observation
features, Kalman state estimation, and OOB-based composite checkpoint selection
(Phase 5 STEP 2).
"""

import os
import sys
import argparse
import math
from typing import Optional

# Ensure repository root and examples/powergrid are in sys.path
_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

_PG_DIR = os.path.join(_ROOT_DIR, "examples", "powergrid")
if os.path.exists(_PG_DIR) and _PG_DIR not in sys.path:
    sys.path.append(_PG_DIR)

import numpy as np
import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from grid_env import GridDefenseEnv
from estimator_interface import KalmanEstimator, PlaceholderEstimator
from config import (
    W_BAND, W_DISTRUST, W_DISTRUST_ATTACK, W_SWITCH,
    REWARD_VARIANTS, OBS_EXTENDED, EPISODE_LEN,
    CKPT_DISTRUST_WEIGHT, CKPT_EVAL_EPISODES, CKPT_EVAL_FREQ,
    SEED_SETS, ATTACK_TYPES,
)


# ---------------------------------------------------------------------------
# Phase 5 – Custom OOB-based checkpoint callback
# ---------------------------------------------------------------------------
class OobCheckpointCallback(BaseCallback):
    """
    Evaluates on CKPT_EVAL seeds every `eval_freq` steps.
    Saves best model when the composite score is minimised:
        score = mean_OOB_steps + CKPT_DISTRUST_WEIGHT * clean_distrust_rate
    Ties broken by higher mean return.

    Formula is documented in config.CKPT_DISTRUST_WEIGHT.
    """

    def __init__(self, make_env_fn, save_path: str, eval_freq: int,
                 episodes_per_type: int, seeds, distrust_weight: float,
                 verbose: int = 1):
        super().__init__(verbose)
        self.make_env_fn = make_env_fn
        self.save_path = save_path
        self.eval_freq = eval_freq
        self.episodes_per_type = episodes_per_type
        self.seeds = list(seeds)
        self.distrust_weight = distrust_weight
        self.best_score = math.inf
        self.best_return = -math.inf
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        env = self.make_env_fn()
        oob_all, distrust_all, reward_all = [], [], []

        seed_idx = 0
        for attack_type in ATTACK_TYPES:
            for _ in range(self.episodes_per_type):
                seed = self.seeds[seed_idx % len(self.seeds)]
                seed_idx += 1
                obs, info = env.reset(seed=seed, options={"attack_type": attack_type})
                done = False
                ep_reward, ep_oob = 0.0, 0
                clean_steps, clean_distrust = 0, 0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(int(action))
                    ep_reward += float(reward)
                    if info["out_of_band"]:
                        ep_oob += 1
                    if not info["attack_active"]:
                        clean_steps += 1
                        if int(action) == 1:
                            clean_distrust += 1
                    done = terminated or truncated
                oob_all.append(ep_oob)
                reward_all.append(ep_reward)
                dr = (clean_distrust / clean_steps) if clean_steps > 0 else 0.0
                distrust_all.append(dr)
        env.close()

        mean_oob = float(np.mean(oob_all))
        mean_dr = float(np.mean(distrust_all))
        mean_ret = float(np.mean(reward_all))
        score = mean_oob + self.distrust_weight * mean_dr

        if self.verbose >= 1:
            print(f"[OobCkpt @ {self.num_timesteps}] "
                  f"score={score:.3f} (OOB={mean_oob:.2f}, DR={mean_dr*100:.1f}%), "
                  f"return={mean_ret:.1f}")

        improved = (score < self.best_score) or (
            abs(score - self.best_score) < 1e-6 and mean_ret > self.best_return
        )
        if improved:
            self.best_score = score
            self.best_return = mean_ret
            ckpt_path = os.path.join(self.save_path, "best_model")
            self.model.save(ckpt_path)
            if self.verbose >= 1:
                print(f"  -> New best! Saved to {ckpt_path}.zip")
        return True


def parse_args():
    parser = argparse.ArgumentParser(description="Train RL Agent for Power Grid Cyber Defense")
    parser.add_argument("--algo", type=str, choices=["dqn", "ppo"], default="dqn",
                        help="RL algorithm: 'dqn' (default) or 'ppo'")
    parser.add_argument("--reward-variant", type=str, choices=list(REWARD_VARIANTS.keys()), default="R1_baseline",
                        help="Named reward variant: R1_baseline, R2_costly, R3_degraded, R4_switch")
    parser.add_argument("--timesteps", type=int, default=200000,
                        help="Total training timesteps (default: 200000)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for environment and model (default: 0)")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Run identifier (default format: <algo>_<variant>_s<seed>)")
    parser.add_argument("--run-suffix", type=str, default="",
                        help="Optional suffix appended to run name (e.g. '_v2')")
    parser.add_argument("--estimator", type=str, choices=["placeholder", "kalman"], default="kalman",
                        help="Estimator type: 'kalman' (default) or 'placeholder'")
    parser.add_argument("--obs-extended", dest="obs_extended", action="store_true", default=True,
                        help="Enable extended 8-dimensional observation vector (default: True)")
    parser.add_argument("--no-obs-extended", dest="obs_extended", action="store_false",
                        help="Use legacy 5-dimensional observation vector")
    parser.add_argument("--w-band", type=float, default=None,
                        help="Override penalty weight for frequency out-of-band")
    parser.add_argument("--w-distrust", type=float, default=None,
                        help="Override penalty weight for needless distrust")
    parser.add_argument("--eval-freq", type=int, default=CKPT_EVAL_FREQ,
                        help=f"Checkpoint evaluation frequency in timesteps (default: {CKPT_EVAL_FREQ})")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load reward configuration from chosen named variant
    reward_cfg = dict(REWARD_VARIANTS[args.reward_variant])
    if args.w_band is not None:
        reward_cfg["w_band"] = float(args.w_band)
    if args.w_distrust is not None:
        reward_cfg["w_distrust"] = float(args.w_distrust)

    # Format run name: <algo>_<variant>_s<seed><suffix>
    if args.run_name is None:
        run_name = f"{args.algo}_{args.reward_variant}_s{args.seed}{args.run_suffix}"
    else:
        run_name = args.run_name

    print("======================================================================")
    print(f"Starting {args.algo.upper()} Defender Training: Run '{run_name}'")
    print(f"  - Algorithm           : {args.algo.upper()}")
    print(f"  - Reward Variant      : {args.reward_variant}")
    print(f"  - Estimator           : {args.estimator.upper()}")
    print(f"  - Obs Extended        : {args.obs_extended} (dim: {8 if args.obs_extended else 5})")
    print(f"  - Total Timesteps     : {args.timesteps}")
    print(f"  - Random Seed         : {args.seed}")
    print(f"  - Reward Weights      : {reward_cfg}")
    print("======================================================================\n")

    # Setup directories
    models_dir = os.path.join("models", run_name)
    best_model_dir = os.path.join(models_dir, "best_model")
    logs_dir = os.path.join("logs", run_name)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Factory for creating environment with the chosen estimator and options
    def make_env():
        if args.estimator == "kalman":
            est = KalmanEstimator()
        else:
            est = PlaceholderEstimator()
        return GridDefenseEnv(
            estimator=est,
            reward_cfg=reward_cfg,
            obs_extended=args.obs_extended,
        )

    # 1. Instantiate training environment with component logging
    info_keys = ("r_band", "r_distrust", "r_distrust_attack", "r_switch", "out_of_band")
    train_env = Monitor(
        make_env(),
        filename=os.path.join(logs_dir, "train_monitor.csv"),
        info_keywords=info_keys,
    )

    # Re-seed training environment
    train_env.reset(seed=args.seed)

    # 2. Setup OOB-based checkpoint callback (Phase 5 STEP 2)
    effective_eval_freq = min(args.eval_freq, max(1000, args.timesteps // 20))
    oob_callback = OobCheckpointCallback(
        make_env_fn=make_env,
        save_path=best_model_dir,
        eval_freq=effective_eval_freq,
        episodes_per_type=CKPT_EVAL_EPISODES,
        seeds=SEED_SETS["CKPT_EVAL"],
        distrust_weight=CKPT_DISTRUST_WEIGHT,
        verbose=1,
    )

    # 3. Instantiate RL Model
    if args.algo == "ppo":
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            seed=args.seed,
            tensorboard_log=logs_dir,
            verbose=1,
        )
    else:
        model = DQN(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=1e-3,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=64,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            exploration_fraction=0.15,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            seed=args.seed,
            tensorboard_log=logs_dir,
            verbose=1,
        )

    # 4. Train Model
    print("Training started...")
    print(f"Logging to {logs_dir}")
    model.learn(
        total_timesteps=args.timesteps,
        callback=oob_callback,
        progress_bar=False,
    )

    # 5. Save Final Model
    final_model_path = os.path.join(models_dir, "final_model.zip")
    model.save(final_model_path)
    print("\n======================================================================")
    print(f"Final model saved cleanly to: {final_model_path}")
    best_path = os.path.join(best_model_dir, 'best_model.zip')
    if os.path.exists(best_path):
        print(f"Best model (OOB-score) saved to: {best_path}")
    else:
        print(f"Best model checkpoint: none saved yet (score never improved past inf)")
    print(f"TensorBoard logs saved to: {logs_dir}")
    print("======================================================================\n")


if __name__ == "__main__":
    main()
