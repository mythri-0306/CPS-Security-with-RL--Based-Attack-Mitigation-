"""
Demo simulation engine – wraps the *real* GridDefenseEnv and every defense
policy so the demo app never duplicates physics / model code.

Every defence is instantiated from the project's own modules:
  - GridDefenseEnv + KalmanEstimator (grid_env.py, estimator_interface.py)
  - RuleBasedDefense, KalmanOnlyDefense (deliverable2/defenses.py)
  - DetectorThresholdTimingPolicy (detector_threshold_policy.py)
  - SB3 DQN models (models/dqn_R2_costly_s*_v2/)
"""

import os
import sys
import numpy as np

# ---------------------------------------------------------------------------
# Path setup – ensure project root and examples/powergrid are importable
# ---------------------------------------------------------------------------
_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_DEMO_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
_PG_DIR = os.path.join(_ROOT_DIR, "examples", "powergrid")
if os.path.exists(_PG_DIR) and _PG_DIR not in sys.path:
    sys.path.append(_PG_DIR)

from grid_env import GridDefenseEnv
from estimator_interface import KalmanEstimator
from detector_threshold_policy import DetectorThresholdTimingPolicy
from config import (
    NOMINAL_FREQ, SAFE_BAND_LOW, SAFE_BAND_HIGH, DT, EPISODE_LEN,
    ATTACK_INTENSITY_RANGES, DETECTOR_THRESHOLDS_TIMING, REWARD_VARIANTS,
)
from deliverable2.defenses import RuleBasedDefense, KalmanOnlyDefense

# ---------------------------------------------------------------------------
# UI ↔ env mappings
# ---------------------------------------------------------------------------
ATTACK_UI_TO_ENV = {
    "none": "none",
    "spoof": "spoof",
    "stealthy_drift": "stealthy_drift",
    "replay": "replay",
    "telemetry_freeze": "delay",          # delay & dos share the same stale-hold path
}

INTENSITY_RANGES = {                       # training-range sliders for the UI
    "none":              (0.0, 0.0),
    "spoof":             (48.00, 49.20),   # Hz
    "stealthy_drift":    (0.02, 0.08),     # Hz/s
    "replay":            (5.0, 15.0),      # seconds
    "telemetry_freeze":  (400.0, 1200.0),  # ms (not actually used by step())
}

DEFENSE_METHODS = [
    "No Defense", "Rule-Based", "Kalman-Only",
    "Detector-Timing", "RL DQN", "Oracle",
]


# ---------------------------------------------------------------------------
# DQN model helpers
# ---------------------------------------------------------------------------
def get_dqn_model_path(seed: int = 0) -> str | None:
    """Return the best-checkpoint path for a DQN R2_costly model seed."""
    for suffix in [f"_v2", ""]:
        p = os.path.join(
            _ROOT_DIR, "models",
            f"dqn_R2_costly_s{seed}{suffix}", "best_model", "best_model.zip",
        )
        if os.path.exists(p):
            return p
    return None


import functools

@functools.lru_cache(maxsize=8)
def load_dqn_model(seed: int = 0):
    """Load and return an SB3 DQN model (lazy import to avoid hard dep)."""
    from stable_baselines3 import DQN
    path = get_dqn_model_path(seed)
    if path is None:
        raise FileNotFoundError(f"DQN model not found for seed {seed}")
    return DQN.load(path)


# ---------------------------------------------------------------------------
# Defence adapters  (thin wrappers – all logic lives in the real modules)
# ---------------------------------------------------------------------------
class _BaseAdapter:
    def reset(self, seed=None):
        pass

    def get_action(self, obs: np.ndarray, info: dict) -> int:
        raise NotImplementedError


class NoDefenseAdapter(_BaseAdapter):
    """Always trust the sensor measurement."""
    def get_action(self, obs, info):
        return 0


class RuleBasedAdapter(_BaseAdapter):
    def __init__(self):
        self.defense = RuleBasedDefense()

    def reset(self, seed=None):
        self.defense.reset()

    def get_action(self, obs, info):
        res = self.defense.process(
            measurement_y=info["measured_freq"],
            p_gen=info.get("p_gen", 100.0),
            time_sec=info.get("step", 0) * DT,
            dt=DT,
        )
        return 1 if res.is_attack_flagged else 0


class KalmanOnlyAdapter(_BaseAdapter):
    def __init__(self):
        self.defense = KalmanOnlyDefense()

    def reset(self, seed=None):
        self.defense.reset()

    def get_action(self, obs, info):
        res = self.defense.process(
            measurement_y=info["measured_freq"],
            p_gen=info.get("p_gen", 100.0),
            time_sec=info.get("step", 0) * DT,
            dt=DT,
        )
        return 1 if res.is_attack_flagged else 0


class DetectorTimingAdapter(_BaseAdapter):
    def __init__(self):
        self.policy = DetectorThresholdTimingPolicy(
            t_res=DETECTOR_THRESHOLDS_TIMING["t_res"],
            t_age=DETECTOR_THRESHOLDS_TIMING["t_age"],
            hold_steps=int(DETECTOR_THRESHOLDS_TIMING["hold_steps"]),
        )

    def reset(self, seed=None):
        self.policy.reset(seed=seed)

    def get_action(self, obs, info):
        return self.policy.get_action(obs, info)


class RLDQNAdapter(_BaseAdapter):
    def __init__(self, seed: int = 0):
        self.model = load_dqn_model(seed)

    def get_action(self, obs, info):
        act, _ = self.model.predict(obs, deterministic=True)
        return int(act)


class OracleAdapter(_BaseAdapter):
    """Causal oracle – distrusts one step after attack begins."""
    def __init__(self):
        self._prev_attack = False

    def reset(self, seed=None):
        self._prev_attack = False

    def get_action(self, obs, info):
        action = 1 if self._prev_attack else 0
        self._prev_attack = info["attack_active"]
        return action


def make_adapter(method: str, dqn_seed: int = 0) -> _BaseAdapter:
    """Factory: return a defence adapter by UI name."""
    if method == "No Defense":
        return NoDefenseAdapter()
    if method == "Rule-Based":
        return RuleBasedAdapter()
    if method == "Kalman-Only":
        return KalmanOnlyAdapter()
    if method == "Detector-Timing":
        return DetectorTimingAdapter()
    if method == "RL DQN":
        return RLDQNAdapter(seed=dqn_seed)
    if method == "Oracle":
        return OracleAdapter()
    raise ValueError(f"Unknown defense: {method}")


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------
def run_episode(
    attack_type: str,
    defense_method: str,
    env_seed: int = 42,
    intensity: float | None = None,
    attack_start: float = 20.0,
    dqn_seed: int = 0,
) -> dict:
    """Run one full episode and return trajectory + metrics.

    Parameters
    ----------
    attack_type : str   UI name (none / spoof / stealthy_drift / replay / telemetry_freeze)
    defense_method : str   one of DEFENSE_METHODS
    env_seed : int   environment RNG seed
    intensity : float or None   override intensity (None → env default)
    attack_start : float   attack onset in seconds
    dqn_seed : int   which DQN model seed (0-4)

    Returns
    -------
    dict   trajectory arrays + scalar metrics
    """
    env_attack = ATTACK_UI_TO_ENV.get(attack_type, attack_type)

    # Build env
    env = GridDefenseEnv(estimator=KalmanEstimator(), obs_extended=True)
    options: dict = {"attack_type": env_attack, "attack_start_time": attack_start}
    if intensity is not None:
        options["intensity"] = intensity

    # Build defence
    adapter = make_adapter(defense_method, dqn_seed=dqn_seed)
    adapter.reset(seed=env_seed)

    obs, info = env.reset(seed=env_seed, options=options)

    # Storage
    t, f_true, f_meas, f_pred = [0.0], [info["true_freq"]], [info["measured_freq"]], [info["kalman_pred"]]
    actions, att_active, oob_flags = [0], [info["attack_active"]], [info["out_of_band"]]

    total_reward = 0.0
    oob_steps = 0
    max_dev = abs(info["true_freq"] - NOMINAL_FREQ)
    clean_steps = 0
    clean_distrust = 0
    attack_onset_step = None
    first_distrust_in_attack = None

    done = False
    step_i = 0
    while not done:
        action = adapter.get_action(obs, info)

        if info["attack_active"] and attack_onset_step is None:
            attack_onset_step = step_i
        if info["attack_active"] and action == 1 and first_distrust_in_attack is None:
            first_distrust_in_attack = step_i
        if not info["attack_active"]:
            clean_steps += 1
            if action == 1:
                clean_distrust += 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        step_i += 1

        t.append(step_i * DT)
        f_true.append(info["true_freq"])
        f_meas.append(info["measured_freq"])
        f_pred.append(info["kalman_pred"])
        actions.append(action)
        att_active.append(info["attack_active"])
        oob_flags.append(info["out_of_band"])

        dev = abs(info["true_freq"] - NOMINAL_FREQ)
        if dev > max_dev:
            max_dev = dev
        if info["out_of_band"]:
            oob_steps += 1

        done = terminated or truncated

    env.close()

    detection_delay = None
    if attack_onset_step is not None and first_distrust_in_attack is not None:
        detection_delay = first_distrust_in_attack - attack_onset_step

    return {
        # Arrays
        "t": np.array(t),
        "f_true": np.array(f_true),
        "f_meas": np.array(f_meas),
        "f_pred": np.array(f_pred),
        "action": np.array(actions),
        "attack_active": np.array(att_active),
        "oob": np.array(oob_flags),
        # Scalars
        "time_oob": oob_steps * DT,
        "max_deviation": max_dev,
        "detection_delay": detection_delay,
        "clean_distrust_rate": (clean_distrust / clean_steps * 100) if clean_steps > 0 else 0.0,
        "episode_return": total_reward,
        "oob_steps": oob_steps,
        "total_steps": step_i,
        # Params
        "attack_type": attack_type,
        "env_attack_type": env_attack,
        "defense_method": defense_method,
        "intensity": env.attack_intensity,
        "attack_start": env.attack_start_time,
        "seed": env_seed,
    }
