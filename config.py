"""
Phase 1-5 RL Defender Configuration - config.py

Centralized configuration holding all shared constants, attack parameters,
normalization bounds, metric definitions, reward variants, checkpoint-selection
weights, and disjoint seed sets for the Gym power-grid defense environment.
"""

from typing import List, Dict, Tuple

# -----------------------------------------------------------------------------
# Grid Physics Constants
# -----------------------------------------------------------------------------
NOMINAL_FREQ: float = 50.00          # Nominal grid frequency in Hz (f0)
SAFE_BAND_LOW: float = 49.80         # Lower boundary of safe operating frequency band in Hz
SAFE_BAND_HIGH: float = 50.20        # Upper boundary of safe operating frequency band in Hz

# Simulation Timing Constants
DT: float = 0.1                      # Simulation timestep dt in seconds (100 ms forward Euler integration)
EPISODE_LEN: int = 600               # Fixed episode length in steps (600 steps * 0.1s = 60.0s episode duration)

# -----------------------------------------------------------------------------
# Reward Weights
# -----------------------------------------------------------------------------
W_BAND: float = 10.0          # Penalty weight per step for true frequency outside safe band
W_DISTRUST: float = 0.1       # Penalty weight per step for needless distrust (action=1 when clean)
W_DISTRUST_ATTACK: float = 0.0 # Penalty weight per step for distrusting during active attack (degraded control)
W_SWITCH: float = 0.0          # Penalty weight per step for switching action between steps (anti-flapping)

# Phase 4 Named Reward Variants for GOAL A
REWARD_VARIANTS: Dict[str, Dict[str, float]] = {
    "R1_baseline": {
        "w_band": 10.0,
        "w_distrust": 0.1,
        "w_distrust_attack": 0.0,
        "w_switch": 0.0,
    },
    "R2_costly": {
        "w_band": 10.0,
        "w_distrust": 1.0,
        "w_distrust_attack": 0.0,
        "w_switch": 0.0,
    },
    "R3_degraded": {
        "w_band": 10.0,
        "w_distrust": 1.0,
        "w_distrust_attack": 0.3,
        "w_switch": 0.0,
    },
    "R4_switch": {
        "w_band": 10.0,
        "w_distrust": 1.0,
        "w_distrust_attack": 0.3,
        "w_switch": 0.5,
    },
}

# Physical Plant Parameters (matching physics.py and utils.py)
INERTIA_H: float = 5.0               # Generator inertia constant H in seconds
BASE_POWER_MVA: float = 100.0        # System power base rating in MVA / MW
DAMPING_D: float = 1.0               # Damping factor D in p.u./Hz
NOMINAL_LOAD: float = 100.0          # Base electrical load demand in MW
INITIAL_GEN_SETPOINT: float = 100.0  # Initial generation dispatch in MW

# Primary Controller Parameters (Droop + Secondary PI matching controller_device.py)
CONTROLLER_KP: float = 15.0          # Controller proportional gain (MW/Hz)
CONTROLLER_KI: float = 2.5           # Controller integral gain (MW/(Hz*s))
CONTROLLER_DEADBAND_HZ: float = 0.005 # Insensitive deadband around nominal frequency (Hz)
CONTROLLER_INTEGRAL_MAX: float = 30.0 # Anti-windup limit for integral error accumulator
GEN_MIN_MW: float = 10.0             # Physical minimum generator capacity clamp (MW)
GEN_MAX_MW: float = 300.0            # Physical maximum generator capacity clamp (MW)

# -----------------------------------------------------------------------------
# Sensor & Noise Model
# -----------------------------------------------------------------------------
SENSOR_NOISE_SIGMA: float = 0.03162  # Gaussian sensor measurement noise std dev (sqrt(1e-3))
SCALE_FACTOR: float = 100.0          # Modbus integer scaling factor (0.01 Hz precision)
PLACEHOLDER_NOISE_SIGMA: float = 0.02 # Placeholder estimator Gaussian noise sigma in Hz

# -----------------------------------------------------------------------------
# Attack Types & Parameter Ranges
# -----------------------------------------------------------------------------
ATTACK_TYPES: List[str] = [
    "none",           # Normal operation (no attack injected)
    "spoof",          # Abrupt constant frequency spoofing (e.g. 48.50 Hz)
    "stealthy_drift", # Gradual frequency ramp down (stub: 50.0 -> 49.9 -> 49.8...)
    "replay",         # Replay historical nominal telemetry
    "delay",          # Packet delay / hold stale reading
    "dos",            # Packet loss / blackout
]

# Attack Timing Ranges
ATTACK_START_TIME_RANGE: Tuple[float, float] = (10.0, 30.0)  # Attack onset start time window in seconds [min, max]
ATTACK_DEFAULT_DURATION: float = 20.0                        # Default attack duration in seconds
STEALTHY_DRIFT_TARGET_FREQ: float = 48.50                    # Target lower saturation frequency for stealthy drift (Hz)

# Per-Type Attack Intensity Ranges
# - spoof: target false frequency (Hz)
# - stealthy_drift: drift rate in Hz/s (stub: 50.0 -> 49.9 -> 49.8...)
# - replay: buffer history offset in seconds
# - delay: network latency in ms
# - dos: packet loss probability [0.0, 1.0]
# - none: inactive (0.0, 0.0)
ATTACK_INTENSITY_RANGES: Dict[str, Tuple[float, float]] = {
    "none": (0.0, 0.0),
    "spoof": (48.00, 49.20),
    "stealthy_drift": (0.02, 0.08),  # Drift rate: 50.0 -> 49.9 -> 49.8 Hz at 0.02-0.08 Hz/s
    "replay": (5.0, 15.0),
    "delay": (400.0, 1200.0),
    "dos": (0.70, 0.95),
}

# -----------------------------------------------------------------------------
# Phase 5 STEP 2 & 3: Fair Baseline Detector-Threshold Hyperparameters
# Tuned strictly on SEED_SETS["CKPT_EVAL"] minimizing mean_OOB + 20 * clean_distrust
# -----------------------------------------------------------------------------
DETECTOR_THRESHOLDS_FULL: Dict[str, float] = {
    "t_res": 2.0,   # |residual_norm| > 2.0 (innovation residual > 2.0 * 0.04122 = 0.0824 Hz)
    "t_age": 0.1,   # packet_age_norm > 0.1 (packet age > 2 steps)
    "t_var": 2.0,   # residual_var_norm > 2.0 (rolling residual var > 0.010)
    "hold_steps": 0,
    "T_RES": 2.0,
    "T_AGE": 0.1,
    "T_VAR": 2.0,
}
# Stale-telemetry + residual only (no variance) - tuned on CKPT_EVAL (Score: 0.211, Clean DR: 1.06%)
DETECTOR_THRESHOLDS_TIMING: Dict[str, float] = {
    "t_res": 4.0,   # |residual_norm| > 4.0 (innovation residual > 0.165 Hz)
    "t_age": 0.05,  # packet_age_norm > 0.05 (packet age > 1 step)
    "hold_steps": 0,
    "T_RES": 4.0,
    "T_AGE": 0.05,
}
# Stale-telemetry + residual with anti-flapping latch hold - tuned on CKPT_EVAL (Score: 0.951, Clean DR: 4.76%)
DETECTOR_THRESHOLDS_TIMING_HOLD: Dict[str, float] = {
    "t_res": 4.0,   # |residual_norm| > 4.0 (innovation residual > 0.165 Hz)
    "t_age": 0.05,  # packet_age_norm > 0.05 (packet age > 1 step)
    "hold_steps": 3,
    "T_RES": 4.0,
    "T_AGE": 0.05,
}
# Default aliases for backward compatibility
DETECTOR_THRESHOLDS = DETECTOR_THRESHOLDS_FULL
DETECTOR_HOLD_STEPS: int = 0

# -----------------------------------------------------------------------------
# Phase 5 STEP 4: Widened Spoof Range & Adjusted Held-Out Range
# -----------------------------------------------------------------------------
WIDE_SPOOF_RANGE: Tuple[float, float] = (46.50, 49.20)              # Widened spoof training range (Hz)
HELDOUT_SPOOF_ADJUSTED_RANGE: Tuple[float, float] = (45.50, 46.40)  # Disjoint from widened training range (Hz)

# -----------------------------------------------------------------------------
# Phase 5 STEP 3: Held-Out Attack Parameters (Generalization & Robustness Test)
# Start time, duration, and intensities strictly beyond training ranges
# -----------------------------------------------------------------------------
HELDOUT_START_TIME_RANGE: Tuple[float, float] = (32.0, 42.0)  # Later onset window (s) vs training (10.0, 30.0)
HELDOUT_DEFAULT_DURATION: float = 25.0                        # Extended duration (s) vs training 20.0s
HELDOUT_ATTACK_INTENSITY_RANGES: Dict[str, Tuple[float, float]] = {
    "none": (0.0, 0.0),
    "spoof": (47.00, 47.90),         # Severe frequency depression (Hz) vs training (48.00, 49.20)
    "stealthy_drift": (0.09, 0.15),  # Faster drift rate (Hz/s) vs training (0.02, 0.08)
    "replay": (18.0, 25.0),          # Longer buffer history offset (s) vs training (5.0, 15.0)
    "delay": (1400.0, 2500.0),       # Higher latency (ms) vs training (400.0, 1200.0)
    "dos": (0.96, 0.99),             # Extreme packet loss probability vs training (0.70, 0.95)
}

# -----------------------------------------------------------------------------
# Observation Space Normalization Scales
# Observation vector layout:
# [0] measured_freq_norm       : (measured_freq - NOMINAL_FREQ) / OBS_FREQ_NORM_SCALE
# [1] kalman_pred_norm         : (kalman_pred - NOMINAL_FREQ) / OBS_FREQ_NORM_SCALE
# [2] residual_norm            : residual / OBS_RESIDUAL_NORM_SCALE
# [3] residual_roc_norm        : residual_rate_of_change / OBS_ROCOF_NORM_SCALE
# [4] timestep_frac            : current_step / EPISODE_LEN
# -----------------------------------------------------------------------------
RESIDUAL_SCALE: float = 0.04122       # Clean-episode residual standard deviation from 50 clean rollouts (Hz)
OBS_FREQ_NORM_SCALE: float = 0.5      # Scale factor for frequency deviation: (f - 50.0) / 0.5 (±0.20Hz maps to ±0.40)
OBS_RESIDUAL_NORM_SCALE: float = RESIDUAL_SCALE  # Scale factor for innovation residual
OBS_ROCOF_NORM_SCALE: float = RESIDUAL_SCALE     # Scale factor for residual RoCoF
OBS_CLIP_LIMIT: float = 10.0          # Symmetric clipping limit for normalized features [-10.0, 10.0]

# Phase 4 Extended Observation Space Settings (GOAL B)
OBS_EXTENDED: bool = True             # Enable 8-dimensional observation vector (includes timing/staleness)
PACKET_AGE_NORM_SCALE: float = 20.0   # Scale factor for packet staleness: packet_age / 20.0
RESIDUAL_VAR_WINDOW: int = 10         # Sliding window size for short-window residual variance
RESIDUAL_VAR_NORM_SCALE: float = 0.005 # Normalization scale for residual variance: var / 0.005

# -----------------------------------------------------------------------------
# Phase 5 Checkpoint Selection
# -----------------------------------------------------------------------------
# Custom callback saves a checkpoint when the following composite score is minimised:
#   score = mean_OOB_steps + CKPT_DISTRUST_WEIGHT * clean_distrust_rate
# where clean_distrust_rate is a fraction in [0, 1] and mean_OOB_steps is in steps.
# Ties are broken by higher mean return.
CKPT_DISTRUST_WEIGHT: float = 20.0    # Weight for clean distrust rate in composite checkpoint score: score = mean_OOB_steps + 20 * clean_distrust_rate_fraction
CKPT_EVAL_EPISODES: int = 5           # Episodes per attack type in checkpoint callback eval (5 per type * 6 types = 30 episodes total)
CKPT_EVAL_FREQ: int = 10000           # Checkpoint evaluation frequency in timesteps

# Disjoint seed sets for training, checkpoint evaluation, and final evaluation.
# TRAIN seeds: used by train_dqn.py make_env / SB3 model seed.
# CKPT_EVAL seeds: used by the custom OOB-based checkpoint callback at each eval (30 distinct seeds).
# FINAL_EVAL seeds: used exclusively in final_eval.py; never seen during training or checkpoint eval.
SEED_SETS = {
    "TRAIN":      list(range(0, 10)),        # seeds 0-9 (10 seeds)
    "CKPT_EVAL":  list(range(1000, 1030)),    # seeds 1000-1029 (30 distinct seeds for checkpoint eval)
    "FINAL_EVAL": list(range(20000, 22000)),  # seeds 20000-21999 (2000 strictly held-out evaluation seeds)
}


# -----------------------------------------------------------------------------
# Evaluation Metric Definitions
# -----------------------------------------------------------------------------
def time_outside_safe_band():
    """time_outside_safe_band: Cumulative time in seconds where true frequency is outside [SAFE_BAND_LOW, SAFE_BAND_HIGH]."""
    pass

def max_deviation():
    """max_deviation: Maximum absolute deviation of true frequency from nominal: max(|f_true - NOMINAL_FREQ|) in Hz."""
    pass

def attack_success_rate():
    """attack_success_rate: Binary success flag (or fraction) where attack causes true frequency excursion outside safe band for > 3.0 cumulative seconds."""
    pass

def recovery_time():
    """recovery_time: Elapsed time in seconds from attack termination or defense activation until true frequency returns and remains within the safe band."""
    pass
