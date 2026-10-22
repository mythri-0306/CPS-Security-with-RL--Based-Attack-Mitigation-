"""
Phase 1 RL Defender Configuration - config.py

Centralized configuration holding all shared constants, attack parameters,
normalization bounds, and metric definitions for the Gym power-grid defense environment.
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
