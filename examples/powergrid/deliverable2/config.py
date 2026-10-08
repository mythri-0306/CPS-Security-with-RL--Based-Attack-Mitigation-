"""
Deliverable 2 Configuration - config.py

Centralized configuration object holding all hyperparameters for:
1. Grid Physical Constants
2. Kalman Filter State Estimator (Q, R, P0)
3. Multi-Signal Attack Detector (Thresholds, Weights, Window sizes)
4. Defense Baselines
5. Paths and Logging
"""

import os
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class GridConfig:
    """Physical power grid parameters matching Deliverable 1."""
    f0: float = 50.00                 # Nominal grid frequency (Hz)
    H: float = 5.0                    # System inertia constant (seconds)
    S_base: float = 100.00            # System base rating (MVA / MW)
    D: float = 1.0                    # Load-frequency damping coefficient (p.u./Hz)
    nominal_load: float = 100.00      # Nominal electrical load (MW)
    initial_p_gen: float = 100.00     # Initial generator setpoint (MW)
    dt_physics: float = 0.1           # Physics integration period (s)
    dt_controller: float = 0.2        # Controller polling period (s)
    safe_band: Tuple[float, float] = (49.80, 50.20)  # Safe operating frequency band (Hz)
    scale_factor: float = 100.0       # Modbus integer scaling factor (Hz * scale)


@dataclass
class KalmanConfig:
    """Kalman Filter State Estimator hyperparameters."""
    process_noise_cov_Q: float = 1e-4      # Q: Process noise covariance (load fluctuations)
    measurement_noise_cov_R: float = 1e-3  # R: Sensor measurement noise covariance
    initial_error_cov_P0: float = 1e-2     # Initial state estimation error covariance
    initial_state_f: float = 50.00         # Initial estimated frequency (Hz)


@dataclass
class DetectorConfig:
    """Multi-Signal Attack Detector hyperparameters."""
    # Signal Weights (sum to 1.0)
    w_residual: float = 0.40          # Weight for Kalman innovation residual score
    w_sudden: float = 0.25            # Weight for sudden change / RoCoF score
    w_timing: float = 0.20            # Weight for inter-arrival timing anomaly score
    w_replay: float = 0.15            # Weight for duplicate / replay pattern score

    # Combined Detection Decision Threshold
    attack_score_threshold: float = 0.50  # Anomaly score >= threshold -> Attack declared

    # Signal 1: Kalman Residual Thresholds
    residual_threshold_hz: float = 0.12    # Absolute residual magnitude threshold (Hz)
    residual_chi2_threshold: float = 9.0   # Normalized residual squared threshold (chi-sq)

    # Signal 2: Sudden Change (RoCoF) Thresholds
    max_rate_of_change_hz_per_sec: float = 0.60  # Max physical df/dt (Hz/s)
    max_step_change_hz: float = 0.15             # Max jump per sample (Hz)

    # Signal 3: Timing / Network Delay Thresholds
    expected_period_sec: float = 0.20            # Expected sensor packet polling interval
    timing_jitter_tolerance_sec: float = 0.10    # Acceptable arrival jitter window (s)
    timeout_threshold_sec: float = 0.60          # Blackout threshold (s)

    # Signal 4: Replay / Duplicate Detection
    replay_window_size: int = 50                 # Number of historical packets to keep
    max_consecutive_identical: int = 15          # Max identical readings allowed before flagging


@dataclass
class DefenseConfig:
    """Defense Baseline hyperparameters."""
    rule_safe_min: float = 49.80                 # Rule-based lower bound (Hz)
    rule_safe_max: float = 50.20                 # Rule-based upper bound (Hz)
    rule_max_jump_hz: float = 0.20               # Rule-based max single-step jump (Hz)
    kalman_only_threshold_hz: float = 0.15       # Kalman-only residual threshold (Hz)


@dataclass
class Deliverable2Config:
    """Master Deliverable 2 Configuration Object."""
    grid: GridConfig = field(default_factory=GridConfig)
    kalman: KalmanConfig = field(default_factory=KalmanConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    defense: DefenseConfig = field(default_factory=DefenseConfig)
    results_dir: str = field(
        default_factory=lambda: os.path.abspath(
            os.path.join(os.path.dirname(__file__), "results")
        )
    )
    random_seed: int = 42

    def __post_init__(self):
        os.makedirs(self.results_dir, exist_ok=True)


# Default global instance for easy import
DEFAULT_CONFIG = Deliverable2Config()
