"""
Deliverable 2: Multi-Signal Attack Detector - detector.py

Implements a 4-channel multi-signal cyber-physical anomaly detector:
1. Kalman Innovation Residual Anomaly (Physics Law Violation)
2. Sudden Change / Rate of Change of Frequency (RoCoF) Anomaly
3. Network Timing & Inter-Arrival Jitter Anomaly
4. Duplicate / Stale-Value Replay Pattern Anomaly

CPS Evidence Fusion:
    Combines weighted multi-sensor correlation with single-channel critical
    overrides to effectively detect both multi-modal attacks (e.g. FDI spoofing)
    and single-domain attacks (e.g. link delay, DoS blackout, replay).
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Deque
import numpy as np

from .config import Deliverable2Config, DEFAULT_CONFIG
from .estimator import EstimationResult


@dataclass
class DetectionResult:
    """Detailed record of multi-signal anomaly detector outputs at step k."""
    time_sec: float
    is_attack: bool                  # Boolean decision: True = Attack, False = Normal
    combined_score: float            # Combined attack score S_attack in [0.0, 1.0]
    score_residual: float            # Signal 1 sub-score in [0.0, 1.0]
    score_sudden: float              # Signal 2 sub-score in [0.0, 1.0]
    score_timing: float              # Signal 3 sub-score in [0.0, 1.0]
    score_replay: float              # Signal 4 sub-score in [0.0, 1.0]
    raw_residual_hz: float           # Raw Kalman residual r(k) in Hz
    raw_rocof_hz_per_sec: float      # Raw Rate of Change of Frequency (Hz/s)
    raw_inter_arrival_sec: float     # Inter-arrival delay between packets (s)
    consecutive_identical_count: int # Number of identical readings in a row


class ResidualAnomalyDetector:
    """Signal 1: Evaluates Kalman innovation residual against physics thresholds."""

    def __init__(self, config: Deliverable2Config):
        self.cfg = config.detector
        # Nominal threshold accounts for normal load disturbance transients (~0.08 Hz)
        self.thresh_hz = 0.25
        self.noise_floor = 0.08

    def evaluate(self, kf_res: EstimationResult) -> float:
        """Compute anomaly sub-score based on residual magnitude."""
        abs_res = abs(kf_res.residual)
        if abs_res <= self.noise_floor:
            return 0.0
        score = (abs_res - self.noise_floor) / (self.thresh_hz - self.noise_floor)
        return float(np.clip(score, 0.0, 1.0))


class SuddenChangeDetector:
    """Signal 2: Evaluates Rate of Change of Frequency (RoCoF) and step jumps."""

    def __init__(self, config: Deliverable2Config):
        self.cfg = config.detector
        self.max_rocof = 2.0         # Hz/s (physical inertia limits df/dt)
        self.max_step = 0.20          # Max single-step jump (Hz)
        self.noise_step_floor = 0.06  # Normal transient step ceiling
        self.last_measurement: Optional[float] = None
        self.last_time: Optional[float] = None

    def reset(self):
        self.last_measurement = None
        self.last_time = None

    def evaluate(self, measurement: float, time_sec: float) -> tuple[float, float]:
        """Compute sudden-change sub-score. Returns (sub_score, raw_rocof)."""
        if self.last_measurement is None:
            self.last_measurement = measurement
            self.last_time = time_sec
            return 0.0, 0.0

        dt = max(1e-4, time_sec - self.last_time)
        step_diff = abs(measurement - self.last_measurement)
        rocof = step_diff / dt

        self.last_measurement = measurement
        self.last_time = time_sec

        if step_diff <= self.noise_step_floor:
            score = 0.0
        else:
            score_step = (step_diff - self.noise_step_floor) / (self.max_step - self.noise_step_floor)
            score_rocof = rocof / self.max_rocof
            score = max(score_step, score_rocof)

        return float(np.clip(score, 0.0, 1.0)), float(rocof)


class TimingAnomalyDetector:
    """Signal 3: Evaluates packet arrival intervals, jitter, and network blackouts."""

    def __init__(self, config: Deliverable2Config):
        self.cfg = config.detector
        self.expected_period = self.cfg.expected_period_sec   # 0.20s
        self.jitter_tol = 0.15                                # 0.15s tolerance
        self.timeout_thresh = 0.50                            # 0.50s blackout threshold
        self.last_arrival_time: Optional[float] = None

    def reset(self):
        self.last_arrival_time = None

    def evaluate(self, arrival_time: float, is_timeout_flag: bool = False) -> tuple[float, float]:
        """Compute timing anomaly sub-score. Returns (sub_score, raw_inter_arrival)."""
        if is_timeout_flag:
            return 1.0, self.timeout_thresh

        if self.last_arrival_time is None:
            self.last_arrival_time = arrival_time
            return 0.0, self.expected_period

        inter_arrival = max(0.0, arrival_time - self.last_arrival_time)
        self.last_arrival_time = arrival_time

        delay_excess = inter_arrival - self.expected_period
        if delay_excess <= self.jitter_tol:
            score = 0.0
        else:
            score = (delay_excess - self.jitter_tol) / (self.timeout_thresh - self.expected_period - self.jitter_tol)

        return float(np.clip(score, 0.0, 1.0)), float(inter_arrival)


class ReplayDuplicateDetector:
    """Signal 4: Identifies historical replay sequences and frozen duplicate values."""

    def __init__(self, config: Deliverable2Config):
        self.cfg = config.detector
        self.history: Deque[float] = deque(maxlen=1000)
        self.consecutive_identical = 0
        self.last_val: Optional[float] = None
        self.pattern_length = 8  # 8 samples = 0.8s sequence fingerprint

    def reset(self):
        self.history.clear()
        self.consecutive_identical = 0
        self.last_val = None

    def evaluate(self, measurement: float) -> tuple[float, int]:
        """Compute replay/duplicate sub-score using sequence fingerprint matching.
        
        Returns (sub_score, consecutive_count).
        """
        # 1. Track identical constant readings
        if self.last_val is not None and abs(measurement - self.last_val) < 1e-4:
            self.consecutive_identical += 1
        else:
            self.consecutive_identical = 1
        self.last_val = measurement

        score_frozen = 0.0
        if self.consecutive_identical >= 4:
            score_frozen = min(1.0, (self.consecutive_identical - 3) / 5.0)

        # 2. Historical Sequence Fingerprint Matching (Replay Detection)
        score_replay = 0.0
        L = self.pattern_length
        if len(self.history) >= (L + 30):
            hist_list = list(self.history)
            curr_window = np.array(hist_list[-(L-1):] + [measurement])

            # Search in past history (excluding the immediate recent past of 20 samples)
            search_bound = len(hist_list) - 20
            best_match_diff = float("inf")

            for i in range(0, search_bound - L + 1):
                past_window = np.array(hist_list[i : i + L])
                # Max absolute error across the window
                max_diff = float(np.max(np.abs(curr_window - past_window)))
                if max_diff < best_match_diff:
                    best_match_diff = max_diff

            # If an exact or near-exact match is found in past history
            if best_match_diff <= 0.005:  # Exact match down to sensor quantization
                score_replay = 1.0
            elif best_match_diff <= 0.015:
                score_replay = (0.015 - best_match_diff) / 0.010

        self.history.append(measurement)
        total_score = max(score_frozen, score_replay)
        return float(np.clip(total_score, 0.0, 1.0)), self.consecutive_identical


class MultiSignalDetector:
    """Master Multi-Signal Cyber-Physical Attack Detector."""

    def __init__(self, config: Optional[Deliverable2Config] = None):
        if config is None:
            config = DEFAULT_CONFIG
        self.cfg = config

        self.res_detector = ResidualAnomalyDetector(config=self.cfg)
        self.sudden_detector = SuddenChangeDetector(config=self.cfg)
        self.timing_detector = TimingAnomalyDetector(config=self.cfg)
        self.replay_detector = ReplayDuplicateDetector(config=self.cfg)

        self.w_res = float(self.cfg.detector.w_residual)
        self.w_sud = float(self.cfg.detector.w_sudden)
        self.w_tim = float(self.cfg.detector.w_timing)
        self.w_rep = float(self.cfg.detector.w_replay)
        self.threshold = float(self.cfg.detector.attack_score_threshold)

        self.history: List[DetectionResult] = []

    def reset(self):
        """Reset all sub-detectors and history."""
        self.sudden_detector.reset()
        self.timing_detector.reset()
        self.replay_detector.reset()
        self.history.clear()

    def evaluate(self, kf_res: EstimationResult, time_sec: float,
                 is_timeout_flag: bool = False) -> DetectionResult:
        """Evaluate all 4 signals and compute combined attack score S_attack."""
        y_k = kf_res.measurement_y

        # Signal 1: Kalman Residual
        s_res = self.res_detector.evaluate(kf_res)

        # Signal 2: Sudden Change (RoCoF)
        s_sud, raw_rocof = self.sudden_detector.evaluate(y_k, time_sec)

        # Signal 3: Timing / Latency
        s_tim, raw_arrival = self.timing_detector.evaluate(time_sec, is_timeout_flag=is_timeout_flag)

        # Signal 4: Replay / Duplicate
        s_rep, ident_count = self.replay_detector.evaluate(y_k)

        # 1. Weighted sum across channels
        weighted_score = (
            (self.w_res * s_res) +
            (self.w_sud * s_sud) +
            (self.w_tim * s_tim) +
            (self.w_rep * s_rep)
        )

        # 2. Critical single-channel overrides (for pure timing, pure replay, or massive residual attacks)
        dominant_override = max(
            0.90 * s_res,
            0.90 * s_sud,
            0.90 * s_tim,
            0.90 * s_rep
        )

        # Master Combined Score
        combined_score = max(weighted_score, dominant_override)
        combined_score = float(np.clip(combined_score, 0.0, 1.0))

        # Binary Decision Thresholding (tau = 0.50)
        is_attack = bool(combined_score >= self.threshold)

        result = DetectionResult(
            time_sec=time_sec,
            is_attack=is_attack,
            combined_score=combined_score,
            score_residual=s_res,
            score_sudden=s_sud,
            score_timing=s_tim,
            score_replay=s_rep,
            raw_residual_hz=kf_res.residual,
            raw_rocof_hz_per_sec=raw_rocof,
            raw_inter_arrival_sec=raw_arrival,
            consecutive_identical_count=ident_count
        )

        self.history.append(result)
        return result
