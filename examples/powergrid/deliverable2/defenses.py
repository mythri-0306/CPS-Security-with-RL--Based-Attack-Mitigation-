"""
Deliverable 2: Baseline & Proposed Cyber-Physical Defenses - defenses.py

Implements modular defense architectures for CPS attack mitigation:
1. Rule-Based Defense (Baseline A): Static threshold clamping & jump rejection.
2. Kalman-Only Defense (Baseline B): Innovation residual thresholding & Kalman state substitution.
3. Multi-Signal Defense (Proposed): Master 4-channel detector + state-feedback mitigation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any

from .config import Deliverable2Config, DEFAULT_CONFIG
from .estimator import GridKalmanFilter, EstimationResult
from .detector import MultiSignalDetector, DetectionResult


@dataclass
class DefenseAction:
    """Detailed record of a defense module's evaluation and mitigation decision."""
    time_sec: float
    raw_measurement: float               # Received sensor reading y(k) (Hz)
    controlled_frequency: float          # Mitigated frequency passed to controller law (Hz)
    is_attack_flagged: bool              # Did the defense detect an attack?
    mitigation_active: bool              # Is active mitigation (isolation/substitution) engaged?
    defense_name: str                    # Name of the active defense
    info: Dict[str, Any]                 # Diagnostic metrics


class BaseDefense(ABC):
    """Abstract Base Class for CPS Defenses."""

    def __init__(self, config: Optional[Deliverable2Config] = None):
        if config is None:
            config = DEFAULT_CONFIG
        self.cfg = config

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def process(self, measurement_y: float, p_gen: float, time_sec: float,
                dt: float, is_timeout: bool = False) -> DefenseAction:
        pass


class RuleBasedDefense(BaseDefense):
    """Baseline A: Rule-Based Defense using Static Thresholds & Jump Rejection."""

    def __init__(self, config: Optional[Deliverable2Config] = None):
        super().__init__(config)
        self.safe_min = float(self.cfg.defense.rule_safe_min)   # 49.80 Hz
        self.safe_max = float(self.cfg.defense.rule_safe_max)   # 50.20 Hz
        self.max_jump = float(self.cfg.defense.rule_max_jump_hz) # 0.20 Hz

        self.last_valid_freq = float(self.cfg.grid.f0)
        self.last_measurement: Optional[float] = None

    def reset(self) -> None:
        self.last_valid_freq = float(self.cfg.grid.f0)
        self.last_measurement = None

    def process(self, measurement_y: float, p_gen: float, time_sec: float,
                dt: float, is_timeout: bool = False) -> DefenseAction:
        is_attack = False
        mitigation_active = False

        # Rule 1: Static frequency bound violation (< 49.80 Hz or > 50.20 Hz)
        out_of_bounds = (measurement_y < self.safe_min or measurement_y > self.safe_max)

        # Rule 2: Excessive single-step jump (> 0.20 Hz)
        step_jump = 0.0
        if self.last_measurement is not None:
            step_jump = abs(measurement_y - self.last_measurement)
        jump_violation = (step_jump > self.max_jump)

        self.last_measurement = measurement_y

        if out_of_bounds or jump_violation or is_timeout:
            is_attack = True
            mitigation_active = True
            # Mitigation: Clamp to safe range or hold last known valid frequency
            if out_of_bounds:
                controlled_freq = max(self.safe_min, min(self.safe_max, measurement_y))
            else:
                controlled_freq = self.last_valid_freq
        else:
            controlled_freq = measurement_y
            self.last_valid_freq = measurement_y

        return DefenseAction(
            time_sec=time_sec,
            raw_measurement=measurement_y,
            controlled_frequency=controlled_freq,
            is_attack_flagged=is_attack,
            mitigation_active=mitigation_active,
            defense_name="Rule-Based Defense",
            info={
                "out_of_bounds": out_of_bounds,
                "jump_violation": jump_violation,
                "step_jump": step_jump
            }
        )


class KalmanOnlyDefense(BaseDefense):
    """Baseline B: Kalman-Only Defense using Innovation Residual Thresholding."""

    def __init__(self, config: Optional[Deliverable2Config] = None):
        super().__init__(config)
        self.threshold_hz = float(self.cfg.defense.kalman_only_threshold_hz)  # 0.15 Hz
        self.kf = GridKalmanFilter(config=self.cfg)

    def reset(self) -> None:
        self.kf.reset()

    def process(self, measurement_y: float, p_gen: float, time_sec: float,
                dt: float, is_timeout: bool = False) -> DefenseAction:
        # 1. Physics Predict Step
        predicted_freq = self.kf.predict(p_gen=p_gen, dt=dt)

        # 2. Compute Innovation Residual
        residual = measurement_y - predicted_freq
        abs_res = abs(residual)

        # 3. Anomaly Decision & Mitigation
        if abs_res > self.threshold_hz or is_timeout:
            is_attack = True
            mitigation_active = True
            # Mitigate: Do NOT poison Kalman state with corrupt reading; use physics prediction
            self.kf.x_hat = predicted_freq
            controlled_freq = predicted_freq
        else:
            is_attack = False
            mitigation_active = False
            # Clean reading: Perform normal Kalman measurement update
            kf_res = self.kf.update(y_k=measurement_y, time_sec=time_sec)
            controlled_freq = kf_res.estimated_freq

        return DefenseAction(
            time_sec=time_sec,
            raw_measurement=measurement_y,
            controlled_frequency=controlled_freq,
            is_attack_flagged=is_attack,
            mitigation_active=mitigation_active,
            defense_name="Kalman-Only Defense",
            info={
                "residual_hz": residual,
                "abs_residual_hz": abs_res,
                "predicted_freq": predicted_freq,
                "threshold_hz": self.threshold_hz
            }
        )


class MultiSignalDefense(BaseDefense):
    """Proposed Defense: 4-Channel Multi-Signal Detector + Kalman State Mitigation."""

    def __init__(self, config: Optional[Deliverable2Config] = None):
        super().__init__(config)
        self.kf = GridKalmanFilter(config=self.cfg)
        self.detector = MultiSignalDetector(config=self.cfg)

    def reset(self) -> None:
        self.kf.reset()
        self.detector.reset()

    def process(self, measurement_y: float, p_gen: float, time_sec: float,
                dt: float, is_timeout: bool = False) -> DefenseAction:
        # 1. Physics Predict Step
        predicted_freq = self.kf.predict(p_gen=p_gen, dt=dt)

        # 2. Compute temporary estimation result for detector
        temp_kf_res = EstimationResult(
            time_sec=time_sec,
            measurement_y=measurement_y,
            predicted_freq=predicted_freq,
            estimated_freq=predicted_freq,
            residual=measurement_y - predicted_freq,
            residual_cov_S=self.kf.P_prior + self.kf.R,
            normalized_residual=(measurement_y - predicted_freq) / max(1e-6, (self.kf.P_prior + self.kf.R)**0.5),
            kalman_gain=self.kf.P_prior / (self.kf.P_prior + self.kf.R),
            error_cov_P=self.kf.P_prior,
            p_gen_mw=p_gen
        )

        # 3. Master Multi-Signal Detector Evaluation
        det_res = self.detector.evaluate(
            kf_res=temp_kf_res,
            time_sec=time_sec,
            is_timeout_flag=is_timeout
        )

        # 4. Mitigation Decision
        if det_res.is_attack:
            is_attack = True
            mitigation_active = True
            # Isolate bad sensor: propagate state via physics model only
            self.kf.x_hat = predicted_freq
            controlled_freq = predicted_freq
        else:
            is_attack = False
            mitigation_active = False
            # Normal operation: incorporate sensor measurement
            updated_res = self.kf.update(y_k=measurement_y, time_sec=time_sec)
            controlled_freq = updated_res.estimated_freq

        return DefenseAction(
            time_sec=time_sec,
            raw_measurement=measurement_y,
            controlled_frequency=controlled_freq,
            is_attack_flagged=is_attack,
            mitigation_active=mitigation_active,
            defense_name="Multi-Signal Defense",
            info={
                "combined_attack_score": det_res.combined_score,
                "score_residual": det_res.score_residual,
                "score_sudden": det_res.score_sudden,
                "score_timing": det_res.score_timing,
                "score_replay": det_res.score_replay,
                "predicted_freq": predicted_freq
            }
        )
