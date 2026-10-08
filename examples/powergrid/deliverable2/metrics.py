"""
Deliverable 2: Evaluation Metrics - metrics.py

Defines and computes standardized Cyber-Physical System (CPS) security metrics:

1. Cyber-Layer Anomaly Detection Metrics (Per-Timestep Confusion Matrix):
   - Ground Truth: y_true(t) in {0, 1} (Attack active flag from state DB)
   - Prediction:   y_pred(t) in {0, 1} (Detector boolean decision)
   - TP (True Positive):  y_true=1 and y_pred=1 (Attack correctly flagged)
   - FP (False Positive): y_true=0 and y_pred=1 (False alarm during normal operation)
   - TN (True Negative):  y_true=0 and y_pred=0 (Normal operation correctly allowed)
   - FN (False Negative): y_true=1 and y_pred=0 (Attack missed)
   - Precision:  TP / (TP + FP)  (Fraction of alarms that are true attacks)
   - Recall:     TP / (TP + FN)  (Fraction of attack timesteps caught)
   - False Alarm Rate (FAR): FP / (FP + TN) (Probability of false alarm when normal)
   - Detection Delay: Time in seconds from attack onset (t_attack_start) to first True Positive detection.

2. Physical-Layer Consequence & Resilience Metrics:
   - Maximum Frequency Deviation: max |f_true(t) - f_nominal| (Hz)
   - Time Outside Safe Range: Cumulative seconds where f_true < 49.80 Hz or f_true > 50.20 Hz
   - Attack Success Rate (ASR):
     Defined as ASR = 1.0 (100%) if the attack causes physical frequency excursion outside
     the safe operating envelope for > 3.0 cumulative seconds WITHOUT mitigation, and ASR = 0.0 (0%)
     if the defense maintains grid stability within safe limits.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


@dataclass
class ScenarioMetrics:
    """Comprehensive evaluation record combining cyber detection and physical resilience metrics."""
    scenario: str
    defense_mode: str
    total_steps: int
    # Cyber Detection Metrics
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    far: float
    detection_delay_sec: float
    # Physical Impact Metrics
    max_freq_dev_hz: float
    time_outside_safe_sec: float
    attack_success: bool
    attack_success_rate: float


def compute_metrics(df: pd.DataFrame, scenario_name: str, defense_mode: str,
                    safe_band: tuple = (49.80, 50.20), f_nominal: float = 50.00) -> ScenarioMetrics:
    """Compute all cyber-layer and physical-layer metrics for a single simulation run."""
    t = df["sim_time_sec"].to_numpy()
    dt = float(t[1] - t[0]) if len(t) > 1 else 0.1
    f_true = df["true_frequency"].to_numpy()

    # 1. Cyber Detection Ground Truth & Predictions
    y_true = df["attack_active"].astype(bool).to_numpy()
    
    # Use defense attack flag if defense is active, else master detector flag
    if "is_defense_attack_flag" in df.columns and defense_mode != "none":
        y_pred = df["is_defense_attack_flag"].astype(bool).to_numpy()
    elif "is_attack_detected" in df.columns:
        y_pred = df["is_attack_detected"].astype(bool).to_numpy()
    else:
        y_pred = np.zeros_like(y_true, dtype=bool)

    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum((~y_true) & y_pred))
    tn = int(np.sum((~y_true) & (~y_pred)))
    fn = int(np.sum(y_true & (~y_pred)))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else (1.0 if np.sum(y_pred) == 0 else 0.0)
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else (1.0 if not np.any(y_true) else 0.0)
    far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # Detection Delay
    attack_indices = np.where(y_true)[0]
    if len(attack_indices) > 0:
        first_attack_t = t[attack_indices[0]]
        detected_indices = np.where(y_true & y_pred)[0]
        if len(detected_indices) > 0:
            first_det_t = t[detected_indices[0]]
            detection_delay = float(max(0.0, first_det_t - first_attack_t))
        else:
            detection_delay = float("inf")  # Attack not detected
    else:
        detection_delay = 0.0  # Normal operation (no attack present)

    # 2. Physical Consequence Metrics
    max_dev = float(np.max(np.abs(f_true - f_nominal)))
    low_b, high_b = safe_band
    unsafe_mask = (f_true < low_b) | (f_true > high_b)
    time_outside_safe = float(np.sum(unsafe_mask) * dt)

    # Attack Success: True if unsafe physical consequence occurs (> 3.0s excursion)
    attack_success = bool(time_outside_safe > 3.0 and scenario_name.lower() != "normal")
    asr = 1.0 if attack_success else 0.0

    return ScenarioMetrics(
        scenario=scenario_name.upper(),
        defense_mode=defense_mode,
        total_steps=len(df),
        tp=tp, fp=fp, tn=tn, fn=fn,
        precision=precision,
        recall=recall,
        far=far,
        detection_delay_sec=detection_delay,
        max_freq_dev_hz=max_dev,
        time_outside_safe_sec=time_outside_safe,
        attack_success=attack_success,
        attack_success_rate=asr
    )
