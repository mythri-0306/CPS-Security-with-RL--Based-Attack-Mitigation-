"""
State Estimator Interface - estimator_interface.py

Defines the abstract base class for power grid state estimators, a
PlaceholderEstimator for synthetic Gaussian noise baselines, and a
KalmanEstimator adapter wrapping Mythri's GridKalmanFilter.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import os
import sys
import numpy as np

# Ensure path resolution for deliverable2 modules
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PG_DIR = os.path.join(_CURRENT_DIR, "examples", "powergrid")
for p in [_CURRENT_DIR, _PG_DIR]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

try:
    from config import PLACEHOLDER_NOISE_SIGMA, INITIAL_GEN_SETPOINT
except ImportError:
    from .config import PLACEHOLDER_NOISE_SIGMA, INITIAL_GEN_SETPOINT


class EstimatorInterface(ABC):
    """Abstract interface for power grid frequency state estimators."""

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset internal state and RNG of the estimator at episode start.

        Args:
            seed (Optional[int]): Random seed for reproducibility.
        """
        pass

    @abstractmethod
    def get_estimate(
        self,
        measured_freq: float,
        true_freq: Optional[float] = None,
        p_gen: Optional[float] = None,
        dt: Optional[float] = None,
        trust: bool = True,
    ) -> Tuple[float, float]:
        """Compute estimated/predicted frequency and innovation residual.

        Args:
            measured_freq (float): Measured frequency received from sensor/network (Hz).
            true_freq (Optional[float]): Ground truth physical frequency (used ONLY by
                PlaceholderEstimator for baseline comparison; ignored by KalmanEstimator).
            p_gen (Optional[float]): Active generation setpoint (MW) for physical model prediction.
            dt (Optional[float]): Integration period in seconds.
            trust (bool): If True, measurement is trusted and used to update estimator state.
                If False, measurement update is gated/skipped (predict-only fallback).

        Returns:
            Tuple[float, float]: (kalman_pred, residual)
                kalman_pred (float): Estimated or predicted grid frequency in Hz.
                residual (float): Innovation residual = measured_freq - kalman_pred in Hz.
        """
        pass


class PlaceholderEstimator(EstimatorInterface):
    """Placeholder estimator generating synthetic predictions with Gaussian noise.

    Allows training and testing the Gym environment with synthetic residuals:
        kalman_pred = true_freq + N(0, sigma^2)
        residual = measured_freq - kalman_pred
    """

    def __init__(self, sigma: float = PLACEHOLDER_NOISE_SIGMA, seed: Optional[int] = None):
        self.sigma = float(sigma)
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset RNG state with optional seed."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def get_estimate(
        self,
        measured_freq: float,
        true_freq: Optional[float] = None,
        p_gen: Optional[float] = None,
        dt: Optional[float] = None,
        trust: bool = True,
    ) -> Tuple[float, float]:
        """Generate placeholder estimate and innovation residual."""
        # Use true_freq if provided, else fallback to measured_freq
        base_freq = float(true_freq if true_freq is not None else measured_freq)
        noise = float(self.rng.normal(0.0, self.sigma))
        kalman_pred = float(base_freq + noise)
        residual = float(measured_freq - kalman_pred)
        return kalman_pred, residual


class KalmanEstimator(EstimatorInterface):
    """Kalman Filter adapter wrapping Mythri's GridKalmanFilter.

    Adheres strictly to the Cyber-Physical principle that the estimator must
    NOT observe ground truth (true_freq). Instead, it relies purely on sensor
    telemetry (measured_freq), actuator feedback (p_gen), and the swing equation
    physics state-space model.
    """

    def __init__(self, config=None, initial_p_gen: float = INITIAL_GEN_SETPOINT):
        # Dynamically import GridKalmanFilter from deliverable2
        try:
            from deliverable2.estimator import GridKalmanFilter
        except ImportError:
            try:
                from examples.powergrid.deliverable2.estimator import GridKalmanFilter
            except ImportError:
                # Add examples/powergrid to path if not already resolved
                pg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples", "powergrid")
                if pg_path not in sys.path:
                    sys.path.insert(0, pg_path)
                from deliverable2.estimator import GridKalmanFilter

        self.kf = GridKalmanFilter(config=config)
        self.initial_p_gen = float(initial_p_gen)

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset Kalman filter state back to nominal initial conditions."""
        self.kf.reset(initial_state=50.00, initial_p_gen=self.initial_p_gen)

    def get_estimate(
        self,
        measured_freq: float,
        true_freq: Optional[float] = None,
        p_gen: Optional[float] = None,
        dt: Optional[float] = None,
        trust: bool = True,
    ) -> Tuple[float, float]:
        """Compute prior state prediction and innovation residual from measurement.

        Args:
            measured_freq: Observed sensor measurement y_k in Hz.
            true_freq: IGNORED by KalmanEstimator (no ground truth leakage).
            p_gen: Generator setpoint in MW used for physical model predict step.
            dt: Timestep in seconds.
            trust: If True (action 0), perform full Kalman update with y_k.
                   If False (action 1 / distrust), gate / skip the measurement
                   update so spoofed measurements cannot contaminate the state
                   estimate. State remains the prior prediction.

        Returns:
            Tuple[float, float]: (kalman_pred, residual)
                kalman_pred: Prior prediction x_hat(k|k-1) in Hz.
                residual: Innovation residual = measured_freq - kalman_pred in Hz.
        """
        # Step 1: Predict next state using physics swing equation
        self.kf.predict(p_gen=p_gen, dt=dt)
        kalman_pred = float(self.kf.x_hat_prior)
        residual = float(measured_freq - kalman_pred)

        # Step 2: Measurement update gated by trust action
        if trust:
            # Trusted measurement: perform standard Kalman innovation and state update
            self.kf.update(y_k=float(measured_freq))
        else:
            # Distrusted measurement: skip correction; retain prior prediction
            # This prevents spoofed sensor telemetry from dragging the state estimate
            self.kf.x_hat = float(self.kf.x_hat_prior)
            self.kf.P = float(self.kf.P_prior)

        return kalman_pred, residual

