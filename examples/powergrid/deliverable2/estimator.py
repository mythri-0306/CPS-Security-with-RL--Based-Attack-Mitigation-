"""
Deliverable 2: Kalman Filter / State Estimator - estimator.py

Implements a Discrete-Time Linear Kalman Filter based on the linearized
swing equation of the physical power grid model (Deliverable 1).

CPS Physics Background:
-----------------------
The rotational dynamics of the power grid are governed by the swing equation:
    df/dt = (f0 / (2 * H * S_base)) * (P_gen - P_load) - D * (f - f0)

Discretized with forward Euler integration for timestep dt:
    f(k+1) = f(k) + dt * [ (f0 / (2 * H * S_base)) * (P_gen(k) - P_load_nominal) - D * (f(k) - f0) ]
           = (1 - D * dt) * f(k) + (D * dt * f0) + [ (f0 * dt) / (2 * H * S_base) ] * (P_gen(k) - P_load_nominal)

State-Space Representation:
---------------------------
    State: x(k) = f(k) (Grid frequency in Hz)
    Control Input: u(k) = P_gen(k) - P_load_nominal (Active power mismatch in MW)
    Measurement: y(k) = Measured frequency from sensor (Hz)

    State Transition:
        x(k|k-1) = A * x(k-1|k-1) + B * u(k-1) + c + w(k)
        where:
            A = 1.0 - D * dt
            B = (f0 * dt) / (2.0 * H * S_base)
            c = D * dt * f0
            w(k) ~ N(0, Q)  [Process noise: unmodeled load variations]

    Measurement Equation:
        y(k) = C * x(k) + v(k)
        where:
            C = 1.0
            v(k) ~ N(0, R)  [Measurement noise: sensor inaccuracy / quantization]

Kalman Filter Execution Cycle:
------------------------------
1. PREDICT (Time Update):
   - Predicted state: x_hat(k|k-1) = A * x_hat(k-1|k-1) + B * u(k-1) + c
   - Predicted covariance: P(k|k-1) = A * P(k-1|k-1) * A + Q

2. MEASURE & INNOVATION:
   - Measurement residual (innovation): r(k) = y(k) - C * x_hat(k|k-1)
   - Innovation covariance: S(k) = C * P(k|k-1) * C + R

3. CORRECT (Measurement Update):
   - Kalman Gain: K(k) = P(k|k-1) * C / S(k)
   - Corrected state: x_hat(k|k) = x_hat(k|k-1) + K(k) * r(k)
   - Corrected covariance: P(k|k) = (1 - K(k) * C) * P(k|k-1)
"""

from dataclasses import dataclass
from typing import Optional
from .config import Deliverable2Config, DEFAULT_CONFIG


@dataclass
class EstimationResult:
    """Detailed output record from a single Kalman filter execution step."""
    time_sec: float
    measurement_y: float             # Raw sensor reading received (Hz)
    predicted_freq: float            # Prior state estimate x_hat(k|k-1) (Hz)
    estimated_freq: float            # Posterior state estimate x_hat(k|k) (Hz)
    residual: float                  # Innovation residual r(k) = y(k) - x_hat(k|k-1) (Hz)
    residual_cov_S: float            # Innovation covariance S(k) = P(k|k-1) + R
    normalized_residual: float       # Normalized residual = r(k) / sqrt(S(k))
    kalman_gain: float               # Kalman gain K(k)
    error_cov_P: float               # Posterior estimation error covariance P(k|k)
    p_gen_mw: float                  # Generator setpoint used in prediction (MW)


class GridKalmanFilter:
    """Kalman Filter State Estimator tailored to the Power Grid Swing Equation."""

    def __init__(self, config: Optional[Deliverable2Config] = None):
        if config is None:
            config = DEFAULT_CONFIG
        self.cfg = config

        # Physical constants
        self.f0 = float(self.cfg.grid.f0)
        self.H = float(self.cfg.grid.H)
        self.S_base = float(self.cfg.grid.S_base)
        self.D = float(self.cfg.grid.D)
        self.p_load_nominal = float(self.cfg.grid.nominal_load)

        # Kalman Filter noise hyperparameters
        self.Q = float(self.cfg.kalman.process_noise_cov_Q)
        self.R = float(self.cfg.kalman.measurement_noise_cov_R)

        # State initialization
        self.x_hat = float(self.cfg.kalman.initial_state_f)     # Current state estimate x_hat(k|k)
        self.P = float(self.cfg.kalman.initial_error_cov_P0)   # Estimation error covariance P(k|k)

        # Prior predicted state storage
        self.x_hat_prior = self.x_hat
        self.P_prior = self.P
        self.last_p_gen = float(self.cfg.grid.initial_p_gen)

        # Step count & history
        self.step_count = 0
        self.history = []

    def reset(self, initial_state: Optional[float] = None, initial_p_gen: Optional[float] = None) -> None:
        """Reset estimator state back to initial conditions."""
        self.x_hat = float(initial_state if initial_state is not None else self.cfg.kalman.initial_state_f)
        self.P = float(self.cfg.kalman.initial_error_cov_P0)
        self.x_hat_prior = self.x_hat
        self.P_prior = self.P
        self.last_p_gen = float(initial_p_gen if initial_p_gen is not None else self.cfg.grid.initial_p_gen)
        self.step_count = 0
        self.history.clear()

    def predict(self, p_gen: Optional[float] = None, dt: Optional[float] = None) -> float:
        """Step 1: Kalman Predict (Time Update).

        Computes the a priori state prediction x_hat(k|k-1) and covariance P(k|k-1)
        using the swing equation physics and the previous generator power setpoint.

        :param p_gen: Generator setpoint in MW (defaults to last known setpoint)
        :param dt: Time interval in seconds (defaults to physics dt)
        :returns: Predicted frequency in Hz
        """
        if p_gen is not None:
            self.last_p_gen = float(p_gen)
        if dt is None:
            dt = float(self.cfg.grid.dt_physics)

        # State-space coefficients for discretized swing equation
        A = 1.0 - (self.D * dt)
        B = (self.f0 * dt) / (2.0 * self.H * self.S_base)
        c = self.D * dt * self.f0

        # Active power mismatch against nominal load
        u = self.last_p_gen - self.p_load_nominal

        # 1. State Prediction: x_hat(k|k-1) = A * x_hat(k-1|k-1) + B * u + c
        self.x_hat_prior = (A * self.x_hat) + (B * u) + c

        # 2. Covariance Prediction: P(k|k-1) = A * P(k-1|k-1) * A + Q
        self.P_prior = (A * self.P * A) + self.Q

        return self.x_hat_prior

    def update(self, y_k: float, time_sec: float = 0.0) -> EstimationResult:
        """Step 2 & 3: Kalman Innovation and Update (Measurement Update).

        Computes the innovation residual r(k), Kalman gain K(k), and updates the
        state estimate x_hat(k|k) and error covariance P(k|k).

        :param y_k: Observed sensor measurement in Hz
        :param time_sec: Current simulation time in seconds
        :returns: EstimationResult containing prediction, estimate, and residual
        """
        C = 1.0  # Observation matrix: measurement directly observes frequency

        # 1. Measurement residual (innovation): r(k) = y(k) - C * x_hat(k|k-1)
        residual = y_k - (C * self.x_hat_prior)

        # 2. Innovation covariance: S(k) = C * P(k|k-1) * C + R
        S = (C * self.P_prior * C) + self.R

        # 3. Kalman Gain: K(k) = P(k|k-1) * C / S(k)
        K = (self.P_prior * C) / S

        # 4. Corrected State: x_hat(k|k) = x_hat(k|k-1) + K * r(k)
        self.x_hat = self.x_hat_prior + (K * residual)

        # 5. Corrected Covariance: P(k|k) = (1 - K * C) * P(k|k-1)
        self.P = (1.0 - (K * C)) * self.P_prior

        # Normalized residual (dimensionless standard deviations)
        std_S = max(1e-9, S ** 0.5)
        norm_residual = residual / std_S

        res = EstimationResult(
            time_sec=time_sec,
            measurement_y=y_k,
            predicted_freq=self.x_hat_prior,
            estimated_freq=self.x_hat,
            residual=residual,
            residual_cov_S=S,
            normalized_residual=norm_residual,
            kalman_gain=K,
            error_cov_P=self.P,
            p_gen_mw=self.last_p_gen
        )

        self.step_count += 1
        self.history.append(res)
        return res

    def step(self, y_k: float, p_gen: Optional[float] = None, dt: Optional[float] = None,
             time_sec: float = 0.0) -> EstimationResult:
        """Execute a full Predict -> Measure -> Update Kalman step in one call."""
        self.predict(p_gen=p_gen, dt=dt)
        return self.update(y_k=y_k, time_sec=time_sec)
