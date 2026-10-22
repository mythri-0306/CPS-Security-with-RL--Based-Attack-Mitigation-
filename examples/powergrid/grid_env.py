"""
Phase 1 & 2 Gym Environment: Power Grid Attack Defense - grid_env.py

Wraps the physical power grid simulation (GridPhysics) into a standard
Gymnasium environment for training and evaluating reinforcement learning
defense agents against cyber-attacks.
"""

import os
import sys
from collections import deque
from typing import Optional, Dict, Any, Tuple, Deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Ensure both local directory and examples/powergrid are in sys.path
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

_PG_DIR = os.path.join(_CURRENT_DIR, "examples", "powergrid")
if os.path.exists(_PG_DIR) and _PG_DIR not in sys.path:
    sys.path.insert(0, _PG_DIR)

from physics import GridPhysics
from config import (
    NOMINAL_FREQ, SAFE_BAND_LOW, SAFE_BAND_HIGH,
    DT, EPISODE_LEN, INERTIA_H, BASE_POWER_MVA, DAMPING_D,
    NOMINAL_LOAD, INITIAL_GEN_SETPOINT,
    CONTROLLER_KP, CONTROLLER_KI, CONTROLLER_DEADBAND_HZ, CONTROLLER_INTEGRAL_MAX,
    GEN_MIN_MW, GEN_MAX_MW,
    W_BAND, W_DISTRUST, W_DISTRUST_ATTACK, W_SWITCH,
    SENSOR_NOISE_SIGMA, SCALE_FACTOR,
    ATTACK_TYPES, ATTACK_START_TIME_RANGE, ATTACK_DEFAULT_DURATION, ATTACK_INTENSITY_RANGES,
    STEALTHY_DRIFT_TARGET_FREQ,
    OBS_FREQ_NORM_SCALE, OBS_RESIDUAL_NORM_SCALE, OBS_ROCOF_NORM_SCALE, OBS_CLIP_LIMIT,
    RESIDUAL_SCALE, OBS_EXTENDED, PACKET_AGE_NORM_SCALE, RESIDUAL_VAR_WINDOW, RESIDUAL_VAR_NORM_SCALE,
)
from estimator_interface import EstimatorInterface, PlaceholderEstimator, KalmanEstimator


class GridDefenseEnv(gym.Env):
    """Gymnasium environment wrapping power grid frequency dynamics and cyber-attacks.

    Actions:
        Discrete(2):
            0 = Trust measurement: Pass measured frequency to the primary controller.
            1 = Distrust measurement: Use the estimator's predicted frequency instead.

    Observations:
        Base Layout (Box(5,)):
            [0] measured_freq_norm  : (measured_freq - 50.0) / OBS_FREQ_NORM_SCALE
            [1] kalman_pred_norm    : (kalman_pred - 50.0) / OBS_FREQ_NORM_SCALE
            [2] residual_norm       : residual / RESIDUAL_SCALE
            [3] residual_roc_norm   : residual_rate_of_change / RESIDUAL_SCALE
            [4] timestep_frac       : step_count / EPISODE_LEN

        Extended Layout (Box(8,) when obs_extended=True):
            [5] packet_age_norm     : packet_age / PACKET_AGE_NORM_SCALE
            [6] duplicate_flag      : 1.0 if identical reading repeated >= 2 steps, else 0.0
            [7] residual_var_norm   : rolling_residual_var / RESIDUAL_VAR_NORM_SCALE

    Rewards:
        Reward per step = -W_BAND * out_of_band
                        - W_DISTRUST * (action == 1 and not attack_active)
                        - W_DISTRUST_ATTACK * (action == 1 and attack_active)
                        - W_SWITCH * (action != prev_action)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        estimator: Optional[EstimatorInterface] = None,
        episode_len: int = EPISODE_LEN,
        dt: float = DT,
        apply_disturbance: bool = True,
        disturbance_time_sec: float = 25.0,
        disturbance_load_step_mw: float = 3.0,
        reward_cfg: Optional[Dict[str, float]] = None,
        obs_extended: Optional[bool] = None,
    ):
        super().__init__()

        self.episode_len = int(episode_len)
        self.dt = float(dt)
        self.apply_disturbance = bool(apply_disturbance)
        self.disturbance_time_sec = float(disturbance_time_sec)
        self.disturbance_load_step_mw = float(disturbance_load_step_mw)

        # Reward weights (overridable via reward_cfg for Phase 4 experiments)
        self.reward_cfg = dict(reward_cfg) if reward_cfg is not None else {}
        self.w_band = float(self.reward_cfg.get("w_band", W_BAND))
        self.w_distrust = float(self.reward_cfg.get("w_distrust", W_DISTRUST))
        self.w_distrust_attack = float(self.reward_cfg.get("w_distrust_attack", W_DISTRUST_ATTACK))
        self.w_switch = float(self.reward_cfg.get("w_switch", W_SWITCH))

        # Extended Observation Flag
        self.obs_extended = bool(obs_extended if obs_extended is not None else OBS_EXTENDED)

        # Estimator hook: defaults to KalmanEstimator for real state estimation
        if estimator is None:
            self.estimator: EstimatorInterface = KalmanEstimator()
        else:
            self.estimator = estimator

        # Action Space: Discrete(2) [0: trust measurement, 1: distrust / use estimate]
        self.action_space = spaces.Discrete(2)

        # Observation Space bounds
        obs_dim = 8 if self.obs_extended else 5
        low_bounds = np.full((obs_dim,), -OBS_CLIP_LIMIT, dtype=np.float32)
        high_bounds = np.full((obs_dim,), OBS_CLIP_LIMIT, dtype=np.float32)
        low_bounds[4] = 0.0
        high_bounds[4] = 1.0
        if self.obs_extended:
            low_bounds[5] = 0.0  # packet age >= 0
            low_bounds[6] = 0.0  # duplicate_flag in [0, 1]
            high_bounds[6] = 1.0
            low_bounds[7] = 0.0  # residual variance >= 0

        self.observation_space = spaces.Box(
            low=low_bounds,
            high=high_bounds,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Phase 4 Timing & Staleness tracking
        self.packet_age: int = 0
        self.consecutive_identical: int = 0
        self.prev_reading: float = NOMINAL_FREQ
        self.residual_history: Deque[float] = deque(maxlen=RESIDUAL_VAR_WINDOW)
        self.prev_action: int = 0

        # Simulation internal state attributes
        self.physics: Optional[GridPhysics] = None
        self.step_count: int = 0
        self.current_load: float = NOMINAL_LOAD
        self.p_gen: float = INITIAL_GEN_SETPOINT
        self.integral_error: float = 0.0

        # Telemetry tracking
        self.current_true_freq: float = NOMINAL_FREQ
        self.current_measured_freq: float = NOMINAL_FREQ
        self.current_kalman_pred: float = NOMINAL_FREQ
        self.current_residual: float = 0.0
        self.prev_residual: float = 0.0
        self.last_valid_measured_freq: float = NOMINAL_FREQ

        # Attack randomization hooks
        self.attack_type: str = "none"
        self.attack_start_time: float = 20.0
        self.attack_duration: float = ATTACK_DEFAULT_DURATION
        self.attack_intensity: float = 0.0
        self.replay_buffer: list = []

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state, re-seed RNG, randomize attack, and return initial obs.

        Args:
            seed (Optional[int]): Random seed for reproducibility.
            options (Optional[Dict[str, Any]]): Optional overrides for attack_type, etc.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: (initial_observation, info_dict)
        """
        super().reset(seed=seed)

        # Re-seed estimator
        estimator_seed = int(self.np_random.integers(0, 2**31 - 1))
        self.estimator.reset(seed=estimator_seed)

        # Instantiate physical grid model
        self.physics = GridPhysics(
            f0=NOMINAL_FREQ,
            H=INERTIA_H,
            S_base=BASE_POWER_MVA,
            D=DAMPING_D,
            dt=self.dt,
            p_load=NOMINAL_LOAD,
            p_gen=INITIAL_GEN_SETPOINT,
        )

        # Reset controller and trajectory tracking
        self.step_count = 0
        self.current_load = NOMINAL_LOAD
        self.p_gen = float(INITIAL_GEN_SETPOINT)
        self.integral_error = 0.0
        self.replay_buffer = []

        # ---------------------------------------------------------------------
        # Randomize cyber-attack parameters using seeded RNG (self.np_random)
        # ---------------------------------------------------------------------
        options = options or {}
        if "attack_type" in options and options["attack_type"] in ATTACK_TYPES:
            self.attack_type = str(options["attack_type"])
        else:
            self.attack_type = str(self.np_random.choice(ATTACK_TYPES))

        if "attack_start_time" in options:
            self.attack_start_time = float(options["attack_start_time"])
        elif "attack_start" in options:
            self.attack_start_time = float(options["attack_start"])
        else:
            self.attack_start_time = float(
                self.np_random.uniform(
                    ATTACK_START_TIME_RANGE[0], ATTACK_START_TIME_RANGE[1]
                )
            )

        self.attack_duration = float(
            options.get("attack_duration", ATTACK_DEFAULT_DURATION)
        )

        # Sample attack intensity from configured range
        if "intensity" in options:
            self.attack_intensity = float(options["intensity"])
        else:
            intensity_min, intensity_max = ATTACK_INTENSITY_RANGES.get(
                self.attack_type, (0.0, 0.0)
            )
            if intensity_min == intensity_max:
                self.attack_intensity = float(intensity_min)
            else:
                self.attack_intensity = float(
                    self.np_random.uniform(intensity_min, intensity_max)
                )

        # Initialize physical frequency & sensor reading
        self.current_true_freq = float(NOMINAL_FREQ)
        noise = float(self.np_random.normal(0.0, SENSOR_NOISE_SIGMA))
        clean_reading = round((self.current_true_freq + noise) * SCALE_FACTOR) / SCALE_FACTOR
        self.current_measured_freq = clean_reading
        self.last_valid_measured_freq = clean_reading
        self.replay_buffer.append(clean_reading)

        # Initial estimate and residual (trusted initial reading)
        kalman_pred, residual = self.estimator.get_estimate(
            measured_freq=self.current_measured_freq,
            true_freq=self.current_true_freq,
            p_gen=self.p_gen,
            dt=self.dt,
            trust=True,
        )
        self.current_kalman_pred = float(kalman_pred)
        self.current_residual = float(residual)
        self.prev_residual = float(residual)

        # Reset Phase 4 staleness & duplicate tracking
        self.packet_age = 0
        self.consecutive_identical = 0
        self.prev_reading = self.current_measured_freq
        self.residual_history.clear()
        self.residual_history.append(float(residual))
        self.prev_action = 0

        obs = self._build_observation(residual_roc=0.0, duplicate_flag=0.0, res_var=0.0)
        info = self._build_info(
            attack_active=False,
            action=0,
            r_band=0.0,
            r_distrust=0.0,
            r_distrust_attack=0.0,
            r_switch=0.0,
        )

        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance simulation by one timestep given agent's action.

        Args:
            action (int): 0 = trust measurement, 1 = distrust measurement (use kalman_pred).

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
                (obs, reward, terminated, truncated, info)
        """
        assert self.action_space.contains(action), f"Invalid action {action}"

        # 1. RL Action Selection: determine which frequency reaches the controller
        if action == 0:
            controlled_freq = self.current_measured_freq  # Trust sensor
        else:
            controlled_freq = self.current_kalman_pred    # Distrust; fallback to estimate

        # 2. Primary Droop + Secondary PI Controller Actuation
        freq_error = NOMINAL_FREQ - controlled_freq
        if abs(freq_error) > CONTROLLER_DEADBAND_HZ:
            self.integral_error += freq_error * self.dt
            self.integral_error = float(
                np.clip(
                    self.integral_error,
                    -CONTROLLER_INTEGRAL_MAX,
                    CONTROLLER_INTEGRAL_MAX,
                )
            )
            p_act = CONTROLLER_KP * freq_error
            i_act = CONTROLLER_KI * self.integral_error
            self.p_gen = INITIAL_GEN_SETPOINT + p_act + i_act
        self.p_gen = float(np.clip(self.p_gen, GEN_MIN_MW, GEN_MAX_MW))

        # 3. Electrical Load Demand (optional step disturbance at t >= 25s)
        current_time = self.step_count * self.dt
        if self.apply_disturbance and current_time >= self.disturbance_time_sec:
            self.current_load = NOMINAL_LOAD + self.disturbance_load_step_mw
        else:
            self.current_load = NOMINAL_LOAD
        self.physics.set_load(self.current_load)

        # 4. Advance Grid Physical Dynamics via Swing Equation
        self.current_true_freq = float(self.physics.step(self.p_gen))

        # Advance timestep
        self.step_count += 1
        t_next = self.step_count * self.dt

        # 5. Determine Cyber-Attack Status for this timestep
        attack_end_time = self.attack_start_time + self.attack_duration
        attack_active = bool(
            self.attack_start_time <= t_next < attack_end_time
            and self.attack_type != "none"
        )

        # 6. Sensor Sampling, Timing/Staleness Tracking & Cyber-Attack Injection
        meas_noise = float(self.np_random.normal(0.0, SENSOR_NOISE_SIGMA))
        clean_sensor_sample = self.current_true_freq + meas_noise
        clean_reading = round(clean_sensor_sample * SCALE_FACTOR) / SCALE_FACTOR

        if not attack_active:
            self.current_measured_freq = clean_reading
            self.last_valid_measured_freq = clean_reading
            self.packet_age = 0  # Fresh measurement received
            if t_next < self.attack_start_time:
                self.replay_buffer.append(clean_reading)
        else:
            # Active attack manipulation
            if self.attack_type == "spoof":
                self.current_measured_freq = float(self.attack_intensity)
                self.packet_age = 0

            elif self.attack_type == "stealthy_drift":
                drift_rate = float(self.attack_intensity)
                elapsed_att = t_next - self.attack_start_time
                drift_freq = float(NOMINAL_FREQ - (drift_rate * elapsed_att))
                self.current_measured_freq = max(STEALTHY_DRIFT_TARGET_FREQ, drift_freq)
                self.packet_age = 0

            elif self.attack_type == "replay":
                if self.replay_buffer:
                    replay_idx = int((t_next - self.attack_start_time) / self.dt) % len(self.replay_buffer)
                    self.current_measured_freq = float(self.replay_buffer[replay_idx])
                else:
                    self.current_measured_freq = float(NOMINAL_FREQ)
                self.packet_age = 0

            elif self.attack_type in ("delay", "dos"):
                # Latency blackout or packet loss holds last known valid telemetry
                self.current_measured_freq = float(self.last_valid_measured_freq)
                self.packet_age += 1  # No fresh measurement: age increments

            else:
                self.current_measured_freq = clean_reading
                self.packet_age = 0

            self.current_measured_freq = (
                round(self.current_measured_freq * SCALE_FACTOR) / SCALE_FACTOR
            )

        # Duplicate tracking
        if abs(self.current_measured_freq - self.prev_reading) < 1e-4:
            self.consecutive_identical += 1
        else:
            self.consecutive_identical = 0
        duplicate_flag = 1.0 if self.consecutive_identical >= 2 else 0.0
        self.prev_reading = self.current_measured_freq

        # 7. State Estimation & Innovation Residual Update (Gated on RL Action)
        # When action == 1 (distrust), measurement update is skipped (predict only)
        # so that spoofed telemetry cannot drag the state estimate.
        trust_measurement = bool(action == 0)
        kalman_pred, residual = self.estimator.get_estimate(
            measured_freq=self.current_measured_freq,
            true_freq=self.current_true_freq,
            p_gen=self.p_gen,
            dt=self.dt,
            trust=trust_measurement,
        )
        self.current_kalman_pred = float(kalman_pred)
        residual_roc = float((residual - self.current_residual) / self.dt)
        self.prev_residual = self.current_residual
        self.current_residual = float(residual)

        # Residual history & rolling variance
        self.residual_history.append(float(residual))
        res_var = float(np.var(self.residual_history)) if len(self.residual_history) >= 2 else 0.0

        # 8. Construct Observation Vector
        obs = self._build_observation(residual_roc=residual_roc, duplicate_flag=duplicate_flag, res_var=res_var)

        # 9. Compute Reward (Phase 4 Multi-Objective Reward Formulation)
        out_of_band = bool(
            self.current_true_freq < SAFE_BAND_LOW
            or self.current_true_freq > SAFE_BAND_HIGH
        )
        r_band = -self.w_band if out_of_band else 0.0
        r_distrust = -self.w_distrust if (action == 1 and not attack_active) else 0.0
        r_distrust_attack = -self.w_distrust_attack if (action == 1 and attack_active) else 0.0
        r_switch = -self.w_switch if (action != self.prev_action) else 0.0
        reward: float = float(r_band + r_distrust + r_distrust_attack + r_switch)
        self.prev_action = int(action)

        # 10. Termination & Truncation
        terminated: bool = False
        truncated: bool = bool(self.step_count >= self.episode_len)

        # 11. Diagnostic Information
        info = self._build_info(
            attack_active=attack_active,
            action=action,
            r_band=r_band,
            r_distrust=r_distrust,
            r_distrust_attack=r_distrust_attack,
            r_switch=r_switch,
            packet_age=self.packet_age,
            duplicate_flag=duplicate_flag,
        )

        return obs, reward, terminated, truncated, info

    def _build_observation(
        self,
        residual_roc: float,
        duplicate_flag: float = 0.0,
        res_var: float = 0.0,
    ) -> np.ndarray:
        """Construct normalized Box float32 observation vector."""
        timestep_frac = float(self.step_count) / float(self.episode_len)

        norm_meas = (self.current_measured_freq - NOMINAL_FREQ) / OBS_FREQ_NORM_SCALE
        norm_pred = (self.current_kalman_pred - NOMINAL_FREQ) / OBS_FREQ_NORM_SCALE
        norm_res = self.current_residual / RESIDUAL_SCALE
        norm_roc = residual_roc / RESIDUAL_SCALE
        norm_frac = timestep_frac

        features = [norm_meas, norm_pred, norm_res, norm_roc, norm_frac]
        if self.obs_extended:
            norm_age = float(self.packet_age) / PACKET_AGE_NORM_SCALE
            norm_dup = float(duplicate_flag)
            norm_var = float(res_var) / RESIDUAL_VAR_NORM_SCALE
            features.extend([norm_age, norm_dup, norm_var])

        raw_obs = np.array(features, dtype=np.float32)
        # Clip to ensure observation strictly stays within observation_space bounds
        obs = np.clip(raw_obs, self.observation_space.low, self.observation_space.high)
        return obs.astype(np.float32)

    def _build_info(
        self,
        attack_active: bool,
        action: int,
        r_band: float = 0.0,
        r_distrust: float = 0.0,
        r_distrust_attack: float = 0.0,
        r_switch: float = 0.0,
        packet_age: int = 0,
        duplicate_flag: float = 0.0,
    ) -> Dict[str, Any]:
        """Construct standardized info dictionary for metrics calculation."""
        out_of_band = bool(
            self.current_true_freq < SAFE_BAND_LOW
            or self.current_true_freq > SAFE_BAND_HIGH
        )
        return {
            "true_freq": float(self.current_true_freq),
            "measured_freq": float(self.current_measured_freq),
            "attack_active": bool(attack_active),
            "attack_type": str(self.attack_type),
            "attack_start": float(self.attack_start_time),
            "intensity": float(self.attack_intensity),
            "out_of_band": out_of_band,
            "step": int(self.step_count),
            "action": int(action),
            "kalman_pred": float(self.current_kalman_pred),
            "residual": float(self.current_residual),
            "p_gen": float(self.p_gen),
            "load_demand": float(self.current_load),
            "r_band": float(r_band),
            "r_distrust": float(r_distrust),
            "r_distrust_attack": float(r_distrust_attack),
            "r_switch": float(r_switch),
            "packet_age": int(packet_age),
            "duplicate_flag": float(duplicate_flag),
        }
