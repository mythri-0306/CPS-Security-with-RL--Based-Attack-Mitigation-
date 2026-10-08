"""
Deliverable 2: Simulation Engine - simulation_engine.py

Unified simulation engine executing the power grid physics, sensor sampling,
controller polling, attack injection, Kalman state estimation, and attack detection.
"""

from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

from physics import GridPhysics
from .config import Deliverable2Config, DEFAULT_CONFIG
from .estimator import GridKalmanFilter
from .detector import MultiSignalDetector


class SimulationEngine:
    """Unified discrete-time CPS simulator for Power Grid Frequency Control."""

    def __init__(self, config: Optional[Deliverable2Config] = None):
        if config is None:
            config = DEFAULT_CONFIG
        self.cfg = config

    def run_scenario(
        self,
        scenario: str = "normal",           # "normal", "spoof", "slow_drift", "replay", "delay", "dos"
        defense_mode: str = "none",         # "none", "rule_based", "kalman_only", "multi_signal"
        controller_mode: str = "pi",        # "pi" (droop + secondary PI) or "threshold"
        duration_sec: float = 60.0,
        attack_start_sec: float = 20.0,
        attack_duration_sec: float = 20.0,
        apply_disturbance: bool = True,
        disturbance_time_sec: float = 25.0,
        disturbance_load_step_mw: float = 3.0,
        spoof_freq_hz: float = 48.50,
        delay_ms: float = 800.0,
        dos_loss_pct: float = 90.0,
        add_sensor_noise: bool = True,
        seed: int = 42
    ) -> pd.DataFrame:
        """Execute a complete simulation run and return detailed time-series DataFrame."""
        np.random.seed(seed)

        dt = float(self.cfg.grid.dt_physics)
        total_steps = int(duration_sec / dt)
        ctrl_interval = int(round(self.cfg.grid.dt_controller / dt))

        attack_end_sec = attack_start_sec + attack_duration_sec

        # 1. Instantiate Grid Physics
        physics = GridPhysics(
            f0=self.cfg.grid.f0,
            H=self.cfg.grid.H,
            S_base=self.cfg.grid.S_base,
            D=self.cfg.grid.D,
            dt=dt,
            p_load=self.cfg.grid.nominal_load,
            p_gen=self.cfg.grid.initial_p_gen
        )

        # 2. Instantiate Estimator, Detector, and Defense
        kf = GridKalmanFilter(config=self.cfg)
        detector = MultiSignalDetector(config=self.cfg)
        kf.reset()
        detector.reset()

        defense = None
        if defense_mode == "rule_based":
            from .defenses import RuleBasedDefense
            defense = RuleBasedDefense(config=self.cfg)
        elif defense_mode == "kalman_only":
            from .defenses import KalmanOnlyDefense
            defense = KalmanOnlyDefense(config=self.cfg)
        elif defense_mode == "multi_signal":
            from .defenses import MultiSignalDefense
            defense = MultiSignalDefense(config=self.cfg)

        # Replay attack memory buffer
        replay_buffer = []

        # State tracking
        p_gen = float(self.cfg.grid.initial_p_gen)
        last_ctrl_received_freq = float(self.cfg.grid.f0)
        missed_polls = 0
        last_packet_arrival_time = 0.0
        integral_error = 0.0

        records = []

        for step in range(total_steps):
            t = step * dt

            # Ground-truth attack active window
            attack_active = bool(attack_start_sec <= t < attack_end_sec and scenario != "normal")

            # Physical Load Disturbance
            if apply_disturbance and t >= disturbance_time_sec:
                current_load = self.cfg.grid.nominal_load + disturbance_load_step_mw
            else:
                current_load = self.cfg.grid.nominal_load
            physics.set_load(current_load)

            # Physics Step (Ground Truth)
            true_freq = physics.step(p_gen)

            # Sensor Sampling with Noise & Quantization
            meas_noise = np.random.normal(0.0, np.sqrt(self.cfg.kalman.measurement_noise_cov_R)) if add_sensor_noise else 0.0
            sensor_sampled = true_freq + meas_noise
            sensor_scaled_int = int(round(sensor_sampled * self.cfg.grid.scale_factor))
            sensor_freq_sent = sensor_scaled_int / self.cfg.grid.scale_factor

            # Buffer readings for replay attack during early normal operation (t < attack_start)
            if t < attack_start_sec:
                replay_buffer.append(sensor_freq_sent)

            # Network Transmission / Attack Injection
            packet_arrived = True
            packet_time = t
            is_timeout_event = False

            if attack_active:
                if scenario == "spoof":
                    # Attacker directly overwrites Modbus register with abrupt fake frequency
                    controller_received_freq = float(spoof_freq_hz)

                elif scenario == "slow_drift":
                    # Stealthy gradual ramp down (e.g. 0.05 Hz/s)
                    drift_rate = 0.05
                    elapsed_att = t - attack_start_sec
                    controller_received_freq = max(float(spoof_freq_hz), float(self.cfg.grid.f0) - (drift_rate * elapsed_att))

                elif scenario == "replay":
                    # Attacker replays historical pre-disturbance nominal readings
                    if replay_buffer:
                        replay_idx = int((t - attack_start_sec) / dt) % len(replay_buffer)
                        controller_received_freq = replay_buffer[replay_idx]
                    else:
                        controller_received_freq = float(self.cfg.grid.f0)

                elif scenario == "delay":
                    # Modbus TCP synchronous polling under link latency:
                    # An 800ms latency stretches the polling round-trip to 1.0s (10 physics steps).
                    # Telemetry packets only arrive once every delay cycle; other steps experience delay blackout.
                    delay_steps = int(round((delay_ms / 1000.0) / dt))
                    if (step % max(1, delay_steps)) == 0:
                        packet_arrived = True
                        packet_time = t
                        delayed_idx = max(0, step - delay_steps)
                        controller_received_freq = records[delayed_idx]["sensor_reading_sent"] if delayed_idx < len(records) else sensor_freq_sent
                        last_packet_arrival_time = t
                    else:
                        packet_arrived = False
                        # Inter-arrival delay builds up at controller
                        packet_time = last_packet_arrival_time
                        controller_received_freq = last_ctrl_received_freq
                        if (t - last_packet_arrival_time) >= self.cfg.detector.timing_jitter_tolerance_sec:
                            is_timeout_event = True

                elif scenario == "dos":
                    # Packet drop DoS: overwhelming majority of polls fail
                    if np.random.rand() < (dos_loss_pct / 100.0):
                        packet_arrived = False
                        missed_polls += 1
                        packet_time = last_packet_arrival_time
                        if missed_polls >= 3:
                            is_timeout_event = True
                        controller_received_freq = last_ctrl_received_freq
                    else:
                        missed_polls = 0
                        packet_arrived = True
                        packet_time = t
                        last_packet_arrival_time = t
                        controller_received_freq = sensor_freq_sent
                else:
                    controller_received_freq = sensor_freq_sent
            else:
                controller_received_freq = sensor_freq_sent
                missed_polls = 0
                last_packet_arrival_time = t

            # Controller & Estimator Execution
            if packet_arrived:
                last_ctrl_received_freq = controller_received_freq
                last_packet_arrival_time = packet_time

            # Kalman Filter Step
            kf_res = kf.step(
                y_k=controller_received_freq,
                p_gen=p_gen,
                dt=dt,
                time_sec=t
            )

            # Multi-Signal Detector Step (Always evaluated for benchmark logging)
            det_res = detector.evaluate(
                kf_res=kf_res,
                time_sec=packet_time,
                is_timeout_flag=is_timeout_event
            )

            # Defense Processing & Mitigation
            if defense is not None:
                def_action = defense.process(
                    measurement_y=controller_received_freq,
                    p_gen=p_gen,
                    time_sec=t,
                    dt=dt,
                    is_timeout=is_timeout_event
                )
                controlled_freq_input = def_action.controlled_frequency
                is_mitigation_active = def_action.mitigation_active
                is_defense_attack_flag = def_action.is_attack_flagged
            else:
                controlled_freq_input = controller_received_freq
                is_mitigation_active = False
                is_defense_attack_flag = False

            # Controller Actuation (Threshold or PI Control Loop using mitigated frequency)
            if step % ctrl_interval == 0:
                ctrl_reading = controlled_freq_input
                if controller_mode == "pi":
                    freq_err = self.cfg.grid.f0 - ctrl_reading
                    if abs(freq_err) > 0.005:
                        integral_error += freq_err * self.cfg.grid.dt_controller
                        integral_error = max(-30.0, min(30.0, integral_error))
                        p_act = 15.0 * freq_err
                        i_act = 2.5 * integral_error
                        p_gen = self.cfg.grid.initial_p_gen + p_act + i_act
                else:
                    if ctrl_reading < 49.80:
                        p_gen += 0.05
                    elif ctrl_reading > 50.20:
                        p_gen -= 0.05

                p_gen = max(10.0, min(300.0, p_gen))

            # Record timestep data
            records.append({
                "sim_time_sec": t,
                "true_frequency": true_freq,
                "sensor_reading_sent": sensor_freq_sent,
                "controller_reading_received": controller_received_freq,
                "controlled_frequency_input": controlled_freq_input,
                "generation_command": p_gen,
                "load_demand_mw": current_load,
                "attack_active": attack_active,
                "scenario": scenario,
                "defense_mode": defense_mode,
                "is_mitigation_active": is_mitigation_active,
                "is_defense_attack_flag": is_defense_attack_flag,
                # Kalman Filter outputs
                "predicted_freq": kf_res.predicted_freq,
                "estimated_freq": kf_res.estimated_freq,
                "residual": kf_res.residual,
                "residual_cov_S": kf_res.residual_cov_S,
                "normalized_residual": kf_res.normalized_residual,
                "kalman_gain": kf_res.kalman_gain,
                # Detector outputs
                "is_attack_detected": det_res.is_attack,
                "combined_attack_score": det_res.combined_score,
                "score_residual": det_res.score_residual,
                "score_sudden": det_res.score_sudden,
                "score_timing": det_res.score_timing,
                "score_replay": det_res.score_replay,
                "raw_rocof_hz_per_sec": det_res.raw_rocof_hz_per_sec,
                "raw_inter_arrival_sec": det_res.raw_inter_arrival_sec,
                "consecutive_identical_count": det_res.consecutive_identical_count
            })

        return pd.DataFrame(records)
