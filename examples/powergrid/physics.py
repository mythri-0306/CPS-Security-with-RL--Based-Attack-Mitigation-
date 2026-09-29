"""
Power Grid Frequency Control CPS Testbed - physics.py

Implements GridPhysics (swing equation physical model) and the
PowerGridProcess continuous simulation loop using MiniCPS's state abstraction.
"""

import time
import sys
try:
    from minicps.devices import Device
except ImportError:
    class Device(object):
        def __init__(self, *args, **kwargs):
            pass
from utils import (
    STATE, FREQ_TAG, GEN_SETPOINT_TAG, LOAD_DEMAND_TAG,
    SENSOR_FREQ_TAG, CONTROLLER_FREQ_TAG, ATTACK_ACTIVE_TAG,
    NOMINAL_FREQ, INERTIA_H, BASE_POWER_MVA, DAMPING_D,
    PHYSICS_PERIOD_SEC, NOMINAL_LOAD, INITIAL_GEN_SETPOINT
)
from logger import SimulationLogger


class GridPhysics:
    """Power grid rotational frequency physical model.

    Governed by the scaled swing equation:
        df/dt = (f0 / (2*H)) * ((P_gen - P_load) / S_base) - D * (f - f0)
    Discretized via forward Euler integration:
        f(t + dt) = f(t) + df/dt * dt
    """

    def __init__(self, f0=NOMINAL_FREQ, H=INERTIA_H, S_base=BASE_POWER_MVA,
                 D=DAMPING_D, dt=PHYSICS_PERIOD_SEC,
                 p_load=NOMINAL_LOAD, p_gen=INITIAL_GEN_SETPOINT):
        self.frequency = float(f0)
        self.f0 = float(f0)
        self.H = float(H)
        self.S_base = float(S_base)
        self.D = float(D)
        self.dt = float(dt)
        self.p_load = float(p_load)
        self.p_gen = float(p_gen)

    def set_load(self, p_load):
        """Update current electrical load demand."""
        self.p_load = float(p_load)

    def step(self, generation_setpoint):
        """Execute one Euler integration step given generator setpoint.

        :param float generation_setpoint: Active generator power output P_gen (MW)
        :returns float: Updated grid frequency (Hz)
        """
        self.p_gen = float(generation_setpoint)
        # 1. Active power mismatch in per-unit (p.u.)
        delta_p_pu = (self.p_gen - self.p_load) / self.S_base
        # 2. System load damping effect in Hz
        damping_hz = self.D * (self.frequency - self.f0)
        # 3. Scaled swing equation (RoCoF in Hz/s)
        df_dt = (self.f0 / (2.0 * self.H)) * delta_p_pu - damping_hz
        self.frequency += df_dt * self.dt
        return self.frequency


class PowerGridProcess(Device):
    """MiniCPS Device process executing the continuous grid simulation."""

    def __init__(self, name="physics", duration=60.0, run_label="baseline_run1",
                 apply_disturbance=True):
        self.duration = float(duration)
        self.run_label = run_label
        self.apply_disturbance = apply_disturbance
        self.physics = GridPhysics(
            f0=NOMINAL_FREQ,
            H=INERTIA_H,
            S_base=BASE_POWER_MVA,
            D=DAMPING_D,
            dt=PHYSICS_PERIOD_SEC,
            p_load=NOMINAL_LOAD,
            p_gen=INITIAL_GEN_SETPOINT
        )
        self.logger = SimulationLogger(run_label=self.run_label)
        super(PowerGridProcess, self).__init__(
            name=name,
            protocol=None,
            state=STATE
        )

    def _start(self):
        self.pre_loop()
        self.main_loop()

    def _stop(self):
        self.logger.close()

    def pre_loop(self):
        """Initialize state database values."""
        self.set(FREQ_TAG, f"{NOMINAL_FREQ:.4f}")
        self.set(GEN_SETPOINT_TAG, f"{INITIAL_GEN_SETPOINT:.2f}")
        self.set(LOAD_DEMAND_TAG, f"{NOMINAL_LOAD:.2f}")
        self.set(SENSOR_FREQ_TAG, f"{NOMINAL_FREQ:.4f}")
        self.set(CONTROLLER_FREQ_TAG, f"{NOMINAL_FREQ:.4f}")
        self.set(ATTACK_ACTIVE_TAG, "0")
        print("INFO [Physics] Initialized physical state: f=50.00 Hz, P_gen=100.00 MW, P_load=100.00 MW")

    def main_loop(self):
        """Continuous physical simulation loop."""
        sim_time = 0.0
        total_steps = int(self.duration / PHYSICS_PERIOD_SEC)
        step_count = 0

        print(f"INFO [Physics] Starting simulation loop for {self.duration}s ({total_steps} steps, label: {self.run_label})...")

        while step_count < total_steps:
            start_wall = time.time()

            # Optional load disturbance: at t = 10s, step load up to 103.0 MW
            # to trigger under-frequency and controller compensation
            if self.apply_disturbance and sim_time >= 10.0:
                current_load = NOMINAL_LOAD + 3.0  # 103.0 MW (+3% load step)
            else:
                current_load = NOMINAL_LOAD

            self.physics.set_load(current_load)
            self.set(LOAD_DEMAND_TAG, f"{current_load:.2f}")

            # Read generator setpoint from state database
            try:
                p_gen = float(self.get(GEN_SETPOINT_TAG))
            except Exception:
                p_gen = self.physics.p_gen

            # Step physical model (ground truth before Modbus scaling/transmission)
            true_freq = self.physics.step(p_gen)

            # Write updated ground truth frequency to state database
            self.set(FREQ_TAG, f"{true_freq:.4f}")

            # Read sensor sent reading, controller received reading, and attack flag
            try:
                sensor_reading_sent = float(self.get(SENSOR_FREQ_TAG))
            except Exception:
                sensor_reading_sent = true_freq

            try:
                controller_reading_received = float(self.get(CONTROLLER_FREQ_TAG))
            except Exception:
                controller_reading_received = true_freq

            try:
                attack_val = self.get(ATTACK_ACTIVE_TAG)
                attack_active = (attack_val in ['1', 'True', 'true', 1, True])
            except Exception:
                attack_active = False

            # Log current state
            self.logger.log(
                sim_time_sec=sim_time,
                true_frequency=true_freq,
                sensor_reading_sent=sensor_reading_sent,
                controller_reading_received=controller_reading_received,
                generation_command=p_gen,
                load_demand_mw=current_load,
                attack_active=attack_active
            )

            if step_count % 50 == 0:
                print(f"[Physics t={sim_time:5.1f}s] TrueFreq: {true_freq:.4f} Hz | Sensor: {sensor_reading_sent:.4f} Hz | Controller: {controller_reading_received:.4f} Hz | P_gen: {p_gen:6.2f} MW | P_load: {current_load:6.2f} MW")

            sim_time += PHYSICS_PERIOD_SEC
            step_count += 1

            # Maintain real-time pacing
            elapsed = time.time() - start_wall
            sleep_time = PHYSICS_PERIOD_SEC - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        print("INFO [Physics] Simulation completed cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Power Grid Physical Process Simulation")
    parser.add_argument("--duration", type=float, default=60.0, help="Simulation duration in seconds")
    parser.add_argument("--run-label", type=str, default="baseline_run1", help="Label for the run output CSV")
    parser.add_argument("--no-disturbance", action="store_true", help="Disable step load disturbance")
    args = parser.parse_args()

    process = PowerGridProcess(
        duration=args.duration,
        run_label=args.run_label,
        apply_disturbance=not args.no_disturbance
    )
