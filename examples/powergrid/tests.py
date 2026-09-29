"""
Unit and integration tests for Power Grid Frequency Control Testbed.
"""

import unittest
import os
import sys

# Ensure repo root is on sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd
from physics import GridPhysics
from logger import SimulationLogger
from plots import plot_simulation_run
from utils import (
    NOMINAL_FREQ, INERTIA_H, BASE_POWER_MVA, DAMPING_D,
    PHYSICS_PERIOD_SEC, NOMINAL_LOAD, INITIAL_GEN_SETPOINT, SCALE_FACTOR,
    FREQ_LOW_THRESH, FREQ_HIGH_THRESH, RAMP_STEP,
    CONTROLLER_KP, CONTROLLER_KI, CONTROLLER_DEADBAND_HZ
)


class TestGridPhysics(unittest.TestCase):
    """Test the physical swing equation simulation."""

    def setUp(self):
        self.physics = GridPhysics(
            f0=NOMINAL_FREQ,
            H=INERTIA_H,
            S_base=BASE_POWER_MVA,
            D=DAMPING_D,
            dt=PHYSICS_PERIOD_SEC,
            p_load=NOMINAL_LOAD,
            p_gen=INITIAL_GEN_SETPOINT
        )

    def test_equilibrium(self):
        """When P_gen == P_load, frequency must remain unchanged."""
        for _ in range(50):
            f = self.physics.step(INITIAL_GEN_SETPOINT)
        self.assertAlmostEqual(f, NOMINAL_FREQ, places=4)

    def test_underfrequency_on_load_step(self):
        """When load increases (P_gen < P_load), frequency must drop."""
        self.physics.set_load(NOMINAL_LOAD + 5.0)  # 105 MW load
        f_start = self.physics.frequency
        f_next = self.physics.step(INITIAL_GEN_SETPOINT)
        self.assertLess(f_next, f_start)

    def test_overfrequency_on_generation_excess(self):
        """When generation increases (P_gen > P_load), frequency must rise."""
        f_start = self.physics.frequency
        f_next = self.physics.step(INITIAL_GEN_SETPOINT + 5.0)
        self.assertGreater(f_next, f_start)

    def test_damping_self_regulation(self):
        """Without controller, damping causes frequency to stabilize at a non-zero steady-state offset."""
        self.physics.set_load(102.0)  # 2 MW disturbance
        for _ in range(500):
            f = self.physics.step(INITIAL_GEN_SETPOINT)
        # With D=1.0, steady state delta_f = (f0 / (2H * D)) * delta_p_pu ... frequency settles finite
        self.assertGreater(f, 40.0)
        self.assertLess(f, NOMINAL_FREQ)

    def test_threshold_control_recovery(self):
        """Simulate threshold control loop keeping frequency within bounds under large load disturbance."""
        self.physics.set_load(106.0)  # 6 MW extra load to cross 49.80 Hz threshold
        p_gen = INITIAL_GEN_SETPOINT

        for step in range(500):
            f = self.physics.step(p_gen)
            # Controller runs every 2 physics steps (0.2s)
            if step % 2 == 0:
                if f < FREQ_LOW_THRESH:
                    p_gen += RAMP_STEP
                elif f > FREQ_HIGH_THRESH:
                    p_gen -= RAMP_STEP

        # Verify that frequency remains bounded around nominal 50 Hz (+/- 0.6 Hz)
        self.assertAlmostEqual(self.physics.frequency, NOMINAL_FREQ, delta=0.6)
        self.assertGreaterEqual(p_gen, 102.0)

    def test_pi_control_recovery(self):
        """Simulate PI control loop eliminating steady-state frequency error."""
        self.physics.set_load(103.0)  # 3 MW extra load
        p_gen = INITIAL_GEN_SETPOINT
        integral_err = 0.0
        dt_ctrl = 0.2

        for step in range(600):
            f = self.physics.step(p_gen)
            if step % 2 == 0:
                err = NOMINAL_FREQ - f
                if abs(err) > CONTROLLER_DEADBAND_HZ:
                    integral_err += err * dt_ctrl
                    integral_err = max(-20.0, min(20.0, integral_err))
                    p_gen = INITIAL_GEN_SETPOINT + CONTROLLER_KP * err + CONTROLLER_KI * integral_err

        # PI control should restore frequency tightly to 50.00 Hz (+/- 0.05 Hz)
        self.assertAlmostEqual(self.physics.frequency, NOMINAL_FREQ, delta=0.05)
        self.assertAlmostEqual(p_gen, 103.0, delta=0.2)


class TestModbusScaling(unittest.TestCase):
    """Test integer scaling for 16-bit Modbus registers."""

    def test_scaling_precision(self):
        freq = 49.82
        scaled = int(round(freq * SCALE_FACTOR))
        self.assertEqual(scaled, 4982)
        recovered = float(scaled) / SCALE_FACTOR
        self.assertAlmostEqual(recovered, freq, places=2)


class TestLoggerAndPlotting(unittest.TestCase):
    """Test Phase 2 logger schema and plotting generation."""

    def test_logger_schema_and_dataframe(self):
        test_csv = "logs/test_unit_log.csv"
        if os.path.exists(test_csv):
            os.remove(test_csv)

        with SimulationLogger(filename=test_csv) as log:
            log.log(
                sim_time_sec=0.0,
                true_frequency=50.0,
                sensor_reading_sent=50.0,
                controller_reading_received=50.0,
                generation_command=100.0,
                load_demand_mw=100.0,
                attack_active=False
            )
            log.log(
                sim_time_sec=0.1,
                true_frequency=49.9,
                sensor_reading_sent=49.9,
                controller_reading_received=49.9,
                generation_command=100.5,
                load_demand_mw=103.0,
                attack_active=True
            )
            df = log.to_dataframe()

        self.assertEqual(len(df), 2)
        for expected_col in SimulationLogger.HEADER:
            self.assertIn(expected_col, df.columns)

        # Test plot generation
        png_path = plot_simulation_run(df, run_label="test_unit_plot", save_path="logs/test_unit_plot.png")
        self.assertTrue(os.path.exists(png_path))
        self.assertGreater(os.path.getsize(png_path), 1000)

        # Cleanup
        if os.path.exists(test_csv):
            os.remove(test_csv)
        if os.path.exists(png_path):
            os.remove(png_path)


if __name__ == "__main__":
    unittest.main()
