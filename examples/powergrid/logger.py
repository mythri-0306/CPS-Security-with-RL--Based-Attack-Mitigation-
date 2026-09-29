"""
Power Grid Frequency Control CPS Testbed - logger.py

Enhanced SimulationLogger recording at every timestep:
- timestamp: Wall clock Unix timestamp
- sim_time_sec: Simulated time from t=0
- true_frequency: Ground-truth frequency from GridPhysics (Hz)
- sensor_reading_sent: Value sent/held by Sensor Modbus server (Hz)
- controller_reading_received: Value read by Controller Modbus client (Hz)
- generation_command: Generator setpoint decided by controller (MW)
- load_demand_mw: Electrical load demand (MW)
- attack_active: Boolean flag (False for baseline)
"""

import csv
import os
import time
import pandas as pd


class SimulationLogger:
    """CSV and DataFrame logger for Power Grid CPS simulation runs."""

    HEADER = [
        "timestamp",
        "sim_time_sec",
        "true_frequency",
        "sensor_reading_sent",
        "controller_reading_received",
        "generation_command",
        "load_demand_mw",
        "attack_active"
    ]

    def __init__(self, filename=None, run_label="baseline_run1", log_dir="logs"):
        self.run_label = run_label
        self.log_dir = log_dir

        if filename is not None:
            self.filename = filename
        else:
            self.filename = os.path.join(self.log_dir, f"{self.run_label}.csv")

        target_dir = os.path.dirname(self.filename)
        if target_dir and not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)

        self.records = []
        self.file = open(self.filename, mode='w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(self.HEADER)
        self.file.flush()

    def log(self, sim_time_sec, true_frequency, sensor_reading_sent,
            controller_reading_received, generation_command,
            load_demand_mw, attack_active=False):
        """Record a single simulation timestep."""
        wall_time = time.time()
        attack_bool = bool(attack_active)

        record = {
            "timestamp": wall_time,
            "sim_time_sec": float(sim_time_sec),
            "true_frequency": float(true_frequency),
            "sensor_reading_sent": float(sensor_reading_sent),
            "controller_reading_received": float(controller_reading_received),
            "generation_command": float(generation_command),
            "load_demand_mw": float(load_demand_mw),
            "attack_active": attack_bool
        }
        self.records.append(record)

        self.writer.writerow([
            f"{wall_time:.4f}",
            f"{sim_time_sec:.2f}",
            f"{true_frequency:.4f}",
            f"{sensor_reading_sent:.4f}",
            f"{controller_reading_received:.4f}",
            f"{generation_command:.2f}",
            f"{load_demand_mw:.2f}",
            "True" if attack_bool else "False"
        ])
        self.file.flush()

    def to_dataframe(self):
        """Return the collected simulation data as a pandas DataFrame."""
        if self.records:
            return pd.DataFrame(self.records)
        elif os.path.exists(self.filename):
            return pd.read_csv(self.filename)
        else:
            return pd.DataFrame(columns=self.HEADER)

    def close(self):
        """Flush and close the CSV log file."""
        if self.file and not self.file.closed:
            self.file.flush()
            self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
