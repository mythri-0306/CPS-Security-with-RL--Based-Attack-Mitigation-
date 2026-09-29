"""
Power Grid Frequency Control CPS Testbed - sensor_device.py

MiniCPS PLC definition for the Sensor host:
- Runs Modbus/TCP server on port 502 exposing holding registers.
- Synchronizes physical grid frequency from statedb to Modbus HR 0 (scaled by 100).
- Synchronizes controller generation setpoint from Modbus HR 1 (scaled by 100) to statedb.
"""

import time
import sys
import argparse
from minicps.devices import PLC
from utils import (
    STATE, SENSOR_PROTOCOL, SENSOR_ADDR,
    FREQ_TAG, GEN_SETPOINT_TAG, SENSOR_FREQ_TAG,
    FREQ_HR_TAG, GEN_SETPOINT_HR_TAG,
    SCALE_FACTOR, SENSOR_PERIOD_SEC,
    NOMINAL_FREQ, INITIAL_GEN_SETPOINT
)


class GridSensorPLC(PLC):
    """Sensor PLC bridging physical simulation state with Modbus/TCP server."""

    def pre_loop(self, sleep=0.5):
        print("INFO [Sensor PLC] Starting up Modbus/TCP Server...")
        time.sleep(sleep)
        # Initialize Modbus registers
        init_scaled_freq = int(round(NOMINAL_FREQ * SCALE_FACTOR))
        init_scaled_setpoint = int(round(INITIAL_GEN_SETPOINT * SCALE_FACTOR))
        try:
            self.send(FREQ_HR_TAG, init_scaled_freq, SENSOR_ADDR)
            self.send(GEN_SETPOINT_HR_TAG, init_scaled_setpoint, SENSOR_ADDR)
            print(f"INFO [Sensor PLC] Modbus registers initialized: HR0={init_scaled_freq}, HR1={init_scaled_setpoint}")
        except Exception as err:
            print(f"WARNING [Sensor PLC] Initial register write warning: {err}")

    def main_loop(self):
        """Continuously bridge state database and Modbus registers."""
        print("INFO [Sensor PLC] Enters main loop.")
        while True:
            try:
                # 1. Read true physical frequency from state DB and update Modbus register 0
                freq_str = self.get(FREQ_TAG)
                if freq_str is not None:
                    freq_val = float(freq_str)
                    scaled_freq = int(round(freq_val * SCALE_FACTOR))
                    # Clamp to 16-bit unsigned integer range
                    scaled_freq = max(0, min(65535, scaled_freq))
                    self.send(FREQ_HR_TAG, scaled_freq, SENSOR_ADDR)
                    self.set(SENSOR_FREQ_TAG, f"{scaled_freq / SCALE_FACTOR:.4f}")

                # 2. Read latest generation setpoint from Modbus register 1 and update state DB
                scaled_setpoint = self.receive(GEN_SETPOINT_HR_TAG, SENSOR_ADDR)
                if scaled_setpoint is not None:
                    if isinstance(scaled_setpoint, list):
                        scaled_setpoint = scaled_setpoint[0]
                    p_gen_val = float(scaled_setpoint) / SCALE_FACTOR
                    if p_gen_val > 0:
                        self.set(GEN_SETPOINT_TAG, f"{p_gen_val:.2f}")

            except Exception as e:
                # Print debug message if communication glitch occurs
                pass

            time.sleep(SENSOR_PERIOD_SEC)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Power Grid Sensor PLC")
    parser.parse_args()

    sensor_plc = GridSensorPLC(
        name='sensor',
        state=STATE,
        protocol=SENSOR_PROTOCOL
    )
