"""
Power Grid Frequency Control CPS Testbed - controller_device.py

MiniCPS PLC definition for the Controller host:
- Runs Modbus/TCP client polling frequency register (HR 0) from Sensor host.
- Applies threshold control rule:
    if f < 49.8 Hz: ramp generation up (setpoint += RAMP_STEP)
    elif f > 50.2 Hz: ramp generation down (setpoint -= RAMP_STEP)
    else: hold setpoint
- Writes updated generation setpoint back to Modbus HR 1 on Sensor host.
"""

import time
import sys
import argparse
from minicps.devices import PLC
from utils import (
    STATE, CONTROLLER_PROTOCOL, SENSOR_ADDR,
    FREQ_HR_TAG, GEN_SETPOINT_HR_TAG, CONTROLLER_FREQ_TAG,
    SCALE_FACTOR, CONTROLLER_PERIOD_SEC,
    FREQ_LOW_THRESH, FREQ_HIGH_THRESH, RAMP_STEP,
    INITIAL_GEN_SETPOINT
)


class GridControllerPLC(PLC):
    """Controller PLC implementing threshold-based primary frequency control."""

    def __init__(self, name="controller", state=STATE, protocol=CONTROLLER_PROTOCOL):
        self.current_setpoint = float(INITIAL_GEN_SETPOINT)
        super(GridControllerPLC, self).__init__(
            name=name,
            state=state,
            protocol=protocol
        )

    def pre_loop(self, sleep=1.0):
        print("INFO [Controller PLC] Waiting for sensor server to be ready...")
        time.sleep(sleep)
        # Write initial setpoint
        scaled_setpoint = int(round(self.current_setpoint * SCALE_FACTOR))
        try:
            self.send(GEN_SETPOINT_HR_TAG, scaled_setpoint, SENSOR_ADDR)
            print(f"INFO [Controller PLC] Initial setpoint sent: {self.current_setpoint:.2f} MW (HR1={scaled_setpoint})")
        except Exception as e:
            print(f"WARNING [Controller PLC] Initial send warning: {e}")

    def main_loop(self):
        """Periodic control execution loop."""
        print("INFO [Controller PLC] Enters main control loop.")
        while True:
            try:
                # 1. Poll frequency register HR 0 from Sensor
                raw_freq = self.receive(FREQ_HR_TAG, SENSOR_ADDR)
                if raw_freq is not None:
                    if isinstance(raw_freq, list):
                        raw_freq = raw_freq[0]

                    frequency = float(raw_freq) / SCALE_FACTOR
                    self.set(CONTROLLER_FREQ_TAG, f"{frequency:.4f}")

                    # 2. Apply threshold control logic
                    action = "HOLD"
                    if frequency < FREQ_LOW_THRESH:
                        self.current_setpoint += RAMP_STEP
                        action = f"RAMP_UP (+{RAMP_STEP} MW)"
                    elif frequency > FREQ_HIGH_THRESH:
                        self.current_setpoint -= RAMP_STEP
                        action = f"RAMP_DOWN (-{RAMP_STEP} MW)"

                    # Prevent unreasonable bounds (e.g. between 10 MW and 300 MW)
                    self.current_setpoint = max(10.0, min(300.0, self.current_setpoint))

                    # 3. Write updated setpoint back to Modbus HR 1
                    scaled_setpoint = int(round(self.current_setpoint * SCALE_FACTOR))
                    self.send(GEN_SETPOINT_HR_TAG, scaled_setpoint, SENSOR_ADDR)

                    print(f"[Controller] Read f={frequency:.2f} Hz -> Action: {action} -> Setpoint: {self.current_setpoint:.2f} MW")

            except Exception as err:
                # Ignore transient network startup errors
                pass

            time.sleep(CONTROLLER_PERIOD_SEC)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Power Grid Controller PLC")
    parser.parse_args()

    controller_plc = GridControllerPLC(
        name='controller',
        state=STATE,
        protocol=CONTROLLER_PROTOCOL
    )
