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
    SCALE_FACTOR, CONTROLLER_PERIOD_SEC, NOMINAL_FREQ,
    FREQ_LOW_THRESH, FREQ_HIGH_THRESH, RAMP_STEP,
    CONTROLLER_KP, CONTROLLER_KI, CONTROLLER_DEADBAND_HZ,
    TIMEOUT_MAX_CYCLES, INITIAL_GEN_SETPOINT
)


class GridControllerPLC(PLC):
    """Controller PLC implementing primary frequency control (Threshold or Droop+PI)."""

    def __init__(self, name="controller", state=STATE, protocol=CONTROLLER_PROTOCOL, mode="threshold"):
        self.current_setpoint = float(INITIAL_GEN_SETPOINT)
        self.mode = mode.lower()
        self.integral_error = 0.0
        self.missed_cycles = 0
        super(GridControllerPLC, self).__init__(
            name=name,
            state=state,
            protocol=protocol
        )

    def pre_loop(self, sleep=1.0):
        print(f"INFO [Controller PLC] Starting up in '{self.mode}' control mode...")
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
        print(f"INFO [Controller PLC] Enters main control loop (mode: {self.mode}).")
        while True:
            try:
                # 1. Poll frequency register HR 0 from Sensor
                raw_freq = self.receive(FREQ_HR_TAG, SENSOR_ADDR)
                
                if raw_freq is not None:
                    if self.missed_cycles >= TIMEOUT_MAX_CYCLES:
                        print(f"INFO [Controller PLC] Communication restored with sensor after {self.missed_cycles} missed cycles.")
                    self.missed_cycles = 0

                    if isinstance(raw_freq, list):
                        raw_freq = raw_freq[0]

                    frequency = float(raw_freq) / SCALE_FACTOR
                    self.set(CONTROLLER_FREQ_TAG, f"{frequency:.4f}")

                    # 2. Execute Control Law
                    if self.mode == "pi":
                        # Standard Primary Droop + Secondary PI Control
                        freq_error = NOMINAL_FREQ - frequency  # positive when grid is under-frequency
                        if abs(freq_error) > CONTROLLER_DEADBAND_HZ:
                            self.integral_error += freq_error * CONTROLLER_PERIOD_SEC
                            # Anti-windup clamping
                            self.integral_error = max(-30.0, min(30.0, self.integral_error))
                            p_act = CONTROLLER_KP * freq_error
                            i_act = CONTROLLER_KI * self.integral_error
                            self.current_setpoint = INITIAL_GEN_SETPOINT + p_act + i_act
                            action = f"PI_CONTROL (err={freq_error:+.3f}Hz, P={p_act:+.2f}MW, I={i_act:+.2f}MW)"
                        else:
                            action = "HOLD_DEADBAND"
                    else:
                        # Default Baseline #1: Threshold / Deadband Control
                        if frequency < FREQ_LOW_THRESH:
                            self.current_setpoint += RAMP_STEP
                            action = f"RAMP_UP (+{RAMP_STEP:.2f} MW)"
                        elif frequency > FREQ_HIGH_THRESH:
                            self.current_setpoint -= RAMP_STEP
                            action = f"RAMP_DOWN (-{RAMP_STEP:.2f} MW)"
                        else:
                            action = "HOLD"

                    # Prevent unreasonable bounds (e.g. between 10 MW and 300 MW)
                    self.current_setpoint = max(10.0, min(300.0, self.current_setpoint))

                    # 3. Write updated setpoint back to Modbus HR 1
                    scaled_setpoint = int(round(self.current_setpoint * SCALE_FACTOR))
                    self.send(GEN_SETPOINT_HR_TAG, scaled_setpoint, SENSOR_ADDR)

                    print(f"[Controller] Read f={frequency:.2f} Hz -> Action: {action} -> Setpoint: {self.current_setpoint:.2f} MW")

                else:
                    # Packet missed or unreadable
                    self._handle_timeout()

            except Exception as err:
                self._handle_timeout(err)

            time.sleep(CONTROLLER_PERIOD_SEC)

    def _handle_timeout(self, err=None):
        """Handle communication failure / packet timeout."""
        self.missed_cycles += 1
        elapsed_loss_sec = self.missed_cycles * CONTROLLER_PERIOD_SEC
        if self.missed_cycles == TIMEOUT_MAX_CYCLES:
            print(f"WARNING [Controller PLC] Communication timeout threshold reached! "
                  f"Missed {self.missed_cycles} consecutive polls ({elapsed_loss_sec:.1f}s). "
                  f"Sensor is unreachable (DoS/loss). Holding last setpoint: {self.current_setpoint:.2f} MW")
        elif self.missed_cycles > TIMEOUT_MAX_CYCLES and self.missed_cycles % 10 == 0:
            print(f"WARNING [Controller PLC] Ongoing communication blackout: {elapsed_loss_sec:.1f}s without sensor telemetry.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Power Grid Controller PLC")
    parser.add_argument("--mode", type=str, default="threshold", choices=["threshold", "pi"],
                        help="Control law: 'threshold' (naive baseline) or 'pi' (droop + secondary PI)")
    args = parser.parse_args()

    controller_plc = GridControllerPLC(
        name='controller',
        state=STATE,
        protocol=CONTROLLER_PROTOCOL,
        mode=args.mode
    )
