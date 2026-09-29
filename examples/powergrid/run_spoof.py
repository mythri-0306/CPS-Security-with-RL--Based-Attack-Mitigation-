"""
Power Grid Frequency Control CPS Testbed - run_spoof.py

Orchestrates the spoof attack scenario:
1.  Initialise SQLite database.
2.  Boot Mininet network (PowerGridTopo).
3.  Launch Sensor PLC on sensor host.
4.  Launch Controller PLC on controller host.
5.  Launch attacks/spoof.py on ATTACKER host — overwrites HR 0 with a
    fake frequency value from t=60s to t=180s.
6.  Launch physics.py on sensor host (ground truth + logger).
7.  Wait for physics to complete (300s total).
8.  Tear down all processes and Mininet network.
9.  Auto-generate CSV + PNG via plots.py.

The attack flag (ATTACK_ACTIVE) is written to the shared SQLite DB by
attacks/spoof.py, picked up by physics.py each timestep, and forwarded
to logger.py → CSV → plots.py without any modification to those files.
"""

import sys
import os
import time
import argparse
import subprocess

from mininet.node import OVSBridge
from mininet.net import Mininet
from mininet.cli import CLI
from minicps.mcps import MiniCPS

from topo import PowerGridTopo
from utils import PATH, SCHEMA, SCHEMA_INIT, IP, MODBUS_PORT
from minicps.states import SQLiteState
from plots import plot_simulation_run

# Attack scenario defaults — edit here or via CLI flags
ATTACK_START_SEC    = 60.0    # attack window opens at t=60s
ATTACK_DURATION_SEC = 120.0   # attack active for 120s  (closes at t=180s)
FAKE_FREQ_HZ        = 48.50   # injected fake frequency (Hz)
SENSOR_IP           = IP['sensor']


class PowerGridCPS_Spoof(MiniCPS):
    """MiniCPS Orchestrator for the value-spoofing attack scenario."""

    def __init__(self, name, net, duration=300.0, run_label="spoof_run1",
                 interactive=False, no_disturbance=False,
                 attack_start=ATTACK_START_SEC,
                 attack_duration=ATTACK_DURATION_SEC,
                 fake_freq=FAKE_FREQ_HZ):

        self.name = name
        self.net = net
        self.duration = float(duration)
        self.run_label = run_label
        self.interactive = interactive
        self.no_disturbance = no_disturbance
        self.attack_start = attack_start
        self.attack_duration = attack_duration
        self.fake_freq = fake_freq

        os.makedirs("logs", exist_ok=True)

        print("\n=======================================================")
        print(f"  Power Grid CPS — SPOOF ATTACK SCENARIO")
        print(f"  Run label : {self.run_label}  ({self.duration}s)")
        print(f"  Fake freq : {self.fake_freq:.2f} Hz")
        print(f"  Window    : t={self.attack_start:.0f}s → "
              f"t={self.attack_start + self.attack_duration:.0f}s")
        print("=======================================================\n")

        # 1. Start Mininet
        self.net.start()

        # 2. Connectivity check
        print("[Mininet] Verifying network connectivity...")
        self.net.pingAll()

        # 3. Get host handles
        sensor     = self.net.get('sensor')
        controller = self.net.get('controller')
        attacker   = self.net.get('attacker')

        py_exec = sys.executable

        _REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        py_prefix = f"PYTHONPATH={_REPO_ROOT} "

        # 4. Launch Sensor PLC
        print("[Simulation] Launching Sensor PLC on 'sensor'...")
        sensor_proc = sensor.popen(
            f"{py_prefix}{py_exec} -u sensor_device.py",
            shell=True,
            stdout=open("logs/sensor_spoof.log", "w"),
            stderr=subprocess.STDOUT
        )
        time.sleep(2.0)   # Allow Modbus server to bind

        # 5. Launch Controller PLC
        print("[Simulation] Launching Controller PLC on 'controller'...")
        controller_proc = controller.popen(
            f"{py_prefix}{py_exec} -u controller_device.py",
            shell=True,
            stdout=open("logs/controller_spoof.log", "w"),
            stderr=subprocess.STDOUT
        )
        time.sleep(1.0)

        # 6. Launch spoof attack on attacker host
        print(f"[Attack] Launching spoof.py on attacker host "
              f"(window t={self.attack_start:.0f}s→"
              f"t={self.attack_start + self.attack_duration:.0f}s, "
              f"fake={self.fake_freq:.2f} Hz)...")
        attack_cmd = (
            f"{py_prefix}{py_exec} -u attacks/spoof.py "
            f"--sensor-ip {SENSOR_IP} "
            f"--fake-freq {self.fake_freq} "
            f"--attack-start {self.attack_start} "
            f"--attack-duration {self.attack_duration} "
            f"--db-path {PATH}"
        )
        attack_proc = attacker.popen(
            attack_cmd,
            shell=True,
            stdout=open("logs/attack_spoof.log", "w"),
            stderr=subprocess.STDOUT
        )

        # 7. Launch physics / logger
        dist_flag = "--no-disturbance" if self.no_disturbance else ""
        physics_cmd = (
            f"{py_prefix}{py_exec} -u physics.py "
            f"--duration {self.duration} "
            f"--run-label {self.run_label} {dist_flag}"
        )
        print(f"[Simulation] Starting physics loop for {self.duration}s...")
        physics_proc = sensor.popen(
            physics_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        if self.interactive:
            print("\n[Mininet] Interactive CLI — type 'exit' to stop.")
            CLI(self.net)
        else:
            print(f"[Simulation] Live simulation output ({self.duration}s total):")
            with open("logs/physics_spoof.log", "w") as f_log:
                try:
                    for line in iter(physics_proc.stdout.readline, ''):
                        if not line:
                            break
                        print(line, end="", flush=True)
                        f_log.write(line)
                        f_log.flush()
                except KeyboardInterrupt:
                    print("\n[Simulation] Interrupted by user.")
            physics_proc.wait()

        # 8. Clean teardown
        print("\n[Simulation] Tearing down processes...")
        for proc in (sensor_proc, controller_proc, attack_proc, physics_proc):
            try:
                proc.kill()
            except Exception:
                pass

        self.net.stop()
        print("[Mininet] Network stopped.")

        # 9. Plot
        csv_file = os.path.join("logs", f"{self.run_label}.csv")
        if os.path.exists(csv_file):
            print(f"\n[Plots] Generating plot for {csv_file} ...")
            try:
                png_path = plot_simulation_run(csv_file, run_label=self.run_label)
                print(f"[Plots] Saved to: {png_path}")
            except Exception as exc:
                import traceback
                print(f"[Plots] Error generating plot: {exc}")
                traceback.print_exc()
        else:
            print(f"[Plots] WARNING: CSV file '{csv_file}' was not generated.")


def init_database():
    """Ensure a clean database before running."""
    if os.path.exists(PATH):
        try:
            os.remove(PATH)
        except Exception:
            pass
    try:
        SQLiteState._create(PATH, SCHEMA)
        SQLiteState._init(PATH, SCHEMA_INIT)
        print(f"[Database] Initialised '{PATH}'.")
    except Exception as err:
        print(f"[Database] Warning: {err}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Power Grid CPS — Spoof Attack Scenario"
    )
    parser.add_argument("--duration",          type=float, default=300.0)
    parser.add_argument("--run-label",         type=str,   default="spoof_run1")
    parser.add_argument("--cli",               action="store_true")
    parser.add_argument("--no-disturbance",    action="store_true")
    parser.add_argument("--attack-start",      type=float, default=ATTACK_START_SEC)
    parser.add_argument("--attack-duration",   type=float, default=ATTACK_DURATION_SEC)
    parser.add_argument("--fake-freq",         type=float, default=FAKE_FREQ_HZ)
    args = parser.parse_args()

    init_database()

    topo = PowerGridTopo()
    net  = Mininet(topo=topo, switch=OVSBridge, controller=None)

    PowerGridCPS_Spoof(
        name="powergrid_spoof",
        net=net,
        duration=args.duration,
        run_label=args.run_label,
        interactive=args.cli,
        no_disturbance=args.no_disturbance,
        attack_start=args.attack_start,
        attack_duration=args.attack_duration,
        fake_freq=args.fake_freq,
    )
