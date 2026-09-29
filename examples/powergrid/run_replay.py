"""
Power Grid Frequency Control CPS Testbed - run_replay.py

Orchestrates the stale-value replay attack scenario:
1.  Initialise SQLite database.
2.  Boot Mininet (PowerGridTopo).
3.  Launch Sensor PLC on sensor host.
4.  Launch Controller PLC on controller host.
5.  Launch attacks/replay.py on ATTACKER host.
       - Record phase  : t=5s → t=55s  (FC 3 read-only, passive)
       - Attack phase  : t=60s → t=180s (FC 6 write stale values)
6.  Launch physics.py on sensor host.
7.  Wait for completion, tear down, auto-plot.
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

# Attack scenario defaults
RECORD_START_SEC    = 5.0
RECORD_DURATION_SEC = 50.0    # record t=5→55s
ATTACK_START_SEC    = 60.0    # replay window opens at t=60s
ATTACK_DURATION_SEC = 120.0   # active t=60→180s
SENSOR_IP           = IP['sensor']


class PowerGridCPS_Replay(MiniCPS):
    """MiniCPS Orchestrator for the stale-value replay attack scenario."""

    def __init__(self, name, net, duration=300.0, run_label="replay_run1",
                 interactive=False, no_disturbance=False,
                 record_start=RECORD_START_SEC,
                 record_duration=RECORD_DURATION_SEC,
                 attack_start=ATTACK_START_SEC,
                 attack_duration=ATTACK_DURATION_SEC):

        self.name = name
        self.net = net
        self.duration = float(duration)
        self.run_label = run_label
        self.interactive = interactive
        self.no_disturbance = no_disturbance
        self.record_start = record_start
        self.record_duration = record_duration
        self.attack_start = attack_start
        self.attack_duration = attack_duration

        os.makedirs("logs", exist_ok=True)

        print("\n=======================================================")
        print(f"  Power Grid CPS — REPLAY ATTACK SCENARIO")
        print(f"  Run label  : {self.run_label}  ({self.duration}s)")
        print(f"  Record     : t={self.record_start:.0f}s → "
              f"t={self.record_start + self.record_duration:.0f}s")
        print(f"  Attack     : t={self.attack_start:.0f}s → "
              f"t={self.attack_start + self.attack_duration:.0f}s")
        print("=======================================================\n")

        self.net.start()
        print("[Mininet] Verifying network connectivity...")
        self.net.pingAll()

        sensor     = self.net.get('sensor')
        controller = self.net.get('controller')
        attacker   = self.net.get('attacker')
        py_exec    = sys.executable

        _REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        py_prefix = f"PYTHONPATH={_REPO_ROOT} "

        # Sensor PLC
        print("[Simulation] Launching Sensor PLC...")
        sensor_proc = sensor.popen(
            f"{py_prefix}{py_exec} -u sensor_device.py",
            shell=True,
            stdout=open("logs/sensor_replay.log", "w"),
            stderr=subprocess.STDOUT
        )
        time.sleep(2.0)

        # Controller PLC
        print("[Simulation] Launching Controller PLC...")
        controller_proc = controller.popen(
            f"{py_prefix}{py_exec} -u controller_device.py",
            shell=True,
            stdout=open("logs/controller_replay.log", "w"),
            stderr=subprocess.STDOUT
        )
        time.sleep(1.0)

        # Replay attack — launched immediately so it can enter its record phase
        print(f"[Attack] Launching replay.py on attacker host...")
        attack_cmd = (
            f"{py_prefix}{py_exec} -u attacks/replay.py "
            f"--sensor-ip {SENSOR_IP} "
            f"--record-start {self.record_start} "
            f"--record-duration {self.record_duration} "
            f"--attack-start {self.attack_start} "
            f"--attack-duration {self.attack_duration} "
            f"--db-path {PATH}"
        )
        attack_proc = attacker.popen(
            attack_cmd,
            shell=True,
            stdout=open("logs/attack_replay.log", "w"),
            stderr=subprocess.STDOUT
        )

        # Physics / logger
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
            with open("logs/physics_replay.log", "w") as f_log:
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

        print("\n[Simulation] Tearing down...")
        for proc in (sensor_proc, controller_proc, attack_proc, physics_proc):
            try:
                proc.kill()
            except Exception:
                pass

        self.net.stop()
        print("[Mininet] Network stopped.")

        csv_file = os.path.join("logs", f"{self.run_label}.csv")
        if os.path.exists(csv_file):
            print(f"\n[Plots] Generating plot for {csv_file}...")
            try:
                png_path = plot_simulation_run(csv_file, run_label=self.run_label)
                print(f"[Plots] Saved to: {png_path}")
            except Exception as exc:
                print(f"[Plots] Error: {exc}")


def init_database():
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
        description="Run Power Grid CPS — Replay Attack Scenario"
    )
    parser.add_argument("--duration",         type=float, default=300.0)
    parser.add_argument("--run-label",        type=str,   default="replay_run1")
    parser.add_argument("--cli",              action="store_true")
    parser.add_argument("--no-disturbance",   action="store_true")
    parser.add_argument("--record-start",     type=float, default=RECORD_START_SEC)
    parser.add_argument("--record-duration",  type=float, default=RECORD_DURATION_SEC)
    parser.add_argument("--attack-start",     type=float, default=ATTACK_START_SEC)
    parser.add_argument("--attack-duration",  type=float, default=ATTACK_DURATION_SEC)
    args = parser.parse_args()

    init_database()

    topo = PowerGridTopo()
    net  = Mininet(topo=topo, switch=OVSBridge, controller=None)

    PowerGridCPS_Replay(
        name="powergrid_replay",
        net=net,
        duration=args.duration,
        run_label=args.run_label,
        interactive=args.cli,
        no_disturbance=args.no_disturbance,
        record_start=args.record_start,
        record_duration=args.record_duration,
        attack_start=args.attack_start,
        attack_duration=args.attack_duration,
    )
