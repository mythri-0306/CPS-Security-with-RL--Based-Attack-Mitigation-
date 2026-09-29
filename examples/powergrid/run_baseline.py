"""
Power Grid Frequency Control CPS Testbed - run_baseline.py

Main orchestrator script:
1. Initializes SQLite database.
2. Boots Mininet network (PowerGridTopo).
3. Launches Sensor PLC on sensor host.
4. Launches Controller PLC on controller host.
5. Launches physical simulation loop.
6. Runs for specified duration and cleanly shuts down.
7. Automatically generates time-series plot via plots.py.
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
from utils import PATH, SCHEMA, SCHEMA_INIT
from minicps.states import SQLiteState
from plots import plot_simulation_run


class PowerGridCPS(MiniCPS):
    """MiniCPS Orchestrator for Power Grid Simulation."""

    def __init__(self, name, net, duration=300.0, run_label="baseline_run1",
                 interactive=False, no_disturbance=False):
        self.name = name
        self.net = net
        self.duration = float(duration)
        self.run_label = run_label
        self.interactive = interactive
        self.no_disturbance = no_disturbance

        os.makedirs("logs", exist_ok=True)

        print("\n=======================================================")
        print(f"  Starting Power Grid Simulation: {self.run_label} ({self.duration}s)")
        print("=======================================================")

        # 1. Start Mininet Network
        self.net.start()

        # 2. Test Reachability
        print("\n[Mininet] Verifying network connectivity across hosts:")
        self.net.pingAll()

        # 3. Get Host Nodes
        sensor = self.net.get('sensor')
        controller = self.net.get('controller')
        attacker = self.net.get('attacker')

        # 4. Launch Devices on Respective Hosts
        py_exec = sys.executable
        _REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        py_prefix = f"PYTHONPATH={_REPO_ROOT} "

        print("\n[Simulation] Launching Sensor PLC on host 'sensor'...")
        sensor_proc = sensor.popen(
            f"{py_prefix}{py_exec} -u sensor_device.py",
            shell=True,
            stdout=open("logs/sensor.log", "w"),
            stderr=subprocess.STDOUT
        )
        time.sleep(2.0)  # Allow Modbus server to bind

        print("[Simulation] Launching Controller PLC on host 'controller'...\n")
        controller_proc = controller.popen(
            f"{py_prefix}{py_exec} -u controller_device.py",
            shell=True,
            stdout=open("logs/controller.log", "w"),
            stderr=subprocess.STDOUT
        )
        time.sleep(1.0)

        # 5. Launch Physics Process
        dist_flag = "--no-disturbance" if self.no_disturbance else ""
        print(f"[Simulation] Starting physical process loop ({self.duration}s, label: {self.run_label})...")
        physics_cmd = f"{py_prefix}{py_exec} -u physics.py --duration {self.duration} --run-label {self.run_label} {dist_flag}"
        physics_proc = sensor.popen(
            physics_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        if self.interactive:
            print("\n[Mininet] Interactive CLI active. Type 'exit' to terminate simulation.")
            CLI(self.net)
        else:
            print(f"[Simulation] Live simulation output ({self.duration}s total):")
            with open("logs/physics.log", "w") as f_log:
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

        # 6. Clean Teardown
        print("\n[Simulation] Tearing down processes...")
        try:
            sensor_proc.kill()
            controller_proc.kill()
            physics_proc.kill()
        except Exception:
            pass

        self.net.stop()
        print("[Mininet] Network stopped successfully.")

        # 7. Generate Visualization Plot
        csv_file = os.path.join("logs", f"{self.run_label}.csv")
        if os.path.exists(csv_file):
            print(f"\n[Plots] Generating visualization for {csv_file}...")
            try:
                png_path = plot_simulation_run(csv_file, run_label=self.run_label)
                print(f"[Plots] Saved plot to: {png_path}")
            except Exception as e:
                print(f"[Plots] Error generating plot: {e}")


def init_database():
    """Ensure clean database setup before running."""
    if os.path.exists(PATH):
        try:
            os.remove(PATH)
        except Exception:
            pass
    try:
        SQLiteState._create(PATH, SCHEMA)
        SQLiteState._init(PATH, SCHEMA_INIT)
        print(f"[Database] Initialized state database '{PATH}'.")
    except Exception as err:
        print(f"[Database] Warning initializing database: {err}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Power Grid Frequency Control Simulation")
    parser.add_argument("--duration", type=float, default=300.0, help="Duration in seconds (default: 300)")
    parser.add_argument("--run-label", type=str, default="baseline_run1", help="Label for run (default: baseline_run1)")
    parser.add_argument("--cli", action="store_true", help="Launch interactive Mininet CLI")
    parser.add_argument("--no-disturbance", action="store_true", help="Disable step load disturbance at t=10s")
    args = parser.parse_args()

    # Re-initialize DB
    init_database()

    # Build topology and network
    topo = PowerGridTopo()
    net = Mininet(topo=topo, switch=OVSBridge, controller=None)

    cps = PowerGridCPS(
        name="powergrid",
        net=net,
        duration=args.duration,
        run_label=args.run_label,
        interactive=args.cli,
        no_disturbance=args.no_disturbance
    )
