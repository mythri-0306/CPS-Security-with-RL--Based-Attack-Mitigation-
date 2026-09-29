"""
Power Grid Frequency Control CPS Testbed - run_delay.py

Orchestrates the tc/netem link-latency attack scenario:
1.  Initialise SQLite database.
2.  Boot Mininet (PowerGridTopo).
3.  Launch Sensor PLC on sensor host.
4.  Launch Controller PLC on controller host.
5.  Determine the sensor host's virtual Ethernet interface name.
6.  Launch attacks/delay.py INSIDE THE SENSOR HOST's network namespace
    (via sensor.popen) so it can apply tc on the sensor's own interface.
7.  Launch physics.py on sensor host.
8.  Wait, tear down, auto-plot.

Why sensor.popen for delay.py / dos.py?
    tc qdisc operates on a specific network interface inside a Linux
    network namespace.  In Mininet each host lives in its own namespace.
    Running the delay script from within the sensor host's namespace
    (sensor.popen) lets it apply `tc` to sensor-eth0 without requiring
    cross-namespace plumbing.  The ATTACK_ACTIVE SQLite flag still
    coordinates the timing signal to physics.py/logger.py unchanged.
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

ATTACK_START_SEC    = 60.0
ATTACK_DURATION_SEC = 120.0
DELAY_MS            = 800      # 800 ms one-way delay — well above Modbus timeout


class PowerGridCPS_Delay(MiniCPS):
    """MiniCPS Orchestrator for the tc/netem latency-injection attack."""

    def __init__(self, name, net, duration=300.0, run_label="delay_run1",
                 interactive=False, no_disturbance=False,
                 attack_start=ATTACK_START_SEC,
                 attack_duration=ATTACK_DURATION_SEC,
                 delay_ms=DELAY_MS):

        self.name = name
        self.net = net
        self.duration = float(duration)
        self.run_label = run_label
        self.interactive = interactive
        self.no_disturbance = no_disturbance
        self.attack_start = attack_start
        self.attack_duration = attack_duration
        self.delay_ms = delay_ms

        os.makedirs("logs", exist_ok=True)

        print("\n=======================================================")
        print(f"  Power Grid CPS — LINK-DELAY ATTACK SCENARIO")
        print(f"  Run label  : {self.run_label}  ({self.duration}s)")
        print(f"  Delay      : {self.delay_ms} ms")
        print(f"  Window     : t={self.attack_start:.0f}s → "
              f"t={self.attack_start + self.attack_duration:.0f}s")
        print("=======================================================\n")

        self.net.start()
        print("[Mininet] Verifying network connectivity...")
        self.net.pingAll()

        sensor     = self.net.get('sensor')
        controller = self.net.get('controller')
        py_exec    = sys.executable
        _REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        py_prefix = f"PYTHONPATH={_REPO_ROOT} "

        # Sensor PLC
        print("[Simulation] Launching Sensor PLC...")
        sensor_proc = sensor.popen(
            f"{py_prefix}{py_exec} -u sensor_device.py",
            shell=True,
            stdout=open("logs/sensor_delay.log", "w"),
            stderr=subprocess.STDOUT
        )
        time.sleep(2.0)

        # Controller PLC
        print("[Simulation] Launching Controller PLC...")
        controller_proc = controller.popen(
            f"{py_prefix}{py_exec} -u controller_device.py",
            shell=True,
            stdout=open("logs/controller_delay.log", "w"),
            stderr=subprocess.STDOUT
        )
        time.sleep(1.0)

        # Determine sensor's veth interface name inside its namespace
        # sensor.intf() returns the first interface (the veth connected to s1)
        sensor_iface = sensor.intf().name
        print(f"[Attack] Sensor interface inside Mininet namespace: {sensor_iface}")

        # Launch delay.py INSIDE sensor namespace via sensor.popen
        # so that the tc command targets the correct interface
        print(f"[Attack] Launching delay.py inside sensor namespace "
              f"(delay={self.delay_ms}ms, window={self.attack_start:.0f}s→"
              f"{self.attack_start + self.attack_duration:.0f}s)...")
        attack_cmd = (
            f"{py_prefix}{py_exec} -u attacks/delay.py "
            f"--interface {sensor_iface} "
            f"--delay-ms {self.delay_ms} "
            f"--attack-start {self.attack_start} "
            f"--attack-duration {self.attack_duration} "
            f"--db-path {PATH}"
        )
        attack_proc = sensor.popen(
            attack_cmd,
            shell=True,
            stdout=open("logs/attack_delay.log", "w"),
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
            with open("logs/physics_delay.log", "w") as f_log:
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
        # Ensure tc qdisc is cleaned up even if attack_proc already exited
        sensor.cmd(f"tc qdisc del dev {sensor_iface} root 2>/dev/null || true")
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
        description="Run Power Grid CPS — Link-Delay Attack Scenario"
    )
    parser.add_argument("--duration",         type=float, default=300.0)
    parser.add_argument("--run-label",        type=str,   default="delay_run1")
    parser.add_argument("--cli",              action="store_true")
    parser.add_argument("--no-disturbance",   action="store_true")
    parser.add_argument("--attack-start",     type=float, default=ATTACK_START_SEC)
    parser.add_argument("--attack-duration",  type=float, default=ATTACK_DURATION_SEC)
    parser.add_argument("--delay-ms",         type=int,   default=DELAY_MS)
    args = parser.parse_args()

    init_database()

    topo = PowerGridTopo()
    net  = Mininet(topo=topo, switch=OVSBridge, controller=None)

    PowerGridCPS_Delay(
        name="powergrid_delay",
        net=net,
        duration=args.duration,
        run_label=args.run_label,
        interactive=args.cli,
        no_disturbance=args.no_disturbance,
        attack_start=args.attack_start,
        attack_duration=args.attack_duration,
        delay_ms=args.delay_ms,
    )
