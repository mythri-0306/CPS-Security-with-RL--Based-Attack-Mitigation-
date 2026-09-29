"""
Power Grid Frequency Control CPS Testbed - run_dos.py

Orchestrates the tc/netem packet-loss DoS attack scenario:
1.  Initialise SQLite database.
2.  Boot Mininet (PowerGridTopo).
3.  Launch Sensor PLC on sensor host.
4.  Launch Controller PLC on controller host.
5.  Determine sensor's veth interface name.
6.  Launch attacks/dos.py INSIDE THE SENSOR HOST's network namespace
    (sensor.popen), passing the sensor process PID for SIGSTOP fallback.
7.  Launch physics.py on sensor host.
8.  Wait, tear down, auto-plot.

Known gap documented (not patched per Phase 3 constraints)
----------------------------------------------------------
When packet loss is ~90%, the controller's Modbus read() calls fail with
an exception, caught silently by the `except: pass` in controller_device.py
(L80-82).  The controller then holds its last setpoint indefinitely — it
cannot distinguish "sensor unreachable" from "frequency in-band, no action".
This architectural gap is flagged in attacks/dos.py and should be discussed
in the writeup.
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
LOSS_PCT            = 90       # Drop 90% of packets → near-total blackout


class PowerGridCPS_DoS(MiniCPS):
    """MiniCPS Orchestrator for the tc/netem packet-loss DoS attack."""

    def __init__(self, name, net, duration=300.0, run_label="dos_run1",
                 interactive=False, no_disturbance=False,
                 attack_start=ATTACK_START_SEC,
                 attack_duration=ATTACK_DURATION_SEC,
                 loss_pct=LOSS_PCT):

        self.name = name
        self.net = net
        self.duration = float(duration)
        self.run_label = run_label
        self.interactive = interactive
        self.no_disturbance = no_disturbance
        self.attack_start = attack_start
        self.attack_duration = attack_duration
        self.loss_pct = loss_pct

        os.makedirs("logs", exist_ok=True)

        print("\n=======================================================")
        print(f"  Power Grid CPS — PACKET-LOSS DoS ATTACK SCENARIO")
        print(f"  Run label  : {self.run_label}  ({self.duration}s)")
        print(f"  Loss       : {self.loss_pct}%")
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
            stdout=open("logs/sensor_dos.log", "w"),
            stderr=subprocess.STDOUT
        )
        time.sleep(2.0)
        # Capture PID for SIGSTOP fallback in dos.py
        sensor_pid = sensor_proc.pid

        # Controller PLC
        print("[Simulation] Launching Controller PLC...")
        controller_proc = controller.popen(
            f"{py_prefix}{py_exec} -u controller_device.py",
            shell=True,
            stdout=open("logs/controller_dos.log", "w"),
            stderr=subprocess.STDOUT
        )
        time.sleep(1.0)

        # Determine sensor veth name
        sensor_iface = sensor.intf().name
        print(f"[Attack] Sensor interface inside Mininet namespace: {sensor_iface}")

        # Launch dos.py inside sensor namespace
        print(f"[Attack] Launching dos.py inside sensor namespace "
              f"(loss={self.loss_pct}%, sensor_pid={sensor_pid}, "
              f"window={self.attack_start:.0f}s→"
              f"{self.attack_start + self.attack_duration:.0f}s)...")
        attack_cmd = (
            f"{py_prefix}{py_exec} -u attacks/dos.py "
            f"--interface {sensor_iface} "
            f"--loss-pct {self.loss_pct} "
            f"--attack-start {self.attack_start} "
            f"--attack-duration {self.attack_duration} "
            f"--sensor-pid {sensor_pid} "
            f"--db-path {PATH}"
        )
        attack_proc = sensor.popen(
            attack_cmd,
            shell=True,
            stdout=open("logs/attack_dos.log", "w"),
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
            with open("logs/physics_dos.log", "w") as f_log:
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
        # Belt-and-suspenders: remove tc qdisc if attack script exited early
        sensor.cmd(f"tc qdisc del dev {sensor_iface} root 2>/dev/null || true")
        # Resume sensor if SIGSTOP was left in place
        try:
            import signal as _sig
            sensor_proc.send_signal(_sig.SIGCONT)
        except Exception:
            pass
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
        description="Run Power Grid CPS — Packet-Loss DoS Attack Scenario"
    )
    parser.add_argument("--duration",         type=float, default=300.0)
    parser.add_argument("--run-label",        type=str,   default="dos_run1")
    parser.add_argument("--cli",              action="store_true")
    parser.add_argument("--no-disturbance",   action="store_true")
    parser.add_argument("--attack-start",     type=float, default=ATTACK_START_SEC)
    parser.add_argument("--attack-duration",  type=float, default=ATTACK_DURATION_SEC)
    parser.add_argument("--loss-pct",         type=int,   default=LOSS_PCT)
    args = parser.parse_args()

    init_database()

    topo = PowerGridTopo()
    net  = Mininet(topo=topo, switch=OVSBridge, controller=None)

    PowerGridCPS_DoS(
        name="powergrid_dos",
        net=net,
        duration=args.duration,
        run_label=args.run_label,
        interactive=args.cli,
        no_disturbance=args.no_disturbance,
        attack_start=args.attack_start,
        attack_duration=args.attack_duration,
        loss_pct=args.loss_pct,
    )
