"""
attacks/delay.py — tc/netem Link-Latency Injection Attack

Mechanism
---------
Uses the Linux tc (traffic control) subsystem with the netem (Network
Emulator) qdisc to impose configurable artificial latency on the sensor
host's Ethernet interface for a configured attack window, then removes
the qdisc afterward.

No Modbus client code is used. This is standard Linux kernel traffic
shaping (tc/netem) applied to a Mininet virtual interface to demonstrate
timing-integrity vulnerabilities in ICS polling architectures.

This script MUST run inside the sensor host's network namespace so that
the `tc` command targets the correct virtual interface. The run script
launches it via `sensor.popen(...)`.

Academic framing (for writeup)
-------------------------------
Modbus/TCP defines no timing-integrity mechanism: there are no sequence
numbers, message timestamps, or maximum-age fields that allow a receiver
to detect stale readings. By imposing link latency, this script causes the
controller to receive delayed frequency data. Because the controller cannot
distinguish delayed data from live data, it responds to the past system
state — a documented weakness of polling-based ICS protocols
(IEC 62351-7 threat taxonomy, category: "timeliness violation").

Known gap (not patched per Phase 3 constraints)
------------------------------------------------
controller_device.py silences all receive() exceptions with `except: pass`
(lines 80-82). Under high latency the Modbus client may time out, at which
point the controller silently holds its last setpoint. This is
indistinguishable from "frequency within safe band" — the controller has no
liveness/heartbeat mechanism. This secondary finding should be documented
in the writeup as an architectural gap in the existing controller design.

Usage (launched inside sensor namespace via run_delay.py):
    python attacks/delay.py \\
        --interface    sensor-eth0 \\
        --delay-ms     800 \\
        --attack-start  60 \\
        --attack-duration 120 \\
        [--db-path powergrid_db.sqlite]
"""

import sys
import os
import time
import argparse
import subprocess
import sqlite3


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
        print(f"[delay] WARNING: ATTACK_ACTIVE update failed: {exc}", file=sys.stderr)


# ──────────────────────────────────────────────────────────────────────────────
# tc/netem helpers
# ──────────────────────────────────────────────────────────────────────────────

def _tc_add_delay(interface: str, delay_ms: int) -> bool:
    """Apply a netem delay qdisc on *interface*. Returns True on success."""
    cmd = [
        'tc', 'qdisc', 'add', 'dev', interface,
        'root', 'netem', 'delay', f'{delay_ms}ms'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[delay] WARNING: tc add failed (rc={result.returncode}): "
              f"{result.stderr.strip()}", file=sys.stderr)
        return False
    print(f"[delay] tc netem delay {delay_ms}ms applied on {interface}")
    return True


def _tc_del_delay(interface: str) -> None:
    """Remove the root qdisc from *interface*."""
    cmd = ['tc', 'qdisc', 'del', 'dev', interface, 'root']
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[delay] WARNING: tc del failed (rc={result.returncode}): "
              f"{result.stderr.strip()}", file=sys.stderr)
    else:
        print(f"[delay] Removed netem qdisc from {interface}")


# ──────────────────────────────────────────────────────────────────────────────
# Attack logic
# ──────────────────────────────────────────────────────────────────────────────

def run_delay(
    interface: str,
    delay_ms: int,
    attack_start: float,
    attack_duration: float,
    db_path: str,
) -> None:
    """Sleep → apply tc delay → hold window → clean up."""

    print(f"[delay] ── Configuration ─────────────────────────────────────")
    print(f"[delay]   Interface    : {interface}")
    print(f"[delay]   Delay        : {delay_ms} ms")
    print(f"[delay]   Attack window: t={attack_start:.1f}s → "
          f"t={attack_start + attack_duration:.1f}s  "
          f"(duration {attack_duration:.0f}s)")
    print(f"[delay]   State DB     : {db_path}")
    print(f"[delay] ─────────────────────────────────────────────────────")

    # ── Pre-attack sleep ─────────────────────────────────────────────────────
    t0 = time.monotonic()
    wait = attack_start - (time.monotonic() - t0)
    if wait > 0:
        print(f"[delay] Waiting {wait:.1f}s before attack window opens...")
        time.sleep(wait)

    # ── Apply netem delay ────────────────────────────────────────────────────
    tc_ok = _tc_add_delay(interface, delay_ms)
    _set_attack_flag(db_path, '1')
    print(f"[delay] ATTACK_ACTIVE = True  ← attack window open")

    # ── Hold for attack duration ─────────────────────────────────────────────
    try:
        time.sleep(attack_duration)
    finally:
        # ── Cleanup ──────────────────────────────────────────────────────────
        if tc_ok:
            _tc_del_delay(interface)
        _set_attack_flag(db_path, '0')
        print(f"[delay] ATTACK_ACTIVE = False ← attack window closed")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='tc/netem link-latency attack — Power Grid CPS Testbed Phase 3',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--interface',
                        default='sensor-eth0',
                        help='Network interface inside sensor namespace to shape')
    parser.add_argument('--delay-ms',
                        type=int, default=800,
                        help='Artificial one-way link delay to apply (ms)')
    parser.add_argument('--attack-start',
                        type=float, default=60.0,
                        help='Seconds after script launch to open attack window')
    parser.add_argument('--attack-duration',
                        type=float, default=120.0,
                        help='Duration of active attack window (s)')
    parser.add_argument('--db-path',
                        default='powergrid_db.sqlite',
                        help='Path to shared SQLite state database')
    args = parser.parse_args()

    run_delay(
        interface=args.interface,
        delay_ms=args.delay_ms,
        attack_start=args.attack_start,
        attack_duration=args.attack_duration,
        db_path=args.db_path,
    )
