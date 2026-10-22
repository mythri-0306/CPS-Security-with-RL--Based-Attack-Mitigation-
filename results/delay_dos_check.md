# Phase 5 Investigation: Delay vs DoS Telemetry & Mechanism Verification

## 1. Executive Summary & Verdict

**Verdict:** `delay` and `dos` in this benchmark share **one identical underlying mechanism**:
- Both attacks freeze telemetry to the last known valid sensor measurement and increment `packet_age` on every step of the attack window.
- When executed under identical PRNG seeds and identical background load steps, **50 out of 50 episodes (100.0%) produce bitwise identical telemetry streams**, identical packet ages, and identical physical frequency dynamics.
- Statistically, the telemetry streams are completely indistinguishable under identical seeds.
- **Policy:** Per project specification, `delay` and `dos` are kept separate in tables and evaluation scripts (to preserve compatibility with Phase 1-5 benchmarks), but are annotated with an explicit note that they represent the same stale-telemetry phenomenon.

## 2. Empirical Verification Across 50 Episodes (Identical Seeds 20000..20049)

- **Bitwise Identical Episodes:** 50 / 50 (100.0%)
- **Mean Packet Age During Attack:**
  - `delay`: 100.50 steps (Max: 200.0 steps)
  - `dos`  : 100.50 steps (Max: 200.0 steps)
- **Stale Step Fraction During Attack:**
  - `delay`: 100.00%
  - `dos`  : 100.00%
- **Stale-Run Length Distribution (steps per contiguous blackout run):**
  - `delay`: {200: 50}
  - `dos`  : {200: 50}

### First 15 Measured Frequencies Post-Attack Onset (Seed 20000):
- `delay`: [49.93, 49.93, 49.93, 49.93, 49.93, 49.93, 49.93, 49.93, 49.93, 49.93, 49.93, 49.93, 49.93, 49.93, 49.93]
- `dos`  : [49.93, 49.93, 49.93, 49.93, 49.93, 49.93, 49.93, 49.93, 49.93, 49.93, 49.93, 49.93, 49.93, 49.93, 49.93]

## 3. Code Inspection (`grid_env.py` lines 365–369)

```python
elif self.attack_type in ("delay", "dos"):
    # Latency blackout or packet loss holds last known valid telemetry
    self.current_measured_freq = float(self.last_valid_measured_freq)
    self.packet_age += 1  # No fresh measurement: age increments
```

Neither attack uses its configured intensity parameter during execution (`step()`):
- `delay` configured range `(400.0, 1200.0) ms` does not buffer/delay packets by $N$ steps.
- `dos` configured range `(0.70, 0.95)` does not probabilistically drop packets.
Both execute an unconditional blackout for the entire 20.0s (`attack_duration`).

## 4. Evaluation Impact & Reporting Note

- In earlier Phase 5 evaluation, seeds 20000..20099 were reused across all attack types, causing delay and dos to receive identical numbers.
- In Phase 5b, seed partitioning (`delay`: 20400–20499; `dos`: 20500–20599) provided distinct background step-load sequences, leading to slight scenario noise (2.36s vs 2.83s Always-Trust OOB).
- Per STEP 1 instructions, `final_summary.md` has been updated with a clear note explaining that delay and dos are reported separately but share one stale-telemetry mechanism.
