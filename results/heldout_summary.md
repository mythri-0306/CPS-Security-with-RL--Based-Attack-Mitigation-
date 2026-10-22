# Phase 5 Held-Out Attack Evaluation Summary (Generalization Test)

Evaluated across **100 episodes per attack type** using out-of-distribution attack parameters (later start times 32-42s, longer 25s durations, and unseen intensity ranges).

> [!NOTE]
> **Telemetry Freeze (`delay` and `dos`) Clarification:** Both `telemetry_freeze (delay)` and `telemetry_freeze (dos)` run the exact same stale-hold code path in `grid_env.py` (freezing sensor measurements to `last_valid_measured_freq` and incrementing `packet_age`). The configured latency (ms) and drop-rate parameters are not used in `step()`, and the held-out delay/dos ranges therefore tested nothing. Both columns are reported side-by-side to maintain historical benchmark alignment.

## Evaluated Defense Architectures
1. **Always-Trust (No Defense)**: Baseline passing raw sensor measurements directly to controller.
2. **Rule-Based Defense (Mythri)**: Static threshold clamping [49.80, 50.20] Hz + rate-of-change jump rejection (>0.20 Hz).
3. **Kalman-Only Defense (Mythri)**: Linearized swing-equation innovation thresholding (>0.15 Hz) with physics fallback.
4. **Detector-Threshold-Full**: Hand-set heuristic observing 4 features: residual, age, duplicate, and rolling variance (tuned on CKPT_EVAL seeds).
5. **Detector-Threshold-Timing**: Hand-set heuristic observing stale-telemetry (age, duplicate) and residual ONLY (no variance, K=0, tuned on CKPT_EVAL).
6. **Detector-Threshold-Timing-Hold**: Stale-telemetry + residual with anti-flapping latch hold K=3 steps (tuned on CKPT_EVAL).
7. **RL Defender (DQN R2_costly)**: Deep Q-Network trained with composite OOB checkpoint selection.
8. **PPO Defender (R2_costly)**: Proximal Policy Optimization trained on R2_costly with composite OOB checkpoint selection.
9. **Oracle (Causal)**: Upper-bound baseline with perfect 1-step causal knowledge of attack state.
10. **Always-Distrust**: Baseline rejecting all measurements and running purely on estimator state.

---

## Comprehensive Performance Tables by Metric (Mean ± Std)

### Time Outside Safe Band (s)

| Method | `none` | `spoof` | `stealthy_drift` | `replay` | `telemetry_freeze (delay)` | `telemetry_freeze (dos)` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Always-Trust (No Defense)** | 0.00 ± 0.00 | 23.16 ± 2.90 | 19.76 ± 2.83 | 0.00 ± 0.00 | 0.58 ± 2.06 | 0.35 ± 1.64 |
| **Rule-Based (Mythri)** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.59 ± 2.09 | 0.35 ± 1.67 |
| **Kalman-Only (Mythri)** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.01 ± 0.05 | 0.02 ± 0.10 |
| **Detector-Threshold-Full** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| **Detector-Threshold-Timing** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| **Detector-Threshold-Timing-Hold** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| **RL (DQN R2, 5-Seed Avg)** | 0.00 ± 0.00 | 7.89 ± 10.74 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| **Oracle (Causal)** | 0.00 ± 0.00 | 0.56 ± 0.37 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |

### Return per Episode (under R2 weights: W_BAND=10.0, W_DISTRUST=1.0)

| Method | `none` | `spoof` | `stealthy_drift` | `replay` | `telemetry_freeze (delay)` | `telemetry_freeze (dos)` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Always-Trust (No Defense)** | 0.00 ± 0.00 | -2315.90 ± 289.90 | -1975.60 ± 283.08 | 0.00 ± 0.00 | -58.10 ± 205.74 | -34.60 ± 164.21 |
| **Rule-Based (Mythri)** | 0.00 ± 0.00 | -0.72 ± 0.98 | -0.62 ± 0.97 | -0.02 ± 0.14 | -59.28 ± 209.84 | -35.37 ± 167.40 |
| **Kalman-Only (Mythri)** | -0.47 ± 0.72 | -1.14 ± 1.59 | -0.82 ± 1.31 | -0.18 ± 0.50 | -0.81 ± 5.04 | -2.16 ± 9.97 |
| **Detector-Threshold-Full** | -81.78 ± 11.44 | -35.45 ± 8.89 | -34.60 ± 7.93 | -33.23 ± 8.71 | -35.87 ± 9.28 | -35.43 ± 9.74 |
| **Detector-Threshold-Timing** | -4.78 ± 2.57 | -3.53 ± 2.08 | -3.49 ± 2.15 | -3.11 ± 1.90 | -3.70 ± 2.19 | -3.94 ± 2.16 |
| **Detector-Threshold-Timing-Hold** | -17.87 ± 8.20 | -14.15 ± 8.75 | -13.36 ± 8.55 | -11.52 ± 6.67 | -13.81 ± 8.48 | -14.87 ± 8.54 |
| **RL (DQN R2, 5-Seed Avg)** | -2.25 ± 3.41 | -790.56 ± 1073.87 | -1.24 ± 1.84 | -0.65 ± 0.98 | -1.30 ± 1.86 | -1.29 ± 1.81 |
| **Oracle (Causal)** | 0.00 ± 0.00 | -56.40 ± 37.06 | -0.59 ± 0.91 | -0.56 ± 0.90 | -0.53 ± 0.88 | -0.53 ± 0.88 |

### r_band Penalty Total (from OOB Steps: -(10.0 * OOB_steps))

| Method | `none` | `spoof` | `stealthy_drift` | `replay` | `telemetry_freeze (delay)` | `telemetry_freeze (dos)` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Always-Trust (No Defense)** | 0.00 ± 0.00 | -2315.90 ± 289.90 | -1975.60 ± 283.08 | 0.00 ± 0.00 | -58.10 ± 205.74 | -34.60 ± 164.21 |
| **Rule-Based (Mythri)** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | -58.90 ± 208.65 | -35.10 ± 166.51 |
| **Kalman-Only (Mythri)** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | -0.50 ± 5.00 | -2.00 ± 9.85 |
| **Detector-Threshold-Full** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| **Detector-Threshold-Timing** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| **Detector-Threshold-Timing-Hold** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| **RL (DQN R2, 5-Seed Avg)** | 0.00 ± 0.00 | -789.48 ± 1074.06 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| **Oracle (Causal)** | 0.00 ± 0.00 | -55.70 ± 37.18 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |

### r_distrust Penalty Total (from Needless Distrust: -(1.0 * clean_distrust_steps))

| Method | `none` | `spoof` | `stealthy_drift` | `replay` | `telemetry_freeze (delay)` | `telemetry_freeze (dos)` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Always-Trust (No Defense)** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| **Rule-Based (Mythri)** | 0.00 ± 0.00 | -0.72 ± 0.98 | -0.62 ± 0.97 | -0.02 ± 0.14 | -0.38 ± 1.52 | -0.27 ± 1.28 |
| **Kalman-Only (Mythri)** | -0.47 ± 0.72 | -1.14 ± 1.59 | -0.82 ± 1.31 | -0.18 ± 0.50 | -0.31 ± 0.87 | -0.16 ± 0.37 |
| **Detector-Threshold-Full** | -81.78 ± 11.44 | -35.45 ± 8.89 | -34.60 ± 7.93 | -33.23 ± 8.71 | -35.87 ± 9.28 | -35.43 ± 9.74 |
| **Detector-Threshold-Timing** | -4.78 ± 2.57 | -3.53 ± 2.08 | -3.49 ± 2.15 | -3.11 ± 1.90 | -3.70 ± 2.19 | -3.94 ± 2.16 |
| **Detector-Threshold-Timing-Hold** | -17.87 ± 8.20 | -14.15 ± 8.75 | -13.36 ± 8.55 | -11.52 ± 6.67 | -13.81 ± 8.48 | -14.87 ± 8.54 |
| **RL (DQN R2, 5-Seed Avg)** | -2.25 ± 3.41 | -1.08 ± 1.52 | -1.24 ± 1.84 | -0.65 ± 0.98 | -1.30 ± 1.86 | -1.29 ± 1.81 |
| **Oracle (Causal)** | 0.00 ± 0.00 | -0.70 ± 0.96 | -0.59 ± 0.91 | -0.56 ± 0.90 | -0.53 ± 0.88 | -0.53 ± 0.88 |

### Clean Step Distrust Rate (%) [fraction of clean steps where action==1]

| Method | `none` | `spoof` | `stealthy_drift` | `replay` | `telemetry_freeze (delay)` | `telemetry_freeze (dos)` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Always-Trust (No Defense)** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| **Rule-Based (Mythri)** | 0.00 ± 0.00 | 0.11 ± 0.14 | 0.09 ± 0.15 | 0.01 ± 0.04 | 0.11 ± 0.43 | 0.08 ± 0.37 |
| **Kalman-Only (Mythri)** | 0.08 ± 0.12 | 0.22 ± 0.36 | 0.15 ± 0.28 | 0.05 ± 0.14 | 0.07 ± 0.21 | 0.03 ± 0.09 |
| **Detector-Threshold-Full** | 13.63 ± 1.91 | 9.50 ± 2.27 | 9.26 ± 2.02 | 8.88 ± 2.16 | 9.47 ± 2.53 | 9.45 ± 2.57 |
| **Detector-Threshold-Timing** | 0.80 ± 0.43 | 0.86 ± 0.54 | 0.85 ± 0.55 | 0.83 ± 0.50 | 0.91 ± 0.56 | 0.99 ± 0.56 |
| **Detector-Threshold-Timing-Hold** | 2.98 ± 1.37 | 3.79 ± 2.43 | 3.52 ± 2.29 | 3.09 ± 1.81 | 3.63 ± 2.30 | 3.96 ± 2.33 |
| **RL (DQN R2, 5-Seed Avg)** | 0.38 ± 0.57 | 0.21 ± 0.39 | 0.26 ± 0.47 | 0.17 ± 0.26 | 0.28 ± 0.46 | 0.28 ± 0.46 |
| **Oracle (Causal)** | 0.00 ± 0.00 | 0.10 ± 0.14 | 0.08 ± 0.13 | 0.08 ± 0.13 | 0.07 ± 0.13 | 0.07 ± 0.13 |

### False Alarm Rate on Clean Episodes (%) [episodes of attack_type=='none']

| Method | `none` | `spoof` | `stealthy_drift` | `replay` | `telemetry_freeze (delay)` | `telemetry_freeze (dos)` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Always-Trust (No Defense)** | 0.00 ± 0.00% | — (see `none`) | — (see `none`) | — (see `none`) | — (see `none`) | — (see `none`) |
| **Rule-Based (Mythri)** | 0.00 ± 0.00% | — (see `none`) | — (see `none`) | — (see `none`) | — (see `none`) | — (see `none`) |
| **Kalman-Only (Mythri)** | 0.08 ± 0.12% | — (see `none`) | — (see `none`) | — (see `none`) | — (see `none`) | — (see `none`) |
| **Detector-Threshold-Full** | 13.63 ± 1.91% | — (see `none`) | — (see `none`) | — (see `none`) | — (see `none`) | — (see `none`) |
| **Detector-Threshold-Timing** | 0.80 ± 0.43% | — (see `none`) | — (see `none`) | — (see `none`) | — (see `none`) | — (see `none`) |
| **Detector-Threshold-Timing-Hold** | 2.98 ± 1.37% | — (see `none`) | — (see `none`) | — (see `none`) | — (see `none`) | — (see `none`) |
| **RL (DQN R2, 5-Seed Avg)** | 0.38 ± 0.57% | — (see `none`) | — (see `none`) | — (see `none`) | — (see `none`) | — (see `none`) |
| **Oracle (Causal)** | 0.00 ± 0.00% | — (see `none`) | — (see `none`) | — (see `none`) | — (see `none`) | — (see `none`) |

### Distrust Recall (%)

| Method | `none` | `spoof` | `stealthy_drift` | `replay` | `telemetry_freeze (delay)` | `telemetry_freeze (dos)` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Always-Trust (No Defense)** | 100.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| **Rule-Based (Mythri)** | 100.00 ± 0.00 | 100.00 ± 0.00 | 92.07 ± 1.64 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| **Kalman-Only (Mythri)** | 100.00 ± 0.00 | 100.00 ± 0.00 | 93.40 ± 1.29 | 0.01 ± 0.07 | 2.70 ± 9.87 | 1.70 ± 8.38 |
| **Detector-Threshold-Full** | 100.00 ± 0.00 | 100.00 ± 0.00 | 96.99 ± 0.57 | 4.16 ± 1.39 | 99.63 ± 0.19 | 99.64 ± 0.18 |
| **Detector-Threshold-Timing** | 100.00 ± 0.00 | 100.00 ± 0.00 | 92.21 ± 1.54 | 0.83 ± 0.71 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| **Detector-Threshold-Timing-Hold** | 100.00 ± 0.00 | 100.00 ± 0.00 | 92.53 ± 1.52 | 2.86 ± 2.24 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| **RL (DQN R2, 5-Seed Avg)** | 100.00 ± 0.00 | 84.12 ± 23.51 | 91.60 ± 3.56 | 0.08 ± 0.22 | 73.44 ± 29.91 | 73.61 ± 30.67 |
| **Oracle (Causal)** | 100.00 ± 0.00 | 99.56 ± 0.05 | 99.55 ± 0.05 | 99.55 ± 0.05 | 99.54 ± 0.05 | 99.55 ± 0.04 |

### Attack Success Rate (%)

| Method | `none` | `spoof` | `stealthy_drift` | `replay` | `telemetry_freeze (delay)` | `telemetry_freeze (dos)` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Always-Trust (No Defense)** | 0.0% | 100.0% | 100.0% | 0.0% | 8.0% | 3.0% |
| **Rule-Based (Mythri)** | 0.0% | 0.0% | 0.0% | 0.0% | 8.0% | 3.0% |
| **Kalman-Only (Mythri)** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **Detector-Threshold-Full** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **Detector-Threshold-Timing** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **Detector-Threshold-Timing-Hold** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **RL (DQN R2, 5-Seed Avg)** | 0.0% | 41.4% | 0.0% | 0.0% | 0.0% | 0.0% |
| **Oracle (Causal)** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

### Recovery Time (s)

| Method | `none` | `spoof` | `stealthy_drift` | `replay` | `telemetry_freeze (delay)` | `telemetry_freeze (dos)` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Always-Trust (No Defense)** | 0.00 ± 0.00 | 19.61 ± 1.75 | 18.68 ± 1.80 | 0.00 ± 0.00 | 0.58 ± 2.06 | 0.35 ± 1.64 |
| **Rule-Based (Mythri)** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.59 ± 2.09 | 0.35 ± 1.67 |
| **Kalman-Only (Mythri)** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.01 ± 0.05 | 0.02 ± 0.10 |
| **Detector-Threshold-Full** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| **Detector-Threshold-Timing** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| **Detector-Threshold-Timing-Hold** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| **RL (DQN R2, 5-Seed Avg)** | 0.00 ± 0.00 | 6.80 ± 9.05 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| **Oracle (Causal)** | 0.00 ± 0.00 | 0.56 ± 0.37 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |

### Max Frequency Deviation (Hz)

| Method | `none` | `spoof` | `stealthy_drift` | `replay` | `telemetry_freeze (delay)` | `telemetry_freeze (dos)` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Always-Trust (No Defense)** | 0.08 ± 0.00 | 5.56 ± 0.20 | 3.91 ± 0.50 | 0.09 ± 0.01 | 0.12 ± 0.06 | 0.12 ± 0.06 |
| **Rule-Based (Mythri)** | 0.08 ± 0.00 | 0.14 ± 0.00 | 0.13 ± 0.00 | 0.09 ± 0.01 | 0.12 ± 0.06 | 0.12 ± 0.06 |
| **Kalman-Only (Mythri)** | 0.08 ± 0.00 | 0.14 ± 0.00 | 0.14 ± 0.00 | 0.09 ± 0.01 | 0.11 ± 0.04 | 0.11 ± 0.04 |
| **Detector-Threshold-Full** | 0.09 ± 0.01 | 0.14 ± 0.00 | 0.14 ± 0.00 | 0.10 ± 0.01 | 0.14 ± 0.00 | 0.14 ± 0.00 |
| **Detector-Threshold-Timing** | 0.08 ± 0.00 | 0.14 ± 0.00 | 0.14 ± 0.00 | 0.09 ± 0.01 | 0.14 ± 0.00 | 0.14 ± 0.00 |
| **Detector-Threshold-Timing-Hold** | 0.08 ± 0.00 | 0.14 ± 0.00 | 0.14 ± 0.00 | 0.09 ± 0.01 | 0.14 ± 0.00 | 0.14 ± 0.00 |
| **RL (DQN R2, 5-Seed Avg)** | 0.08 ± 0.00 | 1.73 ± 2.14 | 0.14 ± 0.01 | 0.09 ± 0.01 | 0.13 ± 0.02 | 0.13 ± 0.02 |
| **Oracle (Causal)** | 0.08 ± 0.00 | 0.23 ± 0.03 | 0.14 ± 0.00 | 0.14 ± 0.00 | 0.14 ± 0.00 | 0.14 ± 0.00 |

### Detection Delay (steps)

| Method | `none` | `spoof` | `stealthy_drift` | `replay` | `telemetry_freeze (delay)` | `telemetry_freeze (dos)` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Always-Trust (No Defense)** | MISSED (0/100) | MISSED (100/100) | MISSED (100/100) | MISSED (100/100) | MISSED (100/100) | MISSED (100/100) |
| **Rule-Based (Mythri)** | MISSED (0/100) | 0.0 ± 0.0 (0 miss) | 17.7 ± 2.7 (0 miss) | MISSED (100/100) | MISSED (100/100) | MISSED (100/100) |
| **Kalman-Only (Mythri)** | MISSED (0/100) | 0.0 ± 0.0 (0 miss) | 14.8 ± 2.4 (0 miss) | 18.5 ± 16.3 (98 miss) | 168.7 ± 39.5 (91 miss) | 183.9 ± 62.6 (92 miss) |
| **Detector-Threshold-Full** | MISSED (0/100) | 0.0 ± 0.0 (0 miss) | 6.8 ± 1.2 (0 miss) | 23.5 ± 21.3 (0 miss) | 0.8 ± 0.4 (0 miss) | 0.8 ± 0.4 (0 miss) |
| **Detector-Threshold-Timing** | MISSED (0/100) | 0.0 ± 0.0 (0 miss) | 16.7 ± 3.1 (0 miss) | 81.4 ± 63.3 (18 miss) | 0.0 ± 0.0 (0 miss) | 0.0 ± 0.0 (0 miss) |
| **Detector-Threshold-Timing-Hold** | MISSED (0/100) | 0.0 ± 0.0 (0 miss) | 16.6 ± 3.5 (0 miss) | 82.0 ± 62.7 (18 miss) | 0.0 ± 0.0 (0 miss) | 0.0 ± 0.0 (0 miss) |
| **RL (DQN R2, 5-Seed Avg)** | MISSED (0/500) | 0.9 ± 2.3 (0 miss) | 18.7 ± 7.5 (0 miss) | 73.3 ± 56.0 (435 miss) | 35.0 ± 43.9 (18 miss) | 34.0 ± 43.6 (25 miss) |
| **Oracle (Causal)** | MISSED (0/100) | 1.0 ± 0.0 (0 miss) | 1.0 ± 0.0 (0 miss) | 1.0 ± 0.0 (0 miss) | 1.0 ± 0.0 (0 miss) | 1.0 ± 0.0 (0 miss) |

### Distrust Precision (%)

| Method | `none` | `spoof` | `stealthy_drift` | `replay` | `telemetry_freeze (delay)` | `telemetry_freeze (dos)` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Always-Trust (No Defense)** | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| **Rule-Based (Mythri)** | 100.00 ± 0.00 | 99.85 ± 0.20 | 99.86 ± 0.23 | 98.00 ± 14.07 | 88.00 ± 32.66 | 88.00 ± 32.66 |
| **Kalman-Only (Mythri)** | 65.00 ± 47.94 | 99.67 ± 0.52 | 99.77 ± 0.41 | 86.33 ± 34.20 | 87.87 ± 32.62 | 88.00 ± 32.66 |
| **Detector-Threshold-Full** | 0.00 ± 0.00 | 86.47 ± 3.47 | 86.31 ± 3.41 | 21.85 ± 6.20 | 85.94 ± 3.50 | 86.43 ± 3.61 |
| **Detector-Threshold-Timing** | 1.00 ± 10.00 | 98.62 ± 0.84 | 98.47 ± 1.05 | 37.15 ± 21.35 | 98.46 ± 0.94 | 98.40 ± 0.91 |
| **Detector-Threshold-Timing-Hold** | 1.00 ± 10.00 | 94.43 ± 3.13 | 94.19 ± 3.65 | 36.37 ± 21.66 | 94.33 ± 3.20 | 94.05 ± 3.13 |
| **RL (DQN R2, 5-Seed Avg)** | 50.40 ± 50.05 | 99.53 ± 0.81 | 99.56 ± 0.76 | 65.07 ± 45.37 | 99.21 ± 4.69 | 98.76 ± 8.16 |
| **Oracle (Causal)** | 100.00 ± 0.00 | 99.86 ± 0.19 | 99.88 ± 0.18 | 99.89 ± 0.18 | 99.90 ± 0.18 | 99.90 ± 0.18 |

---

## Return Decomposition & Sanity Check

Verifying exact penalty accounting under R2 weights: $\text{Return} = - (W_{\text{BAND}} \times \text{OOB steps}) - (W_{\text{DISTRUST}} \times \text{clean distrust steps})$:

| Method Group | Mean Return | r_band Mean | r_distrust Mean | Decomposed Sum | Discrepancy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Always-Trust (No Defense)** | -730.70 | -730.70 | 0.00 | -730.70 | 0.0000 |
| **Rule-Based (Mythri)** | -16.00 | -15.67 | -0.34 | -16.00 | 0.0000 |
| **Kalman-Only (Mythri)** | -0.93 | -0.42 | -0.51 | -0.93 | 0.0000 |
| **Detector-Threshold-Full** | -42.73 | 0.00 | -42.73 | -42.73 | 0.0000 |
| **Detector-Threshold-Timing** | -3.76 | 0.00 | -3.76 | -3.76 | 0.0000 |
| **Detector-Threshold-Timing-Hold** | -14.26 | 0.00 | -14.26 | -14.26 | 0.0000 |
| **RL (DQN R2, 5-Seed Avg)** | -132.88 | -131.58 | -1.30 | -132.88 | 0.0000 |
| **Oracle (Causal)** | -9.77 | -9.28 | -0.48 | -9.77 | 0.0000 |

---

## RL Defender: Per-Seed Breakdown (Mean OOB Steps in Seconds)

| RL Seed | `none` | `spoof` | `stealthy_drift` | `replay` | `telemetry_freeze (delay)` | `telemetry_freeze (dos)` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seed 0** | 0.00s | 0.00s | 0.00s | 0.00s | 0.00s | 0.00s |
| **Seed 1** | 0.00s | 23.16s | 0.00s | 0.00s | 0.00s | 0.00s |
| **Seed 2** | 0.00s | 0.22s | 0.00s | 0.00s | 0.00s | 0.00s |
| **Seed 3** | 0.00s | 14.32s | 0.00s | 0.00s | 0.00s | 0.00s |
| **Seed 4** | 0.00s | 1.78s | 0.00s | 0.00s | 0.00s | 0.00s |

---

## Statistical Paired Comparisons: Difference in Time Outside Safe Band (s)

### (a) Pooled Episodes Comparison

| Comparison | Attack Type | Mean Difference (s) | 95% Bootstrap CI (s) | Verdict |
| :--- | :--- | :---: | :---: | :--- |
| RL vs Kalman-Only | `none` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Kalman-Only | `spoof` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Kalman-Only | `stealthy_drift` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Kalman-Only | `replay` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Kalman-Only | `delay` | -0.005 | [-0.015, +0.000] | 95% CI includes zero |
| RL vs Kalman-Only | `dos` | -0.020 | [-0.040, -0.005] | 95% CI excludes zero (RL lower OOB) |
| RL vs Detector-Threshold-Full | `none` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Detector-Threshold-Full | `spoof` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Detector-Threshold-Full | `stealthy_drift` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Detector-Threshold-Full | `replay` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Detector-Threshold-Full | `delay` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Detector-Threshold-Full | `dos` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Detector-Threshold-Timing | `none` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Detector-Threshold-Timing | `spoof` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Detector-Threshold-Timing | `stealthy_drift` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Detector-Threshold-Timing | `replay` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Detector-Threshold-Timing | `delay` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Detector-Threshold-Timing | `dos` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Detector-Threshold-Timing-Hold | `none` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Detector-Threshold-Timing-Hold | `spoof` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Detector-Threshold-Timing-Hold | `stealthy_drift` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Detector-Threshold-Timing-Hold | `replay` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Detector-Threshold-Timing-Hold | `delay` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Detector-Threshold-Timing-Hold | `dos` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |

### (b) Per-Seed Paired Comparison (over 5 RL Seeds)

| Comparison | Attack Type | Mean Difference across Seeds (s) | 95% Bootstrap CI over Seeds (s) | Verdict |
| :--- | :--- | :---: | :---: | :--- |
| RL vs Kalman-Only | `none` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Kalman-Only | `spoof` | +7.895 | [+0.399, +16.759] | 95% CI excludes zero (Baseline lower OOB) |
| RL vs Kalman-Only | `stealthy_drift` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Kalman-Only | `replay` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Kalman-Only | `delay` | -0.005 | [-0.005, -0.005] | 95% CI excludes zero (RL lower OOB) |
| RL vs Kalman-Only | `dos` | -0.020 | [-0.020, -0.020] | 95% CI excludes zero (RL lower OOB) |
| RL vs Detector-Threshold-Full | `none` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Detector-Threshold-Full | `spoof` | +7.895 | [+0.399, +16.759] | 95% CI excludes zero (Baseline lower OOB) |
| RL vs Detector-Threshold-Full | `stealthy_drift` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Detector-Threshold-Full | `replay` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Detector-Threshold-Full | `delay` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Detector-Threshold-Full | `dos` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Detector-Threshold-Timing | `none` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Detector-Threshold-Timing | `spoof` | +7.895 | [+0.399, +16.759] | 95% CI excludes zero (Baseline lower OOB) |
| RL vs Detector-Threshold-Timing | `stealthy_drift` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Detector-Threshold-Timing | `replay` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Detector-Threshold-Timing | `delay` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Detector-Threshold-Timing | `dos` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Detector-Threshold-Timing-Hold | `none` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Detector-Threshold-Timing-Hold | `spoof` | +7.895 | [+0.399, +16.759] | 95% CI excludes zero (Baseline lower OOB) |
| RL vs Detector-Threshold-Timing-Hold | `stealthy_drift` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Detector-Threshold-Timing-Hold | `replay` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Detector-Threshold-Timing-Hold | `delay` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Detector-Threshold-Timing-Hold | `dos` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
