# Phase 5 Final Evaluation & Benchmark Summary

Evaluated across **100 episodes per attack type** using strictly partitioned, held-out seeds from `SEED_SETS['FINAL_EVAL']`.

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
| **Always-Trust (No Defense)** | 0.00 ± 0.00 | 39.21 ± 5.82 | 26.11 ± 7.60 | 0.00 ± 0.00 | 2.36 ± 4.43 | 2.83 ± 4.96 |
| **Rule-Based (Mythri)** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.07 ± 0.31 | 0.00 ± 0.00 | 2.51 ± 4.77 | 3.08 ± 5.39 |
| **Kalman-Only (Mythri)** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.02 ± 0.10 | 0.00 ± 0.00 | 2.26 ± 4.79 | 3.00 ± 5.39 |
| **Detector-Threshold-Full** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| **Detector-Threshold-Timing** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| **Detector-Threshold-Timing-Hold** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| **RL (DQN R2, 5-Seed Avg)** | 0.00 ± 0.00 | 3.12 ± 8.53 | 0.23 ± 0.63 | 0.00 ± 0.00 | 0.16 ± 0.93 | 0.22 ± 1.11 |
| **PPO (R2, 3-Seed Avg)** | 0.00 ± 0.00 | 13.65 ± 17.34 | 7.36 ± 10.50 | 0.00 ± 0.00 | 0.33 ± 1.87 | 0.50 ± 2.20 |
| **Oracle (Causal)** | 0.00 ± 0.00 | 0.03 ± 0.12 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| **Always-Distrust** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |

### Return per Episode (under R2 weights: W_BAND=10.0, W_DISTRUST=1.0)

| Method | `none` | `spoof` | `stealthy_drift` | `replay` | `telemetry_freeze (delay)` | `telemetry_freeze (dos)` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Always-Trust (No Defense)** | 0.00 ± 0.00 | -3920.60 ± 581.63 | -2611.30 ± 759.57 | 0.00 ± 0.00 | -235.60 ± 442.70 | -282.90 ± 495.82 |
| **Rule-Based (Mythri)** | -0.02 ± 0.14 | -2.09 ± 0.29 | -9.23 ± 31.14 | -0.21 ± 0.46 | -254.13 ± 482.17 | -311.85 ± 545.08 |
| **Kalman-Only (Mythri)** | -0.52 ± 0.67 | -2.58 ± 1.66 | -4.79 ± 10.73 | -1.23 ± 1.25 | -229.62 ± 483.71 | -304.24 ± 543.82 |
| **Detector-Threshold-Full** | -86.43 ± 22.31 | -118.93 ± 47.11 | -126.86 ± 62.51 | -86.02 ± 43.82 | -112.76 ± 53.07 | -112.89 ± 48.91 |
| **Detector-Threshold-Timing** | -4.70 ± 2.39 | -4.95 ± 2.21 | -5.17 ± 2.42 | -4.09 ± 2.46 | -5.08 ± 2.01 | -5.47 ± 2.15 |
| **Detector-Threshold-Timing-Hold** | -17.71 ± 8.80 | -25.86 ± 9.79 | -27.04 ± 13.55 | -14.98 ± 8.35 | -26.90 ± 11.54 | -27.74 ± 11.16 |
| **RL (DQN R2, 5-Seed Avg)** | -2.39 ± 3.54 | -321.24 ± 876.31 | -25.83 ± 63.06 | -1.65 ± 2.16 | -19.82 ± 94.12 | -25.70 ± 112.08 |
| **PPO (R2, 3-Seed Avg)** | -0.01 ± 0.08 | -1365.54 ± 1733.77 | -737.39 ± 1049.37 | -0.02 ± 0.13 | -33.55 ± 187.08 | -50.67 ± 219.70 |
| **Oracle (Causal)** | 0.00 ± 0.00 | -5.20 ± 12.38 | -2.00 ± 0.00 | -2.00 ± 0.00 | -2.00 ± 0.00 | -2.00 ± 0.00 |
| **Always-Distrust** | -600.00 ± 0.00 | -400.00 ± 0.00 | -400.00 ± 0.00 | -400.00 ± 0.00 | -400.00 ± 0.00 | -400.00 ± 0.00 |

### r_band Penalty Total (from OOB Steps: -(10.0 * OOB_steps))

| Method | `none` | `spoof` | `stealthy_drift` | `replay` | `telemetry_freeze (delay)` | `telemetry_freeze (dos)` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Always-Trust (No Defense)** | 0.00 ± 0.00 | -3920.60 ± 581.63 | -2611.30 ± 759.57 | 0.00 ± 0.00 | -235.60 ± 442.70 | -282.90 ± 495.82 |
| **Rule-Based (Mythri)** | 0.00 ± 0.00 | 0.00 ± 0.00 | -7.20 ± 31.14 | 0.00 ± 0.00 | -251.40 ± 477.01 | -307.90 ± 538.82 |
| **Kalman-Only (Mythri)** | 0.00 ± 0.00 | 0.00 ± 0.00 | -1.80 ± 10.19 | 0.00 ± 0.00 | -225.80 ± 478.70 | -299.70 ± 538.64 |
| **Detector-Threshold-Full** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| **Detector-Threshold-Timing** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| **Detector-Threshold-Timing-Hold** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| **RL (DQN R2, 5-Seed Avg)** | 0.00 ± 0.00 | -312.40 ± 853.12 | -23.06 ± 63.33 | 0.00 ± 0.00 | -16.24 ± 92.93 | -21.88 ± 110.51 |
| **PPO (R2, 3-Seed Avg)** | 0.00 ± 0.00 | -1364.70 ± 1734.26 | -736.47 ± 1049.64 | 0.00 ± 0.00 | -32.50 ± 187.00 | -49.60 ± 219.61 |
| **Oracle (Causal)** | 0.00 ± 0.00 | -3.20 ± 12.38 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| **Always-Distrust** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |

### r_distrust Penalty Total (from Needless Distrust: -(1.0 * clean_distrust_steps))

| Method | `none` | `spoof` | `stealthy_drift` | `replay` | `telemetry_freeze (delay)` | `telemetry_freeze (dos)` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Always-Trust (No Defense)** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| **Rule-Based (Mythri)** | -0.02 ± 0.14 | -2.09 ± 0.29 | -2.03 ± 0.17 | -0.21 ± 0.46 | -2.73 ± 6.08 | -3.95 ± 8.07 |
| **Kalman-Only (Mythri)** | -0.52 ± 0.67 | -2.58 ± 1.66 | -2.99 ± 1.96 | -1.23 ± 1.25 | -3.82 ± 5.86 | -4.54 ± 6.45 |
| **Detector-Threshold-Full** | -86.43 ± 22.31 | -118.93 ± 47.11 | -126.86 ± 62.51 | -86.02 ± 43.82 | -112.76 ± 53.07 | -112.89 ± 48.91 |
| **Detector-Threshold-Timing** | -4.70 ± 2.39 | -4.95 ± 2.21 | -5.17 ± 2.42 | -4.09 ± 2.46 | -5.08 ± 2.01 | -5.47 ± 2.15 |
| **Detector-Threshold-Timing-Hold** | -17.71 ± 8.80 | -25.86 ± 9.79 | -27.04 ± 13.55 | -14.98 ± 8.35 | -26.90 ± 11.54 | -27.74 ± 11.16 |
| **RL (DQN R2, 5-Seed Avg)** | -2.39 ± 3.54 | -8.84 ± 26.54 | -2.77 ± 2.56 | -1.65 ± 2.16 | -3.58 ± 3.61 | -3.82 ± 4.29 |
| **PPO (R2, 3-Seed Avg)** | -0.01 ± 0.08 | -0.84 ± 0.66 | -0.92 ± 0.55 | -0.02 ± 0.13 | -1.05 ± 0.25 | -1.07 ± 0.41 |
| **Oracle (Causal)** | 0.00 ± 0.00 | -2.00 ± 0.00 | -2.00 ± 0.00 | -2.00 ± 0.00 | -2.00 ± 0.00 | -2.00 ± 0.00 |
| **Always-Distrust** | -600.00 ± 0.00 | -400.00 ± 0.00 | -400.00 ± 0.00 | -400.00 ± 0.00 | -400.00 ± 0.00 | -400.00 ± 0.00 |

### Clean Step Distrust Rate (%) [fraction of clean steps where action==1]

| Method | `none` | `spoof` | `stealthy_drift` | `replay` | `telemetry_freeze (delay)` | `telemetry_freeze (dos)` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Always-Trust (No Defense)** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| **Rule-Based (Mythri)** | 0.00 ± 0.02 | 0.27 ± 0.07 | 0.26 ± 0.04 | 0.05 ± 0.11 | 0.68 ± 1.52 | 0.99 ± 2.02 |
| **Kalman-Only (Mythri)** | 0.09 ± 0.11 | 0.40 ± 0.42 | 0.50 ± 0.49 | 0.31 ± 0.31 | 0.94 ± 1.47 | 1.12 ± 1.62 |
| **Detector-Threshold-Full** | 14.40 ± 3.72 | 29.50 ± 11.77 | 31.48 ± 15.63 | 21.50 ± 10.95 | 27.96 ± 13.25 | 27.99 ± 12.22 |
| **Detector-Threshold-Timing** | 0.78 ± 0.40 | 0.99 ± 0.55 | 1.04 ± 0.60 | 1.02 ± 0.61 | 1.02 ± 0.50 | 1.12 ± 0.54 |
| **Detector-Threshold-Timing-Hold** | 2.95 ± 1.47 | 6.22 ± 2.45 | 6.52 ± 3.39 | 3.74 ± 2.08 | 6.48 ± 2.90 | 6.70 ± 2.79 |
| **RL (DQN R2, 5-Seed Avg)** | 0.40 ± 0.59 | 1.97 ± 6.65 | 0.44 ± 0.64 | 0.41 ± 0.54 | 0.66 ± 0.90 | 0.72 ± 1.06 |
| **PPO (R2, 3-Seed Avg)** | 0.00 ± 0.01 | 0.03 ± 0.10 | 0.02 ± 0.09 | 0.00 ± 0.03 | 0.01 ± 0.06 | 0.02 ± 0.10 |
| **Oracle (Causal)** | 0.00 ± 0.00 | 0.25 ± 0.00 | 0.25 ± 0.00 | 0.25 ± 0.00 | 0.25 ± 0.00 | 0.25 ± 0.00 |
| **Always-Distrust** | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |

### False Alarm Rate on Clean Episodes (%) [episodes of attack_type=='none']

| Method | `none` | `spoof` | `stealthy_drift` | `replay` | `telemetry_freeze (delay)` | `telemetry_freeze (dos)` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Always-Trust (No Defense)** | 0.00 ± 0.00% | — (see `none`) | — (see `none`) | — (see `none`) | — (see `none`) | — (see `none`) |
| **Rule-Based (Mythri)** | 0.00 ± 0.02% | — (see `none`) | — (see `none`) | — (see `none`) | — (see `none`) | — (see `none`) |
| **Kalman-Only (Mythri)** | 0.09 ± 0.11% | — (see `none`) | — (see `none`) | — (see `none`) | — (see `none`) | — (see `none`) |
| **Detector-Threshold-Full** | 14.40 ± 3.72% | — (see `none`) | — (see `none`) | — (see `none`) | — (see `none`) | — (see `none`) |
| **Detector-Threshold-Timing** | 0.78 ± 0.40% | — (see `none`) | — (see `none`) | — (see `none`) | — (see `none`) | — (see `none`) |
| **Detector-Threshold-Timing-Hold** | 2.95 ± 1.47% | — (see `none`) | — (see `none`) | — (see `none`) | — (see `none`) | — (see `none`) |
| **RL (DQN R2, 5-Seed Avg)** | 0.40 ± 0.59% | — (see `none`) | — (see `none`) | — (see `none`) | — (see `none`) | — (see `none`) |
| **PPO (R2, 3-Seed Avg)** | 0.00 ± 0.01% | — (see `none`) | — (see `none`) | — (see `none`) | — (see `none`) | — (see `none`) |
| **Oracle (Causal)** | 0.00 ± 0.00% | — (see `none`) | — (see `none`) | — (see `none`) | — (see `none`) | — (see `none`) |
| **Always-Distrust** | 100.00 ± 0.00% | — (see `none`) | — (see `none`) | — (see `none`) | — (see `none`) | — (see `none`) |

### Distrust Recall (%)

| Method | `none` | `spoof` | `stealthy_drift` | `replay` | `telemetry_freeze (delay)` | `telemetry_freeze (dos)` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Always-Trust (No Defense)** | 100.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| **Rule-Based (Mythri)** | 100.00 ± 0.00 | 100.00 ± 0.00 | 76.17 ± 10.46 | 0.01 ± 0.05 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| **Kalman-Only (Mythri)** | 100.00 ± 0.00 | 100.00 ± 0.00 | 76.39 ± 9.25 | 0.00 ± 0.00 | 2.16 ± 8.56 | 0.84 ± 4.50 |
| **Detector-Threshold-Full** | 100.00 ± 0.00 | 100.00 ± 0.00 | 90.09 ± 1.68 | 2.32 ± 1.27 | 99.56 ± 0.16 | 99.56 ± 0.16 |
| **Detector-Threshold-Timing** | 100.00 ± 0.00 | 100.00 ± 0.00 | 80.50 ± 2.70 | 1.00 ± 0.93 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| **Detector-Threshold-Timing-Hold** | 100.00 ± 0.00 | 100.00 ± 0.00 | 88.55 ± 7.50 | 3.55 ± 2.96 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| **RL (DQN R2, 5-Seed Avg)** | 100.00 ± 0.00 | 95.82 ± 10.94 | 76.64 ± 11.43 | 0.01 ± 0.08 | 70.51 ± 27.95 | 70.69 ± 27.68 |
| **PPO (R2, 3-Seed Avg)** | 100.00 ± 0.00 | 79.83 ± 25.41 | 57.67 ± 23.44 | 0.00 ± 0.00 | 78.67 ± 16.19 | 77.94 ± 17.49 |
| **Oracle (Causal)** | 100.00 ± 0.00 | 99.50 ± 0.00 | 99.50 ± 0.00 | 99.50 ± 0.00 | 99.50 ± 0.00 | 99.50 ± 0.00 |
| **Always-Distrust** | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |

### Attack Success Rate (%)

| Method | `none` | `spoof` | `stealthy_drift` | `replay` | `telemetry_freeze (delay)` | `telemetry_freeze (dos)` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Always-Trust (No Defense)** | 0.0% | 100.0% | 100.0% | 0.0% | 27.0% | 28.0% |
| **Rule-Based (Mythri)** | 0.0% | 0.0% | 0.0% | 0.0% | 27.0% | 28.0% |
| **Kalman-Only (Mythri)** | 0.0% | 0.0% | 0.0% | 0.0% | 23.0% | 27.0% |
| **Detector-Threshold-Full** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **Detector-Threshold-Timing** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **Detector-Threshold-Timing-Hold** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **RL (DQN R2, 5-Seed Avg)** | 0.0% | 16.2% | 1.0% | 0.0% | 2.6% | 3.2% |
| **PPO (R2, 3-Seed Avg)** | 0.0% | 48.0% | 34.3% | 0.0% | 2.7% | 5.0% |
| **Oracle (Causal)** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **Always-Distrust** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

### Recovery Time (s)

| Method | `none` | `spoof` | `stealthy_drift` | `replay` | `telemetry_freeze (delay)` | `telemetry_freeze (dos)` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Always-Trust (No Defense)** | 0.00 ± 0.00 | 19.98 ± 0.15 | 18.83 ± 2.64 | 0.00 ± 0.00 | 2.30 ± 4.42 | 2.70 ± 4.84 |
| **Rule-Based (Mythri)** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.07 ± 0.31 | 0.00 ± 0.00 | 2.51 ± 4.77 | 3.08 ± 5.39 |
| **Kalman-Only (Mythri)** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.02 ± 0.10 | 0.00 ± 0.00 | 2.20 ± 4.77 | 2.87 ± 5.25 |
| **Detector-Threshold-Full** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| **Detector-Threshold-Timing** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| **Detector-Threshold-Timing-Hold** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| **RL (DQN R2, 5-Seed Avg)** | 0.00 ± 0.00 | 2.16 ± 4.88 | 0.23 ± 0.63 | 0.00 ± 0.00 | 0.16 ± 0.93 | 0.22 ± 1.11 |
| **PPO (R2, 3-Seed Avg)** | 0.00 ± 0.00 | 7.76 ± 8.87 | 6.16 ± 8.30 | 0.00 ± 0.00 | 0.33 ± 1.87 | 0.50 ± 2.20 |
| **Oracle (Causal)** | 0.00 ± 0.00 | 0.03 ± 0.12 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| **Always-Distrust** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |

### Max Frequency Deviation (Hz)

| Method | `none` | `spoof` | `stealthy_drift` | `replay` | `telemetry_freeze (delay)` | `telemetry_freeze (dos)` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Always-Trust (No Defense)** | 0.08 ± 0.00 | 4.16 ± 0.84 | 1.73 ± 0.70 | 0.16 ± 0.01 | 0.16 ± 0.07 | 0.17 ± 0.08 |
| **Rule-Based (Mythri)** | 0.08 ± 0.00 | 0.14 ± 0.00 | 0.15 ± 0.03 | 0.16 ± 0.01 | 0.16 ± 0.07 | 0.17 ± 0.08 |
| **Kalman-Only (Mythri)** | 0.08 ± 0.00 | 0.14 ± 0.00 | 0.15 ± 0.02 | 0.16 ± 0.01 | 0.15 ± 0.07 | 0.17 ± 0.08 |
| **Detector-Threshold-Full** | 0.09 ± 0.01 | 0.15 ± 0.00 | 0.15 ± 0.00 | 0.16 ± 0.01 | 0.15 ± 0.00 | 0.15 ± 0.00 |
| **Detector-Threshold-Timing** | 0.08 ± 0.00 | 0.14 ± 0.00 | 0.15 ± 0.02 | 0.16 ± 0.01 | 0.15 ± 0.00 | 0.15 ± 0.00 |
| **Detector-Threshold-Timing-Hold** | 0.08 ± 0.01 | 0.14 ± 0.00 | 0.15 ± 0.01 | 0.16 ± 0.01 | 0.15 ± 0.00 | 0.15 ± 0.00 |
| **RL (DQN R2, 5-Seed Avg)** | 0.08 ± 0.00 | 0.46 ± 0.66 | 0.16 ± 0.04 | 0.16 ± 0.01 | 0.15 ± 0.02 | 0.15 ± 0.03 |
| **PPO (R2, 3-Seed Avg)** | 0.08 ± 0.00 | 1.16 ± 1.20 | 0.46 ± 0.47 | 0.16 ± 0.01 | 0.15 ± 0.02 | 0.15 ± 0.03 |
| **Oracle (Causal)** | 0.08 ± 0.00 | 0.16 ± 0.02 | 0.15 ± 0.00 | 0.15 ± 0.00 | 0.15 ± 0.00 | 0.15 ± 0.00 |
| **Always-Distrust** | 0.15 ± 0.00 | 0.15 ± 0.00 | 0.15 ± 0.00 | 0.15 ± 0.00 | 0.15 ± 0.00 | 0.15 ± 0.00 |

### Detection Delay (steps)

| Method | `none` | `spoof` | `stealthy_drift` | `replay` | `telemetry_freeze (delay)` | `telemetry_freeze (dos)` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Always-Trust (No Defense)** | MISSED (0/100) | MISSED (100/100) | MISSED (100/100) | MISSED (100/100) | MISSED (100/100) | MISSED (100/100) |
| **Rule-Based (Mythri)** | MISSED (0/100) | 0.0 ± 0.0 (0 miss) | 47.7 ± 20.9 (0 miss) | 0.0 ± nan (99 miss) | MISSED (100/100) | MISSED (100/100) |
| **Kalman-Only (Mythri)** | MISSED (0/100) | 0.0 ± 0.0 (0 miss) | 47.2 ± 18.5 (0 miss) | MISSED (100/100) | 146.0 ± 32.8 (92 miss) | 158.0 ± 20.4 (96 miss) |
| **Detector-Threshold-Full** | MISSED (0/100) | 0.0 ± 0.0 (0 miss) | 10.8 ± 7.1 (0 miss) | 36.6 ± 35.6 (2 miss) | 0.9 ± 0.3 (0 miss) | 0.9 ± 0.3 (0 miss) |
| **Detector-Threshold-Timing** | MISSED (0/100) | 0.0 ± 0.0 (0 miss) | 19.8 ± 16.2 (0 miss) | 61.9 ± 48.0 (21 miss) | 0.0 ± 0.0 (0 miss) | 0.0 ± 0.0 (0 miss) |
| **Detector-Threshold-Timing-Hold** | MISSED (0/100) | 0.0 ± 0.0 (0 miss) | 19.4 ± 16.4 (0 miss) | 60.2 ± 47.9 (21 miss) | 0.0 ± 0.0 (0 miss) | 0.0 ± 0.0 (0 miss) |
| **RL (DQN R2, 5-Seed Avg)** | MISSED (0/500) | 2.2 ± 3.5 (0 miss) | 43.4 ± 25.1 (0 miss) | 67.3 ± 45.6 (493 miss) | 46.7 ± 49.7 (17 miss) | 45.3 ± 47.1 (19 miss) |
| **PPO (R2, 3-Seed Avg)** | MISSED (0/300) | 5.4 ± 4.0 (0 miss) | 54.0 ± 30.8 (2 miss) | MISSED (300/300) | 37.7 ± 29.8 (0 miss) | 37.9 ± 30.8 (0 miss) |
| **Oracle (Causal)** | MISSED (0/100) | 1.0 ± 0.0 (0 miss) | 1.0 ± 0.0 (0 miss) | 1.0 ± 0.0 (0 miss) | 1.0 ± 0.0 (0 miss) | 1.0 ± 0.0 (0 miss) |
| **Always-Distrust** | MISSED (0/100) | 0.0 ± 0.0 (0 miss) | 0.0 ± 0.0 (0 miss) | 0.0 ± 0.0 (0 miss) | 0.0 ± 0.0 (0 miss) | 0.0 ± 0.0 (0 miss) |

### Distrust Precision (%)

| Method | `none` | `spoof` | `stealthy_drift` | `replay` | `telemetry_freeze (delay)` | `telemetry_freeze (dos)` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Always-Trust (No Defense)** | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| **Rule-Based (Mythri)** | 98.00 ± 14.07 | 99.46 ± 0.14 | 99.32 ± 0.13 | 81.00 ± 39.43 | 59.00 ± 49.43 | 51.00 ± 50.24 |
| **Kalman-Only (Mythri)** | 57.00 ± 49.76 | 99.22 ± 0.81 | 98.69 ± 1.32 | 34.00 ± 47.61 | 45.90 ± 49.99 | 42.91 ± 49.65 |
| **Detector-Threshold-Full** | 0.00 ± 0.00 | 64.12 ± 8.49 | 61.02 ± 11.02 | 5.79 ± 3.43 | 65.71 ± 10.10 | 65.49 ± 9.68 |
| **Detector-Threshold-Timing** | 2.00 ± 14.07 | 98.07 ± 1.06 | 97.49 ± 1.43 | 29.83 ± 20.28 | 98.01 ± 0.96 | 97.82 ± 1.03 |
| **Detector-Threshold-Timing-Hold** | 2.00 ± 14.07 | 89.10 ± 3.78 | 87.42 ± 5.60 | 29.75 ± 19.56 | 88.75 ± 4.40 | 88.39 ± 4.23 |
| **RL (DQN R2, 5-Seed Avg)** | 48.80 ± 50.04 | 96.45 ± 10.82 | 98.85 ± 1.82 | 43.85 ± 49.40 | 98.08 ± 3.80 | 97.72 ± 6.17 |
| **PPO (R2, 3-Seed Avg)** | 99.33 ± 8.15 | 99.94 ± 0.20 | 99.94 ± 0.22 | 98.33 ± 12.82 | 99.95 ± 0.44 | 99.90 ± 0.97 |
| **Oracle (Causal)** | 100.00 ± 0.00 | 99.50 ± 0.00 | 99.50 ± 0.00 | 99.50 ± 0.00 | 99.50 ± 0.00 | 99.50 ± 0.00 |
| **Always-Distrust** | 0.00 ± 0.00 | 33.33 ± 0.00 | 33.33 ± 0.00 | 33.33 ± 0.00 | 33.33 ± 0.00 | 33.33 ± 0.00 |

---

## Return Decomposition & Sanity Check

Verifying exact penalty accounting under R2 weights: $\text{Return} = - (W_{\text{BAND}} \times \text{OOB steps}) - (W_{\text{DISTRUST}} \times \text{clean distrust steps})$:

| Method Group | Mean Return | r_band Mean | r_distrust Mean | Decomposed Sum | Discrepancy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Always-Trust (No Defense)** | -1175.07 | -1175.07 | 0.00 | -1175.07 | 0.0000 |
| **Rule-Based (Mythri)** | -96.25 | -94.42 | -1.84 | -96.26 | 0.0000 |
| **Kalman-Only (Mythri)** | -90.50 | -87.88 | -2.61 | -90.50 | 0.0000 |
| **Detector-Threshold-Full** | -107.31 | 0.00 | -107.31 | -107.31 | 0.0000 |
| **Detector-Threshold-Timing** | -4.91 | 0.00 | -4.91 | -4.91 | 0.0000 |
| **Detector-Threshold-Timing-Hold** | -23.37 | 0.00 | -23.37 | -23.37 | 0.0000 |
| **RL (DQN R2, 5-Seed Avg)** | -66.11 | -62.26 | -3.84 | -66.11 | 0.0000 |
| **PPO (R2, 3-Seed Avg)** | -364.53 | -363.88 | -0.65 | -364.53 | 0.0000 |
| **Oracle (Causal)** | -2.20 | -0.53 | -1.67 | -2.20 | 0.0000 |
| **Always-Distrust** | -433.33 | 0.00 | -433.33 | -433.33 | 0.0000 |

---

## RL Defender: Per-Seed Breakdown (Mean OOB Steps in Seconds)

| RL Seed | `none` | `spoof` | `stealthy_drift` | `replay` | `telemetry_freeze (delay)` | `telemetry_freeze (dos)` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seed 0** | 0.00s | 0.00s | 0.00s | 0.00s | 0.00s | 0.00s |
| **Seed 1** | 0.00s | 11.66s | 0.00s | 0.00s | 0.13s | 0.18s |
| **Seed 2** | 0.00s | 0.00s | 0.18s | 0.00s | 0.42s | 0.58s |
| **Seed 3** | 0.00s | 3.96s | 0.90s | 0.00s | 0.27s | 0.33s |
| **Seed 4** | 0.00s | 0.00s | 0.07s | 0.00s | 0.00s | 0.00s |

---

## Statistical Paired Comparisons: Difference in Time Outside Safe Band (s)

### (a) Pooled Episodes Comparison

| Comparison | Attack Type | Mean Difference (s) | 95% Bootstrap CI (s) | Verdict |
| :--- | :--- | :---: | :---: | :--- |
| RL vs Kalman-Only | `none` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Kalman-Only | `spoof` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Kalman-Only | `stealthy_drift` | -0.018 | [-0.039, -0.002] | 95% CI excludes zero (RL lower OOB) |
| RL vs Kalman-Only | `replay` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Kalman-Only | `delay` | -2.258 | [-3.181, -1.344] | 95% CI excludes zero (RL lower OOB) |
| RL vs Kalman-Only | `dos` | -2.997 | [-4.110, -1.976] | 95% CI excludes zero (RL lower OOB) |
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
| RL vs Kalman-Only | `spoof` | +3.124 | [+0.000, +7.059] | 95% CI includes zero |
| RL vs Kalman-Only | `stealthy_drift` | +0.213 | [-0.003, +0.558] | 95% CI includes zero |
| RL vs Kalman-Only | `replay` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Kalman-Only | `delay` | -2.096 | [-2.232, -1.955] | 95% CI excludes zero (RL lower OOB) |
| RL vs Kalman-Only | `dos` | -2.778 | [-2.960, -2.584] | 95% CI excludes zero (RL lower OOB) |
| RL vs Detector-Threshold-Full | `none` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Detector-Threshold-Full | `spoof` | +3.124 | [+0.000, +7.059] | 95% CI includes zero |
| RL vs Detector-Threshold-Full | `stealthy_drift` | +0.231 | [+0.015, +0.576] | 95% CI excludes zero (Baseline lower OOB) |
| RL vs Detector-Threshold-Full | `replay` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Detector-Threshold-Full | `delay` | +0.162 | [+0.026, +0.303] | 95% CI excludes zero (Baseline lower OOB) |
| RL vs Detector-Threshold-Full | `dos` | +0.219 | [+0.037, +0.413] | 95% CI excludes zero (Baseline lower OOB) |
| RL vs Detector-Threshold-Timing | `none` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Detector-Threshold-Timing | `spoof` | +3.124 | [+0.000, +7.059] | 95% CI includes zero |
| RL vs Detector-Threshold-Timing | `stealthy_drift` | +0.231 | [+0.015, +0.576] | 95% CI excludes zero (Baseline lower OOB) |
| RL vs Detector-Threshold-Timing | `replay` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Detector-Threshold-Timing | `delay` | +0.162 | [+0.026, +0.303] | 95% CI excludes zero (Baseline lower OOB) |
| RL vs Detector-Threshold-Timing | `dos` | +0.219 | [+0.037, +0.413] | 95% CI excludes zero (Baseline lower OOB) |
| RL vs Detector-Threshold-Timing-Hold | `none` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Detector-Threshold-Timing-Hold | `spoof` | +3.124 | [+0.000, +7.059] | 95% CI includes zero |
| RL vs Detector-Threshold-Timing-Hold | `stealthy_drift` | +0.231 | [+0.015, +0.576] | 95% CI excludes zero (Baseline lower OOB) |
| RL vs Detector-Threshold-Timing-Hold | `replay` | +0.000 | [+0.000, +0.000] | 95% CI includes zero |
| RL vs Detector-Threshold-Timing-Hold | `delay` | +0.162 | [+0.026, +0.303] | 95% CI excludes zero (Baseline lower OOB) |
| RL vs Detector-Threshold-Timing-Hold | `dos` | +0.219 | [+0.037, +0.413] | 95% CI excludes zero (Baseline lower OOB) |
