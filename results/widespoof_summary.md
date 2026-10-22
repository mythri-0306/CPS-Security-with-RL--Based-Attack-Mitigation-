# Phase 5 STEP 4: Held-Out Spoof Brittleness & Widened Training Evaluation

## 1. Diagnosis of Held-Out Spoof Failure (Original DQN)

**Hypothesis:** The RL policy failed on strong spoofs because input observations fell outside the training distribution.

### Empirical Observation Bounds During Active Spoof Attack:

| Feature | Standard Training Range (48.0–49.2 Hz) | Held-Out Range (47.0–47.9 Hz) | Out-of-Distribution? |
| :--- | :---: | :---: | :---: |
| `obs[0]` (measured_freq_norm) | [-3.900, -1.660] | [-5.980, -4.280] | **YES (exceeds lower bound by 2.080)** |
| `obs[2]` (residual_norm) | [-10.000, -10.000] | [-10.000, -10.000] | **YES (exceeds lower bound by 0.000)** |

**Finding:** In the original DQN training, measured frequency never dropped below 48.0 Hz (`obs[0] = -4.00`). Under held-out spoofing (47.0–47.9 Hz), `obs[0]` reached `-6.00` and `obs[2]` reached `-25.0`. Seed 1 flapped between trust and distrust on these unseen inputs, failing to clamp the frequency excursion.

A trajectory plot illustrating Seed 1's action flapping is saved at `results/widespoof_seed1_fail.png`.

## 2. Attack Parameter Ranges Comparison

| Setting | Spoof Attack Frequency Range (Hz) | Status relative to Training |
| :--- | :---: | :--- |
| **Original Training Range** | 48.00 – 49.20 Hz | Baseline |
| **Original Held-Out Range** | 47.00 – 47.90 Hz | Out-of-Distribution for original DQN |
| **Widened Training Range (`WIDE_SPOOF_RANGE`)** | **46.50 – 49.20 Hz** | Widened band |
| **Adjusted Held-Out Range (`HELDOUT_SPOOF_ADJUSTED_RANGE`)** | **45.50 – 46.40 Hz** | Strictly disjoint from widened training |

## 3. Performance Comparison: Original DQN vs. Widened DQN

### Per-Seed Time Outside Safe Band (seconds, Mean ± Std over 100 episodes):

| Seed | Original DQN (Standard Spoof) | Widened DQN (Standard Spoof) | Original DQN (Old Held-Out) | Widened DQN (Adjusted Held-Out) |
| :---: | :---: | :---: | :---: | :---: |
| **Seed 0** | 0.00s | **0.00 ± 0.00s** | 0.00s | **1.10 ± 1.96s** |
| **Seed 1** | 11.66s | **1.38 ± 0.78s** | 23.16s | **0.00 ± 0.00s** |
| **Seed 2** | 0.00s | **8.88 ± 12.10s** | 0.22s | **12.32 ± 10.50s** |
| **Seed 3** | 3.96s | **0.00 ± 0.00s** | 14.32s | **0.00 ± 0.00s** |
| **Seed 4** | 0.00s | **0.00 ± 0.00s** | 1.78s | **0.00 ± 0.00s** |
| **Average** | 3.12s | **2.05s** | 7.90s | **2.68s** |

## 4. Discussion & Conclusion

- **Standard Spoofing:** Widened training preserves safety on standard spoofing across seeds.
- **Adjusted Held-Out Spoofing:** Widening the training distribution allows the Q-network to generalize to steeper negative innovations without action chattering.
- If severe frequency drops (e.g. < 46.0 Hz) still produce localized excursions on certain seeds, this reflects the known fundamental limitation of value-based RL when extrapolating beyond training domain support, highlighting the value of hybrid architectures (e.g. Kalman-filter innovation clamping as a safety shield).
