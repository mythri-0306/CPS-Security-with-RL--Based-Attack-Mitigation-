# Phase 4 Multi-Reward Benchmark Summary

Evaluated with **Kalman State Estimator** and **Extended Observation Features** across 50 episodes per attack type.

## Reward Variants Tested

| Variant | W_BAND | W_DISTRUST | W_DISTRUST_ATTACK | W_SWITCH | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **R1_baseline** | 10.0 | 0.1 | 0.0 | 0.0 | Baseline |
| **R2_costly** | 10.0 | 1.0 | 0.0 | 0.0 | Costly Distrust |
| **R3_degraded** | 10.0 | 1.0 | 0.3 | 0.0 | Degraded Fallback |
| **R4_switch** | 10.0 | 1.0 | 0.3 | 0.5 | Anti-Flapping |

---

## Aggregated Performance Table (Reward-Independent Metrics)

| Variant / Policy | Attack Type | Mean OOB (steps) | Clean Distrust Rate (%) | Detection Delay (steps) | Missed Attacks |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Always-Trust | `none` | 0.0 | 0.0% | N/A | 0 |
| Always-Trust | `spoof` | 394.7 | 0.0% | N/A | 50 |
| Always-Trust | `stealthy_drift` | 252.0 | 0.0% | N/A | 50 |
| Always-Trust | `replay` | 0.0 | 0.0% | N/A | 50 |
| Always-Trust | `delay` | 26.8 | 0.0% | N/A | 50 |
| Always-Trust | `dos` | 20.1 | 0.0% | N/A | 50 |
| Always-Distrust | `none` | 0.0 | 100.0% | N/A | 0 |
| Always-Distrust | `spoof` | 0.0 | 100.0% | 0.0 | 0 |
| Always-Distrust | `stealthy_drift` | 0.0 | 100.0% | 0.0 | 0 |
| Always-Distrust | `replay` | 0.0 | 100.0% | 0.0 | 0 |
| Always-Distrust | `delay` | 0.0 | 100.0% | 0.0 | 0 |
| Always-Distrust | `dos` | 0.0 | 100.0% | 0.0 | 0 |
| Oracle (Causal) | `none` | 0.0 | 0.0% | N/A | 0 |
| Oracle (Causal) | `spoof` | 0.0 | 0.2% | 0.0 | 0 |
| Oracle (Causal) | `stealthy_drift` | 0.0 | 0.2% | 0.0 | 0 |
| Oracle (Causal) | `replay` | 0.0 | 0.2% | 0.0 | 0 |
| Oracle (Causal) | `delay` | 0.0 | 0.2% | 0.0 | 0 |
| Oracle (Causal) | `dos` | 0.0 | 0.2% | 0.0 | 0 |
| **DQN R1_baseline** | `none` | 0.0 ± 0.0 | 31.8% ± 23.0% | N/A | 0 |
| **DQN R1_baseline** | `spoof` | 0.0 ± 0.0 | 28.5% ± 20.1% | 0.0 | 0 |
| **DQN R1_baseline** | `stealthy_drift` | 0.0 ± 0.0 | 28.4% ± 19.5% | 14.4 | 0 |
| **DQN R1_baseline** | `replay` | 0.0 ± 0.0 | 28.8% ± 21.1% | 26.9 | 5 |
| **DQN R1_baseline** | `delay` | 7.6 ± 11.4 | 30.2% ± 22.2% | 33.8 | 19 |
| **DQN R1_baseline** | `dos` | 6.3 ± 10.3 | 29.4% ± 20.9% | 28.5 | 12 |
| **DQN R2_costly** | `none` | 0.0 ± 0.0 | 0.2% ± 0.2% | N/A | 0 |
| **DQN R2_costly** | `spoof` | 43.7 ± 75.8 | 4.1% ± 3.0% | 0.6 | 0 |
| **DQN R2_costly** | `stealthy_drift` | 0.7 ± 1.2 | 1.6% ± 1.1% | 39.0 | 0 |
| **DQN R2_costly** | `replay` | 0.0 ± 0.0 | 0.5% ± 0.5% | 160.5 | 143 |
| **DQN R2_costly** | `delay` | 1.8 ± 1.6 | 0.9% ± 0.8% | 23.8 | 0 |
| **DQN R2_costly** | `dos` | 1.7 ± 2.0 | 1.0% ± 0.9% | 23.3 | 1 |
| **DQN R3_degraded** | `none` | 0.0 ± 0.0 | 0.1% ± 0.2% | N/A | 0 |
| **DQN R3_degraded** | `spoof` | 58.8 ± 100.3 | 4.3% ± 4.7% | 9.0 | 3 |
| **DQN R3_degraded** | `stealthy_drift` | 0.0 ± 0.0 | 1.5% ± 1.1% | 39.0 | 0 |
| **DQN R3_degraded** | `replay` | 0.0 ± 0.0 | 0.3% ± 0.5% | 193.0 | 149 |
| **DQN R3_degraded** | `delay` | 5.2 ± 7.3 | 1.0% ± 1.2% | 74.4 | 27 |
| **DQN R3_degraded** | `dos` | 3.9 ± 5.9 | 1.1% ± 1.3% | 70.8 | 37 |
| **DQN R4_switch** | `none` | 0.0 ± 0.0 | 0.0% ± 0.0% | N/A | 0 |
| **DQN R4_switch** | `spoof` | 4.9 ± 4.9 | 1.9% ± 1.4% | 1.2 | 0 |
| **DQN R4_switch** | `stealthy_drift` | 3.2 ± 5.3 | 1.8% ± 1.3% | 47.0 | 0 |
| **DQN R4_switch** | `replay` | 0.0 ± 0.0 | 0.1% ± 0.1% | 193.0 | 149 |
| **DQN R4_switch** | `delay` | 9.7 ± 15.4 | 0.3% ± 0.0% | 96.5 | 73 |
| **DQN R4_switch** | `dos` | 6.8 ± 11.2 | 0.2% ± 0.1% | 75.1 | 86 |
| **PPO R3_degraded** | `none` | 0.0 | 0.0% | N/A | 0 |
| **PPO R3_degraded** | `spoof` | 0.0 | 0.2% | 0.0 | 0 |
| **PPO R3_degraded** | `stealthy_drift` | 183.0 | 0.1% | 66.7 | 0 |
| **PPO R3_degraded** | `replay` | 0.0 | 0.0% | N/A | 50 |
| **PPO R3_degraded** | `delay` | 26.8 | 0.0% | N/A | 50 |
| **PPO R3_degraded** | `dos` | 20.1 | 0.0% | N/A | 50 |
