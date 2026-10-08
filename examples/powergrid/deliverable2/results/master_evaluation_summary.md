# Master Evaluation Summary — Deliverable 2

### Cyber Detection & Physical Resilience Matrix

| scenario | defense_mode | precision | recall | far | detection_delay_sec | max_freq_dev_hz | time_outside_safe_sec | attack_success_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NORMAL | none | 0.0% | 100.0% | 0.83% | 0.00s | 0.087 Hz | 0.0 s | 0% |
| SPOOF | none | 93.5% | 100.0% | 3.50% | 0.00s | 4.567 Hz | 39.8 s | 100% |
| SPOOF | rule_based | 87.3% | 100.0% | 7.25% | 0.00s | 0.481 Hz | 15.9 s | 100% |
| SPOOF | kalman_only | 99.0% | 100.0% | 0.50% | 0.00s | 0.154 Hz | 0.0 s | 0% |
| SPOOF | multi_signal | 98.5% | 100.0% | 0.75% | 0.00s | 0.155 Hz | 0.0 s | 0% |
| SLOW_DRIFT | none | 93.8% | 75.5% | 2.50% | 4.90s | 1.699 Hz | 28.0 s | 100% |
| SLOW_DRIFT | rule_based | 90.9% | 79.5% | 4.00% | 4.10s | 0.428 Hz | 10.5 s | 100% |
| SLOW_DRIFT | kalman_only | 96.1% | 73.5% | 1.50% | 5.30s | 0.139 Hz | 0.0 s | 0% |
| SLOW_DRIFT | multi_signal | 97.9% | 69.5% | 0.75% | 6.10s | 0.135 Hz | 0.0 s | 0% |
| REPLAY | none | 97.1% | 67.0% | 1.00% | 0.70s | 0.158 Hz | 0.0 s | 0% |
| REPLAY | rule_based | 100.0% | 0.0% | 0.00% | Never | 0.158 Hz | 0.0 s | 0% |
| REPLAY | kalman_only | 0.0% | 0.0% | 0.50% | Never | 0.155 Hz | 0.0 s | 0% |
| REPLAY | multi_signal | 97.9% | 68.5% | 0.75% | 0.70s | 0.156 Hz | 0.0 s | 0% |
| DELAY | none | 97.9% | 95.5% | 1.00% | 0.10s | 0.142 Hz | 0.0 s | 0% |
| DELAY | rule_based | 100.0% | 83.0% | 0.00% | 0.10s | 0.142 Hz | 0.0 s | 0% |
| DELAY | kalman_only | 99.4% | 83.5% | 0.25% | 0.10s | 0.135 Hz | 0.0 s | 0% |
| DELAY | multi_signal | 98.5% | 96.5% | 0.75% | 0.10s | 0.150 Hz | 0.0 s | 0% |
| DOS | none | 95.7% | 78.0% | 1.75% | 0.20s | 0.120 Hz | 0.0 s | 0% |
| DOS | rule_based | 100.0% | 70.5% | 0.00% | 0.20s | 0.120 Hz | 0.0 s | 0% |
| DOS | kalman_only | 100.0% | 72.0% | 0.00% | 0.20s | 0.127 Hz | 0.0 s | 0% |
| DOS | multi_signal | 96.2% | 76.0% | 1.50% | 0.20s | 0.130 Hz | 0.0 s | 0% |


