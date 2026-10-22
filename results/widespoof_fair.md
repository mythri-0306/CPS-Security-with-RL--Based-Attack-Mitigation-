# Fair Widened-Spoof Comparison

Evaluates the **Original 5 DQN models** and the **Widened 5 DQN models** on the **exact same test scenarios** (100 episodes per seed per test set, identical random attack parameters and seeds). No retraining was performed.

## 1. Test Sets & Training Range Overlap

| Test Set | Test Frequency Range (Hz) | Original Training [48.00, 49.20] Hz | Widened Training [46.50, 49.20] Hz |
| :--- | :---: | :---: | :---: |
| **Standard Spoof [48.00 - 49.20 Hz]** | 48.00 - 49.20 Hz | OVERLAP (In-Distribution / Full Coverage) | OVERLAP (In-Distribution / Full Coverage) |
| **Old Held-Out Spoof [47.00 - 47.90 Hz]** | 47.00 - 47.90 Hz | NO OVERLAP (Strictly Out-of-Distribution) | OVERLAP (In-Distribution / Full Coverage) |
| **New Held-Out Spoof [45.50 - 46.40 Hz]** | 45.50 - 46.40 Hz | NO OVERLAP (Strictly Out-of-Distribution) | NO OVERLAP (Strictly Out-of-Distribution) |

## 2. Per-Seed Safe Band Out-of-Band Time (seconds, Mean +/- Std over 100 episodes)

| Seed | (a) Standard: Orig | (a) Standard: Wide | (b) Old Held-Out: Orig | (b) Old Held-Out: Wide | (c) New Held-Out: Orig | (c) New Held-Out: Wide |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seed 0** | 0.00 +/- 0.00s | 0.00 +/- 0.00s | 0.00 +/- 0.00s | 0.00 +/- 0.00s | 0.00 +/- 0.00s | 1.10 +/- 1.96s |
| **Seed 1** | 9.58 +/- 14.66s | 1.27 +/- 0.75s | 23.16 +/- 2.88s | 0.18 +/- 0.35s | 23.26 +/- 2.88s | 0.00 +/- 0.00s |
| **Seed 2** | 0.00 +/- 0.00s | 8.25 +/- 11.48s | 0.22 +/- 0.45s | 22.90 +/- 3.06s | 19.30 +/- 4.45s | 12.32 +/- 10.50s |
| **Seed 3** | 2.89 +/- 1.05s | 0.00 +/- 0.00s | 14.32 +/- 9.60s | 0.00 +/- 0.00s | 23.16 +/- 2.88s | 0.00 +/- 0.00s |
| **Seed 4** | 0.00 +/- 0.00s | 0.00 +/- 0.00s | 1.78 +/- 6.49s | 0.00 +/- 0.00s | 23.16 +/- 2.88s | 0.00 +/- 0.00s |
| **Mean +/- Std** | **2.49 +/- 3.72s** | **1.90 +/- 3.21s** | **7.89 +/- 9.30s** | **4.62 +/- 9.14s** | **17.78 +/- 9.01s** | **2.68 +/- 4.84s** |

## 3. Computed Comparative Verdicts per Test Set

### (a) Standard Spoof [48.00 - 49.20 Hz]
- **Original DQN Mean OOB:** 2.49 +/- 3.72s
- **Widened DQN Mean OOB :** 1.90 +/- 3.21s
- **Per-Seed Differences (wide minus orig):** [+0.00, -8.31, +8.25, -2.89, +0.00]
- **Seed Improvement Counts:** 2 improved, 1 worsened, 2 unchanged
- **Verdict:** **inconclusive** (per-seed diffs: [+0.00, -8.31, +8.25, -2.89, +0.00]; 2/5 seeds improved, 1/5 worsened)

### (b) Old Held-Out Spoof [47.00 - 47.90 Hz]
- **Original DQN Mean OOB:** 7.89 +/- 9.30s
- **Widened DQN Mean OOB :** 4.62 +/- 9.14s
- **Per-Seed Differences (wide minus orig):** [+0.00, -22.98, +22.68, -14.32, -1.78]
- **Seed Improvement Counts:** 3 improved, 1 worsened, 1 unchanged
- **Verdict:** **inconclusive** (per-seed diffs: [+0.00, -22.98, +22.68, -14.32, -1.78]; 3/5 seeds improved, 1/5 worsened)
- **Note on Test Set (b):** Test set (b) lies inside the widened training range [46.50, 49.20] Hz, so it is not a held-out test for that group.

### (c) New Held-Out Spoof [45.50 - 46.40 Hz]
- **Original DQN Mean OOB:** 17.78 +/- 9.01s
- **Widened DQN Mean OOB :** 2.68 +/- 4.84s
- **Per-Seed Differences (wide minus orig):** [+1.10, -23.26, -6.98, -23.16, -23.16]
- **Seed Improvement Counts:** 4 improved, 1 worsened, 0 unchanged
- **Verdict:** **better** (per-seed diffs: [+1.10, -23.26, -6.98, -23.16, -23.16]; 4/5 seeds improved, 1/5 worsened)

