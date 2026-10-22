# Grid Defense – Interactive Demo

## Quick Start

```bash
# 1. Install dependencies (from project root)
pip install streamlit plotly

# 2. (Optional) Pre-compute cached trajectories for instant display
python demo/precompute.py

# 3. (Optional) Export offline GIF animations as backup
python demo/export_video.py

# 4. Launch the app
streamlit run demo/app.py
```

The app opens in your browser at `http://localhost:8501`.

---

## 5-Minute Demo Script

### Scene 1 – Spoof Attack (Tab 1, ~90 s)

1. **Sidebar**: Attack = `spoof`, Intensity = `48.50 Hz`, Start = `20 s`,
   Seed = `42`, Defense = `No Defense`.
2. Click **Run Simulation → Show All**.
3. Point out: true frequency crashes to ~46 Hz, OOB > 30 s, massive
   deviation.  Status badge turns red "ATTACK ACTIVE".
4. Switch Defense to **RL DQN** (seed 0), re-run.
5. Show: true frequency stays in the safe band, 0 OOB steps.
   The action strip turns red (distrust) right as the attack starts.
6. Switch to **Kalman-Only**, re-run – also 0 OOB.  Mention: "Both
   Kalman-Only and RL catch spoof, but on other seeds RL can fail
   (seed 1: 11.66 s OOB)."

### Scene 2 – Stealthy Drift (Tab 1, ~60 s)

1. Change Attack to `stealthy_drift`, Intensity = `0.05 Hz/s`.
2. Defense = `No Defense` → run.  Frequency slowly drifts out of band.
3. Defense = `RL DQN` → run.  RL detects with some delay but keeps
   frequency in band.
4. Defense = `Kalman-Only` → run.  Also succeeds (0.02 s OOB on average).

### Scene 3 – Telemetry Freeze (Tab 1, ~60 s)

1. Attack = `telemetry_freeze`.
2. Defense = `No Defense` → run.  Frequency drifts modestly; stale
   readings mask the deviation.
3. Defense = `RL DQN` → run.  RL catches it (low OOB).
4. Defense = `Detector-Timing` → run.  **0.0 s OOB** – the hand-tuned
   timing heuristic checks packet_age directly and wins.

### Scene 4 – Compare All Defenses (Tab 2, ~45 s)

1. Keep the same attack (`spoof`, intensity 48.50).
2. Switch to **Compare Defenses** tab.
3. Click **Run All Defenses**.
4. Walk through the 2×3 small-multiples grid.
5. Highlight the metrics table: best per column is shaded green.

### Scene 5 – Results & Limitations (Tab 3, ~45 s)

1. Switch to **Results** tab.
2. Show the OOB bar chart grouped by attack type.
3. Scroll to the per-seed RL table: "Seed 1 is the weak spot."
4. Show the RL vs Detector-Timing vs Kalman-Only comparison.
5. End with the **Limitations** box – read or paraphrase the four points.

---

## Files

| File | Purpose |
|------|---------|
| `demo/app.py` | Streamlit dashboard (3 tabs) |
| `demo/sim_engine.py` | Simulation runner wrapping real env & policies |
| `demo/precompute.py` | Batch-cache trajectories for instant playback |
| `demo/export_video.py` | Offline GIF animations as backup |
| `demo/cache/` | Pickle cache (created by `precompute.py`) |
| `demo/videos/` | GIF files (created by `export_video.py`) |

## Troubleshooting

- **`ModuleNotFoundError: stable_baselines3`** – `pip install stable-baselines3[extra]`
- **Slow first run** – Use `python demo/precompute.py` first, then toggle
  "Use cached runs" in the sidebar.
- **Streamlit won't start** – Try `python -m streamlit run demo/app.py`.
- **No videos generated** – Install `pillow` (`pip install pillow`) for GIF
  export.
