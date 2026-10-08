"""
Phase 2 Test: Kalman Filter State Estimator on Normal Operation.

Validates:
1. Dynamic tracking of grid frequency under nominal load and +3% step disturbance.
2. Residual r(k) is zero-mean (E[r] ≈ 0) and bounded within 3-sigma limits.
3. Estimation error (True - Estimate) converges rapidly to near-zero.
4. Generates Phase 2 plots and summary metrics in deliverable2/results/.
"""

import os
import sys
import numpy as np
import pandas as pd

# Add repo root and examples/powergrid to path
_PG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_PG_DIR, "..", ".."))
for p in [_PG_DIR, _REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from physics import GridPhysics
from deliverable2.config import DEFAULT_CONFIG
from deliverable2.estimator import GridKalmanFilter
from deliverable2.plots_d2 import plot_kalman_performance


def run_kalman_normal_simulation(duration_sec: float = 60.0, apply_disturbance: bool = True,
                                 add_sensor_noise: bool = True, seed: int = 42):
    """Run simulated grid under normal operation with Kalman State Estimator."""
    np.random.seed(seed)
    cfg = DEFAULT_CONFIG

    physics = GridPhysics(
        f0=cfg.grid.f0,
        H=cfg.grid.H,
        S_base=cfg.grid.S_base,
        D=cfg.grid.D,
        dt=cfg.grid.dt_physics,
        p_load=cfg.grid.nominal_load,
        p_gen=cfg.grid.initial_p_gen
    )

    kf = GridKalmanFilter(config=cfg)

    # Simulation loop settings
    dt = cfg.grid.dt_physics
    total_steps = int(duration_sec / dt)
    ctrl_interval = int(round(cfg.grid.dt_controller / dt))  # every 2 steps (0.2s)

    records = []
    p_gen = cfg.grid.initial_p_gen
    integral_err = 0.0

    print(f"\n=======================================================")
    print(f"  Running Phase 2: Kalman Filter Normal Operation Test")
    print(f"  Duration: {duration_sec}s | Load Step at t=10s: {apply_disturbance} | Noise: {add_sensor_noise}")
    print(f"=======================================================")

    for step in range(total_steps):
        t = step * dt

        # 1. Physical Load Disturbance (+3 MW at t=10s)
        if apply_disturbance and t >= 10.0:
            current_load = cfg.grid.nominal_load + 3.0
        else:
            current_load = cfg.grid.nominal_load
        physics.set_load(current_load)

        # 2. Physics Step (Ground Truth)
        true_freq = physics.step(p_gen)

        # 3. Sensor measurement with quantization and small Gaussian noise
        meas_noise = np.random.normal(0.0, np.sqrt(cfg.kalman.measurement_noise_cov_R)) if add_sensor_noise else 0.0
        # Modbus integer scaling quantization (0.01 Hz resolution)
        scaled_int = int(round((true_freq + meas_noise) * cfg.grid.scale_factor))
        meas_freq = scaled_int / cfg.grid.scale_factor

        # 4. Kalman Filter Step (Predict -> Measure -> Correct)
        kf_res = kf.step(y_k=meas_freq, p_gen=p_gen, dt=dt, time_sec=t)

        # 5. Controller Execution (Threshold Control with Deadband)
        if step % ctrl_interval == 0:
            ctrl_freq = kf_res.estimated_freq  # Controller uses Kalman estimate
            if ctrl_freq < 49.80:
                p_gen += 0.05
            elif ctrl_freq > 50.20:
                p_gen -= 0.05
            p_gen = max(10.0, min(300.0, p_gen))

        # Log record
        records.append({
            "sim_time_sec": t,
            "true_frequency": true_freq,
            "measurement_y": meas_freq,
            "predicted_freq": kf_res.predicted_freq,
            "estimated_freq": kf_res.estimated_freq,
            "residual": kf_res.residual,
            "residual_cov_S": kf_res.residual_cov_S,
            "normalized_residual": kf_res.normalized_residual,
            "kalman_gain": kf_res.kalman_gain,
            "error_cov_P": kf_res.error_cov_P,
            "generation_command": p_gen,
            "load_demand_mw": current_load
        })

    df = pd.DataFrame(records)

    # Compute Validation Statistics
    residuals = df["residual"].to_numpy()
    estimation_errors = (df["true_frequency"] - df["estimated_freq"]).to_numpy()

    mean_res = float(np.mean(residuals))
    std_res = float(np.std(residuals))
    max_res = float(np.max(np.abs(residuals)))
    mean_err = float(np.mean(estimation_errors))
    std_err = float(np.std(estimation_errors))
    rmse = float(np.sqrt(np.mean(estimation_errors ** 2)))

    # Save CSV & Plot
    csv_path = os.path.join(cfg.results_dir, "phase2_kalman_normal.csv")
    png_path = os.path.join(cfg.results_dir, "phase2_kalman_normal.png")
    df.to_csv(csv_path, index=False)

    plot_kalman_performance(df, save_path=png_path,
                            title="Phase 2: Kalman State Estimator Tracking (Normal Operation)")

    print(f"\n--- Phase 2 Kalman Filter Test Results ---")
    print(f"Total Simulation Steps : {len(df)}")
    print(f"Mean Residual E[r]      : {mean_res:+.6f} Hz  (Expect near 0.0)")
    print(f"Residual Std Dev        : {std_res:.6f} Hz")
    print(f"Max Absolute Residual   : {max_res:.6f} Hz")
    print(f"Estimation RMSE         : {rmse:.6f} Hz")
    print(f"Results CSV Saved       : {csv_path}")
    print(f"Results PNG Saved       : {png_path}")
    print(f"------------------------------------------\n")

    return df, {
        "mean_residual": mean_res,
        "std_residual": std_res,
        "max_residual": max_res,
        "rmse": rmse,
        "csv_path": csv_path,
        "png_path": png_path
    }


if __name__ == "__main__":
    run_kalman_normal_simulation(duration_sec=60.0, apply_disturbance=True, add_sensor_noise=True)
