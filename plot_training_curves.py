"""
Plot Training Curves - plot_training_curves.py

Overlays episode reward mean (ep_rew_mean) per reward variant, aggregated across seeds,
into results/training_curves.png using monitor CSV files.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
_LOGS_DIR = os.path.join(_ROOT_DIR, "logs")
_RESULTS_DIR = os.path.join(_ROOT_DIR, "results")


def load_monitor_data(csv_path: str):
    """Load SB3 monitor CSV skipping comments header."""
    try:
        df = pd.read_csv(csv_path, skiprows=1)
        if "r" in df.columns and "l" in df.columns:
            df["cumulative_steps"] = df["l"].cumsum()
            return df
    except Exception:
        pass
    return None


def main():
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    variants = ["R1_baseline", "R2_costly", "R3_degraded", "R4_switch"]
    colors = {
        "R1_baseline": "#1b9e77",
        "R2_costly": "#d95f02",
        "R3_degraded": "#7570b3",
        "R4_switch": "#e7298a",
    }

    plt.figure(figsize=(10, 6))
    plotted_any = False

    for var in variants:
        # Search for runs matching dqn_{var}_s*
        pattern = os.path.join(_LOGS_DIR, f"dqn_{var}_s*", "eval_monitor.csv")
        eval_files = glob.glob(pattern)
        if not eval_files:
            pattern = os.path.join(_LOGS_DIR, f"dqn_{var}_s*", "train_monitor.csv")
            eval_files = glob.glob(pattern)

        if not eval_files:
            continue

        all_steps = []
        all_rewards = []

        for f in eval_files:
            data = load_monitor_data(f)
            if data is not None and len(data) > 0:
                steps = data["cumulative_steps"].values
                rewards = data["r"].values
                all_steps.append(steps)
                all_rewards.append(rewards)

        if all_rewards:
            # Interpolate to common step grid
            min_len = min(len(r) for r in all_rewards)
            truncated_r = [r[:min_len] for r in all_rewards]
            common_steps = all_steps[0][:min_len]
            mean_r = np.mean(truncated_r, axis=0)
            std_r = np.std(truncated_r, axis=0) if len(truncated_r) > 1 else np.zeros_like(mean_r)

            # Rolling smoothing
            window = max(1, len(mean_r) // 10)
            smooth_mean = pd.Series(mean_r).rolling(window, min_periods=1).mean().values
            smooth_std = pd.Series(std_r).rolling(window, min_periods=1).mean().values

            c = colors.get(var, "#333333")
            plt.plot(common_steps, smooth_mean, label=f"DQN {var}", color=c, linewidth=2.0)
            plt.fill_between(common_steps, smooth_mean - smooth_std, smooth_mean + smooth_std, color=c, alpha=0.15)
            plotted_any = True

    if plotted_any:
        plt.title("Phase 4 Multi-Reward Training Curves (Mean Episode Reward)", fontsize=14, fontweight="bold")
        plt.xlabel("Training Timesteps", fontsize=12)
        plt.ylabel("Mean Episode Reward", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="lower right", framealpha=0.9)
        plt.tight_layout()
        out_png = os.path.join(_RESULTS_DIR, "training_curves.png")
        plt.savefig(out_png, dpi=200)
        plt.close()
        print(f"[Plot Saved] {out_png}")
    else:
        print("No monitor logs found to plot yet.")


if __name__ == "__main__":
    main()
