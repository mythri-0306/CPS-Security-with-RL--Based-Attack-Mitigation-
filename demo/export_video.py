"""
demo/export_video.py  –  Render offline MP4 / GIF animations comparing
No Defense vs Kalman-Only vs RL DQN for each attack type.

Usage:  python demo/export_video.py
Output: demo/videos/*.gif  (one per attack type)
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_DEMO_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
if _DEMO_DIR not in sys.path:
    sys.path.insert(1, _DEMO_DIR)
_PG_DIR = os.path.join(_ROOT_DIR, "examples", "powergrid")
if os.path.exists(_PG_DIR) and _PG_DIR not in sys.path:
    sys.path.append(_PG_DIR)

from sim_engine import run_episode, ATTACK_UI_TO_ENV
from config import SAFE_BAND_LOW, SAFE_BAND_HIGH, NOMINAL_FREQ

VIDEO_DIR = os.path.join(_DEMO_DIR, "videos")
os.makedirs(VIDEO_DIR, exist_ok=True)

# Defenses to compare in each video
COMPARE_DEFENSES = ["No Defense", "Kalman-Only", "RL DQN"]
COLORS = {"No Defense": "#ef476f", "Kalman-Only": "#06d6a0", "RL DQN": "#118ab2"}

# Default attack intensities
DEFAULT_INTENSITIES = {
    "none":              None,
    "spoof":             48.50,
    "stealthy_drift":    0.05,
    "replay":            10.0,
    "telemetry_freeze":  None,
}

SEED = 42
ATTACK_START = 20.0
FRAME_SKIP = 5      # show every 5th step for smooth but fast GIF


def export_one(attack_type: str):
    """Generate a GIF for one attack type comparing three defenses."""
    print(f"  Generating trajectories for '{attack_type}' ...")
    trajs = {}
    for dm in COMPARE_DEFENSES:
        try:
            trajs[dm] = run_episode(
                attack_type=attack_type,
                defense_method=dm,
                env_seed=SEED,
                intensity=DEFAULT_INTENSITIES.get(attack_type),
                attack_start=ATTACK_START,
                dqn_seed=0,
            )
        except Exception as e:
            print(f"    [WARN] {dm}: {e}")

    if not trajs:
        return

    n_steps = min(len(tr["t"]) for tr in trajs.values())
    frames_idx = list(range(0, n_steps, FRAME_SKIP))
    if frames_idx[-1] != n_steps - 1:
        frames_idx.append(n_steps - 1)

    # Setup figure
    fig, axes = plt.subplots(len(trajs), 1, figsize=(10, 3 * len(trajs)),
                              sharex=True, facecolor="#0f0c29")

    if len(trajs) == 1:
        axes = [axes]

    lines = {}
    for ax, (dm, tr) in zip(axes, trajs.items()):
        ax.set_facecolor("#1a1a2e")
        ax.set_ylabel("Freq (Hz)", color="white", fontsize=9)
        ax.set_title(f"{dm}  (OOB = {tr['time_oob']:.1f}s)", color="white", fontsize=10, pad=6)
        ax.axhspan(SAFE_BAND_LOW, SAFE_BAND_HIGH, alpha=0.10, color="#FFC107")
        ax.axhline(SAFE_BAND_LOW,  color="#FFC107", lw=0.7, ls="--")
        ax.axhline(SAFE_BAND_HIGH, color="#FFC107", lw=0.7, ls="--")

        # Attack window rectangle
        att = tr["attack_active"].astype(bool)
        if att.any():
            t0, t1 = tr["t"][att][0], tr["t"][att][-1]
            ax.axvspan(t0, t1, alpha=0.12, color="#ff006e")

        ln, = ax.plot([], [], color=COLORS.get(dm, "#00b4d8"), lw=1.5, label="True Freq")
        lines[dm] = (ln, tr)

        ax.set_xlim(0, tr["t"][-1])
        ax.set_ylim(min(SAFE_BAND_LOW - 0.5, tr["f_true"].min() - 0.2),
                     max(SAFE_BAND_HIGH + 0.5, tr["f_true"].max() + 0.2))
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#444")

    axes[-1].set_xlabel("Time (s)", color="white", fontsize=9)
    fig.suptitle(f"Attack: {attack_type}", color="white", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    def init():
        for dm, (ln, tr) in lines.items():
            ln.set_data([], [])
        return [ln for ln, _ in lines.values()]

    def animate(frame):
        idx = frames_idx[frame]
        for dm, (ln, tr) in lines.items():
            ln.set_data(tr["t"][:idx+1], tr["f_true"][:idx+1])
        return [ln for ln, _ in lines.values()]

    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(frames_idx), interval=60, blit=True,
    )

    out_path = os.path.join(VIDEO_DIR, f"{attack_type}.gif")
    print(f"  Saving -> {out_path}")
    try:
        anim.save(out_path, writer="pillow", fps=18)
    except Exception as e:
        print(f"    [WARN] pillow writer failed ({e}); trying imagemagick...")
        try:
            anim.save(out_path, writer="imagemagick", fps=18)
        except Exception as e2:
            print(f"    [WARN] imagemagick failed too ({e2}); skipping.")
    plt.close(fig)


def export_all():
    attack_types = [at for at in ATTACK_UI_TO_ENV.keys() if at != "none"]
    print(f"Exporting {len(attack_types)} videos -> {VIDEO_DIR}")
    for at in attack_types:
        export_one(at)
    print("\nDone.")


if __name__ == "__main__":
    export_all()
