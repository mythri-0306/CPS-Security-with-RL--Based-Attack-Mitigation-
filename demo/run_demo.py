"""
Standalone power-grid attack/defense demo  (demo/run_demo.py)
=============================================================

A no-UI, command-line demonstration of the cyber-physical attack on the
simulated power-grid frequency-control loop, and how the defenses respond.

It uses the SAME real GridDefenseEnv / Kalman filter / baselines / DQN models
as the Streamlit app (via demo/sim_engine.py) -- no copied physics, no fake
numbers.  It just skips the browser UI and prints results to the terminal
plus an optional PNG plot.

Examples
--------
    # Spoof attack: undefended vs RL DQN, print summary + save plot
    python demo/run_demo.py --attack spoof

    # Replay attack, compare EVERY defense, your seeds from the screenshot
    python demo/run_demo.py --attack replay --intensity 10 --start 20 \
        --env-seed 46 --dqn-seed 1 --all

    # Stealthy drift, just the numbers, no plot
    python demo/run_demo.py --attack stealthy_drift --no-plot

Attacks   : none | spoof | stealthy_drift | replay | telemetry_freeze
Defenses  : No Defense | Rule-Based | Kalman-Only | Detector-Timing | RL DQN | Oracle
"""

import os
import sys
import argparse

_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_DEMO_DIR)
for p in (_ROOT_DIR, _DEMO_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

from sim_engine import (
    run_episode, DEFENSE_METHODS, ATTACK_UI_TO_ENV, INTENSITY_RANGES,
    NOMINAL_FREQ, SAFE_BAND_LOW, SAFE_BAND_HIGH,
)


# ---------------------------------------------------------------------------
def _fmt_delay(d):
    return "n/a" if d is None else f"{d} steps ({d * 0.1:.1f}s)"


def summarize(traj: dict) -> dict:
    """Pull the headline metrics out of a trajectory."""
    return {
        "defense": traj["defense_method"],
        "time_oob": traj["time_oob"],
        "max_dev": traj["max_deviation"],
        "det_delay": traj["detection_delay"],
        "clean_distrust": traj["clean_distrust_rate"],
        "return": traj["episode_return"],
    }


def print_table(rows):
    """Pretty fixed-width summary table."""
    hdr = (f"{'Defense':<16}{'Time OOB':>10}{'Max Dev':>11}"
           f"{'Det. Delay':>16}{'Clean Distrust':>16}{'Return':>10}")
    print("\n" + hdr)
    print("-" * len(hdr))
    # best time-oob (lower better) for a safety verdict
    best_oob = min(r["time_oob"] for r in rows)
    for r in rows:
        flag = "  <- safest" if r["time_oob"] == best_oob and len(rows) > 1 else ""
        print(f"{r['defense']:<16}"
              f"{r['time_oob']:>8.1f} s"
              f"{r['max_dev']:>9.3f} Hz"
              f"{_fmt_delay(r['det_delay']):>16}"
              f"{r['clean_distrust']:>14.1f} %"
              f"{r['return']:>10.1f}"
              f"{flag}")
    print()


def verdict(traj: dict):
    """One-line human verdict for a single run."""
    oob = traj["time_oob"]
    dev = traj["max_deviation"]
    name = traj["defense_method"]
    if oob == 0.0:
        return f"[OK]   {name}: frequency never left the safe band (max dev {dev:.3f} Hz)."
    return (f"[RISK] {name}: {oob:.1f}s outside the 49.8-50.2 Hz safe band, "
            f"peak deviation {dev:.3f} Hz.")


# ---------------------------------------------------------------------------
def make_plot(trajs, attack, intensity, out_path):
    """Save a comparison plot of true frequency for each run."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(plot skipped: matplotlib not available -- {e})")
        return None

    n = len(trajs)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.6 * n + 0.5),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]

    for ax, tr in zip(axes, trajs):
        t = tr["t"]
        # safe band
        ax.axhspan(SAFE_BAND_LOW, SAFE_BAND_HIGH, color="#2ecc71", alpha=0.10)
        ax.axhline(SAFE_BAND_LOW, color="#f1c40f", ls=":", lw=1)
        ax.axhline(SAFE_BAND_HIGH, color="#f1c40f", ls=":", lw=1)
        # attack window
        att = tr["attack_active"].astype(bool)
        if att.any():
            ax.axvspan(t[att][0], t[att][-1], color="#e74c3c", alpha=0.08)
            ax.axvline(t[att][0], color="#e74c3c", ls="--", lw=1, alpha=0.6)
        # signals
        ax.plot(t, tr["f_meas"], color="#e67e22", lw=1, ls=":",
                label="Sensor (measured)")
        ax.plot(t, tr["f_pred"], color="#16a085", lw=1, ls="--",
                label="Kalman prediction")
        ax.plot(t, tr["f_true"], color="#2980b9", lw=1.8, label="True frequency")
        ax.set_ylabel("Freq (Hz)")
        ax.set_ylim(min(49.5, tr["f_true"].min() - 0.1),
                    max(50.5, tr["f_true"].max() + 0.1))
        ax.set_title(f"{tr['defense_method']}   "
                     f"(OOB {tr['time_oob']:.1f}s, max dev {tr['max_deviation']:.3f} Hz)",
                     fontsize=10, loc="left")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, loc="lower left", ncol=3)

    axes[-1].set_xlabel("Time (s)")
    int_str = f" @ {intensity}" if intensity is not None else ""
    fig.suptitle(f"Power-grid '{attack}' attack{int_str} — defense comparison",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Standalone power-grid attack/defense demo (no UI).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--attack", default="spoof",
                    choices=list(ATTACK_UI_TO_ENV.keys()))
    ap.add_argument("--intensity", type=float, default=None,
                    help="Attack intensity (Hz / Hz-per-s / s). Default = env default.")
    ap.add_argument("--start", type=float, default=20.0,
                    help="Attack onset time in seconds.")
    ap.add_argument("--env-seed", type=int, default=42)
    ap.add_argument("--dqn-seed", type=int, default=0, choices=[0, 1, 2, 3, 4])
    ap.add_argument("--defenses", nargs="+", default=["No Defense", "RL DQN"],
                    help="Which defenses to run (ignored if --all).")
    ap.add_argument("--all", action="store_true",
                    help="Run every available defense.")
    ap.add_argument("--no-plot", action="store_true", help="Skip PNG plot.")
    ap.add_argument("--out", default=None, help="Plot output path.")
    args = ap.parse_args()

    defenses = DEFENSE_METHODS if args.all else args.defenses
    bad = [d for d in defenses if d not in DEFENSE_METHODS]
    if bad:
        ap.error(f"Unknown defense(s): {bad}. Choose from {DEFENSE_METHODS}")

    lo, hi = INTENSITY_RANGES.get(args.attack, (None, None))
    int_hint = "" if args.intensity is not None or args.attack in ("none", "telemetry_freeze") \
        else f"  (default; training range {lo}-{hi})"

    print("=" * 70)
    print(f"  POWER-GRID CPS ATTACK DEMO")
    print(f"  attack={args.attack}  intensity={args.intensity}{int_hint}"
          f"  start={args.start}s")
    print(f"  env_seed={args.env_seed}  dqn_seed={args.dqn_seed}")
    print("=" * 70)

    trajs, rows = [], []
    for dm in defenses:
        print(f"  running: {dm:<16} ...", end="", flush=True)
        try:
            tr = run_episode(args.attack, dm, args.env_seed,
                             args.intensity, args.start, args.dqn_seed)
        except Exception as e:
            print(f" FAILED ({e})")
            continue
        print(" done")
        trajs.append(tr)
        rows.append(summarize(tr))

    if not rows:
        print("No runs succeeded.")
        return 1

    print_table(rows)
    print("Verdict")
    print("-------")
    for tr in trajs:
        print("  " + verdict(tr))
    print()

    if not args.no_plot:
        out = args.out or os.path.join(
            _DEMO_DIR, f"demo_{args.attack}_seed{args.env_seed}.png")
        saved = make_plot(trajs, args.attack, args.intensity, out)
        if saved:
            print(f"Plot saved: {saved}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
