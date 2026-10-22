"""
demo/precompute.py  –  Pre-compute and cache trajectories for every
attack type × defence at a few fixed seeds so the Streamlit app can
run in "cached mode" without live inference.

Usage:  python demo/precompute.py
Output: demo/cache/*.pkl  (pickle files, one per run)
"""

import os, sys, pickle, itertools, time

_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_DEMO_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
if _DEMO_DIR not in sys.path:
    sys.path.insert(1, _DEMO_DIR)
_PG_DIR = os.path.join(_ROOT_DIR, "examples", "powergrid")
if os.path.exists(_PG_DIR) and _PG_DIR not in sys.path:
    sys.path.append(_PG_DIR)

from sim_engine import (
    run_episode, DEFENSE_METHODS, ATTACK_UI_TO_ENV, INTENSITY_RANGES,
)

CACHE_DIR = os.path.join(_DEMO_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Fixed seeds for the demo cache
DEMO_SEEDS = [42, 123, 456]

# Representative intensities (training-range midpoints)
DEFAULT_INTENSITIES = {
    "none":              None,
    "spoof":             48.50,
    "stealthy_drift":    0.05,
    "replay":            10.0,
    "telemetry_freeze":  None,    # not used by step()
}

DEFAULT_START = 20.0


def _cache_key(attack_type, defense_method, seed, intensity, attack_start):
    int_str = f"{intensity:.4f}" if intensity is not None else "default"
    return f"{attack_type}__{defense_method}__{seed}__{int_str}__{attack_start:.1f}.pkl"


def precompute_all():
    combos = list(itertools.product(
        ATTACK_UI_TO_ENV.keys(),   # attack types
        DEFENSE_METHODS,           # defences
        DEMO_SEEDS,                # seeds
    ))
    total = len(combos)
    print(f"Pre-computing {total} trajectories -> {CACHE_DIR}")
    t0 = time.time()

    for i, (at, dm, seed) in enumerate(combos, 1):
        intensity = DEFAULT_INTENSITIES.get(at)
        key = _cache_key(at, dm, seed, intensity, DEFAULT_START)
        out_path = os.path.join(CACHE_DIR, key)

        if os.path.exists(out_path):
            print(f"  [{i}/{total}] SKIP  {key}")
            continue

        print(f"  [{i}/{total}] RUN   {at:20s} | {dm:18s} | seed={seed} ...", end=" ", flush=True)
        try:
            result = run_episode(
                attack_type=at,
                defense_method=dm,
                env_seed=seed,
                intensity=intensity,
                attack_start=DEFAULT_START,
                dqn_seed=0,
            )
            with open(out_path, "wb") as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"OK  (OOB={result['time_oob']:.1f}s)")
        except Exception as e:
            print(f"FAIL  ({e})")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s.  Cached files: {len(os.listdir(CACHE_DIR))}")


if __name__ == "__main__":
    precompute_all()
