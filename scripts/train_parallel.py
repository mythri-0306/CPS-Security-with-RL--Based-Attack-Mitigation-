"""
Parallel Training Script for Phase 5 Models

Trains DQN R2_costly seeds 0, 1, 2, 3, 4 and PPO seeds 1, 2 in parallel.
Uses Python multiprocessing to leverage available CPU cores.
"""

import os
import sys
import subprocess
from concurrent.futures import ProcessPoolExecutor

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def train_job(task):
    algo, seed, timesteps = task
    run_name = f"{algo}_R2_costly_s{seed}_v2"
    cmd = [
        sys.executable,
        os.path.join(_ROOT_DIR, "train_dqn.py"),
        "--algo", algo,
        "--reward-variant", "R2_costly",
        "--seed", str(seed),
        "--timesteps", str(timesteps),
        "--estimator", "kalman",
        "--run-name", run_name,
        "--obs-extended",
    ]
    print(f"[Starting {run_name}] {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=_ROOT_DIR, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[ERROR in {run_name}] Exit code {res.returncode}:\n{res.stderr[-500:]}")
    else:
        print(f"[COMPLETED {run_name}] Successfully trained and saved.")
    return run_name, res.returncode


def main():
    timesteps = 200000
    tasks = [
        ("dqn", 0, timesteps),
        ("dqn", 1, timesteps),
        ("dqn", 2, timesteps),
        ("dqn", 3, timesteps),
        ("dqn", 4, timesteps),
        ("ppo", 1, timesteps),
        ("ppo", 2, timesteps),
    ]

    print(f"Launching {len(tasks)} parallel training jobs on 32-core host...")
    with ProcessPoolExecutor(max_workers=len(tasks)) as executor:
        results = list(executor.map(train_job, tasks))

    print("\n--- All parallel training jobs completed! ---")
    for r_name, code in results:
        print(f"  {r_name}: {'SUCCESS' if code == 0 else f'FAILED ({code})'}")


if __name__ == "__main__":
    main()
