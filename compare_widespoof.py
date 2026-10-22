"""
Fair Widened-Spoof Comparison (compare_widespoof.py)

Evaluates the ORIGINAL 5 DQN models and the WIDENED 5 models on the exact
same test sets (100 episodes per seed per test set, identical scenarios):
  (a) Standard spoof range [48.00, 49.20] Hz (FINAL_EVAL seeds 20100..20199)
  (b) Old held-out spoof range [47.00, 47.90] Hz (FINAL_EVAL seeds 20700..20799)
  (c) New held-out spoof range [45.50, 46.40] Hz (FINAL_EVAL seeds 20700..20799)

Outputs:
  - results/widespoof_fair.md
  - results/widespoof_fair.csv
"""

import os
import sys
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

_ROOT_DIR = r"c:\Users\harsh\Desktop\icps_project"
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
_PG_DIR = os.path.join(_ROOT_DIR, "examples", "powergrid")
if os.path.exists(_PG_DIR) and _PG_DIR not in sys.path:
    sys.path.append(_PG_DIR)

from stable_baselines3 import DQN
from grid_env import GridDefenseEnv
from estimator_interface import KalmanEstimator
from config import (
    SEED_SETS, DT,
    ATTACK_INTENSITY_RANGES, ATTACK_START_TIME_RANGE, ATTACK_DEFAULT_DURATION,
    HELDOUT_START_TIME_RANGE, HELDOUT_DEFAULT_DURATION, HELDOUT_ATTACK_INTENSITY_RANGES,
    WIDE_SPOOF_RANGE, HELDOUT_SPOOF_ADJUSTED_RANGE, REWARD_VARIANTS
)

RESULTS_DIR = os.path.join(_ROOT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

ORIG_TRAIN_RANGE = ATTACK_INTENSITY_RANGES["spoof"]  # (48.00, 49.20)
WIDE_TRAIN_RANGE = WIDE_SPOOF_RANGE                  # (46.50, 49.20)

TEST_SETS = {
    "standard": {
        "name": "Standard Spoof [48.00 - 49.20 Hz]",
        "range": ATTACK_INTENSITY_RANGES["spoof"],
        "start_range": ATTACK_START_TIME_RANGE,
        "duration": ATTACK_DEFAULT_DURATION,
        "seeds": SEED_SETS["FINAL_EVAL"][100:200],  # 20100..20199
    },
    "old_heldout": {
        "name": "Old Held-Out Spoof [47.00 - 47.90 Hz]",
        "range": HELDOUT_ATTACK_INTENSITY_RANGES["spoof"],
        "start_range": HELDOUT_START_TIME_RANGE,
        "duration": HELDOUT_DEFAULT_DURATION,
        "seeds": SEED_SETS["FINAL_EVAL"][700:800],  # 20700..20799
    },
    "new_heldout": {
        "name": "New Held-Out Spoof [45.50 - 46.40 Hz]",
        "range": HELDOUT_SPOOF_ADJUSTED_RANGE,
        "start_range": HELDOUT_START_TIME_RANGE,
        "duration": HELDOUT_DEFAULT_DURATION,
        "seeds": SEED_SETS["FINAL_EVAL"][700:800],  # 20700..20799
    },
}

def check_overlap(train_range: Tuple[float, float], test_range: Tuple[float, float]) -> Tuple[bool, str]:
    overlap = not (train_range[1] < test_range[0] or train_range[0] > test_range[1])
    if overlap:
        if train_range[0] <= test_range[0] and train_range[1] >= test_range[1]:
            desc = "OVERLAP (In-Distribution / Full Coverage)"
        else:
            desc = "OVERLAP (Partial overlap)"
    else:
        desc = "NO OVERLAP (Strictly Out-of-Distribution)"
    return overlap, desc

def load_models() -> Tuple[Dict[int, DQN], Dict[int, DQN]]:
    orig_models = {}
    wide_models = {}
    for s in range(5):
        orig_p = os.path.join(_ROOT_DIR, "models", f"dqn_R2_costly_s{s}_v2", "best_model", "best_model.zip")
        wide_p = os.path.join(_ROOT_DIR, "models", f"dqn_R2_costly_s{s}_widespoof", "best_model.zip")
        if not os.path.exists(orig_p):
            raise FileNotFoundError(f"Original model not found: {orig_p}")
        if not os.path.exists(wide_p):
            raise FileNotFoundError(f"Widened model not found: {wide_p}")
        orig_models[s] = DQN.load(orig_p)
        wide_models[s] = DQN.load(wide_p)
    return orig_models, wide_models

def run_comparison():
    print("=" * 80)
    print("FAIR WIDENED-SPOOF EVALUATION")
    print("=" * 80)
    print(f"Original DQN Training Range: {ORIG_TRAIN_RANGE[0]:.2f} - {ORIG_TRAIN_RANGE[1]:.2f} Hz")
    print(f"Widened DQN Training Range : {WIDE_TRAIN_RANGE[0]:.2f} - {WIDE_TRAIN_RANGE[1]:.2f} Hz")
    print("-" * 80)

    for ts_key, ts_info in TEST_SETS.items():
        _, orig_desc = check_overlap(ORIG_TRAIN_RANGE, ts_info["range"])
        _, wide_desc = check_overlap(WIDE_TRAIN_RANGE, ts_info["range"])
        print(f"Test Set '{ts_key}': {ts_info['name']}")
        print(f"  Original Training ({ORIG_TRAIN_RANGE[0]:.2f}-{ORIG_TRAIN_RANGE[1]:.2f} Hz): {orig_desc}")
        print(f"  Widened Training  ({WIDE_TRAIN_RANGE[0]:.2f}-{WIDE_TRAIN_RANGE[1]:.2f} Hz): {wide_desc}")
    print("=" * 80 + "\n")

    orig_models, wide_models = load_models()
    eval_env = GridDefenseEnv(
        estimator=KalmanEstimator(),
        obs_extended=True,
        reward_cfg=REWARD_VARIANTS["R2_costly"]
    )

    results_summary = {ts_key: {"orig": {s: [] for s in range(5)}, "wide": {s: [] for s in range(5)}} for ts_key in TEST_SETS}
    csv_rows = []

    for ts_key, ts_info in TEST_SETS.items():
        print(f"\nEvaluating Test Set: {ts_info['name']} (100 episodes per seed)...")
        seeds = ts_info["seeds"]
        s_min, s_max = ts_info["start_range"]
        i_min, i_max = ts_info["range"]
        dur = ts_info["duration"]

        scenarios = []
        for seed_val in seeds:
            rng = np.random.default_rng(seed_val)
            scenarios.append({
                "seed": seed_val,
                "attack_type": "spoof",
                "attack_start_time": float(rng.uniform(s_min, s_max)),
                "attack_duration": dur,
                "intensity": float(rng.uniform(i_min, i_max)),
            })

        for s in range(5):
            orig_m = orig_models[s]
            wide_m = wide_models[s]

            # Original Model
            for ep_idx, sc in enumerate(scenarios):
                obs, info = eval_env.reset(seed=sc["seed"], options=sc)
                done = False
                oob_steps = 0
                while not done:
                    act, _ = orig_m.predict(obs, deterministic=True)
                    obs, _, term, trunc, info = eval_env.step(int(act))
                    if info["out_of_band"]:
                        oob_steps += 1
                    done = term or trunc
                oob_sec = oob_steps * DT
                results_summary[ts_key]["orig"][s].append(oob_sec)
                csv_rows.append({
                    "test_set": ts_key,
                    "model_group": "original",
                    "seed": s,
                    "episode": ep_idx,
                    "scenario_seed": sc["seed"],
                    "intensity": sc["intensity"],
                    "oob_sec": oob_sec
                })

            # Widened Model
            for ep_idx, sc in enumerate(scenarios):
                obs, info = eval_env.reset(seed=sc["seed"], options=sc)
                done = False
                oob_steps = 0
                while not done:
                    act, _ = wide_m.predict(obs, deterministic=True)
                    obs, _, term, trunc, info = eval_env.step(int(act))
                    if info["out_of_band"]:
                        oob_steps += 1
                    done = term or trunc
                oob_sec = oob_steps * DT
                results_summary[ts_key]["wide"][s].append(oob_sec)
                csv_rows.append({
                    "test_set": ts_key,
                    "model_group": "widened",
                    "seed": s,
                    "episode": ep_idx,
                    "scenario_seed": sc["seed"],
                    "intensity": sc["intensity"],
                    "oob_sec": oob_sec
                })

            orig_mean = np.mean(results_summary[ts_key]["orig"][s])
            wide_mean = np.mean(results_summary[ts_key]["wide"][s])
            print(f"  Seed {s}: Orig OOB = {orig_mean:.2f}s | Widened OOB = {wide_mean:.2f}s")

    eval_env.close()

    df_csv = pd.DataFrame(csv_rows)
    csv_path = os.path.join(RESULTS_DIR, "widespoof_fair.csv")
    df_csv.to_csv(csv_path, index=False)
    print(f"\n[Saved CSV] {csv_path}")

    generate_markdown(results_summary)

def generate_markdown(results_summary):
    md_path = os.path.join(RESULTS_DIR, "widespoof_fair.md")
    lines = [
        "# Fair Widened-Spoof Comparison\n\n",
        "Evaluates the **Original 5 DQN models** and the **Widened 5 DQN models** on the **exact same test scenarios** (100 episodes per seed per test set, identical random attack parameters and seeds). No retraining was performed.\n\n",
        "## 1. Test Sets & Training Range Overlap\n\n",
        "| Test Set | Test Frequency Range (Hz) | Original Training [48.00, 49.20] Hz | Widened Training [46.50, 49.20] Hz |\n",
        "| :--- | :---: | :---: | :---: |\n",
    ]

    for ts_key, ts_info in TEST_SETS.items():
        _, orig_desc = check_overlap(ORIG_TRAIN_RANGE, ts_info["range"])
        _, wide_desc = check_overlap(WIDE_TRAIN_RANGE, ts_info["range"])
        lines.append(f"| **{ts_info['name']}** | {ts_info['range'][0]:.2f} - {ts_info['range'][1]:.2f} Hz | {orig_desc} | {wide_desc} |\n")

    lines.append("\n## 2. Per-Seed Safe Band Out-of-Band Time (seconds, Mean +/- Std over 100 episodes)\n\n")
    lines.append("| Seed | (a) Standard: Orig | (a) Standard: Wide | (b) Old Held-Out: Orig | (b) Old Held-Out: Wide | (c) New Held-Out: Orig | (c) New Held-Out: Wide |\n")
    lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")

    col_means = {k: {"orig": [], "wide": []} for k in TEST_SETS}

    for s in range(5):
        row = [f"**Seed {s}**"]
        for ts_key in ["standard", "old_heldout", "new_heldout"]:
            o_arr = results_summary[ts_key]["orig"][s]
            w_arr = results_summary[ts_key]["wide"][s]
            o_m, o_s = np.mean(o_arr), np.std(o_arr)
            w_m, w_s = np.mean(w_arr), np.std(w_arr)
            col_means[ts_key]["orig"].append(o_m)
            col_means[ts_key]["wide"].append(w_m)
            row.append(f"{o_m:.2f} +/- {o_s:.2f}s")
            row.append(f"{w_m:.2f} +/- {w_s:.2f}s")
        lines.append("| " + " | ".join(row) + " |\n")

    summary_row = ["**Mean +/- Std**"]
    for ts_key in ["standard", "old_heldout", "new_heldout"]:
        o_list = col_means[ts_key]["orig"]
        w_list = col_means[ts_key]["wide"]
        summary_row.append(f"**{np.mean(o_list):.2f} +/- {np.std(o_list):.2f}s**")
        summary_row.append(f"**{np.mean(w_list):.2f} +/- {np.std(w_list):.2f}s**")
    lines.append("| " + " | ".join(summary_row) + " |\n\n")

    lines.append("## 3. Computed Comparative Verdicts per Test Set\n\n")

    for ts_key, ts_name in [
        ("standard", "(a) Standard Spoof [48.00 - 49.20 Hz]"),
        ("old_heldout", "(b) Old Held-Out Spoof [47.00 - 47.90 Hz]"),
        ("new_heldout", "(c) New Held-Out Spoof [45.50 - 46.40 Hz]")
    ]:
        o_list = col_means[ts_key]["orig"]
        w_list = col_means[ts_key]["wide"]
        o_m = float(np.mean(o_list))
        o_s = float(np.std(o_list))
        w_m = float(np.mean(w_list))
        w_s = float(np.std(w_list))

        diffs = [round(float(w - o), 2) for o, w in zip(o_list, w_list)]
        improved = sum(1 for d in diffs if d < -1e-6)
        worsened = sum(1 for d in diffs if d > 1e-6)
        unchanged = len(diffs) - improved - worsened

        if w_m < o_m and improved >= 4:
            verdict = "better"
        elif w_m > o_m and worsened >= 4:
            verdict = "worse"
        else:
            verdict = "inconclusive"

        diffs_str = "[" + ", ".join([f"{d:+.2f}" for d in diffs]) + "]"

        lines.append(f"### {ts_name}\n")
        lines.append(f"- **Original DQN Mean OOB:** {o_m:.2f} +/- {o_s:.2f}s\n")
        lines.append(f"- **Widened DQN Mean OOB :** {w_m:.2f} +/- {w_s:.2f}s\n")
        lines.append(f"- **Per-Seed Differences (wide minus orig):** {diffs_str}\n")
        lines.append(f"- **Seed Improvement Counts:** {improved} improved, {worsened} worsened, {unchanged} unchanged\n")
        lines.append(f"- **Verdict:** **{verdict}** (per-seed diffs: {diffs_str}; {improved}/5 seeds improved, {worsened}/5 worsened)\n")
        if ts_key == "old_heldout":
            lines.append(
                "- **Note on Test Set (b):** Test set (b) lies inside the widened training range "
                f"[{WIDE_TRAIN_RANGE[0]:.2f}, {WIDE_TRAIN_RANGE[1]:.2f}] Hz, so it is not a held-out test for that group.\n"
            )
        lines.append("\n")

    with open(md_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"\n[Saved Fair Widespoof Report] {md_path}")

if __name__ == "__main__":
    run_comparison()
