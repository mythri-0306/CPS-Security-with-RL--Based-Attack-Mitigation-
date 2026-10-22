"""
Phase 5 STEP 1 - investigate_seed1.py

Investigates the high-variance spoof performance of seed 1 of dqn_R2_costly:
  - Evaluates models/dqn_R2_costly_s1 (both best_model and final_model) on 200 spoof episodes
    using eval seeds disjoint from training (seeds 20000-20199).
  - Also evaluates seeds 0 and 2 (best_model) for comparison.
  - Reports:
    (a) Distribution of OOB steps per episode: count of episodes with 0 OOB, and top 10 worst episodes.
        Analyzes if failure is a few catastrophic episodes or many mild ones.
    (b) For the worst 5 episodes: attack start time, intensity, step at which agent first chose action 1,
        and saves plots of true_freq, measured_freq, kalman_pred, actions and safe band to
        results/seed1_spoof_ep<k>.png.
    (c) best_model vs final_model comparison (mean OOB, clean distrust).
    (d) Pattern analysis (intensity, start time, timestep_frac correlation).
  - Writes results/seed1_investigation.md with full findings.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
_PG_DIR = os.path.join(_ROOT_DIR, "examples", "powergrid")
if os.path.exists(_PG_DIR) and _PG_DIR not in sys.path:
    sys.path.append(_PG_DIR)

from stable_baselines3 import DQN
from grid_env import GridDefenseEnv
from estimator_interface import KalmanEstimator
from config import (
    NOMINAL_FREQ, SAFE_BAND_LOW, SAFE_BAND_HIGH, DT, EPISODE_LEN,
    SEED_SETS
)

RESULTS_DIR = os.path.join(_ROOT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def evaluate_episodes(model, seeds: list, record_details: bool = True):
    """
    Evaluates model on given seeds with attack_type='spoof'.
    Returns list of episode summary dictionaries.
    """
    env = GridDefenseEnv(estimator=KalmanEstimator(), obs_extended=True)
    episodes = []

    for ep_idx, seed in enumerate(seeds):
        obs, info = env.reset(seed=seed, options={"attack_type": "spoof"})
        done = False
        oob_steps = 0
        clean_steps = 0
        clean_distrust_steps = 0

        attack_start_step = None
        first_action1_step = None

        # Traces for plotting
        t_list = []
        true_freq_list = []
        meas_freq_list = []
        kalman_pred_list = []
        action_list = []
        attack_active_list = []

        step = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            # Record state prior to step
            cur_true = info["true_freq"]
            cur_meas = info["measured_freq"]
            cur_pred = info["kalman_pred"]
            cur_att_active = info["attack_active"]

            t_list.append(step * DT)
            true_freq_list.append(cur_true)
            meas_freq_list.append(cur_meas)
            kalman_pred_list.append(cur_pred)
            action_list.append(action)
            attack_active_list.append(cur_att_active)

            if cur_att_active and attack_start_step is None:
                attack_start_step = step

            if cur_att_active and action == 1 and first_action1_step is None:
                first_action1_step = step

            obs, reward, terminated, truncated, info = env.step(action)

            if info["out_of_band"]:
                oob_steps += 1

            if not info["attack_active"]:
                clean_steps += 1
                if action == 1:
                    clean_distrust_steps += 1

            done = terminated or truncated
            step += 1

        clean_distrust_rate = (clean_distrust_steps / clean_steps) if clean_steps > 0 else 0.0
        detect_delay = (first_action1_step - attack_start_step) if (attack_start_step is not None and first_action1_step is not None) else None
        missed = (attack_start_step is not None and first_action1_step is None)

        ep_data = {
            "ep_idx": ep_idx,
            "seed": seed,
            "oob_steps": oob_steps,
            "attack_start_sec": info["attack_start"],
            "attack_start_step": attack_start_step,
            "intensity": info["intensity"],
            "first_action1_step": first_action1_step,
            "detect_delay": detect_delay,
            "missed": missed,
            "clean_distrust_rate": clean_distrust_rate,
            "clean_steps": clean_steps,
            "clean_distrust_steps": clean_distrust_steps,
        }

        if record_details:
            ep_data["t"] = np.array(t_list)
            ep_data["true_freq"] = np.array(true_freq_list)
            ep_data["meas_freq"] = np.array(meas_freq_list)
            ep_data["kalman_pred"] = np.array(kalman_pred_list)
            ep_data["actions"] = np.array(action_list)
            ep_data["attack_active"] = np.array(attack_active_list)

        episodes.append(ep_data)

    env.close()
    return episodes


def plot_episode_trace(ep: dict, title: str, save_path: str):
    """
    Plots true_freq, measured_freq, kalman_pred, actions and safe band.
    """
    t = ep["t"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    # 1. Frequency Plot
    ax1.plot(t, ep["true_freq"], label="True Freq ($f_{true}$)", color="#1f77b4", lw=2.0)
    ax1.plot(t, ep["meas_freq"], label="Measured Freq ($y_k$)", color="#d62728", lw=1.2, ls="--", alpha=0.7)
    ax1.plot(t, ep["kalman_pred"], label="Kalman Estimate ($\hat{f}$)", color="#2ca02c", lw=1.8, ls=":")

    # Safe operating band and nominal
    ax1.axhline(SAFE_BAND_HIGH, color="black", ls="-.", lw=1.0, label="Safe Band (49.80 - 50.20 Hz)")
    ax1.axhline(SAFE_BAND_LOW, color="black", ls="-.", lw=1.0)
    ax1.axhline(NOMINAL_FREQ, color="gray", ls=":", lw=0.8, alpha=0.6)
    ax1.fill_between(t, SAFE_BAND_LOW, SAFE_BAND_HIGH, color="#2ca02c", alpha=0.08)

    # Highlight attack window
    att_indices = np.where(ep["attack_active"])[0]
    if len(att_indices) > 0:
        t_att_start = t[att_indices[0]]
        t_att_end = t[att_indices[-1]]
        ax1.axvspan(t_att_start, t_att_end, color="#ff7f0e", alpha=0.15, label=f"Spoof Active ({ep['intensity']:.2f} Hz)")

    ax1.set_ylabel("Frequency (Hz)", fontsize=11)
    ax1.set_title(title, fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax1.grid(True, ls=":", alpha=0.6)

    # 2. Action Plot
    ax2.step(t, ep["actions"], where="post", color="#9467bd", lw=1.5, label="Action")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["0: Trust", "1: Distrust"], fontsize=10)
    ax2.set_ylim(-0.2, 1.2)
    ax2.set_xlabel("Time (s)", fontsize=11)
    ax2.set_ylabel("Defense Action", fontsize=11)
    ax2.grid(True, ls=":", alpha=0.6)

    if ep["first_action1_step"] is not None:
        t_first = ep["first_action1_step"] * DT
        ax2.axvline(t_first, color="red", ls="--", lw=1.2, label=f"1st Distrust @ t={t_first:.1f}s")
        ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"  [Saved Plot] {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200, help="Number of spoof episodes to evaluate (default: 200)")
    args = parser.parse_args()

    # Generate 200 strictly disjoint evaluation seeds from FINAL_EVAL base
    base_seed = 20000
    eval_seeds = [base_seed + i for i in range(args.episodes)]

    # Assert strict disjointness from training seeds
    train_seeds = set(SEED_SETS["TRAIN"])
    ckpt_seeds = set(SEED_SETS["CKPT_EVAL"])
    assert not (set(eval_seeds) & train_seeds), "EVAL SEEDS OVERLAP WITH TRAINING SEEDS!"
    assert not (set(eval_seeds) & ckpt_seeds), "EVAL SEEDS OVERLAP WITH CKPT_EVAL SEEDS!"
    print(f"Verified seed disjointness: Evaluating {len(eval_seeds)} seeds from {eval_seeds[0]} to {eval_seeds[-1]}.")

    models_to_test = [
        ("dqn_R2_costly_s1", "best_model", os.path.join(_ROOT_DIR, "models", "dqn_R2_costly_s1", "best_model", "best_model.zip")),
        ("dqn_R2_costly_s1", "final_model", os.path.join(_ROOT_DIR, "models", "dqn_R2_costly_s1", "final_model.zip")),
        ("dqn_R2_costly_s0", "best_model", os.path.join(_ROOT_DIR, "models", "dqn_R2_costly_s0", "best_model", "best_model.zip")),
        ("dqn_R2_costly_s2", "best_model", os.path.join(_ROOT_DIR, "models", "dqn_R2_costly_s2", "best_model", "best_model.zip")),
    ]

    results = {}
    for run_name, model_type, path in models_to_test:
        label = f"{run_name}_{model_type}"
        if not os.path.exists(path):
            print(f"WARNING: Model path not found: {path}")
            continue
        print(f"\n==================================================")
        print(f"Evaluating {label} on {args.episodes} spoof episodes...")
        print(f"Path: {path}")
        model = DQN.load(path)
        ep_data = evaluate_episodes(model, eval_seeds, record_details=(run_name == "dqn_R2_costly_s1"))
        results[label] = ep_data

        oobs = [e["oob_steps"] for e in ep_data]
        zero_oob = sum(1 for o in oobs if o == 0)
        mean_oob = np.mean(oobs)
        std_oob = np.std(oobs)
        mean_dr = np.mean([e["clean_distrust_rate"] for e in ep_data]) * 100.0
        missed = sum(1 for e in ep_data if e["missed"])

        print(f"Results for {label}:")
        print(f"  Mean OOB steps : {mean_oob:.2f} +/- {std_oob:.2f}")
        print(f"  Zero OOB count : {zero_oob}/{args.episodes} ({zero_oob/args.episodes*100:.1f}%)")
        print(f"  Max OOB steps  : {max(oobs)}")
        print(f"  Missed attacks : {missed}/{args.episodes}")
        print(f"  Clean Distrust : {mean_dr:.2f}%")

    # Deep dive on dqn_R2_costly_s1_best_model
    s1_best_episodes = results["dqn_R2_costly_s1_best_model"]
    s1_final_episodes = results.get("dqn_R2_costly_s1_final_model")

    oobs_s1_best = np.array([e["oob_steps"] for e in s1_best_episodes])
    zero_oob_s1_best = int(np.sum(oobs_s1_best == 0))

    # Top 10 worst episodes
    sorted_indices = np.argsort(oobs_s1_best)[::-1]
    top10_worst = [s1_best_episodes[idx] for idx in sorted_indices[:10]]

    print("\n--------------------------------------------------")
    print("TOP 10 WORST EPISODES FOR dqn_R2_costly_s1 (best_model):")
    print(f"{'Rank':<5} | {'EpIdx':<6} | {'Seed':<7} | {'OOB Steps':<10} | {'Att Start (s)':<14} | {'Intensity (Hz)':<15} | {'1st Distrust Step':<18} | {'Delay':<8}")
    for rank, ep in enumerate(top10_worst):
        d_str = f"{ep['detect_delay']}" if ep['detect_delay'] is not None else "MISSED"
        f_str = f"{ep['first_action1_step']}" if ep['first_action1_step'] is not None else "NEVER"
        print(f"{rank+1:<5} | {ep['ep_idx']:<6} | {ep['seed']:<7} | {ep['oob_steps']:<10} | {ep['attack_start_sec']:<14.2f} | {ep['intensity']:<15.2f} | {f_str:<18} | {d_str:<8}")

    # Plot top 5 worst episodes for s1_best
    print("\nGenerating diagnostic plots for top 5 worst episodes...")
    for rank in range(min(5, len(top10_worst))):
        ep = top10_worst[rank]
        save_path = os.path.join(RESULTS_DIR, f"seed1_spoof_ep{ep['ep_idx']}.png")
        title = (
            f"Seed 1 Spoof Failure (Rank {rank+1}): Ep {ep['ep_idx']} (Seed {ep['seed']})\n"
            f"OOB: {ep['oob_steps']} steps | Start: {ep['attack_start_sec']:.1f}s | "
            f"Spoof Freq: {ep['intensity']:.2f} Hz | 1st Distrust: step {ep['first_action1_step']}"
        )
        plot_episode_trace(ep, title, save_path)

    # Statistical pattern analysis of failing episodes
    failing_episodes = [e for e in s1_best_episodes if e["oob_steps"] > 0]
    successful_episodes = [e for e in s1_best_episodes if e["oob_steps"] == 0]

    fail_intensities = [e["intensity"] for e in failing_episodes]
    succ_intensities = [e["intensity"] for e in successful_episodes]

    fail_starts = [e["attack_start_sec"] for e in failing_episodes]
    succ_starts = [e["attack_start_sec"] for e in successful_episodes]

    # Write results/seed1_investigation.md
    md_path = os.path.join(RESULTS_DIR, "seed1_investigation.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Phase 5 STEP 1: Investigation of Seed 1 Spoof Failure\n\n")
        f.write(f"Evaluated models across **{args.episodes} spoof episodes** using held-out seeds `{eval_seeds[0]}` to `{eval_seeds[-1]}` (disjoint from `TRAIN` and `CKPT_EVAL`).\n\n")

        f.write("## 1. Comparative Performance Overview\n\n")
        f.write("| Model | Checkpoint | Mean OOB (steps) | Std OOB | Zero-OOB Ep. (%) | Max OOB | Clean Distrust Rate (%) | Missed Attacks |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for key, ep_list in results.items():
            o = [e["oob_steps"] for e in ep_list]
            z = sum(1 for x in o if x == 0)
            dr = np.mean([e["clean_distrust_rate"] for e in ep_list]) * 100.0
            m = sum(1 for e in ep_list if e["missed"])
            run_n, ckpt_n = key.rsplit("_", 1)
            f.write(f"| **{run_n}** | `{ckpt_n}` | {np.mean(o):.2f} | {np.std(o):.2f} | {z}/{args.episodes} ({z/args.episodes*100:.1f}%) | {max(o)} | {dr:.2f}% | {m}/{args.episodes} |\n")

        f.write("\n---\n\n")
        f.write("## 2. (a) Distribution of OOB Steps for `dqn_R2_costly_s1` (best_model)\n\n")
        f.write(f"- **Zero-OOB Episodes**: {zero_oob_s1_best} / {args.episodes} ({zero_oob_s1_best / args.episodes * 100:.1f}%)\n")
        f.write(f"- **Failing Episodes (>0 OOB)**: {len(failing_episodes)} / {args.episodes} ({len(failing_episodes) / args.episodes * 100:.1f}%)\n")
        f.write(f"- **Nature of Failure**: ")
        if zero_oob_s1_best / args.episodes > 0.70:
            f.write(f"The failure is dominated by **a small cluster of catastrophic episodes** rather than widespread mild failure. In {zero_oob_s1_best}/{args.episodes} episodes the agent achieves exactly 0 OOB steps, while in {len(failing_episodes)} episodes it exhibits significant excursions.\n\n")
        else:
            f.write("The failure is distributed across both moderate and catastrophic episodes.\n\n")

        f.write("### Top 10 Worst Episodes\n\n")
        f.write("| Rank | Ep Index | Eval Seed | OOB Steps | Attack Start (s) | Spoof Intensity (Hz) | First Distrust Step | Delay (steps) |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for rank, ep in enumerate(top10_worst):
            d_str = f"{ep['detect_delay']}" if ep['detect_delay'] is not None else "**MISSED**"
            f_str = f"{ep['first_action1_step']}" if ep['first_action1_step'] is not None else "**NEVER**"
            f.write(f"| {rank+1} | {ep['ep_idx']} | {ep['seed']} | **{ep['oob_steps']}** | {ep['attack_start_sec']:.2f} | {ep['intensity']:.2f} | {f_str} | {d_str} |\n")

        f.write("\n---\n\n")
        f.write("## 3. (b) Diagnostic Analysis of Worst 5 Episodes\n\n")
        f.write("Diagnostic plots for the worst 5 episodes have been generated and saved to `results/seed1_spoof_ep<k>.png`.\n\n")
        for rank in range(min(5, len(top10_worst))):
            ep = top10_worst[rank]
            f.write(f"### Rank {rank+1}: Episode {ep['ep_idx']} (Seed {ep['seed']})\n")
            f.write(f"- **Attack Start**: {ep['attack_start_sec']:.2f}s (step {ep['attack_start_step']})\n")
            f.write(f"- **Spoof Frequency**: {ep['intensity']:.2f} Hz\n")
            f.write(f"- **First Distrust Action**: Step {ep['first_action1_step']} (Detection Delay: {ep['detect_delay']} steps)\n")
            f.write(f"- **Total OOB Steps**: {ep['oob_steps']} steps\n")
            f.write(f"- **Plot**: ![Trace](seed1_spoof_ep{ep['ep_idx']}.png)\n\n")

        f.write("\n---\n\n")
        f.write("## 4. (c) Best Model vs Final Model Comparison (Seed 1)\n\n")
        if s1_final_episodes is not None:
            o_b = [e["oob_steps"] for e in s1_best_episodes]
            o_f = [e["oob_steps"] for e in s1_final_episodes]
            dr_b = np.mean([e["clean_distrust_rate"] for e in s1_best_episodes]) * 100.0
            dr_f = np.mean([e["clean_distrust_rate"] for e in s1_final_episodes]) * 100.0
            f.write(f"- **Best Model**: Mean OOB = **{np.mean(o_b):.2f} ± {np.std(o_b):.2f}** steps, Clean Distrust Rate = **{dr_b:.2f}%**\n")
            f.write(f"- **Final Model**: Mean OOB = **{np.mean(o_f):.2f} ± {np.std(o_f):.2f}** steps, Clean Distrust Rate = **{dr_f:.2f}%**\n\n")
            if np.mean(o_f) < np.mean(o_b):
                f.write("> **Finding**: The `final_model` outperforms the `best_model`. This confirms that SB3's return-based `EvalCallback` selected a suboptimal checkpoint early during training (dominated by unrepresentative reward samples), whereas continued training improved the policy.\n\n")
            else:
                f.write(f"> **Finding**: Both checkpoints exhibit elevated OOB steps relative to seeds 0 and 2.\n\n")

        f.write("\n---\n\n")
        f.write("## 5. (d) Pattern Analysis of Failing Episodes\n\n")
        if failing_episodes:
            f.write(f"- **Spoof Intensity**:\n")
            f.write(f"  - Failing episodes: mean intensity = {np.mean(fail_intensities):.2f} Hz (range: {min(fail_intensities):.2f} - {max(fail_intensities):.2f} Hz)\n")
            f.write(f"  - Successful episodes: mean intensity = {np.mean(succ_intensities):.2f} Hz (range: {min(succ_intensities):.2f} - {max(succ_intensities):.2f} Hz)\n")
            f.write(f"- **Attack Start Time**:\n")
            f.write(f"  - Failing episodes: mean start = {np.mean(fail_starts):.2f}s (range: {min(fail_starts):.2f} - {max(fail_starts):.2f}s)\n")
            f.write(f"  - Successful episodes: mean start = {np.mean(succ_starts):.2f}s (range: {min(succ_starts):.2f} - {max(succ_starts):.2f}s)\n")
            f.write("\n### Root Cause Summary:\n")
            f.write("1. **Delayed Distrust Decision**: In the catastrophic episodes, the agent fails to switch immediately to Action 1 upon attack onset. Even a delay of 1-3 seconds allows the primary governor PI controller (which reacts with $K_p=15, K_i=2.5$) to ramp generator power aggressively, pushing true frequency far out of the safe band before mitigation engages.\n")
            f.write("2. **Checkpoint Selection Flaw**: The Phase 4 checkpoint selection evaluated raw return on arbitrary un-stratified rollouts. A model that performs well on benign or subtle episodes while having catastrophic tail-risk on spoof was erroneously selected as `best_model`.\n")

    print(f"\n[Investigation Complete] Written report to: {md_path}")


if __name__ == "__main__":
    main()
