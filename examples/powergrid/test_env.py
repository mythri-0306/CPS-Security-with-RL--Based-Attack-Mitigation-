"""
Phase 1, 2 & 3 Environment Tests - test_env.py

Executes validation checks on GridDefenseEnv:
0. Gymnasium check_env compliance verification
1. Check (a): reset/step shapes and dtypes match observation and action spaces
2. Check (b): Determinism / reproducibility with identical seed
3. Check (c): Full random-action rollout runs to EPISODE_LEN (600 steps) without error
4. Check (d): Reward is negative when out_of_band is True and zero when in band with action 0
5. Phase 3 Test (a): Kalman estimator determinism across identical seeds & reset behavior
6. Phase 3 Test (b): Observation never contains true_freq (no ground-truth leakage)
7. Phase 3 Test (c): Stealthy slow-drift attack residual is non-trivially above clean baseline
"""

import os
import sys
import numpy as np

# Ensure local directory and examples/powergrid are in sys.path
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

_PG_DIR = os.path.join(_CURRENT_DIR, "examples", "powergrid")
if os.path.exists(_PG_DIR) and _PG_DIR not in sys.path:
    sys.path.insert(0, _PG_DIR)

from gymnasium.utils.env_checker import check_env
from grid_env import GridDefenseEnv
from estimator_interface import KalmanEstimator, PlaceholderEstimator
from config import EPISODE_LEN, W_BAND, W_DISTRUST, NOMINAL_FREQ, OBS_FREQ_NORM_SCALE


def test_gymnasium_check_env():
    """Run gymnasium's official check_env utility to verify standard compliance."""
    print("======================================================================")
    print("Running Gymnasium check_env with KalmanEstimator...")
    env = GridDefenseEnv(estimator=KalmanEstimator())
    check_env(env)
    print("PASSED: check_env completed with zero errors.")
    print("======================================================================\n")


def test_shapes_and_dtypes():
    """Check (a): reset and step shapes and dtypes match spaces, info keys present."""
    print("Check (a): Verifying reset/step shapes, dtypes, and spaces...")
    env_ext = GridDefenseEnv(estimator=KalmanEstimator(), obs_extended=True)
    env_base = GridDefenseEnv(estimator=KalmanEstimator(), obs_extended=False)

    assert env_ext.observation_space.shape == (8,), f"Extended obs shape {env_ext.observation_space.shape} != (8,)"
    assert env_base.observation_space.shape == (5,), f"Base obs shape {env_base.observation_space.shape} != (5,)"

    # Reset test on extended env
    obs, info = env_ext.reset(seed=42)
    assert isinstance(obs, np.ndarray), f"Expected np.ndarray, got {type(obs)}"
    assert obs.shape == (8,), f"Obs shape {obs.shape} != (8,)"
    assert obs.dtype == env_ext.observation_space.dtype, f"Obs dtype {obs.dtype} != {env_ext.observation_space.dtype}"
    assert env_ext.observation_space.contains(obs), f"Observation {obs} not in observation_space"
    assert isinstance(info, dict), f"Info must be dict, got {type(info)}"

    # Required info keys (including Phase 4 additions)
    required_keys = [
        "true_freq", "measured_freq", "attack_active", "attack_type",
        "attack_start", "intensity", "out_of_band", "r_band", "r_distrust",
        "r_distrust_attack", "r_switch", "packet_age", "duplicate_flag"
    ]
    for k in required_keys:
        assert k in info, f"Missing required info key '{k}' in reset info"

    # Step test
    action = env_ext.action_space.sample()
    next_obs, reward, terminated, truncated, next_info = env_ext.step(action)

    assert isinstance(next_obs, np.ndarray), f"Expected np.ndarray, got {type(next_obs)}"
    assert next_obs.shape == (8,), f"Next obs shape {next_obs.shape} != (8,)"
    assert next_obs.dtype == env_ext.observation_space.dtype, f"Next obs dtype {next_obs.dtype} != {env_ext.observation_space.dtype}"
    assert env_ext.observation_space.contains(next_obs), f"Next obs {next_obs} not in observation_space"

    assert isinstance(reward, (float, np.floating)), f"Reward must be float, got {type(reward)}"
    assert isinstance(terminated, (bool, np.bool_)), f"Terminated must be bool, got {type(terminated)}"
    assert isinstance(truncated, (bool, np.bool_)), f"Truncated must be bool, got {type(truncated)}"
    assert not terminated, "Terminated should be False for fixed-length episodes"

    for k in required_keys:
        assert k in next_info, f"Missing required info key '{k}' in step info"

    print("  - Extended Obs shape : (8,)")
    print("  - Base Obs shape     : (5,)")
    print("  - Observation dtype  : float32")
    print(f"  - Initial obs sample : {obs}")
    print(f"  - Action space       : {env_ext.action_space}")
    print(f"  - Required info keys : {required_keys}")
    print("PASSED: Check (a) verified successfully.\n")


def test_seed_reproducibility():
    """Check (b): Same seed gives identical rollouts."""
    print("Check (b): Verifying seed reproducibility across identical rollouts...")
    env1 = GridDefenseEnv(estimator=KalmanEstimator())
    env2 = GridDefenseEnv(estimator=KalmanEstimator())

    seed = 101
    obs1, info1 = env1.reset(seed=seed)
    obs2, info2 = env2.reset(seed=seed)

    np.testing.assert_array_equal(obs1, obs2, err_msg="Initial observations differ with same seed!")
    assert info1["attack_type"] == info2["attack_type"], "Attack types differ with same seed!"
    assert info1["attack_start"] == info2["attack_start"], "Attack start times differ with same seed!"
    assert info1["intensity"] == info2["intensity"], "Attack intensities differ with same seed!"

    # Run 50 identical steps
    actions = [i % 2 for i in range(50)]
    for step_idx, act in enumerate(actions):
        o1, r1, term1, trunc1, inf1 = env1.step(act)
        o2, r2, term2, trunc2, inf2 = env2.step(act)

        np.testing.assert_array_equal(
            o1, o2,
            err_msg=f"Observations diverged at step {step_idx} with same seed!"
        )
        assert r1 == r2, f"Rewards diverged at step {step_idx}"
        assert term1 == term2, f"Terminated diverged at step {step_idx}"
        assert trunc1 == trunc2, f"Truncated diverged at step {step_idx}"
        assert inf1["true_freq"] == inf2["true_freq"], f"True freq diverged at step {step_idx}"
        assert inf1["measured_freq"] == inf2["measured_freq"], f"Measured freq diverged at step {step_idx}"
        assert inf1["attack_active"] == inf2["attack_active"], f"Attack active flag diverged at step {step_idx}"
        assert inf1["r_band"] == inf2["r_band"], f"r_band diverged at step {step_idx}"
        assert inf1["r_distrust"] == inf2["r_distrust"], f"r_distrust diverged at step {step_idx}"

    print(f"  - Tested 50 identical steps with seed={seed}.")
    print("  - Trajectories (obs, rewards, info, flags) were 100% bitwise identical.")
    print("PASSED: Check (b) verified successfully.\n")


def test_random_action_full_rollout():
    """Check (c): A random-action episode runs to EPISODE_LEN without error."""
    print(f"Check (c): Verifying full random-action rollout to EPISODE_LEN={EPISODE_LEN}...")
    env = GridDefenseEnv(estimator=KalmanEstimator(), episode_len=EPISODE_LEN)
    obs, info = env.reset(seed=999)

    step_count = 0
    truncated = False
    terminated = False

    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1

    assert truncated is True, f"Expected truncated=True at end of episode, got {truncated}"
    assert terminated is False, f"Expected terminated=False, got {terminated}"
    assert step_count == EPISODE_LEN, f"Expected {EPISODE_LEN} steps, ran {step_count}"
    assert info["step"] == EPISODE_LEN, f"Info step {info['step']} != {EPISODE_LEN}"

    print(f"  - Attack type simulated : {info['attack_type']}")
    print(f"  - Final true frequency   : {info['true_freq']:.4f} Hz")
    print(f"  - Final measured freq    : {info['measured_freq']:.4f} Hz")
    print(f"  - Out of band at finish  : {info['out_of_band']}")
    print(f"  - Total steps completed  : {step_count}/{EPISODE_LEN}")
    print("PASSED: Check (c) verified successfully.\n")


def test_reward_behavior():
    """Check (d): Reward is negative when out_of_band is True and zero-ish when in band with action 0."""
    print("Check (d): Verifying Phase 2 reward formulation...")
    env = GridDefenseEnv(estimator=KalmanEstimator())
    obs, info = env.reset(seed=77)

    # Step 1: Initial state is nominal (50.0 Hz), clean operation, action=0 (trust)
    obs, reward_clean_trust, term, trunc, info_clean_trust = env.step(0)
    assert not info_clean_trust["out_of_band"], "Expected in-band frequency during clean step"
    assert not info_clean_trust["attack_active"], "Expected clean step before attack start"
    assert reward_clean_trust == 0.0, f"Expected reward 0.0 for in-band with action 0, got {reward_clean_trust}"
    assert info_clean_trust["r_band"] == 0.0
    assert info_clean_trust["r_distrust"] == 0.0

    # Step 2: Needless distrust during clean step, action=1
    obs, reward_clean_distrust, term, trunc, info_clean_distrust = env.step(1)
    expected_distrust_penalty = -W_DISTRUST
    assert not info_clean_distrust["out_of_band"]
    assert reward_clean_distrust == expected_distrust_penalty, (
        f"Expected {expected_distrust_penalty}, got {reward_clean_distrust}"
    )
    assert info_clean_distrust["r_distrust"] == expected_distrust_penalty
    assert info_clean_distrust["r_band"] == 0.0

    # Step 3: Out-of-band excursion
    # Force physical frequency outside safe band [49.80, 50.20]
    env.physics.frequency = 49.50  # Unsafe under-frequency
    obs, reward_oob, term, trunc, info_oob = env.step(0)
    assert info_oob["out_of_band"] is True, "Frequency should be flagged out_of_band"
    assert reward_oob < 0.0, f"Reward must be negative when out_of_band is True, got {reward_oob}"
    assert info_oob["r_band"] == -W_BAND, f"Expected r_band = {-W_BAND}, got {info_oob['r_band']}"

    print(f"  - In-band + Action 0 reward    : {reward_clean_trust:.4f} (expected: 0.0)")
    print(f"  - Needless distrust reward     : {reward_clean_distrust:.4f} (expected: {-W_DISTRUST})")
    print(f"  - Out-of-band penalty reward   : {reward_oob:.4f} (expected: {-W_BAND})")
    print("PASSED: Check (d) verified successfully.\n")


def test_kalman_reset_determinism():
    """Phase 3 Test (a): Same seed gives identical rollouts with KalmanEstimator and filter resets properly."""
    print("Phase 3 Test (a): Verifying KalmanEstimator determinism and filter state resets...")
    env = GridDefenseEnv(estimator=KalmanEstimator())

    # Episode 1 with seed=555
    obs1_list = []
    env.reset(seed=555)
    for _ in range(30):
        o, _, _, _, _ = env.step(0)
        obs1_list.append(o)

    # Episode 2 with different seed=777 (alters state)
    env.reset(seed=777)
    for _ in range(30):
        env.step(1)

    # Episode 3 with seed=555 again (must reproduce Episode 1 bitwise!)
    obs3_list = []
    env.reset(seed=555)
    for _ in range(30):
        o, _, _, _, _ = env.step(0)
        obs3_list.append(o)

    for i, (o1, o3) in enumerate(zip(obs1_list, obs3_list)):
        np.testing.assert_array_equal(
            o1, o3,
            err_msg=f"Kalman state did not reset cleanly! Step {i} diverged across identical seed resets."
        )

    print("  - Filter state correctly cleared on reset().")
    print("  - Re-seeded rollout matched initial rollout 100% bitwise.")
    print("PASSED: Phase 3 Test (a) verified successfully.\n")


def test_observation_no_ground_truth():
    """Phase 3 & 4 Test (b): Observation never contains true_freq (zero ground truth leakage)."""
    print("Phase 3 & 4 Test (b): Verifying zero ground-truth leakage across all observation features...")
    env = GridDefenseEnv(estimator=KalmanEstimator(), obs_extended=True)

    # Test KalmanEstimator directly with arbitrary/fake true_freq
    est = KalmanEstimator()
    est.reset()
    p1, r1 = est.get_estimate(measured_freq=48.50, true_freq=50.00, p_gen=100.0, dt=0.1)
    est.reset()
    p2, r2 = est.get_estimate(measured_freq=48.50, true_freq=999.99, p_gen=100.0, dt=0.1)
    assert p1 == p2, "KalmanEstimator prediction changed when true_freq changed! It must ignore true_freq."
    assert r1 == r2, "KalmanEstimator residual changed when true_freq changed! It must ignore true_freq."

    # Test during active spoof attack where true_freq (50.0) != measured_freq (48.0)
    obs, info = env.reset(seed=12, options={"attack_type": "spoof", "attack_start": 0.0, "intensity": 48.0})
    obs, _, _, _, info = env.step(0)

    # Normalized ground-truth deviation vs measured deviation
    true_norm = (info["true_freq"] - NOMINAL_FREQ) / OBS_FREQ_NORM_SCALE
    meas_norm = (info["measured_freq"] - NOMINAL_FREQ) / OBS_FREQ_NORM_SCALE

    assert abs(meas_norm - obs[0]) < 1e-4, f"obs[0] should be measured_freq_norm {meas_norm}, got {obs[0]}"
    assert abs(obs[0] - true_norm) > 0.5, "obs[0] leaked true_freq!"
    for i in range(len(obs)):
        assert abs(obs[i] - true_norm) > 1e-3 or abs(obs[i]) < 1e-5, f"Feature index {i} suspicious match with true_norm"

    print("  - Verified KalmanEstimator outputs are strictly invariant to true_freq.")
    print(f"  - Under attack: true_freq={info['true_freq']} Hz, measured_freq={info['measured_freq']} Hz.")
    print(f"  - obs[0] reflects measured ({obs[0]:.3f}) and NOT ground truth ({true_norm:.3f}).")
    print(f"  - All 8 observation features verified free from ground-truth leakage.")
    print("PASSED: Phase 3 & 4 Test (b) verified successfully.\n")


def test_stealthy_drift_residual():
    """Phase 3 Test (c): During stealthy_drift, residual is non-trivially above clean-episode level."""
    print("Phase 3 Test (c): Verifying stealthy slow-drift attack residual signature...")
    env_clean = GridDefenseEnv(estimator=KalmanEstimator())
    env_drift = GridDefenseEnv(estimator=KalmanEstimator())

    # 1. Clean episode
    env_clean.reset(seed=42, options={"attack_type": "none"})
    clean_residuals = []
    for _ in range(350):
        _, _, _, _, info = env_clean.step(0)
        clean_residuals.append(abs(info["residual"]))

    # 2. Stealthy drift episode: start at t=10.0s (step 100), drift_rate=0.05 Hz/s
    env_drift.reset(seed=42, options={
        "attack_type": "stealthy_drift",
        "attack_start": 10.0,
        "attack_duration": 20.0,
        "intensity": 0.05,
    })
    drift_residuals = []
    attack_window_residuals = []
    for step_i in range(350):
        _, _, _, _, info = env_drift.step(0)
        res_mag = abs(info["residual"])
        drift_residuals.append(res_mag)
        if info["attack_active"]:
            attack_window_residuals.append(res_mag)

    max_clean_res = float(np.max(clean_residuals))
    max_drift_res = float(np.max(attack_window_residuals))

    print(f"  - Maximum residual in clean episode         : {max_clean_res:.4f} Hz")
    print(f"  - Maximum residual during stealthy drift    : {max_drift_res:.4f} Hz")
    print(f"  - Residual amplification factor             : {max_drift_res / max(1e-6, max_clean_res):.1f}x")

    assert max_drift_res > 2.0 * max_clean_res, (
        f"Expected stealthy drift residual ({max_drift_res:.4f}) to clearly exceed clean ({max_clean_res:.4f})"
    )
    assert max_drift_res > 0.08, f"Expected drift residual > 0.08 Hz, got {max_drift_res:.4f}"
    print("PASSED: Phase 3 Test (c) verified successfully.\n")


if __name__ == "__main__":
    test_gymnasium_check_env()
    test_shapes_and_dtypes()
    test_seed_reproducibility()
    test_random_action_full_rollout()
    test_reward_behavior()
    test_kalman_reset_determinism()
    test_observation_no_ground_truth()
    test_stealthy_drift_residual()
    print("======================================================================")
    print("ALL ENVIRONMENT TESTS (PHASES 1, 2 & 3) PASSED CLEANLY!")
    print("======================================================================")
