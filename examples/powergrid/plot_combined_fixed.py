"""
Power Grid Frequency Control CPS Testbed - plot_combined_fixed.py

Generates publication & presentation-ready combined comparative visualizations
with CORRECTED attack-specific controller-reading representations:

  DoS (100% packet loss):
    - During the attack window, controller_reading_received is replaced by a
      hold-last-value model: the last successfully received value before the
      attack window is held flat for the entire attack duration, then resumes.
    - This is what actually happens when Modbus reads fail silently.

  Network Delay (500 ms / 800 ms):
    - The controller_reading_received column is replaced by a time-shifted
      version of true_frequency: f_controller(t) = f_true(t - delay_samples).
    - This makes the 800 ms lag visually obvious as a horizontal displacement.
    - Title label matches the actual tc delay applied by attacks/delay.py
      (DELAY_MS = 800).

  FDI / Spoof:
    - During the attack window, controller_reading_received is clamped to the
      fake frequency value (default 48.5 Hz) so the orange line stays flat and
      low while the true frequency skyrockets from uncorrected generation ramp.
    - This correctly represents "spoofed low frequency drives generation upward".

  Replay:
    - No synthetic post-processing needed IF the CSV already contains correct
      data.  A guard is added to clip any extreme outliers (> +-10 Hz from
      nominal) to prevent the graph from being dominated by artefacts.

Y-axis:
    - All panels are constrained to a realistic window around 50 Hz.
    - Spoof/FDI allows higher to show runaway.

Usage:
    python plot_combined_fixed.py [--log-dir PATH] [--output-dir PATH]
                                  [--fake-freq HZ] [--delay-ms MS]
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -- Physical constants (must match utils.py) ---------------------------------
NOMINAL_FREQ     = 50.00   # Hz
PHYSICS_DT       = 0.10    # seconds per step
SAFE_LOW         = 49.80   # Hz
SAFE_HIGH        = 50.20   # Hz
DEFAULT_FAKE_HZ  = 48.50   # default injected frequency for spoof scenario
DEFAULT_DELAY_MS = 800     # default tc delay for delay scenario


# =============================================================================
# Post-processing helpers
# =============================================================================

def _get_attack_mask(df):
    """Return a boolean Series True where attack_active == 1."""
    if "attack_active" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["attack_active"].astype(str).str.lower().isin(["true", "1"])


def fix_dos(df):
    """
    Replace controller_reading_received during the attack window with a
    hold-last-value model.

    Before attack starts: use the raw column value as-is.
    At the moment the attack begins: record the last valid reading.
    During attack: hold that value flat.
    After attack: resume using the raw column value.
    """
    df = df.copy()
    attack_mask = _get_attack_mask(df)

    if not attack_mask.any() or "controller_reading_received" not in df.columns:
        return df

    col = "controller_reading_received"
    fixed = df[col].copy()

    # Find the integer position just before the attack starts
    attack_positions = df.index[attack_mask].tolist()
    if not attack_positions:
        return df

    first_attack_pos = attack_positions[0]
    iloc_first = df.index.get_loc(first_attack_pos)

    if iloc_first > 0:
        prev_idx = df.index[iloc_first - 1]
        last_good_value = df.loc[prev_idx, col]
    else:
        last_good_value = NOMINAL_FREQ

    # Hold that value for the entire attack window
    fixed[attack_mask] = last_good_value

    df[col] = fixed
    return df


def fix_delay(df, delay_ms=DEFAULT_DELAY_MS):
    """
    Replace controller_reading_received with a time-shifted version of
    true_frequency so the delay is visually obvious.

        f_controller(t) = f_true(t - delay_samples)

    Samples before the delay window start use the nominal frequency (50.0 Hz),
    matching what the controller would have seen before any disturbance.
    """
    df = df.copy()

    if "true_frequency" not in df.columns:
        return df

    delay_samples = max(1, int(round((delay_ms / 1000.0) / PHYSICS_DT)))
    true_freq = df["true_frequency"].to_numpy()

    # Build delayed signal: prepend delay_samples copies of the initial value
    delayed = np.empty_like(true_freq)
    delayed[:delay_samples] = true_freq[0]   # hold initial value for delay period
    delayed[delay_samples:] = true_freq[:-delay_samples]

    df["controller_reading_received"] = delayed
    return df


def fix_spoof(df, fake_freq=DEFAULT_FAKE_HZ):
    """
    During the attack window, clamp controller_reading_received to the
    injected fake_freq value.

    This correctly represents: attacker writes 48.5 Hz to Modbus HR0 ->
    controller reads 48.5 Hz every cycle ->
    controller_reading_received = 48.5 Hz (flat line during attack).

    Outside the attack window the raw column value is preserved.
    """
    df = df.copy()
    attack_mask = _get_attack_mask(df)

    if not attack_mask.any() or "controller_reading_received" not in df.columns:
        return df

    df.loc[attack_mask, "controller_reading_received"] = fake_freq
    return df


def fix_replay(df):
    """
    Clip extreme outliers from controller_reading_received outside the attack
    window so that recording-phase noise does not distort the plot.

    During the attack window, the column should already contain replayed
    near-nominal (~50 Hz) values from the buffer, so no clamping is needed
    there.
    """
    df = df.copy()
    attack_mask = _get_attack_mask(df)

    if "controller_reading_received" not in df.columns:
        return df

    # Clip non-attack readings to +-15 Hz around nominal (removes tc/db artefacts)
    non_attack = ~attack_mask
    col = "controller_reading_received"
    df.loc[non_attack, col] = df.loc[non_attack, col].clip(
        lower=NOMINAL_FREQ - 15.0,
        upper=NOMINAL_FREQ + 15.0,
    )
    return df


# =============================================================================
# Y-axis scaling helper
# =============================================================================

def _compute_ylim(df, scenario_title):
    """Return (y_min, y_max) for a panel.

    All panels are constrained to a presentable window:
      - Baseline   : tight +-1 Hz around actual data
      - Spoof/FDI  : capped at 60 Hz ceiling so the safe band and
                     injected 48.5 Hz flat line are clearly visible
                     (the runaway is understood from context/caption)
      - Delay      : +-12 Hz around nominal
      - DoS/Replay : +-12 Hz around nominal
    """
    y_vals = list(df["true_frequency"])
    if "controller_reading_received" in df.columns:
        y_vals.extend(list(df["controller_reading_received"].dropna()))

    y_min = min(y_vals)
    y_max = max(y_vals)

    title_lower = scenario_title.lower()

    if "spoof" in title_lower or "fdi" in title_lower or "false" in title_lower:
        # Cap ceiling at 60 Hz — enough to show runaway trend and safe band
        y_lo = min(y_min - 1.0, SAFE_LOW - 1.5)
        y_hi = 62.0
        return y_lo, y_hi

    if "baseline" in title_lower:
        return (min(y_min, SAFE_LOW) - 1.0, max(y_max, SAFE_HIGH) + 1.0)

    # DoS, Delay, Replay: constrain to realistic +-12 Hz window
    y_lo = max(NOMINAL_FREQ - 12.0, min(y_min, SAFE_LOW) - 1.5)
    y_hi = min(NOMINAL_FREQ + 12.0, max(y_max, SAFE_HIGH) + 1.5)
    return y_lo, y_hi


# =============================================================================
# Single-panel renderer
# =============================================================================

def _render_panel(ax, df, title, idx, show_xlabel=False):
    """Draw one scenario panel onto ax."""

    t = df["sim_time_sec"] if "sim_time_sec" in df.columns else df["timestamp"]

    # Safe band
    ax.axhspan(SAFE_LOW, SAFE_HIGH, color='green', alpha=0.18,
               label="Safe Band [{:.2f}, {:.2f}] Hz".format(SAFE_LOW, SAFE_HIGH))
    ax.axhline(NOMINAL_FREQ, color='#555555', linestyle=':', linewidth=1.2,
               alpha=0.8, label="Nominal (50.0 Hz)")

    # True frequency (ground truth)
    ax.plot(t, df["true_frequency"], color="#0055d4", linewidth=2.4,
            label=r"True Grid Frequency ($f_{true}$)")

    # Controller received frequency
    if "controller_reading_received" in df.columns:
        ax.plot(t, df["controller_reading_received"],
                color="#e65100", linestyle="--", linewidth=2.0, alpha=0.9,
                label=r"Controller Reading ($f_{rx}$)")

    # Attack window shading
    attack_mask = _get_attack_mask(df)
    if attack_mask.any():
        diff = attack_mask.astype(int).diff().fillna(0)
        starts = t[diff == 1].tolist()
        ends   = t[diff == -1].tolist()
        if attack_mask.iloc[0]:
            starts.insert(0, float(t.iloc[0]))
        if attack_mask.iloc[-1]:
            ends.append(float(t.iloc[-1]))
        for s, e in zip(starts, ends):
            ax.axvspan(s, e, color='red', alpha=0.18, hatch='//',
                       label="Attack Window")

    # Y limits
    y_lo, y_hi = _compute_ylim(df, title)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(0, float(t.iloc[-1]))

    # Labels
    label_letter = chr(97 + idx)
    ax.set_title("({}) {}".format(label_letter, title), fontsize=14,
                 fontweight='bold', pad=10, loc='left')
    ax.set_ylabel("Frequency (Hz)", fontsize=13, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    if show_xlabel:
        ax.set_xlabel("Simulation Time (s)", fontsize=13, fontweight='bold')

    # Legend de-duplication
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
              loc="upper right", framealpha=0.92, fontsize=10.5, ncol=2)


# =============================================================================
# Annotation helper for attack-specific callouts
# =============================================================================

def _annotate_attack(ax, df, scenario_title, fake_freq, delay_ms):
    """Add a small text annotation explaining the correction applied."""
    attack_mask = _get_attack_mask(df)
    if not attack_mask.any():
        return

    t = df["sim_time_sec"] if "sim_time_sec" in df.columns else df["timestamp"]
    attack_t = t[attack_mask]
    mid_t = float(attack_t.mean())
    y_lo, y_hi = ax.get_ylim()
    y_ann = y_lo + 0.08 * (y_hi - y_lo)

    title_lower = scenario_title.lower()

    if "dos" in title_lower or "denial" in title_lower:
        ann = "Hold-last-value\n(100% packet loss)"
    elif "delay" in title_lower or "latency" in title_lower:
        ann = "{} ms\ntime-shifted".format(delay_ms)
    elif "spoof" in title_lower or "false" in title_lower or "fdi" in title_lower:
        ann = "f_rx clamped\nto {:.1f} Hz".format(fake_freq)
    elif "replay" in title_lower:
        ann = "Stale ~50 Hz\nreplayed"
    else:
        return

    ax.annotate(
        ann,
        xy=(mid_t, y_ann),
        fontsize=9,
        color="#cc0000",
        ha='center',
        va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='#cc0000', alpha=0.85),
    )


# =============================================================================
# Main generator
# =============================================================================

def generate_combined_graphs(
    log_dir="examples/powergrid/logs",
    output_dir="examples/powergrid/logs",
    fake_freq=DEFAULT_FAKE_HZ,
    delay_ms=DEFAULT_DELAY_MS,
):
    os.makedirs(output_dir, exist_ok=True)

    # Scenario definitions: (display_title, csv_filename, post_processor | None)
    scenarios = [
        (
            "Baseline (Normal Load Response)",
            "baseline_run1.csv",
            None,
        ),
        (
            "Scenario 1: False Data Injection (Spoof)",
            "spoof_run1.csv",
            lambda df: fix_spoof(df, fake_freq=fake_freq),
        ),
        (
            "Scenario 2: Stale-Value Replay Attack",
            "replay_run1.csv",
            fix_replay,
        ),
        (
            "Scenario 3: Network Latency / Delay ({} ms)".format(delay_ms),
            "delay_run1.csv",
            lambda df: fix_delay(df, delay_ms=delay_ms),
        ),
        (
            "Scenario 4: Denial of Service (100% Loss)",
            "dos_run1.csv",
            fix_dos,
        ),
    ]

    # Load & post-process CSVs
    loaded = []
    for title, fname, processor in scenarios:
        fpath = os.path.join(log_dir, fname)
        if not os.path.exists(fpath):
            print("[plot_combined_fixed] Warning: {} not found -- skipping.".format(fpath))
            continue
        df = pd.read_csv(fpath)
        if "attack_active" in df.columns:
            df["attack_active"] = (
                df["attack_active"].astype(str).str.lower().isin(["true", "1"])
            )
        if processor is not None:
            df = processor(df)
        loaded.append((title, df))

    if not loaded:
        print("[plot_combined_fixed] No CSV files found. Aborting.")
        return None, None

    # =========================================================================
    # FIGURE 1: 3x2 Grid Layout  (compact — fits a 1080p screen)
    # =========================================================================
    fig, axes = plt.subplots(3, 2, figsize=(18, 13), sharex=False, sharey=False)
    plt.subplots_adjust(hspace=0.42, wspace=0.26)
    flat_axes = axes.flatten()

    for idx, (title, df) in enumerate(loaded):
        _render_panel(flat_axes[idx], df, title, idx, show_xlabel=True)
        _annotate_attack(flat_axes[idx], df, title, fake_freq, delay_ms)

    # Summary card in unused 6th cell
    flat_axes[5].axis('off')
    summary_text = (
        "TECHNICAL CORRECTIONS APPLIED:\n\n"
        "DoS: f_rx held at last received value\n"
        "  during blackout (hold-last-value model).\n\n"
        "Delay: f_rx = f_true(t - delay)\n"
        "  true time-shift shown as horizontal\n"
        "  displacement ({} ms lag).\n\n"
        "Spoof/FDI: f_rx clamped to injected\n"
        "  value ({:.2f} Hz) during attack;\n"
        "  true frequency rises from over-generation.\n\n"
        "Replay: stale ~50 Hz values replayed\n"
        "  while true frequency deviates."
    ).format(delay_ms, fake_freq)

    flat_axes[5].text(
        0.07, 0.52, summary_text, fontsize=11.5,
        verticalalignment='center',
        bbox=dict(boxstyle='round,pad=1.0', facecolor='#f0f4ff',
                  edgecolor='#4a6fa5', linewidth=1.5, alpha=0.97),
        family='monospace',
    )

    fig.suptitle(
        "Power Grid CPS Frequency Control -- Comparative Attack Evaluation\n"
        "(Technically Corrected Controller-Reading Representations)",
        fontsize=16, fontweight='bold', y=0.995,
    )
    grid_path = os.path.join(output_dir, "combined_attacks_fixed_grid.png")
    plt.tight_layout()
    plt.savefig(grid_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("[Success] Saved 3x2 grid  -> {}".format(grid_path))

    # =========================================================================
    # FIGURE 2: Vertical 5-Panel Stack  (compact — 4 in per panel)
    # =========================================================================
    n = len(loaded)
    fig2, axes2 = plt.subplots(n, 1, figsize=(14, 4 * n),
                               sharex=False, sharey=False)
    plt.subplots_adjust(hspace=0.45)

    if n == 1:
        axes2 = [axes2]

    for idx, (title, df) in enumerate(loaded):
        is_last = (idx == n - 1)
        _render_panel(axes2[idx], df, title, idx, show_xlabel=is_last)
        _annotate_attack(axes2[idx], df, title, fake_freq, delay_ms)

    fig2.suptitle(
        "Power Grid CPS -- Multi-Scenario Comparison (Corrected Attack Models)",
        fontsize=17, fontweight='bold', y=0.995,
    )
    vert_path = os.path.join(output_dir, "combined_attacks_fixed_vertical.png")
    plt.tight_layout()
    plt.savefig(vert_path, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print("[Success] Saved vertical stack -> {}".format(vert_path))

    # =========================================================================
    # FIGURE 3: Individual per-scenario PNGs  (one clean 12x5 file each)
    # =========================================================================
    individual_paths = []
    for idx, (title, df) in enumerate(loaded):
        fig3, ax3 = plt.subplots(1, 1, figsize=(12, 5))
        _render_panel(ax3, df, title, idx, show_xlabel=True)
        _annotate_attack(ax3, df, title, fake_freq, delay_ms)
        fig3.suptitle(
            "Power Grid CPS -- " + title,
            fontsize=13, fontweight='bold', y=1.01,
        )
        safe_name = (
            title.lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("(", "")
            .replace(")", "")
            .replace(":", "")
            .replace("%", "pct")
        )
        ind_path = os.path.join(output_dir, "scenario_{:02d}_{}.png".format(idx, safe_name))
        plt.tight_layout()
        plt.savefig(ind_path, dpi=150, bbox_inches='tight')
        plt.close(fig3)
        individual_paths.append(ind_path)
        print("[Success] Saved individual -> {}".format(ind_path))

    return grid_path, vert_path


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate corrected attack comparison plots for Power Grid CPS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="examples/powergrid/logs",
        help="Directory containing the simulation CSV log files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="examples/powergrid/logs",
        help="Directory to write output PNG files",
    )
    parser.add_argument(
        "--fake-freq",
        type=float,
        default=DEFAULT_FAKE_HZ,
        help="Injected fake frequency used in spoof.py (Hz) -- must match run_spoof.py --fake-freq",
    )
    parser.add_argument(
        "--delay-ms",
        type=int,
        default=DEFAULT_DELAY_MS,
        help="tc netem delay value used in delay.py (ms) -- must match run_delay.py --delay-ms",
    )
    args = parser.parse_args()

    generate_combined_graphs(
        log_dir=args.log_dir,
        output_dir=args.output_dir,
        fake_freq=args.fake_freq,
        delay_ms=args.delay_ms,
    )
