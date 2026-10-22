"""
Grid-Defense Demo – Streamlit App  (demo/app.py)

Three-tab interactive dashboard for demonstrating RL-based cyber-physical
defense on a simulated power grid.  Uses the **real** GridDefenseEnv,
Kalman filter, Mythri's baselines, Detector-Timing heuristic, and the
saved DQN models – no copied physics or fabricated numbers.

Run:  streamlit run demo/app.py
"""

import os, sys, time, pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
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
    run_episode, DEFENSE_METHODS, ATTACK_UI_TO_ENV,
    INTENSITY_RANGES, NOMINAL_FREQ, SAFE_BAND_LOW, SAFE_BAND_HIGH, DT,
)

# ---------------------------------------------------------------------------
# Page config & custom CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Grid Defense – RL CPS Demo",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ---------- Dark-gradient page background ---------- */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #0f0c29 100%);
}

/* ---------- Metric cards ---------- */
.metric-card {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px;
    padding: 16px 18px;
    text-align: center;
}
.metric-card .label { font-size: 0.78rem; color: #94a3b8; margin-bottom: 4px; }
.metric-card .value { font-size: 1.45rem; font-weight: 700; }

/* ---------- Status badges ---------- */
.badge {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.82rem;
    margin: 0 5px;
}
.badge-normal   { background: #06d6a0; color: #0f0c29; }
.badge-attack   { background: #ff006e; color: #fff; animation: pulse 1s infinite; }
.badge-distrust { background: #ffd166; color: #0f0c29; }
.badge-oob      { background: #ef476f; color: #fff; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.65} }

/* ---------- Limitations box ---------- */
.limits-box {
    background: rgba(239,71,111,0.08);
    border-left: 4px solid #ef476f;
    border-radius: 8px;
    padding: 14px 18px;
    margin-top: 12px;
    font-size: 0.88rem;
}

/* ---------- Table highlight ---------- */
.best-cell { background: rgba(6,214,160,0.22) !important; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Plotly colour palette
# ---------------------------------------------------------------------------
C_TRUE  = "#00B4D8"
C_MEAS  = "#FF6B6B"
C_PRED  = "#06D6A0"
C_BAND  = "#FFC107"
C_ATK   = "rgba(255,0,110,0.12)"
C_TRUST = "#06d6a0"
C_DISTR = "#ff006e"

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif"),
    margin=dict(l=50, r=20, t=40, b=30),
)

# ---------------------------------------------------------------------------
# Helper: build the main frequency chart up to step N
# ---------------------------------------------------------------------------
def build_freq_chart(traj: dict, up_to: int | None = None, height: int = 420) -> go.Figure:
    """Return a Plotly figure of the trajectory up to step `up_to`."""
    n = len(traj["t"]) if up_to is None else min(up_to + 1, len(traj["t"]))
    t = traj["t"][:n]
    # Attack window shading
    att = traj["attack_active"][:n].astype(bool)
    att_start_t = t[att][0] if att.any() else None
    att_end_t   = t[att][-1] if att.any() else None

    fig = go.Figure()
    # Safe-band shading
    fig.add_hrect(y0=SAFE_BAND_LOW, y1=SAFE_BAND_HIGH,
                  fillcolor="rgba(255,193,7,0.08)", line_width=0)
    fig.add_hline(y=SAFE_BAND_LOW,  line=dict(color=C_BAND, width=1, dash="dot"))
    fig.add_hline(y=SAFE_BAND_HIGH, line=dict(color=C_BAND, width=1, dash="dot"))
    # Attack window
    if att_start_t is not None:
        fig.add_vrect(x0=att_start_t, x1=att_end_t,
                      fillcolor=C_ATK, line_width=0,
                      annotation_text="Attack", annotation_position="top left",
                      annotation_font_color="#ff006e")
    # Traces
    fig.add_trace(go.Scatter(x=t, y=traj["f_true"][:n],  mode="lines",
                             name="True Freq",     line=dict(color=C_TRUE, width=2)))
    fig.add_trace(go.Scatter(x=t, y=traj["f_meas"][:n],  mode="lines",
                             name="Sensor (meas)", line=dict(color=C_MEAS, width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=t, y=traj["f_pred"][:n],  mode="lines",
                             name="Kalman Pred",   line=dict(color=C_PRED, width=1.5, dash="dash")))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=height,
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig


def build_action_strip(traj: dict, up_to: int | None = None, height: int = 90) -> go.Figure:
    """Colour strip showing defender actions over time."""
    n = len(traj["t"]) if up_to is None else min(up_to + 1, len(traj["t"]))
    t = traj["t"][:n]
    act = traj["action"][:n]
    colours = [C_TRUST if a == 0 else C_DISTR for a in act]
    fig = go.Figure(go.Bar(x=t, y=[1]*len(t), marker_color=colours,
                           hovertext=["Trust" if a == 0 else "Distrust" for a in act]))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=height,
        xaxis_title="Time (s)",
        yaxis=dict(visible=False),
        showlegend=False,
        bargap=0,
    )
    return fig


# ---------------------------------------------------------------------------
# Helper: metric card HTML
# ---------------------------------------------------------------------------
def metric_html(label: str, value: str, colour: str = "#e0e0e0") -> str:
    return (f'<div class="metric-card">'
            f'<div class="label">{label}</div>'
            f'<div class="value" style="color:{colour}">{value}</div></div>')


# ---------------------------------------------------------------------------
# Helper: status badges
# ---------------------------------------------------------------------------
def status_badges(traj: dict, step_idx: int) -> str:
    badges = []
    if step_idx < len(traj["attack_active"]):
        if traj["attack_active"][step_idx]:
            badges.append('<span class="badge badge-attack">⚠ ATTACK ACTIVE</span>')
        else:
            badges.append('<span class="badge badge-normal">✔ NORMAL</span>')
        if traj["action"][step_idx] == 1:
            badges.append('<span class="badge badge-distrust">🛡 DEFENDER DISTRUSTING</span>')
        if step_idx < len(traj["oob"]) and traj["oob"][step_idx]:
            badges.append('<span class="badge badge-oob">⚡ FREQ OUT OF BAND</span>')
    return " ".join(badges)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
CACHE_DIR = os.path.join(_DEMO_DIR, "cache")

def _cache_key(attack_type, defense_method, seed, intensity, attack_start):
    int_str = f"{intensity:.4f}" if intensity is not None else "default"
    return f"{attack_type}__{defense_method}__{seed}__{int_str}__{attack_start:.1f}.pkl"

def load_cached(attack_type, defense_method, seed, intensity, attack_start):
    fp = os.path.join(CACHE_DIR, _cache_key(attack_type, defense_method, seed, intensity, attack_start))
    if os.path.exists(fp):
        with open(fp, "rb") as f:
            return pickle.load(f)
    return None


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR  (shared parameters)
# ═══════════════════════════════════════════════════════════════════════════
st.sidebar.title("⚡ Grid Defense Demo")
st.sidebar.caption("RL-based Cyber-Physical Security")

use_cache = st.sidebar.toggle("Use cached runs", value=False,
                               help="Load pre-computed trajectories for instant display")

attack_options = list(ATTACK_UI_TO_ENV.keys())
attack_type = st.sidebar.selectbox("Attack type", attack_options, index=1)

# Intensity slider (hidden for none / telemetry_freeze)
intensity_val = None
if attack_type not in ("none", "telemetry_freeze"):
    lo, hi = INTENSITY_RANGES[attack_type]
    unit = {"spoof": "Hz", "stealthy_drift": "Hz/s", "replay": "s"}.get(attack_type, "")
    intensity_val = st.sidebar.slider(
        f"Intensity ({unit})", min_value=float(lo), max_value=float(hi),
        value=float((lo + hi) / 2), step=float((hi - lo) / 40),
    )
else:
    if attack_type == "telemetry_freeze":
        st.sidebar.info("Telemetry freeze holds the last valid reading; intensity is not used.")

attack_start = st.sidebar.slider("Attack start (s)", 10.0, 30.0, 20.0, 1.0)
env_seed = st.sidebar.number_input("Env seed", 0, 100000, 42)
dqn_seed = st.sidebar.selectbox("DQN model seed", [0, 1, 2, 3, 4], index=0)

speed_map = {"Slow": (2, 0.15), "Normal": (5, 0.08), "Fast": (10, 0.04), "Very Fast": (25, 0.015)}
play_speed = st.sidebar.select_slider("Play speed", list(speed_map.keys()), value="Normal")
step_skip, frame_delay = speed_map[play_speed]


# ═══════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🎬 Live Attack Simulation", "📊 Compare Defenses", "📈 Results"])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 – Live Attack Simulation
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    defense_method = st.selectbox("Defense method", DEFENSE_METHODS, index=4,
                                  key="t1_defense")
    # ---------- Init session state ----------
    for k, v in [("traj", None), ("playing", False), ("anim_step", 0)]:
        if k not in st.session_state:
            st.session_state[k] = v

    # ---------- Run / control buttons ----------
    bcol1, bcol2, bcol3, bcol4 = st.columns(4)
    run_clicked = bcol1.button("▶ Run Simulation", use_container_width=True, type="primary")
    play_clicked = bcol2.button("⏯ Play / Pause", use_container_width=True)
    restart_clicked = bcol3.button("🔄 Restart", use_container_width=True)
    show_all_clicked = bcol4.button("⏩ Show All", use_container_width=True)

    if run_clicked:
        with st.spinner("Running episode…"):
            cached = None
            if use_cache:
                cached = load_cached(attack_type, defense_method, env_seed,
                                     intensity_val, attack_start)
            if cached is not None:
                st.session_state.traj = cached
            else:
                st.session_state.traj = run_episode(
                    attack_type, defense_method, env_seed,
                    intensity_val, attack_start, dqn_seed,
                )
        st.session_state.anim_step = 0
        st.session_state.playing = False

    if play_clicked:
        st.session_state.playing = not st.session_state.playing
    if restart_clicked:
        st.session_state.anim_step = 0
        st.session_state.playing = False
    if show_all_clicked and st.session_state.traj is not None:
        st.session_state.anim_step = len(st.session_state.traj["t"]) - 1
        st.session_state.playing = False

    traj = st.session_state.traj
    if traj is not None:
        total_frames = len(traj["t"])
        cur = min(st.session_state.anim_step, total_frames - 1)

        # Status badges
        st.markdown(status_badges(traj, cur), unsafe_allow_html=True)

        # Frequency chart
        st.plotly_chart(build_freq_chart(traj, up_to=cur), use_container_width=True, key="freq_chart")

        # Action strip
        st.caption("Defender action  (🟢 Trust  /  🔴 Distrust)")
        st.plotly_chart(build_action_strip(traj, up_to=cur), use_container_width=True, key="act_strip")

        # Progress slider (manual scrub)
        new_step = st.slider("Timestep", 0, total_frames - 1, cur, key="step_slider")
        if new_step != cur:
            st.session_state.anim_step = new_step

        # Metric cards
        oob_so_far = int(traj["oob"][:cur+1].sum())
        max_dev_so_far = float(np.max(np.abs(traj["f_true"][:cur+1] - NOMINAL_FREQ)))
        act_arr = traj["action"][:cur+1]
        att_arr = traj["attack_active"][:cur+1]
        clean_mask = ~att_arr.astype(bool)
        cd_rate = (act_arr[clean_mask].sum() / max(1, clean_mask.sum()) * 100) if clean_mask.any() else 0.0

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.markdown(metric_html("Time OOB", f"{oob_so_far*DT:.1f} s",
                                "#ef476f" if oob_so_far > 0 else C_TRUST), unsafe_allow_html=True)
        m2.markdown(metric_html("Max Deviation", f"{max_dev_so_far:.3f} Hz",
                                "#ef476f" if max_dev_so_far > 0.2 else C_TRUST), unsafe_allow_html=True)
        det_str = f"{traj['detection_delay']} steps" if traj["detection_delay"] is not None else "—"
        m3.markdown(metric_html("Detection Delay", det_str, "#ffd166"), unsafe_allow_html=True)
        m4.markdown(metric_html("Clean Distrust", f"{cd_rate:.1f} %", "#00b4d8"), unsafe_allow_html=True)
        m5.markdown(metric_html("Episode Return", f"{traj['episode_return']:.1f}", "#e0e0e0"),
                    unsafe_allow_html=True)

        # Auto-advance animation
        if st.session_state.playing and cur < total_frames - 1:
            time.sleep(frame_delay)
            st.session_state.anim_step = min(cur + step_skip, total_frames - 1)
            st.rerun()
        elif st.session_state.playing and cur >= total_frames - 1:
            st.session_state.playing = False

    else:
        st.info("👆 Click **Run Simulation** to generate a trajectory.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 – Compare Defenses
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Side-by-Side Defense Comparison")
    st.caption("Same attack scenario – all defenses evaluated on the same seed.")
    if attack_type == "telemetry_freeze":
        st.markdown(
            "ℹ️ **telemetry_freeze**: a full stale-hold of the last valid measurement. "
            "Both `delay` and `dos` in the env share this identical code path; the "
            "configured latency / drop-rate parameters are not used.",
            help="Telemetry freeze holds the last valid reading indefinitely, "
                 "incrementing packet_age each step.",
        )

    if st.button("🚀 Run All Defenses", key="run_compare", type="primary"):
        results = {}
        prog = st.progress(0, text="Running defenses…")
        for i, dm in enumerate(DEFENSE_METHODS):
            try:
                cached = None
                if use_cache:
                    cached = load_cached(attack_type, dm, env_seed,
                                         intensity_val, attack_start)
                if cached is not None:
                    results[dm] = cached
                else:
                    results[dm] = run_episode(
                        attack_type, dm, env_seed,
                        intensity_val, attack_start, dqn_seed,
                    )
            except Exception as e:
                st.warning(f"⚠ {dm}: {e}")
            prog.progress((i + 1) / len(DEFENSE_METHODS), text=f"{dm} done")
        prog.empty()
        st.session_state.compare_results = results

    if "compare_results" in st.session_state and st.session_state.compare_results:
        results = st.session_state.compare_results

        # Small-multiples grid  (2 rows × 3 cols)
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=list(results.keys()),
            vertical_spacing=0.12, horizontal_spacing=0.06,
        )
        for idx, (dm, tr) in enumerate(results.items()):
            r, c = divmod(idx, 3)
            r += 1; c += 1
            # Safe band
            fig.add_hline(y=SAFE_BAND_LOW,  line=dict(color=C_BAND, width=0.7, dash="dot"), row=r, col=c)
            fig.add_hline(y=SAFE_BAND_HIGH, line=dict(color=C_BAND, width=0.7, dash="dot"), row=r, col=c)
            # True freq
            fig.add_trace(go.Scatter(x=tr["t"], y=tr["f_true"], mode="lines",
                                     line=dict(color=C_TRUE, width=1.5),
                                     name="True Freq", showlegend=(idx == 0)),
                          row=r, col=c)
            # Kalman pred
            fig.add_trace(go.Scatter(x=tr["t"], y=tr["f_pred"], mode="lines",
                                     line=dict(color=C_PRED, width=1, dash="dash"),
                                     name="Kalman Pred", showlegend=(idx == 0)),
                          row=r, col=c)
        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=520,
            legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Metrics table
        rows = []
        for dm, tr in results.items():
            rows.append({
                "Defense": dm,
                "Time OOB (s)": tr["time_oob"],
                "Max Dev (Hz)": round(tr["max_deviation"], 4),
                "Det. Delay (steps)": tr["detection_delay"] if tr["detection_delay"] is not None else "—",
                "Clean Distrust (%)": round(tr["clean_distrust_rate"], 2),
                "Return (R2)": round(tr["episode_return"], 1),
            })
        df = pd.DataFrame(rows)

        # Highlight best per numeric column
        def highlight_best(s):
            if s.dtype == object:
                return [""] * len(s)
            if s.name == "Return (R2)":
                best = s.max()
            else:
                best = s.min()
            return ["background: rgba(6,214,160,0.22); font-weight:700" if v == best else "" for v in s]

        styled = df.style.apply(highlight_best, axis=0).format(precision=2)
        st.dataframe(styled, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 – Results (from saved files)
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📈 Evaluation Results (from final_eval.py)")

    results_dir = os.path.join(_ROOT_DIR, "results")
    csv_path = os.path.join(results_dir, "final_results.csv")
    ws_csv   = os.path.join(results_dir, "widespoof_fair.csv")
    summary_path = os.path.join(results_dir, "final_summary.md")
    ws_md    = os.path.join(results_dir, "widespoof_fair.md")

    # ---------- Load final_results.csv ----------
    if os.path.exists(csv_path):
        df_all = pd.read_csv(csv_path)

        # Rename delay/dos → telemetry_freeze for display
        if "attack_type" in df_all.columns:
            df_all["attack_label"] = df_all["attack_type"].replace({
                "delay": "telemetry_freeze (delay)",
                "dos":   "telemetry_freeze (dos)",
            })

        # --- Bar chart: OOB per attack per method ---
        st.markdown("### Time Outside Safe Band by Attack & Defense")

        # Aggregate
        if "method" in df_all.columns and "time_outside_safe_band" in df_all.columns:
            agg = (df_all.groupby(["method", "attack_label"])["time_outside_safe_band"]
                   .agg(["mean", "std"]).reset_index())
            agg.columns = ["method", "attack", "mean_oob", "std_oob"]
            agg["std_oob"] = agg["std_oob"].fillna(0)

            attacks = agg["attack"].unique()
            methods = agg["method"].unique()

            palette = [
                "#ef476f", "#ffd166", "#06d6a0", "#118ab2",
                "#073b4c", "#8338ec", "#ff006e", "#3a86a7",
                "#fb5607", "#ffbe0b",
            ]

            fig_bar = go.Figure()
            for mi, method in enumerate(methods):
                sub = agg[agg["method"] == method]
                fig_bar.add_trace(go.Bar(
                    name=method,
                    x=sub["attack"],
                    y=sub["mean_oob"],
                    error_y=dict(type="data", array=sub["std_oob"], visible=True),
                    marker_color=palette[mi % len(palette)],
                ))
            fig_bar.update_layout(
                **PLOTLY_LAYOUT,
                barmode="group",
                height=460,
                xaxis_title="Attack Type",
                yaxis_title="Mean Time OOB (s)",
                legend=dict(font=dict(size=10)),
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("CSV columns not in expected format.")

        # --- Per-seed RL table ---
        st.markdown("### RL DQN Per-Seed Breakdown (Time OOB in seconds)")
        rl_mask = df_all["method"].str.contains("DQN", case=False, na=False)
        if rl_mask.any():
            rl_df = df_all[rl_mask]
            if "seed" in rl_df.columns or "model_seed" in rl_df.columns:
                seed_col = "model_seed" if "model_seed" in rl_df.columns else "seed"
                pivot = (rl_df.groupby([seed_col, "attack_label"])["time_outside_safe_band"]
                         .mean().unstack(fill_value=0).round(2))
                st.dataframe(pivot, use_container_width=True)
            else:
                st.info("Seed column not found; showing aggregate.")
                st.dataframe(rl_df.head(20))

        # --- RL vs Detector-Timing vs Kalman-Only comparison ---
        st.markdown("### RL vs Detector-Timing vs Kalman-Only")
        compare_methods = ["RL (DQN R2", "Detector-Threshold-Timing", "Kalman-Only"]
        mask = df_all["method"].apply(
            lambda m: any(cm in str(m) for cm in compare_methods)
        )
        if mask.any():
            comp_df = df_all[mask]
            comp_agg = (comp_df.groupby(["method", "attack_label"])["time_outside_safe_band"]
                        .agg(["mean", "std"]).reset_index())
            comp_agg.columns = ["method", "attack", "mean_oob", "std_oob"]
            comp_agg["std_oob"] = comp_agg["std_oob"].fillna(0)

            fig_comp = go.Figure()
            comp_palette = {"Kalman": "#06d6a0", "Detector": "#ffd166", "RL": "#00b4d8"}
            for method in comp_agg["method"].unique():
                sub = comp_agg[comp_agg["method"] == method]
                col = "#00b4d8"
                for k, v in comp_palette.items():
                    if k in method:
                        col = v
                        break
                fig_comp.add_trace(go.Bar(
                    name=method, x=sub["attack"], y=sub["mean_oob"],
                    error_y=dict(type="data", array=sub["std_oob"], visible=True),
                    marker_color=col,
                ))
            fig_comp.update_layout(
                **PLOTLY_LAYOUT, barmode="group", height=400,
                xaxis_title="Attack Type", yaxis_title="Mean OOB (s)",
            )
            st.plotly_chart(fig_comp, use_container_width=True)

    else:
        st.warning(f"Results file not found: `{csv_path}`")

    # ---------- Widespoof fair results ----------
    if os.path.exists(ws_md):
        with st.expander("📄 Widened-Spoof Fair Comparison (full report)"):
            st.markdown(open(ws_md, encoding="utf-8").read())

    # ---------- Limitations ----------
    st.markdown("### ⚠️ Known Limitations")
    st.markdown("""
<div class="limits-box">
<ol>
<li><b>RL does not beat Kalman-Only on spoof / stealthy drift.</b>  
   The hand-crafted innovation-residual threshold detects abrupt or gradual spoofs
   on the first anomalous sample, whereas the RL agent occasionally trusts a few
   corrupt readings before reacting.</li>
<li><b>A hand-tuned timing detector matches or beats RL on telemetry freeze.</b>  
   Detector-Timing achieves <b>0.0 s OOB</b> across all freeze scenarios because
   it directly inspects packet-age / duplicate flags, while the RL policy learned
   a noisier proxy from the raw observation vector.</li>
<li><b>RL is seed-sensitive and brittle outside its training range.</b>  
   Per-seed OOB varies dramatically (Seed 0: 0.0 s, Seed 1: 11.66 s on spoof).
   The widened-spoof experiment showed partial improvement but seeds can swap
   winners unpredictably.</li>
<li><b>Delay and DOS share one mechanism.</b>  
   Both <code>delay</code> and <code>dos</code> attack types execute the same
   stale-hold code path in <code>grid_env.py</code>; the configured latency / drop
   parameters are never read by <code>step()</code>.</li>
</ol>
</div>
""", unsafe_allow_html=True)
