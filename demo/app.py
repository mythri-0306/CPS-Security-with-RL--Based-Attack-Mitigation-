"""
Grid-Defense Demo -- Streamlit App  (demo/app.py)
=================================================

Scroll-through page:
  1. Animated title / landing section.
  2. Attack controls (type, strength, onset).
  3. Live substation simulation + five-defense frequency graph (one synced
     Play / Pause / Restart).

Theme: glacier blue + navy + lemon.  Real GridDefenseEnv, no fabricated numbers.

Run:  streamlit run demo/app.py
"""

import os
import sys
import streamlit as st
import streamlit.components.v1 as components

# --- path bootstrap (root config.py must win over examples/powergrid) -------
_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_DEMO_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
if _DEMO_DIR not in sys.path:
    sys.path.insert(1, _DEMO_DIR)
_PG_DIR = os.path.join(_ROOT_DIR, "examples", "powergrid")
if os.path.isdir(_PG_DIR) and _PG_DIR not in sys.path:
    sys.path.append(_PG_DIR)

from sim_engine import run_episode, DEFENSE_METHODS
from stage import render_stage

ENV_SEED = 42
DQN_SEED = 0

# label -> (engine key, intensity label, lo, hi, default, step, fmt)
ATTACKS = {
    "Spoof":            ("spoof",            "Spoofed frequency (Hz)", 48.00, 49.20, 48.50, 0.05, "%.2f"),
    "Stealthy drift":   ("stealthy_drift",   "Drift rate (Hz/s)",       0.02,  0.08,  0.05, 0.005, "%.3f"),
    "Replay":           ("replay",           "Replay window (s)",       5.0,  15.0,  10.0, 1.0,  "%.0f"),
    "Telemetry freeze": ("telemetry_freeze", None, 0, 0, 0, 0, None),
}

# ---------------------------------------------------------------------------
st.set_page_config(page_title="Power-Grid Cyber-Attack Simulation",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
  #MainMenu, footer, header {visibility:hidden;}
  [data-testid="stSidebar"]{display:none;}
  .block-container{padding-top:0rem;max-width:1180px;}
  html, body, [class*="css"]{font-family:Inter,-apple-system,Segoe UI,sans-serif;}
  [data-testid="stAppViewContainer"]{background:#eef5f9;}
  .sec-t{font-size:1.35rem;font-weight:800;color:#0E2A47;margin:10px 0 2px;}
  .sec-s{font-size:.92rem;color:#5b7488;margin-bottom:10px;}
  label{font-weight:600 !important;color:#2E6E8E !important;}
  div[data-baseweb="select"]>div{border-radius:9px;border-color:#cfe2ec;}
</style>
""", unsafe_allow_html=True)

# ===================== 1. ANIMATED TITLE / LANDING =========================
HERO = r"""
<div id="hero">
  <canvas id="bg"></canvas>
  <div id="ov">
    <div id="kick">INDUSTRIAL CYBER-PHYSICAL SECURITY</div>
    <h1 id="title">POWER-GRID<br><span>CYBER-ATTACK</span> SIMULATION</h1>
    <div id="sub">A reinforcement-learning defender against live sensor spoofing on a 50&nbsp;Hz grid</div>
    <div id="scroll">scroll to run an attack &#8595;</div>
  </div>
</div>
<style>
  #hero{position:relative;width:100%;height:440px;border-radius:18px;overflow:hidden;
    background:radial-gradient(120% 120% at 70% 10%,#14365a 0%,#0E2A47 55%,#081a30 100%);
    font-family:Inter,-apple-system,Segoe UI,sans-serif;}
  #bg{position:absolute;inset:0;width:100%;height:100%;display:block;}
  #ov{position:absolute;inset:0;display:flex;flex-direction:column;justify-content:center;
    padding:0 7%;opacity:0;transform:translateY(14px);animation:rise 1.1s .15s forwards cubic-bezier(.2,.7,.3,1);}
  #kick{color:#8fd0e8;font-size:.82rem;font-weight:700;letter-spacing:.34em;margin-bottom:12px;}
  #title{margin:0;color:#fff;font-weight:900;line-height:1.02;letter-spacing:-.01em;
    font-size:clamp(2.1rem,5.6vw,4.0rem);}
  #title span{color:#F2E24B;}
  #sub{margin-top:16px;color:#cfe4f0;font-size:clamp(.9rem,1.8vw,1.1rem);max-width:640px;}
  #scroll{margin-top:26px;color:#8fd0e8;font-size:.82rem;font-weight:600;letter-spacing:.08em;
    animation:bob 1.6s ease-in-out infinite;}
  @keyframes rise{to{opacity:1;transform:none;}}
  @keyframes bob{0%,100%{transform:translateY(0)}50%{transform:translateY(6px)}}
</style>
<script>
(function(){
  const cv=document.getElementById("bg"),c=cv.getContext("2d");
  let W,H,dpr=Math.max(1,window.devicePixelRatio||1);
  function size(){W=cv.clientWidth;H=cv.clientHeight;cv.width=W*dpr;cv.height=H*dpr;c.setTransform(dpr,0,0,dpr,0,0);}
  size();window.addEventListener("resize",size);
  const waves=[{a:14,k:0.012,s:0.6,y:.52,col:"rgba(63,142,176,.55)",w:2},
               {a:22,k:0.008,s:0.9,y:.6,col:"rgba(143,208,232,.35)",w:1.5},
               {a:10,k:0.02,s:1.4,y:.46,col:"rgba(242,226,75,.18)",w:1.5}];
  const pk=[]; for(let i=0;i<8;i++)pk.push({x:Math.random(),y:.4+Math.random()*.3,
    v:.03+Math.random()*.05,red:Math.random()<.22});
  let ph=0;
  function frame(){ c.clearRect(0,0,W,H);
    ph+=0.016;
    for(const wv of waves){ c.beginPath();
      for(let x=0;x<=W;x+=6){const y=H*wv.y+Math.sin(x*wv.k+ph*wv.s)*wv.a
        +Math.sin(x*wv.k*0.5+ph*wv.s*1.4)*wv.a*0.4;
        x?c.lineTo(x,y):c.moveTo(x,y);}
      c.strokeStyle=wv.col;c.lineWidth=wv.w;c.stroke(); }
    for(const p of pk){ p.x+=p.v*0.004; if(p.x>1.04){p.x=-.04;p.red=Math.random()<.22;}
      const px=p.x*W, py=H*p.y+Math.sin(px*0.012+ph*0.6)*14;
      c.beginPath();c.arc(px,py,p.red?3.4:2.6,0,7);
      c.fillStyle=p.red?"#E0523F":"#9fe0f4";c.shadowColor=c.fillStyle;c.shadowBlur=10;c.fill();c.shadowBlur=0; }
    requestAnimationFrame(frame); }
  frame();
})();
</script>
"""
components.html(HERO, height=452, scrolling=False)

# ============================ 2. CONTROLS ==================================
st.markdown('<div class="sec-t">Configure the attack</div>', unsafe_allow_html=True)
st.markdown('<div class="sec-s">Pick the attack, its strength, and when it begins. '
            'All five defenses run automatically on the real grid model.</div>',
            unsafe_allow_html=True)

c1, c2, c3 = st.columns([1.1, 1.3, 1.1])
with c1:
    attack_label = st.selectbox("Attack", list(ATTACKS.keys()), index=0)
key, int_label, lo, hi, dflt, step, fmt = ATTACKS[attack_label]
with c2:
    if int_label is not None:
        intensity = st.slider(int_label, float(lo), float(hi), float(dflt), float(step), format=fmt)
    else:
        intensity = None
        st.markdown("<div style='padding-top:30px;color:#8aa6b5;font-size:.86rem'>"
                    "Telemetry freeze holds the last valid reading &mdash; no strength to set.</div>",
                    unsafe_allow_html=True)
with c3:
    attack_start = st.slider("Attack starts at (s)", 5.0, 40.0, 15.0, 1.0)

# ------- run all five defenses (cached) -------
@st.cache_data(show_spinner=False)
def run_all(attack_key, intensity, attack_start):
    series, t, start, end = {}, None, attack_start, attack_start
    for m in DEFENSE_METHODS:
        r = run_episode(attack_key, m, env_seed=ENV_SEED, intensity=intensity,
                        attack_start=attack_start, dqn_seed=DQN_SEED)
        series[m] = [float(x) for x in r["f_true"]]
        if t is None:
            t = [float(x) for x in r["t"]]
            start = float(r["attack_start"])
            aa = r["attack_active"]
            idx = [i for i, v in enumerate(aa) if v]
            end = float(t[idx[-1]]) if idx else start
    return series, t, start, end

with st.spinner("Simulating all five defenses on the real grid model…"):
    series, t, start, end = run_all(key, intensity, attack_start)

# ===================== 3. SIMULATION + GRAPH ===============================
st.write("")
st.markdown('<div class="sec-t">Live simulation</div>', unsafe_allow_html=True)
if int_label is not None:
    unit = int_label.split('(')[-1].rstrip(')')
    scn = f"{attack_label} · {(fmt % intensity)} {unit} · onset {attack_start:.0f}s"
else:
    scn = f"{attack_label} · onset {attack_start:.0f}s"

render_stage(series, t, start, end, scn, height=720)

st.markdown("<div style='color:#8aa6b5;font-size:.82rem;margin-top:4px'>"
            "Attacker injects packets over Modbus/TCP during the shaded window. "
            "Safe band 49.80&ndash;50.20 Hz. Each line is the true grid frequency that defense held.</div>",
            unsafe_allow_html=True)
