"""
Clean single-panel animated dashboard for the Grid-Defense demo (demo/dashboard.py)
===================================================================================

Renders ONE professional, self-contained HTML/JS canvas that animates the true
grid frequency of every defense on the same attack, with Play / Pause / Restart
and a scrub bar that all work client-side (no Streamlit reruns, so the buttons
never break).  No emojis, no clutter.

Public API:
    render_dashboard(series, t, attack_start, scenario, height)
"""

import json
import streamlit.components.v1 as components

# Professional, restrained palette (RL is the dark hero line).
COLORS = {
    "No Defense":      "#C0485B",   # muted red  (danger)
    "Rule-Based":      "#AEB6BF",   # light grey
    "Kalman-Only":     "#6FA8C7",   # soft blue
    "Detector-Timing": "#6E9E6B",   # muted green
    "RL DQN":          "#1C3D5A",   # deep navy  (hero)
}
ORDER = ["No Defense", "Rule-Based", "Kalman-Only", "Detector-Timing", "RL DQN"]


_TEMPLATE = r"""
<div id="wrap">
  <div id="scn">__SCN__</div>
  <div id="legend"></div>
  <canvas id="cv"></canvas>
  <div id="bar">
    <button class="btn" id="play">Play</button>
    <button class="btn ghost" id="pause">Pause</button>
    <button class="btn ghost" id="restart">Restart</button>
    <input id="scrub" type="range" min="0" max="1000" value="0">
    <span id="clock">0.0 s</span>
  </div>
</div>
<style>
  #wrap{font-family:Inter,-apple-system,Segoe UI,sans-serif;color:#1f2a37;
        background:#ffffff;border:1px solid #e6e8eb;border-radius:14px;
        padding:16px 18px 14px;box-shadow:0 1px 3px rgba(16,24,40,.06)}
  #scn{font-size:.82rem;font-weight:600;letter-spacing:.02em;color:#64748b;margin-bottom:8px}
  #legend{display:flex;flex-wrap:wrap;gap:16px;margin-bottom:6px}
  .lg{display:flex;align-items:center;gap:7px;font-size:.8rem;color:#334155}
  .sw{width:22px;height:3px;border-radius:2px;display:inline-block}
  .lg.hero{font-weight:700;color:#1C3D5A}
  canvas{display:block;width:100%;height:auto}
  #bar{display:flex;align-items:center;gap:10px;margin-top:10px}
  .btn{font:600 .82rem Inter,sans-serif;border:none;cursor:pointer;border-radius:9px;
       padding:9px 18px;background:#1C3D5A;color:#fff;transition:background .15s}
  .btn:hover{background:#274d70}
  .btn.ghost{background:#eef1f4;color:#334155}
  .btn.ghost:hover{background:#e2e7ec}
  #scrub{flex:1;accent-color:#1C3D5A;height:4px}
  #clock{font:600 .82rem ui-monospace,monospace;color:#475569;min-width:52px;text-align:right}
</style>
<script>
(function(){
  const D = __DATA__;
  const cv = document.getElementById("cv"), ctx = cv.getContext("2d");
  const t = D.t, N = t.length, tmax = t[N-1];
  const methods = D.order, S = D.series, COL = D.colors;

  // y-range from all series + the safe band, padded
  let ymin=D.low, ymax=D.high;
  for(const m of methods){ for(const v of S[m]){ if(v<ymin)ymin=v; if(v>ymax)ymax=v; } }
  const pad=(ymax-ymin)*0.08||0.1; ymin-=pad; ymax+=pad;

  // legend
  const lg=document.getElementById("legend");
  for(const m of methods){ const d=document.createElement("div");
    d.className="lg"+(m==="RL DQN"?" hero":"");
    d.innerHTML='<span class="sw" style="background:'+COL[m]+'"></span>'+(m==="RL DQN"?"RL Defender":m);
    lg.appendChild(d); }

  const ML=56, MR=18, MT=14, MB=30;  // plot margins
  let W=0, H=0, dpr=Math.max(1,window.devicePixelRatio||1);
  function size(){ W=cv.clientWidth; H=Math.round(W*0.46); H=Math.max(300,Math.min(470,H));
    cv.width=W*dpr; cv.height=H*dpr; cv.style.height=H+"px"; ctx.setTransform(dpr,0,0,dpr,0,0); }
  const X=x=> ML + (x/tmax)*(W-ML-MR);
  const Y=y=> (H-MB) - ((y-ymin)/(ymax-ymin))*(H-MT-MB);

  function draw(f){
    f=Math.max(0,Math.min(f,N-1));
    ctx.clearRect(0,0,W,H);
    // safe band
    ctx.fillStyle="#eef6ef"; ctx.fillRect(ML,Y(D.high),W-ML-MR,Y(D.low)-Y(D.high));
    ctx.strokeStyle="#cfe3d2"; ctx.lineWidth=1; ctx.setLineDash([4,4]);
    ctx.beginPath();ctx.moveTo(ML,Y(D.high));ctx.lineTo(W-MR,Y(D.high));
    ctx.moveTo(ML,Y(D.low));ctx.lineTo(W-MR,Y(D.low));ctx.stroke(); ctx.setLineDash([]);
    // attack onset marker
    if(D.attack_start<tmax){ ctx.strokeStyle="#e5c7c7"; ctx.lineWidth=1.2; ctx.setLineDash([5,5]);
      ctx.beginPath();ctx.moveTo(X(D.attack_start),MT);ctx.lineTo(X(D.attack_start),H-MB);ctx.stroke();
      ctx.setLineDash([]); ctx.fillStyle="#b08585"; ctx.font="11px Inter,sans-serif";
      ctx.textAlign="left"; ctx.fillText("attack starts", X(D.attack_start)+4, MT+11); }
    // axes
    ctx.strokeStyle="#e6e8eb"; ctx.lineWidth=1;
    ctx.fillStyle="#94a3b8"; ctx.font="11px Inter,sans-serif";
    // y ticks
    const yt=5; ctx.textAlign="right";
    for(let i=0;i<=yt;i++){ const val=ymin+(ymax-ymin)*i/yt, py=Y(val);
      ctx.beginPath();ctx.moveTo(ML,py);ctx.lineTo(W-MR,py);ctx.strokeStyle="#f1f3f5";ctx.stroke();
      ctx.fillText(val.toFixed(2), ML-8, py+3); }
    // x ticks
    ctx.textAlign="center";
    for(let s=0;s<=Math.floor(tmax);s+=10){ const px=X(s);
      ctx.fillText(s+"s", px, H-MB+16); }
    // axis labels
    ctx.fillStyle="#64748b"; ctx.font="12px Inter,sans-serif";
    ctx.save();ctx.translate(14,(MT+H-MB)/2);ctx.rotate(-Math.PI/2);
    ctx.textAlign="center";ctx.fillText("Grid frequency (Hz)",0,0);ctx.restore();
    // lines up to frame f
    for(const m of methods){ const arr=S[m], hero=(m==="RL DQN");
      ctx.strokeStyle=COL[m]; ctx.lineWidth=hero?2.8:1.7; ctx.globalAlpha=hero?1:0.9;
      ctx.lineJoin="round"; ctx.beginPath();
      for(let i=0;i<=f;i++){ const px=X(t[i]),py=Y(arr[i]); i?ctx.lineTo(px,py):ctx.moveTo(px,py); }
      ctx.stroke(); ctx.globalAlpha=1;
      // leading dot
      ctx.fillStyle=COL[m]; ctx.beginPath();ctx.arc(X(t[f]),Y(arr[f]),hero?4:3,0,7);ctx.fill(); }
    // playhead
    ctx.strokeStyle="#cbd5e1"; ctx.lineWidth=1; ctx.setLineDash([3,3]);
    ctx.beginPath();ctx.moveTo(X(t[f]),MT);ctx.lineTo(X(t[f]),H-MB);ctx.stroke();ctx.setLineDash([]);
  }

  // ---- animation / controls (all client-side) ----
  let frame=0, playing=false, last=0;
  const SPEED=110;  // sim-steps per wall-second
  const scrub=document.getElementById("scrub"), clock=document.getElementById("clock");
  function paint(){ const fi=Math.floor(frame); draw(fi);
    scrub.value=Math.round(frame/(N-1)*1000); clock.textContent=(fi*D.dt).toFixed(1)+" s"; }
  function setPlay(p){ playing=p; last=0; }
  document.getElementById("play").onclick   =()=>{ if(frame>=N-1)frame=0; setPlay(true); };
  document.getElementById("pause").onclick   =()=> setPlay(false);
  document.getElementById("restart").onclick =()=>{ frame=0; setPlay(true); paint(); };
  scrub.oninput=e=>{ setPlay(false); frame=(+e.target.value/1000)*(N-1); paint(); };
  function loop(ts){ if(!last)last=ts; const dt=Math.min(.05,(ts-last)/1000); last=ts;
    if(playing){ frame+=SPEED*dt; if(frame>=N-1){frame=N-1; playing=false;} paint(); }
    requestAnimationFrame(loop); }
  window.addEventListener("resize",()=>{ size(); paint(); });
  size(); paint(); requestAnimationFrame(loop);
  setTimeout(()=>setPlay(true),450);
})();
</script>
"""


def render_dashboard(series: dict, t, attack_start: float, scenario: str,
                     dt: float = 0.1, low: float = 49.80, high: float = 50.20,
                     height: int = 560):
    """Embed the animated comparison dashboard.

    series : {method_name: list_of_f_true}
    t      : list of time values (s)
    """
    data = {
        "order":  [m for m in ORDER if m in series],
        "series": {m: [round(float(v), 4) for v in series[m]] for m in series},
        "colors": COLORS,
        "t":      [round(float(x), 3) for x in t],
        "attack_start": round(float(attack_start), 1),
        "dt": dt, "low": low, "high": high,
    }
    html = (_TEMPLATE
            .replace("__DATA__", json.dumps(data, separators=(",", ":")))
            .replace("__SCN__", scenario))
    components.html(html, height=height, scrolling=False)
