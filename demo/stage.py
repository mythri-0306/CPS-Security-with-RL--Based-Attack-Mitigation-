"""
Combined "stage" component for the Grid-Defense demo (demo/stage.py)
===================================================================

One self-contained HTML/JS panel that shows BOTH:
  * a live substation schematic -- generator (spinning turbine), sensor (PMU),
    controller, attacker (emits red packets during the attack window), load;
  * the five-defense frequency graph underneath.

A single Play / Pause / Restart + scrub drives both, entirely client-side, so
the buttons never break on a Streamlit rerun.

Theme: glacier blue + navy + lemon.

Public API:
    render_stage(series, t, attack_start, attack_end, scenario, ...)
"""

import json
import streamlit.components.v1 as components

# Glacier / navy / lemon line palette (No Defense = coral villain, RL = navy hero)
COLORS = {
    "No Defense":      "#E0523F",   # coral  (attack succeeds)
    "Rule-Based":      "#9AA7B0",   # slate grey
    "Kalman-Only":     "#3E8EB0",   # glacier blue
    "Detector-Timing": "#D8B22E",   # lemon
    "RL DQN":          "#0E2A47",   # navy   (hero)
}
ORDER = ["No Defense", "Rule-Based", "Kalman-Only", "Detector-Timing", "RL DQN"]


_TEMPLATE = r"""
<div id="stage">
  <div id="scn">__SCN__</div>
  <svg id="sch" viewBox="0 0 820 196" preserveAspectRatio="xMidYMid meet"></svg>
  <div id="legend"></div>
  <canvas id="cv"></canvas>
  <div id="bar">
    <button class="btn" id="play">&#9654;&nbsp; Play</button>
    <button class="btn ghost" id="pause">&#10073;&#10073;&nbsp; Pause</button>
    <button class="btn ghost" id="restart">&#8635;&nbsp; Restart</button>
    <input id="scrub" type="range" min="0" max="1000" value="0">
    <span id="clock">0.0 s</span>
  </div>
</div>
<style>
  html,body{margin:0;padding:0;background:transparent}
  #stage{box-sizing:border-box;font-family:Inter,-apple-system,Segoe UI,sans-serif;color:#0E2A47;
    background:linear-gradient(180deg,#f3fafd,#ffffff);border:1px solid #d6e6ee;
    border-radius:16px;padding:14px 18px 14px;box-shadow:0 6px 22px -16px rgba(14,42,71,.5)}
  #scn{font-size:.82rem;font-weight:700;letter-spacing:.03em;color:#2E6E8E;
    text-transform:uppercase;margin-bottom:6px}
  #sch{display:block;width:100%;height:210px}
  #legend{display:flex;flex-wrap:wrap;gap:15px;margin:4px 0 2px}
  .lg{display:flex;align-items:center;gap:7px;font-size:.8rem;color:#33506a}
  .sw{width:22px;height:3px;border-radius:2px;display:inline-block}
  .lg.hero{font-weight:800;color:#0E2A47}
  canvas{display:block;width:100%;height:auto}
  #bar{display:flex;align-items:center;gap:10px;margin-top:10px}
  .btn{font:700 .82rem Inter,sans-serif;border:none;cursor:pointer;border-radius:10px;
    padding:9px 16px;background:#0E2A47;color:#fff;transition:.15s}
  .btn:hover{background:#1b4068}
  .btn.ghost{background:#e7f0f5;color:#2E6E8E}
  .btn.ghost:hover{background:#d8e8ef}
  #scrub{flex:1;accent-color:#3E8EB0;height:4px}
  #clock{font:700 .82rem ui-monospace,monospace;color:#2E6E8E;min-width:54px;text-align:right}
</style>
<script>
(function(){
  const D = __DATA__;
  const SVGNS="http://www.w3.org/2000/svg";
  const t=D.t, N=t.length, tmax=t[N-1], dt=D.dt;
  const methods=D.order, S=D.series, COL=D.colors;
  const aStart=D.attack_start, aEnd=D.attack_end;

  // ============================ SCHEMATIC =================================
  const sch=document.getElementById("sch");
  function el(tag,a,p){const e=document.createElementNS(SVGNS,tag);for(const k in a)e.setAttribute(k,a[k]);(p||sch).appendChild(e);return e;}
  const CGLAC="#3E8EB0", CNAVY="#0E2A47", CLEM="#E4C83A", CRED="#E0523F",
        CWIRE="#bcd4e0", CNODE="#ffffff", CSTROKE="#9cc3d6", CDIM="#6f94a8";
  const GEN={x:92,y:120}, SENS={x:250,y:120}, CTRL={x:508,y:120},
        ATK={x:384,y:48}, LOAD={x:690,y:120}, LY=120;

  // defs: glow
  const defs=el("defs",{});
  defs.innerHTML='<filter id="gl" x="-60%" y="-60%" width="220%" height="220%">'+
    '<feGaussianBlur stdDeviation="2.2" result="b"/><feMerge><feMergeNode in="b"/>'+
    '<feMergeNode in="SourceGraphic"/></feMerge></filter>';
  // wires
  el("line",{x1:GEN.x+32,y1:LY,x2:SENS.x-28,y2:LY,stroke:CWIRE,"stroke-width":3});
  el("line",{x1:SENS.x+28,y1:LY,x2:CTRL.x-32,y2:LY,stroke:CWIRE,"stroke-width":3});
  el("line",{x1:CTRL.x+32,y1:LY,x2:LOAD.x-26,y2:LY,stroke:CWIRE,"stroke-width":3});
  el("path",{d:`M${CTRL.x} ${LY+30} L${CTRL.x} 170 L${GEN.x} 170 L${GEN.x} ${LY+32}`,
    fill:"none",stroke:"#d4e6ee","stroke-width":2,"stroke-dasharray":"4 5"});
  el("text",{x:(GEN.x+CTRL.x)/2,y:186,fill:CDIM,"font-size":10,"text-anchor":"middle",
    "font-family":"ui-monospace,monospace"}).textContent="control loop → hold 50 Hz";
  el("line",{x1:ATK.x,y1:ATK.y+22,x2:ATK.x,y2:LY,stroke:"#e4b7ae","stroke-width":2,"stroke-dasharray":"3 4"});

  function label(x,y,tt,s,c){el("text",{x,y,fill:c||CNAVY,"font-size":11,"text-anchor":"middle",
    "font-weight":"700","font-family":"Inter,sans-serif"}).textContent=tt;
    if(s)el("text",{x,y:y+12,fill:CDIM,"font-size":8.5,"text-anchor":"middle",
      "font-family":"ui-monospace,monospace"}).textContent=s;}

  // generator + turbine
  el("circle",{cx:GEN.x,cy:GEN.y,r:30,fill:CNODE,stroke:CSTROKE,"stroke-width":2});
  const turb=el("g",{});
  for(let i=0;i<6;i++){const a=i*Math.PI/3;el("line",{x1:GEN.x,y1:GEN.y,
    x2:GEN.x+22*Math.cos(a),y2:GEN.y+22*Math.sin(a),stroke:CGLAC,"stroke-width":3,
    "stroke-linecap":"round"},turb);}
  el("circle",{cx:GEN.x,cy:GEN.y,r:4.5,fill:CNAVY},turb);
  label(GEN.x,GEN.y+46,"GENERATOR","turbine + inertia");
  // sensor (PMU)
  el("rect",{x:SENS.x-26,y:SENS.y-24,width:52,height:48,rx:9,fill:CNODE,stroke:CSTROKE,"stroke-width":2});
  const led=el("circle",{cx:SENS.x,cy:SENS.y-6,r:4,fill:CLEM});
  el("text",{x:SENS.x,y:SENS.y+10,fill:CDIM,"font-size":8.5,"text-anchor":"middle",
    "font-family":"ui-monospace,monospace"}).textContent="PMU";
  label(SENS.x,SENS.y+44,"SENSOR","Modbus/TCP");
  // controller
  const cbox=el("rect",{x:CTRL.x-28,y:CTRL.y-26,width:56,height:52,rx:9,fill:CNODE,stroke:CSTROKE,"stroke-width":2});
  el("path",{d:`M${CTRL.x-12} ${CTRL.y-4} h24 M${CTRL.x-12} ${CTRL.y+5} h24`,stroke:CGLAC,
    "stroke-width":2.5,"stroke-linecap":"round"});
  label(CTRL.x,CTRL.y+44,"CONTROLLER","PI control");
  // attacker
  const an=el("g",{opacity:.35});
  const abox=el("rect",{x:ATK.x-24,y:ATK.y-20,width:48,height:42,rx:9,fill:"#fdecea",
    stroke:"#eeb6ac","stroke-width":2},an);
  el("path",{d:`M${ATK.x} ${ATK.y-9} l7 14 h-14 z`,fill:"none",stroke:CRED,"stroke-width":2,
    "stroke-linejoin":"round"},an);
  el("text",{x:ATK.x,y:ATK.y+7,fill:CRED,"font-size":12,"text-anchor":"middle","font-weight":"800",
    "font-family":"ui-monospace,monospace"},an).textContent="!";
  label(ATK.x,ATK.y-28,"ATTACKER","192.168.1.77",CRED);
  // load / city
  el("rect",{x:LOAD.x-22,y:LOAD.y-26,width:44,height:52,rx:7,fill:CNODE,stroke:CSTROKE,"stroke-width":2});
  const cityG=el("g",{});
  for(let r=0;r<4;r++)for(let c=0;c<3;c++)el("rect",{x:LOAD.x-15+c*11,y:LOAD.y-19+r*11,
    width:6,height:6,rx:1.3,fill:CLEM,class:"win",opacity:.9},cityG);
  label(LOAD.x,LOAD.y+44,"LOAD","city");
  const pk=el("g",{});

  let packets=[], spawnT=0, genAngle=0;
  function spawn(kind){ if(kind==="attack")packets.push({kind,x:ATK.x,y:ATK.y+22,phase:"drop"});
    else packets.push({kind,x:SENS.x+28,y:LY,phase:"run"}); }

  function schUpdate(frameDt, attackLive){
    // turbine spin + sensor/attacker state
    genAngle += (1.1+ (attackLive?0.8:0))*frameDt*60;
    turb.setAttribute("transform",`rotate(${genAngle} ${GEN.x} ${GEN.y})`);
    led.setAttribute("fill", attackLive?CRED:CLEM);
    an.setAttribute("opacity", attackLive?1:.35);
    abox.setAttribute("stroke", attackLive?CRED:"#eeb6ac");
    cbox.setAttribute("stroke", attackLive?CRED:CSTROKE);
    // spawn packets
    spawnT-=frameDt; if(spawnT<=0){ spawnT=.34; spawn("sensor"); if(attackLive)spawn("attack"); }
    const v=165;
    for(const p of packets){ if(p.phase==="drop"){p.y+=v*frameDt; if(p.y>=LY){p.y=LY;p.phase="run";p.x=ATK.x;}}
      else p.x+=v*frameDt;
      if(p.x>=CTRL.x-30) p.dead=true; }
    packets=packets.filter(p=>!p.dead);
    // redraw packets
    pk.textContent="";
    for(const p of packets){ const col=p.kind==="attack"?CRED:(attackLive?CLEM:CGLAC);
      const c=el("circle",{cx:p.x,cy:p.y,r:p.kind==="attack"?4.6:3.4,fill:col,filter:"url(#gl)"},pk); }
  }

  // ============================== GRAPH ==================================
  const cv=document.getElementById("cv"), ctx=cv.getContext("2d");
  let ymin=D.low, ymax=D.high;
  for(const m of methods)for(const v of S[m]){ if(v<ymin)ymin=v; if(v>ymax)ymax=v; }
  const pad=(ymax-ymin)*0.08||0.1; ymin-=pad; ymax+=pad;

  const lg=document.getElementById("legend");
  for(const m of methods){ const d=document.createElement("div");
    d.className="lg"+(m==="RL DQN"?" hero":"");
    d.innerHTML='<span class="sw" style="background:'+COL[m]+'"></span>'+(m==="RL DQN"?"RL Defender":m);
    lg.appendChild(d); }

  const ML=54,MR=16,MT=12,MB=26; let W=0,H=0,dpr=Math.max(1,window.devicePixelRatio||1);
  function size(){ W=cv.clientWidth; H=Math.max(230,Math.min(300,Math.round(W*0.34)));
    cv.width=W*dpr; cv.height=H*dpr; cv.style.height=H+"px"; ctx.setTransform(dpr,0,0,dpr,0,0); }
  const X=x=>ML+(x/tmax)*(W-ML-MR), Y=y=>(H-MB)-((y-ymin)/(ymax-ymin))*(H-MT-MB);

  function drawGraph(f){
    ctx.clearRect(0,0,W,H);
    ctx.fillStyle="#eaf5ef"; ctx.fillRect(ML,Y(D.high),W-ML-MR,Y(D.low)-Y(D.high));
    ctx.strokeStyle="#cde7d6"; ctx.lineWidth=1; ctx.setLineDash([4,4]);
    ctx.beginPath();ctx.moveTo(ML,Y(D.high));ctx.lineTo(W-MR,Y(D.high));
    ctx.moveTo(ML,Y(D.low));ctx.lineTo(W-MR,Y(D.low));ctx.stroke();ctx.setLineDash([]);
    if(aStart<tmax){ ctx.strokeStyle="#e9c3bc";ctx.lineWidth=1.2;ctx.setLineDash([5,5]);
      ctx.beginPath();ctx.moveTo(X(aStart),MT);ctx.lineTo(X(aStart),H-MB);ctx.stroke();ctx.setLineDash([]);
      ctx.fillStyle="#c08075";ctx.font="11px Inter,sans-serif";ctx.textAlign="left";
      ctx.fillText("attack starts",X(aStart)+4,MT+11); }
    ctx.font="11px Inter,sans-serif";
    const yt=5; ctx.textAlign="right";
    for(let i=0;i<=yt;i++){ const val=ymin+(ymax-ymin)*i/yt,py=Y(val);
      ctx.strokeStyle="#eef3f6";ctx.beginPath();ctx.moveTo(ML,py);ctx.lineTo(W-MR,py);ctx.stroke();
      ctx.fillStyle="#8aa6b5";ctx.fillText(val.toFixed(2),ML-8,py+3); }
    ctx.textAlign="center";ctx.fillStyle="#8aa6b5";
    for(let s=0;s<=Math.floor(tmax);s+=10)ctx.fillText(s+"s",X(s),H-MB+15);
    ctx.fillStyle="#2E6E8E";ctx.font="12px Inter,sans-serif";
    ctx.save();ctx.translate(13,(MT+H-MB)/2);ctx.rotate(-Math.PI/2);ctx.textAlign="center";
    ctx.fillText("Grid frequency (Hz)",0,0);ctx.restore();
    for(const m of methods){ const arr=S[m],hero=(m==="RL DQN");
      ctx.strokeStyle=COL[m];ctx.lineWidth=hero?2.8:1.7;ctx.globalAlpha=hero?1:.92;
      ctx.lineJoin="round";ctx.beginPath();
      for(let i=0;i<=f;i++){const px=X(t[i]),py=Y(arr[i]);i?ctx.lineTo(px,py):ctx.moveTo(px,py);}
      ctx.stroke();ctx.globalAlpha=1;
      ctx.fillStyle=COL[m];ctx.beginPath();ctx.arc(X(t[f]),Y(arr[f]),hero?4:3,0,7);ctx.fill(); }
    ctx.strokeStyle="#c3d6e0";ctx.lineWidth=1;ctx.setLineDash([3,3]);
    ctx.beginPath();ctx.moveTo(X(t[f]),MT);ctx.lineTo(X(t[f]),H-MB);ctx.stroke();ctx.setLineDash([]);
  }

  // ========================= SHARED CONTROLS ============================
  let frame=0, playing=false, last=0;
  const SPEED=100;
  const scrub=document.getElementById("scrub"), clock=document.getElementById("clock");
  function paint(){ const fi=Math.floor(frame); drawGraph(fi);
    scrub.value=Math.round(frame/(N-1)*1000); clock.textContent=(fi*dt).toFixed(1)+" s"; }
  function setPlay(p){ playing=p; last=0; }
  document.getElementById("play").onclick   =()=>{ if(frame>=N-1)frame=0; setPlay(true); };
  document.getElementById("pause").onclick   =()=> setPlay(false);
  document.getElementById("restart").onclick =()=>{ frame=0; packets=[]; setPlay(true); paint(); };
  scrub.oninput=e=>{ setPlay(false); frame=(+e.target.value/1000)*(N-1); paint(); };

  function loop(ts){ if(!last)last=ts; const d=Math.min(.05,(ts-last)/1000); last=ts;
    const tt=Math.floor(frame)*dt; const attackLive=(tt>=aStart && tt<aEnd);
    if(playing){ frame+=SPEED*d; if(frame>=N-1){frame=N-1;playing=false;} paint(); }
    schUpdate(playing? d : 0, attackLive);
    requestAnimationFrame(loop); }
  window.addEventListener("resize",()=>{ size(); paint(); });
  size(); paint(); requestAnimationFrame(loop);
  setTimeout(()=>setPlay(true),500);
})();
</script>
"""


def render_stage(series: dict, t, attack_start: float, attack_end: float,
                 scenario: str, dt: float = 0.1, low: float = 49.80,
                 high: float = 50.20, height: int = 620):
    data = {
        "order":  [m for m in ORDER if m in series],
        "series": {m: [round(float(v), 4) for v in series[m]] for m in series},
        "colors": COLORS,
        "t":      [round(float(x), 3) for x in t],
        "attack_start": round(float(attack_start), 2),
        "attack_end":   round(float(attack_end), 2),
        "dt": dt, "low": low, "high": high,
    }
    html = (_TEMPLATE
            .replace("__DATA__", json.dumps(data, separators=(",", ":")))
            .replace("__SCN__", scenario))
    components.html(html, height=height, scrolling=False)
