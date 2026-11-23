"""
Animated power-grid "setup" view for the Streamlit demo (demo/grid_view.py)
===========================================================================

Renders a live one-line substation schematic -- generator, Modbus sensor,
controller, attacker host, and the city/load -- that animates through a single
trajectory produced by sim_engine.run_episode().  Embedded in app.py via
streamlit.components.v1.html so it plays smoothly in its own iframe (no server
round-trips).  Pure SVG, no external libraries.

Light "ember" theme: warm cream background, orange/red accents.

Public API:
    grid_view_html(traj, attack_label, defense_label) -> str   # self-contained HTML
    render_grid_view(traj, attack_label, defense_label, height) # calls st.components
"""

import json


def _traj_to_json(traj: dict) -> str:
    """Pull just what the animation needs, as compact JSON (lists, not ndarrays)."""
    def arr(k, dec=4):
        return [round(float(x), dec) for x in traj[k]]

    payload = {
        "f_true": arr("f_true"),
        "f_meas": arr("f_meas"),
        "f_pred": arr("f_pred"),
        "action": [int(x) for x in traj["action"]],
        "attack_active": [int(x) for x in traj["attack_active"]],
        "oob": [int(x) for x in traj["oob"]],
        "attack_start": round(float(traj.get("attack_start", 20.0)), 1),
        "time_oob": round(float(traj.get("time_oob", 0.0)), 2),
        "max_dev": round(float(traj.get("max_deviation", 0.0)), 3),
        "dt": 0.1, "nominal": 50.0, "low": 49.8, "high": 50.2,
    }
    return json.dumps(payload, separators=(",", ":"))


_TEMPLATE = r"""
<div id="root" style="font-family:Inter,system-ui,sans-serif;color:#43281a;
     background:linear-gradient(180deg,#fffaf4,#fff1e3);border:1px solid #f3d3ba;
     border-radius:16px;padding:12px 14px 14px;overflow:hidden;
     box-shadow:0 6px 22px -14px rgba(180,80,20,.5)">
  <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:2px;flex-wrap:wrap">
    <div style="font:700 .68rem/1 'JetBrains Mono',monospace;letter-spacing:.16em;color:#b06a3a;text-transform:uppercase">
      &#9673; SUBSTATION ONE-LINE &middot; __SCN__</div>
    <div style="display:flex;align-items:center;gap:10px">
      <span id="schSt" style="font:700 .68rem/1 'JetBrains Mono',monospace;letter-spacing:.12em;color:#15935f">SECURE</span>
      <button id="btnPlay" style="font:600 .76rem/1 Inter,sans-serif;color:#fff;cursor:pointer;
        border:none;background:#ef6c1a;border-radius:9px;padding:8px 14px;box-shadow:0 2px 8px -2px rgba(239,108,26,.6)">&#9654; Play</button>
    </div>
  </div>
  <svg id="sch" viewBox="0 0 760 264" style="display:block;width:100%;height:auto"></svg>
  <div style="display:flex;align-items:center;gap:12px;margin-top:2px">
    <span id="clock" style="font:700 .76rem/1 'JetBrains Mono',monospace;color:#b06a3a;white-space:nowrap">T+0.0s</span>
    <input id="scrub" type="range" min="0" max="600" value="0" step="1" style="flex:1;accent-color:#ef6c1a">
    <span id="freq" style="font:800 1.05rem/1 'JetBrains Mono',monospace;color:#ef6c1a;white-space:nowrap">50.00 Hz</span>
    <span id="pill" style="font:800 .66rem/1 'JetBrains Mono',monospace;letter-spacing:.08em;text-transform:uppercase;
      padding:5px 11px;border-radius:999px;border:1px solid #bfe6d1;color:#15935f;background:#e9f7ef;white-space:nowrap">NOMINAL</span>
  </div>
  <div id="note" style="font:500 .78rem/1.45 Inter,sans-serif;color:#7a4a2c;margin-top:8px;min-height:2.3em"></div>
</div>
<script>
(function(){
  const T = __TRAJ__;
  const DT=T.dt, NOM=T.nominal, LO=T.low, HI=T.high, TRIP_HI=51.5, TRIP_LO=48.5;
  const N = T.f_true.length;
  const ATK="__ATK__", DEF="__DEF__";
  // ---- palette (light ember) ----
  const C={ wire:"#e7c3a6", node:"#ffffff", nodeStroke:"#e2b48f", ink:"#43281a", dim:"#9a6b4f",
            orange:"#ef6c1a", red:"#dc2626", amber:"#e8a30c", green:"#15935f", shield:"#0d9488",
            lit:"#f6a929", dark:"#d8c3b2" };
  const BLURB={
    spoof:"Attacker writes a fake low frequency (Modbus FC6); the controller over-generates.",
    replay:"Attacker replays stale nominal readings, blinding the controller to the real drift.",
    "stealthy drift":"Attacker nudges the reading a hair each step — a slow, quiet poisoning.",
    "telemetry freeze":"Attacker freezes the last valid packet; the reading goes stale.",
    none:"No attack — a clean baseline run."
  };
  const SVGNS="http://www.w3.org/2000/svg";
  const SCH=document.getElementById("sch");
  function el(t,a,p){const e=document.createElementNS(SVGNS,t);for(const k in a)e.setAttribute(k,a[k]);(p||SCH).appendChild(e);return e;}
  const GEN={x:92,y:118}, SENS={x:262,y:118}, CTRL={x:532,y:118}, ATK_N={x:397,y:46}, CITY={x:668,y:118}, LY=118;

  const defs=el("defs",{});
  defs.innerHTML='<filter id="gl" x="-60%" y="-60%" width="220%" height="220%"><feGaussianBlur stdDeviation="2.4" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>';
  // wires
  el("line",{x1:GEN.x+34,y1:LY,x2:SENS.x-30,y2:LY,stroke:C.wire,"stroke-width":3});
  el("line",{x1:SENS.x+30,y1:LY,x2:CTRL.x-34,y2:LY,stroke:C.wire,"stroke-width":3});
  el("line",{x1:CTRL.x+34,y1:LY,x2:CITY.x-28,y2:LY,stroke:C.wire,"stroke-width":3});
  el("path",{d:`M${CTRL.x} ${LY+32} L${CTRL.x} 208 L${GEN.x} 208 L${GEN.x} ${LY+34}`,fill:"none",stroke:"#eccbb1","stroke-width":2,"stroke-dasharray":"4 5"});
  el("text",{x:(GEN.x+CTRL.x)/2,y:225,fill:C.dim,"font-size":10,"text-anchor":"middle","font-family":"JetBrains Mono,monospace"}).textContent="setpoint P_gen → hold 50 Hz";
  el("text",{x:(SENS.x+CTRL.x)/2,y:105,fill:C.dim,"font-size":10,"text-anchor":"middle","font-family":"JetBrains Mono,monospace"}).textContent="Modbus/TCP · HR0=freq×100";
  el("line",{x1:ATK_N.x,y1:ATK_N.y+24,x2:ATK_N.x,y2:LY,stroke:"#f0b9a3","stroke-width":2,"stroke-dasharray":"3 4"});

  function label(x,y,t,s,c){el("text",{x,y,fill:c||C.ink,"font-size":11,"text-anchor":"middle","font-weight":"700","font-family":"'Space Grotesk',Inter,sans-serif"}).textContent=t;
    if(s)el("text",{x,y:y+12,fill:C.dim,"font-size":9,"text-anchor":"middle","font-family":"JetBrains Mono,monospace"}).textContent=s;}

  // generator + turbine
  el("circle",{cx:GEN.x,cy:GEN.y,r:32,fill:C.node,stroke:C.nodeStroke,"stroke-width":2});
  const turb=el("g",{id:"turb"});
  for(let i=0;i<6;i++){const a=i*Math.PI/3;el("line",{x1:GEN.x,y1:GEN.y,x2:GEN.x+24*Math.cos(a),y2:GEN.y+24*Math.sin(a),stroke:"#e08a3e","stroke-width":3,"stroke-linecap":"round"},turb);}
  el("circle",{cx:GEN.x,cy:GEN.y,r:5,fill:C.orange},turb);
  label(GEN.x,GEN.y+48,"GENERATOR","turbine + inertia");
  // sensor
  el("rect",{x:SENS.x-28,y:SENS.y-26,width:56,height:52,rx:9,fill:C.node,stroke:C.nodeStroke,"stroke-width":2});
  const led=el("circle",{cx:SENS.x,cy:SENS.y-7,r:4,fill:C.amber,id:"led"});
  el("text",{x:SENS.x,y:SENS.y+11,fill:C.dim,"font-size":9,"text-anchor":"middle","font-family":"JetBrains Mono,monospace"}).textContent="PMU";
  label(SENS.x,SENS.y+46,"SENSOR","Modbus server");
  // controller
  const cbox=el("rect",{x:CTRL.x-30,y:CTRL.y-28,width:60,height:56,rx:9,fill:C.node,stroke:C.nodeStroke,"stroke-width":2,id:"cbox"});
  el("path",{d:`M${CTRL.x-13} ${CTRL.y-5} h26 M${CTRL.x-13} ${CTRL.y+5} h26`,stroke:C.green,"stroke-width":2.5,"stroke-linecap":"round"});
  label(CTRL.x,CTRL.y+46,"CONTROLLER","PI control law");
  const sh=el("g",{id:"shield",opacity:0});
  el("path",{d:`M${CTRL.x+28} ${CTRL.y-32} l15 6 v11 c0 10 -7 17 -15 21 c-8 -4 -15 -11 -15 -21 v-11 z`,fill:"rgba(13,148,136,.15)",stroke:C.shield,"stroke-width":2},sh);
  el("path",{d:`M${CTRL.x+21} ${CTRL.y-6} l5 6 l11 -12`,fill:"none",stroke:C.shield,"stroke-width":2.5,"stroke-linecap":"round","stroke-linejoin":"round"},sh);
  // attacker
  const an=el("g",{id:"atkn",opacity:.4});
  const abox=el("rect",{x:ATK_N.x-26,y:ATK_N.y-22,width:52,height:46,rx:9,fill:"#fdecea",stroke:"#f3c1b8","stroke-width":2,id:"abox"},an);
  el("path",{d:`M${ATK_N.x} ${ATK_N.y-10} l8 15 h-16 z`,fill:"none",stroke:C.red,"stroke-width":2,"stroke-linejoin":"round"},an);
  el("text",{x:ATK_N.x,y:ATK_N.y+7,fill:C.red,"font-size":12,"text-anchor":"middle","font-weight":"700","font-family":"JetBrains Mono,monospace"},an).textContent="!";
  label(ATK_N.x,ATK_N.y-30,"ATTACKER","192.168.1.77",C.red);
  // city / load
  const city=el("g",{id:"city"});
  el("rect",{x:CITY.x-24,y:CITY.y-28,width:48,height:56,rx:7,fill:C.node,stroke:C.nodeStroke,"stroke-width":2});
  let wi=0;for(let r=0;r<4;r++)for(let c=0;c<3;c++){el("rect",{x:CITY.x-16+c*12,y:CITY.y-20+r*12,width:6,height:6,rx:1.5,fill:C.lit,class:"win",opacity:.95},city);}
  label(CITY.x,CITY.y+46,"LOAD","city / consumers");
  const pk=el("g",{id:"pk"});

  let packets=[], sparks=[], spawnT=0, genAngle=0, playing=false, last=0, stepF=0;
  const SPS=32;

  function spawn(kind){ if(kind==="attack")packets.push({kind,x:ATK_N.x,y:ATK_N.y+24,phase:"drop"}); else packets.push({kind,x:SENS.x+30,y:LY,phase:"run"}); }
  function spark(x,y){sparks.push({x,y,t:0});}
  const wins=()=>city.querySelectorAll(".win");
  const pill=document.getElementById("pill"), schSt=document.getElementById("schSt");
  function setPill(txt,fg,bg,bd){pill.textContent=txt;pill.style.color=fg;pill.style.background=bg;pill.style.borderColor=bd;}

  function render(i){
    i=Math.max(0,Math.min(i,N-1));
    const f=T.f_true[i], aa=T.attack_active[i], dis=T.action[i];
    const tripped=f>TRIP_HI||f<TRIP_LO, oob=f>HI||f<LO;
    const fq=document.getElementById("freq");
    fq.textContent=f.toFixed(2)+" Hz";
    fq.style.color = tripped?C.red:(oob?C.amber:C.orange);
    document.getElementById("clock").textContent="T+"+(i*DT).toFixed(1)+"s";
    document.getElementById("scrub").value=i;
    if(tripped){setPill("RELAY TRIP","#fff",C.red,C.red);schSt.textContent="BLACKOUT";schSt.style.color=C.red;}
    else if(oob){setPill("OUT OF BAND","#b4740c","#fff0d6","#f5d79a");schSt.textContent="UNSTABLE";schSt.style.color=C.amber;}
    else{setPill("NOMINAL",C.green,"#e9f7ef","#bfe6d1");schSt.textContent=aa?"UNDER ATTACK":"SECURE";schSt.style.color=aa?C.amber:C.green;}
    sh.setAttribute("opacity",dis?1:0);
    cbox.setAttribute("stroke",dis?C.shield:C.nodeStroke);
    an.setAttribute("opacity",aa?1:.4); abox.setAttribute("stroke",aa?C.red:"#f3c1b8");
    led.setAttribute("fill",aa?C.red:C.amber);
    wins().forEach(w=>{ if(tripped){w.setAttribute("fill",C.dark);w.setAttribute("opacity",.6);}
      else if(oob){w.setAttribute("fill",C.amber);w.setAttribute("opacity",Math.random()>.4?.95:.35);}
      else {w.setAttribute("fill",C.lit);w.setAttribute("opacity",.95);} });
    const note=document.getElementById("note");
    if(aa) note.innerHTML='<b style="color:'+C.red+'">Attack live.</b> '+(BLURB[ATK]||"")+(dis?' <b style="color:'+C.shield+'">Controller is distrusting the sensor.</b>':'');
    else if(i*DT < T.attack_start) note.innerHTML='Generator holds <b style="color:'+C.ink+'">'+f.toFixed(2)+' Hz</b>. Controller trusts the sensor over Modbus/TCP.';
    else note.innerHTML='Attack ended. Grid recovering — <b style="color:'+C.ink+'">'+f.toFixed(2)+' Hz</b>.';
  }
  function updatePackets(dt,i){
    const aa=T.attack_active[i], dis=T.action[i];
    spawnT-=dt; if(spawnT<=0){spawnT=.34;spawn("sensor");if(aa)spawn("attack");}
    const v=170;
    for(const p of packets){ if(p.phase==="drop"){p.y+=v*dt;if(p.y>=LY){p.y=LY;p.phase="run";p.x=ATK_N.x;}} else p.x+=v*dt;
      if(p.x>=CTRL.x-34){ if(p.kind==="attack"&&dis){p.dead=true;spark(p.x,LY);} else p.dead=true; } }
    packets=packets.filter(p=>!p.dead&&p.x<CTRL.x+10);
    const f=T.f_true[i]; genAngle+=(.6+(f-NOM)*2.2)*dt*60; turb.setAttribute("transform",`rotate(${genAngle} ${GEN.x} ${GEN.y})`);
    pk.innerHTML="";
    for(const p of packets){const col=p.kind==="attack"?C.red:(aa&&p.kind==="sensor"?C.amber:C.green);
      const c=document.createElementNS(SVGNS,"circle");c.setAttribute("cx",p.x);c.setAttribute("cy",p.y);c.setAttribute("r",p.kind==="attack"?4.8:3.6);c.setAttribute("fill",col);c.setAttribute("filter","url(#gl)");pk.appendChild(c);}
    sparks.forEach(s=>s.t+=dt); sparks=sparks.filter(s=>s.t<.35);
    for(const s of sparks){const r=4+s.t*40,o=1-s.t/.35;const c=document.createElementNS(SVGNS,"circle");c.setAttribute("cx",s.x);c.setAttribute("cy",s.y);c.setAttribute("r",r);c.setAttribute("fill","none");c.setAttribute("stroke",C.shield);c.setAttribute("stroke-width",2);c.setAttribute("opacity",o.toFixed(2));pk.appendChild(c);}
  }
  function setPlay(p){playing=p;document.getElementById("btnPlay").innerHTML=p?"&#10073;&#10073; Pause":"&#9654; Play";}
  document.getElementById("btnPlay").onclick=()=>{ if(stepF>=N-1)stepF=0; setPlay(!playing); };
  document.getElementById("scrub").oninput=e=>{setPlay(false);stepF=+e.target.value;render(Math.floor(stepF));};
  document.getElementById("scrub").max=N-1;
  function loop(ts){ if(!last)last=ts; const dt=Math.min(.05,(ts-last)/1000); last=ts; const i=Math.floor(stepF);
    if(playing){ stepF+=SPS*dt; if(stepF>=N-1){stepF=N-1;setPlay(false);} render(Math.floor(stepF)); }
    updatePackets(dt,Math.min(i,N-1)); requestAnimationFrame(loop); }
  render(0); requestAnimationFrame(loop);
  setTimeout(()=>{setPlay(true);},400);
})();
</script>
"""


def grid_view_html(traj: dict, attack_label: str, defense_label: str) -> str:
    scn = f"{attack_label} &rarr; {defense_label}"
    return (_TEMPLATE
            .replace("__TRAJ__", _traj_to_json(traj))
            .replace("__ATK__", attack_label)
            .replace("__DEF__", defense_label)
            .replace("__SCN__", scn))


def render_grid_view(traj: dict, attack_label: str, defense_label: str, height: int = 400):
    import streamlit.components.v1 as components
    components.html(grid_view_html(traj, attack_label, defense_label), height=height, scrolling=False)
