import json
from pathlib import Path

import numpy as np
import pandas as pd

import argparse

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser(description="Interactive player-movement map from a LacrosseNet trajectories.csv")
ap.add_argument("--trajectories", default=str(ROOT / "results/laxnet_hfx_at_ga_2026-05-02/trajectories.csv"))
ap.add_argument("--out", default=str(ROOT / "analysis_visuals/player_movement_map.html"))
args = ap.parse_args()
SOURCE = Path(args.trajectories)
OUT = Path(args.out)

if not SOURCE.exists():
    raise SystemExit(f"Source file not found: {SOURCE}")

df = pd.read_csv(SOURCE)
required = ["t", "x", "y", "shot_id", "track_id", "team",
            "measured", "outlier", "pos_std", "speed"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise SystemExit(f"Missing required columns: {missing}")

# Keep only usable coordinates; do not modify the source CSV.
df = df.dropna(subset=["t", "x", "y", "shot_id", "track_id"]).copy()
df["shot_id"] = df["shot_id"].astype(int)
df["track_id"] = df["track_id"].astype(str)
df["team"] = df["team"].fillna("unknown").astype(str)
df["measured"] = df["measured"].fillna(False).astype(bool)
df["outlier"] = df["outlier"].fillna(False).astype(bool)

optional = ["number", "name"]
for c in optional:
    if c not in df.columns:
        df[c] = None

cols = ["t", "x", "y", "shot_id", "track_id", "team",
        "measured", "outlier", "pos_std", "speed", "number", "name"]
data = df[cols].replace({np.nan: None}).to_dict("records")
payload = json.dumps(data, allow_nan=False, separators=(",", ":"))

html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LacrosseNet | Player Movement Explorer</title>
<style>
:root{color-scheme:dark;--bg:#11151c;--panel:#1c2430;--text:#edf2f7;--muted:#a9b6c7}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:15px system-ui,-apple-system,sans-serif}
main{max-width:1250px;margin:auto;padding:22px}
h1{font-size:25px;margin:0 0 5px}
.sub{color:var(--muted);margin-bottom:18px}
.controls,.stats{display:flex;flex-wrap:wrap;gap:12px;align-items:center}
.controls{background:var(--panel);padding:14px;border-radius:12px;margin-bottom:12px}
label{display:flex;gap:7px;align-items:center}
select,input,button{accent-color:#b5c7e5}
select,button{background:#2a3545;color:white;border:1px solid #536176;border-radius:7px;padding:8px}
input[type=range]{min-width:160px}
.stats{margin:12px 0}
.stat{background:var(--panel);border-radius:9px;padding:10px 14px;min-width:125px}
.stat small{display:block;color:var(--muted)}
.stat strong{font-size:19px}
.canvaswrap{background:#18212a;border-radius:12px;padding:10px}
canvas{display:block;width:100%;height:auto}
.legend{display:flex;flex-wrap:wrap;gap:16px;color:var(--muted);font-size:13px;padding:12px 4px}
.dot{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:5px}
.note{color:var(--muted);font-size:13px;line-height:1.5;margin-top:12px}
#timeLabel{font-variant-numeric:tabular-nums;min-width:85px}
</style>
</head>
<body>
<main>
<h1>LacrosseNet — Player Movement Explorer</h1>
<div class="sub">Original game run · floor coordinates in feet · read-only source</div>

<div class="controls">
<label>Shot <select id="shot"></select></label>
<label>Time <input id="time" type="range" min="0" max="1" value="0"></label>
<strong id="timeLabel">—</strong>
<label>Trail <select id="trail">
<option value="1">1 second</option><option value="2" selected>2 seconds</option>
<option value="4">4 seconds</option><option value="8">8 seconds</option>
</select></label>
<label><input id="ids" type="checkbox"> Track IDs</label>
<label><input id="officials" type="checkbox"> Officials</label>
<label><input id="bad" type="checkbox" checked> Show outliers</label>
<button id="play">▶ Play</button>
</div>

<div class="stats">
<div class="stat"><small>Shot</small><strong id="sShot">—</strong></div>
<div class="stat"><small>Players on frame</small><strong id="sPlayers">—</strong></div>
<div class="stat"><small>Measured points</small><strong id="sMeasured">—</strong></div>
<div class="stat"><small>Predicted / interpolated</small><strong id="sPredicted">—</strong></div>
<div class="stat"><small>Outlier flags</small><strong id="sOutliers">—</strong></div>
</div>

<div class="canvaswrap"><canvas id="field" width="1200" height="610"></canvas></div>
<div class="legend">
<span><i class="dot" style="background:#9b4dff"></i>HFX</span>
<span><i class="dot" style="background:#e0a100"></i>GA</span>
<span><i class="dot" style="background:#b7c3ce"></i>Official</span>
<span>● Filled = measured</span><span>○ Hollow = predicted/interpolated</span>
<span style="color:#ff5e68">× Outlier flag</span>
</div>
<div class="note">
Field dimensions and markings come from the upstream configuration (200 × 85 ft,
28 ft corner radius). Trails show recent recorded trajectory points. Hollow markers
are rows marked unmeasured by the pipeline; an outlier is a model flag, not proof
that a position is wrong. This is a tracking diagnostic, not yet a validated tactical
measurement system. Player locations outside the field drawing can indicate calibration
or tracking issues.
</div>
</main>
<script>
const rows = DATA_PLACEHOLDER;
const colors = {HFX:"#9b4dff",GA:"#e0a100",official:"#b7c3ce"};
const $ = id => document.getElementById(id);
const shots = [...new Set(rows.map(r=>r.shot_id))].sort((a,b)=>a-b);
for (const s of shots) {
  const o=document.createElement("option"); o.value=s; o.textContent="Shot "+s; $("shot").appendChild(o);
}
const canvas=$("field"), ctx=canvas.getContext("2d");
let times=[], current=0, playing=false, timer=null;

function shotRows(){return rows.filter(r=>r.shot_id===Number($("shot").value));}
function setupShot(){
  const rs=shotRows();
  times=[...new Set(rs.map(r=>r.t))].sort((a,b)=>a-b);
  $("time").min=0; $("time").max=Math.max(0,times.length-1);
  $("time").value=0; current=0; draw();
}
function roundedRect(x,y,w,h,r){
  ctx.beginPath();ctx.moveTo(x+r,y);ctx.lineTo(x+w-r,y);
  ctx.quadraticCurveTo(x+w,y,x+w,y+r);ctx.lineTo(x+w,y+h-r);
  ctx.quadraticCurveTo(x+w,y+h,x+w-r,y+h);ctx.lineTo(x+r,y+h);
  ctx.quadraticCurveTo(x,y+h,x,y+h-r);ctx.lineTo(x,y+r);
  ctx.quadraticCurveTo(x,y,x+r,y);ctx.closePath();
}
function draw(){
  const rs=shotRows();
  if(!times.length)return;
  current=Math.max(0,Math.min(Number($("time").value),times.length-1));
  const t=times[current];
  $("timeLabel").textContent=t.toFixed(2)+" s";
  $("sShot").textContent=$("shot").value;
  const W=canvas.width,H=canvas.height;
  ctx.clearRect(0,0,W,H);
  const scale=Math.min((W-100)/214,(H-70)/99);
  const cx=W/2,cy=H/2;
  const X=x=>cx+x*scale, Y=y=>cy-y*scale;
  const L=200,Wf=85,R=28;
  ctx.fillStyle="#1c302d";ctx.fillRect(0,0,W,H);
  ctx.fillStyle="#205a3a";
  roundedRect(X(-100),Y(42.5),L*scale,Wf*scale,R*scale);ctx.fill();
  ctx.strokeStyle="#d5c58b";ctx.lineWidth=3;
  roundedRect(X(-100),Y(42.5),L*scale,Wf*scale,R*scale);ctx.stroke();

  // Field lines from the configured floor model.
  ctx.strokeStyle="rgba(245,248,238,.76)";ctx.lineWidth=1.8;
  for(const x of [-42.5,0,42.5]){
    ctx.beginPath();ctx.moveTo(X(x),Y(-42.5));ctx.lineTo(X(x),Y(42.5));ctx.stroke();
  }
  ctx.beginPath();ctx.arc(X(0),Y(0),11*scale,0,2*Math.PI);ctx.stroke();
  const goalX=100-12, crease=9.25;
  for(const x of [-goalX,goalX]){
    ctx.beginPath();ctx.arc(X(x),Y(0),crease*scale,0,2*Math.PI);ctx.stroke();
    ctx.strokeStyle="#ef6666";ctx.lineWidth=3;
    ctx.beginPath();ctx.moveTo(X(x),Y(-2.4));ctx.lineTo(X(x),Y(2.4));ctx.stroke();
    ctx.strokeStyle="rgba(245,248,238,.76)";ctx.lineWidth=1.8;
  }
  ctx.fillStyle="#dce7de";ctx.font="13px system-ui";
  ctx.textAlign="center";
  ctx.fillText("LEFT END  (−X)",X(-72),Y(0)-15);
  ctx.fillText("RIGHT END  (+X)",X(72),Y(0)-15);
  ctx.fillText("CENTER",X(0),Y(0)-17);
  ctx.fillStyle="#c2d0c7";ctx.font="12px system-ui";ctx.textAlign="left";
  ctx.fillText("Y+ toward far boards",18,25);
  ctx.fillText("Floor coordinates (ft)",18,H-15);

  const visible=rs.filter(r=>$("officials").checked || r.team!=="official");
  const lo=t-Number($("trail").value);
  const groups={};
  for(const r of visible){
    if(!groups[r.track_id])groups[r.track_id]=[];
    groups[r.track_id].push(r);
  }
  // Draw recent trails, with uncertain observations visually subdued.
  for(const [id,g] of Object.entries(groups)){
    const trail=g.filter(r=>r.t>=lo && r.t<=t).sort((a,b)=>a.t-b.t);
    if(trail.length<2)continue;
    ctx.beginPath();
    trail.forEach((r,i)=>i?ctx.lineTo(X(r.x),Y(r.y)):ctx.moveTo(X(r.x),Y(r.y)));
    ctx.strokeStyle=colors[g[0].team]||"#f0f0f0";
    ctx.globalAlpha=.43;ctx.lineWidth=2;ctx.stroke();ctx.globalAlpha=1;
  }

  const now=visible.filter(r=>r.t===t);
  let measured=0,predicted=0,outliers=0;
  for(const r of now){
    const isOfficial=r.team==="official";
    if(!r.measured)predicted++;
    else measured++;
    if(r.outlier)outliers++;
    if(r.outlier && !$("bad").checked)continue;
    const x=X(r.x),y=Y(r.y),c=colors[r.team]||"#f0f0f0";
    if(r.outlier){
      ctx.strokeStyle="#ff5e68";ctx.lineWidth=3;
      ctx.beginPath();ctx.moveTo(x-6,y-6);ctx.lineTo(x+6,y+6);
      ctx.moveTo(x+6,y-6);ctx.lineTo(x-6,y+6);ctx.stroke();
    }else{
      ctx.beginPath();ctx.arc(x,y,isOfficial?4.5:6,0,2*Math.PI);
      ctx.lineWidth=2;ctx.strokeStyle=c;
      if(r.measured){ctx.fillStyle=c;ctx.fill();}else{ctx.fillStyle="#205a3a";ctx.fill();}
      ctx.stroke();
    }
    if($("ids").checked){
      ctx.font="12px system-ui";ctx.textAlign="left";
      ctx.fillStyle="#fff";ctx.fillText(r.track_id,x+8,y-8);
    }
  }
  $("sPlayers").textContent=now.filter(r=>r.team!=="official").length;
  $("sMeasured").textContent=measured;
  $("sPredicted").textContent=predicted;
  $("sOutliers").textContent=outliers;
}
$("shot").addEventListener("change",setupShot);
$("time").addEventListener("input",draw);
for(const id of ["trail","ids","officials","bad"])$(id).addEventListener("change",draw);
$("play").addEventListener("click",()=>{
  if(playing){clearInterval(timer);playing=false;$("play").textContent="▶ Play";return;}
  playing=true;$("play").textContent="Ⅱ Pause";
  timer=setInterval(()=>{
    if(current>=times.length-1){clearInterval(timer);playing=false;$("play").textContent="▶ Play";return;}
    $("time").value=++current;draw();
  },100);
});
setupShot();
</script>
</body>
</html>
"""

html = html.replace("DATA_PLACEHOLDER", payload)
OUT.write_text(html, encoding="utf-8")

print(f"Created: {OUT}")
print(f"Source rows used: {len(df):,}")
print("Shots:", ", ".join(map(str, sorted(df["shot_id"].unique()))))
print(f"HTML size: {OUT.stat().st_size:,} bytes")
print("Source CSV was not modified.")
