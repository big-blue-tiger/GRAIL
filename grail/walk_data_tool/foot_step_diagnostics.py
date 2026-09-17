"""CPU-only diagnostic plots and self-contained, offline skeleton replay."""

from __future__ import annotations

import html
import json
import textwrap

import numpy as np


def write_plot(path, name, feet, filtered, speeds, fps, selection, error):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    path.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(12, 9), layout="constrained")
    FigureCanvasAgg(figure)
    frame = selection.get("source_frame") if selection else None
    status = error or f"source frame {frame}; {selection['reason']}"
    figure.suptitle(f"{name}\n" + textwrap.fill(status, 110), fontsize=11)
    if feet is None or filtered is None or speeds is None:
        axis = figure.subplots()
        axis.text(0.5, 0.5, "No valid foot trajectory\n" + textwrap.fill(error or "Unknown error", 90),
                  ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
    else:
        axes = figure.subplots(4, 1, sharex=True)
        frames = np.arange(len(feet))
        colors = ("#2563eb", "#60a5fa", "#dc2626", "#fb923c")
        labels = ("L ankle", "L toe", "R ankle", "R toe")
        for coord, axis in enumerate(axes[:3]):
            for point, (color, label) in enumerate(zip(colors, labels)):
                axis.plot(frames, feet[:, point, coord], color=color, alpha=0.25, linewidth=1)
                axis.plot(frames, filtered[:, point, coord], color=color, label=label, linewidth=1)
            axis.set_ylabel(f"world {'XYZ'[coord]} (m)")
        axes[0].legend(ncol=4, fontsize=8, loc="upper right")
        raw_speed = np.zeros(feet.shape[:2])
        raw_speed[1:] = np.linalg.norm(np.diff(feet, axis=0), axis=-1) * fps
        raw_speed = raw_speed.reshape(len(feet), 2, 2).max(axis=2)
        for foot, color in enumerate((colors[0], colors[2])):
            axes[3].plot(frames, raw_speed[:, foot], color=color, alpha=0.25, linewidth=1)
            axes[3].plot(frames, speeds[:, foot], color=color, label=("left", "right")[foot])
        params = selection["parameters"]
        axes[3].axhline(params["speed_on"], color="black", linestyle="--", label="on")
        axes[3].axhline(params["speed_off"], color="gray", linestyle=":", label="stable")
        axes[3].set_ylabel("speed (m/s)")
        # Preserve visibility near micro-step thresholds even alongside a fast stride.
        axes[3].set_yscale("symlog", linthresh=params["speed_on"])
        axes[3].set_ylim(bottom=0)
        axes[3].set_xlabel(f"Original frame (zero-based), {fps:g} FPS; faint=raw, solid=filtered")
        axes[3].legend(ncol=4, fontsize=8)
        for axis in axes:
            axis.grid(alpha=0.2)
            for episode in selection["episodes"]:
                axis.axvspan(episode["start_frame"], episode["end_frame"],
                             color=colors[0 if episode["foot"] == "left" else 2], alpha=0.09)
            if frame is not None:
                axis.axvline(frame, color="#15803d", linewidth=2)
                start = selection["stable_window_start"]
                axis.axvspan(start, start + selection["stable_intervals"], color="#15803d", alpha=0.12)
    figure.savefig(path, dpi=120)
    figure.clear()


def write_replay(path, name, positions, fps, skeleton, selection, error):
    """Two orthographic views with a frame slider; no CDN, codecs, or simulator."""
    selected = selection.get("source_frame") if selection else None
    center = selected if selected is not None else len(positions) - 1
    radius = max(1, int(np.ceil(fps)))
    start, stop = max(0, center - radius), min(len(positions), center + radius + 1)
    names = skeleton.bone_order_names
    edges = [[names.index(parent), i] for i, name in enumerate(names)
             if (parent := skeleton.bone_parents[name]) is not None]
    data = {
        "positions": np.round(positions[start:stop], 6).tolist(), "edges": edges,
        "feet": skeleton.foot_joint_idx, "start": start, "fps": fps,
        "selected": selected, "initial": center - start,
    }
    title = html.escape(name)
    status = html.escape(error or f"Selected source frame: {selected}")
    document = REPLAY_HTML.replace("__TITLE__", title).replace("__STATUS__", status)
    document = document.replace("__DATA__", json.dumps(data, allow_nan=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


REPLAY_HTML = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>__TITLE__ — foot-step review</title>
<style>
body{font:16px system-ui;margin:24px auto;max-width:1000px;padding:0 16px;color:#182335}
canvas{width:100%;border:1px solid #ccd5e2;border-radius:8px;background:#f8fafc}
input[type=range]{width:65%;vertical-align:middle}button{padding:8px 16px;margin:8px}p{line-height:1.5}
</style>
<h2>__TITLE__</h2><p>__STATUS__</p>
<canvas id="view" width="1000" height="570"></canvas>
<div><button id="play">Play</button><button id="selected">Selection / end</button>
<input id="frame" type="range" min="0" step="1" aria-label="Source frame"></div>
<label><input id="feetOnly" type="checkbox"> Foot detail</label>
<p id="label"></p><p>Left foot: blue; right foot: orange. Green border marks the selected frame.
Views use world coordinates (metres). Original, unfiltered skeleton; no object or contact simulation.</p>
<script>
const data = __DATA__;
const canvas=document.getElementById('view'),ctx=canvas.getContext('2d');
const slider=document.getElementById('frame'),button=document.getElementById('play');
const feetOnly=document.getElementById('feetOnly');
slider.max=data.positions.length-1;slider.value=data.initial;
function draw(){
 const visible=feetOnly.checked?data.positions.flatMap(p=>data.feet.map(j=>p[j])):data.positions.flat();
 const bounds=[0,1,2].map(d=>{
  const v=visible.map(p=>p[d]);return [Math.min(...v),Math.max(...v)];});
 const scale=420/Math.max(...bounds.map(b=>b[1]-b[0]),0.08);
 const i=Number(slider.value),points=data.positions[i],original=data.start+i;
 ctx.clearRect(0,0,canvas.width,canvas.height);
 ctx.font='17px system-ui';和抓取后的脚步
 [[0,'X / Z side view'],[1,'Y / Z front view']].forEach(([horizontal,title],panel)=>{
  const center=(bounds[horizontal][0]+bounds[horizontal][1])/2;
  const project=p=>[panel*500+250+(p[horizontal]-center)*scale,
                    505-(p[2]-bounds[2][0])*scale];
  ctx.fillStyle='#182335';ctx.fillText(title,panel*500+20,30);
  const ground=505+bounds[2][0]*scale;
  ctx.strokeStyle='#cbd5e1';ctx.lineWidth=1;ctx.beginPath();
  ctx.moveTo(panel*500+10,ground);ctx.lineTo(panel*500+490,ground);ctx.stroke();
  // Raw foot paths make small relocations visible even when the body sways.
  data.feet.forEach((joint,f)=>{
   ctx.strokeStyle=f<2?'#93c5fd':'#fdba74';ctx.lineWidth=1;ctx.beginPath();
   data.positions.forEach((pose,t)=>{const p=project(pose[joint]);
     if(t===0)ctx.moveTo(...p);else ctx.lineTo(...p);});ctx.stroke();
  });
  ctx.strokeStyle='#64748b';ctx.lineWidth=3;
  data.edges.forEach(([a,b])=>{ctx.beginPath();ctx.moveTo(...project(points[a]));
                            ctx.lineTo(...project(points[b]));ctx.stroke();});
  points.forEach((point,j)=>{
   const f=data.feet.indexOf(j);ctx.fillStyle=f<0?'#475569':f<2?'#2563eb':'#ea580c';
   ctx.beginPath();ctx.arc(...project(point),f<0?3:6,0,2*Math.PI);ctx.fill();
  });
 });
 canvas.style.borderColor=original===data.selected?'#15803d':'#ccd5e2';
 document.getElementById('label').textContent='Original frame '+original+' | '+
  (original/data.fps).toFixed(3)+' s | '+data.fps+' FPS'+
  (original===data.selected?' | SELECTED':'');
}
let timer=null;
button.onclick=()=>{if(timer){clearInterval(timer);timer=null;button.textContent='Play';}
 else{button.textContent='Pause';timer=setInterval(()=>{
 slider.value=(Number(slider.value)+1)%data.positions.length;draw();},1000/data.fps);}};
slider.oninput=draw;
feetOnly.onchange=draw;
document.getElementById('selected').onclick=()=>{slider.value=data.initial;draw();};
draw();
</script></html>
"""
