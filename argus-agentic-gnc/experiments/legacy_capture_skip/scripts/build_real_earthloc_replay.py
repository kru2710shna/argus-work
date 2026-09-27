"""Build a standalone interactive replay for a real EarthLoc agent episode."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
from pathlib import Path


def footprint_from_path(path: str) -> list[dict[str, float]]:
    """Read the four latitude/longitude corners encoded in EarthLoc filenames."""
    values = Path(path).name.split("@")[1:9]
    numbers = [float(value) for value in values]
    return [
        {"lat": numbers[index], "lon": numbers[index + 1]}
        for index in range(0, 8, 2)
    ]


def image_data_url(path: Path) -> str:
    media_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("argus-agentic-gnc/outputs/real_earthloc_agent_history.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("argus-agentic-gnc/outputs/real_earthloc_replay.html"),
    )
    args = parser.parse_args()

    history = json.loads(args.history.read_text())
    image_paths = {Path(frame["query_path"]) for frame in history}
    image_paths.update(
        Path(tile)
        for frame in history
        for tile in frame.get("top_paths", [])
    )

    missing = [path for path in image_paths if not (args.data_root / path).exists()]
    if missing:
        raise FileNotFoundError(f"Cannot build replay; missing image: {missing[0]}")

    assets = {str(path): image_data_url(args.data_root / path) for path in sorted(image_paths)}
    for frame in history:
        frame["query_footprint"] = footprint_from_path(frame["query_path"])
        frame["top_footprints"] = [footprint_from_path(path) for path in frame.get("top_paths", [])]
        frame["query_id"] = Path(frame["query_path"]).name.split("@")[9]
        frame["top_ids"] = [Path(path).name.split("@")[9] for path in frame.get("top_paths", [])]

    payload = json.dumps({"frames": history, "assets": assets}, separators=(",", ":"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Real EarthLoc Agent Replay</title>
  <style>
    :root { color-scheme: dark; --bg:#0b1020; --panel:#121a2d; --line:#2b3959; --text:#e8edf8; --muted:#aab7d1; --capture:#28d17c; --skip:#ff8768; --cyan:#54d9f7; --orange:#ffbf5b; }
    * { box-sizing:border-box; } body { margin:0; background:var(--bg); color:var(--text); font:15px/1.45 system-ui,-apple-system,sans-serif; }
    main { max-width:1500px; margin:auto; padding:20px; } h1,h2,p { margin:0; } h1 { font-size:24px; } h2 { font-size:16px; margin-bottom:10px; } .subtitle { color:var(--muted); margin-top:4px; }
    .controls { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin:16px 0; } button { color:var(--text); background:#1b2843; border:1px solid var(--line); border-radius:7px; padding:8px 12px; cursor:pointer; } button:hover,button.active { background:#2c426b; } input[type=range] { flex:1; min-width:200px; accent-color:var(--cyan); }
    .grid { display:grid; grid-template-columns:1.15fr 1.15fr .9fr; gap:14px; } .panel { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px; min-width:0; }
    .image { width:100%; height:300px; object-fit:contain; background:#050912; border-radius:6px; } .meta { color:var(--muted); font-size:13px; margin-top:7px; word-break:break-word; }
    .result-grid { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:6px; } .result-grid img { width:100%; aspect-ratio:1; object-fit:cover; border:2px solid transparent; border-radius:5px; } .result-grid img.positive { border-color:var(--capture); } .tile-label { font-size:11px; color:var(--muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .stats { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:12px; } .stat { padding:10px; background:#0d1425; border-radius:7px; } .stat strong { display:block; font-size:21px; } .capture { color:var(--capture); } .skip { color:var(--skip); }
    #map { width:100%; height:270px; background:#0b1325; border-radius:6px; } .legend { color:var(--muted); font-size:12px; margin-top:6px; }
    .timeline { display:flex; gap:5px; flex-wrap:wrap; margin-top:14px; } .frame { width:40px; height:38px; padding:0; font-weight:700; } .frame.capture { border-color:var(--capture); } .frame.skip { border-color:var(--skip); }
    @media (max-width:1000px) { .grid { grid-template-columns:1fr 1fr; } .grid .panel:last-child { grid-column:span 2; } } @media (max-width:650px) { main{padding:12px}.grid{grid-template-columns:1fr}.grid .panel:last-child{grid-column:auto}.image{height:250px}.result-grid{grid-template-columns:repeat(3,1fr)} }
  </style>
</head>
<body><main>
  <h1>Real EarthLoc Agent Replay</h1>
  <p class="subtitle">ISS062-E · 12 real query images · EarthLoc retrieval · capture/skip policy</p>
  <div class="controls"><button id="back">◀ Previous</button><button id="play">▶ Play</button><button id="next">Next ▶</button><label>Speed <select id="speed"><option value="1500">1×</option><option value="800">2×</option><option value="400">4×</option></select></label><input id="scrubber" type="range" min="0" max="11" value="0"><span id="counter"></span></div>
  <section class="grid">
    <article class="panel"><h2>Actual ISS query image</h2><img class="image" id="queryImage" alt="Current real ISS query image"><p class="meta" id="queryMeta"></p></article>
    <article class="panel"><h2 id="retrievalTitle">EarthLoc retrieval</h2><div id="retrieval" class="result-grid"></div><p class="meta" id="retrievalMeta"></p></article>
    <article class="panel"><h2>Agent and localization feedback</h2><div class="stats"><div class="stat"><span>Action</span><strong id="action"></strong></div><div class="stat"><span>Images used</span><strong id="images"></strong></div><div class="stat"><span>Known overlap rank</span><strong id="rank"></strong></div><div class="stat"><span>Reward</span><strong id="reward"></strong></div></div><svg id="map" viewBox="0 0 600 270" role="img" aria-label="Geographic footprints for the ISS query and selected reference tile"></svg><p class="legend">cyan: query footprint · orange: retrieved top tile · dots: captured frames</p></article>
  </section>
  <div class="timeline" id="timeline" aria-label="Replay frame selector"></div>
</main><script>
const replay = """ + payload + """;
const {frames, assets} = replay; let index = 0, timer = null;
const $ = id => document.getElementById(id);
const allCorners = frames.flatMap(f => f.query_footprint); const lons=allCorners.map(p=>p.lon), lats=allCorners.map(p=>p.lat);
const lonMin=Math.min(...lons)-.35, lonMax=Math.max(...lons)+.35, latMin=Math.min(...lats)-.35, latMax=Math.max(...lats)+.35;
function point(p){ const x=30+(p.lon-lonMin)/(lonMax-lonMin)*540; const y=240-(p.lat-latMin)/(latMax-latMin)*210; return `${x.toFixed(1)},${y.toFixed(1)}`; }
function polygon(points){ return points.map(point).join(' '); }
function renderMap(frame){ const captured=frames.slice(0,index+1).filter(f=>f.action==='CAPTURE').map(f=>f.query_footprint[0]); const tile=frame.top_footprints?.[0]; $('map').innerHTML=`<rect x="30" y="30" width="540" height="210" fill="#0b1325" stroke="#2b3959"/><path d="M30 135H570 M300 30V240" stroke="#20304d"/><polygon points="${polygon(frame.query_footprint)}" fill="#54d9f733" stroke="#54d9f7" stroke-width="3"/>${tile?`<polygon points="${polygon(tile)}" fill="#ffbf5b33" stroke="#ffbf5b" stroke-width="3"/>`:''}${captured.map(p=>`<circle cx="${point(p).split(',')[0]}" cy="${point(p).split(',')[1]}" r="4" fill="#28d17c"/>`).join('')}<text x="34" y="22" fill="#aab7d1" font-size="12">${latMax.toFixed(1)}°N</text><text x="34" y="258" fill="#aab7d1" font-size="12">${latMin.toFixed(1)}°N</text>`; }
function render(){ const f=frames[index], capture=f.action==='CAPTURE'; $('scrubber').value=index; $('counter').textContent=`Frame ${index+1} / ${frames.length}`; $('queryImage').src=assets[f.query_path]; $('queryMeta').textContent=f.query_id+' · '+(capture?'image captured and sent to EarthLoc':'agent skipped retrieval for this frame'); $('action').textContent=f.action; $('action').className=capture?'capture':'skip'; $('images').textContent=f.images_taken; $('rank').textContent=capture?(f.positive_rank ? '#'+f.positive_rank : 'not in top-5'):'—'; $('reward').textContent=Number(f.reward).toFixed(2); $('retrievalTitle').textContent=capture?'Top-5 retrieved real map tiles':'EarthLoc retrieval skipped'; $('retrieval').innerHTML=capture?f.top_paths.map((path,i)=>`<div><img class="${f.positive_rank===i+1?'positive':''}" src="${assets[path]}" alt="Retrieved tile ${i+1}"><div class="tile-label">#${i+1} ${f.top_ids[i]}</div><div class="tile-label">${f.top_scores[i].toFixed(3)}</div></div>`).join(''):'<p class="meta">No candidate tiles were computed; the agent chose SKIP to save image/compute budget.</p>'; $('retrievalMeta').textContent=capture?`Green border = a known overlapping tile. Top score: ${f.top_scores[0].toFixed(3)}.`:'The query image is still shown at left, but no EarthLoc inference was performed.'; renderMap(f); document.querySelectorAll('.frame').forEach((button,i)=>button.classList.toggle('active',i===index)); }
function setIndex(value){ index=(value+frames.length)%frames.length; render(); }
frames.forEach((f,i)=>{const b=document.createElement('button');b.className='frame '+(f.action==='CAPTURE'?'capture':'skip');b.textContent=i+1;b.title=`Frame ${i+1}: ${f.action}`;b.onclick=()=>setIndex(i);$('timeline').appendChild(b);});
function stop(){ if(timer){clearInterval(timer);timer=null;$('play').textContent='▶ Play';} } function play(){ if(timer){stop();return;} $('play').textContent='❚❚ Pause'; timer=setInterval(()=>setIndex(index+1),Number($('speed').value)); }
$('back').onclick=()=>{stop();setIndex(index-1)}; $('next').onclick=()=>{stop();setIndex(index+1)}; $('play').onclick=play; $('scrubber').oninput=e=>{stop();setIndex(Number(e.target.value))}; $('speed').onchange=()=>{if(timer){stop();play();}}; render();
</script></body></html>"""
    )
    print(f"Created interactive replay: {args.output}")
    print(f"Embedded {len(assets)} real image assets across {len(history)} frames.")


if __name__ == "__main__":
    main()
