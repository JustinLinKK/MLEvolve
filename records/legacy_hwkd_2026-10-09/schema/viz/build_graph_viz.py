"""Render the hardware-feature GraphDB (hardware_knowledge_graph.json) as an
interactive HTML graph. Reads the built GraphDB instance data, NOT the vector
schemas, so it keeps working after the vector schemas are removed.

  Hardware (blue box) -[HAS_FEATURE]-> Feature (green ellipse)
  Feature -[REQUIRES_FEATURE]-> Feature

Clicking a node shows its properties. API symbols and one example_code are
already merged onto each Feature node by build_graph_db.py.

Output: schema/graph_schema_viz.html (self-contained, vis-network via CDN).
"""
from __future__ import annotations
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
GRAPH = HERE.parent / "hardware_knowledge_graph.json"  # data lives in schema/
OUT = HERE / "graph_schema_viz.html"

CAT_COLOR = {
    "precision": "#7fd3a0", "tensor_core": "#5bbf8a",
    "compute_capability": "#9bd6b4", "interconnect": "#c8e6d3",
    "kernel_optimization": "#b6dfc6", "memory_optimization": "#8fcfae",
    "data_pipeline": "#6fbf95", "parallelism": "#52a87a",
    "optimizer": "#9ad6b8", "attention": "#7ec9a0", "other": "#a8cfb8",
}
HARDWARE_COLOR = "#4f9eed"


def main() -> int:
    g = json.loads(GRAPH.read_text())
    # features each hardware has (derived from HAS_FEATURE edges)
    hw_feats: dict[str, list[str]] = {}
    for e in g["edges"]:
        if e["type"] == "HAS_FEATURE":
            hw_feats.setdefault(e["from"], []).append(e["to"].split(":", 1)[1])

    nodes = []
    for n in g["nodes"]:
        p = n["properties"]
        if n["label"] == "Feature":
            nodes.append({
                "kind": "feature", "id": n["id"], "label": p["feature_id"],
                "color": CAT_COLOR.get(p.get("category", "other"), "#a8cfb8"),
                "category": p.get("category", ""),
                "description": p.get("description", ""),
                "url": p.get("source_url", ""),
                "api_symbols": p.get("api_symbols", []),
                "example_code": p.get("example_code", ""),
                "example_code_source": p.get("example_code_source", ""),
            })
        else:
            nodes.append({
                "kind": "hardware", "id": n["id"], "label": p.get("name", n["id"]),
                "color": HARDWARE_COLOR,
                "hardware_id": p.get("hardware_id", ""),
                "architectures": p.get("architectures", []),
                "compute_capabilities": p.get("compute_capabilities", []),
                "vram_MB": p.get("vram_MB"), "vram_type": p.get("vram_type", ""),
                "sm_count": p.get("sm_count"),
                "features": sorted(hw_feats.get(n["id"], [])),
            })

    edges = [{"from": e["from"], "to": e["to"], "type": e["type"]}
             for e in g["edges"]]
    data = {"nodes": nodes, "edges": edges}
    OUT.write_text(TEMPLATE.replace("__DATA__", json.dumps(data)))
    n_hw = sum(1 for n in nodes if n["kind"] == "hardware")
    n_ft = sum(1 for n in nodes if n["kind"] == "feature")
    n_hf = sum(1 for e in edges if e["type"] == "HAS_FEATURE")
    n_rq = sum(1 for e in edges if e["type"] == "REQUIRES_FEATURE")
    print(f"wrote {OUT} ({n_hw} hardware, {n_ft} features, "
          f"{n_hf} HAS_FEATURE, {n_rq} REQUIRES_FEATURE)")
    return 0


TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Hardware-Feature Knowledge Graph</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
  html,body{margin:0;height:100%;font-family:system-ui,sans-serif;background:#1b1d23;color:#e6e6e6}
  #wrap{display:flex;height:100vh}
  #net{flex:1;height:100%}
  #panel{width:360px;overflow-y:auto;padding:16px;background:#23262e;border-left:1px solid #333}
  h1{font-size:15px;margin:0 0 4px}
  h2{font-size:14px;margin:12px 0 6px;color:#9ad}
  .sub{font-size:11px;color:#8b93a1;margin-bottom:10px}
  table{width:100%;border-collapse:collapse;font-size:12px}
  td{padding:3px 6px;border-bottom:1px solid #30343d;vertical-align:top}
  .desc{font-size:12px;line-height:1.5;color:#cfd3da;margin:6px 0}
  .pill{display:inline-block;padding:1px 7px;border-radius:9px;font-size:11px;margin-right:6px}
  .hint{font-size:12px;color:#8b93a1}
  .legend span{display:inline-block;margin:2px 8px 2px 0;font-size:11px}
  .dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;vertical-align:middle}
  code{background:#2c303a;padding:1px 4px;border-radius:3px;font-size:11px}
  pre{background:#15171c;border:1px solid #30343d;border-radius:5px;padding:8px;
      font-size:11px;line-height:1.45;overflow-x:auto;white-space:pre;color:#cfe3d6}
  .codesrc{font-size:10px;color:#8b93a1;margin:8px 0 2px}
</style></head>
<body><div id="wrap">
  <div id="net"></div>
  <div id="panel">
    <h1>Hardware-Feature Knowledge Graph</h1>
    <div class="sub" id="ver"></div>
    <div class="legend">
      <span><span class="dot" style="background:#4f9eed"></span>Hardware</span>
      <span><span class="dot" style="background:#7fd3a0"></span>Feature</span>
      <span><span class="dot" style="background:#37b36b"></span>requires</span>
    </div>
    <div class="hint">Click a node to inspect it.</div>
    <div id="detail"></div>
  </div>
</div>
<script>
const DATA = __DATA__;
const hw = DATA.nodes.filter(n=>n.kind==='hardware');
const ft = DATA.nodes.filter(n=>n.kind==='feature');
const nHF = DATA.edges.filter(e=>e.type==='HAS_FEATURE').length;
const nRQ = DATA.edges.filter(e=>e.type==='REQUIRES_FEATURE').length;
document.getElementById('ver').textContent =
  hw.length+' hardware  ·  '+ft.length+' features  ·  '+nHF+' HAS_FEATURE  ·  '+nRQ+' REQUIRES_FEATURE';

const visNodes=[];
for(const n of hw) visNodes.push({id:n.id,label:n.label,shape:'box',
  color:{background:n.color,border:'#fff'},font:{color:'#fff',size:13},margin:8});
for(const n of ft) visNodes.push({id:n.id,label:n.label,shape:'ellipse',
  color:{background:n.color,border:'#2f6f4c'},font:{color:'#0d2417',size:13}});
const nodes=new vis.DataSet(visNodes);

const visEdges=[];
DATA.edges.forEach((e,i)=>{
  if(e.type==='HAS_FEATURE') visEdges.push({id:i,from:e.from,to:e.to,arrows:'to',
    color:{color:'#454b57',opacity:0.55},width:1});
  else visEdges.push({id:i,from:e.from,to:e.to,label:'requires',arrows:'to',
    color:{color:'#37b36b'},width:2,font:{color:'#7fd3a0',size:10,strokeWidth:3,strokeColor:'#1b1d23'}});
});
const edges=new vis.DataSet(visEdges);

const net=new vis.Network(document.getElementById('net'),{nodes,edges},{
  layout:{improvedLayout:false},
  physics:{stabilization:{enabled:true,iterations:400,updateInterval:25,fit:true},
    barnesHut:{springLength:150,gravitationalConstant:-14000,avoidOverlap:0.1},
    minVelocity:0.75,timestep:0.4},
  interaction:{hover:true,tooltipDelay:120}});
net.once('stabilizationIterationsDone',()=>net.setOptions({physics:false}));
setTimeout(()=>net.setOptions({physics:false}),6000);
net.on('dragEnd',()=>net.setOptions({physics:false}));

function list(a){return a&&a.length?a.map(x=>'<code>'+x+'</code>').join(' '):'<span class="hint">none</span>';}
function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function showHardware(n){
  document.getElementById('detail').innerHTML=
    '<h2><span class="pill" style="background:#4f9eed">HARDWARE</span>'+n.label+'</h2>'
    +'<div class="hint">'+n.hardware_id+'</div><table>'
    +'<tr><td>architectures</td><td>'+list(n.architectures)+'</td></tr>'
    +'<tr><td>compute_capabilities</td><td>'+list(n.compute_capabilities)+'</td></tr>'
    +'<tr><td>vram_MB</td><td>'+(n.vram_MB||'')+'</td></tr>'
    +'<tr><td>vram_type</td><td>'+(n.vram_type||'')+'</td></tr>'
    +'<tr><td>sm_count</td><td>'+(n.sm_count||'')+'</td></tr></table>'
    +'<h2>'+n.features.length+' features (HAS_FEATURE)</h2><div class="desc">'+list(n.features)+'</div>';
}
function showFeature(n){
  const ex=n.example_code?('<div class="codesrc">from '+n.example_code_source+'</div><pre>'+esc(n.example_code)+'</pre>')
    :'<div class="hint">no example code</div>';
  document.getElementById('detail').innerHTML=
    '<h2><span class="pill" style="background:'+n.color+';color:#0d2417">FEATURE</span>'+n.label+'</h2>'
    +'<div class="hint">category: '+n.category+'</div>'
    +'<div class="desc">'+esc(n.description||'')+'</div>'
    +(n.url?'<div class="desc"><a href="'+n.url+'" target="_blank" style="color:#7fd3a0">'+n.url+'</a></div>':'')
    +'<h2>API symbols (merged)</h2><div class="desc">'+list(n.api_symbols)+'</div>'
    +'<h2>example code (1)</h2>'+ex;
}
net.on('click',p=>{
  if(!p.nodes.length) return;
  const n=DATA.nodes.find(x=>x.id===p.nodes[0]); if(!n) return;
  if(n.kind==='hardware') showHardware(n); else showFeature(n);
});
</script></body></html>"""


if __name__ == "__main__":
    raise SystemExit(main())
