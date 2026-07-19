use std::fs;
use std::path::Path;

use serde_json::{Value, json};

use crate::bundler::{VisualizationEdge, VisualizationGraph, VisualizationNode};

pub fn write_visualization(graph: &VisualizationGraph, output: &Path) -> Result<(), String> {
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
    }
    let data = graph_json(graph).to_string().replace("</", "<\\/");
    let html = TEMPLATE.replace("__DIFFPACK_GRAPH_DATA__", &data);
    fs::write(output, html).map_err(|error| format!("cannot write {}: {error}", output.display()))
}

fn graph_json(graph: &VisualizationGraph) -> Value {
    json!({
        "entry": graph.entry,
        "nodes": graph.nodes.iter().map(node_json).collect::<Vec<_>>(),
        "edges": graph.edges.iter().map(edge_json).collect::<Vec<_>>(),
    })
}

fn node_json(node: &VisualizationNode) -> Value {
    json!({
        "id": node.id,
        "denseId": node.dense_id,
        "reachable": node.reachable,
        "entry": node.is_entry,
        "sourceBytes": node.source_bytes,
        "loweredBytes": node.lowered_bytes,
        "flatEligible": node.flat_eligible,
        "directEffects": node.has_direct_effects,
        "declarations": node.declarations,
        "exports": node.exports,
        "foldableConstants": node.foldable_constants,
        "foldableEffects": node.foldable_effects,
        "prunedImports": node.pruned_imports,
    })
}

fn edge_json(edge: &VisualizationEdge) -> Value {
    json!({
        "source": edge.source,
        "target": edge.target,
        "specifier": edge.specifier,
        "dynamic": edge.dynamic,
        "all": edge.all,
        "names": edge.names,
    })
}

const TEMPLATE: &str = r####"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Diffpack Linker Graph</title>
<style>
  :root{color-scheme:dark;--bg:#090b10;--panel:#10141d;--panel2:#151b27;--line:#242c3a;--text:#edf2f7;--muted:#8c98aa;--cyan:#42d3c5;--gold:#ffc857;--orange:#ff8c5a;--purple:#b48cff;--red:#ff627d;--blue:#64a8ff}
  *{box-sizing:border-box}html,body{height:100%;margin:0;overflow:hidden;background:var(--bg);color:var(--text);font:13px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
  body:before{content:"";position:fixed;inset:0;pointer-events:none;background:radial-gradient(circle at 30% 15%,rgba(66,211,197,.08),transparent 30%),radial-gradient(circle at 80% 75%,rgba(180,140,255,.07),transparent 35%)}
  .app{height:100%;display:grid;grid-template-rows:64px 1fr;position:relative}.top{display:flex;align-items:center;gap:16px;padding:0 18px;border-bottom:1px solid var(--line);background:rgba(13,17,24,.9);backdrop-filter:blur(18px);z-index:4}
  .brand{display:flex;align-items:center;gap:11px;min-width:210px}.mark{width:30px;height:30px;border:1px solid #3b4b5d;border-radius:9px;display:grid;place-items:center;background:linear-gradient(145deg,#192532,#0d1219);box-shadow:inset 0 1px rgba(255,255,255,.06)}.mark svg{width:18px}.brand strong{font-size:14px;letter-spacing:.02em}.brand small{display:block;color:var(--muted);font-size:10px;letter-spacing:.12em;text-transform:uppercase}
  .search{position:relative;flex:1;max-width:520px}.search input{width:100%;height:36px;border-radius:9px;border:1px solid var(--line);background:#0b0f16;color:var(--text);outline:none;padding:0 38px 0 34px}.search input:focus{border-color:#3b6d70;box-shadow:0 0 0 3px rgba(66,211,197,.08)}.search svg{position:absolute;left:11px;top:10px;width:16px;color:var(--muted)}.kbd{position:absolute;right:9px;top:8px;border:1px solid #303847;border-radius:5px;padding:1px 6px;color:#768196;font-size:10px}
  .stats{display:flex;gap:18px;margin-left:auto}.stat b{display:block;font-size:14px}.stat span{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.08em}
  .main{min-height:0;display:grid;grid-template-columns:250px 1fr 330px}.side{border-right:1px solid var(--line);background:rgba(14,18,26,.82);padding:17px;overflow:auto;z-index:2}.side.right{border-right:0;border-left:1px solid var(--line);padding:0}.section{margin-bottom:24px}.section h3,.eyebrow{margin:0 0 10px;color:#7f8ca0;font-size:10px;text-transform:uppercase;letter-spacing:.14em;font-weight:700}
  .control{display:flex;align-items:center;justify-content:space-between;padding:7px 0;color:#c6cfdb}.control input{accent-color:var(--cyan)}.legend{display:grid;gap:9px}.legend div{display:flex;align-items:center;gap:9px;color:#aeb8c7}.dot{width:9px;height:9px;border-radius:50%;box-shadow:0 0 10px currentColor}.hint{padding:12px;border:1px solid var(--line);border-radius:9px;background:#0c1017;color:var(--muted);font-size:11px}.hint b{color:#c9d3df}
  .stage{position:relative;min-width:0;overflow:hidden}.stage canvas{width:100%;height:100%;display:block;cursor:grab}.stage canvas.dragging{cursor:grabbing}.tools{position:absolute;left:14px;bottom:14px;display:flex;gap:6px}.tools button,.pill{height:31px;border:1px solid var(--line);border-radius:7px;background:rgba(17,22,31,.92);color:#b8c2d0;padding:0 10px;cursor:pointer}.tools button:hover{border-color:#445064;color:white}.pill{position:absolute;left:14px;top:14px;height:auto;padding:6px 9px;color:var(--muted);pointer-events:none}
  .empty{height:100%;display:grid;place-items:center;text-align:center;color:var(--muted);padding:30px}.inspect{display:none}.inspect.active{display:block}.inspect-head{padding:18px;border-bottom:1px solid var(--line);background:linear-gradient(180deg,#151b26,#10141d)}.inspect-head h2{font-size:15px;margin:3px 0 4px;word-break:break-all}.path{color:var(--muted);font:11px ui-monospace,SFMono-Regular,Menlo,monospace;word-break:break-all}.badges{display:flex;gap:6px;flex-wrap:wrap;margin-top:12px}.badge{border:1px solid #30394a;border-radius:999px;padding:3px 8px;font-size:10px;color:#b7c1cf}.badge.entry{border-color:#6f5a27;color:var(--gold)}.badge.effect{border-color:#70402e;color:var(--orange)}.badge.flat{border-color:#285f5c;color:var(--cyan)}.badge.fallback{border-color:#663140;color:var(--red)}.badge.fold{border-color:#513c75;color:var(--purple)}
  .inspect-body{padding:17px}.metrics{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:20px}.metric{padding:10px;border:1px solid var(--line);border-radius:8px;background:#0c1017}.metric b{display:block;font-size:15px}.metric span{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.1em}.list{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:18px}.token{font:10px ui-monospace,SFMono-Regular,Menlo,monospace;padding:4px 6px;border-radius:5px;background:#1a2130;color:#c9d3e2;border:1px solid #273143}.edge-list{display:grid;gap:6px;margin-bottom:20px}.edge{padding:8px;border:1px solid var(--line);border-radius:7px;background:#0b0f16;cursor:pointer}.edge:hover{border-color:#3e4a5c}.edge strong{display:block;font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.edge span{color:var(--muted);font-size:10px}.code{font:10px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;color:#c7b6ef;background:#0b0f16;border:1px solid var(--line);border-radius:7px;padding:9px;overflow:auto;margin-bottom:6px}
  @media(max-width:900px){.main{grid-template-columns:190px 1fr}.side.right{position:absolute;right:0;top:64px;bottom:0;width:310px;box-shadow:-20px 0 50px #0008}.stats{display:none}} 
</style>
</head>
<body>
<div class="app">
  <header class="top">
    <div class="brand"><div class="mark"><svg viewBox="0 0 24 24" fill="none"><path d="M5 6.5 12 3l7 3.5v8L12 21l-7-6.5v-8Z" stroke="#42d3c5"/><path d="m5 6.5 7 4 7-4M12 10.5V21" stroke="#ffc857"/></svg></div><div><strong>Diffpack</strong><small>Linker graph</small></div></div>
    <label class="search"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/></svg><input id="search" placeholder="Find a module, export, or declaration…"><span class="kbd">/</span></label>
    <div class="stats"><div class="stat"><b id="nodeCount">0</b><span>Modules</span></div><div class="stat"><b id="edgeCount">0</b><span>Imports</span></div><div class="stat"><b id="reachableCount">0</b><span>Reachable</span></div></div>
  </header>
  <main class="main">
    <aside class="side">
      <section class="section"><h3>View</h3><label class="control">Dynamic imports<input id="dynamic" type="checkbox" checked></label><label class="control">Unreachable modules<input id="unreachable" type="checkbox" checked></label><label class="control">Module labels<input id="labels" type="checkbox" checked></label></section>
      <section class="section"><h3>Linker state</h3><div class="legend"><div><i class="dot" style="color:var(--gold);background:var(--gold)"></i>Entry module</div><div><i class="dot" style="color:var(--orange);background:var(--orange)"></i>Direct effects</div><div><i class="dot" style="color:var(--purple);background:var(--purple)"></i>Constant-foldable</div><div><i class="dot" style="color:var(--cyan);background:var(--cyan)"></i>Flat / scope-hoisted</div><div><i class="dot" style="color:var(--red);background:var(--red)"></i>Runtime fallback</div></div></section>
      <section class="section"><h3>How to read this</h3><div class="hint"><b>Solid edges</b> are static imports. <b>Dashed edges</b> are dynamic imports. Click a module to see exactly which exports an edge demands and why the linker keeps it.</div></section>
    </aside>
    <section class="stage" id="stage"><canvas id="canvas"></canvas><div class="pill" id="status">Drag to pan · wheel to zoom · click to inspect</div><div class="tools"><button id="fit">Fit graph</button><button id="entry">Entry</button><button id="reset">Reset filters</button></div></section>
    <aside class="side right"><div class="empty" id="empty"><div><div style="font-size:28px;margin-bottom:8px">⬡</div>Select a module to inspect its linker state.</div></div><div class="inspect" id="inspect"><div class="inspect-head"><div class="eyebrow">Selected module</div><h2 id="title"></h2><div class="path" id="path"></div><div class="badges" id="badges"></div></div><div class="inspect-body"><div class="metrics" id="metrics"></div><div id="details"></div></div></div></aside>
  </main>
</div>
<script id="graph-data" type="application/json">__DIFFPACK_GRAPH_DATA__</script>
<script>
const graph=JSON.parse(document.getElementById('graph-data').textContent);
const canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d'),stage=document.getElementById('stage');
const byId=new Map(graph.nodes.map(n=>[n.denseId,n]));
const outgoing=new Map(graph.nodes.map(n=>[n.denseId,[]])),incoming=new Map(graph.nodes.map(n=>[n.denseId,[]]));
for(const e of graph.edges){if(outgoing.has(e.source)&&incoming.has(e.target)){outgoing.get(e.source).push(e);incoming.get(e.target).push(e)}}
const entryNode=graph.nodes.find(n=>n.entry);let selected=null,hovered=null,scale=1,panX=0,panY=0,drag=null;
const filter={dynamic:true,unreachable:true,labels:true,query:''};
function shortName(id){return id.split(/[\\/]/).pop()||id}
function layout(){const level=new Map();if(entryNode){level.set(entryNode.denseId,0);const q=[entryNode.denseId];for(let i=0;i<q.length;i++){const s=q[i],d=level.get(s);for(const e of outgoing.get(s)||[]){if(!level.has(e.target)){level.set(e.target,d+1);q.push(e.target)}}}}let max=Math.max(0,...level.values());const groups=new Map();for(const n of graph.nodes){const d=level.get(n.denseId)??++max;if(!groups.has(d))groups.set(d,[]);groups.get(d).push(n)}for(const [d,nodes] of groups){nodes.sort((a,b)=>a.id.localeCompare(b.id));const gap=Math.max(34,Math.min(92,1200/Math.sqrt(nodes.length)));nodes.forEach((n,i)=>{n.x=d*240;n.y=(i-(nodes.length-1)/2)*gap})}}
function visibleNode(n){return(filter.unreachable||n.reachable)&&(!filter.query||searchText(n).includes(filter.query))}
function searchText(n){return[n.id,...n.exports,...n.declarations,...n.foldableConstants].join(' ').toLowerCase()}
function nodeColor(n){if(n.entry)return'#ffc857';if(n.foldableConstants.length||n.foldableEffects.length)return'#b48cff';if(n.directEffects)return'#ff8c5a';if(!n.flatEligible)return'#ff627d';return'#42d3c5'}
function resize(){const dpr=devicePixelRatio||1,w=stage.clientWidth,h=stage.clientHeight;canvas.width=w*dpr;canvas.height=h*dpr;canvas.style.width=w+'px';canvas.style.height=h+'px';ctx.setTransform(dpr,0,0,dpr,0,0);draw()}
function worldToScreen(n){return{x:n.x*scale+panX,y:n.y*scale+panY}}
function draw(){const w=stage.clientWidth,h=stage.clientHeight;ctx.clearRect(0,0,w,h);ctx.save();ctx.translate(panX,panY);ctx.scale(scale,scale);const selectedEdges=selected?new Set([...(outgoing.get(selected.denseId)||[]),...(incoming.get(selected.denseId)||[])]):null;for(const e of graph.edges){if(e.dynamic&&!filter.dynamic)continue;const a=byId.get(e.source),b=byId.get(e.target);if(!a||!b||!visibleNode(a)||!visibleNode(b))continue;const hi=selectedEdges?.has(e);ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.strokeStyle=hi?(e.dynamic?'#b48cff':'#64a8ff'):(e.dynamic?'#57436f':'#293444');ctx.lineWidth=(hi?2:1)/scale;ctx.globalAlpha=hi?1:.62;if(e.dynamic)ctx.setLineDash([6/scale,5/scale]);else ctx.setLineDash([]);ctx.stroke()}ctx.setLineDash([]);for(const n of graph.nodes){if(!visibleNode(n))continue;const color=nodeColor(n),active=n===selected||n===hovered,neighbor=selected&&((outgoing.get(selected.denseId)||[]).some(e=>e.target===n.denseId)||(incoming.get(selected.denseId)||[]).some(e=>e.source===n.denseId));ctx.globalAlpha=n.reachable?1:.25;ctx.beginPath();ctx.arc(n.x,n.y,(active?10:n.entry?8:6)/scale,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();if(active||neighbor){ctx.strokeStyle=active?'#fff':color;ctx.lineWidth=2/scale;ctx.stroke()}if(filter.labels&&(scale>.48||active)){ctx.globalAlpha=active?1:.78;ctx.fillStyle=active?'#fff':'#aeb9c8';ctx.font=`${active?'600 ':''}${11/scale}px ui-monospace,monospace`;ctx.fillText(shortName(n.id),n.x+12/scale,n.y+4/scale)}}ctx.restore();ctx.globalAlpha=1;document.getElementById('status').textContent=`${visibleNodes().length.toLocaleString()} visible · ${Math.round(scale*100)}% zoom`}
function visibleNodes(){return graph.nodes.filter(visibleNode)}
function fit(nodes=visibleNodes()){if(!nodes.length)return;const xs=nodes.map(n=>n.x),ys=nodes.map(n=>n.y),minX=Math.min(...xs)-40,maxX=Math.max(...xs)+160,minY=Math.min(...ys)-50,maxY=Math.max(...ys)+50;const w=stage.clientWidth,h=stage.clientHeight;scale=Math.max(.03,Math.min(1.35,(w-50)/(maxX-minX),(h-50)/(maxY-minY)));panX=(w-(minX+maxX)*scale)/2;panY=(h-(minY+maxY)*scale)/2;draw()}
function focus(n){selected=n;const w=stage.clientWidth,h=stage.clientHeight;scale=Math.max(scale,.85);panX=w*.46-n.x*scale;panY=h*.5-n.y*scale;inspect(n);draw()}
function inspect(n){document.getElementById('empty').style.display='none';document.getElementById('inspect').classList.add('active');document.getElementById('title').textContent=shortName(n.id);document.getElementById('path').textContent=n.id;const badges=[];if(n.entry)badges.push(['entry','entry']);if(n.directEffects)badges.push(['effect','direct effects']);badges.push([n.flatEligible?'flat':'fallback',n.flatEligible?'flat eligible':'runtime fallback']);if(n.foldableConstants.length||n.foldableEffects.length)badges.push(['fold','foldable']);if(!n.reachable)badges.push(['','unreachable']);document.getElementById('badges').innerHTML=badges.map(([c,t])=>`<span class="badge ${c}">${t}</span>`).join('');document.getElementById('metrics').innerHTML=metric(n.sourceBytes,'source bytes')+metric(n.loweredBytes,'lowered bytes')+metric((outgoing.get(n.denseId)||[]).length,'imports')+metric((incoming.get(n.denseId)||[]).length,'importers');let html=listSection('Exports',n.exports)+listSection('Top-level declarations',n.declarations)+listSection('Pruned imports',n.prunedImports)+codeSection('Foldable constants',n.foldableConstants)+codeSection('Observable foldable effects',n.foldableEffects)+edgeSection('Imports',outgoing.get(n.denseId)||[],true)+edgeSection('Imported by',incoming.get(n.denseId)||[],false);document.getElementById('details').innerHTML=html;document.querySelectorAll('[data-focus]').forEach(el=>el.onclick=()=>focus(byId.get(Number(el.dataset.focus))))}
function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function metric(v,l){return`<div class="metric"><b>${Number(v).toLocaleString()}</b><span>${l}</span></div>`}
function listSection(title,items){if(!items.length)return'';return`<div class="eyebrow">${title}</div><div class="list">${items.map(x=>`<span class="token">${esc(x)}</span>`).join('')}</div>`}
function codeSection(title,items){if(!items.length)return'';return`<div class="eyebrow">${title}</div>${items.map(x=>`<div class="code">${esc(x)}</div>`).join('')}<div style="height:12px"></div>`}
function edgeSection(title,edges,isOut){if(!edges.length)return'';return`<div class="eyebrow">${title}</div><div class="edge-list">${edges.map(e=>{const other=byId.get(isOut?e.target:e.source),d=e.dynamic?'dynamic':e.all?'all exports':e.names.length?e.names.join(', '):'side effects';return`<div class="edge" data-focus="${other.denseId}"><strong>${esc(shortName(other.id))}</strong><span>${esc(e.specifier)} · ${esc(d)}</span></div>`}).join('')}</div>`}
function pick(x,y){let best=null,dist=16;for(const n of visibleNodes()){const p=worldToScreen(n),d=Math.hypot(p.x-x,p.y-y);if(d<dist){best=n;dist=d}}return best}
canvas.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY,panX,panY,moved:false};canvas.setPointerCapture(e.pointerId);canvas.classList.add('dragging')});canvas.addEventListener('pointermove',e=>{if(drag){const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(Math.abs(dx)+Math.abs(dy)>3)drag.moved=true;panX=drag.panX+dx;panY=drag.panY+dy;draw()}else{hovered=pick(e.offsetX,e.offsetY);draw()}});canvas.addEventListener('pointerup',e=>{if(drag&&!drag.moved){const n=pick(e.offsetX,e.offsetY);if(n){selected=n;inspect(n)}else selected=null}drag=null;canvas.classList.remove('dragging');draw()});canvas.addEventListener('wheel',e=>{e.preventDefault();const factor=Math.exp(-e.deltaY*.001),next=Math.max(.03,Math.min(4,scale*factor)),wx=(e.offsetX-panX)/scale,wy=(e.offsetY-panY)/scale;panX=e.offsetX-wx*next;panY=e.offsetY-wy*next;scale=next;draw()},{passive:false});
document.getElementById('search').addEventListener('input',e=>{filter.query=e.target.value.trim().toLowerCase();const matches=visibleNodes();if(matches.length===1)focus(matches[0]);else{selected=null;fit(matches)}});for(const key of ['dynamic','unreachable','labels'])document.getElementById(key).onchange=e=>{filter[key]=e.target.checked;fit()};document.getElementById('fit').onclick=()=>fit();document.getElementById('entry').onclick=()=>entryNode&&focus(entryNode);document.getElementById('reset').onclick=()=>{filter.dynamic=filter.unreachable=filter.labels=true;filter.query='';for(const k of ['dynamic','unreachable','labels'])document.getElementById(k).checked=true;document.getElementById('search').value='';selected=null;fit()};document.addEventListener('keydown',e=>{if(e.key==='/'&&document.activeElement.tagName!=='INPUT'){e.preventDefault();document.getElementById('search').focus()}if(e.key==='Escape'){document.getElementById('search').value='';filter.query='';fit()}});
document.getElementById('nodeCount').textContent=graph.nodes.length.toLocaleString();document.getElementById('edgeCount').textContent=graph.edges.length.toLocaleString();document.getElementById('reachableCount').textContent=graph.nodes.filter(n=>n.reachable).length.toLocaleString();layout();new ResizeObserver(resize).observe(stage);setTimeout(()=>fit(),30);
</script>
</body>
</html>"####;

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn writes_a_self_contained_and_script_safe_visualization() {
        let directory = tempdir().unwrap();
        let output = directory.path().join("graph.html");
        let graph = VisualizationGraph {
            entry: "/project/entry.js".into(),
            nodes: vec![VisualizationNode {
                id: "/project/</script>entry.js".into(),
                dense_id: 0,
                reachable: true,
                is_entry: true,
                source_bytes: 10,
                lowered_bytes: 8,
                flat_eligible: true,
                has_direct_effects: true,
                declarations: vec!["value".into()],
                exports: vec![],
                foldable_constants: vec![],
                foldable_effects: vec![],
                pruned_imports: vec![],
            }],
            edges: vec![],
        };
        write_visualization(&graph, &output).unwrap();
        let html = fs::read_to_string(output).unwrap();
        assert!(html.contains("Diffpack Linker Graph"));
        assert!(html.contains(r#"<\/script>entry.js"#));
        assert_eq!(html.matches("</script>").count(), 2);
    }
}
