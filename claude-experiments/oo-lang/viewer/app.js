// scry viewer — every pane is sugar over POST /eval {id,source} -> {id,value|error}.
// Refresh = re-eval on an interval / on focus / after an action. Nothing is pushed.
"use strict";

let evalSeq = 0;
const state = {
  view: "index",            // "index" | "table" | "detail"
  typeName: null,           // for table
  ref: null,                // {class, slot, gen} for detail
  breadcrumbs: [],          // [{label, go}]
  schema: [],               // last types() items
  lastCounts: {},           // name -> count (trend)
  countTrend: {},           // name -> "up"|"down"|""
  pollTimer: null,
  lastDetail: null,         // previous detail fields (for flash diff)
  filter: "",
  ifaceOpen: {},
};

// ---------- the one wire op ----------
async function evalSource(source) {
  const id = "e" + (++evalSeq);
  const t0 = performance.now();
  logTx(source, null, false);
  try {
    const res = await fetch("/eval", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id, source }),
    });
    const json = await res.json();
    const dt = performance.now() - t0;
    setConn(dt < 1200 ? "live" : "slow");
    logTx(source, json, json.error != null, true);
    return json;
  } catch (e) {
    setConn("down");
    logTx(source, { error: { kind: "Transport", message: String(e) } }, true, true);
    return { error: { kind: "Transport", message: String(e) } };
  }
}

// ---------- connection indicator ----------
function setConn(s) {
  const el = document.getElementById("conn");
  el.className = "conn " + (s === "down" ? "down" : "live");
  el.querySelector(".conn-label").textContent =
    s === "down" ? "reconnecting…" : s === "slow" ? "live (slow)" : "● live";
}

// ---------- transcript ----------
const txList = [];
function logTx(source, resp, isErr, isResp) {
  if (!isResp) {
    txList.unshift({ source, resp: null, isErr: false, t: new Date() });
  } else {
    // attach response to the most recent matching request
    const e = txList.find((x) => x.source === source && x.resp === null);
    if (e) { e.resp = resp; e.isErr = isErr; }
    else txList.unshift({ source, resp, isErr, t: new Date() });
  }
  if (txList.length > 200) txList.length = 200;
  renderTx();
}
function renderTx() {
  const box = document.getElementById("transcript-list");
  if (document.getElementById("transcript").classList.contains("collapsed")) return;
  box.innerHTML = "";
  for (const e of txList.slice(0, 120)) {
    const d = document.createElement("div");
    d.className = "tx";
    const req = document.createElement("div");
    req.className = "tx-req"; req.textContent = truncate(e.source, 200);
    d.appendChild(req);
    if (e.resp) {
      const r = document.createElement("div");
      r.className = "tx-res" + (e.isErr ? " err" : "");
      r.textContent = truncate(JSON.stringify(e.resp.value ?? e.resp.error), 300);
      d.appendChild(r);
    }
    box.appendChild(d);
  }
}
function truncate(s, n) { s = String(s).replace(/\s+/g, " "); return s.length > n ? s.slice(0, n) + "…" : s; }

// ---------- value renderer (shared everywhere) ----------
function renderValue(v, opts = {}) {
  const span = (cls, txt) => { const s = document.createElement("span"); s.className = cls; s.textContent = txt; return s; };
  if (v == null) return span("v-void", "null");
  switch (v.type) {
    case "Int": case "Float": return span("v-number", String(v.value));
    case "Bool": return span("v-bool", String(v.value));
    case "String": return span("v-string", JSON.stringify(v.value));
    case "Void": return span("v-void", "void");
    case "ref": return refLink(v);
    case "list": return renderCollection(v, opts);
    case "map": return renderMap(v, opts);
    default:
      if (v.case !== undefined) return renderEnum(v);
      if (v.ref !== undefined && v.fields) return refLink({ ...v, class: v.type, summary: v.ref });
      { const s = span("v-void", JSON.stringify(v)); return s; }
  }
}
function renderEnum(v) {
  const wrap = document.createElement("span");
  const pill = document.createElement("span");
  pill.className = "pill";
  pill.textContent = v.type + "." + v.case;
  wrap.appendChild(pill);
  if (v.payload && v.payload.length) {
    wrap.appendChild(document.createTextNode(" "));
    v.payload.forEach((p, i) => { if (i) wrap.appendChild(document.createTextNode(", ")); wrap.appendChild(renderValue(p)); });
  }
  return wrap;
}
function refLink(v) {
  const a = document.createElement("span");
  a.className = "reflink";
  a.textContent = v.summary && v.summary !== v.ref ? `${v.ref} · ${v.summary}` : v.ref;
  a.title = v.ref + " (gen " + v.generation + ")";
  a.onclick = (e) => { e.stopPropagation(); navigateRef(v); };
  return a;
}
function renderCollection(v, opts) {
  const wrap = document.createElement("span");
  const items = v.items || [];
  if (items.length && items[0].type === "ref") {
    items.forEach((it) => { const c = document.createElement("span"); c.className = "chip"; c.appendChild(renderValue(it)); wrap.appendChild(c); });
  } else if (opts.inline === false) {
    items.forEach((it, i) => { if (i) wrap.appendChild(document.createTextNode(", ")); wrap.appendChild(renderValue(it)); });
  } else {
    wrap.appendChild(document.createTextNode(`${v.length} × ${v.elementType}`));
    if (items.length && items.length <= 6) {
      wrap.appendChild(document.createTextNode("  ["));
      items.forEach((it, i) => { if (i) wrap.appendChild(document.createTextNode(", ")); wrap.appendChild(renderValue(it)); });
      wrap.appendChild(document.createTextNode("]"));
    }
  }
  if (v.truncated) { const m = document.createElement("span"); m.className = "list-more"; m.textContent = `  (+${v.length - items.length} more)`; wrap.appendChild(m); }
  return wrap;
}
function renderMap(v, opts) {
  const wrap = document.createElement("span");
  wrap.appendChild(document.createTextNode(`{${v.length} entries} `));
  (v.entries || []).slice(0, 8).forEach(([k, val], i) => {
    if (i) wrap.appendChild(document.createTextNode(", "));
    wrap.appendChild(renderValue(k)); wrap.appendChild(document.createTextNode(": ")); wrap.appendChild(renderValue(val));
  });
  if (v.truncated) { const m = document.createElement("span"); m.className = "list-more"; m.textContent = ` (+${v.length - (v.entries||[]).length} more)`; wrap.appendChild(m); }
  return wrap;
}

// ---------- navigation ----------
function navigateRef(v) {
  const m = /^(.+)#(\d+)$/.exec(v.ref);
  if (!m) return;
  openDetail(v.class || v.type, +m[2], v.generation ?? 0, true);
}
function setBreadcrumbs(crumbs) {
  const bc = document.getElementById("breadcrumbs");
  bc.innerHTML = "";
  crumbs.forEach((c, i) => {
    if (i) { const s = document.createElement("span"); s.className = "sep"; s.textContent = "›"; bc.appendChild(s); }
    const el = document.createElement("span");
    el.className = "crumb" + (c.go ? "" : " static");
    el.textContent = c.label;
    if (c.go) el.onclick = c.go;
    bc.appendChild(el);
  });
}
function stopPoll() { if (state.pollTimer) { clearInterval(state.pollTimer); state.pollTimer = null; } }
function startPoll(fn, ms) { stopPoll(); state.pollTimer = setInterval(() => { if (!document.hidden) fn(); }, ms); }

// ---------- type index ----------
async function refreshTypes() {
  const r = await evalSource("types()");
  if (!r.value) return;
  const items = r.value.items || [];
  // trend
  const trend = {};
  for (const t of items) {
    const prev = state.lastCounts[t.name];
    if (prev != null && t.liveCount !== prev) trend[t.name] = t.liveCount > prev ? "up" : "down";
    else trend[t.name] = state.countTrend[t.name] || "";
    state.lastCounts[t.name] = t.liveCount;
  }
  state.countTrend = trend;
  state.schema = items;
  renderTypeList();
}
function renderTypeList() {
  const ul = document.getElementById("type-list");
  const filt = document.getElementById("type-search").value.toLowerCase();
  ul.innerHTML = "";
  // group interface implementors
  const byIface = {};
  const plain = [];
  for (const t of state.schema) {
    if (t.implements && t.implements.length) {
      for (const i of t.implements) (byIface[i] = byIface[i] || []).push(t);
    } else plain.push(t);
  }
  const shown = new Set();
  for (const iface of Object.keys(byIface).sort()) {
    if (filt && !iface.toLowerCase().includes(filt) && !byIface[iface].some(t => t.name.toLowerCase().includes(filt))) continue;
    const total = byIface[iface].reduce((a, t) => a + t.liveCount, 0);
    const li = document.createElement("li"); li.className = "iface-group";
    const row = document.createElement("div"); row.className = "type-row iface";
    const caret = document.createElement("span"); caret.className = "caret" + (state.ifaceOpen[iface] ? " open" : ""); caret.textContent = "▶";
    row.appendChild(caret);
    const nm = document.createElement("span"); nm.className = "tname"; nm.textContent = iface + " ‹interface›";
    const cnt = document.createElement("span"); cnt.className = "count"; cnt.textContent = total + " live";
    row.appendChild(nm); row.appendChild(cnt);
    row.onclick = () => { state.ifaceOpen[iface] = !state.ifaceOpen[iface]; renderTypeList(); };
    li.appendChild(row);
    if (state.ifaceOpen[iface]) {
      const kids = document.createElement("div"); kids.className = "iface-children";
      for (const t of byIface[iface]) { kids.appendChild(typeRow(t, filt)); shown.add(t.name); }
      li.appendChild(kids);
    } else byIface[iface].forEach(t => shown.add(t.name));
    ul.appendChild(li);
  }
  for (const t of plain) {
    if (shown.has(t.name)) continue;
    if (filt && !t.name.toLowerCase().includes(filt)) continue;
    const li = document.createElement("li"); li.appendChild(typeRow(t, filt)); ul.appendChild(li);
  }
}
function typeRow(t, filt) {
  const row = document.createElement("div");
  row.className = "type-row" + (state.view === "table" && state.typeName === t.name ? " active" : "");
  const tr = state.countTrend[t.name] || "";
  const trend = document.createElement("span"); trend.className = "trend " + tr; trend.textContent = tr === "up" ? "▲" : tr === "down" ? "▼" : "";
  const nm = document.createElement("span"); nm.className = "tname"; nm.textContent = t.name;
  const cnt = document.createElement("span"); cnt.className = "count" + (tr ? " changed" : ""); cnt.textContent = t.liveCount + " live";
  row.appendChild(nm); row.appendChild(cnt); row.appendChild(trend);
  row.onclick = () => openTable(t.name);
  return row;
}

// ---------- instance table ----------
function schemaFor(name) { return state.schema.find((t) => t.name === name); }
function openTable(name) {
  state.view = "table"; state.typeName = name; state.filter = "";
  setBreadcrumbs([{ label: "types", go: goIndex }, { label: name }]);
  renderTableShell();
  refreshTable();
  startPoll(refreshTable, 750);
  renderTypeList();
}
function goIndex() { state.view = "index"; stopPoll(); setBreadcrumbs([{ label: "types" }]);
  document.getElementById("pane").innerHTML = `<div class="pane-title">Entity types</div><div class="pane-sub">Pick a type from the rail to browse its live instances. Counts refresh automatically.</div>`;
  renderTypeList();
}
function renderTableShell() {
  const sc = schemaFor(state.typeName);
  const pane = document.getElementById("pane");
  pane.innerHTML = "";
  const title = document.createElement("div"); title.className = "pane-title"; title.textContent = state.typeName;
  pane.appendChild(title);
  const sub = document.createElement("div"); sub.className = "pane-sub";
  sub.textContent = sc ? `${sc.fields.length} fields · ${sc.methods.length} methods` : "";
  pane.appendChild(sub);
  const tools = document.createElement("div"); tools.className = "tbl-tools";
  const fb = document.createElement("input"); fb.className = "filter-box"; fb.placeholder = 'filter, e.g.  name == "coder"  or  status contains "run"';
  fb.value = state.filter;
  fb.oninput = () => { state.filter = fb.value; };
  fb.onkeydown = (e) => { if (e.key === "Enter") refreshTable(); };
  const meta = document.createElement("div"); meta.className = "tbl-meta"; meta.id = "tbl-meta";
  tools.appendChild(fb); tools.appendChild(meta);
  pane.appendChild(tools);
  const holder = document.createElement("div"); holder.id = "tbl-holder"; pane.appendChild(holder);
}
async function refreshTable() {
  const name = state.typeName;
  const f = state.filter.replace(/"/g, '\\"');
  const r = await evalSource(`${name}.instances(filter: "${f}", offset: 0, limit: 200)`);
  if (state.view !== "table" || state.typeName !== name) return;
  const holder = document.getElementById("tbl-holder");
  if (!holder) return;
  if (r.error) { holder.innerHTML = `<div class="invoke-result invoke-error">${esc(r.error.kind)}: ${esc(r.error.message)}</div>`; return; }
  const items = r.value.items || [];
  const meta = document.getElementById("tbl-meta");
  if (meta) meta.textContent = `${r.value.length} live${r.value.truncated ? " (showing " + items.length + ")" : ""}`;
  const sc = schemaFor(name);
  const cols = sc ? sc.fields.map((f) => f.name) : (items[0] ? Object.keys(items[0].fields) : []);
  const tbl = document.createElement("table"); tbl.className = "itable";
  const thead = document.createElement("thead"); const htr = document.createElement("tr");
  htr.appendChild(th("id")); cols.forEach((c) => htr.appendChild(th(c)));
  thead.appendChild(htr); tbl.appendChild(thead);
  const tb = document.createElement("tbody");
  if (!items.length) { holder.innerHTML = ""; const e = document.createElement("div"); e.className = "empty"; e.textContent = "no live instances match."; holder.appendChild(e); return; }
  for (const it of items) {
    const tr = document.createElement("tr");
    const m = /#(\d+)$/.exec(it.ref);
    tr.onclick = () => openDetail(name, m ? +m[1] : 0, it.generation, true);
    const idc = document.createElement("td"); idc.className = "col-id"; idc.textContent = it.ref; tr.appendChild(idc);
    for (const c of cols) { const td = document.createElement("td"); const fv = it.fields[c]; if (fv) td.appendChild(renderValue(fv)); else td.textContent = "—"; tr.appendChild(td); }
    tb.appendChild(tr);
  }
  tbl.appendChild(tb);
  holder.innerHTML = ""; holder.appendChild(tbl);
}
function th(t) { const e = document.createElement("th"); e.textContent = t; return e; }

// ---------- instance detail ----------
function openDetail(cls, slot, gen, pushCrumb) {
  state.view = "detail"; state.ref = { class: cls, slot, gen }; state.lastDetail = null;
  document.getElementById("repl-context").textContent = `self = ${cls}#${slot}`;
  const crumbLabel = `${cls}#${slot}`;
  if (pushCrumb) {
    const base = state.breadcrumbs.length && state.breadcrumbs[0].label === "types" ? state.breadcrumbs : [{ label: "types", go: goIndex }];
    state.breadcrumbs = base.filter((c) => c.label !== crumbLabel);
    state.breadcrumbs.push({ label: crumbLabel, go: () => openDetail(cls, slot, gen, false) });
  }
  setBreadcrumbs(state.breadcrumbs.length ? state.breadcrumbs : [{ label: "types", go: goIndex }, { label: crumbLabel }]);
  refreshDetail();
  startPoll(refreshDetail, 750);
}
async function refreshDetail() {
  const { class: cls, slot, gen } = state.ref;
  const r = await evalSource(`${cls}.at(${slot}, ${gen})`);
  if (state.view !== "detail" || !state.ref || state.ref.slot !== slot) return;
  const pane = document.getElementById("pane");
  if (r.error) { pane.innerHTML = `<div class="pane-title">${esc(cls)}#${slot}</div><div class="invoke-result invoke-error">${esc(r.error.kind)}: ${esc(r.error.message)}</div>`; return; }
  const inst = r.value;
  const sc = schemaFor(cls);
  const prev = state.lastDetail;
  pane.innerHTML = "";
  const title = document.createElement("div"); title.className = "pane-title"; title.textContent = inst.ref;
  pane.appendChild(title);
  const sub = document.createElement("div"); sub.className = "pane-sub";
  sub.innerHTML = `generation ${inst.generation}`;
  if (sc && sc.implements && sc.implements.length) sub.innerHTML += ` · implements <span class="impl">${sc.implements.join(", ")}</span>`;
  if (sc) {
    const edit = document.createElement("button"); edit.className = "ghost-btn edit-src";
    edit.textContent = "✎ edit source";
    edit.onclick = () => openCodePanel(cls, sc);
    sub.appendChild(edit);
  }
  pane.appendChild(sub);

  // fields
  const fsec = document.createElement("div"); fsec.className = "detail-section";
  fsec.innerHTML = `<h3>fields</h3>`;
  const grid = document.createElement("div"); grid.className = "field-grid";
  const typeOf = (n) => sc ? (sc.fields.find((f) => f.name === n) || {}).type : "";
  for (const [k, val] of Object.entries(inst.fields)) {
    const nameCell = document.createElement("div"); nameCell.className = "fcell fname";
    nameCell.innerHTML = esc(k) + `<span class="ftype">${esc(typeOf(k) || "")}</span>`;
    const valCell = document.createElement("div"); valCell.className = "fcell fval";
    valCell.appendChild(renderValue(val));
    if (prev && JSON.stringify(prev[k]) !== JSON.stringify(val)) valCell.classList.add("flash");
    grid.appendChild(nameCell); grid.appendChild(valCell);
  }
  fsec.appendChild(grid); pane.appendChild(fsec);
  state.lastDetail = inst.fields;

  // methods
  if (sc && sc.methods.length) {
    const msec = document.createElement("div"); msec.className = "detail-section";
    msec.innerHTML = `<h3>methods</h3>`;
    for (const m of sc.methods) msec.appendChild(methodCard(cls, slot, gen, m));
    pane.appendChild(msec);
  }
}
function methodCard(cls, slot, gen, m) {
  const card = document.createElement("div"); card.className = "method";
  const head = document.createElement("div"); head.className = "method-head";
  const sig = document.createElement("span"); sig.className = "method-sig";
  const params = m.params.map((p) => `${p.name}: ${p.type}`).join(", ");
  sig.innerHTML = `${esc(m.name)}(${esc(params)}) <span class="mret">→ ${esc(m.returns)}</span>`;
  const btn = document.createElement("button"); btn.className = "invoke-btn"; btn.textContent = "invoke";
  head.appendChild(sig); head.appendChild(btn);
  const body = document.createElement("div"); body.className = "method-body";
  const inputs = {};
  for (const p of m.params) {
    const row = document.createElement("div"); row.className = "arg-row";
    const lab = document.createElement("label"); lab.textContent = `${p.name}: ${p.type}`;
    const inp = document.createElement("input"); inp.placeholder = literalHint(p.type);
    inputs[p.name] = { inp, type: p.type };
    row.appendChild(lab); row.appendChild(inp); body.appendChild(row);
  }
  const result = document.createElement("div"); result.style.display = "none"; result.className = "invoke-result";
  const doInvoke = async () => {
    const args = m.params.map((p) => literalFor(inputs[p.name].inp.value, p.type)).join(", ");
    const src = `${cls}.at(${slot}, ${gen}).${m.name}(${args})`;
    const r = await evalSource(src);
    result.style.display = "block";
    result.className = "invoke-result flash" + (r.error ? " invoke-error" : "");
    result.innerHTML = "";
    if (r.error) {
      result.appendChild(document.createTextNode(`${r.error.kind}: ${r.error.message}`));
      if (r.error.trace) { const t = document.createElement("div"); t.className = "etrace"; t.textContent = r.error.trace.map((f) => `${f.type}.${f.method} (line ${f.line})`).join(" › "); result.appendChild(t); }
    } else result.appendChild(renderValue(r.value));
    setTimeout(() => refreshDetail(), 60);  // read the mutation back immediately
  };
  head.onclick = (e) => { if (e.target === btn && m.params.length === 0) { doInvoke(); return; } card.classList.toggle("open"); };
  btn.onclick = (e) => { e.stopPropagation(); if (m.params.length) card.classList.add("open"); doInvoke(); };
  const runRow = document.createElement("div"); runRow.className = "arg-row";
  const run = document.createElement("button"); run.className = "invoke-btn"; run.textContent = "run";
  run.onclick = doInvoke;
  if (m.params.length) { runRow.appendChild(run); body.appendChild(runRow); }
  body.appendChild(result);
  card.appendChild(head); card.appendChild(body);
  return card;
}
function literalHint(type) {
  if (type === "String") return '"text"';
  if (type === "Int") return "0"; if (type === "Float") return "0.0"; if (type === "Bool") return "true";
  return "expression";
}
function literalFor(v, type) {
  v = v.trim();
  if (type === "String" && !(v.startsWith('"'))) return JSON.stringify(v);
  return v || "0";
}

// ---------- repl ----------
function replSubmit(src) {
  let source = src;
  if (state.view === "detail" && state.ref) {
    const { class: cls, slot, gen } = state.ref;
    source = src.replace(/\bself\b/g, `${cls}.at(${slot}, ${gen})`);
  }
  const scroll = document.getElementById("repl-scroll");
  const entry = document.createElement("div"); entry.className = "repl-entry";
  const ex = document.createElement("div"); ex.className = "repl-expr"; ex.textContent = src;
  entry.appendChild(ex);
  const out = document.createElement("div"); out.className = "repl-out"; out.textContent = "…";
  entry.appendChild(out);
  scroll.appendChild(entry); scroll.scrollTop = scroll.scrollHeight;
  evalSource(source).then((r) => {
    out.innerHTML = "";
    if (r.error) { out.className = "repl-out err"; out.textContent = `${r.error.kind}: ${r.error.message}`; }
    else out.appendChild(renderValue(r.value, { inline: false }));
    scroll.scrollTop = scroll.scrollHeight;
    if (state.view === "detail") setTimeout(refreshDetail, 60);
  });
}

// ---------- code panel (live code change / redefinition) ----------
// A definition eval is live redefinition: POST the source, show accepted {gen} or the
// rejection diagnostic inline. After acceptance the detail poll picks up the new behavior
// on its own (calls resolve through the swapped method table). Persistent drawer, so the
// 750ms detail refresh never wipes what you are typing.
function cleanType(t) { return String(t).replace(/^\w+:/, ""); }
function classSkeleton(cls, sc) {
  const impl = sc.implements && sc.implements.length ? " implements " + sc.implements.join(", ") : "";
  let s = `class ${cls}${impl} {\n`;
  for (const f of sc.fields) s += `  ${f.name}: ${cleanType(f.type)}\n`;
  if (sc.methods.length) s += "\n";
  for (const m of sc.methods) {
    const params = m.params.map((p) => `${p.name}: ${cleanType(p.type)}`).join(", ");
    const ret = m.returns && m.returns !== "Void" && m.returns !== "()" ? ` -> ${cleanType(m.returns)}` : "";
    s += `  fn ${m.name}(${params})${ret} {\n    // edit this body\n  }\n`;
  }
  return s + "}\n";
}
function openCodePanel(cls, sc) {
  const panel = document.getElementById("code-panel");
  const ed = document.getElementById("code-editor");
  document.getElementById("code-cls").textContent = cls;
  // keep an in-progress draft for this class; otherwise seed from the live schema skeleton
  if (!state.codeDraft || state.codeDraft.cls !== cls) {
    ed.value = classSkeleton(cls, sc);
    state.codeDraft = { cls, text: ed.value };
  } else {
    ed.value = state.codeDraft.text;
  }
  document.getElementById("code-result").textContent = "";
  document.getElementById("code-result").className = "code-result";
  panel.classList.remove("collapsed");
  ed.focus();
}
async function codeDefine() {
  const ed = document.getElementById("code-editor");
  const res = document.getElementById("code-result");
  res.className = "code-result"; res.textContent = "defining…";
  const r = await evalSource(ed.value);
  if (r.error) {
    res.className = "code-result err";
    res.textContent = `✗ ${r.error.kind}: ${r.error.message}`;
  } else {
    const v = r.value || {};
    res.className = "code-result ok flash";
    res.textContent = v.type === "defined"
      ? `✓ ${v.defined} redefined — now at generation ${v.gen}`
      : `✓ ${JSON.stringify(v)}`;
    if (state.view === "detail") setTimeout(refreshDetail, 60);
  }
}
document.getElementById("code-editor").addEventListener("input", (e) => {
  if (state.codeDraft) state.codeDraft.text = e.target.value;
});
document.getElementById("code-define").onclick = codeDefine;
document.getElementById("code-close").onclick = () => document.getElementById("code-panel").classList.add("collapsed");
document.getElementById("code-editor").addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); codeDefine(); }
});

// ---------- global search ----------
async function globalSearch(q) {
  q = q.trim();
  const m = /^([A-Za-z_][\w<>,]*)#(\d+)$/.exec(q);
  if (m) { openDetail(m[1], +m[2], 0, true); return; }
  // cross-type substring over instances(); jump to first hit
  for (const t of state.schema) {
    const r = await evalSource(`${t.name}.instances(filter: "", offset: 0, limit: 200)`);
    const hit = (r.value?.items || []).find((it) => JSON.stringify(it.fields).toLowerCase().includes(q.toLowerCase()));
    if (hit) { const mm = /#(\d+)$/.exec(hit.ref); openDetail(t.name, +mm[1], hit.generation, true); return; }
  }
}

// ---------- utils / boot ----------
function esc(s) { return String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c])); }

document.addEventListener("keydown", (e) => {
  if (e.key === "`" && document.activeElement.id !== "repl-input" && !/input/i.test(document.activeElement.tagName)) {
    e.preventDefault(); toggleRepl();
  } else if (e.key === "`" && document.activeElement.id === "repl-input" && document.getElementById("repl-input").value === "") {
    e.preventDefault(); toggleRepl();
  } else if (e.key === "Escape") { document.getElementById("repl").classList.add("collapsed"); }
});
function toggleRepl() {
  const r = document.getElementById("repl");
  r.classList.toggle("collapsed");
  if (!r.classList.contains("collapsed")) document.getElementById("repl-input").focus();
}
document.getElementById("repl-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && e.target.value.trim()) { replSubmit(e.target.value.trim()); e.target.value = ""; }
});
document.getElementById("type-search").addEventListener("input", renderTypeList);
document.getElementById("global-search").addEventListener("keydown", (e) => { if (e.key === "Enter") globalSearch(e.target.value); });
document.getElementById("transcript-toggle").onclick = () => { document.getElementById("transcript").classList.toggle("collapsed"); renderTx(); };
document.getElementById("transcript-close").onclick = () => document.getElementById("transcript").classList.add("collapsed");
window.addEventListener("focus", () => { if (state.view === "index") refreshTypes(); else if (state.pollTimer) { /* immediate */ } });
document.addEventListener("visibilitychange", () => { if (!document.hidden) { refreshTypes(); } });

// boot
goIndex();
refreshTypes();
setInterval(() => { if (!document.hidden) refreshTypes(); }, 500);
