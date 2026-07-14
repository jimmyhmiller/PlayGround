// scry viewer — React rewrite (htm + React 18, no build step).
// Every pane is still pure sugar over the ONE wire op: POST /eval {id,source} -> {id,value|error}.
// Refresh = re-eval on an interval / on focus / after an action. Nothing is pushed.
//
// Why React: the old vanilla viewer rebuilt panes with innerHTML on every 750ms poll and had
// to hand-capture/restore open method cards, typed args, focus and selection so the refresh
// didn't wipe the form out from under you. Here that state lives in React state/refs inside
// keyed components, so polls only ever setState *data* — open cards, in-flight results, typed
// text, focus, selection and scroll all survive a poll BY CONSTRUCTION.
"use strict";

const { useState, useRef, useEffect, useCallback, useContext, useMemo, createContext } = React;
const { useSyncExternalStore } = React;
const html = htm.bind(React.createElement);

// ===================== the one wire op + cross-cutting stores =====================
// conn + transcript update on *every* eval (many per second). We keep them in tiny external
// stores so the whole App tree does not re-render on each request — only the two subscribed
// widgets (ConnIndicator, TranscriptDrawer) do.
function makeStore(initial) {
  let value = initial;
  const subs = new Set();
  return {
    get: () => value,
    set: (v) => { value = v; subs.forEach((f) => f()); },
    subscribe: (f) => { subs.add(f); return () => subs.delete(f); },
  };
}
function useStore(store) { return useSyncExternalStore(store.subscribe, store.get); }

const connStore = makeStore("connecting");
const txStore = makeStore([]);          // [{source, resp, isErr, t}]
const detailBus = makeStore(0);         // bump to ask the open DetailPane to re-fetch now
const bumpDetail = () => detailBus.set(detailBus.get() + 1);

// Phase 10 portal: every eval is routed through `evalBase`. Standalone (a program serving its
// own viewer directly) leaves this "" -> POST /eval, exactly as before. When the viewer is
// served BY the portal and you click into a program card, evalBase becomes "/p/<id>" so the
// SAME panes POST to /p/<id>/eval and the portal reverse-proxies to that program. One app,
// both modes; nothing below evalSource knows the difference.
let evalBase = "";
function setEvalBase(b) { evalBase = b || ""; }

let evalSeq = 0;
function logReq(source) {
  const arr = txStore.get().slice();
  arr.unshift({ source, resp: null, isErr: false, t: Date.now() });
  if (arr.length > 200) arr.length = 200;
  txStore.set(arr);
}
function logResp(source, resp, isErr) {
  const arr = txStore.get().slice();
  const e = arr.find((x) => x.source === source && x.resp === null);
  if (e) { e.resp = resp; e.isErr = isErr; }
  else arr.unshift({ source, resp, isErr, t: Date.now() });
  txStore.set(arr);
}
async function evalSource(source) {
  const id = "e" + (++evalSeq);
  const t0 = performance.now();
  logReq(source);
  try {
    const res = await fetch(evalBase + "/eval", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id, source }),
    });
    const json = await res.json();
    const dt = performance.now() - t0;
    connStore.set(dt < 1200 ? "live" : "slow");
    logResp(source, json, json.error != null);
    return json;
  } catch (e) {
    connStore.set("down");
    const err = { error: { kind: "Transport", message: String(e) } };
    logResp(source, err, true);
    return err;
  }
}

// ===================== console tools =====================
// A small toolbox exposed on the browser console (the viewer is a live REPL, so debugging
// affordances live here, not as chrome). `tools.gc()` forces a full collection on the running
// program and prints what it reclaimed, then refreshes every pane so the drop in live counts is
// visible immediately. It is pure sugar over the one wire op: it POSTs `gc()`.
const toolsRefresh = () => { window.dispatchEvent(new Event("focus")); bumpDetail(); };
async function runGc(source) {
  const r = await evalSource(source);
  if (r.error) { console.error(`tools.${source.replace("()", "")} failed:`, r.error); return r; }
  const v = r.value || {};
  console.log(
    `%cGC (${v.kind}) — freed ${v.freed} instances: ${v.liveBefore} → ${v.liveAfter} live`,
    "color:#7fd4ff;font-weight:bold");
  if (v.byType && v.byType.length) console.table(v.byType);
  else console.log("(no live entity instances yet — nothing to reclaim)");
  toolsRefresh();  // re-poll every pane so the effect shows up now, not on the next tick
  return v;
}
window.tools = {
  gc: () => runGc("gc()"),           // full major collection (both generations)
  minorGc: () => runGc("minorGc()"), // young-only minor collection (leaves the old generation alone)
  refresh: toolsRefresh,
};
console.log("%cscry%c  tools.gc() (major) · tools.minorGc() (young only) · tools.refresh() re-polls — each shows the effect",
  "color:#7fd4ff;font-weight:bold", "color:inherit");

// ===================== poll helper =====================
// Runs fn now and every ms, but never while the tab is hidden (matches the vanilla
// document.hidden guard). Also re-fires on focus + visibility regain. deps re-arm it.
function usePoll(fn, ms, deps) {
  const saved = useRef(fn);
  saved.current = fn;
  useEffect(() => {
    let cancelled = false;
    const tick = () => { if (!document.hidden && !cancelled) saved.current(); };
    tick();
    const id = setInterval(tick, ms);
    const onWake = () => { if (!document.hidden) saved.current(); };
    document.addEventListener("visibilitychange", onWake);
    window.addEventListener("focus", onWake);
    return () => {
      cancelled = true;
      clearInterval(id);
      document.removeEventListener("visibilitychange", onWake);
      window.removeEventListener("focus", onWake);
    };
  }, deps); // eslint-disable-line
}

// ===================== navigation context =====================
// Refs are clickable anywhere a value is rendered; row clicks and breadcrumbs navigate too.
const NavContext = createContext(null);

// ===================== shared value renderer =====================
function ValueView({ v, inline }) {
  if (v == null) return html`<span class="v-void">null</span>`;
  switch (v.type) {
    case "Int": case "Float": return html`<span class="v-number">${String(v.value)}</span>`;
    case "Bool": return html`<span class="v-bool">${String(v.value)}</span>`;
    case "String": return html`<span class="v-string">${JSON.stringify(v.value)}</span>`;
    case "Void": return html`<span class="v-void">void</span>`;
    case "ref": return html`<${RefLink} v=${v} />`;
    case "list": return html`<${CollectionView} v=${v} inline=${inline} />`;
    case "map": return html`<${MapView} v=${v} />`;
    default:
      if (v.case !== undefined) return html`<${EnumView} v=${v} />`;
      if (v.ref !== undefined && v.fields) return html`<${RefLink} v=${{ ...v, class: v.type, summary: v.ref }} />`;
      return html`<span class="v-void">${JSON.stringify(v)}</span>`;
  }
}
function RefLink({ v }) {
  const nav = useContext(NavContext);
  const label = v.summary && v.summary !== v.ref ? `${v.ref} · ${v.summary}` : v.ref;
  return html`<span class="reflink" title=${v.ref + " (gen " + v.generation + ")"}
    onClick=${(e) => { e.stopPropagation(); nav.navigateRef(v); }}>${label}</span>`;
}
function EnumView({ v }) {
  const payload = v.payload && v.payload.length
    ? v.payload.map((p, i) => html`<${React.Fragment} key=${i}>${i ? ", " : ""}<${ValueView} v=${p} /><//>`)
    : null;
  return html`<span><span class="pill">${v.type + "." + v.case}</span>${payload ? html` ${payload}` : ""}</span>`;
}
function CollectionView({ v, inline }) {
  const items = v.items || [];
  let body;
  if (items.length && items[0].type === "ref") {
    body = items.map((it, i) => html`<span class="chip" key=${i}><${ValueView} v=${it} /></span>`);
  } else if (inline === false) {
    body = items.map((it, i) => html`<${React.Fragment} key=${i}>${i ? ", " : ""}<${ValueView} v=${it} /><//>`);
  } else {
    const head = `${v.length} × ${v.elementType}`;
    if (items.length && items.length <= 6) {
      body = html`${head}  [${items.map((it, i) => html`<${React.Fragment} key=${i}>${i ? ", " : ""}<${ValueView} v=${it} /><//>`)}]`;
    } else {
      body = head;
    }
  }
  const more = v.truncated
    ? html`<span class="list-more">  (+${v.length - items.length} more)</span>` : "";
  return html`<span>${body}${more}</span>`;
}
function MapView({ v }) {
  const entries = (v.entries || []).slice(0, 8);
  const more = v.truncated
    ? html`<span class="list-more"> (+${v.length - (v.entries || []).length} more)</span>` : "";
  return html`<span>${`{${v.length} entries} `}${entries.map(([k, val], i) => html`<${React.Fragment} key=${i}>${i ? ", " : ""}<${ValueView} v=${k} />: <${ValueView} v=${val} /><//>`)}${more}</span>`;
}

// ===================== module identity + focus mode helpers (Phase 3) =====================
// Internal identity from here on keys on `qualified` (Phase 2 wire field on every schema()/
// types()/graph()/functions() node) so two modules that declare the same bare class name
// (tests/run/modules_coexist.scry: modpkg_a.shell.Shell / modpkg_b.shell.Shell) never collide in
// a client-side Map keyed by name. `name` stays bare and is used ONLY for display — via
// shortLabel() below, which qualifies just enough of the module path to disambiguate, never more.
const moduleOf = (n) => (n && n.module) || "";
const qualOf = (n) => (n && (n.qualified || n.name)) || "";
// COMPILER GOTCHA (discovered building this phase, see the final report): a fully-qualified
// name works as an ordinary VALUE reference or CONSTRUCTION (`agent.core.CalcTool(label: "x")`
// resolves with no import) but NOT as the receiver of the reflect-static special forms
// `.at()`/`.instance()`/`.instances()`, nor as an enum-variant access (`agent.core.Role.User`) —
// `compile-dotted-call`'s qualified fast path only recognizes exactly `<module>.<member>(...)`,
// one level deep; `<module>.<Class>.<reflectFn>(...)` falls through to plain dotted-call
// resolution, which doesn't understand a dotted (qualified) receiver and errors "unknown
// identifier". The INTENDED, working mechanism for exactly this (07-modules.md §6) is the eval
// module-HEADER: `module <path>\n<bare-name-source>` sets bare-name resolution context for the
// whole eval, and reflect-static forms + enum variants resolve correctly through it. So every
// eval this viewer sends for .at()/.instance()/.instances()/enum-variant access uses a BARE name
// prefixed with this header (built from the resolved schema node's `module`), never a qualified
// receiver. Fragments EMBEDDED inside a larger expression (an entity/enum ArgInput's picked
// value, spliced into a method-invoke source) still emit bare-only and rely on the OUTER
// expression's single header — a value from a DIFFERENT module than the invoked instance's own
// is a flagged, narrow edge case (no shipped example hits it).
function moduleHeader(mod) { return mod ? `module ${mod}\n` : ""; }
// is module `mod` inside focus module `focus` (itself, or a dotted descendant)? no focus = everywhere.
const inFocus = (mod, focus) => !focus || mod === focus || (mod || "").startsWith(focus + ".");
const isStdModule = (path) => path === "std" || (path || "").startsWith("std.");
// shortest trailing run of module segments that disambiguates `node.name` from same-named
// siblings in `all` — e.g. two `Shell`s under modpkg_a.shell / modpkg_b.shell (same last
// segment "shell") fall through to the full path; two under agents.tools / ui.panels settle on
// one segment ("tools.Shell" / "panels.Shell").
function shortLabel(node, all) {
  if (!node) return "";
  const collide = (all || []).filter((n) => n.name === node.name);
  if (collide.length <= 1) return node.name;
  const segs = moduleOf(node).split(".").filter(Boolean);
  for (let k = 1; k <= segs.length; k++) {
    const suffix = segs.slice(-k).join(".");
    const mine = suffix ? suffix + "." + node.name : node.name;
    const clash = collide.some((o) => {
      if (o === node) return false;
      const os = moduleOf(o).split(".").filter(Boolean);
      return os.slice(-k).join(".") === suffix;
    });
    if (!clash) return mine;
  }
  return (moduleOf(node) ? moduleOf(node) + "." : "") + node.name;
}
// flatten a modules() tree (children[]) into path -> node, for O(1) lookup of counts/labels.
function flattenModules(tree) {
  const out = new Map();
  const walk = (n) => { out.set(n.path, n); (n.children || []).forEach(walk); };
  (tree || []).forEach(walk);
  return out;
}
// non-std top-level modules — the "≥2 modules -> show module chrome" threshold (07-modules.md §7):
// a single-module program (the common case, and almost every existing example) must look
// UNCHANGED, so module rings / rail grouping / breadcrumb only switch on once a real second
// (non-std) module exists. std.* alone never trips it (every program imports std for free).
const nonStdTop = (tree) => (tree || []).filter((n) => !isStdModule(n.path));

// hash-route focus (07-modules.md §7: "/p/<id>/m/<dotted.module>" URL-addressable; see final
// report for why this landed as a hash route, not a portal-proxied path).
function parseHashFocus() {
  const m = /#m=([^&]*)/.exec(location.hash);
  return m ? decodeURIComponent(m[1]) : null;
}
function writeHashFocus(path) {
  const h = path ? "#m=" + encodeURIComponent(path) : "";
  const url = location.pathname + location.search + h;
  if (location.hash !== h) history.pushState(null, "", url);
}
// "expand focus to include <module>" (07-modules.md §7): the longest common dotted prefix of
// the current focus and the boundary-chip's module — the smallest ring that covers both. No
// common ancestor -> clear focus entirely (everywhere).
function lcaModule(a, b) {
  if (!a || !b) return null;
  const as = a.split("."), bs = b.split(".");
  const out = [];
  for (let i = 0; i < Math.min(as.length, bs.length) && as[i] === bs[i]; i++) out.push(as[i]);
  return out.length ? out.join(".") : null;
}

// ===================== type rail =====================
// interface-grouped + plain rows for ONE bucket of types (a module's own declarations, or the
// whole flat list in the no-module-chrome case). Identical to the pre-Phase-3 behavior, just
// keyed by `qualified` throughout instead of bare `name` (the CRITICAL rekey — 07-modules.md
// §7): two types with the same bare name in different modules must not collide in React `key`s,
// the active-route check, or the eval source `nav.openTable` builds.
function typeRailRows(types, allTypes, trend, route, ifaceOpen, setIfaceOpen, nav, filt) {
  const byIface = {};
  const plain = [];
  for (const t of types) {
    if (t.implements && t.implements.length) {
      for (const i of t.implements) (byIface[i] = byIface[i] || []).push(t);
    } else plain.push(t);
  }
  const shown = new Set();
  const rows = [];
  for (const iface of Object.keys(byIface).sort()) {
    if (filt && !iface.toLowerCase().includes(filt) && !byIface[iface].some((t) => t.name.toLowerCase().includes(filt))) continue;
    const total = byIface[iface].reduce((a, t) => a + t.liveCount, 0);
    const open = !!ifaceOpen[iface];
    byIface[iface].forEach((t) => shown.add(t.qualified));
    rows.push(html`
      <li class="iface-group" key=${"iface:" + iface}>
        <div class="type-row iface" onClick=${() => setIfaceOpen(iface, !open)}>
          <span class=${"caret" + (open ? " open" : "")}>▶</span>
          <span class="tname">${iface + " ‹interface›"}</span>
          <span class="count">${total + " live"}</span>
        </div>
        ${open ? html`<div class="iface-children">
          ${byIface[iface].map((t) => html`<${TypeRow} key=${t.qualified} t=${t} label=${shortLabel(t, allTypes)} trend=${trend[t.qualified] || ""} active=${route.view === "table" && route.typeName === t.qualified} onOpen=${() => nav.openTable(t.qualified)} />`)}
        </div>` : ""}
      </li>`);
  }
  for (const t of plain) {
    if (shown.has(t.qualified)) continue;
    if (filt && !t.name.toLowerCase().includes(filt)) continue;
    rows.push(html`<li key=${t.qualified}><${TypeRow} t=${t} label=${shortLabel(t, allTypes)} trend=${trend[t.qualified] || ""} active=${route.view === "table" && route.typeName === t.qualified} onOpen=${() => nav.openTable(t.qualified)} /></li>`);
  }
  return rows;
}
// one module's rail group: collapsible, click-to-focus header showing the modules() aggregate
// counts, own types (iface-grouped) then nested child-module groups — the modules() TREE shape
// (07-modules.md §7 item 3). Skipped entirely if it (and its whole subtree) has no types at all.
function RailModuleGroup({ node, byModule, allTypes, trend, route, ifaceOpen, setIfaceOpen, nav, filt, collapsed, toggle, setFocus, depth }) {
  const own = byModule.get(node.path) || [];
  const kids = node.children || [];
  const hasAny = own.length > 0 || kids.some((c) => moduleHasTypes(c, byModule));
  if (!hasAny) return null;
  const open = !collapsed[node.path];
  const faded = isStdModule(node.path);
  return html`
    <li class=${"mod-group" + (faded ? " faded" : "")} key=${"mod:" + node.path}>
      <div class="type-row mod" onClick=${() => toggle(node.path)}>
        <span class=${"caret" + (open ? " open" : "")}>▶</span>
        <span class="mname" title=${"focus " + node.path}
              onClick=${(e) => { e.stopPropagation(); setFocus(node.path); }}>${node.name}</span>
        <span class="count">${node.liveCount} live · ${node.typeCount} types</span>
      </div>
      ${open ? html`<div class="mod-children" style=${{ marginLeft: (depth || 0) < 3 ? "10px" : "0" }}>
        <ul class="type-list-inner">
          ${typeRailRows(own, allTypes, trend, route, ifaceOpen, setIfaceOpen, nav, filt)}
          ${kids.map((c) => html`<${RailModuleGroup} key=${c.path} node=${c} byModule=${byModule} allTypes=${allTypes}
              trend=${trend} route=${route} ifaceOpen=${ifaceOpen} setIfaceOpen=${setIfaceOpen} nav=${nav} filt=${filt}
              collapsed=${collapsed} toggle=${toggle} setFocus=${setFocus} depth=${(depth || 0) + 1} />`)}
        </ul>
      </div>` : ""}
    </li>`;
}
function moduleHasTypes(node, byModule) {
  if ((byModule.get(node.path) || []).length) return true;
  return (node.children || []).some((c) => moduleHasTypes(c, byModule));
}

function TypeRail({ schema, trend, route, ifaceOpen, setIfaceOpen, modTree, focus, setFocus, everywhere }) {
  const nav = useContext(NavContext);
  const [search, setSearch] = useState("");
  const [collapsed, setCollapsedState] = useState({});
  const toggle = useCallback((path) => setCollapsedState((s) => ({ ...s, [path]: !s[path] })), []);
  const filt = search.toLowerCase();

  // focus (and no "everywhere" override) scopes the whole rail to the subtree — no module
  // headers needed once you're already inside one (07-modules.md §7 item 4).
  const scoped = focus && !everywhere ? schema.filter((t) => inFocus(moduleOf(t), focus)) : schema;
  const showModules = !focus && nonStdTop(modTree).length >= 2;

  let body;
  if (showModules) {
    const byModule = new Map();
    for (const t of schema) { const k = moduleOf(t); (byModule.get(k) || byModule.set(k, []).get(k)).push(t); }
    body = modTree.map((node) => html`<${RailModuleGroup} key=${node.path} node=${node} byModule=${byModule} allTypes=${schema}
        trend=${trend} route=${route} ifaceOpen=${ifaceOpen} setIfaceOpen=${setIfaceOpen} nav=${nav} filt=${filt}
        collapsed=${collapsed} toggle=${toggle} setFocus=${setFocus} depth=${0} />`);
  } else {
    body = typeRailRows(scoped, schema, trend, route, ifaceOpen, setIfaceOpen, nav, filt);
  }

  return html`
    <nav id="rail">
      <div class="rail-head">
        <span>types</span>
        ${focus ? html`<div class="rail-focus-chip">
            <span class="rfc-path">${focus}</span>
            <button class="ghost-btn rfc-clear" onClick=${() => setFocus(null)}>✕ clear focus</button>
          </div>` : ""}
        <input id="type-search" type="text" placeholder="filter…" spellcheck="false" autocomplete="off"
               value=${search} onInput=${(e) => setSearch(e.target.value)} />
      </div>
      <ul id="type-list">${body}</ul>
    </nav>`;
}
function TypeRow({ t, label, trend, active, onOpen }) {
  const arrow = trend === "up" ? "▲" : trend === "down" ? "▼" : "";
  return html`
    <div class=${"type-row" + (active ? " active" : "")} onClick=${onOpen}>
      <span class="tname">${label || t.name}</span>
      <span class=${"count" + (trend ? " changed" : "")}>${t.liveCount + " live"}</span>
      <span class=${"trend " + trend}>${arrow}</span>
    </div>`;
}

// ===================== index pane =====================
function IndexPane() {
  return html`
    <div>
      <div class="pane-title">Entity types</div>
      <div class="pane-sub">Pick a type from the rail to browse its live instances. Counts refresh automatically.</div>
    </div>`;
}

// ===================== instance table =====================
// When a type has no live instances (static inspect / a static project / just not created yet),
// show its CLASS SCHEMA (fields + method signatures) instead of an empty table — so you can
// always inspect a class statically.
function TablePaneSchema({ sc, name, error }) {
  if (error && error.kind !== "StaticInspection")
    return html`<div class="invoke-result invoke-error">${error.kind}: ${error.message}</div>`;
  if (!sc) return html`<div class="empty">no live instances, and no schema for ${name}.</div>`;
  return html`
    <div class="detail-section">
      <div class="pane-sub">no live instances — showing the class schema</div>
      ${sc.fields && sc.fields.length ? html`
        <h3>fields</h3>
        <div class="field-grid">
          ${sc.fields.map((f) => html`<${React.Fragment} key=${f.name}>
            <div class="fcell fname">${f.name}<span class="ftype">${cleanType(f.type)}</span></div>
            <div class="fcell fval"><span class="v-void">⟵ runtime</span></div>
          <//>`)}
        </div>` : ""}
      ${sc.methods && sc.methods.length ? html`
        <h3 class="sec-gap">methods</h3>
        ${sc.methods.map((m) => html`
          <div class="method" key=${m.name}>
            <div class="method-head static">
              <span class="method-sig">${m.name}(${(m.params || []).map((p) => `${p.name}: ${cleanType(p.type)}`).join(", ")}) <span class="mret">→ ${m.returns}</span></span>
            </div>
          </div>`)}` : ""}
    </div>`;
}

function TablePane({ name, schema }) {
  const nav = useContext(NavContext);
  const sc = schema.find((t) => t.qualified === name);
  const label = sc ? shortLabel(sc, schema) : name;
  const [filterInput, setFilterInput] = useState("");   // what you type
  const [applied, setApplied] = useState("");           // what the poll queries
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  // reset the filter when the type changes
  useEffect(() => { setFilterInput(""); setApplied(""); setData(null); setError(null); }, [name]);

  usePoll(async () => {
    const f = applied.replace(/"/g, '\\"');
    const bare = sc ? sc.name : name;
    const src = moduleHeader(sc && sc.module) + `${bare}.instances(filter: "${f}", offset: 0, limit: 200)`;
    const r = await evalSource(src);
    if (r.error) { setError(r.error); setData(null); }
    else { setError(null); setData(r.value); }
  }, 750, [name, applied, sc && sc.module, sc && sc.name]);

  const items = data ? (data.items || []) : [];
  const cols = sc ? sc.fields.map((f) => f.name) : (items[0] ? Object.keys(items[0].fields) : []);
  const meta = data ? `${data.length} live${data.truncated ? " (showing " + items.length + ")" : ""}` : "";

  return html`
    <div>
      <div class="pane-title">${label}</div>
      <div class="pane-sub">${sc ? `${sc.fields.length} fields · ${sc.methods.length} methods` : ""}${sc && sc.module ? html` · <span class="mod-tag">${sc.module}</span>` : ""}</div>
      <div class="tbl-tools">
        <input class="filter-box" spellcheck="false" autocomplete="off"
               placeholder=${'filter, e.g.  name == "coder"  or  status contains "run"'}
               value=${filterInput}
               onInput=${(e) => setFilterInput(e.target.value)}
               onKeyDown=${(e) => { if (e.key === "Enter") setApplied(filterInput); }} />
        <div class="tbl-meta">${meta}</div>
      </div>
      ${!items.length
        ? (applied
            ? html`<div class="empty">no instances match the filter.</div>`
            : html`<${TablePaneSchema} sc=${sc} name=${name} error=${error} />`)
        : html`
            <table class="itable">
              <thead><tr><th>id</th>${cols.map((c) => html`<th key=${c}>${c}</th>`)}</tr></thead>
              <tbody>
                ${items.map((it) => {
                  const m = /#(\d+)$/.exec(it.ref);
                  const slot = m ? +m[1] : 0;
                  return html`
                    <tr key=${it.ref} onClick=${() => nav.openDetail(name, slot, it.generation, true)}>
                      <td class="col-id">${it.ref}</td>
                      ${cols.map((c) => html`<td key=${c}>${it.fields[c] ? html`<${ValueView} v=${it.fields[c]} />` : "—"}</td>`)}
                    </tr>`;
                })}
              </tbody>
            </table>`}
    </div>`;
}

// ===================== instance detail =====================
function DetailPane({ cls, slot, gen, schema, onEditSource }) {
  const sc = schema.find((t) => t.qualified === cls);
  const dispCls = sc ? shortLabel(sc, schema) : cls;
  const [inst, setInst] = useState(null);
  const [error, setError] = useState(null);
  const [actions, setActions] = useState([]);   // this type's declared actions (actions())
  const prevFields = useRef(null);      // previous poll's fields, for flash diffing
  const flashKeys = useRef({});         // field name -> nonce; bump => value cell remounts => flash replays

  // Program-declared actions are static (the desugar is fixed at build) — fetch once per type.
  // actions()'s "target" is a BARE name (unqualified in the wire, unlike schema/types nodes), so
  // match against sc.name (the resolved type's own bare name), never the qualified `cls`.
  useEffect(() => {
    let live = true;
    const bareTarget = sc ? sc.name : cls;
    evalSource("actions()").then((r) => {
      if (live && r.value && r.value.actions) setActions(r.value.actions.filter((a) => a.target === bareTarget));
    });
    return () => { live = false; };
  }, [cls, sc && sc.name]);

  const fetchDetail = useCallback(async () => {
    const bare = sc ? sc.name : cls;
    const src = moduleHeader(sc && sc.module) + `${bare}.at(${slot}, ${gen})`;
    const r = await evalSource(src);
    if (r.error) { setError(r.error); return; }
    setError(null);
    const next = r.value;
    const prev = prevFields.current;
    if (prev) {
      for (const [k, val] of Object.entries(next.fields)) {
        if (JSON.stringify(prev[k]) !== JSON.stringify(val)) {
          flashKeys.current[k] = (flashKeys.current[k] || 0) + 1;
        }
      }
    }
    prevFields.current = next.fields;
    setInst(next);
  }, [cls, slot, gen, sc && sc.module, sc && sc.name]);

  // fresh identity => drop the flash baseline so we don't flash the whole record on arrival
  useEffect(() => { prevFields.current = null; flashKeys.current = {}; setInst(null); setError(null); }, [cls, slot, gen]);

  usePoll(fetchDetail, 750, [cls, slot, gen]);

  // let MethodCard / ReplDock ask for an immediate read-back after a mutation
  const fetchRef = useRef(fetchDetail);
  fetchRef.current = fetchDetail;
  useEffect(() => detailBus.subscribe(() => { if (!document.hidden) fetchRef.current(); }), []);

  if (error) {
    return html`<div><div class="pane-title">${dispCls + "#" + slot}</div>
      <div class="invoke-result invoke-error">${error.kind}: ${error.message}</div></div>`;
  }
  if (!inst) return html`<div><div class="pane-title">${dispCls + "#" + slot}</div></div>`;

  const typeOf = (n) => sc ? (sc.fields.find((f) => f.name === n) || {}).type : "";
  const implementsLine = sc && sc.implements && sc.implements.length
    ? html` · implements <span class="impl">${sc.implements.join(", ")}</span>` : "";
  const modLine = sc && sc.module ? html` · <span class="mod-tag">${sc.module}</span>` : "";

  return html`
    <div>
      <div class="pane-title">${dispCls}<span class="pane-title-sub">#${slot}</span></div>
      <div class="pane-sub">
        generation ${inst.generation}${implementsLine}${modLine}
        ${sc ? html`<button class="ghost-btn edit-src" onClick=${() => onEditSource(cls, sc)}>✎ edit source</button>` : ""}
      </div>

      <div class="detail-section">
        <h3>fields</h3>
        <div class="field-grid">
          ${Object.entries(inst.fields).map(([k, val]) => {
            const fk = flashKeys.current[k] || 0;
            return html`
              <${React.Fragment} key=${k}>
                <div class="fcell fname">${k}<span class="ftype">${typeOf(k) || ""}</span></div>
                <div class=${"fcell fval" + (fk ? " flash" : "")} key=${"v" + fk}><${ValueView} v=${val} /></div>
              <//>`;
          })}
        </div>
      </div>

      ${actions.length ? html`
        <div class="detail-section actions-section">
          <h3>actions</h3>
          <div class="action-grid">
            ${actions.map((a) => html`<${ActionCard} key=${a.invoke} cls=${cls} slot=${slot} gen=${gen} a=${a} schema=${schema} />`)}
          </div>
        </div>` : ""}

      ${sc && sc.methods.length ? html`
        <div class="detail-section">
          <h3>methods</h3>
          ${sc.methods.map((m) => html`<${MethodCard} key=${m.name} cls=${cls} slot=${slot} gen=${gen} m=${m} schema=${schema} />`)}
        </div>` : ""}
    </div>`;
}

// An action is a curated, app-blessed affordance: a prominent button (with an inline arg form if it
// has params) that evals the action's hidden synthetic method against THIS instance through the
// normal invoke path, then bumps detailBus so the mutation is read back live. Styled distinct from
// the raw method list — these are "the things a person would want to do here."
function ActionCard({ cls, slot, gen, a, schema }) {
  const [open, setOpen] = useState(false);
  const [args, setArgs] = useState({});
  const [result, setResult] = useState(null);
  const flash = useRef(0);
  const hasParams = a.params.length > 0;
  const allValid = a.params.every((p) => argValid(p, args[p.name], schema));
  const sc = schema.find((t) => t.qualified === cls);

  const doInvoke = useCallback(async () => {
    if (!a.params.every((p) => argValid(p, args[p.name], schema))) return;  // never send bad source
    const argList = a.params.map((p) => literalFor(args[p.name] || "", p.type)).join(", ");
    const bare = sc ? sc.name : cls;
    const src = moduleHeader(sc && sc.module) + `${bare}.at(${slot}, ${gen}).${a.invoke}(${argList})`;
    const r = await evalSource(src);
    flash.current += 1;
    setResult({ ...r, flash: flash.current });
    setTimeout(bumpDetail, 60); // read the mutation back immediately
  }, [args, cls, slot, gen, a, schema, sc && sc.module, sc && sc.name]);

  return html`
    <div class=${"action-card" + (open ? " open" : "")}>
      <button class="action-btn" title=${hasParams ? "fill args, then run" : "run this action"}
              onClick=${() => { if (hasParams) setOpen((o) => !o); else doInvoke(); }}>
        <span class="action-label">${a.label}</span>
        ${hasParams ? html`<span class="action-caret">${open ? "▾" : "▸"}</span>` : ""}
      </button>
      ${hasParams ? html`
        <div class="action-form">
          ${a.params.map((p) => html`<${ArgInput} key=${p.name} param=${p} schema=${schema} active=${open}
              value=${args[p.name]} onChange=${(val) => setArgs((s) => ({ ...s, [p.name]: val }))} onEnter=${doInvoke} />`)}
          <div class="arg-row"><button class="action-run" onClick=${doInvoke} disabled=${!allValid}>run ${a.label}</button></div>
        </div>` : ""}
      ${result ? html`<${InvokeResult} result=${result} key=${result.flash} />` : ""}
    </div>`;
}

// A method card owns its own open/args/result state. That state is what the vanilla viewer
// had to snapshot-and-restore across every poll; here it simply lives in the component, so a
// poll re-rendering DetailPane never touches it.
function MethodCard({ cls, slot, gen, m, schema }) {
  const [open, setOpen] = useState(false);
  const [args, setArgs] = useState({});                 // param name -> string
  const [result, setResult] = useState(null);           // {error|value, flash}
  const flash = useRef(0);
  const allValid = m.params.every((p) => argValid(p, args[p.name], schema));
  const sc = schema.find((t) => t.qualified === cls);

  const doInvoke = useCallback(async () => {
    if (!m.params.every((p) => argValid(p, args[p.name], schema))) return;  // never send bad source
    const argList = m.params.map((p) => literalFor(args[p.name] || "", p.type)).join(", ");
    const bare = sc ? sc.name : cls;
    const src = moduleHeader(sc && sc.module) + `${bare}.at(${slot}, ${gen}).${m.name}(${argList})`;
    const r = await evalSource(src);
    flash.current += 1;
    setResult({ ...r, flash: flash.current });
    setTimeout(bumpDetail, 60); // read the mutation back immediately
  }, [args, cls, slot, gen, m, schema, sc && sc.module, sc && sc.name]);

  const params = m.params.map((p) => `${p.name}: ${p.type}`).join(", ");

  return html`
    <div class=${"method" + (open ? " open" : "")}>
      <div class="method-head" onClick=${() => setOpen((o) => !o)}>
        <span class="method-sig">${m.name}(${params}) <span class="mret">→ ${m.returns}</span></span>
        <button class="invoke-btn" onClick=${(e) => { e.stopPropagation(); setOpen(true); if (!m.params.length || allValid) doInvoke(); }}>invoke</button>
      </div>
      <div class="method-body">
        ${m.params.map((p) => html`<${ArgInput} key=${p.name} param=${p} schema=${schema} active=${open}
            value=${args[p.name]} onChange=${(val) => setArgs((a) => ({ ...a, [p.name]: val }))} onEnter=${doInvoke} />`)}
        ${m.params.length ? html`<div class="arg-row"><button class="invoke-btn" onClick=${doInvoke} disabled=${!allValid}>run</button></div>` : ""}
        ${result ? html`<${InvokeResult} result=${result} key=${result.flash} />` : ""}
      </div>
    </div>`;
}
// A safe string for any value (never throws). Used as the render fallback.
function safeStringify(v) {
  try { return typeof v === "string" ? v : JSON.stringify(v); }
  catch { return String(v); }
}
// A render error boundary: if a child render throws (a malformed value shape, etc.), show the
// fallback instead of unmounting the whole app (the reported "white-screen" bug). Each invoke
// result is keyed by `flash`, so a fresh invoke remounts the boundary clean.
class RenderBoundary extends React.Component {
  constructor(props) { super(props); this.state = { err: null }; }
  static getDerivedStateFromError(err) { return { err }; }
  render() { return this.state.err ? this.props.fallback(this.state.err) : this.props.children; }
}
// ValueView guarded by the boundary: a throw anywhere in the (possibly deep) value render
// degrades to a plain safe string rather than crashing the viewer.
function SafeValue({ v }) {
  return html`<${RenderBoundary} fallback=${() => html`<span class="v-void">${safeStringify(v)}</span>`}>
    <${ValueView} v=${v} />
  <//>`;
}
// Render an invoke's result or error. Bulletproof: no error/value shape (missing kind/message,
// odd `type`, null, a non-array trace, an unexpected error primitive) may throw during render —
// the scry server is uncrashable and always returns a clean TypeError, so a bad eval must show
// inline and leave the form fully usable.
function InvokeResult({ result }) {
  const r = result || {};
  if (r.error != null) {
    const err = (r.error && typeof r.error === "object") ? r.error : { message: safeStringify(r.error) };
    const kind = err.kind != null ? String(err.kind) : "Error";
    const message = err.message != null ? String(err.message) : safeStringify(err);
    const trace = Array.isArray(err.trace) && err.trace.length
      ? html`<div class="etrace">${err.trace.map((f) => (f && f.type != null) ? `${f.type}.${f.method} (line ${f.line})` : safeStringify(f)).join(" › ")}</div>` : "";
    return html`<div class="invoke-result flash invoke-error">${kind}: ${message}${trace}</div>`;
  }
  return html`<div class="invoke-result flash"><${SafeValue} v=${r.value} /></div>`;
}
function literalHint(type) {
  if (type === "String") return '"text"';
  if (type === "Int") return "0"; if (type === "Float") return "0.0"; if (type === "Bool") return "true";
  return "expression";
}
// Turn a user's raw text into a well-typed literal for the param's type. NEVER emit a bare
// word (which the compiler would read as an unknown variable) — quote/parse/qualify by type.
function literalFor(v, type) {
  v = (v || "").trim();
  const t = (type || "").trim();
  if (t === "String") return v.startsWith('"') ? v : JSON.stringify(v);   // always a real string
  if (t === "Bool") return v === "true" || v === "1" ? "true" : "false";
  if (t === "Int") { const n = parseInt(v, 10); return Number.isFinite(n) ? String(n) : "0"; }
  if (t === "Float") { const f = parseFloat(v); return Number.isFinite(f) ? String(f) : "0.0"; }
  if (v === "") return "0";
  // an instance reference typed as "Agent#3" (ArgInput may hand in a QUALIFIED "a.b.Agent#3" —
  // strip to the bare last segment: .at() is a reflect-static form that only accepts a bare
  // receiver, see the moduleHeader() comment; this fragment is spliced into a larger expression
  // that carries ITS OWN module header, set from the instance being invoked on).
  const rm = /^([A-Za-z_][\w.]*)#(\d+)$/.exec(v);
  if (rm) return `${rm[1].split(".").pop()}.at(${rm[2]}, 0)`;
  // enum variant emitted bare ("AgentStatus.Running") for the same reflect-static-form reason;
  // the ArgInput widgets below emit bare enum variants / (possibly qualified) Type#slot ids /
  // quoted strings, so literalFor is just the FINAL mapper — a bare word only survives for the
  // free-text fallback types.
  return v;
}

// ===================== typed argument entry (ArgInput) =====================
// The proper fix for "unknown identifier": a bare word typed for an entity/enum param used to
// desugar to a raw expression the compiler read as an unbound variable. Instead of one plain
// <input> per param, we render a TYPE-AWARE widget so the form always emits well-typed source,
// and literalFor stays the single final mapper (bool→as-is, enum "Type.Variant"→as-is,
// "Type#slot"→Type.at(slot,0), number→as-is, string→quoted).

// Classify a declared param type against the schema. cleanType strips a `ref:`/`list:` prefix; a
// generic (`List<Card>`) or an explicit `list:`/`map:` type is never a picker — free text.
function classifyParam(type, schema) {
  const raw = String(type || "").trim();
  if (/^(list|map):/i.test(raw) || /[<>]/.test(raw)) return { kind: "text" };
  const t = cleanType(raw);
  if (t === "Bool") return { kind: "bool" };
  if (t === "Int") return { kind: "int" };
  if (t === "Float") return { kind: "float" };
  if (t === "String") return { kind: "string" };
  // param/field/ctor `type` strings are BARE display types (ast-ty-str on the wire — no
  // qualified twin, unlike schema/types nodes), so this lookup stays bare-name. Flagged
  // limitation: two visible types sharing this bare name is ambiguous here (rare in practice —
  // see the final Phase 3 report).
  const node = (schema || []).find((n) => n.name === t);
  if (node) {
    if (node.kind === "enum") return { kind: "enum", node };
    if (node.kind === "interface") return { kind: "interface", node };
    if (node.kind === "class" || node.kind === "object") return { kind: "entity", node };
  }
  return { kind: "text" };  // unknown / generic -> free-text fallback
}
// Is the current string a VALID value for this param? Blocks invoke on bad/missing input:
// numbers must parse; a required entity/enum must have a selection. String/text/bool are always ok.
function argValid(param, value, schema) {
  const { kind } = classifyParam(param.type, schema);
  const v = (value || "").trim();
  if (kind === "int") return /^[+-]?\d+$/.test(v);
  if (kind === "float") return v !== "" && Number.isFinite(Number(v));
  if (kind === "enum" || kind === "entity" || kind === "interface") return v !== "";
  return true;
}

// The type-aware widget for ONE param. `onChange` stores a STRING literalFor turns into valid
// source; `active` gates the entity instance poll to open cards only; `onEnter` invokes.
function ArgInput({ param, value, onChange, schema, active, onEnter, depth }) {
  const info = classifyParam(param.type, schema);
  const label = html`<label>${param.name}: ${cleanType(param.type)}</label>`;
  const enterKey = (e) => { if (e.key === "Enter" && onEnter) onEnter(); };
  if (info.kind === "bool") {
    return html`<div class="arg-row">${label}
      <select class="arg-select" value=${value || "false"} onChange=${(e) => onChange(e.target.value)}>
        <option value="true">true</option><option value="false">false</option>
      </select></div>`;
  }
  if (info.kind === "enum") {
    const variants = info.node.variants || [];
    // enum variant emitted BARE (embedded fragment, relies on the enclosing eval's module
    // header — see moduleHeader()'s comment; a qualified receiver breaks this reflect-like form
    // the same way it breaks .at()/.instances()).
    return html`<div class="arg-row">${label}
      <select class="arg-select" value=${value || ""} onChange=${(e) => onChange(e.target.value)}>
        <option value="">— select —</option>
        ${variants.map((vr) => html`<option key=${vr.name} value=${`${info.node.name}.${vr.name}`}>${vr.name}</option>`)}
      </select></div>`;
  }
  if (info.kind === "entity" || info.kind === "interface") {
    // resolve to QUALIFIED type identifiers (schema is available here) so the eval sources
    // EntityArgInput builds (.instances()/.at()/construction) are collision-safe even when an
    // interface's implementors — or the plain entity itself — share a bare name across modules.
    const qualFor = (bareName) => {
      const n = (schema || []).find((x) => x.name === bareName);
      return n ? n.qualified : bareName;
    };
    const types = info.kind === "interface"
      ? ((info.node.implementors && info.node.implementors.length) ? info.node.implementors.map(qualFor) : [info.node.qualified || info.node.name])
      : [info.node.qualified || info.node.name];
    return html`<${EntityArgInput} label=${label} types=${types}
      value=${value} onChange=${onChange} active=${active} onEnter=${onEnter}
      schema=${schema} depth=${depth || 0} />`;
  }
  // Int / Float / String / (list/map/unknown) -> a text input; numbers are validated inline.
  const numeric = info.kind === "int" || info.kind === "float";
  const bad = numeric && (value || "").trim() !== "" && !argValid(param, value, schema);
  return html`<div class="arg-row">${label}
    <input class=${"arg-input" + (bad ? " invalid" : "")}
           inputMode=${info.kind === "int" ? "numeric" : info.kind === "float" ? "decimal" : undefined}
           placeholder=${literalHint(param.type)} value=${value || ""}
           onInput=${(e) => onChange(e.target.value)} onKeyDown=${enterKey} />
    ${bad ? html`<span class="arg-err">not ${info.kind === "int" ? "an int" : "a float"}</span>` : ""}</div>`;
}
// Deepest CONSTRUCTOR nesting we allow through "+ create new" before we stop offering it and
// force a live-instance pick — an entity whose ctor references its own type (or a cycle) would
// otherwise recurse forever. 3 levels is plenty for the demo entity graphs.
const MAX_CREATE_DEPTH = 3;

// Entity/interface picker: a <select> of LIVE instances (Type#slot · summary) fetched from
// <Type>.instances(...) PLUS, per concrete type, a "+ create new <Type>" option. For an
// interface, every implementor is queried (and each is offered as a constructible concrete
// type). Refetches when the card opens (active), on focus/mousedown, and on a slow interval.
// Choosing "create new" reveals an inline CONSTRUCTOR form: one ArgInput per ctor param (from
// schema()'s `ctor`), RECURSIVELY (an entity ctor param gets its own pick-or-create). While the
// sub-form is incomplete the emitted value is "" so the parent's run button stays disabled
// (argValid). When valid it emits a construction expression `Type(p: v, ...)` that literalFor
// passes through unchanged. Nesting is capped at MAX_CREATE_DEPTH.
function EntityArgInput({ label, types, value, onChange, active, onEnter, schema, depth }) {
  depth = depth || 0;
  const canCreate = depth < MAX_CREATE_DEPTH;
  const [opts, setOpts] = useState(null);   // null = not loaded; [] = loaded, none live
  const [createType, setCreateType] = useState(null); // null = pick mode; else the Type being built
  const [ctorArgs, setCtorArgs] = useState({});       // ctor param name -> string
  const key = types.join(",");
  // `types` are QUALIFIED identifiers now (ArgInput resolves them); label with the short
  // (disambiguated-if-needed) display name instead of the full qualified string.
  const shortFor = (tn) => { const n = (schema || []).find((x) => x.qualified === tn); return n ? shortLabel(n, schema) : tn; };
  const fetchNow = useCallback(async () => {
    const all = [];
    for (const tn of types) {
      // `tn` is qualified for CLIENT identity, but .instances() needs a bare receiver + a
      // `module` eval header (see the moduleHeader() comment — reflect-static forms don't
      // understand a qualified receiver, only ordinary value/construction positions do).
      const node = (schema || []).find((x) => x.qualified === tn);
      const bare = node ? node.name : tn;
      const src = moduleHeader(node && node.module) + `${bare}.instances(filter: "", offset: 0, limit: 50)`;
      const r = await evalSource(src);
      const items = (r && r.value && r.value.items) || [];
      const disp = shortFor(tn);
      for (const it of items) {
        const m = /#(\d+)$/.exec(it.ref); const slot = m ? +m[1] : 0;
        const s = instSummary(it);
        all.push({ value: `${tn}#${slot}`, label: `${disp}#${slot}` + (s ? ` · ${s}` : "") });
      }
    }
    setOpts(all);
  }, [key]); // eslint-disable-line
  useEffect(() => {
    if (!active || createType) return;   // don't poll live instances while building a new one
    fetchNow();
    const id = setInterval(fetchNow, 1500);
    return () => clearInterval(id);
  }, [active, createType, fetchNow]);

  // ctor param list for the type currently being constructed (from schema()'s `ctor`).
  const ctorNode = createType ? (schema || []).find((n) => n.qualified === createType) : null;
  const ctorParams = (ctorNode && ctorNode.ctor) || [];
  // Continuously push the construction expression (or "" while incomplete) up to the parent.
  useEffect(() => {
    if (!createType) return;
    const ok = ctorParams.every((p) => argValid(p, ctorArgs[p.name], schema));
    if (ok) {
      const argList = ctorParams.map((p) => `${p.name}: ${literalFor(ctorArgs[p.name] || "", p.type)}`).join(", ");
      onChange(`${createType}(${argList})`);
    } else {
      onChange("");
    }
  }, [createType, JSON.stringify(ctorArgs)]); // eslint-disable-line

  const enterKey = (e) => { if (e.key === "Enter" && onEnter) onEnter(); };
  const onSelect = (val) => {
    const cm = /^__create__:(.+)$/.exec(val);
    if (cm) { setCreateType(cm[1]); setCtorArgs({}); onChange(""); return; }
    setCreateType(null);
    onChange(val);
  };
  const createOpts = canCreate
    ? types.map((tn) => html`<option key=${"__c:" + tn} value=${"__create__:" + tn}>+ create new ${shortFor(tn)}</option>`)
    : "";

  // ---- create mode: inline constructor form ----
  if (createType) {
    return html`<div class="arg-row create-form">${label}
      <div class="create-head">
        <span class="create-title">new ${shortFor(createType)}</span>
        <button type="button" class="create-back" onClick=${() => { setCreateType(null); onChange(""); }}>use existing</button>
      </div>
      ${ctorParams.length === 0 ? html`<span class="arg-note">no constructor arguments</span>` : ""}
      ${ctorParams.map((p) => html`<${ArgInput} key=${p.name} param=${p} schema=${schema} active=${active}
          depth=${depth + 1} value=${ctorArgs[p.name]}
          onChange=${(v) => setCtorArgs((s) => ({ ...s, [p.name]: v }))} onEnter=${onEnter} />`)}
    </div>`;
  }

  // ---- pick mode ----
  // No live instances AND can't create here (max depth) -> free-text id fallback.
  if (opts !== null && opts.length === 0 && !canCreate) {
    return html`<div class="arg-row">${label}
      <input class="arg-input" placeholder="Type#slot" value=${value || ""}
             onFocus=${fetchNow} onInput=${(e) => onChange(e.target.value)} onKeyDown=${enterKey} />
      <span class="arg-note">max nesting — type an id</span></div>`;
  }
  return html`<div class="arg-row">${label}
    <select class="arg-select" value=${value || ""}
            onMouseDown=${fetchNow} onFocus=${fetchNow} onChange=${(e) => onSelect(e.target.value)}>
      <option value="">${opts === null ? "loading…" : "— select —"}</option>
      ${(opts || []).map((o) => html`<option key=${o.value} value=${o.value}>${o.label}</option>`)}
      ${createOpts}
    </select>
    ${(opts !== null && opts.length === 0 && canCreate) ? html`<span class="arg-note">no live instances — create one</span>` : ""}</div>`;
}

// ===================== breadcrumbs =====================
function Breadcrumbs({ crumbs }) {
  const nav = useContext(NavContext);
  return html`
    <div id="breadcrumbs">
      ${crumbs.map((c, i) => html`
        <${React.Fragment} key=${i}>
          ${i ? html`<span class="sep">›</span>` : ""}
          <span class=${"crumb" + (c.target ? "" : " static")}
                onClick=${c.target ? () => nav.goCrumb(c.target) : undefined}>${c.label}</span>
        <//>`)}
    </div>`;
}

// ===================== repl dock =====================
// flatten a modules() tree into a depth-indented option list for the REPL module picker.
function flattenModuleOptions(tree, depth) {
  depth = depth || 0;
  const out = [];
  for (const n of (tree || [])) {
    out.push({ path: n.path, name: n.name, depth });
    out.push(...flattenModuleOptions(n.children, depth + 1));
  }
  return out;
}
function ReplDock({ open, setOpen, route, modTree, schema }) {
  const [entries, setEntries] = useState([]);       // {expr, out?, error?}
  const [input, setInput] = useState("");
  // module picker (07-modules.md §7 item 5): "" = default (entry module, i.e. no header).
  // Selecting one prepends a `module <path>` header to every eval this dock sends.
  const [replModule, setReplModule] = useState("");
  const scrollRef = useRef(null);
  const inputRef = useRef(null);
  const modOptions = useMemo(() => flattenModuleOptions(modTree), [modTree]);

  useEffect(() => { if (open) inputRef.current && inputRef.current.focus(); }, [open]);
  useEffect(() => { if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight; }, [entries]);

  const submit = useCallback((src) => {
    let source = src;
    let header = replModule ? moduleHeader(replModule) : "";
    if (route.view === "detail" && route.ref) {
      const { class: c, slot, gen } = route.ref;
      // `self` binds to a BARE-name reflect-static .at() (see moduleHeader()'s comment) — the
      // instance's OWN module always wins the header so `self` itself is guaranteed to resolve,
      // even if the module picker is set to something else.
      const sc = (schema || []).find((t) => t.qualified === c);
      const bare = sc ? sc.name : c;
      source = src.replace(/\bself\b/g, `${bare}.at(${slot}, ${gen})`);
      header = moduleHeader(sc && sc.module);
    }
    source = header + source;
    const idx = entries.length;
    setEntries((es) => [...es, { expr: src, pending: true }]);
    evalSource(source).then((r) => {
      setEntries((es) => es.map((e, i) => i === idx ? { expr: src, error: r.error, value: r.value } : e));
      if (route.view === "detail") setTimeout(bumpDetail, 60);
    });
  }, [entries.length, route, replModule, schema]);

  const ctx = route.view === "detail" && route.ref ? `self = ${route.ref.class}#${route.ref.slot}` : "";

  return html`
    <section id="repl" class=${open ? "" : "collapsed"}>
      <div class="repl-head">
        <span class="repl-title">repl</span>
        <span class="repl-context">${ctx}</span>
        ${modOptions.length ? html`<select class="repl-mod-select" title="module context — prepends 'module <path>' to every eval"
              value=${replModule} onChange=${(e) => setReplModule(e.target.value)}>
            <option value="">module: (entry)</option>
            ${modOptions.map((m) => html`<option key=${m.path} value=${m.path}>${"·".repeat(m.depth)} ${m.path}</option>`)}
          </select>` : ""}
        <span class="repl-hint">bound to <code>self</code> · press <kbd>\`</kbd> to toggle</span>
      </div>
      <div id="repl-scroll" ref=${scrollRef}>
        ${entries.map((e, i) => html`
          <div class="repl-entry" key=${i}>
            <div class="repl-expr">${e.expr}</div>
            <div class=${"repl-out" + (e.error ? " err" : "")}>
              ${e.pending ? "…" : e.error ? `${e.error.kind}: ${e.error.message}` : html`<${ValueView} v=${e.value} inline=${false} />`}
            </div>
          </div>`)}
      </div>
      <div class="repl-input-row">
        <span class="repl-prompt">›</span>
        <input id="repl-input" ref=${inputRef} type="text" spellcheck="false" autocomplete="off"
               placeholder="self.conversation.size()  —  or any Scry expression"
               value=${input}
               onInput=${(e) => setInput(e.target.value)}
               onKeyDown=${(e) => {
                 if (e.key === "Enter" && input.trim()) { submit(input.trim()); setInput(""); }
                 else if (e.key === "`" && input === "") { e.preventDefault(); setOpen(false); }
               }} />
      </div>
    </section>`;
}

// ===================== code panel (live redefinition) =====================
function cleanType(t) { return String(t).replace(/^\w+:/, ""); }
// A redefinition eval's class name is always BARE (source syntax, not a qualified reference) —
// module CONTEXT rides in an optional `module a.b.c` source header instead (07-modules.md §6),
// which addresses the redefinition at the right qualified name even under a bare-name collision
// across modules. `cls` may be either bare or qualified (callers vary); `sc.name`/`sc.module` are
// always the authoritative bare name / owning module, so the skeleton is built from those.
function classSkeleton(cls, sc) {
  const bare = (sc && sc.name) || cls;
  const header = sc && sc.module ? `module ${sc.module}\n\n` : "";
  const impl = sc.implements && sc.implements.length ? " implements " + sc.implements.join(", ") : "";
  let s = `${header}class ${bare}${impl} {\n`;
  for (const f of sc.fields) s += `  ${f.name}: ${cleanType(f.type)}\n`;
  if (sc.methods.length) s += "\n";
  for (const m of sc.methods) {
    const params = m.params.map((p) => `${p.name}: ${cleanType(p.type)}`).join(", ");
    const ret = m.returns && m.returns !== "Void" && m.returns !== "()" ? ` -> ${cleanType(m.returns)}` : "";
    s += `  fn ${m.name}(${params})${ret} {\n    // edit this body\n  }\n`;
  }
  return s + "}\n";
}
function CodePanel({ session, text, setText, onClose }) {
  const [res, setRes] = useState(null);  // {error} | {ok, text}
  const flash = useRef(0);
  const editorRef = useRef(null);
  useEffect(() => { if (session) { setRes(null); editorRef.current && editorRef.current.focus(); } }, [session]);

  const define = useCallback(async () => {
    setRes({ pending: true });
    const r = await evalSource(text);
    flash.current += 1;
    if (r.error) setRes({ error: r.error, flash: flash.current });
    else {
      const v = r.value || {};
      const msg = v.type === "defined"
        ? `✓ ${v.defined} redefined — now at generation ${v.gen}`
        : `✓ ${JSON.stringify(v)}`;
      setRes({ ok: msg, flash: flash.current });
      setTimeout(bumpDetail, 60);
    }
  }, [text]);

  const cls = session ? ((session.sc && session.sc.name) || session.cls) : "";
  const resClass = "code-result"
    + (res && res.ok ? " ok flash" : "") + (res && res.error ? " err" : "");
  const resText = !res ? "" : res.pending ? "defining…"
    : res.ok ? res.ok : `✗ ${res.error.kind}: ${res.error.message}`;

  return html`
    <section id="code-panel" class=${session ? "" : "collapsed"}>
      <div class="code-head">
        <span class="code-title">redefine <span class="code-cls">${cls}</span></span>
        <span class="code-hint">live code change — edit a body or add a <code>field: Type = default</code>, then define</span>
        <button class="ghost-btn" onClick=${onClose}>close</button>
      </div>
      <textarea id="code-editor" ref=${editorRef} spellcheck="false" autocomplete="off"
                value=${text}
                onInput=${(e) => setText(e.target.value)}
                onKeyDown=${(e) => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); define(); } }}></textarea>
      <div class="code-row">
        <button class="invoke-btn" onClick=${define}>define</button>
        <div class=${resClass} key=${res && res.flash}>${resText}</div>
      </div>
    </section>`;
}

// ===================== transcript drawer =====================
function truncate(s, n) { s = String(s).replace(/\s+/g, " "); return s.length > n ? s.slice(0, n) + "…" : s; }
function TranscriptDrawer({ open, onClose }) {
  const tx = useStore(txStore);
  return html`
    <aside id="transcript" class=${open ? "" : "collapsed"}>
      <div class="transcript-head"><span>eval transcript</span><button class="ghost-btn" onClick=${onClose}>close</button></div>
      <div id="transcript-list">
        ${open ? tx.slice(0, 120).map((e, i) => html`
          <div class="tx" key=${i}>
            <div class="tx-req">${truncate(e.source, 200)}</div>
            ${e.resp ? html`<div class=${"tx-res" + (e.isErr ? " err" : "")}>${truncate(JSON.stringify(e.resp.value ?? e.resp.error), 300)}</div>` : ""}
          </div>`) : ""}
      </div>
    </aside>`;
}

// ===================== connection indicator + topbar =====================
function ConnIndicator() {
  const s = useStore(connStore);
  const cls = "conn " + (s === "down" ? "down" : "live");
  const label = s === "down" ? "reconnecting…" : s === "slow" ? "live (slow)" : "● live";
  return html`<div class=${cls}><span class="dot"></span><span class="conn-label">${label}</span></div>`;
}
function TopBar({ onGlobalSearch, onToggleTranscript, mode, setMode, onBack, programName, focus, everywhere, setEverywhere }) {
  const [q, setQ] = useState("");
  return html`
    <header id="topbar">
      ${onBack ? html`<button class="ghost-btn back-btn" title="back to the portal" onClick=${onBack}>← portal</button>` : ""}
      <div class="brand">scry<span class="brand-sub">${programName || "live viewer"}</span></div>
      <div class="viewtoggle">
        <button class=${"vt-btn" + (mode === "map" ? " active" : "")} onClick=${() => setMode("map")}>Map</button>
        <button class=${"vt-btn" + (mode === "browse" ? " active" : "")} onClick=${() => setMode("browse")}>List</button>
      </div>
      <input id="global-search" type="text" spellcheck="false" autocomplete="off"
             placeholder=${focus && !everywhere ? `search in ${focus}…` : "jump to Agent#7 or search field values…"}
             value=${q} onInput=${(e) => setQ(e.target.value)}
             onKeyDown=${(e) => { if (e.key === "Enter") onGlobalSearch(q); }} />
      ${focus ? html`<label class="everywhere-toggle" title="search/filter the whole program, not just the focused module">
          <input type="checkbox" checked=${everywhere} onChange=${(e) => setEverywhere(e.target.checked)} /> everywhere
        </label>` : ""}
      <div class="topbar-right">
        <button class="ghost-btn" title="eval transcript" onClick=${onToggleTranscript}>transcript</button>
        <${ConnIndicator} />
      </div>
    </header>`;
}

// ===================== bespoke nested-containment view (Phase V1) =====================
// The DEFAULT live view. NOT a graph — a hand-built instrument panel where OWNERSHIP becomes
// NESTING and SIZE becomes MASS, driven by the graph() eval (every live instance + its
// entity-field references, by stable id) plus schema() (static shape, for domain-vs-infra
// reachability + interface implementors + climbing per-type counts). Layout is DETERMINISTIC:
// positions derive from the stable program structure, not from counts, so climbing counts grow
// stacks/bars IN PLACE and never reflow. Shared instances (a Tool held by many owners) are NOT
// nested and NOT edges — each gets a stable IDENTITY COLOR and renders as that chip everywhere
// it is held; hovering one lights up every appearance. Utility/transport types recede to a
// faded, collapsible infrastructure strip. See docs/DECISIONS.md #15 + docs/06-implementation.md.

const N_ID = 8;                                  // identity palette slots (--id-0 .. --id-7)
const roleClass = (r) => {                       // normalize a Message role -> mrow tint class
  const s = String(r || "").toLowerCase();
  if (s.startsWith("assist") || s === "asst") return "asst";
  if (s === "tool_result" || s === "tres" || s.startsWith("tool_res")) return "tres";
  if (s === "tool" || s === "tool_use" || s === "tuse" || s.startsWith("tool")) return "tool";
  return "user";
};
// scalar accessors over a graph() instance record
const sVal = (inst, name) => {                   // a scalar field's String/number value, or null
  const f = inst.scalars && inst.scalars[name];
  if (!f) return null;
  if (f.case !== undefined) return f.case;       // enum -> its case name
  return f.value !== undefined ? f.value : null;
};
// a "message" = a leaf with a role AND some body text (content/text). Agent also has a `role`
// field, so requiring a body keeps Agents/other role-bearing entities out of message stacks.
const isMsgLike = (inst) => !!(inst.scalars && inst.scalars.role != null &&
  (inst.scalars.content != null || inst.scalars.text != null || inst.scalars.body != null));
const displayName = (inst) =>
  sVal(inst, "name") || sVal(inst, "title") ||
  (sVal(inst, "content") ? truncate(sVal(inst, "content"), 40) : null) || inst.ref;
const refParts = (id) => { const m = /^(.+)#(\d+)$/.exec(id); return m ? { cls: m[1], slot: +m[2] } : null; };
// open a graph()/instances() record's detail: prefer its wire "qualified" (Phase 3 addition,
// collision-safe) and fall back to the bare id-parsed class name only if absent (e.g. an
// un-rebuilt server) — the CRITICAL rekey applied at every drill-in call site.
function openInst(inst, onOpen) {
  if (!inst) return;
  const p = refParts(inst.ref);
  if (!p) return;
  onOpen(inst.qualified || p.cls, p.slot, inst.generation);
}

// The whole derivation: ownership (nesting) vs sharing (identity chips) vs infrastructure,
// from the live instance list + the static schema. Pure + deterministic.
// `moduleAware` (true only while a module FOCUS is scoping the view — see NestedView): ownership
// nesting additionally requires the owner and the owned instance to share a module, so the
// focused module's own content stands alone as roots even when normally nested inside a
// cross-module parent. UNSET by default: ordinary ownership nesting crosses module lines freely
// (an Agent in module `assistant` naturally nests a Conversation from `agent.core`) — genuine
// sharing (>=2 owners) already gets the identity-chip treatment module-blind, which is the
// entirety of "cross-module references reuse the existing shared-entity treatment" (07-modules.md
// §7 item 2) — nothing module-specific needed there.
function computeNested(instances, schema, moduleAware) {
  const byId = new Map(instances.map((i) => [i.ref, i]));
  // graph() instances carry "qualified" (Phase 3 wire addition); fall back to bare "type" so an
  // un-rebuilt server still renders (degraded: bare-name collisions are then possible, as before).
  const qOf = (inst) => (inst && (inst.qualified || inst.type)) || null;
  const typeOf = (id) => qOf(byId.get(id));
  const nodes = schema || [];
  // CRITICAL rekey (07-modules.md §7): keyed by `qualified`, never bare `name` — two modules can
  // declare the same bare class name (tests/run/modules_coexist.scry) and must not collide here.
  const byName = new Map(nodes.map((n) => [n.qualified, n]));
  const entityTypes = nodes.filter((n) => n.kind === "class" || n.kind === "object").map((n) => n.qualified);
  const liveCount = (t) => (byName.get(t) || {}).liveCount || 0;
  const modOf = (t) => moduleOf(byName.get(t));

  // ---- static type-reference graph (qualified), interfaces expanded to their implementors ----
  const implementorsOf = new Map();
  for (const n of nodes) if (n.kind === "interface") {
    const impl = (n.implementors || []).map((bareNm) => { const x = nodes.find((y) => y.name === bareNm); return x ? x.qualified : bareNm; });
    implementorsOf.set(n.qualified, impl);
  }
  const expand = (t) => implementorsOf.has(t) ? implementorsOf.get(t) : [t];
  const refOut = new Map();                       // qualified entity type -> Set(qualified entity type)
  for (const n of nodes) {
    if (n.kind !== "class" && n.kind !== "object") continue;
    const outs = new Set();
    for (const f of n.fields || []) for (const rt of (f.refQualified && f.refQualified.length ? f.refQualified : f.refTypes) || []) for (const e of expand(rt)) outs.add(e);
    refOut.set(n.qualified, outs);
  }
  const referenced = new Set();
  for (const [, outs] of refOut) for (const t of outs) referenced.add(t);
  // worker/scaffolding types (those implementing a BUILTIN interface, i.e. Runnable thread
  // bodies) are execution machinery, not domain roots — they hold a domain object to run it,
  // but the domain owner (Orchestrator) is the real container. `implements` stays bare on the
  // wire (a class's own property, not a cross-module lookup key), so this stays bare-matched.
  const builtinIfaces = new Set(nodes.filter((n) => n.kind === "interface" && n.builtin).map((n) => n.name));
  const isWorker = (t) => ((byName.get(t) || {}).implements || []).some((i) => builtinIfaces.has(i));

  // primary domain root TYPE = the unreferenced container whose static reachable set is largest.
  const reach = (start) => {
    const seen = new Set([start]), st = [start];
    while (st.length) { const c = st.pop(); for (const t of (refOut.get(c) || [])) if (!seen.has(t) && byName.has(t)) { seen.add(t); st.push(t); } }
    return seen;
  };
  const rootTypeCands = entityTypes.filter((t) => (refOut.get(t) || new Set()).size > 0 && !referenced.has(t) && !isWorker(t));
  const sumLive = (set) => { let s = 0; for (const t of set) s += liveCount(t); return s; };
  let domainTypes = new Set(entityTypes);         // fallback: everything is domain
  if (rootTypeCands.length) {
    // Primary root = the candidate spanning the LARGEST connected domain (reach size); ties
    // broken by total LIVE instances in that reach (so a real, populated container like
    // Orchestrator beats a same-shaped-but-empty worker like SubAgentWorker, and a small
    // side-tree like ModelResponse->ToolCall can never win on instance count alone).
    let best = null, bestSize = -1, bestLive = -1;
    for (const c of rootTypeCands) {
      const s = reach(c), sz = s.size, lv = sumLive(s);
      if (sz > bestSize || (sz === bestSize && lv > bestLive)) { best = c; bestSize = sz; bestLive = lv; domainTypes = s; }
    }
  }
  const isDomainInst = (id) => byId.has(id) && domainTypes.has(typeOf(id));

  // ---- instance-level ownership (only among domain instances) ----
  const owners = new Map();                        // targetId -> Set(ownerId), ALL modules
  for (const inst of instances) {
    if (!isDomainInst(inst.ref)) continue;
    for (const r of inst.refs || []) for (const tid of r.ids) {
      if (!isDomainInst(tid) || tid === inst.ref) continue;
      (owners.get(tid) || owners.set(tid, new Set()).get(tid)).add(inst.ref);
    }
  }
  // module-aware ownership scopes "owner" to same-module owners ONLY — a cross-module owner
  // never counts toward nesting/sharing (it gets the boundary-chip treatment in buildNode below).
  const ownerCount = (id) => {
    const all = owners.get(id) || new Set();
    if (!moduleAware) return all.size;
    const om = modOf(typeOf(id));
    let n = 0; for (const oid of all) if (modOf(typeOf(oid)) === om) n++;
    return n;
  };
  const sharedIds = new Set();                     // >= 2 distinct SAME-MODULE domain owners
  for (const inst of instances) if (isDomainInst(inst.ref) && ownerCount(inst.ref) >= 2) sharedIds.add(inst.ref);

  // ---- nesting tree from the domain roots (inCount 0) ----
  const placed = new Set();
  const buildNode = (id) => {
    const inst = byId.get(id);
    placed.add(id);
    const children = [], chipIds = [];
    const myMod = modOf(typeOf(id));
    for (const r of inst.refs || []) for (const tid of r.ids) {
      if (!byId.has(tid) || tid === id) continue;
      const crossModule = moduleAware && modOf(typeOf(tid)) !== myMod;
      if (crossModule || sharedIds.has(tid)) { if (!chipIds.includes(tid)) chipIds.push(tid); continue; }
      if (!isDomainInst(tid)) continue;            // reference into infra -> ignore here
      if (ownerCount(tid) === 1 && !placed.has(tid)) children.push(buildNode(tid));
    }
    let subtree = 1;
    for (const c of children) subtree += c.subtree;
    return { id, inst, children, chipIds, subtree, module: myMod };
  };
  const rootIds = instances.filter((i) => isDomainInst(i.ref) && ownerCount(i.ref) === 0).map((i) => i.ref);
  // singleton objects (a leaf like Session): unreferenced, no entity fields, exactly 1 live.
  const singletonTypes = new Set(entityTypes.filter((t) =>
    !domainTypes.has(t) && liveCount(t) === 1 && (refOut.get(t) || new Set()).size === 0 && !referenced.has(t)));
  const singletons = instances.filter((i) => singletonTypes.has(qOf(i)));
  // container roots become the stage; leaf roots that are singleton-objects go to the header.
  const roots = rootIds.filter((id) => !singletonTypes.has(typeOf(id))).map(buildNode)
    .sort((a, b) => b.subtree - a.subtree);

  // ---- identity color: stable slot per shared id (sorted -> index) ----
  const idColor = new Map([...sharedIds].sort().map((id, i) => [id, i % N_ID]));

  // ---- infrastructure types: entity types with no presence in the domain (nor singletons) ----
  const infraTypes = entityTypes
    .filter((t) => !domainTypes.has(t) && !singletonTypes.has(t) && liveCount(t) > 0)
    .map((t) => ({ qualified: t, name: (byName.get(t) || {}).name || t, module: modOf(t), count: liveCount(t) }))
    .sort((a, b) => b.count - a.count);
  const infraSet = new Set(infraTypes.map((x) => x.qualified));

  // ---- census: every entity type with a live instance, mass by count ----
  const census = entityTypes
    .filter((t) => liveCount(t) > 0)
    .map((t) => ({ qualified: t, name: (byName.get(t) || {}).name || t, module: modOf(t), count: liveCount(t), util: infraSet.has(t) }))
    .sort((a, b) => b.count - a.count);

  return { byId, roots, singletons, sharedIds, idColor, infraTypes, census, ownerCount, byName, moduleAware };
}

// ===================== type-level containment skeleton (Phase V3) =====================
// The STATIC template that live data fills in (DECISIONS #14 unified with #15). Built purely
// from schema() — NO instances — so `scry inspect` (which never runs main()) still shows the
// bespoke class-relationship structure in the nested style: a representative CELL per domain
// TYPE, nested by ownership (Agent-type ▸ Conversation-type ▸ Message-type), shared TYPES as
// identity chips (colored by a stable type-name→palette slot, since there are no instance ids),
// infrastructure types faded in the strip, `object`-kind singletons as obj chips. It mirrors
// computeNested's exact static reachability/ownership rules, just at the TYPE level (owner =
// a domain type whose field references the target type, interfaces expanded to implementors).
function computeTypeSkeleton(schema, moduleAware) {
  const nodes = schema || [];
  // CRITICAL rekey (07-modules.md §7): qualified, not bare name — see computeNested.
  const byName = new Map(nodes.map((n) => [n.qualified, n]));
  const isEntity = (t) => { const n = byName.get(t); return !!n && (n.kind === "class" || n.kind === "object"); };
  const entityTypes = nodes.filter((n) => n.kind === "class" || n.kind === "object").map((n) => n.qualified);
  const modOf = (t) => moduleOf(byName.get(t));

  // interface -> implementors, so a `Tool` field expands to ShellTool/SearchTool/…
  const implementorsOf = new Map();
  for (const n of nodes) if (n.kind === "interface") {
    const impl = (n.implementors || []).map((bareNm) => { const x = nodes.find((y) => y.name === bareNm); return x ? x.qualified : bareNm; });
    implementorsOf.set(n.qualified, impl);
  }
  const expand = (t) => implementorsOf.has(t) ? implementorsOf.get(t) : [t];

  // type -> Set(entity type) it references through a field (interfaces expanded)
  const refOut = new Map();
  for (const n of nodes) {
    if (n.kind !== "class" && n.kind !== "object") continue;
    const outs = new Set();
    for (const f of n.fields || []) for (const rt of (f.refQualified && f.refQualified.length ? f.refQualified : f.refTypes) || []) for (const e of expand(rt)) if (isEntity(e)) outs.add(e);
    refOut.set(n.qualified, outs);
  }
  const referenced = new Set();
  for (const [, outs] of refOut) for (const t of outs) referenced.add(t);

  const builtinIfaces = new Set(nodes.filter((n) => n.kind === "interface" && n.builtin).map((n) => n.name));
  const isWorker = (t) => ((byName.get(t) || {}).implements || []).some((i) => builtinIfaces.has(i));

  const reach = (start) => {
    const seen = new Set([start]), st = [start];
    while (st.length) { const c = st.pop(); for (const t of (refOut.get(c) || [])) if (!seen.has(t) && byName.has(t)) { seen.add(t); st.push(t); } }
    return seen;
  };
  // primary domain root TYPE = unreferenced non-worker container with the largest reachable set;
  // ties broken by qualified name (no live counts to break them, unlike computeNested). Deterministic.
  const rootCands = entityTypes.filter((t) => (refOut.get(t) || new Set()).size > 0 && !referenced.has(t) && !isWorker(t));
  let domainTypes = new Set(entityTypes), rootTypes = [];
  if (rootCands.length) {
    const scored = rootCands.map((c) => ({ c, s: reach(c) })).sort((a, b) => b.s.size - a.s.size || a.c.localeCompare(b.c));
    domainTypes = scored[0].s; rootTypes = [scored[0].c];
  }

  // type-level ownership: owners(U) = domain types T (≠U) whose refOut contains U, same-module
  // only when moduleAware (a cross-module type reference gets the boundary-chip treatment).
  const ownerTypesAll = new Map();
  for (const t of domainTypes) for (const u of (refOut.get(t) || [])) {
    if (!domainTypes.has(u) || u === t) continue;
    (ownerTypesAll.get(u) || ownerTypesAll.set(u, new Set()).get(u)).add(t);
  }
  const ownerCount = (t) => {
    const all = ownerTypesAll.get(t) || new Set();
    if (!moduleAware) return all.size;
    const om = modOf(t);
    let n = 0; for (const o of all) if (modOf(o) === om) n++;
    return n;
  };
  const sharedTypes = new Set([...domainTypes].filter((t) => ownerCount(t) >= 2));

  // message-like TYPE: has a `role` scalar AND a body scalar (content/text/body) — same shape
  // rule as the live isMsgLike, so a Conversation's owned Message type renders as a stack.
  const isMsgType = (t) => { const fs = new Set(((byName.get(t) || {}).fields || []).map((f) => f.name));
    return fs.has("role") && (fs.has("content") || fs.has("text") || fs.has("body")); };

  // nesting tree from the root type: a type with exactly ONE (same-module) owner nests inside
  // it; shared / cross-module types become chips; refs into infra are ignored here.
  const placed = new Set();
  const buildNode = (name) => {
    placed.add(name);
    const children = [], chipTypes = [];
    const myMod = modOf(name);
    for (const u of [...(refOut.get(name) || [])].sort()) {
      if (u === name) continue;
      const crossModule = moduleAware && modOf(u) !== myMod;
      if (crossModule || sharedTypes.has(u)) { if (!chipTypes.includes(u)) chipTypes.push(u); continue; }
      if (!domainTypes.has(u)) continue;
      if (ownerCount(u) === 1 && !placed.has(u)) children.push(buildNode(u));
    }
    let subtree = 1; for (const c of children) subtree += c.subtree;
    return { name, node: byName.get(name), children, chipTypes, subtree, module: myMod };
  };
  // when moduleAware (focused), roots are every domain type with NO same-module owner — mirrors
  // computeNested's per-instance rootIds exactly, so a leaf module's own types stand alone as
  // roots under focus even though `rootTypes` (the single whole-program primary root, e.g.
  // `assistant.Agent`) belongs to a DIFFERENT module and is filtered out of view.
  const rootSet = moduleAware ? [...domainTypes].filter((t) => ownerCount(t) === 0).sort() : rootTypes;
  let roots = rootSet.filter((t) => !placed.has(t)).map(buildNode).sort((a, b) => b.subtree - a.subtree);
  // DEFAULT when the program has no ownership hierarchy (flat types with no entity-to-entity
  // references, or a pure reference cycle with no unreferenced root): rather than an empty stage,
  // show every domain type as its own standalone cell. Gives any program a map to look at.
  if (roots.length === 0) {
    for (const t of [...domainTypes].sort()) if (!placed.has(t)) roots.push(buildNode(t));
    roots.sort((a, b) => b.subtree - a.subtree || a.name.localeCompare(b.name));
  }

  // identity color per shared TYPE (stable: sorted qualified name -> slot), same palette as live.
  const idColor = new Map([...sharedTypes].sort().map((t, i) => [t, i % N_ID]));

  // singleton objects: `object`-kind entity types (language-level singletons like std `Json`).
  // A class can't be known a singleton statically (that's a runtime count), so classes recede
  // to infra even if they end up 1-of at runtime — the honest, count-free call.
  const singletonTypes = entityTypes.filter((t) => !domainTypes.has(t) && byName.get(t).kind === "object");
  const singletonSet = new Set(singletonTypes);

  // infrastructure: entity types outside the domain (and not singleton objects) — faded strip.
  const infraTypes = entityTypes
    .filter((t) => !domainTypes.has(t) && !singletonSet.has(t))
    .map((t) => ({ qualified: t, name: (byName.get(t) || {}).name || t, module: modOf(t) }))
    .sort((a, b) => a.name.localeCompare(b.name));

  // census: every entity type, domain first (root order then name), then singletons, then infra.
  // No live counts, so mass is unknown — the ribbon shows types with no bar.
  const domainOrder = [...domainTypes].sort((a, b) => (rootTypes.includes(b) ? 1 : 0) - (rootTypes.includes(a) ? 1 : 0) || a.localeCompare(b));
  const dispOf = (t) => (byName.get(t) || {}).name || t;
  const census = [
    ...domainOrder.map((t) => ({ qualified: t, name: dispOf(t), module: modOf(t), util: false })),
    ...singletonTypes.map((t) => ({ qualified: t, name: dispOf(t), module: modOf(t), util: true })),
    ...infraTypes.map((x) => ({ qualified: x.qualified, name: x.name, module: x.module, util: true })),
  ];

  return { byName, roots, singletonTypes, sharedTypes, idColor, infraTypes, census, ownerCount, domainTypes, refOut, expand, isMsgType, moduleAware };
}
// follow a dotted field-path over the TYPE graph -> the distinct element TYPES at the end.
// `all` = the target type itself (every instance of it, at runtime). Interfaces are expanded.
function followTypePath(startType, pathStr, model) {
  if (pathStr === "all" || pathStr === "byCount") return [startType];
  let cur = [startType];
  for (const hop of String(pathStr).split(".")) {
    const next = [];
    for (const t of cur) {
      const n = model.byName.get(t); if (!n) continue;
      const f = (n.fields || []).find((x) => x.name === hop);
      if (f) for (const rt of (f.refQualified && f.refQualified.length ? f.refQualified : f.refTypes) || []) for (const e of model.expand(rt)) next.push(e);
    }
    cur = next;
  }
  return [...new Set(cur)];
}
const scalarFieldNames = (model, t) =>
  ((model.byName.get(t) || {}).fields || []).filter((f) => !(f.refTypes && f.refTypes.length)).map((f) => f.name);

// an identity chip for a shared instance (colored, hover-highlights all its appearances)
function IdChip({ id, model, small, onEnter, onLeave, onClick, focus, onExpandFocus }) {
  const slot = model.idColor.get(id) ?? 0;
  const inst = model.byId.get(id);
  const label = inst ? (sVal(inst, "name") || sVal(inst, "title") || inst.type) : id;
  const held = model.ownerCount(id);
  // boundary chip: this reference crosses OUT of the current focus (07-modules.md §7 item 2's
  // "expand focus to include <module>" affordance).
  const outside = focus && inst && inst.module && !inFocus(inst.module, focus);
  return html`
    <span class="chip-wrap">
      <button class=${"chip" + (small ? " sm" : "") + (outside ? " boundary" : "")} data-identity=${id} data-slot=${slot}
              style=${{ "--c": `var(--id-${slot})` }}
              title=${`${id} · shared instance · held by ${held}` + (outside ? ` · outside focus (${inst.module})` : "")}
              onMouseEnter=${() => onEnter(id)} onMouseLeave=${onLeave}
              onClick=${(e) => { e.stopPropagation(); onClick(id); }}>
        <span class="knob"></span>${label}</button>
      ${outside && onExpandFocus ? html`<button class="mod-expand" title=${"expand focus to include " + inst.module}
          onClick=${(e) => { e.stopPropagation(); onExpandFocus(inst.module); }}>⤢</button>` : ""}
    </span>`;
}

// ---------- program-declared views (V2): follow a dotted field-path over graph() records ----------
function followPath(startId, pathStr, byId) {
  let cur = [startId];
  for (const hop of String(pathStr).split(".")) {
    const next = [];
    for (const id of cur) {
      const inst = byId.get(id); if (!inst) continue;
      const r = (inst.refs || []).find((x) => x.field === hop);
      if (r) for (const t of r.ids) next.push(t);
    }
    cur = next;
  }
  return cur;
}
const clauseByKey = (view, key) => (view.clauses || []).find((c) => c.key === key);
const argValue = (clause, name) => ((clause.args || []).find((a) => a.name === name) || {}).value;
// resolve the instance ids a section/clause item points at (path, or `all` = all of a type)
function resolveItemIds(clause, node, model, targetType) {
  if (clause.path === "all") return [...model.byId.values()].filter((i) => i.type === targetType).map((i) => i.ref);
  if (clause.path === "byCount") return [];
  return followPath(node.id, clause.path, model.byId);
}

// one representation (timeline / chips / rows / card / heat), honoring the declared spec
function Representation({ clause, node, model, targetType, onEnter, onLeave, onChip, onOpen }) {
  const rep = clause.representation;
  let ids = resolveItemIds(clause, node, model, targetType);
  const insts = ids.map((id) => model.byId.get(id)).filter(Boolean);
  const orderField = argValue(clause, "order");
  if (orderField) insts.sort((a, b) => String(sVal(a, orderField)).localeCompare(String(sVal(b, orderField))));
  if (rep === "timeline") {
    return html`<div class="timeline">
      ${insts.map((m) => html`<div class=${"tl drill " + roleClass(sVal(m, "role"))} data-mrow=${m.ref} key=${m.ref}
        onClick=${(e) => { e.stopPropagation(); openInst(m, onOpen); }}>
        <span class="role">${sVal(m, "role") || m.type}</span>
        <span class="msg">${truncate(sVal(m, "content") || sVal(m, "text") || displayName(m), 160)}</span></div>`)}
    </div>`;
  }
  if (rep === "chips") {
    return html`<div class="vchips">
      ${insts.map((it) => model.sharedIds.has(it.ref)
        ? html`<${IdChip} key=${it.ref} id=${it.ref} model=${model} onEnter=${onEnter} onLeave=${onLeave} onClick=${onChip} />`
        : html`<button class="chip" key=${it.ref} style=${{ "--c": "var(--fg-faint)" }}
              onClick=${(e) => { e.stopPropagation(); openInst(it, onOpen); }}>
            <span class="knob"></span>${sVal(it, "name") || sVal(it, "title") || it.type}</button>`)}
    </div>`;
  }
  if (rep === "rows") {
    return html`<div class="vrows">
      ${insts.map((it) => html`<div class="vrow" key=${it.ref}
          onClick=${() => openInst(it, onOpen)}>
        <span class="vr-kind">${it.type}</span>${displayName(it)}</div>`)}
    </div>`;
  }
  if (rep === "card") {
    const it = insts[0];
    if (!it) return html`<div class="mini-card"><span class="fk">—</span><span class="fv">no instance</span></div>`;
    return html`<div class="mini-card">
      ${Object.entries(it.scalars || {}).map(([k, f]) => html`<${React.Fragment} key=${k}>
        <span class="fk">${k}</span><span class="fv">${f.case !== undefined ? f.case : String(f.value)}</span><//>`)}
    </div>`;
  }
  if (rep === "heat") {
    const byField = argValue(clause, "by") || "role";
    const roles = [["user", "user"], ["asst", "assistant"], ["tool", "tool_use"], ["tres", "tool_result"]];
    return html`<div>
      <div class="heat-legend">
        ${roles.map(([c, label]) => html`<span class="pl" key=${c}><span class="sw" style=${{ background: `var(--m-${c})` }}></span>${label}</span>`)}
      </div>
      <div class="heat">
        ${insts.map((it, i) => html`<div class=${"hcell drill " + roleClass(sVal(it, byField))} data-hcell=${it.ref} key=${it.ref}
          title=${displayName(it)} onClick=${(e) => { e.stopPropagation(); openInst(it, onOpen); }}
          style=${{ "--mc": `var(--m-${roleClass(sVal(it, byField))})`, "--o": (0.55 + (i % 5) * 0.09).toFixed(2) }}></div>`)}
      </div></div>`;
  }
  return html`<div class="vrows">${insts.map((it) => html`<div class="vrow" key=${it.ref}>${displayName(it)}</div>`)}</div>`;
}

// a declared view rendered in place of the default cell (the mockup's AgentBoard)
function BoardView({ view, node, model, depth, onEnter, onLeave, onChip, onOpen, toggle }) {
  const { inst } = node;
  const titleC = clauseByKey(view, "title");
  const badgeC = clauseByKey(view, "badge");
  const role = sVal(inst, "role");
  const title = (titleC && sVal(inst, titleC.path)) || displayName(inst);
  const badge = badgeC && sVal(inst, badgeC.path);
  const st = badge ? String(badge).toLowerCase() : "";
  const badgeCls = "badge " + (st.includes("run") ? "running" : st.includes("paus") ? "paused" : st.includes("done") ? "done" : "waiting");
  const sections = (view.clauses || []).filter((c) => c.kind === "section" || (c.kind === "clause" && c.representation));
  const cls = "region" + (depth === 0 ? " root" : depth === 1 ? " lvl1" : " lvl2");
  return html`
    <div class=${cls} data-region=${node.id}>
      <div class="board" data-view=${view.name}>
        <div class="board-head">
          <div>
            <div class="board-title">${title}</div>
            <div class="board-sub">${inst.type}${role ? " · " + role : ""} · view ${view.name}</div>
          </div>
          ${badge ? html`<span class=${badgeCls} style=${{ marginLeft: "auto" }}><span class="bd"></span>${String(badge)}</span>` : ""}
          ${toggle}
        </div>
        ${sections.map((c, i) => html`
          <div class="vsection" key=${i}>
            <div class="sh">${c.label || c.key}${c.representation ? html`<span class="as">as ${c.representation}</span>` : ""}</div>
            <${Representation} clause=${c} node=${node} model=${model} targetType=${view.target}
              onEnter=${onEnter} onLeave=${onLeave} onChip=${onChip} onOpen=${onOpen} />
          </div>`)}
      </div>
    </div>`;
}

// a nested region (recursive): header + owned children + a dense message stack + identity chips.
// When the instance's type has a declared `view`, a ▤ cell / ▧ board toggle flips to the bespoke board.
function Region({ node, model, depth, onEnter, onLeave, onChip, onOpen, viewsByType, viewMode, setViewMode, focus, onExpandFocus }) {
  const { inst } = node;
  const view = viewsByType && viewsByType.get(inst.type);
  const boardOn = !!(view && viewMode[node.id] === "board");
  const toggle = view ? html`
    <div class="vtoggle" onClick=${(e) => e.stopPropagation()}>
      <button class=${boardOn ? "" : "on"} onClick=${(e) => { e.stopPropagation(); setViewMode(node.id, "cell"); }}>▤ cell</button>
      <button class=${boardOn ? "on" : ""} onClick=${(e) => { e.stopPropagation(); setViewMode(node.id, "board"); }}>▧ board</button>
    </div>` : "";
  if (boardOn) return html`<${BoardView} view=${view} node=${node} model=${model} depth=${depth}
    onEnter=${onEnter} onLeave=${onLeave} onChip=${onChip} onOpen=${onOpen} toggle=${toggle} />`;
  const status = sVal(inst, "status");
  const role = sVal(inst, "role");
  const msgKids = node.children.filter((c) => isMsgLike(c.inst));
  const subKids = node.children.filter((c) => !isMsgLike(c.inst));
  const cls = "region" + (depth === 0 ? " root" : depth === 1 ? " lvl1" : " lvl2");
  const st = status ? String(status).toLowerCase() : "";
  const badgeCls = "badge " + (st.includes("run") ? "running" : st.includes("paus") ? "paused" : st.includes("done") ? "done" : "waiting");
  const p = refParts(node.id);
  const qcls = inst.qualified || (p && p.cls);
  return html`
    <div class=${cls} data-region=${node.id}>
      <div class="region-head" onClick=${() => p && onOpen(qcls, p.slot, inst.generation)}>
        <span class="node-kind">${inst.type}</span>
        <span class="region-name">${displayName(inst)}</span>
        ${role ? html`<span class="region-role">${role}</span>` : ""}
        ${status ? html`<span class=${badgeCls}><span class="bd"></span>${String(status)}</span>` : ""}
        ${toggle}
      </div>
      ${msgKids.length ? html`
        <div class="conv">
          <div class="conv-label"><span class="k">owns ▸ ${msgKids[0].inst.type}</span>
            <span class="n">count <b>${msgKids.length}</b></span></div>
          <div class="mstack">
            ${msgKids.map((c) => { const mp = refParts(c.id); const mq = c.inst.qualified || (mp && mp.cls); return html`<div class=${"mrow drill " + roleClass(sVal(c.inst, "role"))} key=${c.id}
              data-mrow=${c.id} title=${displayName(c.inst)}
              onClick=${(e) => { e.stopPropagation(); if (mp) onOpen(mq, mp.slot, c.inst.generation); }}>
              <span class="tick"></span><span class="fill"></span></div>`; })}
          </div>
        </div>` : ""}
      ${subKids.length ? html`
        <div class=${"subregions" + (subKids.length > 1 && depth === 0 ? " grid" : "")}>
          ${subKids.map((c) => html`<${Region} key=${c.id} node=${c} model=${model} depth=${depth + 1}
            onEnter=${onEnter} onLeave=${onLeave} onChip=${onChip} onOpen=${onOpen}
            viewsByType=${viewsByType} viewMode=${viewMode} setViewMode=${setViewMode} focus=${focus} onExpandFocus=${onExpandFocus} />`)}
        </div>` : ""}
      ${node.chipIds.length ? html`
        <div class="refs"><span class="rk">references ▸ shared</span>
          ${node.chipIds.map((id) => html`<${IdChip} key=${id} id=${id} model=${model} small=${true}
            onEnter=${onEnter} onLeave=${onLeave} onClick=${onChip} focus=${focus} onExpandFocus=${onExpandFocus} />`)}
        </div>` : ""}
    </div>`;
}

// ---------- type-level (skeleton) renderers: the SAME bespoke vocabulary, drawn from TYPES ----------
// A shared-TYPE chip (colored by the type-name identity slot). Reuses the .chip data-identity
// contract so the live hover-highlight effect lights up every appearance of the SAME type.
function TypeChip({ name, model, small, onEnter, onLeave, onOpen, focus, onExpandFocus }) {
  // `name` is the QUALIFIED type identity (model.byName/idColor are qualified-keyed); the chip
  // DISPLAYS the short/disambiguated label only.
  const node = model.byName.get(name);
  const label = node ? shortLabel(node, [...model.byName.values()]) : name;
  const slot = model.idColor.get(name) ?? 0;
  const kind = (node || {}).kind || "type";
  const nodeMod = moduleOf(node);
  const outside = focus && nodeMod && !inFocus(nodeMod, focus);
  return html`
    <span class="chip-wrap">
      <button class=${"chip" + (small ? " sm" : "") + (outside ? " boundary" : "")} data-identity=${name} data-slot=${slot}
              style=${{ "--c": `var(--id-${slot})` }}
              title=${`${name} · shared ${kind} · referenced by ${model.ownerCount(name)} owners`
                + (outside ? ` · outside focus (${nodeMod})` : "")}
              onMouseEnter=${() => onEnter(name)} onMouseLeave=${onLeave}
              onClick=${(e) => { e.stopPropagation(); onOpen(name); }}>
        <span class="knob"></span>${label}</button>
      ${outside && onExpandFocus ? html`<button class="mod-expand" title=${"expand focus to include " + nodeMod}
          onClick=${(e) => { e.stopPropagation(); onExpandFocus(nodeMod); }}>⤢</button>` : ""}
    </span>`;
}

// a template representation for a declared-view section, resolved over the TYPE graph. Shows the
// element TYPE's shape (field names) rather than data, since no instances exist.
function TypeRepresentation({ clause, node, model, onEnter, onLeave, onChip, onOpen }) {
  const rep = clause.representation;
  const elemTypes = followTypePath(node.name, clause.path, model);
  if (rep === "timeline") {
    const et = elemTypes[0];
    const fields = et ? scalarFieldNames(model, et) : [];
    const roleF = fields.includes("role") ? "role" : (fields[0] || "role");
    const bodyF = fields.find((f) => f === "content" || f === "text" || f === "body") || fields[1] || "…";
    const tint = ["user", "asst", "tool", "tres"];
    return html`<div class="timeline">
      <div class="tmpl-shape">${et || "?"} ⟵ ${fields.join(", ") || "—"}</div>
      ${tint.map((c, i) => html`<div class=${"tl tmpl " + c} key=${i}>
        <span class="role">${roleF}</span>
        <span class="msg">${bodyF} ⟵ ${et || "instance"}</span></div>`)}
    </div>`;
  }
  if (rep === "chips") {
    return html`<div class="vchips">
      ${elemTypes.map((t) => model.sharedTypes.has(t)
        ? html`<${TypeChip} key=${t} name=${t} model=${model} onEnter=${onEnter} onLeave=${onLeave} onOpen=${onChip} />`
        : html`<button class="chip solo" key=${t} style=${{ "--c": "var(--fg-faint)" }}
              onClick=${(e) => { e.stopPropagation(); onOpen(t); }}><span class="knob"></span>${t}</button>`)}
    </div>`;
  }
  if (rep === "rows") {
    return html`<div class="vrows">
      ${elemTypes.map((t) => html`<div class="vrow" key=${t} onClick=${() => onOpen(t)}>
        <span class="vr-kind">${(model.byName.get(t) || {}).kind || "type"}</span>${t}</div>`)}
    </div>`;
  }
  if (rep === "card") {
    const et = elemTypes[0];
    const fields = et ? scalarFieldNames(model, et) : [];
    if (!et) return html`<div class="mini-card"><span class="fk">—</span><span class="fv">no type</span></div>`;
    return html`<div class="mini-card">
      ${fields.length ? fields.map((f) => html`<${React.Fragment} key=${f}>
        <span class="fk">${f}</span><span class="fv">⟵ ${et}</span><//>`)
        : html`<span class="fk">${et}</span><span class="fv">no scalar fields</span>`}
    </div>`;
  }
  if (rep === "heat") {
    const roles = [["user", "user"], ["asst", "assistant"], ["tool", "tool_use"], ["tres", "tool_result"]];
    return html`<div>
      <div class="heat-legend">
        ${roles.map(([c, label]) => html`<span class="pl" key=${c}><span class="sw" style=${{ background: `var(--m-${c})` }}></span>${label}</span>`)}
      </div>
      <div class="heat">
        ${Array.from({ length: 24 }).map((_, i) => html`<div class=${"hcell tmpl " + roles[i % 4][0]} key=${i}
          style=${{ "--mc": `var(--m-${roles[i % 4][0]})` }}></div>`)}
      </div>
      <div class="tmpl-shape">${elemTypes[0] || "?"} instances fill this stream at runtime</div></div>`;
  }
  return html`<div class="vrows">${elemTypes.map((t) => html`<div class="vrow" key=${t}>${t}</div>`)}</div>`;
}

// a declared view rendered as a TEMPLATE against the target TYPE (the inspect-mode AgentBoard):
// title/badge shown as their FIELD-NAME wiring, each section against the element type's shape.
function TypeBoardView({ view, node, model, depth, onEnter, onLeave, onChip, onOpen, toggle }) {
  const titleC = clauseByKey(view, "title");
  const badgeC = clauseByKey(view, "badge");
  const sections = (view.clauses || []).filter((c) => c.kind === "section" || (c.kind === "clause" && c.representation));
  const cls = "region" + (depth === 0 ? " root" : depth === 1 ? " lvl1" : " lvl2");
  return html`
    <div class=${cls} data-region=${node.name}>
      <div class="board tmpl" data-view=${view.name}>
        <div class="board-head">
          <div>
            <div class="board-title">${node.node ? shortLabel(node.node, [...model.byName.values()]) : node.name}<span class="tmpl-wire"> title ⟵ ${titleC ? titleC.path : "—"}</span></div>
            <div class="board-sub">view ${view.name} · template · fills at runtime</div>
          </div>
          ${badgeC ? html`<span class="badge tmpl" style=${{ marginLeft: "auto" }}><span class="bd"></span>badge ⟵ ${badgeC.path}</span>` : ""}
          ${toggle}
        </div>
        ${sections.map((c, i) => html`
          <div class="vsection" key=${i}>
            <div class="sh">${c.label || c.key}${c.representation ? html`<span class="as">as ${c.representation}</span>` : ""}
              <span class="tmpl-path">${c.path}</span></div>
            <${TypeRepresentation} clause=${c} node=${node} model=${model}
              onEnter=${onEnter} onLeave=${onLeave} onChip=${onChip} onOpen=${onOpen} />
          </div>`)}
      </div>
    </div>`;
}

// a nested TYPE region (recursive): a representative cell per domain type. Owned message-like
// types render as a labeled placeholder stack; owned sub-types recurse; shared types are chips.
function TypeRegion({ node, model, depth, onEnter, onLeave, onChip, onOpenType, viewsByType, viewMode, setViewMode, focus, onExpandFocus }) {
  const bareName = (node.node || {}).name || node.name;
  const label = node.node ? shortLabel(node.node, [...model.byName.values()]) : bareName;
  const view = viewsByType && viewsByType.get(bareName);
  const boardOn = !!(view && viewMode[node.name] === "board");
  const toggle = view ? html`
    <div class="vtoggle" onClick=${(e) => e.stopPropagation()}>
      <button class=${boardOn ? "" : "on"} onClick=${(e) => { e.stopPropagation(); setViewMode(node.name, "cell"); }}>▤ cell</button>
      <button class=${boardOn ? "on" : ""} onClick=${(e) => { e.stopPropagation(); setViewMode(node.name, "board"); }}>▧ board</button>
    </div>` : "";
  if (boardOn) return html`<${TypeBoardView} view=${view} node=${node} model=${model} depth=${depth}
    onEnter=${onEnter} onLeave=${onLeave} onChip=${onChip} onOpen=${onOpenType} toggle=${toggle} />`;
  const msgKids = node.children.filter((c) => model.isMsgType(c.name));
  const subKids = node.children.filter((c) => !model.isMsgType(c.name));
  const cls = "region" + (depth === 0 ? " root" : depth === 1 ? " lvl1" : " lvl2");
  const kind = (node.node || {}).kind || "type";
  return html`
    <div class=${cls} data-region=${node.name} data-type-cell=${node.name}>
      <div class="region-head" onClick=${() => onOpenType(node.name)}>
        <span class="node-kind">${kind}</span>
        <span class="region-name">${label}</span>
        <span class="region-role">type</span>
        ${toggle}
      </div>
      ${(() => {
        // DEFAULT representation for a type with no declared view: its scalar fields (name: type).
        // Entity-reference fields are already shown as nested subregions / shared-type chips below.
        const fields = ((node.node || {}).fields || []).filter((f) => !(f.refTypes && f.refTypes.length));
        return fields.length ? html`
          <div class="type-fields">
            ${fields.map((f) => html`<div class="tf-row" key=${f.name}>
              <span class="tf-name">${f.name}</span><span class="tf-type">${cleanType(f.type)}</span></div>`)}
          </div>` : "";
      })()}
      ${msgKids.map((c) => html`
        <div class="conv" key=${c.name}>
          <div class="conv-label"><span class="k">owns ▸ ${(c.node || {}).name || c.name}</span>
            <span class="n">count <b>⟵ runtime</b></span></div>
          <div class="mstack">
            ${["user", "asst", "tool", "tres", "user", "asst"].map((r, i) => html`<div class=${"mrow tmpl " + r} key=${i}>
              <span class="tick"></span><span class="fill"></span></div>`)}
          </div>
        </div>`)}
      ${subKids.length ? html`
        <div class=${"subregions" + (subKids.length > 1 && depth === 0 ? " grid" : "")}>
          ${subKids.map((c) => html`<${TypeRegion} key=${c.name} node=${c} model=${model} depth=${depth + 1}
            onEnter=${onEnter} onLeave=${onLeave} onChip=${onChip} onOpenType=${onOpenType}
            viewsByType=${viewsByType} viewMode=${viewMode} setViewMode=${setViewMode} focus=${focus} onExpandFocus=${onExpandFocus} />`)}
        </div>` : ""}
      ${node.chipTypes.length ? html`
        <div class="refs"><span class="rk">references ▸ shared types</span>
          ${node.chipTypes.map((t) => html`<${TypeChip} key=${t} name=${t} model=${model} small=${true}
            onEnter=${onEnter} onLeave=${onLeave} onOpen=${onChip} focus=${focus} onExpandFocus=${onExpandFocus} />`)}
        </div>` : ""}
    </div>`;
}

// ===================== V4: in-map detail inspector =====================
// A right-docked inspector column that renders the FULL instance detail (the same DetailPane the
// browse screen uses) WITHOUT leaving the Map. Reference fields, method-result refs and REPL refs
// navigate WITHIN the inspector (a back-stack of targets) instead of switching to browse — done by
// overriding NavContext.navigateRef for everything rendered inside the panel. A target is either an
// instance {kind:"instance",cls,slot,gen} or, in static/inspect mode, a type {kind:"type",name}.
function TypeStaticDetail({ name, schema, onEditSource, onOpenType }) {
  const sc = schema.find((t) => t.qualified === name);
  if (!sc) return html`<div><div class="pane-title">${name}</div>
    <div class="pane-sub">type not in schema</div></div>`;
  const label = shortLabel(sc, schema);
  // implementors (and any bare-name field elsewhere on this node) are BARE in the wire — resolve
  // to qualified before handing off navigation so a same-named type in another module can't be
  // silently mis-picked.
  const qualFor = (bareName) => { const n = schema.find((x) => x.name === bareName); return n ? n.qualified : bareName; };
  const isIface = sc.kind === "interface" || (sc.implementors && sc.implementors.length);
  return html`
    <div>
      <div class="pane-title">${label}</div>
      <div class="pane-sub">
        ${isIface ? "interface" : "type"}${sc.implements && sc.implements.length
          ? html` · implements <span class="impl">${sc.implements.join(", ")}</span>` : ""} · static template
        ${sc.module ? html` · <span class="mod-tag">${sc.module}</span>` : ""}
        <button class="ghost-btn edit-src" onClick=${() => onEditSource(name, sc)}>✎ edit source</button>
      </div>
      ${sc.fields && sc.fields.length ? html`
        <div class="detail-section">
          <h3>fields</h3>
          <div class="field-grid">
            ${sc.fields.map((f) => html`<${React.Fragment} key=${f.name}>
              <div class="fcell fname">${f.name}<span class="ftype">${cleanType(f.type)}</span></div>
              <div class="fcell fval"><span class="v-void">⟵ runtime</span></div>
            <//>`)}
          </div>
        </div>` : ""}
      ${sc.methods && sc.methods.length ? html`
        <div class="detail-section">
          <h3>methods</h3>
          ${sc.methods.map((m) => html`
            <div class="method" key=${m.name}>
              <div class="method-head static">
                <span class="method-sig">${m.name}(${m.params.map((p) => `${p.name}: ${cleanType(p.type)}`).join(", ")}) <span class="mret">→ ${m.returns}</span></span>
              </div>
            </div>`)}
        </div>` : ""}
      ${sc.implementors && sc.implementors.length ? html`
        <div class="detail-section">
          <h3>implementors</h3>
          <div class="vchips">
            ${sc.implementors.map((t) => html`<button class="chip" key=${t}
              onClick=${() => onOpenType(qualFor(t))}><span class="knob"></span>${t}</button>`)}
          </div>
        </div>` : ""}
    </div>`;
}

// pick a human-ish label field off a graph/instances record for the instance list
function instSummary(it) {
  const f = it.fields || {};
  for (const k of ["name", "title", "role", "label", "topic", "id"]) {
    if (f[k] && f[k].type === "String" && f[k].value) return f[k].value;
  }
  return "";
}
// Opening a TYPE in the inspector: list its LIVE instances (click one to drill to its full
// detail — fields, methods, and the green ACTIONS buttons). Falls back to the static template
// when there are no instances (inspect / static-project mode).
function TypeLiveDetail({ name, schema, onEditSource, onNavRef }) {
  const [items, setItems] = useState(null);
  const scForFetch = schema.find((t) => t.qualified === name);
  usePoll(async () => {
    const bare = scForFetch ? scForFetch.name : name;
    const src = moduleHeader(scForFetch && scForFetch.module) + `${bare}.instances(filter: "", offset: 0, limit: 200)`;
    const r = await evalSource(src);
    setItems(r && r.value && r.value.items ? r.value.items : []);
  }, 900, [name, scForFetch && scForFetch.module, scForFetch && scForFetch.name]);
  if (items === null) return html`<div class="pane-title">${name}</div><div class="pane-sub">loading…</div>`;
  if (items.length === 0)
    return html`<${TypeStaticDetail} name=${name} schema=${schema} onEditSource=${onEditSource}
      onOpenType=${(n) => onNavRef({ kind: "type", name: n })} />`;
  const sc = schema.find((t) => t.qualified === name);
  const label = sc ? shortLabel(sc, schema) : name;
  return html`
    <div>
      <div class="pane-title">${label}</div>
      <div class="pane-sub">${items.length} live instance${items.length > 1 ? "s" : ""} · click one to inspect
        ${sc && sc.module ? html` · <span class="mod-tag">${sc.module}</span>` : ""}
        ${sc ? html`<button class="ghost-btn edit-src" onClick=${() => onEditSource(name, sc)}>✎ edit source</button>` : ""}</div>
      <div class="detail-section">
        <h3>instances</h3>
        <div class="inst-list">
          ${items.map((it) => {
            const m = /#(\d+)$/.exec(it.ref); const slot = m ? +m[1] : 0; const s = instSummary(it);
            return html`<button class="inst-row" key=${it.ref}
              onClick=${() => onNavRef({ kind: "instance", cls: name, slot, gen: it.generation ?? 0 })}>
              <span class="inst-id">${it.ref}</span>${s ? html`<span class="inst-sum">${s}</span>` : ""}
              <span class="inst-arrow">›</span></button>`;
          })}
        </div>
      </div>
    </div>`;
}

function InspectorPanel({ stack, schema, onEditSource, onNavRef, onCrumb, onBack, onClose, nav }) {
  const top = stack[stack.length - 1];
  const crumbLabel = (t) => {
    if (t.kind === "instance") { const sc = schema.find((n) => n.qualified === t.cls); return `${sc ? shortLabel(sc, schema) : t.cls}#${t.slot}`; }
    if (t.kind === "function") return `${t.name}()`;
    const sc = schema.find((n) => n.qualified === t.name); return sc ? shortLabel(sc, schema) : t.name;
  };
  // refs anywhere in the inspector (field values, method results, REPL) push a new target here.
  const inspNav = useMemo(() => ({
    ...nav,
    navigateRef: (v) => {
      const m = /^(.+)#(\d+)$/.exec(v.ref);
      if (!m) return;
      // a value's "ref" is BARE on the wire (serialize-entity, untouched by Phase 3 — flagged in
      // the report); best-effort resolve to the qualified identity via schema so navigation stays
      // collision-safe whenever the bare name happens to be unambiguous (the common case).
      const bare = v.class || v.type;
      const n = (schema || []).find((x) => x.name === bare);
      onNavRef({ kind: "instance", cls: n ? n.qualified : bare, slot: +m[2], gen: v.generation ?? 0 });
    },
  }), [nav, onNavRef, schema]);
  return html`
    <aside id="inspector">
      <div class="insp-head">
        <div class="insp-crumbs">
          ${stack.map((t, i) => html`<${React.Fragment} key=${i}>
            ${i ? html`<span class="isep">›</span>` : ""}
            <span class=${"icrumb" + (i === stack.length - 1 ? " cur" : "")}
                  onClick=${i === stack.length - 1 ? undefined : () => onCrumb(i)}>${crumbLabel(t)}</span>
          <//>`)}
        </div>
        <div class="insp-actions">
          ${stack.length > 1 ? html`<button class="insp-btn" onClick=${onBack} title="back">←</button>` : ""}
          <button class="insp-btn" onClick=${onClose} title="close inspector">✕</button>
        </div>
      </div>
      <div class="insp-body">
        <${NavContext.Provider} value=${inspNav}>
          ${top.kind === "instance"
            ? html`<${DetailPane} cls=${top.cls} slot=${top.slot} gen=${top.gen} schema=${schema} onEditSource=${onEditSource} />`
            : top.kind === "function"
            ? html`<${FnTraceView} fn=${top} />`
            : html`<${TypeLiveDetail} name=${top.name} schema=${schema} onEditSource=${onEditSource}
                onNavRef=${onNavRef} />`}
        <//>
      </div>
    </aside>`;
}

// ===================== module rings — the map's outermost containment level (Phase 3) =====================
// "Map view: module = the outermost containment ring" (07-modules.md §7 item 1). One deterministic
// region per top-level module, sized by aggregate live mass, nesting sub-modules exactly as the
// modules() tree dictates. Shared for both the live (roots tagged `.module`) and the static
// skeleton (roots tagged `.module` too) stages — `renderRoot` is the only thing that differs.
function moduleRingHasContent(node, rootsByModule) {
  if ((rootsByModule.get(node.path) || []).length) return true;
  return (node.children || []).some((c) => moduleRingHasContent(c, rootsByModule));
}
function ModuleRing({ node, rootsByModule, renderRoot, depth, collapsed, toggle, setFocus }) {
  const own = rootsByModule.get(node.path) || [];
  const kids = node.children || [];
  if (!own.length && !kids.some((c) => moduleRingHasContent(c, rootsByModule))) return null;
  const faded = isStdModule(node.path);
  const open = !collapsed[node.path];
  return html`
    <div class=${"module-ring" + (faded ? " faded" : "") + (depth === 0 ? " top" : "")} data-module=${node.path}>
      <div class="module-ring-head" onClick=${() => toggle(node.path)}>
        <span class=${"caret" + (open ? " open" : "")}>▶</span>
        <span class="mr-path" title=${"focus " + node.path}
              onClick=${(e) => { e.stopPropagation(); setFocus(node.path); }}>${node.path}</span>
        <span class="mr-meta">${node.liveCount} live · ${node.typeCount} types${node.fnCount ? ` · ${node.fnCount} fns` : ""}</span>
      </div>
      ${open ? html`<div class="module-ring-body">
        ${own.map(renderRoot)}
        ${kids.map((c) => html`<${ModuleRing} key=${c.path} node=${c} rootsByModule=${rootsByModule}
            renderRoot=${renderRoot} depth=${(depth || 0) + 1} collapsed=${collapsed} toggle=${toggle} setFocus=${setFocus} />`)}
      </div>` : ""}
    </div>`;
}

function NestedView({ onInspect, selectedId, modTree, focus, setFocus, everywhere }) {
  const [instances, setInstances] = useState([]);
  const [schema, setSchema] = useState([]);
  const [functions, setFunctions] = useState([]);  // top-level functions (functions()) — first-class Map citizens
  const [views, setViews] = useState([]);          // program-declared view specs (views())
  const [viewMode, setViewModeState] = useState({});// instance id -> "cell" | "board"
  const [hoverId, setHoverId] = useState(null);
  const [infraOpen, setInfraOpen] = useState(false);
  const [ringCollapsed, setRingCollapsed] = useState({});  // module path -> collapsed (default open)
  const toggleRing = useCallback((path) => setRingCollapsed((s) => ({ ...s, [path]: !s[path] })), []);
  const prevCounts = useRef({});
  const [live, setLive] = useState({});           // type -> true when its count just climbed
  const freshRef = useRef(new Set());             // instance ids seen last poll (for pulse)
  const wrapRef = useRef(null);

  usePoll(async () => {
    const [g, s, v, fns] = await Promise.all([evalSource("graph()"), evalSource("schema()"), evalSource("views()"), evalSource("functions()")]);
    if (fns.value && fns.value.type === "Functions") setFunctions(fns.value.functions || []);
    if (s.value && s.value.nodes) {
      const nl = {};
      for (const n of s.value.nodes) {
        // CRITICAL rekey: trend/live-climb tracking by `qualified`, not bare name.
        const prev = prevCounts.current[n.qualified];
        nl[n.qualified] = prev != null && (n.liveCount || 0) > prev;
        prevCounts.current[n.qualified] = n.liveCount || 0;
      }
      setLive(nl); setSchema(s.value.nodes);
    }
    if (g.value && g.value.instances) setInstances(g.value.instances);
    if (v.value && v.value.views) setViews(v.value.views);
  }, 800, []);

  // first declared view per target type — the type's default custom board. views()' "target" is
  // BARE (follows the type, not the declaring module — 07-modules.md §7 last bullet); matched
  // against bare inst.type / the resolved schema node's bare name elsewhere, so stays bare here.
  const viewsByType = useMemo(() => {
    const m = new Map();
    for (const v of views) if (!m.has(v.target)) m.set(v.target, v);
    return m;
  }, [views]);
  const setViewMode = useCallback((id, m) => setViewModeState((s) => ({ ...s, [id]: m })), []);

  // >=2 non-std top-level modules -> module rings (chrome only). Below that threshold every
  // existing single-module program renders IDENTICALLY to pre-Phase-3 (07-modules.md §7 item 1's
  // "must look essentially unchanged").
  const moduleAware = nonStdTop(modTree).length >= 2;
  // focus scopes census/map/infra/functions to the subtree (07-modules.md §7 item 4), unless the
  // "everywhere" toggle is on. Rings render only when UNFOCUSED — focusing already zooms to one
  // subtree, so the outer ring chrome (and its siblings) would be redundant chrome around it.
  const scoping = !!focus && !everywhere;
  const showModuleChrome = moduleAware && !scoping;
  // Ownership-nesting itself is UNRESTRICTED by default (exactly the pre-Phase-3 algorithm — a
  // root nests its owned instances regardless of which module declares their TYPE, e.g. an
  // Agent in module `assistant` naturally nests a Conversation from `agent.core`; genuine
  // sharing already gets the identity-chip treatment module-blind, which is what
  // "cross-module references reuse the existing shared-entity treatment" means — nothing new to
  // build there). Nesting is restricted to same-module-only ONLY while a focus is actively
  // scoping the view, so a leaf module's own content can stand alone as roots under that focus
  // even when it's normally nested inside a cross-module parent (otherwise focusing a leaf
  // module would show an empty stage — its instances are all non-roots of the whole-program tree).
  const model = useMemo(() => computeNested(instances, schema, scoping), [instances, schema, scoping]);
  // the static TYPE-level template (from schema alone). When there are no live instances
  // (`scry inspect`, or the split-second before main() populates the arenas) we render THIS —
  // the same bespoke view at zero fill. Once instances arrive, the live model above takes over.
  const typeModel = useMemo(() => computeTypeSkeleton(schema, scoping), [schema, scoping]);
  const noInstances = instances.length === 0;
  const showSkeleton = noInstances && typeModel.roots.length > 0;
  const onExpandFocus = useCallback((mod) => setFocus(lcaModule(focus, mod)), [focus, setFocus]);

  // fresh-instance pulse: mark the DOM rows/regions that are new since the previous poll.
  useEffect(() => {
    const now = new Set(instances.map((i) => i.ref));
    if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches && freshRef.current.size) {
      for (const id of now) if (!freshRef.current.has(id)) {
        const el = wrapRef.current && wrapRef.current.querySelector(`[data-mrow="${CSS.escape(id)}"]`);
        if (el) { el.classList.remove("fresh"); void el.offsetWidth; el.classList.add("fresh"); }
      }
    }
    freshRef.current = now;
  }, [instances]);

  // identity highlight: light up EVERY appearance of the hovered instance id (imperative, like
  // the mockup — survives poll re-renders because it re-runs on both hoverId and instances).
  useEffect(() => {
    const wrap = wrapRef.current; if (!wrap) return;
    wrap.querySelectorAll(".chip.id-hi").forEach((c) => c.classList.remove("id-hi"));
    if (hoverId) wrap.querySelectorAll(`.chip[data-identity="${CSS.escape(hoverId)}"]`).forEach((c) => c.classList.add("id-hi"));
  }, [hoverId, instances]);

  // V4 selection highlight: ring EVERY appearance of the inspected instance (its region, its
  // message row, its identity chip) so the map shows where the open inspector target lives.
  useEffect(() => {
    const wrap = wrapRef.current; if (!wrap) return;
    wrap.querySelectorAll(".sel-inspect").forEach((c) => c.classList.remove("sel-inspect"));
    if (!selectedId) return;
    const esc = CSS.escape(selectedId);
    wrap.querySelectorAll(`[data-region="${esc}"], [data-mrow="${esc}"], [data-hcell="${esc}"], .chip[data-identity="${esc}"]`)
      .forEach((c) => c.classList.add("sel-inspect"));
  }, [selectedId, instances]);

  const onEnter = useCallback((id) => setHoverId(id), []);
  const onLeave = useCallback(() => setHoverId(null), []);
  // V4: every drill-in in the Map opens the docked in-map inspector — we never switch to browse.
  const openDetail = useCallback((cls, slot, gen) => onInspect({ kind: "instance", cls, slot, gen: gen ?? 0 }), [onInspect]);
  const openType = useCallback((name) => onInspect({ kind: "type", name }), [onInspect]);
  // functions are first-class Map citizens: clicking one opens a trace view in the same in-map inspector.
  const openFn = useCallback((f) => onInspect({ kind: "function", name: f.name, params: f.params || [], returns: f.returns, module: f.module }), [onInspect]);
  const onChip = useCallback((id) => openInst(model.byId.get(id), openDetail), [model, openDetail]);

  const maxCount = Math.max(1, ...model.census.map((c) => c.count));
  const barW = (n) => Math.max(3, Math.pow(n / maxCount, 0.72) * 100);

  // --- census: live mass by count, OR (skeleton) the static type roster with no bars ---
  // Scoped to the current focus (unless "everywhere"), per 07-modules.md §7 item 4.
  const inScope = (m) => !scoping || inFocus(m, focus);
  const census = (showSkeleton ? typeModel.census : model.census).filter((c) => inScope(c.module));
  const infra = (showSkeleton ? typeModel.infraTypes : model.infraTypes).filter((u) => inScope(u.module));
  const scopedFunctions = functions.filter((f) => inScope(f.module));
  const roots = (showSkeleton ? typeModel.roots : model.roots).filter((r) => inScope(r.module));
  const singletons = showSkeleton
    ? typeModel.singletonTypes.filter((t) => inScope(moduleOf(typeModel.byName.get(t))))
    : model.singletons.filter((s) => inScope(s.module));

  return html`
    <div id="nested" class=${(hoverId ? "id-active " : "") + (showSkeleton ? "skeleton" : "")} ref=${wrapRef}>
      <div class="nested-bar">
        <span class="nested-title">structure <span class="nsub">${showSkeleton ? "ownership = nesting · this is the type-level template live data fills in" : "ownership = nesting · size = mass"}</span></span>
        ${showSkeleton ? html`<span class="schema-affordance"><span class="sdot"></span>schema · not running</span>` : ""}
        ${focus ? html`<span class="focus-badge">
            focused:
            ${focus.split(".").map((seg, i, segs) => {
              const path = segs.slice(0, i + 1).join(".");
              const isLast = i === segs.length - 1;
              return html`<${React.Fragment} key=${path}>
                ${i ? html`<span class="fb-sep">▸</span>` : ""}
                ${isLast
                  ? html`<b>${seg}</b>`
                  : html`<span class="fb-crumb" onClick=${() => setFocus(path)}>${seg}</span>`}
              <//>`;
            })}
            ${everywhere ? html`<span class="fb-everywhere">· showing everywhere</span>` : ""}
            <button class="ghost-btn" onClick=${() => setFocus(null)}>✕</button></span>` : ""}
      </div>

      <div class="census">
        <div class="census-head"><span class="h">${showSkeleton ? "type structure · mass fills at runtime" : "live heap · mass by instance count"}</span>
          <span class="meta">${showSkeleton ? "static schema · 0 instances" : "watching · refresh 800ms"}</span></div>
        <div class="census-grid">
          ${census.map((c) => html`
            <div class=${"cx-row" + (c.util ? " util" : "")} key=${c.qualified}
                 onClick=${() => openType(c.qualified)} title=${"inspect " + c.qualified}>
              <div class="cx-name" title=${c.module ? `${c.name} · ${c.module}` : c.name}>
                <span class="cx-primary">${c.name}</span>${moduleAware && c.module ? html`<span class="cx-mod">${c.module}</span>` : ""}
              </div>
              <div class=${"cx-track" + (showSkeleton ? " tmpl" : "")}>${showSkeleton ? "" : html`<div class=${"cx-bar" + (live[c.qualified] ? " live" : "")} style=${{ width: barW(c.count) + "%" }}></div>`}</div>
              ${showSkeleton
                ? html`<div class="cx-count tmpl">×<b>—</b></div>`
                : html`<div class=${"cx-count" + (live[c.qualified] ? " live" : "")}>×<b>${c.count}</b>${live[c.qualified] ? html`<span class="trend">▲</span>` : ""}</div>`}
            </div>`)}
        </div>
      </div>

      ${showSkeleton
        ? (typeModel.sharedTypes.size ? html`
            <div class="legend"><span class="lbl">shared types</span>
              ${[...typeModel.sharedTypes].sort().map((t) => html`<${TypeChip} key=${t} name=${t} model=${typeModel}
                onEnter=${onEnter} onLeave=${onLeave} onOpen=${openType} focus=${focus} onExpandFocus=${onExpandFocus} />`)}
            </div>` : "")
        : (model.sharedIds.size ? html`
            <div class="legend"><span class="lbl">shared instances</span>
              ${[...model.sharedIds].sort().map((id) => html`<${IdChip} key=${id} id=${id} model=${model}
                onEnter=${onEnter} onLeave=${onLeave} onClick=${onChip} focus=${focus} onExpandFocus=${onExpandFocus} />`)}
            </div>` : "")}

      <div class="stage-wrap">
        ${singletons.length ? html`
          <div class="singletons">
            ${showSkeleton
              ? singletons.map((t) => html`
                  <button class="singleton-obj" key=${t} onClick=${() => openType(t)}>
                    <span class="node-kind">obj</span>${(typeModel.byName.get(t) || {}).name || t} <span class="dim">singleton</span></button>`)
              : singletons.map((s) => html`
                  <button class="singleton-obj" key=${s.ref} onClick=${() => openInst(s, openDetail)}>
                    <span class="node-kind">obj</span>${s.type} <span class="dim">×1</span></button>`)}
          </div>` : ""}

        ${showSkeleton ? html`
          ${roots.length === 0 ? html`<div class="stage-empty">nothing in focus.</div>` : ""}
          ${showModuleChrome
            ? modTree.map((n) => html`<${ModuleRing} key=${n.path} node=${n}
                rootsByModule=${(() => { const m = new Map(); for (const r of roots) (m.get(r.module) || m.set(r.module, []).get(r.module)).push(
                  html`<div class="orch" key=${r.name}><${TypeRegion} node=${r} model=${typeModel} depth=${0}
                    onEnter=${onEnter} onLeave=${onLeave} onChip=${openType} onOpenType=${openType}
                    viewsByType=${viewsByType} viewMode=${viewMode} setViewMode=${setViewMode} focus=${focus} onExpandFocus=${onExpandFocus} /></div>`); return m; })()}
                renderRoot=${(x) => x} depth=${0} collapsed=${ringCollapsed} toggle=${toggleRing} setFocus=${setFocus} />`)
            : roots.map((r) => html`
                <div class="orch" key=${r.name}>
                  <${TypeRegion} node=${r} model=${typeModel} depth=${0}
                    onEnter=${onEnter} onLeave=${onLeave} onChip=${openType} onOpenType=${openType}
                    viewsByType=${viewsByType} viewMode=${viewMode} setViewMode=${setViewMode} focus=${focus} onExpandFocus=${onExpandFocus} />
                </div>`)}
        ` : html`
          ${roots.length === 0 ? html`
            <div class="stage-empty">${instances.length === 0 ? "no live instances yet — run the program, or switch to " : "nothing in focus — switch to "}<b>List</b> to browse types.</div>` : ""}
          ${showModuleChrome
            ? modTree.map((n) => html`<${ModuleRing} key=${n.path} node=${n}
                rootsByModule=${(() => { const m = new Map(); for (const r of roots) (m.get(r.module) || m.set(r.module, []).get(r.module)).push(
                  html`<div class="orch" key=${r.id}><${Region} node=${r} model=${model} depth=${0}
                    onEnter=${onEnter} onLeave=${onLeave} onChip=${onChip} onOpen=${openDetail}
                    viewsByType=${viewsByType} viewMode=${viewMode} setViewMode=${setViewMode} focus=${focus} onExpandFocus=${onExpandFocus} /></div>`); return m; })()}
                renderRoot=${(x) => x} depth=${0} collapsed=${ringCollapsed} toggle=${toggleRing} setFocus=${setFocus} />`)
            : roots.map((r) => html`
                <div class="orch" key=${r.id}>
                  <${Region} node=${r} model=${model} depth=${0}
                    onEnter=${onEnter} onLeave=${onLeave} onChip=${onChip} onOpen=${openDetail}
                    viewsByType=${viewsByType} viewMode=${viewMode} setViewMode=${setViewMode} focus=${focus} onExpandFocus=${onExpandFocus} />
                </div>`)}
        `}

        ${infra.length ? html`
          <div class=${"infra" + (infraOpen ? " open" : "")}>
            <div class="infra-head" onClick=${() => setInfraOpen((v) => !v)}>
              <span class="caret">▸</span><span class="k">infrastructure</span>
              <span class="sub">${infra.length} utility type${infra.length === 1 ? "" : "s"} · faded — click to ${infraOpen ? "collapse" : "expand"}</span>
            </div>
            <div class="infra-body">
              <p class="infra-note">Transport &amp; parsing noise — present and browsable, but they don't own domain state, so the default view keeps them out of the way.</p>
              <div class="infra-grid">
                ${infra.map((u) => html`
                  <button class="util" key=${u.qualified} onClick=${() => openType(u.qualified)}>
                    <span class="un" title=${u.module ? `${u.name} · ${u.module}` : u.name}>${u.name}${moduleAware && u.module ? html`<span class="cx-mod">${u.module}</span>` : ""}</span>
                    <span class=${"uc" + (live[u.qualified] ? " live" : "")}>${showSkeleton ? "type" : "×" + u.count}${live[u.qualified] ? " ▲" : ""}</span>
                  </button>`)}
              </div>
            </div>
          </div>` : ""}

        ${scopedFunctions.length ? html`
          <div class="fnsec">
            <div class="fnsec-head">
              <span class="k">functions</span>
              <span class="sub">${scopedFunctions.length} top-level function${scopedFunctions.length === 1 ? "" : "s"} · ${showSkeleton ? "run the program to trace a call — click to open" : "click one to trace a call"}</span>
            </div>
            <div class="fnsec-grid">
              ${scopedFunctions.map((f) => html`
                <button class="fn-item" key=${f.qualified || f.name} onClick=${() => openFn(f)} title=${"trace " + f.name}>
                  <span class="fn-name">${f.name}</span>${moduleAware && f.module ? html`<span class="fn-mod">${f.module}</span>` : ""}
                  <span class="fn-sig">(${(f.params || []).map((p) => `${p.name}: ${cleanType(p.type)}`).join(", ")})</span>
                  <span class="fn-ret">→ ${f.returns}</span>
                </button>`)}
            </div>
          </div>` : ""}
      </div>
    </div>`;
}

// ===================== app root =====================
// ===================== Function trace inspector (Phase F1/F2) =====================
// A FUNCTION is a first-class Map citizen: clicking one in the Map's `functions` section opens
// THIS view in the in-map inspector. The COMPUTATION counterpart to an entity's DATA detail — it
// evals `trace(<call>)` (the server runs it with the call recorder on) and renders the returned
// call tree as a collapsible recursion tree — each node `fn(args) = result` — plus a stats strip
// (total calls, per-function counts, max depth, truncation). The trace input is pre-filled from
// the function's signature (`name(a, b)`), so a 0-arg function traces in one click and a
// parametered one is ready to have its args typed. Statically (a portal project with no process,
// where `trace(...)` returns StaticInspection) we show the signature + a "run the program to
// trace" note instead of a dead/empty tree. `scry inspect` runs pure functions live, so tracing
// still works there. Re-tracing on submit; the flat node list is assembled into a tree by parent.
function fnCallExpr(fn) { return `${fn.name}(${(fn.params || []).map((p) => p.name).join(", ")})`; }
function FnTraceView({ fn }) {
  const zeroArg = !(fn.params && fn.params.length);
  const [expr, setExpr] = useState(() => fnCallExpr(fn));
  const [submitted, setSubmitted] = useState(() => (zeroArg ? fnCallExpr(fn) : ""));
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [staticNote, setStaticNote] = useState(false);
  const [busy, setBusy] = useState(false);
  const [collapsed, setCollapsed] = useState(() => new Set());

  // when the inspected function changes, reset the input to its signature; auto-trace 0-arg fns.
  useEffect(() => {
    const ini = fnCallExpr(fn);
    setExpr(ini); setData(null); setError(null); setStaticNote(false);
    setSubmitted(zeroArg ? ini : "");
  }, [fn.name]); // eslint-disable-line

  const runTrace = useCallback(async (src) => {
    src = (src || "").trim();
    if (!src) return;
    setBusy(true); setError(null); setStaticNote(false);
    // an ordinary (non reflect-static) bare function call resolves via normal §4 rules — own
    // module, then imports — so a function from a non-entry module needs its OWN module's header
    // or it reads as "unknown name" (functions() now carries `.module` for exactly this).
    const r = await evalSource(moduleHeader(fn.module) + `trace(${src})`);
    setBusy(false);
    // static/portal-static: the program isn't running, so a trace can't execute — show the
    // signature + a "run to trace" note rather than surfacing the raw StaticInspection error.
    if (r.error && r.error.kind === "StaticInspection") { setStaticNote(true); setData(null); return; }
    if (r.error || !r.value || r.value.type !== "Trace") {
      setError(r.error || { kind: "BadResult", message: "not a trace result" });
      setData(null); return;
    }
    const v = r.value;
    // large trees: start collapsed below depth 1 so the browser stays responsive; small ones expand fully.
    const init = new Set();
    if (v.nodes.length > 300) for (const n of v.nodes) if (n.depth >= 1) init.add(n.i);
    setCollapsed(init);
    setData(v);
  }, []);

  useEffect(() => { if (submitted) runTrace(submitted); }, [submitted, runTrace]);

  const kids = useMemo(() => {
    const m = new Map();
    if (data && data.nodes) for (const n of data.nodes) {
      if (!m.has(n.parent)) m.set(n.parent, []);
      m.get(n.parent).push(n);
    }
    return m;
  }, [data]);

  const toggle = useCallback((i) => setCollapsed((s) => {
    const n = new Set(s); if (n.has(i)) n.delete(i); else n.add(i); return n;
  }), []);
  const expandAll = useCallback(() => setCollapsed(new Set()), []);
  const collapseAll = useCallback(() => setCollapsed(() => {
    const n = new Set(); if (data && data.nodes) for (const x of data.nodes) if (kids.has(x.i)) n.add(x.i); return n;
  }), [data, kids]);

  const roots = kids.get(-1) || [];
  const submit = () => setSubmitted(expr.trim());
  const sig = `(${(fn.params || []).map((p) => `${p.name}: ${cleanType(p.type)}`).join(", ")})`;

  return html`
    <div class="trace-panel fn-trace">
      <div class="pane-title">${fn.name}<span class="fn-sig">${sig} → ${fn.returns}</span></div>
      <div class="pane-sub">function · trace a call to watch its recursion tree</div>
      <div class="trace-controls">
        <span class="trace-prompt">trace(</span>
        <input class="trace-input" type="text" spellcheck="false" autocomplete="off"
               placeholder=${fnCallExpr(fn)} value=${expr}
               onInput=${(e) => setExpr(e.target.value)}
               onKeyDown=${(e) => { if (e.key === "Enter") submit(); }} />
        <span class="trace-prompt">)</span>
        <button class="trace-run" onClick=${submit} disabled=${busy}>${busy ? "running…" : "trace"}</button>
        ${data ? html`<span class="trace-spacer"></span>
          <button class="ghost-btn tsm" onClick=${expandAll}>expand all</button>
          <button class="ghost-btn tsm" onClick=${collapseAll}>collapse all</button>` : ""}
      </div>
      ${staticNote ? html`<div class="trace-note">▷ run the program to trace a call — this is a static view (no live process).</div>` : ""}
      ${error ? html`<div class="trace-error"><span class="te-kind">${error.kind}</span> ${error.message}</div>` : ""}
      ${data ? html`
        <${TraceStats} stats=${data.stats} value=${data.value} expr=${submitted} />
        <div class="trace-tree">
          ${roots.length ? roots.map((n) => html`<${TraceNodeRow} key=${n.i} node=${n} kids=${kids} collapsed=${collapsed} toggle=${toggle} />`)
            : html`<div class="trace-empty">no function calls — <code>${submitted}</code> evaluated without calling any user function.</div>`}
        </div>`
        : (busy ? html`<div class="trace-empty">tracing…</div>`
           : (!error && !staticNote ? html`<div class="trace-empty">edit the arguments above and press <b>trace</b>.</div>` : ""))}
    </div>`;
}

function TraceStats({ stats, value, expr }) {
  const perFn = (stats.perFn || []).slice().sort((a, b) => b.calls - a.calls);
  const maxCalls = perFn.length ? perFn[0].calls : 1;
  return html`
    <div class="trace-stats">
      <div class="ts-headline">
        <code class="ts-expr">${expr}</code>
        <span class="ts-eq">=</span>
        <span class="ts-result"><${ValueView} v=${value} /></span>
      </div>
      <div class="ts-strip">
        <div class="ts-tile"><div class="ts-num">${stats.totalCalls}</div><div class="ts-lbl">calls</div></div>
        <div class="ts-tile"><div class="ts-num">${stats.maxDepth}</div><div class="ts-lbl">max depth</div></div>
        <div class="ts-tile"><div class="ts-num">${perFn.length}</div><div class="ts-lbl">functions</div></div>
        ${stats.truncated ? html`<div class="ts-tile ts-warn"><div class="ts-num">${stats.nodeCount}</div><div class="ts-lbl">shown (truncated)</div></div>` : ""}
      </div>
      <div class="ts-perfn">
        ${perFn.map((f) => html`
          <div class="ts-fnrow" key=${f.fn}>
            <span class="ts-fnname">${f.fn}</span>
            <span class="ts-fnbar"><span class="ts-fnfill" style=${{ width: Math.max(4, (f.calls / maxCalls) * 100) + "%" }}></span></span>
            <span class="ts-fncount">×${f.calls}</span>
          </div>`)}
      </div>
    </div>`;
}

function TraceNodeRow({ node, kids, collapsed, toggle }) {
  const children = kids.get(node.i) || [];
  const hasKids = children.length > 0;
  const isCollapsed = collapsed.has(node.i);
  return html`
    <div class="tnode">
      <div class=${"tnode-row" + (hasKids ? " has-kids" : "")} onClick=${() => hasKids && toggle(node.i)}>
        <span class="tnode-tw">${hasKids ? (isCollapsed ? "▶" : "▼") : ""}</span>
        <span class="tnode-fn">${node.fn}</span>
        <span class="tnode-paren">(</span>
        ${(node.args || []).map((a, i) => html`<${React.Fragment} key=${i}>${i ? html`<span class="tnode-comma">, </span>` : ""}<${ValueView} v=${a} inline=${false} /><//>`)}
        <span class="tnode-paren">)</span>
        <span class="tnode-eq"> = </span>
        <span class="tnode-res">${node.hasResult ? html`<${ValueView} v=${node.result} inline=${false} />` : html`<span class="v-void">…</span>`}</span>
        ${hasKids && isCollapsed ? html`<span class="tnode-badge">${subtreeCount(node, kids)}</span>` : ""}
      </div>
      ${hasKids && !isCollapsed ? html`<div class="tnode-kids">
        ${children.map((c) => html`<${TraceNodeRow} key=${c.i} node=${c} kids=${kids} collapsed=${collapsed} toggle=${toggle} />`)}
      </div>` : ""}
    </div>`;
}

// total descendants of a node (shown on a collapsed node so mass stays legible)
function subtreeCount(node, kids) {
  let total = 0;
  const stack = [...(kids.get(node.i) || [])];
  while (stack.length) { const n = stack.pop(); total++; const c = kids.get(n.i); if (c) for (const x of c) stack.push(x); }
  return total;
}

function App({ onBack, programName } = {}) {
  const [schema, setSchema] = useState([]);
  const [trend, setTrend] = useState({});
  const lastCounts = useRef({});
  const trendRef = useRef({});

  const [route, setRoute] = useState({ view: "index", typeName: null, ref: null });
  const [mode, setMode] = useState("map");   // "map" (bespoke nested landing) | "browse" (rail + panes)
  const [crumbs, setCrumbs] = useState([{ label: "types" }]);
  const [ifaceOpen, setIfaceOpenState] = useState({});
  const [replOpen, setReplOpen] = useState(false);
  const [txOpen, setTxOpen] = useState(false);
  const [codeSession, setCodeSession] = useState(null);   // {cls, sc}
  const [codeText, setCodeText] = useState("");
  const codeDraftCls = useRef(null);
  // V4 in-map inspector: a back-stack of targets ({kind:"instance"|...}). null = closed.
  const [inspect, setInspect] = useState(null);
  // Phase 3: module focus + "everywhere" search/filter override, and the modules() tree that
  // drives the rail groups, map rings, and the REPL module picker.
  const [focus, setFocusState] = useState(parseHashFocus);
  const [everywhere, setEverywhere] = useState(false);
  const [modTree, setModTree] = useState([]);
  useEffect(() => {
    const onHash = () => setFocusState(parseHashFocus());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  const setFocus = useCallback((path) => { setFocusState(path || null); writeHashFocus(path || null); }, []);
  usePoll(async () => {
    const r = await evalSource("modules()");
    if (r.value && r.value.children) setModTree(r.value.children);
  }, 3000, []);

  // rail: refresh type counts every 500ms, always (matches vanilla)
  usePoll(async () => {
    const r = await evalSource("types()");
    if (!r.value) return;
    const items = r.value.items || [];
    const nt = {};
    for (const t of items) {
      // CRITICAL rekey (07-modules.md §7): trend keyed by `qualified`, never bare `name`.
      const prev = lastCounts.current[t.qualified];
      if (prev != null && t.liveCount !== prev) nt[t.qualified] = t.liveCount > prev ? "up" : "down";
      else nt[t.qualified] = trendRef.current[t.qualified] || "";
      lastCounts.current[t.qualified] = t.liveCount;
    }
    trendRef.current = nt;
    setTrend(nt);
    setSchema(items);
  }, 500, []);

  // types() (above) drives the rail's live counts/trend, but its nodes carry no `kind`/`variants`
  // and omit enums — so the DetailPane's typed ArgInput widgets need the RICHER schema() payload
  // (kind + enum variants + interface implementors) to render the right per-param picker.
  const [fullSchema, setFullSchema] = useState([]);
  usePoll(async () => {
    const r = await evalSource("schema()");
    if (r.value && r.value.nodes) setFullSchema(r.value.nodes);
  }, 1500, []);

  const setIfaceOpen = useCallback((iface, v) => setIfaceOpenState((m) => ({ ...m, [iface]: v })), []);

  // breadcrumb/rail labels always show the short (bare-unless-ambiguous) display name; `name`/
  // `cls` themselves are QUALIFIED (the CRITICAL rekey — every route/nav identifier from here on).
  const labelFor = useCallback((qualified) => {
    const sc = schema.find((t) => t.qualified === qualified);
    return sc ? shortLabel(sc, schema) : qualified;
  }, [schema]);

  const goIndex = useCallback(() => {
    setRoute({ view: "index", typeName: null, ref: null });
    setCrumbs([{ label: "types" }]);
  }, []);
  const openTable = useCallback((name) => {
    setRoute({ view: "table", typeName: name, ref: null });
    setCrumbs([{ label: "types", target: { kind: "index" } }, { label: labelFor(name) }]);
  }, [labelFor]);
  const openDetail = useCallback((cls, slot, gen, pushCrumb) => {
    const crumbLabel = `${labelFor(cls)}#${slot}`;
    setRoute({ view: "detail", typeName: cls, ref: { class: cls, slot, gen } });
    setCrumbs((prev) => {
      if (!pushCrumb) return prev.length ? prev : [{ label: "types", target: { kind: "index" } }, { label: crumbLabel, target: { kind: "detail", cls, slot, gen } }];
      const base = prev.length && prev[0].label === "types" && prev[0].target ? prev : [{ label: "types", target: { kind: "index" } }];
      const filtered = base.filter((c) => c.label !== crumbLabel);
      filtered.push({ label: crumbLabel, target: { kind: "detail", cls, slot, gen } });
      return filtered;
    });
  }, [labelFor]);
  const navigateRef = useCallback((v) => {
    const m = /^(.+)#(\d+)$/.exec(v.ref);
    if (!m) return;
    // a value's "ref"/"class"/"type" stay BARE on the wire (serialize-entity, flagged in the
    // Phase 3 report as a deliberately-not-fixed gap) — best-effort resolve to qualified via the
    // rail's schema so navigation stays collision-safe whenever the bare name is unambiguous.
    const bare = v.class || v.type;
    const sc = schema.find((t) => t.name === bare);
    openDetail(sc ? sc.qualified : bare, +m[2], v.generation ?? 0, true);
  }, [openDetail, schema]);
  const goCrumb = useCallback((target) => {
    if (target.kind === "index") goIndex();
    else if (target.kind === "table") openTable(target.name);
    else if (target.kind === "detail") openDetail(target.cls, target.slot, target.gen, false);
  }, [goIndex, openTable, openDetail]);

  const openCodePanel = useCallback((cls, sc) => {
    if (codeDraftCls.current !== cls) { setCodeText(classSkeleton(cls, sc)); codeDraftCls.current = cls; }
    setCodeSession({ cls, sc });
  }, []);

  // Search scopes to the current focus unless "everywhere" is on (07-modules.md §7 item 4).
  const globalSearch = useCallback(async (q) => {
    q = q.trim();
    if (!q) return;
    const scoped = focus && !everywhere ? schema.filter((t) => inFocus(moduleOf(t), focus)) : schema;
    const m = /^([A-Za-z_][\w.<>,]*)#(\d+)$/.exec(q);
    if (m) {
      const sc = scoped.find((t) => t.name === m[1] || t.qualified === m[1]);
      openDetail(sc ? sc.qualified : m[1], +m[2], 0, true);
      return;
    }
    for (const t of scoped) {
      const src = moduleHeader(t.module) + `${t.name}.instances(filter: "", offset: 0, limit: 200)`;
      const r = await evalSource(src);
      const hit = (r.value?.items || []).find((it) => JSON.stringify(it.fields).toLowerCase().includes(q.toLowerCase()));
      if (hit) { const mm = /#(\d+)$/.exec(hit.ref); openDetail(t.qualified, +mm[1], hit.generation, true); return; }
    }
  }, [schema, openDetail, focus, everywhere]);

  // global keys: backtick toggles repl (unless typing in another input); Esc closes repl and,
  // if no repl was open (or after closing it), clears module focus (07-modules.md §7 item 2).
  useEffect(() => {
    const onKey = (e) => {
      const ae = document.activeElement;
      const inInput = ae && (ae.id === "repl-input" || /input|textarea/i.test(ae.tagName));
      if (e.key === "`") {
        if (!inInput) { e.preventDefault(); setReplOpen((o) => !o); }
        else if (ae.id === "repl-input" && ae.value === "") { e.preventDefault(); setReplOpen((o) => !o); }
      } else if (e.key === "Escape") {
        setReplOpen(false);
        if (!inInput) setFocus(null);
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);

  const nav = useMemo(() => ({ openTable, openDetail, goIndex, navigateRef, goCrumb }),
    [openTable, openDetail, goIndex, navigateRef, goCrumb]);

  // V4: clicking ANY entity in the Map opens the in-map inspector — never switches to browse.
  const openInspect = useCallback((t) => setInspect({ stack: [t] }), []);
  const inspectPush = useCallback((t) => setInspect((s) => s ? { stack: [...s.stack, t] } : { stack: [t] }), []);
  const inspectGoto = useCallback((i) => setInspect((s) => s && i < s.stack.length ? { stack: s.stack.slice(0, i + 1) } : s), []);
  const inspectBack = useCallback(() => setInspect((s) => s && s.stack.length > 1 ? { stack: s.stack.slice(0, -1) } : null), []);
  const inspectClose = useCallback(() => setInspect(null), []);
  // switching to browse (List) closes the inspector; it belongs to the Map.
  const changeMode = useCallback((m) => { setMode(m); if (m !== "map") setInspect(null); }, []);

  const inspTop = inspect && inspect.stack[inspect.stack.length - 1];
  // selectedId must match graph()'s BARE ref id format ("Type#slot" — data-region/data-mrow/chip
  // data-identity are all still bare there, Phase 3 left graph()'s ref/type id shape unchanged),
  // so resolve inspTop.cls (qualified) back to its bare display name via schema for the DOM match.
  const inspBareCls = inspTop && inspTop.kind === "instance"
    ? ((fullSchema.length ? fullSchema : schema).find((t) => t.qualified === inspTop.cls) || {}).name || inspTop.cls
    : null;
  const inspSelId = inspTop && inspTop.kind === "instance" ? `${inspBareCls}#${inspTop.slot}` : null;
  // the shared bottom REPL dock binds `self` to the inspector's open instance while it's up in Map.
  const replRoute = (mode === "map" && inspTop && inspTop.kind === "instance")
    ? { view: "detail", ref: { class: inspTop.cls, slot: inspTop.slot, gen: inspTop.gen } }
    : route;

  let pane;
  if (route.view === "table") pane = html`<${TablePane} name=${route.typeName} schema=${schema} />`;
  else if (route.view === "detail") pane = html`<${DetailPane} cls=${route.ref.class} slot=${route.ref.slot} gen=${route.ref.gen} schema=${fullSchema.length ? fullSchema : schema} onEditSource=${openCodePanel} />`;
  else pane = html`<${IndexPane} />`;

  return html`
    <${NavContext.Provider} value=${nav}>
      <${TopBar} onGlobalSearch=${globalSearch} onToggleTranscript=${() => setTxOpen((o) => !o)} mode=${mode} setMode=${changeMode} onBack=${onBack} programName=${programName}
        focus=${focus} everywhere=${everywhere} setEverywhere=${setEverywhere} />
      ${mode === "map"
        ? html`<div id="layout" class=${"nested-layout" + (inspect ? " has-inspector" : "")}>
            <div class="nested-stage-col"><${NestedView} onInspect=${openInspect} selectedId=${inspSelId}
                modTree=${modTree} focus=${focus} setFocus=${setFocus} everywhere=${everywhere} /></div>
            ${inspect ? html`<${InspectorPanel} stack=${inspect.stack} schema=${fullSchema.length ? fullSchema : schema} onEditSource=${openCodePanel}
                onNavRef=${inspectPush} onCrumb=${inspectGoto} onBack=${inspectBack} onClose=${inspectClose} nav=${nav} />` : ""}
          </div>`
        : html`<div id="layout">
            <${TypeRail} schema=${schema} trend=${trend} route=${route} ifaceOpen=${ifaceOpen} setIfaceOpen=${setIfaceOpen}
                modTree=${modTree} focus=${focus} setFocus=${setFocus} everywhere=${everywhere} />
            <main id="content">
              <${Breadcrumbs} crumbs=${crumbs} />
              <div id="pane">${pane}</div>
            </main>
          </div>`}
      <${ReplDock} open=${replOpen} setOpen=${setReplOpen} route=${replRoute} modTree=${modTree} schema=${fullSchema.length ? fullSchema : schema} />
      <${CodePanel} session=${codeSession} text=${codeText} setText=${setCodeText} onClose=${() => setCodeSession(null)} />
      <${TranscriptDrawer} open=${txOpen} onClose=${() => setTxOpen(false)} />
    <//>`;
}

// ===================== portal landing =====================
// A card per registered program (GET /api/programs, polled ~1s so cards POP UP when a program
// launches and grey when it exits). Each card fetches a cheap types() through the proxy to show
// live counts. Clicking a card sets evalBase="/p/<id>" and drops you into the inspector (graph
// default, per Phase 9) for that program.
function ProgramCard({ prog, onOpen }) {
  const [stats, setStats] = useState(null);   // { types, instances }
  const exited = prog.status === "exited";
  usePoll(async () => {
    if (exited) return;
    try {
      const res = await fetch(`/p/${prog.id}/eval`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: "card", source: "types()" }),
      });
      const json = await res.json();
      const items = json.value?.items || [];
      setStats({ types: items.length, instances: items.reduce((s, t) => s + (t.liveCount || 0), 0) });
    } catch (e) { /* leave last-known stats */ }
  }, 1500, [prog.id, exited]);
  const started = prog.startTime ? new Date(prog.startTime * 1000).toLocaleTimeString() : "";
  return html`
    <button class=${"pcard" + (exited ? " exited" : "")} disabled=${exited}
            onClick=${() => !exited && onOpen(prog)}>
      <div class="pcard-top">
        <span class="pcard-name">${prog.name}</span>
        <span class=${"pcard-badge " + prog.mode}>${prog.mode === "inspect" ? "inspect" : "running"}</span>
      </div>
      <div class="pcard-meta">
        <span class=${"pcard-status " + prog.status}>${exited ? "exited" : "live"}</span>
        <span class="pcard-port">:${prog.port}</span>
        ${started ? html`<span class="pcard-time">${started}</span>` : ""}
      </div>
      <div class="pcard-stats">
        ${exited
          ? html`<span class="pcard-dim">program gone</span>`
          : stats
            ? html`<span><b>${stats.instances}</b> instance${stats.instances === 1 ? "" : "s"}</span><span class="pcard-dot">·</span><span><b>${stats.types}</b> types</span>`
            : html`<span class="pcard-dim">loading…</span>`}
      </div>
    </button>`;
}

// A card per DISCOVERED project (GET /api/projects). A project is STATIC — pure function of its
// source (typecheck -> schema/views/actions), served by the portal from a cached `scry schema-json`
// dump with NO running process (DECISIONS #16). Clicking sets evalBase="/proj/<id>" and opens the
// SAME inspector; since graph() is empty there, NestedView renders the type-skeleton (schema · not
// running). If the project is ALSO running live, a jump affordance hops to its live program card.
function ProjectCard({ proj, onOpen, onJumpLive }) {
  const [nodes, setNodes] = useState(null);
  useEffect(() => {
    let alive = true;
    fetch(`/proj/${proj.id}/eval`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: "card", source: "schema()" }),
    }).then((r) => r.json()).then((j) => { if (alive) setNodes((j.value?.nodes || []).length); })
      .catch(() => {});
    return () => { alive = false; };
  }, [proj.id, proj.mtime]);
  const live = proj.alsoRunning != null && proj.alsoRunning !== undefined && proj.alsoRunning !== null;
  return html`
    <button class="pcard project" onClick=${() => onOpen(proj)}>
      <div class="pcard-top">
        <span class="pcard-name">${proj.name}</span>
        <span class="pcard-badge inspect">schema</span>
      </div>
      <div class="pcard-meta">
        <span class="pcard-status static">static</span>
        ${live
          ? html`<span class="pcard-live-jump" onClick=${(e) => { e.stopPropagation(); onJumpLive(proj); }}>● live — jump</span>`
          : ""}
      </div>
      <div class="pcard-stats">
        ${nodes != null
          ? html`<span><b>${nodes}</b> type${nodes === 1 ? "" : "s"}</span><span class="pcard-dot">·</span><span class="pcard-dim">no process</span>`
          : html`<span class="pcard-dim">schema…</span>`}
      </div>
    </button>`;
}

function Landing({ onOpen, onOpenProject }) {
  const [progs, setProgs] = useState([]);
  const [projects, setProjects] = useState([]);
  usePoll(async () => {
    try {
      const r = await fetch("/api/programs");
      if (r.status === 200) setProgs(await r.json());
    } catch (e) { /* keep last list */ }
    try {
      const rp = await fetch("/api/projects");
      if (rp.status === 200) setProjects(await rp.json());
    } catch (e) { /* keep last list */ }
  }, 1000, []);
  // jump from a project card to its live program card (alsoRunning = the running program's id)
  const jumpLive = useCallback((proj) => {
    const p = progs.find((x) => x.id === proj.alsoRunning);
    if (p) onOpen(p);
  }, [progs, onOpen]);
  return html`
    <div id="portal-landing">
      <header id="portal-head">
        <div class="brand">scry<span class="brand-sub">portal</span></div>
        <div class="portal-tag">running programs & known projects — click a card to drill in</div>
      </header>
      <section class="portal-section">
        <div class="portal-section-head">running <span class="psh-sub">live processes — pop up on ${"`scry run`"}</span></div>
        ${progs.length === 0
          ? html`<div id="portal-empty">
              <div class="pe-title">no programs running</div>
              <div class="pe-sub">run one in a terminal — it pops up here:</div>
              <code class="pe-cmd">scry run examples/assistant.scry</code>
            </div>`
          : html`<div id="portal-grid">
              ${progs.map((p) => html`<${ProgramCard} key=${p.id} prog=${p} onOpen=${onOpen} />`)}
            </div>`}
      </section>
      <section class="portal-section">
        <div class="portal-section-head">projects <span class="psh-sub">discovered — statically inspectable anytime, no process</span></div>
        ${projects.length === 0
          ? html`<div class="portal-empty-slim">no .scry projects discovered in the portal's working tree</div>`
          : html`<div id="portal-grid">
              ${projects.map((p) => html`<${ProjectCard} key=${p.id} proj=${p} onOpen=${onOpenProject} onJumpLive=${jumpLive} />`)}
            </div>`}
      </section>
    </div>`;
}

// ===================== dual-mode root =====================
// Decide portal vs standalone ONCE: if GET /api/programs is 200 we are served by the portal ->
// show the landing grid; a 404 means a program is serving its own viewer directly -> go straight
// to the inspector with evalBase="" (today's behavior). One app, both modes.
function Root() {
  const [portalMode, setPortalMode] = useState(null);   // null=probing, true, false
  const [selected, setSelected] = useState(null);       // the opened program {id,name,mode}
  useEffect(() => {
    fetch("/api/programs")
      .then((r) => setPortalMode(r.status === 200))
      .catch(() => setPortalMode(false));
  }, []);
  const openProg = useCallback((p) => { setEvalBase(`/p/${p.id}`); setSelected(p); }, []);
  // open a STATIC project: evalBase="/proj/<id>". graph() is empty there so the inspector shows
  // the type-skeleton; non-static evals surface the StaticInspection error gracefully.
  const openProject = useCallback((p) => { setEvalBase(`/proj/${p.id}`); setSelected({ id: `proj-${p.id}`, name: p.name, static: true }); }, []);
  const back = useCallback(() => { setEvalBase(""); setSelected(null); }, []);
  if (portalMode === null) return html`<div id="root-probe">connecting…</div>`;
  if (!portalMode) return html`<${App} />`;                       // standalone program
  if (!selected) return html`<${Landing} onOpen=${openProg} onOpenProject=${openProject} />`;  // portal landing
  return html`<${App} key=${selected.id} onBack=${back} programName=${selected.name} />`;
}

// A render exception anywhere in the tree must NOT white-screen the viewer — the actual Scry
// program keeps running regardless (the viewer is just a REPL client). Catch it, show a
// recoverable message, and let the user dismiss (re-render) or reload.
class ErrorBoundary extends React.Component {
  constructor(p) { super(p); this.state = { err: null }; }
  static getDerivedStateFromError(err) { return { err }; }
  componentDidCatch(err, info) { try { console.error("viewer render error:", err, info); } catch (_) {} }
  render() {
    if (this.state.err) {
      const msg = String((this.state.err && this.state.err.message) || this.state.err);
      return html`<div class="crash-fallback">
        <div class="crash-title">the viewer hit a rendering error</div>
        <div class="crash-msg">${msg}</div>
        <div class="crash-hint">Your program is still running — this is only a UI glitch. Dismiss to retry, or reload the page.</div>
        <div class="crash-btns">
          <button class="insp-btn" onClick=${() => this.setState({ err: null })}>dismiss</button>
          <button class="insp-btn" onClick=${() => location.reload()}>reload</button>
        </div>
      </div>`;
    }
    return this.props.children;
  }
}
ReactDOM.createRoot(document.getElementById("app")).render(html`<${ErrorBoundary}><${Root} /><//>`);
