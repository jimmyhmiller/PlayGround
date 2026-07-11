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

// ===================== type rail =====================
function TypeRail({ schema, trend, route, ifaceOpen, setIfaceOpen }) {
  const nav = useContext(NavContext);
  const [search, setSearch] = useState("");
  const filt = search.toLowerCase();

  const byIface = {};
  const plain = [];
  for (const t of schema) {
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
    byIface[iface].forEach((t) => shown.add(t.name));
    rows.push(html`
      <li class="iface-group" key=${"iface:" + iface}>
        <div class="type-row iface" onClick=${() => setIfaceOpen(iface, !open)}>
          <span class=${"caret" + (open ? " open" : "")}>▶</span>
          <span class="tname">${iface + " ‹interface›"}</span>
          <span class="count">${total + " live"}</span>
        </div>
        ${open ? html`<div class="iface-children">
          ${byIface[iface].map((t) => html`<${TypeRow} key=${t.name} t=${t} trend=${trend[t.name] || ""} active=${route.view === "table" && route.typeName === t.name} onOpen=${() => nav.openTable(t.name)} />`)}
        </div>` : ""}
      </li>`);
  }
  for (const t of plain) {
    if (shown.has(t.name)) continue;
    if (filt && !t.name.toLowerCase().includes(filt)) continue;
    rows.push(html`<li key=${t.name}><${TypeRow} t=${t} trend=${trend[t.name] || ""} active=${route.view === "table" && route.typeName === t.name} onOpen=${() => nav.openTable(t.name)} /></li>`);
  }

  return html`
    <nav id="rail">
      <div class="rail-head">
        <span>types</span>
        <input id="type-search" type="text" placeholder="filter…" spellcheck="false" autocomplete="off"
               value=${search} onInput=${(e) => setSearch(e.target.value)} />
      </div>
      <ul id="type-list">${rows}</ul>
    </nav>`;
}
function TypeRow({ t, trend, active, onOpen }) {
  const arrow = trend === "up" ? "▲" : trend === "down" ? "▼" : "";
  return html`
    <div class=${"type-row" + (active ? " active" : "")} onClick=${onOpen}>
      <span class="tname">${t.name}</span>
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
function TablePane({ name, schema }) {
  const nav = useContext(NavContext);
  const sc = schema.find((t) => t.name === name);
  const [filterInput, setFilterInput] = useState("");   // what you type
  const [applied, setApplied] = useState("");           // what the poll queries
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  // reset the filter when the type changes
  useEffect(() => { setFilterInput(""); setApplied(""); setData(null); setError(null); }, [name]);

  usePoll(async () => {
    const f = applied.replace(/"/g, '\\"');
    const r = await evalSource(`${name}.instances(filter: "${f}", offset: 0, limit: 200)`);
    if (r.error) { setError(r.error); setData(null); }
    else { setError(null); setData(r.value); }
  }, 750, [name, applied]);

  const items = data ? (data.items || []) : [];
  const cols = sc ? sc.fields.map((f) => f.name) : (items[0] ? Object.keys(items[0].fields) : []);
  const meta = data ? `${data.length} live${data.truncated ? " (showing " + items.length + ")" : ""}` : "";

  return html`
    <div>
      <div class="pane-title">${name}</div>
      <div class="pane-sub">${sc ? `${sc.fields.length} fields · ${sc.methods.length} methods` : ""}</div>
      <div class="tbl-tools">
        <input class="filter-box" spellcheck="false" autocomplete="off"
               placeholder=${'filter, e.g.  name == "coder"  or  status contains "run"'}
               value=${filterInput}
               onInput=${(e) => setFilterInput(e.target.value)}
               onKeyDown=${(e) => { if (e.key === "Enter") setApplied(filterInput); }} />
        <div class="tbl-meta">${meta}</div>
      </div>
      ${error
        ? html`<div class="invoke-result invoke-error">${error.kind}: ${error.message}</div>`
        : !items.length
          ? html`<div class="empty">no live instances match.</div>`
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
  const sc = schema.find((t) => t.name === cls);
  const [inst, setInst] = useState(null);
  const [error, setError] = useState(null);
  const prevFields = useRef(null);      // previous poll's fields, for flash diffing
  const flashKeys = useRef({});         // field name -> nonce; bump => value cell remounts => flash replays

  const fetchDetail = useCallback(async () => {
    const r = await evalSource(`${cls}.at(${slot}, ${gen})`);
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
  }, [cls, slot, gen]);

  // fresh identity => drop the flash baseline so we don't flash the whole record on arrival
  useEffect(() => { prevFields.current = null; flashKeys.current = {}; setInst(null); setError(null); }, [cls, slot, gen]);

  usePoll(fetchDetail, 750, [cls, slot, gen]);

  // let MethodCard / ReplDock ask for an immediate read-back after a mutation
  const fetchRef = useRef(fetchDetail);
  fetchRef.current = fetchDetail;
  useEffect(() => detailBus.subscribe(() => { if (!document.hidden) fetchRef.current(); }), []);

  if (error) {
    return html`<div><div class="pane-title">${cls + "#" + slot}</div>
      <div class="invoke-result invoke-error">${error.kind}: ${error.message}</div></div>`;
  }
  if (!inst) return html`<div><div class="pane-title">${cls + "#" + slot}</div></div>`;

  const typeOf = (n) => sc ? (sc.fields.find((f) => f.name === n) || {}).type : "";
  const implementsLine = sc && sc.implements && sc.implements.length
    ? html` · implements <span class="impl">${sc.implements.join(", ")}</span>` : "";

  return html`
    <div>
      <div class="pane-title">${inst.ref}</div>
      <div class="pane-sub">
        generation ${inst.generation}${implementsLine}
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

      ${sc && sc.methods.length ? html`
        <div class="detail-section">
          <h3>methods</h3>
          ${sc.methods.map((m) => html`<${MethodCard} key=${m.name} cls=${cls} slot=${slot} gen=${gen} m=${m} />`)}
        </div>` : ""}
    </div>`;
}

// A method card owns its own open/args/result state. That state is what the vanilla viewer
// had to snapshot-and-restore across every poll; here it simply lives in the component, so a
// poll re-rendering DetailPane never touches it.
function MethodCard({ cls, slot, gen, m }) {
  const [open, setOpen] = useState(false);
  const [args, setArgs] = useState({});                 // param name -> string
  const [result, setResult] = useState(null);           // {error|value, flash}
  const flash = useRef(0);

  const doInvoke = useCallback(async () => {
    const argList = m.params.map((p) => literalFor(args[p.name] || "", p.type)).join(", ");
    const src = `${cls}.at(${slot}, ${gen}).${m.name}(${argList})`;
    const r = await evalSource(src);
    flash.current += 1;
    setResult({ ...r, flash: flash.current });
    setTimeout(bumpDetail, 60); // read the mutation back immediately
  }, [args, cls, slot, gen, m]);

  const params = m.params.map((p) => `${p.name}: ${p.type}`).join(", ");

  return html`
    <div class=${"method" + (open ? " open" : "")}>
      <div class="method-head" onClick=${() => setOpen((o) => !o)}>
        <span class="method-sig">${m.name}(${params}) <span class="mret">→ ${m.returns}</span></span>
        <button class="invoke-btn" onClick=${(e) => { e.stopPropagation(); if (m.params.length) setOpen(true); doInvoke(); }}>invoke</button>
      </div>
      <div class="method-body">
        ${m.params.map((p) => html`
          <div class="arg-row" key=${p.name}>
            <label>${p.name}: ${p.type}</label>
            <input placeholder=${literalHint(p.type)} value=${args[p.name] || ""}
                   onInput=${(e) => setArgs((a) => ({ ...a, [p.name]: e.target.value }))}
                   onKeyDown=${(e) => { if (e.key === "Enter") doInvoke(); }} />
          </div>`)}
        ${m.params.length ? html`<div class="arg-row"><button class="invoke-btn" onClick=${doInvoke}>run</button></div>` : ""}
        ${result ? html`<${InvokeResult} result=${result} key=${result.flash} />` : ""}
      </div>
    </div>`;
}
function InvokeResult({ result }) {
  if (result.error) {
    const trace = result.error.trace
      ? html`<div class="etrace">${result.error.trace.map((f) => `${f.type}.${f.method} (line ${f.line})`).join(" › ")}</div>` : "";
    return html`<div class="invoke-result flash invoke-error">${result.error.kind}: ${result.error.message}${trace}</div>`;
  }
  return html`<div class="invoke-result flash"><${ValueView} v=${result.value} /></div>`;
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
function ReplDock({ open, setOpen, route }) {
  const [entries, setEntries] = useState([]);       // {expr, out?, error?}
  const [input, setInput] = useState("");
  const scrollRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => { if (open) inputRef.current && inputRef.current.focus(); }, [open]);
  useEffect(() => { if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight; }, [entries]);

  const submit = useCallback((src) => {
    let source = src;
    if (route.view === "detail" && route.ref) {
      const { class: c, slot, gen } = route.ref;
      source = src.replace(/\bself\b/g, `${c}.at(${slot}, ${gen})`);
    }
    const idx = entries.length;
    setEntries((es) => [...es, { expr: src, pending: true }]);
    evalSource(source).then((r) => {
      setEntries((es) => es.map((e, i) => i === idx ? { expr: src, error: r.error, value: r.value } : e));
      if (route.view === "detail") setTimeout(bumpDetail, 60);
    });
  }, [entries.length, route]);

  const ctx = route.view === "detail" && route.ref ? `self = ${route.ref.class}#${route.ref.slot}` : "";

  return html`
    <section id="repl" class=${open ? "" : "collapsed"}>
      <div class="repl-head">
        <span class="repl-title">repl</span>
        <span class="repl-context">${ctx}</span>
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

  const cls = session ? session.cls : "";
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
function TopBar({ onGlobalSearch, onToggleTranscript, mode, setMode, onBack, programName }) {
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
             placeholder="jump to Agent#7 or search field values…"
             value=${q} onInput=${(e) => setQ(e.target.value)}
             onKeyDown=${(e) => { if (e.key === "Enter") onGlobalSearch(q); }} />
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

// The whole derivation: ownership (nesting) vs sharing (identity chips) vs infrastructure,
// from the live instance list + the static schema. Pure + deterministic.
function computeNested(instances, schema) {
  const byId = new Map(instances.map((i) => [i.ref, i]));
  const typeOf = (id) => (byId.get(id) || {}).type;
  const nodes = schema || [];
  const byName = new Map(nodes.map((n) => [n.name, n]));
  const entityTypes = nodes.filter((n) => n.kind === "class" || n.kind === "object").map((n) => n.name);
  const liveCount = (t) => (byName.get(t) || {}).liveCount || 0;

  // ---- static type-reference graph, interfaces expanded to their implementors ----
  const implementorsOf = new Map();
  for (const n of nodes) if (n.kind === "interface") implementorsOf.set(n.name, n.implementors || []);
  const expand = (t) => implementorsOf.has(t) ? implementorsOf.get(t) : [t];
  const refOut = new Map();                       // entity type -> Set(entity type) it can point at
  for (const n of nodes) {
    if (n.kind !== "class" && n.kind !== "object") continue;
    const outs = new Set();
    for (const f of n.fields || []) for (const rt of f.refTypes || []) for (const e of expand(rt)) outs.add(e);
    refOut.set(n.name, outs);
  }
  const referenced = new Set();
  for (const [, outs] of refOut) for (const t of outs) referenced.add(t);
  // worker/scaffolding types (those implementing a BUILTIN interface, i.e. Runnable thread
  // bodies) are execution machinery, not domain roots — they hold a domain object to run it,
  // but the domain owner (Orchestrator) is the real container. Excluding them as root
  // candidates keeps them (and their held instances' second owner) out of the domain spine, so
  // e.g. a sub-agent held by BOTH the Orchestrator and its worker still nests under the
  // Orchestrator instead of counting as "shared". They recede to the infrastructure strip.
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
  const owners = new Map();                        // targetId -> Set(ownerId)
  for (const inst of instances) {
    if (!isDomainInst(inst.ref)) continue;
    for (const r of inst.refs || []) for (const tid of r.ids) {
      if (!isDomainInst(tid) || tid === inst.ref) continue;
      (owners.get(tid) || owners.set(tid, new Set()).get(tid)).add(inst.ref);
    }
  }
  const ownerCount = (id) => (owners.get(id) || new Set()).size;
  const sharedIds = new Set();                     // >= 2 distinct domain owners
  for (const inst of instances) if (isDomainInst(inst.ref) && ownerCount(inst.ref) >= 2) sharedIds.add(inst.ref);

  // ---- nesting tree from the domain roots (inCount 0) ----
  const placed = new Set();
  const buildNode = (id) => {
    const inst = byId.get(id);
    placed.add(id);
    const children = [], chipIds = [];
    for (const r of inst.refs || []) for (const tid of r.ids) {
      if (!byId.has(tid) || tid === id) continue;
      if (sharedIds.has(tid)) { chipIds.push(tid); continue; }
      if (!isDomainInst(tid)) continue;            // reference into infra -> ignore here
      if (ownerCount(tid) === 1 && !placed.has(tid)) children.push(buildNode(tid));
    }
    let subtree = 1;
    for (const c of children) subtree += c.subtree;
    return { id, inst, children, chipIds, subtree };
  };
  const rootIds = instances.filter((i) => isDomainInst(i.ref) && ownerCount(i.ref) === 0).map((i) => i.ref);
  // singleton objects (a leaf like Session): unreferenced, no entity fields, exactly 1 live.
  const singletonTypes = new Set(entityTypes.filter((t) =>
    !domainTypes.has(t) && liveCount(t) === 1 && (refOut.get(t) || new Set()).size === 0 && !referenced.has(t)));
  const singletons = instances.filter((i) => singletonTypes.has(i.type));
  // container roots become the stage; leaf roots that are singleton-objects go to the header.
  const roots = rootIds.filter((id) => !singletonTypes.has(typeOf(id))).map(buildNode)
    .sort((a, b) => b.subtree - a.subtree);

  // ---- identity color: stable slot per shared id (sorted -> index) ----
  const idColor = new Map([...sharedIds].sort().map((id, i) => [id, i % N_ID]));

  // ---- infrastructure types: entity types with no presence in the domain (nor singletons) ----
  const infraTypes = entityTypes
    .filter((t) => !domainTypes.has(t) && !singletonTypes.has(t) && liveCount(t) > 0)
    .map((t) => ({ name: t, count: liveCount(t) }))
    .sort((a, b) => b.count - a.count);
  const infraSet = new Set(infraTypes.map((x) => x.name));

  // ---- census: every entity type with a live instance, mass by count ----
  const census = entityTypes
    .filter((t) => liveCount(t) > 0)
    .map((t) => ({ name: t, count: liveCount(t), util: infraSet.has(t) }))
    .sort((a, b) => b.count - a.count);

  return { byId, roots, singletons, sharedIds, idColor, infraTypes, census, ownerCount };
}

// an identity chip for a shared instance (colored, hover-highlights all its appearances)
function IdChip({ id, model, small, onEnter, onLeave, onClick }) {
  const slot = model.idColor.get(id) ?? 0;
  const inst = model.byId.get(id);
  const label = inst ? (sVal(inst, "name") || sVal(inst, "title") || inst.type) : id;
  const held = model.ownerCount(id);
  return html`
    <button class=${"chip" + (small ? " sm" : "")} data-identity=${id} data-slot=${slot}
            style=${{ "--c": `var(--id-${slot})` }}
            title=${`${id} · shared instance · held by ${held}`}
            onMouseEnter=${() => onEnter(id)} onMouseLeave=${onLeave}
            onClick=${(e) => { e.stopPropagation(); onClick(id); }}>
      <span class="knob"></span>${label}</button>`;
}

// a nested region (recursive): header + owned children + a dense message stack + identity chips
function Region({ node, model, depth, onEnter, onLeave, onChip, onOpen }) {
  const { inst } = node;
  const status = sVal(inst, "status");
  const role = sVal(inst, "role");
  const msgKids = node.children.filter((c) => isMsgLike(c.inst));
  const subKids = node.children.filter((c) => !isMsgLike(c.inst));
  const cls = "region" + (depth === 0 ? " root" : depth === 1 ? " lvl1" : " lvl2");
  const st = status ? String(status).toLowerCase() : "";
  const badgeCls = "badge " + (st.includes("run") ? "running" : st.includes("paus") ? "paused" : st.includes("done") ? "done" : "waiting");
  const p = refParts(node.id);
  return html`
    <div class=${cls} data-region=${node.id}>
      <div class="region-head" onClick=${() => p && onOpen(p.cls, p.slot, inst.generation)}>
        <span class="node-kind">${inst.type}</span>
        <span class="region-name">${displayName(inst)}</span>
        ${role ? html`<span class="region-role">${role}</span>` : ""}
        ${status ? html`<span class=${badgeCls}><span class="bd"></span>${String(status)}</span>` : ""}
      </div>
      ${msgKids.length ? html`
        <div class="conv">
          <div class="conv-label"><span class="k">owns ▸ ${msgKids[0].inst.type}</span>
            <span class="n">count <b>${msgKids.length}</b></span></div>
          <div class="mstack">
            ${msgKids.map((c) => html`<div class=${"mrow " + roleClass(sVal(c.inst, "role"))} key=${c.id}
              data-mrow=${c.id}><span class="tick"></span><span class="fill"></span></div>`)}
          </div>
        </div>` : ""}
      ${subKids.length ? html`
        <div class=${"subregions" + (subKids.length > 1 && depth === 0 ? " grid" : "")}>
          ${subKids.map((c) => html`<${Region} key=${c.id} node=${c} model=${model} depth=${depth + 1}
            onEnter=${onEnter} onLeave=${onLeave} onChip=${onChip} onOpen=${onOpen} />`)}
        </div>` : ""}
      ${node.chipIds.length ? html`
        <div class="refs"><span class="rk">references ▸ shared</span>
          ${node.chipIds.map((id) => html`<${IdChip} key=${id} id=${id} model=${model} small=${true}
            onEnter=${onEnter} onLeave=${onLeave} onClick=${onChip} />`)}
        </div>` : ""}
    </div>`;
}

function NestedView({ onOpenType, onOpenDetail }) {
  const [instances, setInstances] = useState([]);
  const [schema, setSchema] = useState([]);
  const [hoverId, setHoverId] = useState(null);
  const [linksOn, setLinksOn] = useState(false);
  const [infraOpen, setInfraOpen] = useState(false);
  const prevCounts = useRef({});
  const [live, setLive] = useState({});           // type -> true when its count just climbed
  const freshRef = useRef(new Set());             // instance ids seen last poll (for pulse)
  const wrapRef = useRef(null);
  const overlayRef = useRef(null);

  usePoll(async () => {
    const [g, s] = await Promise.all([evalSource("graph()"), evalSource("schema()")]);
    if (s.value && s.value.nodes) {
      const nl = {};
      for (const n of s.value.nodes) {
        const prev = prevCounts.current[n.name];
        nl[n.name] = prev != null && (n.liveCount || 0) > prev;
        prevCounts.current[n.name] = n.liveCount || 0;
      }
      setLive(nl); setSchema(s.value.nodes);
    }
    if (g.value && g.value.instances) setInstances(g.value.instances);
  }, 800, []);

  const model = useMemo(() => computeNested(instances, schema), [instances, schema]);

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

  const onEnter = useCallback((id) => setHoverId(id), []);
  const onLeave = useCallback(() => setHoverId(null), []);
  const openDetail = useCallback((cls, slot, gen) => onOpenDetail(cls, slot, gen ?? 0), [onOpenDetail]);
  const onChip = useCallback((id) => { const p = refParts(id); if (p) openDetail(p.cls, p.slot, (model.byId.get(id) || {}).generation); }, [model, openDetail]);

  // "show links" overlay: hub-and-spoke faded connectors between every appearance of hoverId.
  useEffect(() => {
    const svg = overlayRef.current, wrap = wrapRef.current;
    if (!svg || !wrap) return;
    svg.innerHTML = "";
    if (!linksOn || !hoverId) return;
    const wb = wrap.getBoundingClientRect();
    const pts = [...wrap.querySelectorAll(`.chip[data-identity="${CSS.escape(hoverId)}"]`)].map((c) => {
      const r = c.getBoundingClientRect(); return { x: r.left + r.width / 2 - wb.left, y: r.top + r.height / 2 - wb.top };
    });
    if (pts.length < 2) return;
    const hub = { x: pts.reduce((s, p) => s + p.x, 0) / pts.length, y: pts.reduce((s, p) => s + p.y, 0) / pts.length };
    const slot = model.idColor.get(hoverId) ?? 0;
    for (const p of pts) {
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", `M${p.x},${p.y} Q${(p.x + hub.x) / 2},${p.y} ${hub.x},${hub.y}`);
      path.setAttribute("class", "link show"); path.style.setProperty("--c", `var(--id-${slot})`);
      svg.appendChild(path);
    }
  }, [linksOn, hoverId, model, instances]);

  const maxCount = Math.max(1, ...model.census.map((c) => c.count));
  const barW = (n) => Math.max(3, Math.pow(n / maxCount, 0.72) * 100);

  return html`
    <div id="nested" class=${hoverId ? "id-active" : ""} ref=${wrapRef}>
      <div class="nested-bar">
        <span class="nested-title">structure <span class="nsub">ownership = nesting · size = mass</span></span>
        <button class=${"ctl" + (linksOn ? " on" : "")} onClick=${() => setLinksOn((v) => !v)}>
          <span class="sw"></span>show links</button>
      </div>

      <div class="census">
        <div class="census-head"><span class="h">live heap · mass by instance count</span>
          <span class="meta">watching · refresh 800ms</span></div>
        <div class="census-grid">
          ${model.census.map((c) => html`
            <${React.Fragment} key=${c.name}>
              <div class=${"cx-name" + (c.util ? " util" : "")}>${c.name}</div>
              <div class="cx-track"><div class=${"cx-bar" + (live[c.name] ? " live" : "")} style=${{ width: barW(c.count) + "%" }}></div></div>
              <div class=${"cx-count" + (live[c.name] ? " live" : "")}>×<b>${c.count}</b>${live[c.name] ? html`<span class="trend">▲</span>` : ""}</div>
            <//>`)}
        </div>
      </div>

      ${model.sharedIds.size ? html`
        <div class="legend"><span class="lbl">shared instances</span>
          ${[...model.sharedIds].sort().map((id) => html`<${IdChip} key=${id} id=${id} model=${model}
            onEnter=${onEnter} onLeave=${onLeave} onClick=${onChip} />`)}
        </div>` : ""}

      <div class="stage-wrap">
        <svg class="link-overlay" ref=${overlayRef} aria-hidden="true"></svg>
        ${model.roots.length === 0 ? html`
          <div class="stage-empty">no live instances yet — run the program, or switch to <b>List</b> to browse types.</div>` : ""}
        ${model.roots.map((r) => html`
          <div class="orch" key=${r.id}>
            ${model.singletons.length && r === model.roots[0] ? html`
              <div class="singletons">
                ${model.singletons.map((s) => { const p = refParts(s.ref); return html`
                  <button class="singleton-obj" key=${s.ref} onClick=${() => p && openDetail(p.cls, p.slot, s.generation)}>
                    <span class="node-kind">obj</span>${s.type} <span class="dim">×1</span></button>`; })}
              </div>` : ""}
            <${Region} node=${r} model=${model} depth=${0}
              onEnter=${onEnter} onLeave=${onLeave} onChip=${onChip} onOpen=${openDetail} />
          </div>`)}

        ${model.infraTypes.length ? html`
          <div class=${"infra" + (infraOpen ? " open" : "")}>
            <div class="infra-head" onClick=${() => setInfraOpen((v) => !v)}>
              <span class="caret">▸</span><span class="k">infrastructure</span>
              <span class="sub">${model.infraTypes.length} utility type${model.infraTypes.length === 1 ? "" : "s"} · faded — click to ${infraOpen ? "collapse" : "expand"}</span>
            </div>
            <div class="infra-body">
              <p class="infra-note">Transport &amp; parsing noise — present and browsable, but they don't own domain state, so the default view keeps them out of the way.</p>
              <div class="infra-grid">
                ${model.infraTypes.map((u) => html`
                  <button class="util" key=${u.name} onClick=${() => onOpenType(u.name)}>
                    <span class="un">${u.name}</span>
                    <span class=${"uc" + (live[u.name] ? " live" : "")}>×${u.count}${live[u.name] ? " ▲" : ""}</span>
                  </button>`)}
              </div>
            </div>
          </div>` : ""}
      </div>
    </div>`;
}

// ===================== app root =====================
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

  // rail: refresh type counts every 500ms, always (matches vanilla)
  usePoll(async () => {
    const r = await evalSource("types()");
    if (!r.value) return;
    const items = r.value.items || [];
    const nt = {};
    for (const t of items) {
      const prev = lastCounts.current[t.name];
      if (prev != null && t.liveCount !== prev) nt[t.name] = t.liveCount > prev ? "up" : "down";
      else nt[t.name] = trendRef.current[t.name] || "";
      lastCounts.current[t.name] = t.liveCount;
    }
    trendRef.current = nt;
    setTrend(nt);
    setSchema(items);
  }, 500, []);

  const setIfaceOpen = useCallback((iface, v) => setIfaceOpenState((m) => ({ ...m, [iface]: v })), []);

  const goIndex = useCallback(() => {
    setRoute({ view: "index", typeName: null, ref: null });
    setCrumbs([{ label: "types" }]);
  }, []);
  const openTable = useCallback((name) => {
    setRoute({ view: "table", typeName: name, ref: null });
    setCrumbs([{ label: "types", target: { kind: "index" } }, { label: name }]);
  }, []);
  const openDetail = useCallback((cls, slot, gen, pushCrumb) => {
    const crumbLabel = `${cls}#${slot}`;
    setRoute({ view: "detail", typeName: cls, ref: { class: cls, slot, gen } });
    setCrumbs((prev) => {
      if (!pushCrumb) return prev.length ? prev : [{ label: "types", target: { kind: "index" } }, { label: crumbLabel, target: { kind: "detail", cls, slot, gen } }];
      const base = prev.length && prev[0].label === "types" && prev[0].target ? prev : [{ label: "types", target: { kind: "index" } }];
      const filtered = base.filter((c) => c.label !== crumbLabel);
      filtered.push({ label: crumbLabel, target: { kind: "detail", cls, slot, gen } });
      return filtered;
    });
  }, []);
  const navigateRef = useCallback((v) => {
    const m = /^(.+)#(\d+)$/.exec(v.ref);
    if (!m) return;
    openDetail(v.class || v.type, +m[2], v.generation ?? 0, true);
  }, [openDetail]);
  const goCrumb = useCallback((target) => {
    if (target.kind === "index") goIndex();
    else if (target.kind === "table") openTable(target.name);
    else if (target.kind === "detail") openDetail(target.cls, target.slot, target.gen, false);
  }, [goIndex, openTable, openDetail]);

  const openCodePanel = useCallback((cls, sc) => {
    if (codeDraftCls.current !== cls) { setCodeText(classSkeleton(cls, sc)); codeDraftCls.current = cls; }
    setCodeSession({ cls, sc });
  }, []);

  const globalSearch = useCallback(async (q) => {
    q = q.trim();
    if (!q) return;
    const m = /^([A-Za-z_][\w<>,]*)#(\d+)$/.exec(q);
    if (m) { openDetail(m[1], +m[2], 0, true); return; }
    for (const t of schema) {
      const r = await evalSource(`${t.name}.instances(filter: "", offset: 0, limit: 200)`);
      const hit = (r.value?.items || []).find((it) => JSON.stringify(it.fields).toLowerCase().includes(q.toLowerCase()));
      if (hit) { const mm = /#(\d+)$/.exec(hit.ref); openDetail(t.name, +mm[1], hit.generation, true); return; }
    }
  }, [schema, openDetail]);

  // global keys: backtick toggles repl (unless typing in another input); Esc closes repl
  useEffect(() => {
    const onKey = (e) => {
      const ae = document.activeElement;
      const inInput = ae && (ae.id === "repl-input" || /input|textarea/i.test(ae.tagName));
      if (e.key === "`") {
        if (!inInput) { e.preventDefault(); setReplOpen((o) => !o); }
        else if (ae.id === "repl-input" && ae.value === "") { e.preventDefault(); setReplOpen((o) => !o); }
      } else if (e.key === "Escape") { setReplOpen(false); }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);

  const nav = useMemo(() => ({ openTable, openDetail, goIndex, navigateRef, goCrumb }),
    [openTable, openDetail, goIndex, navigateRef, goCrumb]);

  // nested-view "open a type" (an infra chip click): drill into the browse view's instance table.
  const onGraphOpen = useCallback((name) => { setMode("browse"); openTable(name); }, [openTable]);
  // clicking a region/chip in the nested view drills into that instance's detail (browse mode).
  const onNestOpenDetail = useCallback((cls, slot, gen) => { setMode("browse"); openDetail(cls, slot, gen ?? 0, true); }, [openDetail]);

  let pane;
  if (route.view === "table") pane = html`<${TablePane} name=${route.typeName} schema=${schema} />`;
  else if (route.view === "detail") pane = html`<${DetailPane} cls=${route.ref.class} slot=${route.ref.slot} gen=${route.ref.gen} schema=${schema} onEditSource=${openCodePanel} />`;
  else pane = html`<${IndexPane} />`;

  return html`
    <${NavContext.Provider} value=${nav}>
      <${TopBar} onGlobalSearch=${globalSearch} onToggleTranscript=${() => setTxOpen((o) => !o)} mode=${mode} setMode=${setMode} onBack=${onBack} programName=${programName} />
      ${mode === "map"
        ? html`<div id="layout" class="nested-layout"><${NestedView} onOpenType=${onGraphOpen} onOpenDetail=${onNestOpenDetail} /></div>`
        : html`<div id="layout">
            <${TypeRail} schema=${schema} trend=${trend} route=${route} ifaceOpen=${ifaceOpen} setIfaceOpen=${setIfaceOpen} />
            <main id="content">
              <${Breadcrumbs} crumbs=${crumbs} />
              <div id="pane">${pane}</div>
            </main>
          </div>`}
      <${ReplDock} open=${replOpen} setOpen=${setReplOpen} route=${route} />
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

function Landing({ onOpen }) {
  const [progs, setProgs] = useState([]);
  usePoll(async () => {
    try {
      const r = await fetch("/api/programs");
      if (r.status === 200) setProgs(await r.json());
    } catch (e) { /* keep last list */ }
  }, 1000, []);
  return html`
    <div id="portal-landing">
      <header id="portal-head">
        <div class="brand">scry<span class="brand-sub">portal</span></div>
        <div class="portal-tag">running & inspected programs — click a card to drill in</div>
      </header>
      ${progs.length === 0
        ? html`<div id="portal-empty">
            <div class="pe-title">no programs yet</div>
            <div class="pe-sub">run one in a terminal — it pops up here:</div>
            <code class="pe-cmd">scry run examples/assistant.scry</code>
            <code class="pe-cmd">scry inspect examples/agents.scry</code>
          </div>`
        : html`<div id="portal-grid">
            ${progs.map((p) => html`<${ProgramCard} key=${p.id} prog=${p} onOpen=${onOpen} />`)}
          </div>`}
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
  const back = useCallback(() => { setEvalBase(""); setSelected(null); }, []);
  if (portalMode === null) return html`<div id="root-probe">connecting…</div>`;
  if (!portalMode) return html`<${App} />`;                       // standalone program
  if (!selected) return html`<${Landing} onOpen=${openProg} />`;  // portal landing
  return html`<${App} key=${selected.id} onBack=${back} programName=${selected.name} />`;
}

ReactDOM.createRoot(document.getElementById("app")).render(html`<${Root} />`);
