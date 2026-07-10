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
    const res = await fetch("/eval", {
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
function TopBar({ onGlobalSearch, onToggleTranscript }) {
  const [q, setQ] = useState("");
  return html`
    <header id="topbar">
      <div class="brand">scry<span class="brand-sub">live viewer</span></div>
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

// ===================== app root =====================
function App() {
  const [schema, setSchema] = useState([]);
  const [trend, setTrend] = useState({});
  const lastCounts = useRef({});
  const trendRef = useRef({});

  const [route, setRoute] = useState({ view: "index", typeName: null, ref: null });
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

  let pane;
  if (route.view === "table") pane = html`<${TablePane} name=${route.typeName} schema=${schema} />`;
  else if (route.view === "detail") pane = html`<${DetailPane} cls=${route.ref.class} slot=${route.ref.slot} gen=${route.ref.gen} schema=${schema} onEditSource=${openCodePanel} />`;
  else pane = html`<${IndexPane} />`;

  return html`
    <${NavContext.Provider} value=${nav}>
      <${TopBar} onGlobalSearch=${globalSearch} onToggleTranscript=${() => setTxOpen((o) => !o)} />
      <div id="layout">
        <${TypeRail} schema=${schema} trend=${trend} route=${route} ifaceOpen=${ifaceOpen} setIfaceOpen=${setIfaceOpen} />
        <main id="content">
          <${Breadcrumbs} crumbs=${crumbs} />
          <div id="pane">${pane}</div>
        </main>
      </div>
      <${ReplDock} open=${replOpen} setOpen=${setReplOpen} route=${route} />
      <${CodePanel} session=${codeSession} text=${codeText} setText=${setCodeText} onClose=${() => setCodeSession(null)} />
      <${TranscriptDrawer} open=${txOpen} onClose=${() => setTxOpen(false)} />
    <//>`;
}

ReactDOM.createRoot(document.getElementById("app")).render(html`<${App} />`);
