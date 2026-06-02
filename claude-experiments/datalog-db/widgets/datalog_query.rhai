// datalog_query.rhai — point-and-click query builder for datalog-db.
//
// EVENT-DRIVEN. Rhai widgets can't receive typed text (the host only
// forwards navigation keys), so instead of a free-text query box this
// builds a `find` query by clicking: pick a type, toggle which fields
// to return, optionally count / order / limit, then Run. The query is
// shipped to the `datalog` CLI as a one-shot subprocess and the result
// table is shown verbatim.
//
// For arbitrary hand-written queries or asserts, use the `datalog` CLI;
// this widget is for fast structured exploration.
//
// Scoping: only handlers (on_init/on_click/on_frame/render) touch
// `state`; `fn`s are pure helpers (may call host fns, never `state`).

// ---- one-time state init ----
if !("host" in state)     { state.host = "127.0.0.1:5557"; }
if !("proc" in state)     { state.proc = -1; }
if !("busy" in state)     { state.busy = false; }
if !("job" in state)      { state.job = ""; }
if !("out" in state)      { state.out = []; }
if !("schema" in state)   { state.schema = []; }
if !("types" in state)    { state.types = []; }
if !("loaded" in state)   { state.loaded = false; }
if !("selected" in state) { state.selected = ""; }
if !("fields" in state)   { state.fields = []; }   // field names of selected type
if !("picked" in state)   { state.picked = []; }   // chosen projection columns
if !("count" in state)    { state.count = false; } // count(*) mode
if !("order" in state)    { state.order = ""; }    // order-by field
if !("desc" in state)     { state.desc = false; }
if !("limit" in state)    { state.limit = 50; }    // 0 = no limit
if !("result" in state)   { state.result = []; }
if !("ran" in state)      { state.ran = ""; }      // query string last run

// ============================ helpers ============================

fn dl_path() {
    let home = host_env("HOME");
    if home == "" { "datalog" } else { home + "/.cargo/bin/datalog" }
}

fn spawn_args(host, args) {
    let argv = ["--host", host];
    for a in args { argv.push(a); }
    proc_spawn(dl_path(), argv)
}

fn drain(handle) {
    let lines = [];
    loop { let l = proc_read(handle); if l == "" { break; } lines.push(l); }
    let done = false;
    if !proc_alive(handle) {
        loop { let l = proc_read(handle); if l == "" { break; } lines.push(l); }
        done = true;
    }
    #{ lines: lines, done: done }
}

fn is_name_line(line) {
    line.len() >= 3 && line.starts_with("  ") && !line.starts_with("   ")
}

fn parse_types(lines) {
    let types = [];
    let section = "";
    for line in lines {
        if line == "Entity types:" { section = "types"; }
        else if line == "Enums:"   { section = "enums"; }
        else if is_name_line(line) {
            let name = line.sub_string(2);
            if name != "" && name != "(none)" && section == "types" { types.push(name); }
        }
    }
    types
}

// Field names declared under `type` in the schema dump.
fn field_names(lines, type) {
    let target = "  " + type;
    let out = [];
    let in_block = false;
    for line in lines {
        if line == target { in_block = true; }
        else if in_block {
            if line.starts_with("    ") {
                let f = line.sub_string(4);
                let i = f.index_of(":");
                if i > 0 { out.push(f.sub_string(0, i)); }
            } else { break; }
        }
    }
    out
}

fn join(arr, sep) {
    let s = ""; let first = true;
    for x in arr { if first { s = x; first = false; } else { s += sep + x; } }
    s
}

fn has(arr, v) {
    for x in arr { if x == v { return true; } }
    false
}

fn without(arr, v) {
    let out = [];
    for x in arr { if x != v { out.push(x); } }
    out
}

// Build a `find` query string from the current selection. `picked` is
// the projection; if empty, all `fields` are used so the query is valid.
fn build_query(type, picked, fields, count, order, desc, limit) {
    let cols = if picked.len() > 0 { picked } else { fields };
    let binds = [];
    let finds = [];
    for f in cols { binds.push(f + ": ?" + f); finds.push("?" + f); }
    let find_part = if count { "count(*)" } else { join(finds, ", ") };
    let q = "find " + find_part + " where ?e: " + type + " { " + join(binds, ", ") + " }";
    if !count && order != "" {
        q += " order by ?" + order;
        if desc { q += " desc"; }
    }
    if limit > 0 { q += " limit " + limit; }
    q
}

fn next_limit(cur) {
    if cur == 10 { 50 } else if cur == 50 { 200 } else if cur == 200 { 0 } else { 10 }
}

fn limit_label(cur) { if cur == 0 { "all" } else { "" + cur } }

// =========================== handlers ============================

fn on_init() {
    state.proc = spawn_args(state.host, ["schema"]);
    if state.proc != -1 { state.busy = true; state.job = "schema"; state.out = []; set_animating(true); }
    request_render();
}

fn on_frame(dt) {
    if !state.busy { set_animating(false); return; }
    let res = drain(state.proc);
    for l in res.lines { state.out.push(l); }
    if res.done {
        let job = state.job;
        let out = state.out;
        state.busy = false; state.proc = -1; state.job = ""; state.out = [];
        set_animating(false);
        if job == "schema" {
            state.schema = out;
            state.types = parse_types(out);
            state.loaded = true;
        } else if job == "query" {
            state.result = out;
        }
    }
    request_render();
}

fn on_click(x, y, shift, cmd, id) {
    if id == "" { return; }

    if id == "refresh" {
        if !state.busy {
            state.proc = spawn_args(state.host, ["schema"]);
            if state.proc != -1 { state.busy = true; state.job = "schema"; state.out = []; set_animating(true); }
        }
        request_render();
        return;
    }

    if id.starts_with("t:") {
        state.selected = id.sub_string(2);
        state.fields = field_names(state.schema, state.selected);
        state.picked = []; state.count = false; state.order = ""; state.desc = false;
        state.result = []; state.ran = "";
        request_render();
        return;
    }

    if id.starts_with("c:") {           // toggle a projection column
        let f = id.sub_string(2);
        if has(state.picked, f) { state.picked = without(state.picked, f); }
        else { state.picked.push(f); }
        request_render();
        return;
    }

    if id == "count" { state.count = !state.count; request_render(); return; }

    if id.starts_with("o:") {           // order-by field (re-click toggles dir)
        let f = id.sub_string(2);
        if state.order == f { state.desc = !state.desc; }
        else { state.order = f; state.desc = false; }
        request_render();
        return;
    }
    if id == "o:none" { state.order = ""; state.desc = false; request_render(); return; }

    if id == "limit" { state.limit = next_limit(state.limit); request_render(); return; }

    if id == "run" {
        if state.busy || state.selected == "" { return; }
        let q = build_query(state.selected, state.picked, state.fields,
                            state.count, state.order, state.desc, state.limit);
        state.ran = q;
        state.proc = spawn_args(state.host, ["query", q]);
        if state.proc != -1 { state.busy = true; state.job = "query"; state.out = []; set_animating(true); }
        host_log("[datalog_query] " + q);
        request_render();
        return;
    }
}

// ============================ render =============================

fn render(w, h) {
    let rows = [];

    rows.push(#{ type: "hstack", gap: 8.0, align: "center", children: [
        #{ type: "text", value: "datalog query", size: 15.0, color: "#e6e9f0", weight: "bold" },
        #{ type: "badge", value: state.host },
        #{ type: "button", id: "refresh", label: "refresh" },
    ]});

    let status = if state.busy { "running " + state.job + "..." }
                 else if !state.loaded { "connecting..." }
                 else { "pick a type, toggle columns, Run" };
    rows.push(#{ type: "text", value: status, size: 11.0, color: "#8b90a0" });
    rows.push(#{ type: "divider" });

    // type chips
    let type_btns = [];
    for t in state.types {
        let mark = if state.selected == t { "> " } else { "  " };
        type_btns.push(#{ type: "button", id: "t:" + t, label: mark + t });
    }
    rows.push(#{ type: "hstack", gap: 4.0, children: type_btns });

    if state.selected != "" {
        rows.push(#{ type: "divider" });

        // column toggles
        rows.push(#{ type: "text", value: "columns", size: 11.0, color: "#9aa0b0" });
        let col_btns = [];
        for f in state.fields {
            let on = has(state.picked, f);
            let mark = if on { "[x] " } else { "[ ] " };
            col_btns.push(#{ type: "button", id: "c:" + f, label: mark + f });
        }
        rows.push(#{ type: "hstack", gap: 4.0, children: col_btns });

        // option row
        let count_label = if state.count { "[x] count(*)" } else { "[ ] count(*)" };
        let limit_label_s = "limit: " + limit_label(state.limit);
        rows.push(#{ type: "hstack", gap: 6.0, children: [
            #{ type: "button", id: "count", label: count_label },
            #{ type: "button", id: "limit", label: limit_label_s },
            #{ type: "button", id: "run", label: "Run"},
        ]});

        // order-by row (disabled meaning when count is on)
        if !state.count {
            let order_btns = [#{ type: "text", value: "order:", size: 11.0, color: "#9aa0b0" }];
            order_btns.push(#{ type: "button", id: "o:none",
                label: if state.order == "" { "> none" } else { "  none" } });
            for f in state.fields {
                let lbl = if state.order == f { (if state.desc { "> " + f + " v" } else { "> " + f + " ^" }) }
                          else { "  " + f };
                order_btns.push(#{ type: "button", id: "o:" + f, label: lbl });
            }
            rows.push(#{ type: "hstack", gap: 4.0, children: order_btns });
        }

        if state.ran != "" {
            rows.push(#{ type: "text", value: state.ran, size: 10.0, color: "#6f7686", family: "mono" });
        }

        if state.result.len() > 0 {
            rows.push(#{ type: "divider" });
            let body = [];
            for line in state.result {
                body.push(#{ type: "text", value: line, size: 10.0, color: "#cfd3dd", family: "mono" });
            }
            rows.push(#{ type: "scroll", pad: 4.0, gap: 1.0, children: body });
        }
    }

    #{ type: "vstack", pad: 10.0, gap: 4.0, children: rows }
}
