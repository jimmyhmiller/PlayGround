// dlide_results.rhai — datalog IDE: the executor + results table.
//
// The ONLY pane that runs queries. Listens for sql.run, shells out to the
// `datalog` CLI (proc_* one-shot, drained in on_frame while a query is in
// flight — opt-in animation, not idle polling), renders the result table,
// and broadcasts what happened.
//
//   listens: sql.run {sql}            run this command
//            conn.state {host}        (retained) DB host
//   emits:   sql.result {ok,summary,ms}   a command finished
//            schema.changed            after define/drop/purge (DDL)

if !("host" in state)  { state.host = "127.0.0.1:5557"; }
if !("busy" in state)  { state.busy = false; }
if !("proc" in state)  { state.proc = -1; }
if !("out" in state)   { state.out = []; }
if !("ran" in state)   { state.ran = ""; }
if !("t0" in state)    { state.t0 = 0.0; }
if !("ms" in state)    { state.ms = 0; }
if !("rkind" in state) { state.rkind = ""; }
if !("rcols" in state) { state.rcols = []; }
if !("rrows" in state) { state.rrows = []; }
if !("rtext" in state) { state.rtext = []; }

// ---- pure helpers ----
fn dl_path() { let home = host_env("HOME"); if home == "" { "datalog" } else { home + "/.cargo/bin/datalog" } }
fn spawn_args(host, args) { let argv = ["--host", host]; for a in args { argv.push(a); } proc_spawn(dl_path(), argv) }
fn trimmed(s) { s.trim(); s }
fn split_ws(s) { let out = []; for p in s.split(' ') { if p != "" { out.push(p); } } out }
fn first_word(s) { let t = trimmed(s); let i = t.index_of(" "); if i < 0 { t } else { t.sub_string(0, i) } }
fn is_ddl(kw) { kw == "define" || kw == "drop" || kw == "purge" }

fn to_argv(line) {
    let s = trimmed(line);
    if s == "" { return []; }
    let sp = s.index_of(" ");
    let kw = if sp < 0 { s } else { s.sub_string(0, sp) };
    let rest = if sp < 0 { "" } else { trimmed(s.sub_string(sp + 1)) };
    if kw == "status" || kw == "schema" { return [kw]; }
    if kw == "find" { return ["query", s]; }
    if kw == "query" || kw == "define" || kw == "assert" || kw == "retract" || kw == "json" {
        let a = [kw];
        if rest != "" { a.push(rest); }
        return a;
    }
    split_ws(s)
}

fn is_table(lines) { lines.len() >= 2 && lines[1].starts_with("-") }
fn split_cells(line) { let out = []; for c in line.split(" | ") { out.push(trimmed(c)); } out }
fn parse_table(lines) {
    let cols = split_cells(lines[0]);
    let rows = [];
    let i = 2;
    while i < lines.len() {
        let line = lines[i];
        i += 1;
        if line == "" { continue; }
        if line.starts_with("(") { break; }
        rows.push(split_cells(line));
    }
    #{ cols: cols, rows: rows }
}
fn has_error(lines) { for l in lines { if l.starts_with("ERROR") { return true; } } false }
fn join_tab(arr) { let s = ""; let first = true; for x in arr { if first { s = x; first = false; } else { s += "\t" + x; } } s }

// ---- handlers ----
fn on_message(topic, payload, sender) {
    if sender == my_id() { return; }
    if topic == "conn.state" { state.host = payload.host; request_render(); return; }
    if topic == "sql.run" {
        if state.busy { return; }
        let line = payload.sql;
        let h = spawn_args(state.host, to_argv(line));
        if h != -1 {
            state.proc = h; state.busy = true; state.out = [];
            state.ran = trimmed(line); state.t0 = time();
            host_log("[dlide_results] run: " + state.ran);
        }
        request_render();
    }
}

// Event-driven subprocess: lines are pushed as they arrive (no set_animating,
// no on_frame poll). Accumulate quietly; render once on exit.
fn on_proc_output(handle, line) {
    if handle == state.proc { state.out.push(line); }
}

fn on_proc_exit(handle, code) {
    if handle != state.proc { return; }
    let out = state.out;
    state.busy = false; state.proc = -1;
    state.ms = ((time() - state.t0) * 1000.0).to_int();

    let ok = true;
    let summary = "";
    if is_table(out) {
        let t = parse_table(out);
        state.rkind = "table"; state.rcols = t.cols; state.rrows = t.rows; state.rtext = [];
        summary = state.rrows.len() + " rows";
    } else {
        state.rkind = "text"; state.rtext = out; state.rcols = []; state.rrows = [];
        ok = !has_error(out);
        summary = if out.len() > 0 { out[out.len() - 1] } else { "(done)" };
    }
    emit("sql.result", #{ ok: ok, summary: summary, ms: state.ms });
    if is_ddl(first_word(state.ran)) { emit("schema.changed"); }
    host_log("[dlide_results] done: " + summary + " (" + state.ms + "ms)");
    request_render();
}

// ---- style helpers (pure) ----
fn is_number(s) {
    if s == "" { return false; }
    let i = 0; let seen = false;
    while i < s.len() {
        let ch = s[i]; i += 1;
        if ch >= '0' && ch <= '9' { seen = true; }
        else if ch == '.' || ch == '-' || ch == '+' { }
        else { return false; }
    }
    seen
}

// Right-align a column when every non-empty cell in it is numeric.
fn col_aligns(cols, rows) {
    let aligns = [];
    let c = 0;
    while c < cols.len() {
        let numeric = true; let any = false;
        for r in rows {
            if c < r.len() && r[c] != "" { any = true; if !is_number(r[c]) { numeric = false; } }
        }
        aligns.push(if any && numeric { "end" } else { "start" });
        c += 1;
    }
    aligns
}

// Explicit per-column widths that sum to `avail` (the usable canvas width).
// The renderer's auto columns + content-box panel overflow the pane when
// content is wide, so we size columns ourselves: proportional to content
// length (clamped), then scaled to exactly fill `avail` — so the table fills
// its pane and never spills past it. Returns a width per column.
fn col_widths(cols, rows, avail) {
    let n = cols.len();
    let nat = []; let total = 0.0;
    let c = 0;
    while c < n {
        let m = cols[c].len();
        for r in rows { if c < r.len() { let l = r[c].len(); if l > m { m = l; } } }
        let wpx = (m.to_float() + 2.0) * 7.5;   // ~7.5px/char + padding
        if wpx < 36.0 { wpx = 36.0; }
        if wpx > 380.0 { wpx = 380.0; }
        nat.push(wpx); total += wpx; c += 1;
    }
    if total <= 0.0 { return nat; }
    let scale = avail / total;            // fill if room, shrink if over
    let out = [];
    for x in nat { out.push(x * scale); }
    out
}

fn on_click(x, y, shift, cmd, id) {
    if id == "copy" {
        let txt = "";
        if state.rkind == "table" {
            txt += join_tab(state.rcols) + "\n";
            for r in state.rrows { txt += join_tab(r) + "\n"; }
        } else {
            for l in state.rtext { txt += l + "\n"; }
        }
        clipboard_set(txt);
    }
}

// Re-render on pane resize so layout (e.g. table widths) tracks the
// new canvas size — render() depends on `w`/`h`.
fn on_resize(w, h) { request_render(); }

fn render(w, h) {
    // slim status bar
    let head = if state.busy { "running query…" }
               else if state.ran == "" { "Run a query from the editor below." }
               else { state.rrows.len() + " rows · " + state.ms + " ms" };
    let head_color = if state.busy { "warn" } else { "fg_muted" };

    let bar = [#{ type: "text", value: head, size: 12.0, color: head_color }];
    if state.ran != "" {
        bar.push(#{ type: "text", value: "  ·  " + state.ran, size: 11.0, color: "fg_muted", family: "mono" });
    }
    // widget text isn't selectable, so offer copy-to-clipboard of the result
    if state.rkind != "" {
        bar.push(#{ type: "button", id: "copy", label: "copy", kind: "ghost" });
    }

    // result body
    let inner = [];
    if state.rkind == "table" {
        if state.rrows.len() == 0 {
            inner.push(#{ type: "text", value: "no rows", size: 12.0, color: "fg_muted" });
        } else {
            let aligns = col_aligns(state.rcols, state.rrows);
            // usable width = canvas minus our paddings/border minus column gaps
            let nc = state.rcols.len();
            let gaps = if nc > 1 { 12.0 * (nc - 1).to_float() } else { 0.0 };
            let avail = w - 52.0 - gaps;
            if avail < 60.0 { avail = 60.0; }
            let widths = col_widths(state.rcols, state.rrows, avail);
            let cols = [];
            let i = 0;
            while i < nc {
                cols.push(#{ header: state.rcols[i], align: aligns[i], width: widths[i] });
                i += 1;
            }
            inner.push(#{ type: "table", zebra: true, columns: cols, rows: state.rrows });
        }
    } else if state.rkind == "text" {
        for line in state.rtext {
            let col = if line.starts_with("ERROR") { "#e0746f" }
                      else if line.starts_with("OK") { "#6cc08a" }
                      else { "fg" };
            inner.push(#{ type: "text", value: line, size: 12.0, color: col, family: "mono" });
        }
    } else {
        inner.push(#{ type: "text", value: "Results will appear here.", size: 12.0, color: "fg_muted" });
    }

    #{ type: "vstack", gap: 8.0,
       style: #{ background: "surface_1", width: "100%", min_height: "100%",
                 padding: #{ top: 10.0, right: 12.0, bottom: 12.0, left: 12.0 } },
       children: [
        #{ type: "hstack", gap: 6.0, align: "center", children: bar },
        #{ type: "frame", pad: 10.0, gap: 4.0,
           style: #{ background: "surface_2", radius: "radius_md",
                     border: #{ color: "surface_3", width: 1.0 } },
           children: [ #{ type: "scroll", pad: 0.0, gap: 2.0, children: inner } ] },
    ] }
}
