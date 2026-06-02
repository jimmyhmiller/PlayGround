// datalog_query.rhai — a real query console for datalog-db.
//
// Type any datalog command into the box and Cmd/Ctrl+Enter (or Run):
//   find ?n where ?u: User { name: ?n }
//   list User limit 50      schema      status
//   assert User { name: "Carol", email: "c@x.com", age: 22 }
//   define Tag { label: string required }
//   drop User       purge Product
// Query results render as a Table; assert/define/drop/etc show their
// status line. Quick chips run common reads without typing.
//
// Talks to the `datalog` CLI as a one-shot subprocess (proc_* API),
// draining stdout in on_frame. Scoping: only handlers touch `state`;
// `fn`s are pure helpers (may call host fns, never `state`).

// ---- state ----
if !("host" in state)   { state.host = "127.0.0.1:5557"; }
if !("proc" in state)   { state.proc = -1; }
if !("busy" in state)   { state.busy = false; }
if !("job" in state)    { state.job = ""; }
if !("out" in state)    { state.out = []; }
if !("types" in state)  { state.types = []; }   // for quick chips
if !("q" in state)      { state.q = ""; }        // textarea buffer
if !("ran" in state)    { state.ran = ""; }      // last command run
if !("rkind" in state)  { state.rkind = ""; }    // "table" | "text"
if !("rcols" in state)  { state.rcols = []; }
if !("rrows" in state)  { state.rrows = []; }
if !("rtext" in state)  { state.rtext = []; }

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

// Rhai's `trim()` mutates in place and returns (), so use this when you
// want a trimmed *copy* as an expression.
fn trimmed(s) { s.trim(); s }

fn split_ws(s) {
    let out = [];
    for p in s.split(' ') { if p != "" { out.push(p); } }
    out
}

// Map a typed command line to the CLI's one-shot argv, matching how the
// `datalog` binary itself splits one-shot commands. A bare `find ...` is
// treated as a `query`.
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
    // list / backup / drop / purge / anything else → whitespace argv
    split_ws(s)
}

fn parse_types(lines) {
    let types = [];
    let section = "";
    for line in lines {
        if line == "Entity types:" { section = "types"; }
        else if line == "Enums:"   { section = "enums"; }
        else if line.len() >= 3 && line.starts_with("  ") && !line.starts_with("   ") {
            let name = line.sub_string(2);
            if name != "" && name != "(none)" && section == "types" { types.push(name); }
        }
    }
    types
}

// A `find`/`list` result is an ASCII table: header, a `---+---` rule,
// rows, then a `(N rows)` footer. Detect by the rule on line 2.
fn is_table(lines) {
    lines.len() >= 2 && lines[1].starts_with("-")
}

fn split_cells(line) {
    let out = [];
    for c in line.split(" | ") { out.push(trimmed(c)); }
    out
}

fn parse_table(lines) {
    let cols = split_cells(lines[0]);
    let rows = [];
    let i = 2; // skip header + rule
    while i < lines.len() {
        let line = lines[i];
        i += 1;
        if line == "" { continue; }
        if line.starts_with("(") { break; } // "(N rows)" footer
        rows.push(split_cells(line));
    }
    #{ cols: cols, rows: rows }
}

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
            state.types = parse_types(out);
        } else if job == "cmd" {
            if is_table(out) {
                let t = parse_table(out);
                state.rkind = "table"; state.rcols = t.cols; state.rrows = t.rows; state.rtext = [];
            } else {
                state.rkind = "text"; state.rtext = out; state.rcols = []; state.rrows = [];
            }
        }
    }
    request_render();
}

// Build the one-shot child for `line`. Pure helper (host fns only, no
// `state`); returns the handle, or -1. Handlers do the state mutation
// inline because helpers may not touch `state`.
fn launch(host, line) {
    let argv = to_argv(line);
    if argv.len() == 0 { return -1; }
    spawn_args(host, argv)
}

fn on_click(x, y, shift, cmd, id) {
    if id == "" || state.busy { return; }
    let line = "";
    if id == "run" { line = state.q; }
    else if id == "q:schema" { line = "schema"; }
    else if id == "q:status" { line = "status"; }
    else if id.starts_with("ql:") { line = "list " + id.sub_string(3) + " limit 100"; }
    else { return; }

    state.q = line;
    let h = launch(state.host, line);
    if h != -1 {
        state.proc = h; state.busy = true; state.job = "cmd"; state.out = [];
        state.ran = trimmed(line);
        set_animating(true);
        host_log("[datalog_query] " + state.ran);
    }
    request_render();
}

fn on_input_change(id, value) {
    if id == "q" { state.q = value; request_render(); }
}

fn on_input_submit(id, value) {     // Cmd/Ctrl+Enter in the textarea
    if id != "q" || state.busy { return; }
    state.q = value;
    let h = launch(state.host, value);
    if h != -1 {
        state.proc = h; state.busy = true; state.job = "cmd"; state.out = [];
        state.ran = trimmed(value);
        set_animating(true);
        host_log("[datalog_query] " + state.ran);
    }
    request_render();
}

// ============================ render =============================

fn render(w, h) {
    let rows = [];

    rows.push(#{ type: "hstack", gap: 8.0, align: "center", children: [
        #{ type: "text", value: "datalog query", size: 15.0, color: "#e6e9f0", weight: "bold" },
        #{ type: "badge", value: state.host },
    ]});

    // command box
    rows.push(#{ type: "textarea", id: "q", value: state.q, rows: 4,
        placeholder: "find ?n, ?age where ?u: User { name: ?n, age: ?age }   (Cmd+Enter to run)\nlist User limit 50   .   assert/define/drop/purge also work" });

    // action row + quick chips
    let chips = [
        #{ type: "button", id: "run", label: if state.busy { "running..." } else { "Run" } },
        #{ type: "button", id: "q:schema", label: "schema" },
        #{ type: "button", id: "q:status", label: "status" },
    ];
    for t in state.types { chips.push(#{ type: "button", id: "ql:" + t, label: "list " + t }); }
    rows.push(#{ type: "hstack", gap: 6.0, children: chips });

    if state.ran != "" {
        rows.push(#{ type: "text", value: "$ " + state.ran, size: 10.0, color: "#6f7686", family: "mono" });
    }

    // result
    if state.rkind == "table" {
        rows.push(#{ type: "divider" });
        if state.rrows.len() == 0 {
            rows.push(#{ type: "text", value: "(no rows)", size: 11.0, color: "#7f8794" });
        } else {
            let cols = [];
            for c in state.rcols { cols.push(#{ header: c }); }
            rows.push(#{ type: "table", zebra: true, columns: cols, rows: state.rrows });
            rows.push(#{ type: "text", value: state.rrows.len() + " row(s)", size: 10.0, color: "#7f8794" });
        }
    } else if state.rkind == "text" {
        rows.push(#{ type: "divider" });
        for line in state.rtext {
            let col = if line.starts_with("ERROR") { "#e08a8a" }
                      else if line.starts_with("OK") { "#86c08a" }
                      else { "#cfd3dd" };
            rows.push(#{ type: "text", value: line, size: 11.0, color: col, family: "mono" });
        }
    }

    #{ type: "vstack", pad: 10.0, gap: 6.0, children: rows }
}
