// datalog_browser.rhai — explore a running datalog-db service.
//
// EVENT-DRIVEN. Talks to the `datalog` CLI as a one-shot subprocess
// (the same proc_* API the chess widget uses for Stockfish): spawn
// `datalog --host <h> <cmd...>`, drain stdout line-by-line in on_frame
// until the child exits, then act on the captured text.
//
// What it does:
//   - lists every entity type + enum (parsed from `schema`)
//   - click a type to see its fields and browse its rows (`list`)
//   - soft-drop (reversible) or hard-purge (destructive, confirmed) a type
//
// Scoping rules this file obeys (terminal-bevy Rhai widgets):
//   - only handlers (on_init/on_click/on_frame/render) may touch `state`
//   - plain `fn`s are pure helpers: they take values, return values, and
//     may call host fns (proc_*, host_env, ...) but never read `state`.

// ---- one-time state init (top level runs on load/reload) ----
if !("host" in state)    { state.host = "127.0.0.1:5557"; }
if !("proc" in state)    { state.proc = -1; }
if !("busy" in state)    { state.busy = false; }
if !("job" in state)     { state.job = ""; }
if !("out" in state)     { state.out = []; }
if !("schema" in state)  { state.schema = []; }
if !("types" in state)   { state.types = []; }
if !("enums" in state)   { state.enums = []; }
if !("selected" in state){ state.selected = ""; }
if !("rcols" in state)   { state.rcols = []; }   // browsed-rows table columns
if !("rrows" in state)   { state.rrows = []; }   // browsed-rows table rows
if !("rows_for" in state){ state.rows_for = ""; }
if !("msg" in state)     { state.msg = ""; }
if !("confirm" in state) { state.confirm = ""; }
if !("loaded" in state)  { state.loaded = false; }

// ============================ helpers ============================
// (pure — no `state` access; host fns are allowed)

fn dl_path() {
    let home = host_env("HOME");
    if home == "" { "datalog" } else { home + "/.cargo/bin/datalog" }
}

fn build_argv(host, args) {
    let argv = ["--host", host];
    for a in args { argv.push(a); }
    argv
}

// Spawn `datalog --host <host> <args...>`. Returns the child handle, or
// -1 on failure. Caller stores it into state.
fn spawn_job(host, args) {
    proc_spawn(dl_path(), build_argv(host, args))
}

// Drain all stdout lines currently available from `handle`. Returns
// #{ lines: [...new lines...], done: <process exited & fully drained> }.
fn drain(handle) {
    let lines = [];
    loop { let l = proc_read(handle); if l == "" { break; } lines.push(l); }
    let done = false;
    if !proc_alive(handle) {
        // EOF: the reader thread queues every line before flipping
        // alive=false, so one more drain captures the tail.
        loop { let l = proc_read(handle); if l == "" { break; } lines.push(l); }
        done = true;
    }
    #{ lines: lines, done: done }
}

// A schema name line is indented exactly two spaces ("  User"); field
// lines are indented four ("    name: ...").
fn is_name_line(line) {
    line.len() >= 3 && line.starts_with("  ") && !line.starts_with("   ")
}

fn parse_schema(lines) {
    let types = [];
    let enums = [];
    let section = "";
    for line in lines {
        if line == "Entity types:" { section = "types"; }
        else if line == "Enums:"   { section = "enums"; }
        else if is_name_line(line) {
            let name = line.sub_string(2);
            if name != "" && name != "(none)" {
                if section == "types" { types.push(name); }
                else if section == "enums" { enums.push(name); }
            }
        }
    }
    #{ types: types, enums: enums }
}

// Field/variant lines that belong to `type` in the schema dump.
fn fields_of(lines, type) {
    let target = "  " + type;
    let out = [];
    let in_block = false;
    for line in lines {
        if line == target { in_block = true; }
        else if in_block {
            if line.starts_with("    ") { out.push(line.sub_string(4)); }
            else { break; }
        }
    }
    out
}

fn last_line(arr) {
    if arr.len() == 0 { "" } else { arr[arr.len() - 1] }
}

// Rhai's `trim()` mutates in place + returns (); use this for a copy.
fn trimmed(s) { s.trim(); s }

// `list` output is an ASCII table: header, a `---+---` rule, rows, then
// a `(N rows)` footer. Detect by the rule on line 2; parse cells on " | ".
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

// Split a field line "name: string required" into ["name", "string required"].
fn field_pair(fieldline) {
    let i = fieldline.index_of(":");
    if i < 0 { [fieldline, ""] }
    else { [trimmed(fieldline.sub_string(0, i)), trimmed(fieldline.sub_string(i + 1))] }
}

// =========================== handlers ============================

fn on_init() {
    // Kick the first schema load.
    state.proc = spawn_job(state.host, ["schema"]);
    if state.proc != -1 {
        state.busy = true; state.job = "schema"; state.out = [];
        set_animating(true);
    }
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
            let parsed = parse_schema(out);
            state.types = parsed.types;
            state.enums = parsed.enums;
            state.loaded = true;
            host_log("[datalog_browser] schema: " + state.types.len() +
                     " types, " + state.enums.len() + " enums");
        } else if job == "list" {
            if is_table(out) {
                let t = parse_table(out);
                state.rcols = t.cols; state.rrows = t.rows;
            } else {
                state.rcols = []; state.rrows = [];
            }
            state.rows_for = state.selected;
        } else if job == "drop" || job == "purge" {
            state.msg = last_line(out);
            state.selected = ""; state.rcols = []; state.rrows = []; state.rows_for = "";
            // Re-pull the schema so the dropped type disappears.
            state.proc = spawn_job(state.host, ["schema"]);
            if state.proc != -1 {
                state.busy = true; state.job = "schema"; state.out = [];
                set_animating(true);
            }
        }
    }
    request_render();
}

fn on_click(x, y, shift, cmd, id) {
    if id == "" { return; }

    if id == "refresh" {
        state.confirm = ""; state.msg = "";
        if !state.busy {
            state.proc = spawn_job(state.host, ["schema"]);
            if state.proc != -1 { state.busy = true; state.job = "schema"; state.out = []; set_animating(true); }
        }
        request_render();
        return;
    }

    if id.starts_with("t:") {
        state.selected = id.sub_string(2);
        state.confirm = ""; state.msg = "";
        state.rcols = []; state.rrows = []; state.rows_for = "";
        request_render();
        return;
    }

    if state.busy { return; } // one command at a time

    if id == "browse" {
        if state.selected != "" {
            state.proc = spawn_job(state.host, ["list", state.selected, "limit", "100"]);
            if state.proc != -1 { state.busy = true; state.job = "list"; state.out = []; set_animating(true); }
        }
        request_render();
        return;
    }

    if id == "drop" { // soft drop — reversible, no confirm
        if state.selected != "" {
            state.proc = spawn_job(state.host, ["drop", state.selected]);
            if state.proc != -1 { state.busy = true; state.job = "drop"; state.out = []; set_animating(true); }
        }
        request_render();
        return;
    }

    if id == "purge" { state.confirm = "purge:" + state.selected; request_render(); return; }
    if id == "cancel" { state.confirm = ""; request_render(); return; }
    if id == "confirm" {
        let c = state.confirm; state.confirm = "";
        if c.starts_with("purge:") {
            let t = c.sub_string(6);
            state.proc = spawn_job(state.host, ["purge", t]);
            if state.proc != -1 { state.busy = true; state.job = "purge"; state.out = []; set_animating(true); }
        }
        request_render();
        return;
    }
}

// ============================ render =============================

fn render(w, h) {
    let rows = [];

    // --- header ---
    let title = [
        #{ type: "text", value: "datalog", size: 15.0, color: "#e6e9f0", weight: "bold" },
        #{ type: "badge", value: state.host },
        #{ type: "button", id: "refresh", label: "refresh" },
    ];
    rows.push(#{ type: "hstack", gap: 8.0, align: "center", children: title });

    let status = if state.busy { "running " + state.job + "..." }
                 else if !state.loaded { "connecting..." }
                 else { state.types.len() + " types, " + state.enums.len() + " enums" };
    rows.push(#{ type: "text", value: status, size: 11.0, color: "#8b90a0" });
    if state.msg != "" {
        rows.push(#{ type: "text", value: state.msg, size: 11.0, color: "#c8b06a" });
    }
    rows.push(#{ type: "divider" });

    // --- type list ---
    rows.push(#{ type: "text", value: "Types", size: 12.0, color: "#9aa0b0" });
    if state.types.len() == 0 && state.loaded {
        rows.push(#{ type: "text", value: "(none)", size: 11.0, color: "#6b7080" });
    }
    for t in state.types {
        let mark = if state.selected == t { "> " } else { "  " };
        rows.push(#{ type: "button", id: "t:" + t, label: mark + t });
    }

    if state.enums.len() > 0 {
        rows.push(#{ type: "text", value: "Enums", size: 12.0, color: "#9aa0b0" });
        let pills = [];
        for e in state.enums { pills.push(#{ type: "badge", value: e }); }
        rows.push(#{ type: "hstack", gap: 4.0, children: pills });
    }

    // --- selected type detail ---
    if state.selected != "" {
        rows.push(#{ type: "divider" });
        rows.push(#{ type: "text", value: state.selected, size: 13.0, color: "#e6e9f0", weight: "bold" });

        // fields as a 2-col table
        let frows = [];
        for f in fields_of(state.schema, state.selected) {
            let p = field_pair(f);
            frows.push([p[0], p[1]]);
        }
        if frows.len() > 0 {
            rows.push(#{ type: "table", columns: [
                #{ header: "field", width: 120.0 },
                #{ header: "type" },
            ], rows: frows });
        }

        let actions = [
            #{ type: "button", id: "browse", label: "browse rows" },
            #{ type: "button", id: "drop", label: "soft drop" },
            #{ type: "button", id: "purge", label: "purge"},
        ];
        rows.push(#{ type: "hstack", gap: 6.0, children: actions });

        if state.confirm == "purge:" + state.selected {
            rows.push(#{ type: "text",
                value: "Hard-purge '" + state.selected + "'? Deletes all its datoms. Irreversible.",
                size: 11.0, color: "#e08a8a" });
            rows.push(#{ type: "hstack", gap: 6.0, children: [
                #{ type: "button", id: "confirm", label: "yes, purge"},
                #{ type: "button", id: "cancel", label: "cancel" },
            ]});
        }

        // browsed rows as a table
        if state.rows_for == state.selected && state.rrows.len() > 0 {
            rows.push(#{ type: "divider" });
            let cols = [];
            for c in state.rcols { cols.push(#{ header: c }); }
            rows.push(#{ type: "table", zebra: true, columns: cols, rows: state.rrows });
            rows.push(#{ type: "text", value: state.rrows.len() + " row(s)", size: 10.0, color: "#7f8794" });
        }
    }

    #{ type: "vstack", pad: 10.0, gap: 4.0, children: rows }
}
