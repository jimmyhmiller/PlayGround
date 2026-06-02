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
if !("rows" in state)    { state.rows = []; }
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
            state.rows = out;
            state.rows_for = state.selected;
        } else if job == "drop" || job == "purge" {
            state.msg = last_line(out);
            state.selected = ""; state.rows = []; state.rows_for = "";
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
        state.rows = []; state.rows_for = "";
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
        for e in state.enums {
            rows.push(#{ type: "text", value: "  " + e, size: 11.0, color: "#aeb4c4" });
        }
    }

    // --- selected type detail ---
    if state.selected != "" {
        rows.push(#{ type: "divider" });
        rows.push(#{ type: "text", value: state.selected, size: 13.0, color: "#e6e9f0", weight: "bold" });

        for f in fields_of(state.schema, state.selected) {
            rows.push(#{ type: "text", value: "  " + f, size: 11.0, color: "#aeb4c4", family: "mono" });
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

        if state.rows_for == state.selected && state.rows.len() > 0 {
            let body = [];
            for line in state.rows {
                body.push(#{ type: "text", value: line, size: 10.0, color: "#cfd3dd", family: "mono" });
            }
            rows.push(#{ type: "scroll", pad: 4.0, gap: 1.0, children: body });
        }
    }

    #{ type: "vstack", pad: 10.0, gap: 4.0, children: rows }
}
