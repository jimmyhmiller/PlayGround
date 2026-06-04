// dlide_schema.rhai — datalog IDE: schema/table browser.
//
// Reads `schema` to list types + enums. Clicking a type asks the editor
// to prefill a browse; per-type actions ask the results pane to run.
// Auto-reloads whenever any pane reports a schema change.
//
//   listens: schema.changed           reload the type list
//            conn.state {host}         (retained) DB host
//   emits:   sql.use_table {table}     (click type → editor prefill)
//            sql.run {sql}             (browse / drop / purge → results)

if !("host" in state)    { state.host = "127.0.0.1:5557"; }
if !("busy" in state)    { state.busy = false; }
if !("proc" in state)    { state.proc = -1; }
if !("out" in state)     { state.out = []; }
if !("types" in state)   { state.types = []; }
if !("enums" in state)   { state.enums = []; }
if !("selected" in state){ state.selected = ""; }
if !("confirm" in state) { state.confirm = ""; }
if !("loaded" in state)  { state.loaded = false; }

fn dl_path() { let home = host_env("HOME"); if home == "" { "datalog" } else { home + "/.cargo/bin/datalog" } }
fn spawn_args(host, args) { let argv = ["--host", host]; for a in args { argv.push(a); } proc_spawn(dl_path(), argv) }
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
fn is_name_line(line) { line.len() >= 3 && line.starts_with("  ") && !line.starts_with("   ") }
fn parse_schema(lines) {
    let types = []; let enums = []; let section = "";
    for line in lines {
        if line == "Entity types:" { section = "types"; }
        else if line == "Enums:"   { section = "enums"; }
        else if is_name_line(line) {
            let name = line.sub_string(2);
            if name != "" && name != "(none)" {
                if section == "types" { types.push(name); } else if section == "enums" { enums.push(name); }
            }
        }
    }
    #{ types: types, enums: enums }
}

// Start a schema reload (no-op if already busy). Pure helper → handle.
fn load_schema(host) { spawn_args(host, ["schema"]) }

fn on_init() {
    state.proc = load_schema(state.host);
    if state.proc != -1 { state.busy = true; state.out = []; set_animating(true); }
    request_render();
}

fn on_frame(dt) {
    if !state.busy { set_animating(false); return; }
    let res = drain(state.proc);
    for l in res.lines { state.out.push(l); }
    if res.done {
        let parsed = parse_schema(state.out);
        state.types = parsed.types; state.enums = parsed.enums; state.loaded = true;
        state.busy = false; state.proc = -1; state.out = [];
        set_animating(false);
        host_log("[dlide_schema] " + state.types.len() + " types, " + state.enums.len() + " enums");
        request_render();
    }
}

fn on_message(topic, payload, sender) {
    if sender == my_id() { return; }
    if topic == "conn.state" { state.host = payload.host; request_render(); return; }
    if topic == "schema.changed" {
        if !state.busy {
            state.proc = load_schema(state.host);
            if state.proc != -1 { state.busy = true; state.out = []; set_animating(true); }
        }
        request_render();
    }
}

fn on_click(x, y, shift, cmd, id) {
    if id == "" { return; }
    if id == "refresh" {
        if !state.busy {
            state.proc = load_schema(state.host);
            if state.proc != -1 { state.busy = true; state.out = []; set_animating(true); }
        }
        request_render();
        return;
    }
    if id.starts_with("t:") {                 // select + ask editor to prefill
        let t = id.sub_string(2);
        state.selected = t; state.confirm = "";
        emit("sql.use_table", #{ table: t });
        request_render();
        return;
    }
    if id == "browse" {
        if state.selected != "" { emit("sql.run", #{ sql: "list " + state.selected + " limit 50" }); }
        return;
    }
    if id == "drop" {
        if state.selected != "" { emit("sql.run", #{ sql: "drop " + state.selected }); }
        return;
    }
    if id == "purge" { state.confirm = "purge:" + state.selected; request_render(); return; }
    if id == "cancel" { state.confirm = ""; request_render(); return; }
    if id == "confirm" {
        let c = state.confirm; state.confirm = "";
        if c.starts_with("purge:") { emit("sql.run", #{ sql: "purge " + c.sub_string(6) }); }
        request_render();
        return;
    }
}

fn label(text) { #{ type: "text", value: text, size: 10.0, weight: "bold", color: "fg_muted" } }

fn render(w, h) {
    let rows = [];

    // toolbar
    let status = if state.busy { "loading…" }
                 else if !state.loaded { "connecting…" }
                 else { state.types.len() + " types · " + state.enums.len() + " enums" };
    rows.push(#{ type: "hstack", gap: 6.0, align: "center", children: [
        #{ type: "text", value: status, size: 11.0, color: "fg_muted" },
        #{ type: "button", id: "refresh", label: "↻", kind: "ghost" },
    ]});

    // types as selectable rows
    rows.push(label("TYPES"));
    if state.types.len() == 0 && state.loaded {
        rows.push(#{ type: "text", value: "none yet", size: 11.0, color: "fg_muted" });
    }
    for t in state.types {
        rows.push(#{ type: "list-item", id: "t:" + t, selected: state.selected == t,
            pad: 6.0,
            children: [ #{ type: "text", value: t, size: 13.0,
                           color: if state.selected == t { "accent_500" } else { "fg" } } ] });
    }

    if state.enums.len() > 0 {
        rows.push(label("ENUMS"));
        let pills = [];
        for e in state.enums { pills.push(#{ type: "badge", value: e }); }
        rows.push(#{ type: "hstack", gap: 4.0, children: pills });
    }

    // actions for the selected type, in a card
    if state.selected != "" {
        let card = [
            #{ type: "text", value: state.selected, size: 13.0, weight: "bold", color: "fg" },
            #{ type: "hstack", gap: 6.0, children: [
                #{ type: "button", id: "browse", label: "browse", kind: "filled" },
                #{ type: "button", id: "drop", label: "drop", kind: "ghost" },
                #{ type: "button", id: "purge", label: "purge", kind: "ghost" },
            ]},
        ];
        if state.confirm == "purge:" + state.selected {
            card.push(#{ type: "text",
                value: "Hard-purge '" + state.selected + "'? Irreversible.",
                size: 11.0, color: "#e0746f" });
            card.push(#{ type: "hstack", gap: 6.0, children: [
                #{ type: "button", id: "confirm", label: "yes, purge", kind: "filled" },
                #{ type: "button", id: "cancel", label: "cancel", kind: "ghost" },
            ]});
        }
        rows.push(#{ type: "frame", pad: 10.0, gap: 6.0,
            style: #{ background: "surface_2", radius: "radius_md",
                      border: #{ color: "surface_3", width: 1.0 } },
            children: card });
    }

    #{ type: "vstack", gap: 6.0,
       style: #{ background: "surface_1", width: "100%", height: "100%",
                 padding: #{ top: 10.0, right: 10.0, bottom: 10.0, left: 10.0 } },
       children: [ #{ type: "scroll", pad: 0.0, gap: 6.0, children: rows } ] }
}
