// dlide_schema.rhai — datalog IDE: schema browser (types + enums).
//
// Lists types and enums. Click a type → editor prefills a browse + shows
// its fields; click an enum → shows its variants. Per-selection actions:
// browse (types only), drop (soft, confirmed), purge (hard, confirmed) —
// emitted as datalog commands for the results pane to run.
//
//   listens: schema.changed           reload the schema
//            conn.state {host}         (retained) DB host
//   emits:   sql.use_table {table}     (click type → editor prefill)
//            sql.run {sql}             (browse / drop / purge → results)

if !("host" in state)    { state.host = "127.0.0.1:5557"; }
if !("busy" in state)    { state.busy = false; }
if !("proc" in state)    { state.proc = -1; }
if !("out" in state)     { state.out = []; }
if !("schema" in state)  { state.schema = []; }   // raw `schema` lines (for inspect)
if !("types" in state)   { state.types = []; }
if !("enums" in state)   { state.enums = []; }
if !("selected" in state){ state.selected = ""; }
if !("sel_kind" in state){ state.sel_kind = ""; }  // "type" | "enum"
if !("confirm" in state) { state.confirm = ""; }
if !("loaded" in state)  { state.loaded = false; }

fn dl_path() { let home = host_env("HOME"); if home == "" { "datalog" } else { home + "/.cargo/bin/datalog" } }
fn spawn_args(host, args) { let argv = ["--host", host]; for a in args { argv.push(a); } proc_spawn(dl_path(), argv) }
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

// The 4-space-indented lines under "  <name>" within a given section
// header ("Entity types:" → fields, "Enums:" → variants).
fn block_under(lines, header, name) {
    let in_section = false; let in_block = false; let out = [];
    let target = "  " + name;
    for line in lines {
        if line == "Entity types:" { in_section = header == "Entity types:"; in_block = false; }
        else if line == "Enums:" { in_section = header == "Enums:"; in_block = false; }
        else if in_section {
            if line == target { in_block = true; }
            else if in_block {
                if line.starts_with("    ") { out.push(line.sub_string(4)); }
                else { break; }
            }
        }
    }
    out
}

fn load_schema(host) { spawn_args(host, ["schema"]) }

fn on_init() {
    state.proc = load_schema(state.host);
    if state.proc != -1 { state.busy = true; state.out = []; }
    request_render();
}

// Event-driven subprocess (no set_animating / on_frame): accumulate lines,
// parse once on exit.
fn on_proc_output(handle, line) {
    if handle == state.proc { state.out.push(line); }
}

fn on_proc_exit(handle, code) {
    if handle != state.proc { return; }
    let out = state.out;
    let parsed = parse_schema(out);
    state.schema = out;
    state.types = parsed.types; state.enums = parsed.enums; state.loaded = true;
    // Clear a selection that no longer exists (e.g. dropped elsewhere).
    if state.selected != "" {
        let still = false;
        if state.sel_kind == "type" { for t in state.types { if t == state.selected { still = true; } } }
        else if state.sel_kind == "enum" { for e in state.enums { if e == state.selected { still = true; } } }
        if !still { state.selected = ""; state.sel_kind = ""; state.confirm = ""; }
    }
    state.busy = false; state.proc = -1; state.out = [];
    host_log("[dlide_schema] " + state.types.len() + " types, " + state.enums.len() + " enums");
    request_render();
}

fn on_message(topic, payload, sender) {
    if sender == my_id() { return; }
    if topic == "conn.state" { state.host = payload.host; request_render(); return; }
    if topic == "schema.changed" {
        if !state.busy {
            state.proc = load_schema(state.host);
            if state.proc != -1 { state.busy = true; state.out = []; }
        }
        request_render();
    }
}

fn on_click(x, y, shift, cmd, id) {
    if id == "" { return; }
    if id == "refresh" {
        if !state.busy {
            state.proc = load_schema(state.host);
            if state.proc != -1 { state.busy = true; state.out = []; }
        }
        request_render();
        return;
    }
    if id.starts_with("t:") {                 // type: select + prefill editor
        state.selected = id.sub_string(2); state.sel_kind = "type"; state.confirm = "";
        emit("sql.use_table", #{ table: state.selected });
        request_render();
        return;
    }
    if id.starts_with("e:") {                 // enum: select (inspect only)
        state.selected = id.sub_string(2); state.sel_kind = "enum"; state.confirm = "";
        request_render();
        return;
    }
    if id == "browse" {
        if state.selected != "" && state.sel_kind == "type" {
            emit("sql.run", #{ sql: "list " + state.selected + " limit 50" });
        }
        return;
    }
    if id == "drop" { if state.selected != "" { state.confirm = "drop:" + state.selected; } request_render(); return; }
    if id == "purge" { if state.selected != "" { state.confirm = "purge:" + state.selected; } request_render(); return; }
    if id == "cancel" { state.confirm = ""; request_render(); return; }
    if id == "confirm" {
        let c = state.confirm; state.confirm = "";
        let kp = if state.sel_kind == "enum" { "enum " } else { "" };  // "drop enum X" vs "drop X"
        if c.starts_with("purge:") { emit("sql.run", #{ sql: "purge " + kp + state.selected }); }
        else if c.starts_with("drop:") { emit("sql.run", #{ sql: "drop " + kp + state.selected }); }
        // The item is gone now — clear the selection so its card disappears
        // instead of leaving a stale drop/purge target to re-click.
        state.selected = ""; state.sel_kind = "";
        request_render();
        return;
    }
}

fn label(text) { #{ type: "text", value: text, size: 10.0, weight: "bold", color: "fg_muted" } }

// Re-render on pane resize so layout tracks the new canvas size.
fn on_resize(w, h) { request_render(); }

fn render(w, h) {
    let rows = [];

    let status = if state.busy { "loading…" }
                 else if !state.loaded { "connecting…" }
                 else { state.types.len() + " types · " + state.enums.len() + " enums" };
    rows.push(#{ type: "hstack", gap: 6.0, align: "center", children: [
        #{ type: "text", value: status, size: 11.0, color: "fg_muted" },
        #{ type: "button", id: "refresh", label: "↻", kind: "ghost" },
    ]});

    // types
    rows.push(label("TYPES"));
    if state.types.len() == 0 && state.loaded {
        rows.push(#{ type: "text", value: "none yet", size: 11.0, color: "fg_muted" });
    }
    for t in state.types {
        let on = state.sel_kind == "type" && state.selected == t;
        rows.push(#{ type: "list-item", id: "t:" + t, selected: on, pad: 6.0,
            children: [ #{ type: "text", value: t, size: 13.0, color: if on { "accent_500" } else { "fg" } } ] });
    }

    // enums (now selectable, like types)
    if state.enums.len() > 0 {
        rows.push(label("ENUMS"));
        for e in state.enums {
            let on = state.sel_kind == "enum" && state.selected == e;
            rows.push(#{ type: "list-item", id: "e:" + e, selected: on, pad: 6.0,
                children: [ #{ type: "text", value: e, size: 13.0, color: if on { "accent_500" } else { "fg" } } ] });
        }
    }

    // detail + actions for the selection (type or enum)
    if state.selected != "" {
        let is_enum = state.sel_kind == "enum";
        let kw = if is_enum { "enum " } else { "" };
        let card = [
            #{ type: "hstack", gap: 6.0, align: "center", children: [
                #{ type: "text", value: state.selected, size: 13.0, weight: "bold", color: "fg" },
                #{ type: "badge", value: if is_enum { "enum" } else { "type" } },
            ]},
        ];

        // inspect: fields (types) or variants (enums)
        let detail = if is_enum { block_under(state.schema, "Enums:", state.selected) }
                     else { block_under(state.schema, "Entity types:", state.selected) };
        if detail.len() == 0 {
            card.push(#{ type: "text", value: "(no " + (if is_enum { "variants" } else { "fields" }) + ")",
                         size: 11.0, color: "fg_muted" });
        }
        for d in detail {
            card.push(#{ type: "text", value: d, size: 11.0, color: "fg_muted", family: "mono" });
        }

        // actions
        let actions = [];
        if !is_enum { actions.push(#{ type: "button", id: "browse", label: "browse", kind: "filled" }); }
        actions.push(#{ type: "button", id: "drop", label: "drop", kind: "ghost" });
        actions.push(#{ type: "button", id: "purge", label: "purge", kind: "ghost" });
        card.push(#{ type: "hstack", gap: 6.0, children: actions });

        // confirmations
        if state.confirm == "drop:" + state.selected {
            card.push(#{ type: "text",
                value: "Drop " + kw + "'" + state.selected + "'? Removes it from the schema. Re-define to restore.",
                size: 11.0, color: "warn" });
            card.push(#{ type: "hstack", gap: 6.0, children: [
                #{ type: "button", id: "confirm", label: "yes, drop", kind: "filled" },
                #{ type: "button", id: "cancel", label: "cancel", kind: "ghost" },
            ]});
        } else if state.confirm == "purge:" + state.selected {
            card.push(#{ type: "text",
                value: "Hard-purge " + kw + "'" + state.selected + "'? Deletes its definition (and data for types). Irreversible.",
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
       style: #{ background: "surface_1", width: "100%", min_height: "100%",
                 padding: #{ top: 10.0, right: 10.0, bottom: 10.0, left: 10.0 } },
       children: [ #{ type: "scroll", pad: 0.0, gap: 6.0, children: rows } ] }
}
