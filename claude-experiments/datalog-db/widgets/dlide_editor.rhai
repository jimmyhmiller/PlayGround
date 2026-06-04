// dlide_editor.rhai — datalog IDE: the query editor pane.
//
// One of a set of communicating widgets (editor / results / schema /
// history) that talk over the widget message bus (emit / on_message,
// project-scoped, SEPARATE from the Claude bus). The editor owns the
// query text and broadcasts it; it never runs queries itself.
//
//   emits:   sql.run {sql}            (Run / Cmd+Enter)
//   listens: sql.set_editor {sql}     (history: load a past query)
//            sql.use_table {table}    (schema: prefill a browse)
//            sql.result {ok,summary}  (results: show last status)
//            conn.state {host}        (retained: current DB host)

if !("sql" in state)    { state.sql = ""; }
if !("host" in state)   { state.host = "127.0.0.1:5557"; }
if !("status" in state) { state.status = ""; }

fn on_input_change(id, value) {
    if id == "sql" { state.sql = value; request_render(); }
}

fn on_input_submit(id, value) {        // Cmd/Ctrl+Enter in the textarea
    if id == "sql" {
        state.sql = value;
        emit("sql.run", #{ sql: value });
        request_render();
    }
}

fn on_click(x, y, shift, cmd, id) {
    if id == "run" { emit("sql.run", #{ sql: state.sql }); return; }
    if id == "clear" { state.sql = ""; request_render(); return; }
    if id == "ex_schema" { state.sql = "schema"; emit("sql.run", #{ sql: "schema" }); request_render(); return; }
}

fn on_message(topic, payload, sender) {
    if sender == my_id() { return; }
    if topic == "sql.set_editor" {
        state.sql = payload.sql;
        host_log("[dlide_editor] set_editor: " + payload.sql);
        request_render();
    } else if topic == "sql.use_table" {
        state.sql = "list " + payload.table + " limit 50";
        host_log("[dlide_editor] use_table: " + payload.table);
        request_render();
    } else if topic == "conn.state" {
        state.host = payload.host;
        request_render();
    } else if topic == "sql.result" {
        state.status = if payload.ok { "ok · " + payload.summary } else { "error · " + payload.summary };
        request_render();
    }
}

fn render(w, h) {
    let status_color = if state.status.starts_with("error") { "#e0746f" }
                       else if state.status.starts_with("ok") { "#6cc08a" }
                       else { "fg_muted" };

    #{ type: "vstack", gap: 8.0,
       style: #{ background: "surface_1", width: "100%", height: "100%",
                 padding: #{ top: 10.0, right: 12.0, bottom: 12.0, left: 12.0 } },
       children: [
        // editor surface fills available height
        #{ type: "frame", pad: 8.0, gap: 0.0,
           style: #{ background: "surface_2", radius: "radius_md", flex_grow: 1.0,
                     border: #{ color: "surface_3", width: 1.0 } },
           children: [
             // width/rows are the only size levers today — TextArea ignores
             // style.flex_grow/width/height (bug filed). Big fixed size so the
             // editor is usable; will switch to flex once the bug lands.
             #{ type: "textarea", id: "sql", value: state.sql, rows: 12, width: 1480.0,
                style: #{ background: "surface_2" },
                placeholder: "find ?n, ?age where ?u: User { name: ?n, age: ?age }   (Cmd+Enter to run)\nlist · schema · assert · define · drop · purge all work" },
           ] },
        // toolbar: primary Run + secondary actions, status on the right
        #{ type: "hstack", gap: 6.0, align: "center", children: [
            #{ type: "button", id: "run", label: "Run  ⌘↵", kind: "filled" },
            #{ type: "button", id: "clear", label: "Clear", kind: "ghost" },
            #{ type: "button", id: "ex_schema", label: "schema", kind: "ghost" },
            #{ type: "text", value: "   " + state.status, size: 11.0, color: status_color },
        ]},
    ] }
}
