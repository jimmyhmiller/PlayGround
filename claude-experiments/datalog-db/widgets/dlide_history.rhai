// dlide_history.rhai — datalog IDE: query history.
//
// Logs every command the editor runs (heard off the bus) and lets you
// click one to reload it into the editor. Pure bus widget — no DB access.
//
//   listens: sql.run {sql}            append to history
//            sql.result {ok,summary}  annotate the newest entry
//   emits:   sql.set_editor {sql}     (click → editor loads it)

if !("items" in state)  { state.items = []; }   // most recent first: [#{ sql, note }]

fn truncate(s, n) {
    let t = s;
    t.replace("\n", " ");
    if t.len() > n { t.sub_string(0, n) + "…" } else { t }
}

fn on_message(topic, payload, sender) {
    if sender == my_id() { return; }
    if topic == "sql.run" {
        let sql = payload.sql;
        // de-dup if identical to the newest entry
        if state.items.len() == 0 || state.items[0].sql != sql {
            let next = [#{ sql: sql, note: "running…" }];
            for it in state.items { next.push(it); }
            while next.len() > 30 { next.remove(next.len() - 1); }
            state.items = next;
            host_log("[dlide_history] +" + sql + " (" + state.items.len() + " total)");
        }
        request_render();
    } else if topic == "sql.result" {
        if state.items.len() > 0 {
            let head = state.items[0];
            head.note = if payload.ok { "ok · " + payload.summary } else { "error" };
            state.items[0] = head;
            request_render();
        }
    }
}

fn on_click(x, y, shift, cmd, id) {
    if id.starts_with("h:") {
        let idx = parse_int(id.sub_string(2));
        if idx >= 0 && idx < state.items.len() {
            emit("sql.set_editor", #{ sql: state.items[idx].sql });
        }
    }
}

fn note_color(note) {
    if note.starts_with("ok") { "#6cc08a" } else if note.starts_with("error") { "#e0746f" } else { "fg_muted" }
}

fn render(w, h) {
    let rows = [];
    rows.push(#{ type: "text", value: "HISTORY", size: 10.0, weight: "bold", color: "fg_muted" });
    if state.items.len() == 0 {
        rows.push(#{ type: "text", value: "nothing run yet", size: 11.0, color: "fg_muted" });
    }
    let i = 0;
    while i < state.items.len() {
        let it = state.items[i];
        rows.push(#{ type: "list-item", id: "h:" + i, pad: 6.0, children: [
            #{ type: "text", value: truncate(it.sql, 38), size: 11.0, color: "fg", family: "mono" },
            #{ type: "text", value: it.note, size: 10.0, color: note_color(it.note) },
        ] });
        i += 1;
    }
    #{ type: "vstack", gap: 4.0,
       style: #{ background: "surface_1", width: "100%", height: "100%",
                 padding: #{ top: 10.0, right: 10.0, bottom: 10.0, left: 10.0 } },
       children: [ #{ type: "scroll", pad: 0.0, gap: 4.0, children: rows } ] }
}
