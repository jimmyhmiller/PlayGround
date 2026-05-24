// dust.rhai — system script driving the dust effect.
//
// Runs each frame. Pre-populated scope vars:
//   state    — Map. Persistent across ticks. Mutate in place.
//   events   — Array. (Empty for now until the event bus is wired in.)
//   time     — f64. Seconds since app start.
//   dt       — f64. Seconds since last tick.
//
// Engine auto-populates these uniforms on every shader that declares
// them in its UserData struct:
//   time, dt, resolution, mouse_world, focused_pane
// — so you don't need to forward them.

if !("init" in state) {
    state.init = true;
    state.last_focus_at = time;
    state.last_cursor = ();
    state.windex_t = -1.0;
    state.last_windex_req = -1.0;
}

// ---- tunables ----
//   `dust_intensity` 0..1 — set in dev_panel.rhai (or your own UI) via
//   state_set("dust_intensity", value).
//   `wipe_dust_gate_secs` — below this much natural dust, mouse motion
//   doesn't accumulate wipe. Lower it if dust is faint but you still
//   want to be able to smear.
let dust_intensity = state_get("dust_intensity", 1.0);
let gate = state_get("wipe_dust_gate_secs", 60.0);
let brush = state_get("wipe_brush_radius_px", 80.0);

// ---- compute the live dust amount ----
let dust_seconds = time - state.last_focus_at;
uniform_set("dust_seconds", dust_seconds);
uniform_set("dust_intensity", dust_intensity);
uniform_set("fg_muted", state_get("fg_muted_rgba", [0.5, 0.52, 0.56, 1.0]));

// ---- check for Windex trigger ----
// dev_panel sets state.windex_request_t = time when the button is
// clicked. We start the animation when that value changes.
let windex_req = state_get("windex_request_t", -1.0);
if windex_req > 0.0 && windex_req != state.last_windex_req {
    state.last_windex_req = windex_req;
    state.windex_t = time;
}

// ---- Windex animation ----
if state.windex_t >= 0.0 {
    let elapsed = time - state.windex_t;
    let dur = 1.6;
    if elapsed >= dur {
        state.windex_t = -1.0;
        uniform_set("windex_progress", 0.0);
        uniform_set("windex_active", 0.0);
        // Reset everything for the active project: clear smears,
        // reset focus timer so dust starts from 0.
        mask_clear("wipe_mask");
        state.last_focus_at = time;
    } else {
        uniform_set("windex_progress", elapsed / dur);
        uniform_set("windex_active", 1.0);
    }
} else {
    uniform_set("windex_progress", 0.0);
    uniform_set("windex_active", 0.0);
}

// ---- Mouse smear ----
let mouse = uniform_get("mouse_world");
if type_of(mouse) == "array" && mouse.len() == 2 {
    let cx = mouse[0]; let cy = mouse[1];
    if dust_intensity > 0.0 && dust_seconds > gate && cx >= 0.0 {
        let prev = state.last_cursor;
        let px = if type_of(prev) == "()" { cx } else { prev[0] };
        let py = if type_of(prev) == "()" { cy } else { prev[1] };
        state.last_cursor = [cx, cy];

        // Skip if over a pane.
        let over_pane = false;
        for r in pane_rects() {
            if cx >= r.x && cx <= r.x + r.w && cy >= r.y && cy <= r.y + r.h {
                over_pane = true;
                break;
            }
        }
        if !over_pane {
            // Stamp the brush along the segment from prev to cursor.
            let dx = cx - px;
            let dy = cy - py;
            let seg = (dx*dx + dy*dy).sqrt();
            let step = brush * 0.4;
            let raw = (seg / step).ceil().to_int();
            let steps = if raw < 1 { 1 } else if raw > 64 { 64 } else { raw };
            let i = 0;
            while i <= steps {
                let t = i.to_float() / steps.to_float();
                mask_paint("wipe_mask", px + dx*t, py + dy*t, brush, 1.0);
                i += 1;
            }
        }
    }
}
