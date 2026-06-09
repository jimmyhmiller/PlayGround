//! glaze_ui — a showcase of real UI components (cards, stat tiles, a pricing
//! card, a glassy profile card, badges, buttons, toggles) laid out with flex and
//! styled entirely by Glaze: gradient buttons/banners and a glowing avatar are
//! `overlay shader {}` layers; everything else is tokens + box styling.

use std::collections::HashMap;
use std::io::{self, BufRead, Write};

use glaze::{Program, parse};
use widget_bevy::glaze_style::{
    hex, resolve_select_style, resolve_tabs_style, to_bar_style, to_checkbox_style, to_dialog_style,
    to_popover_style, to_radio_style, to_slider_style, to_stepper_style, to_style, to_toast_style,
    to_toggle_style, to_tooltip_style,
};
use widget_bevy::glaze_style::to_table_style;
use widget_bevy::protocol::{
    Align, Border, ButtonKind, Edges, Element, HostEvent, Style, TabItem, TableColumn, Weight,
    WidgetMsg,
};

const SHEET: &str = r#"
    token surface = oklch(0.21 0.012 255)
    token surf2   = oklch(0.26 0.012 255)
    token line    = oklch(0.45 0.012 255 / 0.5)
    token fg      = oklch(0.94 0.008 255)
    token muted   = oklch(0.66 0.012 255)
    token gold    = oklch(0.80 0.13 85)
    token teal    = oklch(0.76 0.10 195)
    token violet  = oklch(0.66 0.16 290)
    token green   = oklch(0.76 0.14 150)
    token rose    = oklch(0.67 0.19 18)

    // ---- functions: a spacing scale + a reusable shader circle ----
    fn space(n)     = n * 4px
    fn circle(p, r) = smoothstep(r, r - 0.04, length(p - vec2(0.5, 0.5)))

    style card {
        fill   surface
        radius space(4)        // 16px, from the spacing fn
        border line 1px
        pad    space(4.5)      // 18px
        grow   1
        min_width 0
    }
    style stat {
        let r  = space(3)      // a local variable
        fill   surf2
        radius r
        border line 1px
        pad    space(4)
        grow   1
        min_width 0
    }

    // Interactive gradient CTA. `hover` is fed by the host every frame and
    // eased in the material runtime, without rebuilding the widget tree.
    style cta {
        radius 10px
        pad    12px 22px
        overlay shader {
            let base = mix(violet, gold, uv.x)
            emit mix(base, gold, hover * 0.35)
        }
    }
    // gradient header banner — violet→teal
    style banner {
        radius 12px
        pad    14px
        overlay shader { emit mix(violet, teal, uv.x) }
    }

    // glassy card: opaque base + a soft top sheen
    style glass {
        fill   surface
        radius 16px
        border line 1px
        pad    18px
        grow   1
        min_width 0
        overlay shader {
            let d = length(uv - vec2(0.5, 0.0))
            emit vec4(1.0, 1.0, 1.0, smoothstep(1.1, 0.0, d) * 0.10)
        }
    }

    // RESPONSIVE: a row of cards that stacks into a column when the pane is
    // narrow. The breakpoint lives here, in Glaze — not in the widget's code.
    style row_resp {
        direction row
        when vw < 560 {
            direction column
        }
    }

    // ---- Phase 1b: the widened layer stack (things flat Style can't say) ----
    // multi-stop linear gradient fill (no shader authored — `gradient` is a layer)
    style g_linear {
        radius 12px
        height 72px
        grow   1
        min_width 0
        pad    space(3)
        gradient 90 violet teal gold
    }
    // angled gradient with explicit stop offsets
    style g_diag {
        radius 12px
        height 72px
        grow   1
        min_width 0
        pad    space(3)
        gradient 45 rose 0% gold 60% teal 100%
    }
    // per-side borders: a thick left accent + a thin bottom rule
    style edged {
        fill   surf2
        radius 0px
        height 72px
        grow   1
        min_width 0
        pad    space(3)
        border_left   gold 4px
        border_bottom teal 2px
    }
    // inner (inset) shadow over a solid fill — a "pressed well". Uses a light
    // teal fill so the dark inner shadow reads clearly.
    style insets {
        fill   teal
        radius 12px
        height 72px
        grow   1
        min_width 0
        pad    space(3)
        inset_shadow oklch(0 0 0 / 0.85) 22px 0 3px
    }
    // outset drop shadow with spread, on a gold tile
    style raised {
        fill   gold
        radius 12px
        height 72px
        grow   1
        min_width 0
        pad    space(3)
        shadow oklch(0 0 0 / 0.55) 20px 9px 2px
    }

    // ---- Phase 1c: a slot-styled component. `Bar` has two slots (track, fill);
    // Glaze styles each `part {}` independently. Here the fill is a 1b gradient
    // and the track carries an inset shadow — composing 1b + 1c.
    style progress {
        track {
            fill   surf2
            radius 999px
            inset_shadow oklch(0 0 0 / 0.6) 7px 0 1px
        }
        fill {
            radius 999px
            gradient 90 teal violet
        }
    }
    style progress_gold {
        track {
            fill   surf2
            radius 999px
        }
        fill {
            fill   gold
            radius 999px
        }
    }

    // ---- Phase 1d: retrofit components, styled by slots + discrete state ----
    style toggle {
        track {
            fill   surf2
            radius 999px
            :checked { fill teal }   // discrete state: on → teal track
        }
        knob {
            fill   fg
            radius 999px
        }
    }
    style tabbar {
        strip {
            fill   surf2
            radius 8px
        }
        tab {
            radius 6px
            :selected { fill line }  // active tab gets a raised cell
        }
        indicator {
            height 3px
            radius 999px
            gradient 90 teal violet  // a 1b gradient under the active tab
        }
    }
    // ---- Phase 2: a net-new component (Slider) built on the slot system ----
    style slider {
        track {
            fill   surf2
            radius 999px
        }
        range {
            radius 999px
            gradient 90 teal violet   // value-driven fill is a 1b gradient
        }
        thumb {
            fill   fg
            radius 999px
            overlay shader {           // a soft glow on the handle (1b shader)
                let d = length(uv - vec2(0.5, 0.5))
                emit vec4(0.55, 0.85, 0.95, smoothstep(0.5, 0.1, d) * 0.5)
            }
        }
    }
    style notif {
        surface {
            fill   surf2
            radius 8px
            border teal 1px
        }
    }
    style pop {
        trigger {
            fill   surf2
            radius 6px
            border line 1px
        }
        surface {
            fill   surface
            radius 10px
            border line 1px
        }
    }
    style modal {
        scrim {
            fill oklch(0 0 0 / 0.6)
        }
        panel {
            fill   surface
            radius 14px
            border line 1px
        }
    }
    style tip {
        bubble {
            fill   surf2
            radius 6px
            border line 1px
        }
    }
    // ---- Phase 3: floating overlay component. The trigger is in-pane; the
    // menu renders on the overlay layer, escaping pane bounds. item:selected.
    style picker {
        trigger {
            fill   surf2
            radius 6px
            border line 1px
        }
        menu {
            fill   surface
            radius 8px
            border line 1px
        }
        item {
            radius 5px
            :selected { fill line }
        }
    }
    style stepper {
        field {
            fill   surf2
            radius 6px
            border line 1px
        }
        button {
            fill   surf2
            radius 6px
            border teal 1px
        }
    }
    style radiogroup {
        dot {
            radius 999px
            gradient 135 teal violet   // the selected dot is a 1b gradient
        }
    }
    style checkbox {
        box {
            fill   surf2
            radius 5px
            border line 1px
            :checked { border teal 1px }
        }
        check {
            radius 3px
            gradient 135 teal violet   // the tick is a 1b gradient
        }
    }
    style datatable {
        panel {
            fill   surface
            radius 10px
            border line 1px
        }
        header {
            fill   surf2
        }
        zebra {
            fill   surf2
        }
    }

    // glowing circular avatar
    style avatar {
        width  58px
        height 58px
        radius 999px
        overlay shader {
            let c    = circle(uv, 0.48)                       // reuse the fn
            let glow = smoothstep(0.5, 0.0, length(uv - vec2(0.5, 0.5))) * 0.5
            emit vec4(0.66 * (c + glow), 0.42 * (c + glow), 0.95 * (c + glow), c)
        }
    }
"#;

// ---- helpers ----------------------------------------------------------------

fn tok(prog: &Program, name: &str) -> String {
    match prog.eval_token(name) {
        Ok(glaze::Value::Color(c)) => hex(c),
        _ => "#ffffff".into(),
    }
}

fn text(value: &str, color: &str, size: f32, bold: bool) -> Element {
    Element::Text {
        value: value.into(),
        color: Some(color.into()),
        size: Some(size),
        weight: bold.then_some(Weight::Bold),
        family: Some("font_family_body".into()),
        selectable: false,
    }
}

fn col(gap: f32, children: Vec<Element>) -> Element {
    Element::Vstack { gap, pad: 0.0, children, style: None }
}
fn row(gap: f32, align: Align, children: Vec<Element>) -> Element {
    Element::Hstack { gap, pad: 0.0, align, children, style: None }
}

/// A Glaze-styled frame; resolve errors render as a loud red tile (house rule).
fn glz(prog: &Program, name: &str, gap: f32, children: Vec<Element>) -> Element {
    match prog.resolve(name, &HashMap::new(), &[]) {
        Ok(c) => Element::Frame { gap, pad: 0.0, children, style: Some(to_style(&c)) },
        Err(e) => Element::Frame {
            gap: 4.0,
            pad: 0.0,
            children: vec![text(&format!("glaze: {e}"), "#ff6b5a", 10.0, true)],
            style: Some(Style {
                background: Some("#2a1414".into()),
                border: Some(Border { color: "#ff6b5a".into(), width: 1.0 }),
                radius: Some("8".into()),
                padding: Some(Edges::all(10.0)),
                ..Default::default()
            }),
        },
    }
}

fn badge(value: &str, color: &str) -> Element {
    Element::Badge { value: value.into(), color: Some(color.into()), selectable: false, style: None }
}

/// A progress bar that fills its container width (so it shrinks/grows with the
/// card) — a track Frame at 100% holding a fill Frame at `frac`%.
fn bar(frac: f32, accent: &str, track: &str) -> Element {
    let pct = (frac * 100.0).clamp(0.0, 100.0);
    let fill = Element::Frame {
        gap: 0.0,
        pad: 0.0,
        children: vec![Element::Spacer { size: 6.0 }],
        style: Some(Style {
            background: Some(accent.into()),
            radius: Some("3".into()),
            width: Some(format!("{pct}%")),
            ..Default::default()
        }),
    };
    Element::Frame {
        gap: 0.0,
        pad: 0.0,
        children: vec![fill],
        style: Some(Style {
            background: Some(track.into()),
            radius: Some("3".into()),
            width: Some("100%".into()),
            ..Default::default()
        }),
    }
}

/// An `Element::Bar` styled by a Glaze slot style (`track` + `fill` parts). A
/// bad slot name in the `.glz` surfaces as a loud red tile via `glz`/our error
/// path; here a clean resolve yields a typed `BarStyle`.
fn glaze_bar(prog: &Program, style_name: &str, frac: f32, width: f32) -> Element {
    let style = prog
        .resolve_slots(style_name, &HashMap::new(), &[])
        .ok()
        .and_then(|s| to_bar_style(&s).ok());
    Element::Bar {
        value: frac,
        max: 1.0,
        color: None,
        track: None,
        width,
        height: 14.0,
        style,
    }
}

/// A slot-styled Slider — the first net-new Phase 2 component. `track`/`range`/
/// `thumb` come from Glaze; the value is owned by the widget (updated on the
/// host's `slider-change` events) and echoed back here.
fn glaze_slider(prog: &Program, id: &str, value: f32) -> Element {
    let style = prog
        .resolve_slots("slider", &HashMap::new(), &[])
        .ok()
        .and_then(|s| to_slider_style(&s).ok());
    Element::Slider {
        id: id.into(),
        value,
        min: 0.0,
        max: 1.0,
        step: 0.0,
        width: 320.0,
        height: 22.0,
        style,
    }
}

/// A Popover (Phase 3) — proves anchored positioning + arbitrary content
/// compose. A click opens a floating card with action buttons.
fn glaze_popover(prog: &Program, muted: &str) -> Element {
    let style = prog
        .resolve_slots("pop", &HashMap::new(), &[])
        .ok()
        .and_then(|s| to_popover_style(&s).ok());
    let content = Element::Vstack {
        gap: 6.0,
        pad: 0.0,
        children: vec![
            text("ACCOUNT", muted, 10.0, true),
            Element::Button {
                id: "pop-profile".into(),
                label: "View profile".into(),
                kind: ButtonKind::Ghost,
                style: None,
            },
            Element::Button {
                id: "pop-signout".into(),
                label: "Sign out".into(),
                kind: ButtonKind::Ghost,
                style: None,
            },
        ],
        style: None,
    };
    Element::Popover {
        id: "acct".into(),
        label: "Account ▾".into(),
        content: Some(Box::new(content)),
        width: 200.0,
        style,
    }
}

/// A modal Dialog (Phase 3) — proves arbitrary slot-styled content (with working
/// buttons) rendered on the overlay layer. Body buttons fire normal `click`
/// events; the scrim/Escape sends `dialog-close`.
fn glaze_dialog(prog: &Program, open: bool, fg: &str, muted: &str) -> Element {
    let style = prog
        .resolve_slots("modal", &HashMap::new(), &[])
        .ok()
        .and_then(|s| to_dialog_style(&s).ok());
    let body = Element::Vstack {
        gap: 16.0,
        pad: 0.0,
        children: vec![
            text(
                "Switch to the Team plan? Your card will be charged the prorated difference today.",
                muted,
                12.0,
                false,
            ),
            row(
                10.0,
                Align::Center,
                vec![
                    Element::Button {
                        id: "dlg-cancel".into(),
                        label: "Cancel".into(),
                        kind: ButtonKind::Outline,
                        style: None,
                    },
                    Element::Button {
                        id: "dlg-confirm".into(),
                        label: "Confirm".into(),
                        kind: ButtonKind::Filled,
                        style: None,
                    },
                ],
            ),
        ],
        style: None,
    };
    let _ = fg;
    Element::Dialog {
        id: "settings".into(),
        open,
        title: "Confirm change".into(),
        body: Some(Box::new(body)),
        width: 360.0,
        style,
    }
}

/// A slot-styled Tooltip (Phase 3). Hovering the label shows the hint on the
/// overlay layer.
fn tooltip(prog: &Program, label: &str, text: &str) -> Element {
    let style = prog
        .resolve_slots("tip", &HashMap::new(), &[])
        .ok()
        .and_then(|s| to_tooltip_style(&s).ok());
    Element::Tooltip {
        label: label.into(),
        text: text.into(),
        style,
    }
}

/// A slot-styled Select (Phase 3). The trigger is in-pane; the dropdown floats
/// on the overlay layer (host-owned open state), emitting `select-change`.
fn glaze_select(prog: &Program, id: &str, options: &[(&str, &str)], value: &str) -> Element {
    let style = resolve_select_style(prog, "picker").ok();
    Element::Select {
        id: id.into(),
        options: options
            .iter()
            .map(|(i, l)| TabItem {
                id: (*i).into(),
                label: (*l).into(),
            })
            .collect(),
        value: value.into(),
        placeholder: "Select…".into(),
        width: 200.0,
        style,
    }
}

/// A slot-styled Stepper. The +/- buttons carry the precomputed value, so a
/// click round-trips as `HostEvent::NumberChange`.
fn glaze_stepper(prog: &Program, id: &str, value: f32) -> Element {
    let style = prog
        .resolve_slots("stepper", &HashMap::new(), &[])
        .ok()
        .and_then(|s| to_stepper_style(&s).ok());
    Element::Stepper {
        id: id.into(),
        value,
        min: 0.0,
        max: 20.0,
        step: 1.0,
        style,
    }
}

/// A slot-styled RadioGroup. The `dot` is a Glaze slot (gradient here); the
/// ring uses the default selected-affordance. Emits `radio-select` on click.
fn glaze_radio(prog: &Program, id: &str, options: &[(&str, &str)], selected: &str) -> Element {
    let style = prog
        .resolve_slots("radiogroup", &HashMap::new(), &[])
        .ok()
        .and_then(|s| to_radio_style(&s).ok());
    Element::RadioGroup {
        id: id.into(),
        options: options
            .iter()
            .map(|(i, l)| TabItem {
                id: (*i).into(),
                label: (*l).into(),
            })
            .collect(),
        selected: selected.into(),
        style,
    }
}

/// A slot-styled Checkbox. Like the Toggle, the `:checked` state is resolved at
/// the widget; the `check` mark renders only when checked. Reuses the `toggle`
/// event so a click round-trips as `HostEvent::Toggle`.
fn glaze_checkbox(prog: &Program, id: &str, label: &str, checked: bool) -> Element {
    let states: &[&str] = if checked { &["checked"] } else { &[] };
    let style = prog
        .resolve_slots("checkbox", &HashMap::new(), states)
        .ok()
        .and_then(|s| to_checkbox_style(&s).ok());
    Element::Checkbox {
        id: id.into(),
        label: label.into(),
        checked,
        style,
    }
}

/// A slot-styled Toggle. The track's `:checked` state is resolved here (CPU
/// discrete-state model): we pass `["checked"]` when on so Glaze lands the
/// teal track; the knob's value-driven x-position is the renderer's job.
fn glaze_toggle(prog: &Program, id: &str, label: &str, checked: bool) -> Element {
    let states: &[&str] = if checked { &["checked"] } else { &[] };
    let style = prog
        .resolve_slots("toggle", &HashMap::new(), states)
        .ok()
        .and_then(|s| to_toggle_style(&s).ok());
    Element::Toggle {
        id: id.into(),
        label: label.into(),
        checked,
        style,
    }
}

/// A slot-styled Tabs. `resolve_tabs_style` precomputes the resting and
/// `:selected` tab plans; the renderer swaps the selected one in for the active
/// tab and draws the gradient indicator under it.
fn glaze_tabs(prog: &Program, id: &str, items: &[(&str, &str)], selected: &str) -> Element {
    let style = resolve_tabs_style(prog, "tabbar").ok();
    Element::Tabs {
        id: id.into(),
        items: items
            .iter()
            .map(|(i, l)| TabItem {
                id: (*i).into(),
                label: (*l).into(),
            })
            .collect(),
        selected: selected.into(),
        style,
    }
}

/// A slot-styled Table (panel / header / zebra surfaces from Glaze).
fn glaze_table(prog: &Program) -> Element {
    let style = prog
        .resolve_slots("datatable", &HashMap::new(), &[])
        .ok()
        .and_then(|s| to_table_style(&s).ok());
    let col = |h: &str, w: Option<f32>| TableColumn {
        header: h.into(),
        width: w,
        align: Align::Start,
    };
    Element::Table {
        columns: vec![col("Service", Some(140.0)), col("Status", Some(90.0)), col("Latency", None)],
        rows: vec![
            vec!["api".into(), "live".into(), "42ms".into()],
            vec!["worker".into(), "live".into(), "8ms".into()],
            vec!["cache".into(), "degraded".into(), "120ms".into()],
            vec!["db".into(), "live".into(), "5ms".into()],
        ],
        zebra: true,
        selectable: true,
        style,
    }
}

fn stat_card(prog: &Program, label: &str, value: &str, tone: &str, frac: f32) -> Element {
    let accent = tok(prog, tone);
    glz(
        prog,
        "stat",
        8.0,
        vec![
            text(label, &tok(prog, "muted"), 10.0, true),
            text(value, &accent, 26.0, true),
            bar(frac, &accent, &tok(prog, "surface")),
        ],
    )
}

fn feature(prog: &Program, label: &str) -> Element {
    row(8.0, Align::Center, vec![badge("✓", &tok(prog, "green")), text(label, &tok(prog, "fg"), 12.0, false)])
}

fn cta(prog: &Program, label: &str) -> Element {
    let style = prog
        .resolve("cta", &HashMap::new(), &[])
        .ok()
        .map(|c| to_style(&c));
    Element::Button {
        id: "get-pro".into(),
        label: label.into(),
        kind: ButtonKind::Filled,
        style,
    }
}

/// A row of cards whose row↔column behaviour is decided by Glaze: the
/// `row_resp` style's `when vw < 560` breakpoint flips its `direction`. We just
/// feed Glaze the current viewport width; the responsive logic is in the `.glz`.
fn resp_row(prog: &Program, width: f32, gap: f32, children: Vec<Element>) -> Element {
    let style = prog
        .resolve_at("row_resp", &HashMap::new(), &[], width, f32::MAX)
        .ok()
        .map(|c| to_style(&c));
    Element::Hstack { gap, pad: 0.0, align: Align::Stretch, children, style }
}

fn build_ui(
    prog: &Program,
    width: f32,
    tab: &str,
    slider: f32,
    checks: &HashMap<String, bool>,
    radio: &str,
    qty: f32,
    country: &str,
    dialog_open: bool,
    toast_shown: bool,
) -> Element {
    let checked = |id: &str| checks.get(id).copied().unwrap_or(false);
    let fg = tok(prog, "fg");
    let muted = tok(prog, "muted");

    let header = col(
        4.0,
        vec![
            text("Glaze UI", &tok(prog, "gold"), 30.0, true),
            text("components · flex layout · shader-styled — all from one .glz", &muted, 12.0, false),
        ],
    );

    let stats = resp_row(
        prog,
        width,
        14.0,
        vec![
            stat_card(prog, "REVENUE", "$48.2k", "gold", 0.72),
            stat_card(prog, "ACTIVE USERS", "1,284", "teal", 0.55),
            stat_card(prog, "CHURN", "2.1%", "rose", 0.21),
        ],
    );

    let pricing = glz(
        prog,
        "card",
        12.0,
        vec![
            glz(
                prog,
                "banner",
                2.0,
                vec![
                    text("PRO PLAN", "#fdf6ec", 10.0, true),
                    text("$29 / mo", "#fdf6ec", 22.0, true),
                ],
            ),
            feature(prog, "Unlimited panes"),
            feature(prog, "Shader-styled components"),
            feature(prog, "Priority support"),
            Element::Spacer { size: 4.0 },
            cta(prog, "Get Pro"),
        ],
    );

    let profile = glz(
        prog,
        "glass",
        12.0,
        vec![
            row(
                12.0,
                Align::Center,
                vec![
                    glz(prog, "avatar", 0.0, vec![]),
                    col(
                        2.0,
                        vec![
                            text("Ada Lovelace", &fg, 14.0, true),
                            text("Design Engineer", &muted, 11.0, false),
                        ],
                    ),
                ],
            ),
            text("Building delightful interfaces with a tiny staged style language.", &muted, 11.0, false),
            row(
                10.0,
                Align::Center,
                vec![
                    Element::Button {
                        id: "follow".into(),
                        label: "Follow".into(),
                        kind: ButtonKind::Filled,
                        style: None,
                    },
                    Element::Toggle {
                        id: "notify".into(),
                        label: "Notify".into(),
                        checked: true,
                        style: None,
                    },
                ],
            ),
        ],
    );

    let middle = resp_row(prog, width, 16.0, vec![pricing, profile]);

    // Phase 1b showcase: gradient / per-side border / inset & spread shadow,
    // each a pure Glaze layer (no hand-authored shader) flowing through the
    // widened paint plan.
    let layers = col(
        8.0,
        vec![
            text("WIDENED LAYER STACK · PHASE 1B", &muted, 10.0, true),
            resp_row(
                prog,
                width,
                12.0,
                vec![
                    glz(prog, "g_linear", 0.0, vec![text("linear gradient", "#fdf6ec", 11.0, true)]),
                    glz(prog, "g_diag", 0.0, vec![text("angled gradient", "#fdf6ec", 11.0, true)]),
                    glz(prog, "edged", 0.0, vec![text("per-side border", &fg, 11.0, true)]),
                    glz(prog, "insets", 0.0, vec![text("inset shadow", "#0c2226", 11.0, true)]),
                    glz(prog, "raised", 0.0, vec![text("drop shadow · spread", "#1a1407", 11.0, true)]),
                ],
            ),
        ],
    );

    let badges = col(
        8.0,
        vec![
            text("STATUS", &muted, 10.0, true),
            row(
                8.0,
                Align::Center,
                vec![
                    badge("● Live", &tok(prog, "green")),
                    badge("Beta", &tok(prog, "violet")),
                    badge("v0.3", &tok(prog, "teal")),
                    badge("3 issues", &tok(prog, "rose")),
                ],
            ),
        ],
    );

    // Phase 1c: slot-styled Bars. Each is one Element::Bar whose track/fill
    // surfaces are styled independently by Glaze `part {}` blocks.
    let slots = col(
        8.0,
        vec![
            text("SLOT-STYLED COMPONENT · PHASE 1C", &muted, 10.0, true),
            text(
                "Element::Bar · track + fill parts — gradient fill over an inset-shadow track",
                &muted,
                11.0,
                false,
            ),
            glaze_bar(prog, "progress", 0.7, 520.0),
            glaze_bar(prog, "progress_gold", 0.4, 520.0),
        ],
    );

    // Phase 1d: retrofit components whose hardcoded render.rs styling is now
    // driven by Glaze slots + discrete state (Tabs strip/tab/indicator with
    // :selected; Toggle track/knob with :checked).
    // Tabs that actually switch the panel below them.
    let tab_panel = match tab {
        // Activity
        "b" => col(
            8.0,
            vec![
                text("RECENT ACTIVITY", &muted, 10.0, true),
                row(10.0, Align::Center, vec![text("Deploys", &fg, 12.0, false), glaze_bar(prog, "progress", 0.82, 360.0)]),
                row(10.0, Align::Center, vec![text("Reviews", &fg, 12.0, false), glaze_bar(prog, "progress_gold", 0.46, 360.0)]),
            ],
        ),
        // Settings — the toggles live here, and they actually toggle now.
        "c" => col(
            10.0,
            vec![
                text("SETTINGS", &muted, 10.0, true),
                row(
                    20.0,
                    Align::Center,
                    vec![
                        glaze_toggle(prog, "notifications", "Notifications", checked("notifications")),
                        glaze_toggle(prog, "autosync", "Auto-sync", checked("autosync")),
                    ],
                ),
            ],
        ),
        // Overview (default)
        _ => glaze_table(prog),
    };
    let retrofit = col(
        10.0,
        vec![
            text("SLOT-RETROFIT COMPONENTS · PHASE 1D — click the tabs", &muted, 10.0, true),
            glaze_tabs(
                prog,
                "demo-tabs",
                &[("a", "Overview"), ("b", "Activity"), ("c", "Settings")],
                tab,
            ),
            tab_panel,
        ],
    );

    // Phase 2: the first net-new interactive component on the slot system.
    let phase2 = col(
        8.0,
        vec![
            text("NEW COMPONENT · PHASE 2 — DRAGGABLE SLIDER", &muted, 10.0, true),
            row(
                14.0,
                Align::Center,
                vec![
                    glaze_slider(prog, "volume", slider),
                    text(
                        &format!("{}%", (slider * 100.0).round() as i32),
                        &tok(prog, "teal"),
                        13.0,
                        true,
                    ),
                ],
            ),
            row(
                24.0,
                Align::Center,
                vec![
                    glaze_checkbox(prog, "terms", "Accept terms", checked("terms")),
                    glaze_checkbox(prog, "news", "Email updates", checked("news")),
                    glaze_checkbox(prog, "beta", "Join beta", checked("beta")),
                ],
            ),
            row(
                40.0,
                Align::Start,
                vec![
                    glaze_radio(
                        prog,
                        "plan",
                        &[("free", "Free"), ("pro", "Pro"), ("team", "Team")],
                        radio,
                    ),
                    col(
                        6.0,
                        vec![
                            text("SEATS", &muted, 10.0, true),
                            glaze_stepper(prog, "seats", qty),
                        ],
                    ),
                ],
            ),
        ],
    );

    // Phase 3: the floating overlay component — a Select whose dropdown escapes
    // the pane onto the overlay layer.
    let phase3 = col(
        8.0,
        vec![
            text("FLOATING OVERLAY · PHASE 3 — SELECT", &muted, 10.0, true),
            row(
                14.0,
                Align::Center,
                vec![
                    glaze_select(
                        prog,
                        "country",
                        &[("us", "United States"), ("ca", "Canada"), ("uk", "United Kingdom"), ("de", "Germany")],
                        country,
                    ),
                    text(&format!("→ {country}"), &tok(prog, "teal"), 12.0, true),
                ],
            ),
            row(
                8.0,
                Align::Center,
                vec![
                    text("Hover the hint:", &muted, 12.0, false),
                    tooltip(prog, "what is this?", "A floating hint on the overlay layer — it escapes the pane."),
                ],
            ),
            row(
                10.0,
                Align::Center,
                vec![
                    Element::Button {
                        id: "open-dialog".into(),
                        label: "Open dialog".into(),
                        kind: ButtonKind::Filled,
                        style: None,
                    },
                    // 0-size in-pane; renders centered on the overlay when open.
                    glaze_dialog(prog, dialog_open, &fg, &muted),
                    glaze_popover(prog, &muted),
                    // 0-size; renders at the bottom-right corner on the overlay.
                    // Click it to dismiss (toast-dismiss → drops it from the frame).
                    if toast_shown {
                        Element::Toast {
                            id: "saved".into(),
                            text: "Settings saved ✓".into(),
                            style: prog
                                .resolve_slots("notif", &HashMap::new(), &[])
                                .ok()
                                .and_then(|s| to_toast_style(&s).ok()),
                        }
                    } else {
                        Element::Spacer { size: 0.0 }
                    },
                ],
            ),
        ],
    );

    Element::Vstack {
        gap: 18.0,
        pad: 4.0,
        children: vec![header, stats, middle, layers, slots, retrofit, phase2, phase3, badges],
        style: None,
    }
}

fn main() {
    let prog = match parse(SHEET) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("glaze_ui: stylesheet failed to compile: {e}");
            std::process::exit(1);
        }
    };
    let stdout = io::stdout();
    let mut out = stdout.lock();
    let _ = writeln!(
        out,
        "{}",
        serde_json::to_string(&WidgetMsg::Title { value: "Glaze UI".into() }).unwrap()
    );
    // Track the pane's content width (row↔column breakpoint), the active tab,
    // and the slider value (the widget owns it; the host drives slider-change).
    let mut width = 700.0_f32;
    let mut tab = "a".to_string();
    let mut slider = 0.65_f32;
    let mut checks: HashMap<String, bool> = HashMap::from([
        ("terms".into(), true),
        ("news".into(), false),
        ("beta".into(), true),
        ("notifications".into(), true),
        ("autosync".into(), false),
    ]);
    let mut radio = "pro".to_string();
    let mut qty = 3.0_f32;
    let mut country = "ca".to_string();
    // Default open when GLAZE_DIALOG is set (for the static snapshot).
    let mut dialog_open = std::env::var("GLAZE_DIALOG").is_ok();
    let mut toast_shown = true;
    emit(&mut out, &prog, width, &tab, slider, &checks, &radio, qty, &country, dialog_open, toast_shown);
    for line in io::stdin().lock().lines() {
        let Ok(line) = line else { break };
        let Ok(evt) = serde_json::from_str::<HostEvent>(&line) else { continue };
        match evt {
            HostEvent::Close => return,
            HostEvent::Init { width: w, .. } => width = w,
            HostEvent::Resize { width: w, .. } => width = w,
            HostEvent::TabSelect { id, tab: t } if id == "demo-tabs" => tab = t,
            HostEvent::SliderChange { id, value } if id == "volume" => slider = value,
            HostEvent::RadioSelect { id, option } if id == "plan" => radio = option,
            HostEvent::NumberChange { id, value } if id == "seats" => qty = value,
            HostEvent::SelectChange { id, value } if id == "country" => country = value,
            HostEvent::Click { id } if id == "open-dialog" => dialog_open = true,
            HostEvent::Click { id } if id == "dlg-cancel" || id == "dlg-confirm" => {
                dialog_open = false;
            }
            HostEvent::DialogClose { id } if id == "settings" => dialog_open = false,
            HostEvent::ToastDismiss { id } if id == "saved" => toast_shown = false,
            HostEvent::Toggle { id, checked } if checks.contains_key(&id) => {
                checks.insert(id, checked);
            }
            _ => {}
        }
        emit(&mut out, &prog, width, &tab, slider, &checks, &radio, qty, &country, dialog_open, toast_shown);
    }
}

#[allow(clippy::too_many_arguments)]
fn emit<W: Write>(
    out: &mut W,
    prog: &Program,
    width: f32,
    tab: &str,
    slider: f32,
    checks: &HashMap<String, bool>,
    radio: &str,
    qty: f32,
    country: &str,
    dialog_open: bool,
    toast_shown: bool,
) {
    if let Ok(s) = serde_json::to_string(&WidgetMsg::Frame {
        root: build_ui(prog, width, tab, slider, checks, radio, qty, country, dialog_open, toast_shown),
    }) {
        let _ = writeln!(out, "{s}");
        let _ = out.flush();
    }
}
