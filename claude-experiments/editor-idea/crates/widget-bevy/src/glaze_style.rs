//! Adapter: a `glaze::CompiledStyle` → the existing `protocol::Style`.
//!
//! This is the whole "Glaze produces the styling the renderer already
//! understands" seam. Colors are stored linear in Glaze; the renderer parses
//! sRGB hex back to linear, so we round-trip linear → sRGB `#rrggbb[aa]`.

use crate::protocol::{
    BarStyle, CheckboxStyle, Edges, GlazeLayer, GradientStop, RadioGroupStyle, SelectStyle, Sides,
    SliderStyle, StepperStyle, Style, TableStyle, TabsStyle, ToggleStyle, TooltipStyle,
};
use glaze::{CompiledSlots, CompiledStyle, Dim, Dir, Layer, Program, Rgba};
use std::collections::HashMap;

fn lin_to_srgb(c: f32) -> f32 {
    if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.clamp(0.0, 1.0).powf(1.0 / 2.4) - 0.055
    }
}

/// linear-rgb → `#rrggbb` (or `#rrggbbaa` when not opaque).
pub fn hex(c: Rgba) -> String {
    let b8 = |x: f32| (lin_to_srgb(x).clamp(0.0, 1.0) * 255.0).round() as u8;
    let (r, g, b) = (b8(c.r), b8(c.g), b8(c.b));
    let a = (c.a.clamp(0.0, 1.0) * 255.0).round() as u8;
    if a == 255 {
        format!("#{r:02x}{g:02x}{b:02x}")
    } else {
        format!("#{r:02x}{g:02x}{b:02x}{a:02x}")
    }
}

fn dim_str(d: Dim) -> String {
    match d {
        Dim::Px(p) => format!("{p}"),
        Dim::Pct(p) => format!("{p}%"),
        Dim::Auto => "auto".into(),
    }
}

/// Convert a compiled Glaze style into a `protocol::Style`.
pub fn to_style(c: &CompiledStyle) -> Style {
    let mut s = Style::default();
    let p = c.box_.padding;
    s.padding = Some(Edges {
        top: p[0],
        right: p[1],
        bottom: p[2],
        left: p[3],
    });
    if c.box_.radius > 0.0 {
        s.radius = Some(format!("{}", c.box_.radius));
    }
    s.width = c.box_.width.map(dim_str);
    s.height = c.box_.height.map(dim_str);
    s.min_width = c.box_.min_width.map(dim_str);
    s.max_width = c.box_.max_width.map(dim_str);
    s.min_height = c.box_.min_height.map(dim_str);
    s.max_height = c.box_.max_height.map(dim_str);
    s.flex_grow = c.box_.flex_grow;
    s.flex_shrink = c.box_.flex_shrink;
    s.flex_direction = c.box_.flex_direction.map(|d| match d {
        Dir::Row => "row".to_string(),
        Dir::Column => "column".to_string(),
    });
    s.glaze_layers = c
        .layers
        .iter()
        .map(|layer| match layer {
            Layer::Fill(rgba) => GlazeLayer::Fill { color: hex(*rgba) },
            Layer::LinearGradient { angle, stops } => GlazeLayer::LinearGradient {
                angle: *angle,
                stops: stops
                    .iter()
                    .map(|s| GradientStop {
                        offset: s.offset,
                        color: hex(s.color),
                    })
                    .collect(),
            },
            Layer::Border {
                color,
                width,
                sides,
            } => GlazeLayer::Border {
                color: hex(*color),
                width: *width,
                sides: Sides {
                    top: sides.top,
                    right: sides.right,
                    bottom: sides.bottom,
                    left: sides.left,
                },
            },
            Layer::Shadow {
                color,
                blur,
                offset_x,
                offset_y,
                spread,
                inset,
            } => GlazeLayer::Shadow {
                color: hex(*color),
                blur: *blur,
                offset_x: *offset_x,
                offset_y: *offset_y,
                spread: *spread,
                inset: *inset,
            },
            Layer::Shader(cs) => GlazeLayer::Shader {
                body: cs.wgsl_body.clone(),
                overlay: cs.overlay,
            },
        })
        .collect();
    s
}

/// Validate a slotted Glaze style against a component's known slot names,
/// returning a load-time error (house rule: loud, never silent) on an unknown
/// slot — surfaced to the author on `.glz` (hot-)reload.
fn validate_slots(slots: &CompiledSlots, known: &[&str], component: &str) -> Result<(), String> {
    for name in slots.slot_names() {
        if !known.contains(&name) {
            return Err(format!(
                "{component} has no slot `{name}` (known: {})",
                known.join(", ")
            ));
        }
    }
    Ok(())
}

/// Convert a resolved slotted Glaze style into a typed [`BarStyle`]. Unknown
/// `part {}` names are rejected. The component's root box (`base`) is not used
/// by `Bar` today — its geometry comes from the element's `width`/`height`.
pub fn to_bar_style(slots: &CompiledSlots) -> Result<BarStyle, String> {
    validate_slots(slots, BarStyle::SLOTS, "bar")?;
    Ok(BarStyle {
        track: slots.slot("track").map(to_style),
        fill: slots.slot("fill").map(to_style),
    })
}

/// Convert a resolved slotted Glaze style into a typed [`ToggleStyle`]. The
/// `track`/`knob` plans should be resolved with the right discrete state in
/// `slots` (i.e. the widget passes `["checked"]` when the toggle is on, so a
/// `track { fill muted; :checked { fill accent } }` lands the on/off color).
pub fn to_toggle_style(slots: &CompiledSlots) -> Result<ToggleStyle, String> {
    validate_slots(slots, ToggleStyle::SLOTS, "toggle")?;
    Ok(ToggleStyle {
        track: slots.slot("track").map(to_style),
        knob: slots.slot("knob").map(to_style),
    })
}

/// Convert a resolved slotted Glaze style into a typed [`StepperStyle`]
/// (`field` / `button`). Unknown slots are rejected.
pub fn to_stepper_style(slots: &CompiledSlots) -> Result<StepperStyle, String> {
    validate_slots(slots, StepperStyle::SLOTS, "stepper")?;
    Ok(StepperStyle {
        field: slots.slot("field").map(to_style),
        button: slots.slot("button").map(to_style),
    })
}

/// Convert a resolved slotted Glaze style into a typed [`RadioGroupStyle`]
/// (`ring` / `dot`). Unknown slots are rejected.
pub fn to_radio_style(slots: &CompiledSlots) -> Result<RadioGroupStyle, String> {
    validate_slots(slots, RadioGroupStyle::SLOTS, "radiogroup")?;
    Ok(RadioGroupStyle {
        ring: slots.slot("ring").map(to_style),
        dot: slots.slot("dot").map(to_style),
    })
}

/// Convert a resolved slotted Glaze style into a typed [`CheckboxStyle`]
/// (`box` / `check`). Unknown slots are rejected.
pub fn to_checkbox_style(slots: &CompiledSlots) -> Result<CheckboxStyle, String> {
    validate_slots(slots, CheckboxStyle::SLOTS, "checkbox")?;
    Ok(CheckboxStyle {
        square: slots.slot("box").map(to_style),
        check: slots.slot("check").map(to_style),
    })
}

/// Convert a resolved slotted Glaze style into a typed [`SliderStyle`]
/// (track / range / thumb). Unknown slots are rejected.
pub fn to_slider_style(slots: &CompiledSlots) -> Result<SliderStyle, String> {
    validate_slots(slots, SliderStyle::SLOTS, "slider")?;
    Ok(SliderStyle {
        track: slots.slot("track").map(to_style),
        range: slots.slot("range").map(to_style),
        thumb: slots.slot("thumb").map(to_style),
    })
}

/// Convert a resolved slotted Glaze style into a typed [`TableStyle`] (panel /
/// header / zebra background surfaces). Unknown slots are rejected.
pub fn to_table_style(slots: &CompiledSlots) -> Result<TableStyle, String> {
    validate_slots(slots, TableStyle::SLOTS, "table")?;
    Ok(TableStyle {
        panel: slots.slot("panel").map(to_style),
        header: slots.slot("header").map(to_style),
        zebra: slots.slot("zebra").map(to_style),
    })
}

/// Convert a resolved slotted Glaze style into a typed [`TooltipStyle`]
/// (`bubble`). Unknown slots are rejected.
pub fn to_tooltip_style(slots: &CompiledSlots) -> Result<TooltipStyle, String> {
    validate_slots(slots, TooltipStyle::SLOTS, "tooltip")?;
    Ok(TooltipStyle {
        bubble: slots.slot("bubble").map(to_style),
    })
}

/// Resolve a `select` slot style into a typed [`SelectStyle`], precomputing the
/// resting and `:selected` `item` plans (the chosen option's row). `trigger`
/// renders in-pane; `menu`/`item` render on the floating overlay layer.
pub fn resolve_select_style(prog: &Program, style: &str) -> Result<SelectStyle, String> {
    let empty: HashMap<String, String> = HashMap::new();
    let base = prog
        .resolve_slots(style, &empty, &[])
        .map_err(|e| e.to_string())?;
    validate_slots(&base, SelectStyle::SLOTS, "select")?;
    let selected = prog
        .resolve_slots(style, &empty, &["selected"])
        .map_err(|e| e.to_string())?;
    Ok(SelectStyle {
        trigger: base.slot("trigger").map(to_style),
        menu: base.slot("menu").map(to_style),
        item: base.slot("item").map(to_style),
        item_selected: selected.slot("item").map(to_style),
    })
}

/// Resolve a `tabs` slot style into a typed [`TabsStyle`], precomputing both the
/// resting and `:selected` `tab` plans (one resolve per state — the discrete
/// state model). The renderer swaps `tab_selected` in for the active tab.
pub fn resolve_tabs_style(prog: &Program, style: &str) -> Result<TabsStyle, String> {
    let empty: HashMap<String, String> = HashMap::new();
    let base = prog
        .resolve_slots(style, &empty, &[])
        .map_err(|e| e.to_string())?;
    validate_slots(&base, TabsStyle::SLOTS, "tabs")?;
    let selected = prog
        .resolve_slots(style, &empty, &["selected"])
        .map_err(|e| e.to_string())?;
    Ok(TabsStyle {
        strip: base.slot("strip").map(to_style),
        tab: base.slot("tab").map(to_style),
        tab_selected: selected.slot("tab").map(to_style),
        indicator: base.slot("indicator").map(to_style),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use glaze::{CompiledShader, CompiledStyle};

    #[test]
    fn to_bar_style_maps_known_slots() {
        let prog = glaze::parse(
            r#"
            token a = #112233
            token b = #445566
            style bar {
                track { fill a }
                fill  { fill b }
            }
        "#,
        )
        .unwrap();
        let slots = prog.resolve_slots("bar", &Default::default(), &[]).unwrap();
        let bs = to_bar_style(&slots).expect("known slots should map");
        assert!(bs.track.is_some());
        assert!(bs.fill.is_some());
        // the fill slot carried its layer through as a glaze_layer plan
        assert!(!bs.fill.unwrap().glaze_layers.is_empty());
    }

    #[test]
    fn toggle_checked_state_changes_track_plan() {
        let prog = glaze::parse(
            r#"
            token off = #333333
            token on  = #00ccaa
            style toggle {
                track {
                    fill off
                    :checked { fill on }
                }
                knob {
                    fill #ffffff
                }
            }
        "#,
        )
        .unwrap();
        let off = to_toggle_style(&prog.resolve_slots("toggle", &Default::default(), &[]).unwrap())
            .unwrap();
        let on = to_toggle_style(
            &prog
                .resolve_slots("toggle", &Default::default(), &["checked"])
                .unwrap(),
        )
        .unwrap();
        // the :checked overlay produces a different track plan (different last fill)
        let last_fill = |s: &Option<Style>| {
            s.as_ref()
                .unwrap()
                .glaze_layers
                .iter()
                .rev()
                .find_map(|l| match l {
                    GlazeLayer::Fill { color } => Some(color.clone()),
                    _ => None,
                })
                .unwrap()
        };
        assert_ne!(last_fill(&off.track), last_fill(&on.track));
    }

    #[test]
    fn resolve_tabs_style_precomputes_selected_plan() {
        let prog = glaze::parse(
            r#"
            token a = #222222
            token b = #00ccaa
            style tabbar {
                strip {
                    fill a
                }
                tab {
                    radius 6px
                    :selected { fill b }
                }
                indicator {
                    height 3px
                }
            }
        "#,
        )
        .unwrap();
        let ts = resolve_tabs_style(&prog, "tabbar").unwrap();
        assert!(ts.strip.is_some());
        assert!(ts.tab.is_some());
        // the selected plan picked up the :selected fill; the resting one didn't
        assert!(ts.tab.as_ref().unwrap().glaze_layers.is_empty());
        assert!(!ts.tab_selected.as_ref().unwrap().glaze_layers.is_empty());
        assert!(ts.indicator.is_some());
    }

    #[test]
    fn resolve_select_style_precomputes_selected_item() {
        let prog = glaze::parse(
            r#"
            token a = #222222
            token b = #00ccaa
            style picker {
                trigger {
                    fill a
                }
                menu {
                    fill a
                }
                item {
                    radius 5px
                    :selected { fill b }
                }
            }
        "#,
        )
        .unwrap();
        let ss = resolve_select_style(&prog, "picker").unwrap();
        assert!(ss.trigger.is_some() && ss.menu.is_some() && ss.item.is_some());
        // resting item has no fill layer; the :selected variant does
        assert!(ss.item.as_ref().unwrap().glaze_layers.is_empty());
        assert!(!ss.item_selected.as_ref().unwrap().glaze_layers.is_empty());

        let bad = glaze::parse(
            r#"
            token a = #222222
            style picker { panl { fill a } }
        "#,
        )
        .unwrap();
        assert!(resolve_select_style(&bad, "picker")
            .unwrap_err()
            .contains("panl"));
    }

    #[test]
    fn to_stepper_style_maps_and_validates() {
        let prog = glaze::parse(
            r#"
            token a = #112233
            style st {
                field {
                    fill a
                }
                button {
                    fill a
                }
            }
        "#,
        )
        .unwrap();
        let ss =
            to_stepper_style(&prog.resolve_slots("st", &Default::default(), &[]).unwrap()).unwrap();
        assert!(ss.field.is_some() && ss.button.is_some());

        let bad = glaze::parse(
            r#"
            token a = #112233
            style st { feild { fill a } }
        "#,
        )
        .unwrap();
        let err =
            to_stepper_style(&bad.resolve_slots("st", &Default::default(), &[]).unwrap()).unwrap_err();
        assert!(err.contains("feild"));
    }

    #[test]
    fn to_radio_style_maps_and_validates() {
        let prog = glaze::parse(
            r#"
            token a = #112233
            style rg {
                ring {
                    fill a
                }
                dot {
                    fill a
                }
            }
        "#,
        )
        .unwrap();
        let rs =
            to_radio_style(&prog.resolve_slots("rg", &Default::default(), &[]).unwrap()).unwrap();
        assert!(rs.ring.is_some() && rs.dot.is_some());

        let bad = glaze::parse(
            r#"
            token a = #112233
            style rg { circle { fill a } }
        "#,
        )
        .unwrap();
        let err =
            to_radio_style(&bad.resolve_slots("rg", &Default::default(), &[]).unwrap()).unwrap_err();
        assert!(err.contains("circle"));
    }

    #[test]
    fn to_checkbox_style_maps_box_slot_to_square_field() {
        let prog = glaze::parse(
            r#"
            token a = #112233
            token b = #445566
            style cb {
                box {
                    fill a
                }
                check {
                    fill b
                }
            }
        "#,
        )
        .unwrap();
        let cs =
            to_checkbox_style(&prog.resolve_slots("cb", &Default::default(), &[]).unwrap()).unwrap();
        assert!(cs.square.is_some(), "the `box` slot maps to the `square` field");
        assert!(cs.check.is_some());

        let bad = glaze::parse(
            r#"
            token a = #112233
            style cb { tick { fill a } }
        "#,
        )
        .unwrap();
        let err =
            to_checkbox_style(&bad.resolve_slots("cb", &Default::default(), &[]).unwrap()).unwrap_err();
        assert!(err.contains("tick"));
    }

    #[test]
    fn to_slider_style_maps_and_validates() {
        let prog = glaze::parse(
            r#"
            token a = #112233
            token b = #445566
            style sl {
                track {
                    fill a
                }
                range {
                    fill b
                }
                thumb {
                    fill #ffffff
                }
            }
        "#,
        )
        .unwrap();
        let ss =
            to_slider_style(&prog.resolve_slots("sl", &Default::default(), &[]).unwrap()).unwrap();
        assert!(ss.track.is_some() && ss.range.is_some() && ss.thumb.is_some());

        let bad = glaze::parse(
            r#"
            token a = #112233
            style sl { thmub { fill a } }
        "#,
        )
        .unwrap();
        let err =
            to_slider_style(&bad.resolve_slots("sl", &Default::default(), &[]).unwrap()).unwrap_err();
        assert!(err.contains("thmub"));
    }

    #[test]
    fn to_table_style_maps_and_validates() {
        let prog = glaze::parse(
            r#"
            token p = #181818
            token h = #222222
            style tbl {
                panel {
                    fill p
                }
                header {
                    fill h
                }
            }
        "#,
        )
        .unwrap();
        let ts =
            to_table_style(&prog.resolve_slots("tbl", &Default::default(), &[]).unwrap()).unwrap();
        assert!(ts.panel.is_some());
        assert!(ts.header.is_some());
        assert!(ts.zebra.is_none());

        let bad = glaze::parse(
            r#"
            token p = #181818
            style tbl { footre { fill p } }
        "#,
        )
        .unwrap();
        let err =
            to_table_style(&bad.resolve_slots("tbl", &Default::default(), &[]).unwrap()).unwrap_err();
        assert!(err.contains("footre"), "error should name the bad slot: {err}");
    }

    #[test]
    fn to_bar_style_rejects_unknown_slot() {
        let prog = glaze::parse(
            r#"
            token a = #112233
            style bar {
                trakc { fill a }   // typo
            }
        "#,
        )
        .unwrap();
        let slots = prog.resolve_slots("bar", &Default::default(), &[]).unwrap();
        let err = to_bar_style(&slots).unwrap_err();
        assert!(err.contains("trakc"), "error should name the bad slot: {err}");
    }

    #[test]
    fn preserves_ordered_layers() {
        let mut compiled = CompiledStyle::default();
        compiled.layers = vec![
            Layer::Fill(Rgba {
                r: 0.1,
                g: 0.2,
                b: 0.3,
                a: 1.0,
            }),
            Layer::Shader(CompiledShader {
                overlay: false,
                wgsl_body: "    return vec4<f32>(1.0);".into(),
                used: vec![],
            }),
            Layer::Border {
                color: Rgba {
                    r: 0.4,
                    g: 0.5,
                    b: 0.6,
                    a: 1.0,
                },
                width: 2.0,
                sides: glaze::Sides::ALL,
            },
            Layer::Shader(CompiledShader {
                overlay: true,
                wgsl_body: "    return vec4<f32>(0.5);".into(),
                used: vec![],
            }),
        ];

        let style = to_style(&compiled);
        assert_eq!(style.glaze_layers.len(), 4);
        assert!(matches!(style.glaze_layers[0], GlazeLayer::Fill { .. }));
        assert!(matches!(
            style.glaze_layers[1],
            GlazeLayer::Shader { overlay: false, .. }
        ));
        assert!(matches!(style.glaze_layers[2], GlazeLayer::Border { .. }));
        assert!(matches!(
            style.glaze_layers[3],
            GlazeLayer::Shader { overlay: true, .. }
        ));
        assert!(style.background.is_none());
        assert!(style.shader.is_none());
    }
}
