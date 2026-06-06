//! glaze_demo — a widget pane whose styling comes entirely from a Glaze
//! stylesheet compiled at runtime.
//!
//! This is the first, deliberately non-invasive integration of the `glaze`
//! crate: nothing in the renderer or protocol changes. Glaze just becomes
//! *another way to produce the existing `Style`* — every styled box below is
//! `compiled_to_style(program.resolve(name, variant, states))`. The `.glz`
//! source is compiled once on startup; tokens, variants and `:state` overlays
//! all resolve through the real compiler.

use std::collections::HashMap;
use std::io::{self, BufRead, Write};

use glaze::{CompiledStyle, Dim, Layer, Program, Rgba, parse};
use widget_bevy::protocol::{
    Align, Border, Edges, Element, HostEvent, Shadow, ShaderSpec, Style, Weight, WidgetMsg,
};

// The stylesheet. Authored in Glaze; compiled at runtime.
const SHEET: &str = r#"
    // primitives
    token gold.500   = oklch(0.74 0.12 85)
    token slate.900  = oklch(0.17 0.012 250)
    token slate.800  = oklch(0.24 0.012 250)
    token slate.700  = oklch(0.30 0.012 250)
    token red.500    = oklch(0.62 0.17 25)
    token green.500  = oklch(0.70 0.13 150)

    // semantic aliases
    token accent.solid   = gold.500
    token accent.hover   = mix(gold.500, slate.700, 0.25)
    token accent.press   = mix(gold.500, slate.900, 0.35)
    token danger.solid   = red.500
    token surface.raised = slate.800
    token surface.sunk   = slate.900
    token border.subtle  = oklch(0.45 0.01 250 / 0.6)
    token focus.ring     = oklch(0.80 0.13 85)
    token text.on_accent = slate.900
    token radius.md      = 10px

    style card {
        fill   surface.raised
        radius radius.md
        border border.subtle 1px
        pad    16px
    }

    style button(intent) {
        fill   intent == danger ? danger.solid : accent.solid
        radius radius.md
        pad    10px 18px

        :hover { fill intent == danger ? danger.solid : accent.hover }
        :press { fill accent.press }
        :focus { border focus.ring 2px }
    }

    // A shader layer: compiled to WGSL by `glaze`, run on the GPU by the host.
    style hero {
        fill   surface.sunk
        radius radius.md
        width  100%
        overlay shader {
            let d    = length(uv - vec2(0.5, 0.5))   // uv → per-fragment
            let ring = smoothstep(0.6, 0.0, d)        // steady radial falloff
            emit ring * accent.solid                  // gold token → folded constant
        }
    }
"#;

// ---- the adapter: glaze CompiledStyle -> the existing protocol Style --------
// This is the *entire* integration surface. No renderer changes.

fn lin_to_srgb(c: f32) -> f32 {
    if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.clamp(0.0, 1.0).powf(1.0 / 2.4) - 0.055
    }
}

/// Glaze stores linear-rgb; the renderer parses sRGB hex back to linear, so we
/// convert linear -> sRGB 8-bit -> `#rrggbb[aa]` (round-trips faithfully).
fn hex(c: Rgba) -> String {
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

fn compiled_to_style(c: &CompiledStyle) -> Style {
    let mut s = Style::default();
    let p = c.box_.padding;
    s.padding = Some(Edges { top: p[0], right: p[1], bottom: p[2], left: p[3] });
    if c.box_.radius > 0.0 {
        s.radius = Some(format!("{}", c.box_.radius));
    }
    if let Some(d) = c.box_.width {
        s.width = Some(dim_str(d));
    }
    if let Some(d) = c.box_.height {
        s.height = Some(dim_str(d));
    }
    for layer in &c.layers {
        match layer {
            // last fill wins (a :hover overlay replaces the base fill)
            Layer::Fill(rgba) => s.background = Some(hex(*rgba)),
            Layer::Border { color, width } => {
                s.border = Some(Border { color: hex(*color), width: *width });
            }
            Layer::Shadow { color, blur, offset_y } => {
                s.shadow = Some(Shadow {
                    token: None,
                    color: Some(hex(*color)),
                    blur: Some(*blur),
                    offset_y: Some(*offset_y),
                });
            }
            // Stage 3: a compiled shader layer → carried over the protocol as
            // its WGSL body; the host runs it on the element's box.
            Layer::Shader(cs) => {
                s.shader = Some(ShaderSpec { body: cs.wgsl_body.clone(), overlay: cs.overlay });
            }
        }
    }
    s
}

// ---- helpers ----------------------------------------------------------------

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

fn vstack(gap: f32, children: Vec<Element>) -> Element {
    Element::Vstack { gap, pad: 0.0, children, style: None }
}

fn hstack(gap: f32, children: Vec<Element>) -> Element {
    Element::Hstack { gap, pad: 0.0, align: Align::Start, children, style: None }
}

/// A frame styled entirely by Glaze. Surfaces resolve errors *loudly* as a
/// red error tile rather than silently dropping the box (house rule).
fn glz(prog: &Program, name: &str, variant: &[(&str, &str)], states: &[&str], gap: f32, children: Vec<Element>) -> Element {
    let v: HashMap<String, String> =
        variant.iter().map(|(k, x)| (k.to_string(), x.to_string())).collect();
    match prog.resolve(name, &v, states) {
        Ok(compiled) => Element::Frame { gap, pad: 0.0, children, style: Some(compiled_to_style(&compiled)) },
        Err(e) => Element::Frame {
            gap: 4.0,
            pad: 0.0,
            children: vec![text(&format!("glaze: {e}"), "#ff6b5a", 11.0, true)],
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

/// A swatch showing a single resolved token color (proves token resolution).
fn swatch(prog: &Program, token: &str) -> Element {
    let color = match prog.eval_token(token) {
        Ok(glaze::Value::Color(c)) => hex(c),
        _ => "#000000".into(),
    };
    vstack(
        4.0,
        vec![
            Element::Frame {
                gap: 0.0,
                pad: 0.0,
                children: vec![Element::Spacer { size: 30.0 }],
                style: Some(Style {
                    background: Some(color),
                    radius: Some("6".into()),
                    padding: Some(Edges::symmetric(22.0, 0.0)),
                    ..Default::default()
                }),
            },
            text(token, "fg_muted", 9.0, false),
        ],
    )
}

fn button(prog: &Program, label: &str, intent: &str, states: &[&str]) -> Element {
    glz(prog, "button", &[("intent", intent)], states, 0.0, vec![text(label, "#1b1f17", 12.0, true)])
}

fn build_frame(prog: &Program) -> Element {
    let header = vstack(
        4.0,
        vec![
            text("Styled by Glaze", "accent", 30.0, true),
            text(
                "every box below — colors · radius · border · padding · variants · :states — \
                 is compiled from a .glz stylesheet at runtime, then adapted into the existing Style.",
                "fg_muted",
                12.0,
                false,
            ),
        ],
    );

    let swatches = vstack(
        6.0,
        vec![
            text("TOKENS  (resolved through alias chains + mix())", "fg_muted", 9.0, true),
            hstack(
                8.0,
                vec![
                    swatch(prog, "accent.solid"),
                    swatch(prog, "accent.hover"),
                    swatch(prog, "accent.press"),
                    swatch(prog, "danger.solid"),
                    swatch(prog, "surface.raised"),
                    swatch(prog, "border.subtle"),
                ],
            ),
        ],
    );

    // cards: one Glaze style `card`, reused; change the .glz once → all restyle.
    let card = |title: &str, body: &str| {
        glz(
            prog,
            "card",
            &[],
            &[],
            6.0,
            vec![text(title, "fg", 14.0, true), text(body, "fg_muted", 11.0, false)],
        )
    };
    let cards = vstack(
        6.0,
        vec![
            text("STYLE `card`  (reused — one definition, many instances)", "fg_muted", 9.0, true),
            hstack(
                12.0,
                vec![
                    card("Surface", "surface.raised fill"),
                    card("Border", "border.subtle 1px"),
                    card("Radius", "radius.md corners"),
                ],
            ),
        ],
    );

    // buttons: one variant style resolved across variant axes + discrete states.
    let buttons = vstack(
        6.0,
        vec![
            text("STYLE `button(intent)`  (variant axis + :hover / :press / :focus plans)", "fg_muted", 9.0, true),
            hstack(
                10.0,
                vec![
                    button(prog, "Primary", "primary", &[]),
                    button(prog, "Hover", "primary", &["hover"]),
                    button(prog, "Press", "primary", &["press"]),
                    button(prog, "Focus", "primary", &["focus"]),
                    button(prog, "Danger", "danger", &[]),
                ],
            ),
        ],
    );

    // A live Glaze shader layer running on the GPU, inside a real pane.
    let hero = vstack(
        6.0,
        vec![
            text("STYLE `hero`  (overlay shader { } → compiled to WGSL, runs on the GPU)", "fg_muted", 9.0, true),
            glz(
                prog,
                "hero",
                &[],
                &[],
                8.0,
                vec![
                    text("overlay shader { }", "fg", 13.0, true),
                    text(
                        "steady radial glow · `uv` per-fragment · gold token folded to a constant",
                        "fg_muted",
                        10.0,
                        false,
                    ),
                    Element::Spacer { size: 64.0 },
                ],
            ),
        ],
    );

    Element::Vstack {
        gap: 22.0,
        pad: 4.0,
        children: vec![header, swatches, cards, buttons, hero],
        style: None,
    }
}

fn main() {
    let prog = match parse(SHEET) {
        Ok(p) => p,
        Err(e) => {
            // A stylesheet that won't compile is a hard, visible failure.
            eprintln!("glaze_demo: stylesheet failed to compile: {e}");
            std::process::exit(1);
        }
    };

    let stdout = io::stdout();
    let mut out = stdout.lock();
    let _ = writeln!(
        out,
        "{}",
        serde_json::to_string(&WidgetMsg::Title { value: "Glaze Demo".into() }).unwrap()
    );
    emit(&mut out, &prog);

    for line in io::stdin().lock().lines() {
        let Ok(line) = line else { break };
        let Ok(evt) = serde_json::from_str::<HostEvent>(&line) else { continue };
        match evt {
            HostEvent::Close => return,
            _ => emit(&mut out, &prog),
        }
    }
}

fn emit<W: Write>(out: &mut W, prog: &Program) {
    let msg = WidgetMsg::Frame { root: build_frame(prog) };
    if let Ok(s) = serde_json::to_string(&msg) {
        let _ = writeln!(out, "{s}");
        let _ = out.flush();
    }
}
