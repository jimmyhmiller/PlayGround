//! glaze_ui — a showcase of real UI components (cards, stat tiles, a pricing
//! card, a glassy profile card, badges, buttons, toggles) laid out with flex and
//! styled entirely by Glaze: gradient buttons/banners and a glowing avatar are
//! `overlay shader {}` layers; everything else is tokens + box styling.

use std::collections::HashMap;
use std::io::{self, BufRead, Write};

use glaze::{Program, parse};
use widget_bevy::glaze_style::{hex, to_style};
use widget_bevy::protocol::{
    Align, Border, ButtonKind, Edges, Element, HostEvent, Style, Weight, WidgetMsg,
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

fn build_ui(prog: &Program, width: f32) -> Element {
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

    Element::Vstack {
        gap: 18.0,
        pad: 4.0,
        children: vec![header, stats, middle, badges],
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
    // Track the pane's content width so we can switch row↔column at a breakpoint.
    let mut width = 700.0_f32;
    emit(&mut out, &prog, width);
    for line in io::stdin().lock().lines() {
        let Ok(line) = line else { break };
        let Ok(evt) = serde_json::from_str::<HostEvent>(&line) else { continue };
        match evt {
            HostEvent::Close => return,
            HostEvent::Init { width: w, .. } => width = w,
            HostEvent::Resize { width: w, .. } => width = w,
            _ => {}
        }
        emit(&mut out, &prog, width);
    }
}

fn emit<W: Write>(out: &mut W, prog: &Program, width: f32) {
    if let Ok(s) = serde_json::to_string(&WidgetMsg::Frame { root: build_ui(prog, width) }) {
        let _ = writeln!(out, "{s}");
        let _ = out.flush();
    }
}
