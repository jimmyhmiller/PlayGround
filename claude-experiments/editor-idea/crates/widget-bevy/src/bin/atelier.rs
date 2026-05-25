//! Atelier — a showcase widget that reproduces the dark-theme design
//! mockup using every new Element variant, design token, and Style
//! override. Verifies the style system is expressive enough end-to-end.
//!
//! Run via the host (e.g. set a widget pane's command to
//! `target/debug/atelier`). The widget is interactive — Tabs, Toggle,
//! and Input all round-trip through the host.

use std::io::{self, BufRead, Write};

use widget_bevy::protocol::{
    Align, Border, ButtonKind, Element, HostEvent, Shadow, Style, TabItem, Weight, WidgetMsg,
};

// ---- helpers ----------------------------------------------------------

fn text(value: &str) -> Element {
    Element::Text {
        value: value.into(),
        color: None,
        size: None,
        weight: None,
        family: None,
    }
}

fn text_sized(value: &str, size: f32, color: Option<&str>) -> Element {
    Element::Text {
        value: value.into(),
        color: color.map(str::to_string),
        size: Some(size),
        weight: None,
        family: None,
    }
}

fn heading(value: &str) -> Element {
    Element::Text {
        value: value.into(),
        color: Some("fg".into()),
        size: Some(20.0),
        weight: Some(Weight::Bold),
        family: Some("font_family_heading".into()),
    }
}

fn label(value: &str) -> Element {
    Element::Text {
        value: value.into(),
        color: Some("fg_muted".into()),
        size: Some(11.0),
        weight: None,
        family: Some("font_family_body".into()),
    }
}

fn divider() -> Element {
    Element::Divider
}

fn spacer(n: f32) -> Element {
    Element::Spacer { size: n }
}

fn vstack(gap: f32, pad: f32, children: Vec<Element>) -> Element {
    Element::Vstack { gap, pad, children, style: None }
}

fn hstack(gap: f32, children: Vec<Element>) -> Element {
    Element::Hstack {
        gap,
        pad: 0.0,
        align: Align::Center,
        children,
        style: None,
    }
}

fn card(children: Vec<Element>, surface: &str) -> Element {
    Element::Frame {
        gap: 8.0,
        pad: 12.0,
        children,
        style: Some(Style {
            background: Some(surface.into()),
            radius: Some("radius_md".into()),
            border: Some(Border {
                color: "chrome_divider".into(),
                width: 1.0,
            }),
            shadow: Some(Shadow {
                token: Some("shadow_sm".into()),
                ..Default::default()
            }),
            ..Default::default()
        }),
    }
}

fn swatch_row() -> Element {
    let colors = [
        "#c9a96a", "#5f7e9e", "#7a8e9c", "#cdb89a", "#b96a4a", "#3e6e6c", "#2a3346", "#1a1f2c",
        "#0f1218",
    ];
    let kids: Vec<Element> = colors
        .iter()
        .map(|c| Element::Swatch {
            color: (*c).into(),
            size: 28.0,
            id: None,
        })
        .collect();
    hstack(6.0, kids)
}

fn radius_row() -> Element {
    let radii = [("xs", "radius_xs"), ("sm", "radius_sm"), ("md", "radius_md"), ("lg", "radius_lg"), ("pill", "radius_pill")];
    let kids: Vec<Element> = radii
        .iter()
        .map(|(_name, token)| {
            Element::Frame {
                gap: 0.0,
                pad: 0.0,
                children: vec![],
                style: Some(Style {
                    background: Some("surface_3".into()),
                    radius: Some((*token).into()),
                    ..Default::default()
                }),
            }
        })
        .collect();
    // Wrap each in an Hstack with a fixed-size Frame inside doesn't work
    // since Frame measures from children — give each its own swatch
    // sized via Spacer trick: stack the frame over a sized spacer.
    let sized: Vec<Element> = kids
        .into_iter()
        .zip(radii.iter())
        .map(|(frame, (_n, _t))| {
            // Use a Frame whose content is a 40×40 spacer.
            let Element::Frame { style, .. } = frame else {
                unreachable!()
            };
            Element::Frame {
                gap: 0.0,
                pad: 0.0,
                children: vec![Element::Spacer { size: 40.0 }],
                style,
            }
        })
        .collect();
    hstack(10.0, sized)
}

fn shadow_row() -> Element {
    let shadows = ["shadow_sm", "shadow_md", "shadow_lg"];
    let kids: Vec<Element> = shadows
        .iter()
        .map(|token| Element::Frame {
            gap: 0.0,
            pad: 0.0,
            children: vec![Element::Spacer { size: 48.0 }],
            style: Some(Style {
                background: Some("surface_3".into()),
                radius: Some("radius_md".into()),
                shadow: Some(Shadow {
                    token: Some((*token).into()),
                    ..Default::default()
                }),
                ..Default::default()
            }),
        })
        .collect();
    hstack(20.0, kids)
}

fn typography_row() -> Element {
    let tiles = [
        ("Aa", "font_family_heading", 28.0, "Display"),
        ("Aa", "font_family_body", 18.0, "Body"),
        ("Aa", "font_family_mono", 14.0, "Mono"),
    ];
    let kids: Vec<Element> = tiles
        .iter()
        .map(|(g, family, size, name)| Element::Frame {
            gap: 4.0,
            pad: 10.0,
            children: vec![
                Element::Text {
                    value: (*g).into(),
                    color: Some("accent".into()),
                    size: Some(*size),
                    weight: Some(Weight::Bold),
                    family: Some((*family).into()),
                },
                label(name),
            ],
            style: Some(Style {
                background: Some("surface_2".into()),
                radius: Some("radius_sm".into()),
                border: Some(Border {
                    color: "chrome_divider".into(),
                    width: 1.0,
                }),
                ..Default::default()
            }),
        })
        .collect();
    hstack(8.0, kids)
}

fn button_row() -> Element {
    hstack(
        8.0,
        vec![
            Element::Button {
                id: "filled".into(),
                label: "Filled".into(),
                kind: ButtonKind::Filled,
                style: None,
            },
            Element::Button {
                id: "outline".into(),
                label: "Outline".into(),
                kind: ButtonKind::Outline,
                style: None,
            },
            Element::Button {
                id: "ghost".into(),
                label: "Ghost".into(),
                kind: ButtonKind::Ghost,
                style: None,
            },
        ],
    )
}

fn dos_donts() -> Element {
    let ok: Vec<&str> = vec![
        "Use the surface ladder",
        "Pair serif heading + sans body",
        "Lift cards with shadow_sm",
    ];
    let bad: Vec<&str> = vec![
        "Mix four accent hues",
        "Stack three nested shadows",
        "Trust radius_pill on tiny chips",
    ];
    let col = |title: &str, color: &str, items: &[&str]| -> Element {
        let mut kids: Vec<Element> = vec![Element::Text {
            value: title.into(),
            color: Some(color.into()),
            size: Some(13.0),
            weight: Some(Weight::Bold),
            family: Some("font_family_body".into()),
        }];
        for it in items {
            kids.push(hstack(
                6.0,
                vec![
                    Element::Text {
                        value: if color.contains("success") || color.starts_with("#5") {
                            "✓".into()
                        } else {
                            "✗".into()
                        },
                        color: Some(color.into()),
                        size: Some(12.0),
                        weight: Some(Weight::Bold),
                        family: None,
                    },
                    text(it),
                ],
            ));
        }
        vstack(4.0, 0.0, kids)
    };
    hstack(
        24.0,
        vec![
            col("Do", "status_success", &ok),
            col("Don't", "status_failed", &bad),
        ],
    )
}

fn build_frame(active_tab: &str, dark_mode: bool, search_value: &str) -> Element {
    let palette_card = card(
        vec![heading("Color palette"), swatch_row()],
        "surface_2",
    );
    let typography_card = card(
        vec![heading("Typography"), typography_row()],
        "surface_2",
    );
    let radius_card = card(
        vec![heading("Radii"), radius_row()],
        "surface_2",
    );
    let shadow_card = card(
        vec![heading("Shadows"), shadow_row()],
        "surface_2",
    );
    let components_card = card(
        vec![
            heading("Components"),
            button_row(),
            Element::Tabs {
                id: "demo".into(),
                items: vec![
                    TabItem { id: "overview".into(), label: "Overview".into() },
                    TabItem { id: "tokens".into(), label: "Tokens".into() },
                    TabItem { id: "components".into(), label: "Components".into() },
                ],
                selected: active_tab.into(),
                style: None,
            },
            hstack(
                12.0,
                vec![
                    Element::Toggle {
                        id: "darkmode".into(),
                        label: "Dark mode".into(),
                        checked: dark_mode,
                        style: None,
                    },
                    Element::Bar {
                        value: 0.62,
                        max: 1.0,
                        color: Some("accent".into()),
                        track: Some("surface_3".into()),
                        width: 160.0,
                        height: 8.0,
                    },
                ],
            ),
            Element::Input {
                id: "search".into(),
                value: search_value.into(),
                placeholder: "Search components…".into(),
                focused: false,
                width: 280.0,
                style: None,
            },
        ],
        "surface_2",
    );
    let list_card = card(
        vec![
            heading("List"),
            Element::ListItem {
                id: "row-1".into(),
                children: vec![text("Primary buttons")],
                gap: 0.0,
                pad: 8.0,
                selected: true,
                style: None,
            },
            Element::ListItem {
                id: "row-2".into(),
                children: vec![text("Secondary toggles")],
                gap: 0.0,
                pad: 8.0,
                selected: false,
                style: None,
            },
            Element::ListItem {
                id: "row-3".into(),
                children: vec![text("Input fields")],
                gap: 0.0,
                pad: 8.0,
                selected: false,
                style: None,
            },
        ],
        "surface_2",
    );
    let dos_card = card(vec![heading("Do & Don't"), dos_donts()], "surface_2");

    vstack(
        16.0,
        16.0,
        vec![
            // Title strip
            hstack(
                12.0,
                vec![
                    Element::Text {
                        value: "Atelier".into(),
                        color: Some("fg".into()),
                        size: Some(28.0),
                        weight: Some(Weight::Bold),
                        family: Some("font_family_heading".into()),
                    },
                    text_sized("Elegant & tactile", 14.0, Some("fg_muted")),
                ],
            ),
            divider(),
            // Top row
            hstack(
                16.0,
                vec![palette_card, typography_card, radius_card, shadow_card],
            ),
            spacer(4.0),
            // Middle
            components_card,
            spacer(4.0),
            hstack(16.0, vec![list_card, dos_card]),
        ],
    )
}

// ---- main loop --------------------------------------------------------

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    let _ = writeln!(
        out,
        "{}",
        serde_json::to_string(&WidgetMsg::Title {
            value: "Atelier".into()
        })
        .unwrap()
    );
    let mut active_tab = String::from("overview");
    let mut dark_mode = true;
    let mut search_value = String::new();
    emit_frame(&mut out, &active_tab, dark_mode, &search_value);

    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let Ok(evt) = serde_json::from_str::<HostEvent>(&line) else {
            continue;
        };
        match evt {
            HostEvent::Close => return,
            HostEvent::TabSelect { id: _, tab } => {
                active_tab = tab;
            }
            HostEvent::Toggle { id, checked } if id == "darkmode" => {
                dark_mode = checked;
            }
            HostEvent::InputChange { id, value } if id == "search" => {
                search_value = value;
            }
            HostEvent::Click { .. }
            | HostEvent::Init { .. }
            | HostEvent::Resize { .. }
            | HostEvent::Refresh => {}
            _ => continue,
        }
        emit_frame(&mut out, &active_tab, dark_mode, &search_value);
    }
}

fn emit_frame<W: Write>(out: &mut W, tab: &str, dark: bool, search: &str) {
    let msg = WidgetMsg::Frame {
        root: build_frame(tab, dark, search),
    };
    if let Ok(s) = serde_json::to_string(&msg) {
        let _ = writeln!(out, "{}", s);
        let _ = out.flush();
    }
}
