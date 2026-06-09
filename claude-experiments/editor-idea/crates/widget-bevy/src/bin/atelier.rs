//! Atelier — a showcase widget that reproduces the dark Atelier
//! mockup using every new Element variant, design token, and Style
//! override.
//!
//! Run via the host (set a widget pane's command to
//! `target/debug/atelier`). The widget is interactive: Tabs, Toggle,
//! Input all round-trip through the host.

use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use widget_bevy::protocol::{
    Align, Border, ButtonKind, Edges, Element, HostEvent, Shadow, Style, TabItem, Weight, WidgetMsg,
};

// ---- helpers ----------------------------------------------------------

fn text(value: &str) -> Element {
    Element::Text {
        value: value.into(),
        color: Some("fg".into()),
        size: Some(13.0),
        weight: None,
        family: Some("font_family_body".into()),
        selectable: false,
    }
}

fn muted(value: &str) -> Element {
    Element::Text {
        value: value.into(),
        color: Some("fg_muted".into()),
        size: Some(11.0),
        weight: None,
        family: Some("font_family_body".into()),
        selectable: false,
    }
}

fn heading(value: &str) -> Element {
    Element::Text {
        value: value.into(),
        color: Some("fg".into()),
        size: Some(15.0),
        weight: Some(Weight::Bold),
        family: Some("font_family_heading".into()),
        selectable: false,
    }
}

fn eyebrow(value: &str) -> Element {
    // All-caps section label. Spaced via uppercase characters; Inter
    // handles tracking better than mono, but at small sizes both read.
    Element::Text {
        value: value
            .to_uppercase()
            .chars()
            .collect::<Vec<_>>()
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(" "),
        color: Some("accent".into()),
        size: Some(9.0),
        weight: Some(Weight::Bold),
        family: Some("font_family_body".into()),
        selectable: false,
    }
}

fn mono(value: &str, color: &str) -> Element {
    Element::Text {
        value: value.into(),
        color: Some(color.into()),
        size: Some(12.0),
        weight: None,
        family: Some("font_family_mono".into()),
        selectable: false,
    }
}

fn divider() -> Element {
    Element::Divider
}

fn spacer(n: f32) -> Element {
    Element::Spacer { size: n }
}

fn vstack(gap: f32, pad: f32, children: Vec<Element>) -> Element {
    Element::Vstack {
        gap,
        pad,
        children,
        style: None,
    }
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

fn hstack_top(gap: f32, children: Vec<Element>) -> Element {
    Element::Hstack {
        gap,
        pad: 0.0,
        // Cards in a row stretch to the row's height — pairs with
        // each card carrying flex_grow:1 so the row also distributes
        // width evenly. Visually: tidy, equal rectangles, no jagged
        // top/bottom edges where one card's content is shorter.
        align: Align::Stretch,
        children,
        style: None,
    }
}

fn card(eyebrow_text: &str, title: &str, body: Vec<Element>) -> Element {
    let mut kids = vec![eyebrow(eyebrow_text), heading(title), spacer(2.0)];
    kids.extend(body);
    Element::Frame {
        gap: 6.0,
        pad: 16.0,
        children: kids,
        style: Some(Style {
            background: Some("surface_2".into()),
            radius: Some("radius_md".into()),
            border: Some(Border {
                color: "chrome_divider".into(),
                width: 1.0,
            }),
            shadow: Some(Shadow {
                token: Some("shadow_sm".into()),
                ..Default::default()
            }),
            // Equal-distribute cards in a row + stretch to fill the
            // row's height. The result: every card in a row is the
            // same width AND height, the magazine-spread look.
            flex_grow: Some(1.0),
            flex_shrink: Some(1.0),
            min_width: Some("0".into()),
            ..Default::default()
        }),
    }
}

// ---- sections ---------------------------------------------------------

fn palette_card() -> Element {
    let colors: &[(&str, &str)] = &[
        ("#c9a96a", "Gold"),
        ("#5f7e9e", "Mist"),
        ("#7a8e9c", "Stone"),
        ("#cdb89a", "Bone"),
        ("#b96a4a", "Clay"),
        ("#3e6e6c", "Teal"),
        ("#2a3346", "Ink"),
        ("#1a1f2c", "Pitch"),
    ];
    let swatch_with_label = |hex: &str, name: &str| -> Element {
        vstack(
            4.0,
            0.0,
            vec![
                Element::Frame {
                    gap: 0.0,
                    pad: 0.0,
                    children: vec![Element::Spacer { size: 28.0 }],
                    style: Some(Style {
                        background: Some(hex.into()),
                        radius: Some("radius_sm".into()),
                        padding: Some(Edges::symmetric(14.0, 0.0)),
                        ..Default::default()
                    }),
                },
                Element::Text {
                    value: name.into(),
                    color: Some("fg_muted".into()),
                    size: Some(9.0),
                    weight: None,
                    family: Some("font_family_body".into()),
                    selectable: false,
                },
            ],
        )
    };
    let tiles: Vec<Element> = colors
        .iter()
        .map(|(hex, name)| swatch_with_label(hex, name))
        .collect();
    let row1 = hstack(6.0, tiles[..4].to_vec());
    let row2 = hstack(6.0, tiles[4..].to_vec());
    card(
        "color palette",
        "Atelier",
        vec![vstack(8.0, 0.0, vec![row1, row2])],
    )
}

fn typography_card() -> Element {
    let tile = |glyph: &str, family: &str, size: f32, name: &str| -> Element {
        Element::Frame {
            gap: 4.0,
            pad: 10.0,
            children: vec![
                Element::Text {
                    value: glyph.into(),
                    color: Some("accent".into()),
                    size: Some(size),
                    weight: Some(Weight::Bold),
                    family: Some(family.into()),
                    selectable: false,
                },
                Element::Text {
                    value: name.into(),
                    color: Some("fg_muted".into()),
                    size: Some(9.0),
                    weight: None,
                    family: Some("font_family_body".into()),
                    selectable: false,
                },
            ],
            style: Some(Style {
                background: Some("surface_3".into()),
                radius: Some("radius_sm".into()),
                padding: Some(Edges::symmetric(12.0, 8.0)),
                ..Default::default()
            }),
        }
    };
    card(
        "typography",
        "Aa",
        vec![hstack(
            8.0,
            vec![
                tile("Aa", "font_family_heading", 26.0, "DISPLAY"),
                tile("Aa", "font_family_body", 18.0, "BODY"),
                tile("Aa", "font_family_mono", 14.0, "MONO"),
            ],
        )],
    )
}

fn radii_card() -> Element {
    let radii = [
        ("xs", "radius_xs"),
        ("sm", "radius_sm"),
        ("md", "radius_md"),
        ("lg", "radius_lg"),
        ("pill", "radius_pill"),
    ];
    let tile = |name: &str, token: &str| -> Element {
        vstack(
            4.0,
            0.0,
            vec![
                Element::Frame {
                    gap: 0.0,
                    pad: 0.0,
                    children: vec![Element::Spacer { size: 30.0 }],
                    style: Some(Style {
                        background: Some("accent_800".into()),
                        radius: Some(token.into()),
                        padding: Some(Edges::symmetric(16.0, 0.0)),
                        ..Default::default()
                    }),
                },
                Element::Text {
                    value: name.into(),
                    color: Some("fg_muted".into()),
                    size: Some(9.0),
                    weight: None,
                    family: Some("font_family_body".into()),
                    selectable: false,
                },
            ],
        )
    };
    let tiles: Vec<Element> = radii.iter().map(|(n, t)| tile(n, t)).collect();
    card("radii", "Corners", vec![hstack(6.0, tiles)])
}

fn shadows_card() -> Element {
    let shadows = [
        ("sm", "shadow_sm"),
        ("md", "shadow_md"),
        ("lg", "shadow_lg"),
    ];
    let tile = |name: &str, token: &str| -> Element {
        vstack(
            6.0,
            0.0,
            vec![
                Element::Frame {
                    gap: 0.0,
                    pad: 0.0,
                    children: vec![Element::Spacer { size: 36.0 }],
                    style: Some(Style {
                        background: Some("surface_1".into()),
                        radius: Some("radius_sm".into()),
                        border: Some(Border {
                            color: "chrome_divider".into(),
                            width: 1.0,
                        }),
                        shadow: Some(Shadow {
                            token: Some(token.into()),
                            ..Default::default()
                        }),
                        padding: Some(Edges::symmetric(22.0, 0.0)),
                        ..Default::default()
                    }),
                },
                Element::Text {
                    value: name.into(),
                    color: Some("fg_muted".into()),
                    size: Some(9.0),
                    weight: None,
                    family: Some("font_family_body".into()),
                    selectable: false,
                },
            ],
        )
    };
    let tiles: Vec<Element> = shadows.iter().map(|(n, t)| tile(n, t)).collect();
    card("shadows", "Elevation", vec![hstack(14.0, tiles)])
}

fn spacing_card() -> Element {
    let steps = [
        ("xs", 4.0),
        ("sm", 8.0),
        ("md", 12.0),
        ("lg", 20.0),
        ("xl", 32.0),
    ];
    let row = |name: &str, w: f32| -> Element {
        hstack(
            8.0,
            vec![
                Element::Frame {
                    gap: 0.0,
                    pad: 0.0,
                    children: vec![Element::Spacer { size: 4.0 }],
                    style: Some(Style {
                        background: Some("accent_600".into()),
                        radius: Some("radius_xs".into()),
                        padding: Some(Edges::symmetric(w * 0.5, 0.0)),
                        ..Default::default()
                    }),
                },
                Element::Text {
                    value: name.into(),
                    color: Some("fg_muted".into()),
                    size: Some(10.0),
                    weight: None,
                    family: Some("font_family_mono".into()),
                    selectable: false,
                },
            ],
        )
    };
    let rows: Vec<Element> = steps.iter().map(|(n, w)| row(n, *w)).collect();
    card("spacing", "Rhythm", vec![vstack(6.0, 0.0, rows)])
}

fn components_card() -> Element {
    let buttons = hstack(
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
    );
    let bars = vstack(
        4.0,
        0.0,
        vec![
            hstack(
                8.0,
                vec![
                    muted("Progress"),
                    Element::Bar {
                        value: 0.72,
                        max: 1.0,
                        color: Some("accent".into()),
                        track: Some("surface_3".into()),
                        width: 140.0,
                        height: 6.0,
                        style: None,
                    },
                ],
            ),
            hstack(
                8.0,
                vec![
                    muted("Storage"),
                    Element::Bar {
                        value: 0.34,
                        max: 1.0,
                        color: Some("status_success".into()),
                        track: Some("surface_3".into()),
                        width: 140.0,
                        height: 6.0,
                        style: None,
                    },
                ],
            ),
        ],
    );
    card("components", "Primitives", vec![buttons, spacer(2.0), bars])
}

fn code_card() -> Element {
    // Hand-tokenized snippet so each token can carry the right
    // SYNTAX_* color. Reads as JS-ish on purpose.
    let line1 = hstack(
        0.0,
        vec![
            mono("const ", "syntax_keyword"),
            mono("theme ", "syntax_variable"),
            mono("= ", "syntax_operator"),
            mono("{", "syntax_punctuation"),
        ],
    );
    let line2 = hstack(
        0.0,
        vec![
            mono("  accent: ", "syntax_property"),
            mono("\"#c9a96a\"", "syntax_string"),
            mono(",", "syntax_punctuation"),
        ],
    );
    let line3 = hstack(
        0.0,
        vec![
            mono("  shadow: ", "syntax_property"),
            mono("Shadow", "syntax_type"),
            mono(".", "syntax_punctuation"),
            mono("md", "syntax_function"),
            mono("(),", "syntax_punctuation"),
        ],
    );
    let line4 = hstack(
        0.0,
        vec![
            mono("  fonts: ", "syntax_property"),
            mono("[", "syntax_punctuation"),
            mono("\"serif\"", "syntax_string"),
            mono(", ", "syntax_punctuation"),
            mono("\"sans\"", "syntax_string"),
            mono("]", "syntax_punctuation"),
            mono(",", "syntax_punctuation"),
        ],
    );
    let line5 = mono("};", "syntax_punctuation");
    let line6 = mono("// rebuild + ship", "syntax_comment");
    let body = Element::Frame {
        gap: 2.0,
        pad: 12.0,
        children: vec![line1, line2, line3, line4, line5, spacer(4.0), line6],
        style: Some(Style {
            background: Some("surface_1".into()),
            radius: Some("radius_sm".into()),
            border: Some(Border {
                color: "chrome_divider".into(),
                width: 1.0,
            }),
            ..Default::default()
        }),
    };
    card("code", "Tokens in use", vec![body])
}

fn list_card(selected: &str) -> Element {
    let items = [
        ("buttons", "Primary buttons", "12 px radius, gold fill"),
        ("inputs", "Input fields", "Slate well, accent caret"),
        ("toggles", "Toggle switches", "Pill track, knob slides"),
        ("tabs", "Tab strips", "Accent underline"),
    ];
    let rows: Vec<Element> = items
        .iter()
        .map(|(id, title, sub)| Element::ListItem {
            id: (*id).into(),
            children: vec![
                Element::Text {
                    value: (*title).into(),
                    color: Some("fg".into()),
                    size: Some(13.0),
                    weight: Some(Weight::Bold),
                    family: Some("font_family_body".into()),
                    selectable: false,
                },
                muted(sub),
            ],
            gap: 2.0,
            pad: 8.0,
            selected: *id == selected,
            style: Some(Style {
                radius: Some("radius_sm".into()),
                padding: Some(Edges {
                    top: 6.0,
                    right: 10.0,
                    bottom: 6.0,
                    left: 10.0,
                }),
                ..Default::default()
            }),
        })
        .collect();
    card("anatomy", "Application shell", rows)
}

fn dos_donts_card() -> Element {
    // Single-column layout — the previous side-by-side hstack of
    // do/don't overflowed because the inner text widths didn't
    // intrinsically fit half a card. A vertical list reads cleanly at
    // any card width.
    let row = |ok: bool, msg: &str| -> Element {
        let mark_color = if ok {
            "status_success"
        } else {
            "status_failed"
        };
        hstack(
            8.0,
            vec![
                Element::Text {
                    value: if ok { "✓".into() } else { "✗".into() },
                    color: Some(mark_color.into()),
                    size: Some(12.0),
                    weight: Some(Weight::Bold),
                    family: Some("font_family_body".into()),
                    selectable: false,
                },
                muted(msg),
            ],
        )
    };
    card(
        "guidelines",
        "Do & Don't",
        vec![
            row(true, "Use the surface ladder"),
            row(true, "Pair serif heading + sans body"),
            row(true, "Lift cards with shadow_sm"),
            spacer(4.0),
            row(false, "Mix four accent hues"),
            row(false, "Stack three nested shadows"),
            row(false, "Trust radius_pill on tiny chips"),
        ],
    )
}

fn textures_card(textures_dir: &str) -> Element {
    let tile = |path: &str, label: &str| -> Element {
        let style = Style {
            background_image: Some(format!("{}/{}", textures_dir, path)),
            radius: Some("radius_sm".into()),
            border: Some(Border {
                color: "chrome_divider".into(),
                width: 1.0,
            }),
            ..Default::default()
        };
        vstack(
            6.0,
            0.0,
            vec![
                Element::Frame {
                    gap: 0.0,
                    pad: 0.0,
                    children: vec![Element::Spacer { size: 78.0 }],
                    style: Some(Style {
                        padding: Some(Edges::symmetric(60.0, 0.0)),
                        ..style
                    }),
                },
                muted(label),
            ],
        )
    };
    card(
        "textures & motifs",
        "Atmosphere",
        vec![hstack_top(
            10.0,
            vec![
                tile("fabric.png", "Linen"),
                tile("landscape.png", "Twilight"),
                tile("candle.png", "Embers"),
            ],
        )],
    )
}

fn header() -> Element {
    let title = Element::Text {
        value: "Atelier".into(),
        color: Some("fg".into()),
        size: Some(48.0),
        weight: Some(Weight::Bold),
        family: Some("font_family_heading".into()),
        selectable: false,
    };
    let subtitle = Element::Text {
        value: "Elegant & tactile".into(),
        color: Some("fg_muted".into()),
        size: Some(15.0),
        weight: None,
        family: Some("font_family_heading".into()),
        selectable: false,
    };
    let release = Element::Frame {
        gap: 0.0,
        pad: 0.0,
        children: vec![Element::Text {
            value: "THEME · 01".into(),
            color: Some("accent".into()),
            size: Some(10.0),
            weight: Some(Weight::Bold),
            family: Some("font_family_body".into()),
            selectable: false,
        }],
        style: Some(Style {
            border: Some(Border {
                color: "accent_700".into(),
                width: 1.0,
            }),
            radius: Some("radius_pill".into()),
            padding: Some(Edges::symmetric(10.0, 4.0)),
            ..Default::default()
        }),
    };
    Element::Hstack {
        gap: 16.0,
        pad: 0.0,
        align: Align::Center,
        children: vec![title, subtitle, Element::Spacer { size: 220.0 }, release],
        style: None,
    }
}

fn intro_card(search: &str, dark_mode: bool) -> Element {
    let blurb = Element::Text {
        value: "A warm, tactile dark theme. Slate cards lifted on \
                soft shadows; gold accent reserved for action; serif \
                heading paired with a clean sans for body."
            .into(),
        color: Some("fg_muted".into()),
        size: Some(12.0),
        weight: None,
        family: Some("font_family_body".into()),
        selectable: false,
    };
    let controls = hstack(
        12.0,
        vec![
            Element::Toggle {
                id: "darkmode".into(),
                label: "Dark mode".into(),
                checked: dark_mode,
                style: None,
            },
            Element::Toggle {
                id: "compact".into(),
                label: "Compact".into(),
                checked: false,
                style: None,
            },
            Element::Spacer { size: 16.0 },
            Element::Input {
                id: "search".into(),
                value: search.into(),
                placeholder: "Search tokens…".into(),
                focused: false,
                width: 220.0,
                style: None,
            },
        ],
    );
    Element::Frame {
        gap: 10.0,
        pad: 18.0,
        children: vec![blurb, controls],
        style: Some(Style {
            background: Some("surface_2".into()),
            radius: Some("radius_lg".into()),
            border: Some(Border {
                color: "chrome_divider".into(),
                width: 1.0,
            }),
            shadow: Some(Shadow {
                token: Some("shadow_md".into()),
                ..Default::default()
            }),
            ..Default::default()
        }),
    }
}

fn tabs_strip(active: &str) -> Element {
    Element::Tabs {
        id: "section".into(),
        items: vec![
            TabItem {
                id: "tokens".into(),
                label: "Tokens".into(),
            },
            TabItem {
                id: "components".into(),
                label: "Components".into(),
            },
            TabItem {
                id: "shell".into(),
                label: "Shell".into(),
            },
            TabItem {
                id: "atmosphere".into(),
                label: "Atmosphere".into(),
            },
        ],
        selected: active.into(),
        style: None,
    }
}

fn build_frame(
    active_tab: &str,
    dark_mode: bool,
    search_value: &str,
    selected_row: &str,
    textures_dir: &str,
) -> Element {
    // Two rows of small tokens cards, four cards each row at ~250 px
    // — fits the pane width without clipping and reads as a magazine
    // spread of design-token reference plates.
    let tokens_row_1 = hstack_top(
        12.0,
        vec![palette_card(), typography_card(), spacing_card()],
    );
    let tokens_row_2 = hstack_top(12.0, vec![radii_card(), shadows_card()]);
    let middle = hstack_top(14.0, vec![components_card(), code_card()]);
    let bottom = hstack_top(14.0, vec![list_card(selected_row), dos_donts_card()]);
    let atmosphere = textures_card(textures_dir);

    Element::Vstack {
        gap: 14.0,
        // No outer pad — h1 sits flush with the top of the content
        // area. The pane chrome already provides MARGIN; doubling it
        // up reads as accidental whitespace.
        pad: 0.0,
        children: vec![
            header(),
            intro_card(search_value, dark_mode),
            tabs_strip(active_tab),
            divider(),
            tokens_row_1,
            tokens_row_2,
            middle,
            bottom,
            atmosphere,
        ],
        style: None,
    }
}

// ---- main loop --------------------------------------------------------

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();

    // Resolve textures dir relative to this binary so the host can
    // launch it from anywhere. Fall back to a CWD-relative path that
    // works when running from the workspace root.
    let textures_dir = textures_dir_path();

    let _ = writeln!(
        out,
        "{}",
        serde_json::to_string(&WidgetMsg::Title {
            value: "Atelier".into()
        })
        .unwrap()
    );
    let mut active_tab = String::from("tokens");
    let mut dark_mode = true;
    let mut search_value = String::new();
    let mut selected_row = String::from("buttons");
    emit_frame(
        &mut out,
        &active_tab,
        dark_mode,
        &search_value,
        &selected_row,
        &textures_dir,
    );

    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let Ok(evt) = serde_json::from_str::<HostEvent>(&line) else {
            continue;
        };
        match evt {
            HostEvent::Close => return,
            HostEvent::TabSelect { id, tab } if id == "section" => {
                active_tab = tab;
            }
            HostEvent::Toggle { id, checked } if id == "darkmode" => {
                dark_mode = checked;
            }
            HostEvent::InputChange { id, value } if id == "search" => {
                search_value = value;
            }
            HostEvent::Click { id } => {
                // ListItem rows emit a plain Click.
                selected_row = id;
            }
            _ => continue,
        }
        emit_frame(
            &mut out,
            &active_tab,
            dark_mode,
            &search_value,
            &selected_row,
            &textures_dir,
        );
    }
}

fn textures_dir_path() -> String {
    // 1. <bin>/../../crates/widget-bevy/assets/textures (running from target/{debug,release})
    if let Ok(exe) = std::env::current_exe() {
        let mut p = exe.clone();
        for _ in 0..3 {
            if !p.pop() {
                break;
            }
        }
        let candidate = p
            .join("crates")
            .join("widget-bevy")
            .join("assets")
            .join("textures");
        if candidate.exists() {
            return candidate.to_string_lossy().into_owned();
        }
    }
    // 2. workspace-root-relative
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let candidate = cwd
        .join("crates")
        .join("widget-bevy")
        .join("assets")
        .join("textures");
    candidate.to_string_lossy().into_owned()
}

fn emit_frame<W: Write>(
    out: &mut W,
    tab: &str,
    dark: bool,
    search: &str,
    selected_row: &str,
    textures_dir: &str,
) {
    let msg = WidgetMsg::Frame {
        root: build_frame(tab, dark, search, selected_row, textures_dir),
    };
    if let Ok(s) = serde_json::to_string(&msg) {
        let _ = writeln!(out, "{}", s);
        let _ = out.flush();
    }
}
