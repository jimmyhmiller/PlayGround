// Atelier — warm gold accent on cool slate.
//
// Source of truth for the engine's *default* palette AND the
// `atelier` style preset. style-bevy embeds this file via
// `include_str!`; presets.rs seeds the same content to
// `~/.terminal-bevy/styles/atelier/theme.rhai` on startup.
//
// Edits here affect both the in-app default look (taken at startup)
// and the seed copy on disk (first run only — user copies aren't
// overwritten).
//
// To override per project, write `~/.terminal-bevy/projects/<id>/
// theme.rhai` with the tokens you want to change.
#{
    // --- core text + accent ---
    bg: "#10121a",
    fg: "#e8e4dd",
    fg_muted: "#878a94",
    accent: "#c9a96a",            // warm gold
    caret: "#e3c280",
    selection: "#6b572f80",
    warn: "#dfb06b",
    err: "#d06b4d",

    // --- pane chrome (SDF material). Focus changes ONLY title-bar
    // text + title-bar background; border/glow stay stable so the
    // pane body doesn't visually shift under focus.
    pane_bg: "#181f20",
    pane_border: "#293032",
    pane_border_focused: "#293032",
    pane_focus_glow: "#00000000",
    pane_corner_radius: 8.0,
    pane_border_width: 1.0,
    pane_border_width_focused: 1.0,
    pane_focus_width: 0.0,
    pane_focus_strength: 0.0,
    pane_shadow_color: "#00000073",
    pane_shadow_blur: 20.0,
    pane_shadow_offset_y: 6.0,

    // --- chrome text + title-strip fill ---
    chrome_title: "#80818c",
    chrome_title_focused: "#e8e4dd",
    chrome_title_bg: "#0d1013",
    chrome_title_bg_focused: "#2a2e37",
    chrome_divider: "#272a32",
    chrome_close: "#80818c",
    chrome_handle: "#383b42",

    // --- syntax — warm Atelier palette ---
    syntax_default: "#d9d9d4",
    syntax_keyword: "#cfb06f",      // warm gold
    syntax_string: "#94be9e",       // sage
    syntax_comment: "#6b6b79",
    syntax_function: "#bad2e0",     // soft sky
    syntax_type: "#d0baea",         // lavender
    syntax_attribute: "#d2aa82",
    syntax_constant: "#e38f6b",     // terracotta
    syntax_operator: "#a6a8af",
    syntax_punctuation: "#9ea0a6",
    syntax_variable: "#d9d9d4",
    syntax_property: "#d0baea",
    syntax_label: "#e38f6b",
    syntax_escape: "#e3c280",
    syntax_constructor: "#d0baea",

    // --- form fields ---
    input_bg: "#13151b",
    input_text: "#b8b6b2",
    input_text_focused: "#e8e4dd",

    // --- buttons ---
    button_bg: "#c9a96a",
    button_bg_hover: "#d6b878",
    button_label: "#181410",
    button_primary_bg: "#6b8f61",
    button_primary_label: "#f0f0e8",

    // --- widget Button SDF shape + shadow ---
    widget_button_corner_radius: 6.0,
    widget_button_border: "#00000000",
    widget_button_border_width: 0.0,
    widget_button_shadow_color: "#05030199",
    widget_button_shadow_blur: 8.0,
    widget_button_shadow_offset_y: 2.0,

    // --- status ---
    status_idle: "#9eb9d1",
    status_running: "#e3c280",
    status_success: "#94be9e",
    status_failed: "#d06b4d",

    // --- diff view ---
    diff_added: "#94be9e",
    diff_removed: "#d77a5f",
    diff_header: "#7f93ad",
    diff_added_bg: "#9bd6a526",
    diff_removed_bg: "#e0795f24",
    diff_gutter: "#6b6b79",

    // --- radial menu ---
    radial_wedge: "#21262e",
    radial_wedge_hover: "#383028",
    radial_deadzone: "#13161b",
    radial_deadzone_ring: "#3c4048",
    radial_label: "#d6d2cb",
    radial_label_hover: "#f8f5ef",
    radial_icon: "#f0ede5",
    radial_backdrop: "#00000052",

    // --- widget protocol extras ---
    widget_bar_track: "#21262e",
    widget_bar_fill: "#c9a96a",
    widget_badge_bg: "#27303a",
    widget_badge_label: "#dec4b3",
    widget_link: "#cfb06f",

    // --- shell (canvas + sidebar) ---
    canvas_bg: "#10121a",
    sidebar_bg: "#141821",
    sidebar_row_active_bg: "#241f1a",
    sidebar_row_renaming_bg: "#22252a",
    sidebar_text_faint: "#61656b",

    // --- design-token scales ---
    space_xs: 4.0,
    space_sm: 8.0,
    space_md: 12.0,
    space_lg: 20.0,
    space_xl: 32.0,

    radius_xs: 2.0,
    radius_sm: 4.0,
    radius_md: 8.0,
    radius_lg: 14.0,
    radius_pill: 9999.0,

    // Shadow ladder — cool black, tight. Warm shadows stacked across
    // many cards in a dense layout get muddy.
    shadow_sm_color: "#00000033",
    shadow_sm_blur: 4.0,
    shadow_sm_offset_y: 1.0,
    shadow_md_color: "#00000052",
    shadow_md_blur: 10.0,
    shadow_md_offset_y: 3.0,
    shadow_lg_color: "#00000073",
    shadow_lg_blur: 24.0,
    shadow_lg_offset_y: 10.0,

    // Surface ladder — cool slate gradient.
    surface_1: "#10121a",
    surface_2: "#181f20",
    surface_3: "#21282a",

    // Accent ramp (warm gold 50..900).
    accent_50: "#fbf8f0",
    accent_100: "#f4ede0",
    accent_200: "#ecdec2",
    accent_300: "#e3cea3",
    accent_400: "#d8bd87",
    accent_500: "#c9a96a",
    accent_600: "#ae8e54",
    accent_700: "#8a703f",
    accent_800: "#664f2c",
    accent_900: "#45351c",

    // Typography roles — string tokens, resolved through FontRegistry.
    font_family_heading: "serif",
    font_family_body: "sans",
    font_family_mono: "mono",
    font_size: 14.0,
    line_height_ratio: 1.3,

    // Wipe / dust shader tuning (kept theme-controllable so projects
    // can dial it down without touching code).
    wipe_dust_gate_secs: 60.0,
    wipe_brush_radius_px: 80.0,
    dust_intensity: 1.0,
}
