use glaze::{GlazeError, Layer, Length, Value, parse};
use std::collections::HashMap;

fn variant(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
}

const SHEET: &str = include_str!("../examples/atelier.glz");

#[test]
fn parses_sample_sheet() {
    let prog = parse(SHEET).expect("sample should parse");
    assert_eq!(prog.tokens.len(), 12);
    assert_eq!(prog.styles.len(), 2);
}

#[test]
fn token_alias_resolves_through_layers() {
    let prog = parse(SHEET).unwrap();
    // accent.solid -> gold.500 -> oklch(...); just confirm it evaluates to a color
    match prog.eval_token("accent.solid").unwrap() {
        Value::Color(_) => {}
        other => panic!("expected color, got {other:?}"),
    }
}

#[test]
fn variant_selects_fill_color() {
    let prog = parse(SHEET).unwrap();
    let primary = prog.resolve("button", &variant(&[("intent", "primary")]), &[]).unwrap();
    let danger = prog.resolve("button", &variant(&[("intent", "danger")]), &[]).unwrap();
    let fill = |c: &glaze::CompiledStyle| match c.layers[0] {
        Layer::Fill(rgba) => rgba,
        _ => panic!("first layer should be a fill"),
    };
    // primary (gold) and danger (red) must differ
    assert_ne!(fill(&primary), fill(&danger));
    // box props folded
    assert_eq!(primary.box_.radius, 8.0);
    assert_eq!(primary.box_.padding, [8.0, 12.0, 8.0, 12.0]);
}

#[test]
fn discrete_state_overlays_apply() {
    let prog = parse(SHEET).unwrap();
    let base = prog.resolve("button", &variant(&[("intent", "primary")]), &[]).unwrap();
    let hover = prog.resolve("button", &variant(&[("intent", "primary")]), &["hover"]).unwrap();
    // base has one fill; :hover adds a second fill overlay
    assert_eq!(base.layers.len(), 1);
    assert_eq!(hover.layers.len(), 2);

    let focus = prog.resolve("button", &variant(&[("intent", "primary")]), &["focus"]).unwrap();
    assert!(matches!(focus.layers.last(), Some(Layer::Border { width, .. }) if *width == 2.0));
}

#[test]
fn card_has_no_variant_and_folds() {
    let prog = parse(SHEET).unwrap();
    let card = prog.resolve("card", &variant(&[]), &[]).unwrap();
    assert_eq!(card.box_.radius, 8.0);
    assert_eq!(card.box_.padding, [16.0; 4]);
    // fill + border
    assert!(matches!(card.layers[0], Layer::Fill(_)));
    assert!(matches!(card.layers[1], Layer::Border { .. }));
}

#[test]
fn shader_layer_compiles_to_wgsl_with_staging() {
    // The explainer's glow: `time`/`hover` are dynamic → uniforms; the gold
    // token and the vec4 literal are static → folded to constants.
    let prog = parse(
        r#"
        token accent.solid = oklch(0.74 0.12 85)
        style glow(intent) {
          fill accent.solid
          overlay shader {
            let pulse = 0.5 + 0.5*sin(time*2)
            emit smoothstep(0,1,hover) * pulse * vec4(1,1,1,0.25)
          }
        }
        "#,
    )
    .unwrap();
    let compiled = prog.resolve("glow", &variant(&[("intent", "primary")]), &[]).unwrap();
    // base fill + a shader layer
    assert_eq!(compiled.layers.len(), 2);
    let shader = match compiled.layers.last().unwrap() {
        Layer::Shader(s) => s,
        other => panic!("expected a shader layer, got {other:?}"),
    };
    assert!(shader.overlay);
    // dynamic builtins became uniforms
    assert!(shader.used.contains(&glaze::Builtin::Time));
    assert!(shader.used.contains(&glaze::Builtin::Hover));
    // `time`/`hover` reference the uniform block; the static vec4 is folded
    let w = &shader.wgsl_body;
    assert!(w.contains("u.time"), "wgsl:\n{w}");
    assert!(w.contains("u.hover"), "wgsl:\n{w}");
    assert!(w.contains("let pulse ="), "the dynamic let survives:\n{w}");
    assert!(w.contains("vec4<f32>(1") || w.contains("vec4<f32>(1.0"), "static vec4 folded:\n{w}");
    // no per-frame builtin leaked into a CPU constant
    assert!(!w.contains("smoothstep(0.0, 1.0, hover)"), "hover must be u.hover:\n{w}");
}

#[test]
fn fully_static_shader_folds_to_a_constant_return() {
    // No dynamic builtin → the whole emit folds; no uniforms used.
    let prog = parse(
        r#"
        token accent.solid = oklch(0.74 0.12 85)
        style s { overlay shader { emit accent.solid } }
        "#,
    )
    .unwrap();
    let compiled = prog.resolve("s", &variant(&[]), &[]).unwrap();
    let shader = match &compiled.layers[0] {
        Layer::Shader(s) => s,
        other => panic!("got {other:?}"),
    };
    assert!(shader.used.is_empty(), "static shader needs no uniforms");
    assert!(shader.wgsl_body.contains("return vec4<f32>("), "folded:\n{}", shader.wgsl_body);
    assert!(!shader.wgsl_body.contains("u."), "no uniform refs:\n{}", shader.wgsl_body);
}

#[test]
fn swizzles_lower_to_member_access() {
    // `resolution.x`/`uv.y` are swizzles on builtins; `accent.solid` stays a token.
    let prog = parse(
        r#"
        token accent.solid = oklch(0.74 0.12 85)
        style s { overlay shader {
          let aspect = resolution.x / resolution.y
          let p      = (uv - vec2(0.5, 0.5)) * vec2(aspect, 1.0)
          emit smoothstep(0.5, 0.0, length(p)) * accent.solid
        } }
        "#,
    )
    .unwrap();
    let c = prog.resolve("s", &variant(&[]), &[]).unwrap();
    let Layer::Shader(sh) = &c.layers[0] else { panic!("expected shader layer") };
    let w = &sh.wgsl_body;
    assert!(w.contains("u.resolution.x"), "{w}");
    assert!(w.contains("u.resolution.y"), "{w}");
    assert!(w.contains("in.uv"), "{w}");
    // the gold token still folds to a constant (not a swizzle)
    assert!(w.contains("vec4<f32>("), "{w}");
    assert!(sh.used.contains(&glaze::Builtin::Uv));
    assert!(sh.used.contains(&glaze::Builtin::Resolution));
}

#[test]
fn ternary_lowers_to_select() {
    let prog = parse(
        r#"
        style s { overlay shader { emit hover > 0.5 ? vec4(1,0,0,1) : vec4(0,0,0,1) } }
        "#,
    )
    .unwrap();
    let compiled = prog.resolve("s", &variant(&[]), &[]).unwrap();
    if let Layer::Shader(s) = &compiled.layers[0] {
        assert!(s.wgsl_body.contains("select("), "wgsl:\n{}", s.wgsl_body);
        assert!(s.wgsl_body.contains("u.hover"));
    } else {
        panic!("expected shader layer");
    }
}

#[test]
fn responsive_when_flips_direction() {
    let prog = parse(
        r#"
        style bar {
          direction row
          when vw < 560 { direction column }
        }
        "#,
    )
    .unwrap();
    let wide = prog.resolve_at("bar", &variant(&[]), &[], 800.0, 600.0).unwrap();
    let narrow = prog.resolve_at("bar", &variant(&[]), &[], 400.0, 600.0).unwrap();
    assert_eq!(wide.box_.flex_direction, Some(glaze::Dir::Row));
    assert_eq!(narrow.box_.flex_direction, Some(glaze::Dir::Column));
    // plain resolve() uses an infinite viewport → no breakpoint fires
    let dflt = prog.resolve("bar", &variant(&[]), &[]).unwrap();
    assert_eq!(dflt.box_.flex_direction, Some(glaze::Dir::Row));
}

#[test]
fn functions_and_let_variables() {
    let prog = parse(
        r#"
        fn space(n)  = n * 4px
        fn tint(c, a) = mix(c, oklch(0.2 0.01 250), a)
        token gold = oklch(0.8 0.13 85)
        style card {
          let p = space(4)     // 16px
          pad    p
          radius space(2)      // 8px
          fill   tint(gold, 0.3)
        }
        "#,
    )
    .unwrap();
    let c = prog.resolve("card", &variant(&[]), &[]).unwrap();
    assert_eq!(c.box_.padding, [16.0; 4]);
    assert_eq!(c.box_.radius, 8.0);
    assert!(matches!(c.layers[0], Layer::Fill(_)));
}

#[test]
fn function_inlines_into_shader() {
    let prog = parse(
        r#"
        fn circle(p, r) = smoothstep(r, r - 0.02, length(p - vec2(0.5, 0.5)))
        style s { overlay shader { emit circle(uv, 0.3) * vec4(1, 1, 1, 1) } }
        "#,
    )
    .unwrap();
    let c = prog.resolve("s", &variant(&[]), &[]).unwrap();
    let Layer::Shader(sh) = &c.layers[0] else { panic!() };
    assert!(sh.wgsl_body.contains("in.uv"), "{}", sh.wgsl_body);
    assert!(sh.wgsl_body.contains("smoothstep"), "{}", sh.wgsl_body);
    assert!(sh.used.contains(&glaze::Builtin::Uv));
}

#[test]
fn recursive_function_rejected() {
    let err = parse("fn f(x) = f(x)\nstyle s { radius 1px }").unwrap_err();
    assert!(matches!(err, GlazeError::Parse(_)), "got {err:?}");
}

#[test]
fn unknown_property_errors() {
    let prog = parse("style x { wobble 3px }").unwrap();
    let err = prog.resolve("x", &variant(&[]), &[]).unwrap_err();
    assert!(matches!(err, GlazeError::Eval(_)), "got {err:?}");
}

#[test]
fn token_cycle_detected() {
    let prog = parse("token a = b\ntoken b = a\nstyle s { fill a }").unwrap();
    let err = prog.resolve("s", &variant(&[]), &[]).unwrap_err();
    match err {
        GlazeError::Eval(m) => assert!(m.contains("cycle"), "got {m}"),
        other => panic!("expected cycle error, got {other:?}"),
    }
}

#[test]
fn lengths_and_units() {
    let prog = parse("style s { width 50%\nheight auto\nradius 4px }").unwrap();
    let c = prog.resolve("s", &variant(&[]), &[]).unwrap();
    assert_eq!(c.box_.width, Some(glaze::Dim::Pct(50.0)));
    assert_eq!(c.box_.height, Some(glaze::Dim::Auto));
    assert_eq!(c.box_.radius, 4.0);
    // sanity: a bare px length round-trips
    let _ = Length::Px(4.0);
}

// ---------- Phase 1b: widened layer stack ----------

const STOPS: &str = r#"
token brand.a = #ff0000
token brand.b = #00ff00
token brand.c = #0000ff

style grad {
    gradient 90 brand.a brand.b brand.c
}
style grad_offsets {
    gradient brand.a 0% brand.b 25% brand.c 100%
}
style edges {
    border_top  brand.a 2px
    border_left brand.b 3px
}
style shadows {
    shadow brand.a 12px 4px 2px
    inset_shadow brand.b 8px 0 0
}
"#;

#[test]
fn gradient_distributes_missing_offsets_evenly() {
    let prog = parse(STOPS).unwrap();
    let c = prog.resolve("grad", &variant(&[]), &[]).unwrap();
    match &c.layers[0] {
        Layer::LinearGradient { angle, stops } => {
            assert_eq!(*angle, 90.0);
            assert_eq!(stops.len(), 3);
            // evenly distributed 0, 0.5, 1
            assert!((stops[0].offset - 0.0).abs() < 1e-5);
            assert!((stops[1].offset - 0.5).abs() < 1e-5);
            assert!((stops[2].offset - 1.0).abs() < 1e-5);
        }
        other => panic!("expected gradient, got {other:?}"),
    }
}

#[test]
fn gradient_honors_explicit_offsets_and_default_angle() {
    let prog = parse(STOPS).unwrap();
    let c = prog.resolve("grad_offsets", &variant(&[]), &[]).unwrap();
    match &c.layers[0] {
        Layer::LinearGradient { angle, stops } => {
            assert_eq!(*angle, 180.0); // omitted → default
            let offs: Vec<f32> = stops.iter().map(|s| s.offset).collect();
            assert_eq!(offs, vec![0.0, 0.25, 1.0]);
        }
        other => panic!("expected gradient, got {other:?}"),
    }
}

#[test]
fn per_side_borders_set_only_their_edge() {
    let prog = parse(STOPS).unwrap();
    let c = prog.resolve("edges", &variant(&[]), &[]).unwrap();
    let sides: Vec<glaze::Sides> = c
        .layers
        .iter()
        .filter_map(|l| match l {
            Layer::Border { sides, .. } => Some(*sides),
            _ => None,
        })
        .collect();
    assert_eq!(sides.len(), 2);
    assert_eq!(sides[0], glaze::Sides::only(true, false, false, false)); // top
    assert!(!sides[0].is_all());
    assert_eq!(sides[1], glaze::Sides::only(false, false, false, true)); // left
}

#[test]
fn shadow_and_inset_shadow_carry_params() {
    let prog = parse(STOPS).unwrap();
    let c = prog.resolve("shadows", &variant(&[]), &[]).unwrap();
    match &c.layers[0] {
        Layer::Shadow { blur, offset_y, spread, inset, .. } => {
            assert_eq!((*blur, *offset_y, *spread), (12.0, 4.0, 2.0));
            assert!(!inset);
        }
        other => panic!("expected shadow, got {other:?}"),
    }
    match &c.layers[1] {
        Layer::Shadow { blur, inset, .. } => {
            assert_eq!(*blur, 8.0);
            assert!(inset);
        }
        other => panic!("expected inset shadow, got {other:?}"),
    }
}

// ---------- Phase 1c: slots (part {}) ----------

const SLOTTED: &str = r#"
token track.bg  = #222222
token fill.lo   = #00ff00
token fill.hi   = #ff0000
token r.pill    = 999px

style bar {
    let h = 8px           // top-level let, visible inside parts
    radius r.pill         // base (root box) prop
    height h
    track {
        fill   track.bg
        radius r.pill
    }
    fill {
        fill   fill.lo
        radius r.pill
        :active { fill fill.hi }
    }
}

style bad_nest {
    track {
        inner {
            fill track.bg
        }
    }
}
"#;

#[test]
fn slots_split_base_from_parts() {
    let prog = parse(SLOTTED).unwrap();
    let s = prog.resolve_slots("bar", &variant(&[]), &[]).unwrap();
    // base = the root box props only (radius + height), no part fills
    assert_eq!(s.base.box_.radius, 999.0);
    assert!(s.base.layers.is_empty(), "base must not absorb part layers");
    // two slots, in source order
    let names: Vec<&str> = s.slot_names().collect();
    assert_eq!(names, vec!["track", "fill"]);
    // a part carries its own layers + box
    let track = s.slot("track").unwrap();
    assert_eq!(track.box_.radius, 999.0);
    assert_eq!(track.layers.len(), 1); // one fill
    assert!(s.slot("nope").is_none());
}

#[test]
fn slot_inherits_top_level_let() {
    let prog = parse(SLOTTED).unwrap();
    let s = prog.resolve_slots("bar", &variant(&[]), &[]).unwrap();
    // `height h` where `let h = 8px` at top level — base picks it up
    assert_eq!(s.base.box_.height, Some(glaze::Dim::Px(8.0)));
}

#[test]
fn slot_state_overlay_applies_within_part() {
    let prog = parse(SLOTTED).unwrap();
    let off = prog.resolve_slots("bar", &variant(&[]), &[]).unwrap();
    let on = prog.resolve_slots("bar", &variant(&[]), &["active"]).unwrap();
    let fill_color = |s: &glaze::CompiledSlots| match s.slot("fill").unwrap().layers.last().unwrap() {
        Layer::Fill(c) => *c,
        other => panic!("expected fill, got {other:?}"),
    };
    // :active overlay flips the fill slot's color (green → red)
    assert_ne!(fill_color(&off), fill_color(&on));
}

#[test]
fn nested_parts_are_rejected() {
    let prog = parse(SLOTTED).unwrap();
    let err = prog.resolve_slots("bad_nest", &variant(&[]), &[]).unwrap_err();
    assert!(matches!(err, GlazeError::Parse(_)), "got {err:?}");
}

#[test]
fn non_slotted_style_still_resolves_flat() {
    // a style with no parts behaves exactly as before through resolve()
    let prog = parse(SLOTTED).unwrap();
    let flat = prog.resolve("bar", &variant(&[]), &[]).unwrap();
    // resolve() ignores parts: base box only, no part layers leaked in
    assert_eq!(flat.box_.radius, 999.0);
    assert!(flat.layers.is_empty());
}
