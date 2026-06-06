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
