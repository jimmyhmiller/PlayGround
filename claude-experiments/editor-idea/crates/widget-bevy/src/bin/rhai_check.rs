//! Parse-and-smoke-test a rhai widget script. Compiles it, primes
//! `state`/`canvas_w`/`canvas_h` in scope, runs the top-level body,
//! then calls `on_init` and `render` and `on_click` for every square
//! to flush out runtime errors in legal-move generation. No host fns
//! are registered beyond no-op stubs for the ones widgets typically
//! call (`request_render`, `set_animating`, etc.).
use rhai::{Dynamic, Engine, Scope};

fn main() {
    let path = std::env::args().nth(1).expect("usage: rhai_check <path>");
    let body = std::fs::read_to_string(&path).expect("read script");
    let mut engine = Engine::new();
    engine.set_max_expr_depths(256, 128);
    engine.register_fn("set_animating", |_on: bool| {});
    engine.register_fn("request_render", || {});
    engine.register_fn("host_log", |msg: &str| eprintln!("[rhai] {}", msg));
    engine.register_fn("hash_str", |s: &str| -> i64 { s.len() as i64 });
    engine.register_fn("rand", || -> f64 { 0.5 });
    engine.register_fn("rand_int", |lo: i64, hi: i64| -> i64 { lo + (hi - lo) / 2 });
    engine.register_fn("time", || -> f64 { 0.0 });
    engine.register_fn("widget_asset", |rel: &str| -> String {
        let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("assets");
        path.push(rel);
        path.to_string_lossy().into_owned()
    });

    let ast = match engine.compile(&body) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("parse error in {}: {}", path, e);
            std::process::exit(1);
        }
    };

    let mut scope = Scope::new();
    scope.push("state", Dynamic::from(rhai::Map::new()));
    scope.push("canvas_w", 600.0_f64);
    scope.push("canvas_h", 600.0_f64);

    if let Err(e) = engine.run_ast_with_scope(&mut scope, &ast) {
        eprintln!("top-level error in {}: {}", path, e);
        std::process::exit(1);
    }

    if let Err(e) = engine.call_fn::<Dynamic>(&mut scope, &ast, "on_init", ()) {
        if !matches!(*e, rhai::EvalAltResult::ErrorFunctionNotFound(_, _)) {
            eprintln!("on_init error: {}", e);
            std::process::exit(1);
        }
    }

    let frame = engine
        .call_fn::<Dynamic>(&mut scope, &ast, "render", (600.0_f64, 600.0_f64))
        .unwrap_or_else(|e| panic!("render error: {}", e));
    if frame.is_unit() {
        println!("render returned unit (no frame)");
    } else {
        // light sanity check on shape
        if let Some(map) = frame.read_lock::<rhai::Map>() {
            let kind = map
                .get("type")
                .and_then(|d| d.clone().into_string().ok())
                .unwrap_or_else(|| "?".to_string());
            println!("render ok: root type = {}", kind);
        } else {
            println!("render ok (non-map root)");
        }
    }

    // Sequence of clicks: 1. e2 (select pawn), 2. e4 (move), 3. e7 (black
    // pawn select), 4. e5 (move). Coords assume 600x600 canvas, 32px bar,
    // ~70px squares. Board origin = (12, 32 + ((600-32-560)/2)) = (12, 36).
    // e-file = col 4, center x ≈ 12 + 4.5 * 70 = 327.
    // rank 2 = row 6 (from top), center y ≈ 36 + 6.5 * 70 = 491.
    // rank 4 = row 4, center y ≈ 36 + 4.5 * 70 = 351.
    // rank 7 = row 1, center y ≈ 36 + 1.5 * 70 = 141.
    // rank 5 = row 3, center y ≈ 36 + 3.5 * 70 = 281.
    let clicks: &[(f64, f64)] = &[(327.0, 491.0), (327.0, 351.0), (327.0, 141.0), (327.0, 281.0)];
    for &(cx, cy) in clicks {
        let args = (cx, cy, false, false, "".to_string());
        if let Err(e) = engine.call_fn::<Dynamic>(&mut scope, &ast, "on_click", args) {
            if !matches!(*e, rhai::EvalAltResult::ErrorFunctionNotFound(_, _)) {
                eprintln!("on_click error at ({},{}): {}", cx, cy, e);
                std::process::exit(1);
            }
        }
    }

    // Render after the opening to make sure post-move legal_from /
    // king-search paths don't blow up.
    let _ = engine.call_fn::<Dynamic>(&mut scope, &ast, "render", (600.0_f64, 600.0_f64))
        .unwrap_or_else(|e| panic!("post-click render error: {}", e));

    // Exercise drag handlers if the script defines them. a2 is row 6,
    // col 0 → center ≈ (47, 491). Drag to a4 → center ≈ (47, 351).
    let drag_seq: &[(&str, f64, f64)] = &[
        ("on_click", 47.0, 491.0),
        ("on_drag", 47.0, 450.0),
        ("on_drag", 47.0, 400.0),
        ("on_release", 47.0, 351.0),
    ];
    for &(name, cx, cy) in drag_seq {
        let result = if name == "on_click" {
            engine.call_fn::<Dynamic>(
                &mut scope,
                &ast,
                name,
                (cx, cy, false, false, "".to_string()),
            )
        } else {
            engine.call_fn::<Dynamic>(&mut scope, &ast, name, (cx, cy))
        };
        if let Err(e) = result {
            if !matches!(*e, rhai::EvalAltResult::ErrorFunctionNotFound(_, _)) {
                eprintln!("{} error at ({},{}): {}", name, cx, cy, e);
                std::process::exit(1);
            }
        }
    }
    let _ = engine
        .call_fn::<Dynamic>(&mut scope, &ast, "render", (600.0_f64, 600.0_f64))
        .unwrap_or_else(|e| panic!("post-drag render error: {}", e));

    // Inspect state.turn and state.last_to to confirm moves landed.
    if let Some(s) = scope.get_value::<Dynamic>("state") {
        if let Some(map) = s.read_lock::<rhai::Map>() {
            let turn = map
                .get("turn")
                .and_then(|d| d.clone().into_string().ok())
                .unwrap_or_default();
            let last_to = map
                .get("last_to")
                .and_then(|d| d.as_int().ok())
                .unwrap_or(-1);
            println!("after opening: turn={} last_to={}", turn, last_to);
        }
    }
    println!("ok: {}", path);
}
