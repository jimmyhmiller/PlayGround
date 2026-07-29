//! The browser demo's scene, driven headlessly: the same program, foreign
//! bindings, and scripted edits the wasm demo ships, checked here so the
//! language-level behaviour is pinned by the test suite rather than by
//! eyeballing a canvas.

use livetype_core::{ActorStatus, Session, Turn, Value};
use std::sync::{Arc, Mutex};

const INTERFACE: &str = include_str!("../demo/interface.lt");
const SCENE: &str = include_str!("../demo/scene.lt");

/// One drawing call. `a`/`b` are the radius and an unused 0 for the round
/// shapes, the width and height for a bar — the same flat encoding the wasm
/// host hands to its canvas.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Shape {
    /// Which foreign function the guest called: the shape it chose to draw.
    kind: &'static str,
    x: i64,
    y: i64,
    a: i64,
    b: i64,
    hue: i64,
}

/// What the guest drew, in order — the browser's canvas in miniature.
#[derive(Default)]
struct Toolkit {
    canvases_opened: u64,
    frames_cleared: u64,
    /// Most recent frame last.
    shapes: Vec<Shape>,
}

fn bind(session: &mut Session, tk: &Arc<Mutex<Toolkit>>) {
    let kind = session
        .foreign_kind("Canvas")
        .expect("Canvas declared by the scene");
    {
        let tk = Arc::clone(tk);
        session
            .register_foreign(
                "open_canvas",
                Box::new(move |_| {
                    let mut t = tk.lock().unwrap();
                    t.canvases_opened += 1;
                    Value::Foreign { kind, ptr: t.canvases_opened }
                }),
            )
            .unwrap();
    }
    {
        let tk = Arc::clone(tk);
        session
            .register_foreign(
                "clear",
                Box::new(move |_| {
                    tk.lock().unwrap().frames_cleared += 1;
                    Value::Unit
                }),
            )
            .unwrap();
    }
    // `circle` and `ring` share an arity and differ only in what they paint,
    // which is the point: the guest picks between them by dispatching on an
    // enum, and this records which one it picked.
    for name in ["circle", "ring"] {
        let tk = Arc::clone(tk);
        session
            .register_foreign(
                name,
                Box::new(move |args| {
                    let [_, Value::I64(x), Value::I64(y), Value::I64(r), Value::I64(hue)] = args
                    else {
                        panic!("{name} called with the wrong argument shapes: {args:?}");
                    };
                    tk.lock().unwrap().shapes.push(Shape {
                        kind: name,
                        x: *x,
                        y: *y,
                        a: *r,
                        b: 0,
                        hue: *hue,
                    });
                    Value::Unit
                }),
            )
            .unwrap();
    }
    {
        let tk = Arc::clone(tk);
        session
            .register_foreign(
                "bar",
                Box::new(move |args| {
                    let [
                        _,
                        Value::I64(x),
                        Value::I64(y),
                        Value::I64(w),
                        Value::I64(h),
                        Value::I64(hue),
                    ] = args
                    else {
                        panic!("bar called with the wrong argument shapes: {args:?}");
                    };
                    tk.lock().unwrap().shapes.push(Shape {
                        kind: "bar",
                        x: *x,
                        y: *y,
                        a: *w,
                        b: *h,
                        hue: *hue,
                    });
                    Value::Unit
                }),
            )
            .unwrap();
    }
}

/// One frame: step until the guest's `yield`, exactly as the browser's
/// requestAnimationFrame driver does.
fn frame(session: &mut Session, actor: &mut livetype_core::Actor) -> Turn {
    for _ in 0..200_000 {
        match session.engine.step(actor) {
            Turn::Progress => {}
            other => return other,
        }
    }
    panic!("frame did not reach a yield within the instruction cap");
}

fn boot() -> (Session, livetype_core::Actor, Arc<Mutex<Toolkit>>) {
    let tk = Arc::new(Mutex::new(Toolkit::default()));
    let mut session = Session::new();
    // Declare the foreign interface, bind its implementations, and only then
    // load the scene — a `letonce` initializer calls `open_canvas` as the
    // scene is evaluated, so the binding has to already be in place.
    session.eval(INTERFACE).expect("interface compiles");
    bind(&mut session, &tk);
    session.eval(SCENE).expect("scene compiles");
    let main = session.fn_id("main").expect("main");
    let actor = session.engine.spawn(main, vec![]).expect("spawn main");
    (session, actor, tk)
}

#[test]
fn the_scene_runs_and_draws_every_frame() {
    let (mut session, mut actor, tk) = boot();
    for _ in 0..5 {
        assert!(matches!(frame(&mut session, &mut actor), Turn::Yielded));
    }
    let t = tk.lock().unwrap();
    assert_eq!(t.canvases_opened, 1, "letonce opened the canvas exactly once");
    assert_eq!(t.frames_cleared, 5);
    assert_eq!(t.shapes.len(), 5 * 18, "18 particles drawn per frame");
    assert!(t.shapes.iter().all(|s| s.a == 14), "radius() is still 14");
    assert!(
        t.shapes.iter().all(|s| s.kind == "circle"),
        "every particle is a filled disc before any edit"
    );
}

#[test]
fn particles_actually_move() {
    let (mut session, mut actor, tk) = boot();
    frame(&mut session, &mut actor);
    let first = tk.lock().unwrap().shapes[0];
    frame(&mut session, &mut actor);
    let second = tk.lock().unwrap().shapes[18];
    assert_ne!((first.x, first.y), (second.x, second.y), "particle 0 moved");
}

#[test]
fn particles_stay_on_the_canvas() {
    let (mut session, mut actor, tk) = boot();
    for _ in 0..400 {
        frame(&mut session, &mut actor);
    }
    let t = tk.lock().unwrap();
    for s in &t.shapes {
        assert!((14..=626).contains(&s.x), "x {} escaped the canvas", s.x);
        assert!((14..=386).contains(&s.y), "y {} escaped the canvas", s.y);
    }
}

/// Changing the seed count and restarting is the demo's "Restart world with
/// this source" path. The particles have to actually land ON the canvas: an
/// earlier layout put everything past the first screenful off-canvas, where
/// `advance` clamped it into a corner, so 200 particles looked like 18 and the
/// edit looked ignored even though it had taken effect.
#[test]
fn a_bigger_seed_count_spreads_across_the_canvas() {
    let scene = SCENE.replace("while i < 18", "while i < 200");
    let tk = Arc::new(Mutex::new(Toolkit::default()));
    let mut session = Session::new();
    session.eval(INTERFACE).expect("interface compiles");
    bind(&mut session, &tk);
    session.eval(&scene).expect("resized scene compiles");
    let main = session.fn_id("main").expect("main");
    let mut actor = session.engine.spawn(main, vec![]).expect("spawn main");

    frame(&mut session, &mut actor);
    let first = {
        let t = tk.lock().unwrap();
        assert_eq!(t.shapes.len(), 200, "all 200 particles are drawn");
        t.shapes.clone()
    };

    // Spread, not a pile. Every particle must be individually visible: distinct
    // positions, not merely distinct columns — a coarse grid passed a
    // per-axis check while stacking four particles per slot.
    let positions: std::collections::BTreeSet<(i64, i64)> =
        first.iter().map(|s| (s.x, s.y)).collect();
    assert_eq!(positions.len(), 200, "every particle is at its own spot");
    let cornered = first.iter().filter(|s| s.x >= 620 && s.y >= 380).count();
    assert!(cornered < 10, "{cornered} particles piled into the corner");

    // And they stay individually visible: identical velocities would make
    // co-located particles move as one for the rest of the run.
    for _ in 0..40 {
        frame(&mut session, &mut actor);
    }
    let latest: Vec<(i64, i64)> = {
        let t = tk.lock().unwrap();
        t.shapes[t.shapes.len() - 200..].iter().map(|s| (s.x, s.y)).collect()
    };
    let distinct: std::collections::BTreeSet<(i64, i64)> = latest.iter().copied().collect();
    assert!(
        distinct.len() > 150,
        "particles stayed distinct while moving, got {}",
        distinct.len()
    );
}

#[test]
fn scenario_1_redefining_a_function_lands_on_the_next_frame() {
    let (mut session, mut actor, tk) = boot();
    frame(&mut session, &mut actor);
    session.eval(include_str!("../demo/edit_1_radius.lt")).unwrap();
    let before = tk.lock().unwrap().shapes.len();
    frame(&mut session, &mut actor);
    let t = tk.lock().unwrap();
    for s in &t.shapes[before..] {
        assert_eq!(s.a, 8 + s.y / 24, "radius() v2 computed from y");
    }
}

#[test]
fn scenario_2_live_particles_migrate_to_the_new_struct_version() {
    let (mut session, mut actor, tk) = boot();
    for _ in 0..3 {
        frame(&mut session, &mut actor);
    }
    // The particles on screen were all built under Particle v1.
    let before = tk.lock().unwrap().shapes.len();
    session.eval(include_str!("../demo/edit_2_migrate.lt")).unwrap();
    frame(&mut session, &mut actor);
    let t = tk.lock().unwrap();
    assert_eq!(t.canvases_opened, 1, "no reseed, no reopen");
    for s in &t.shapes[before..] {
        assert_eq!(s.hue, 20 + s.x / 3, "migrated particle carries the defaulted hue");
    }
}

#[test]
fn scenario_3_a_breaking_edit_is_rejected_and_the_program_runs_on() {
    let (mut session, mut actor, tk) = boot();
    frame(&mut session, &mut actor);
    let err = session
        .eval(include_str!("../demo/edit_3_rejected.lt"))
        .expect_err("a str-returning tint must not install");
    assert!(!err.is_empty());
    let before = tk.lock().unwrap().shapes.len();
    frame(&mut session, &mut actor);
    let t = tk.lock().unwrap();
    assert!(t.shapes.len() > before, "the animation kept running");
    assert!(
        t.shapes[before..].iter().all(|s| s.hue == 205),
        "still drawing with the last good tint"
    );
}

#[test]
fn scenarios_4_to_6_break_freeze_and_resume_in_place() {
    let (mut session, mut actor, tk) = boot();
    frame(&mut session, &mut actor);

    // 4 — introduce the enum and dispatch on it.
    session.eval(include_str!("../demo/edit_4_enum.lt")).unwrap();
    let before = tk.lock().unwrap().shapes.len();
    frame(&mut session, &mut actor);
    {
        let t = tk.lock().unwrap();
        let drawn = &t.shapes[before..];
        // The arms differ in which foreign function they call, so the enum is
        // visible as a shape and not merely as a size. `radius` is untouched at
        // 14 and still sizes both arms.
        for s in drawn {
            let expected = if s.x > 320 { ("ring", 20) } else { ("circle", 14) };
            assert_eq!((s.kind, s.a), expected, "match dispatched on Kind");
        }
        // Both arms have to be exercised, or the check above passes vacuously.
        assert!(drawn.iter().any(|s| s.kind == "ring"), "some particle is a ring");
        assert!(drawn.iter().any(|s| s.kind == "circle"), "some particle is a disc");
    }

    // 5 — adding a variant marks the stale match Broken at install, and the
    //     running animation freezes at its next call to it.
    session.eval(include_str!("../demo/edit_5_break.lt")).unwrap();
    let (drawn_at_break, cleared_at_break) = {
        let t = tk.lock().unwrap();
        (t.shapes.len(), t.frames_cleared)
    };
    let turn = frame(&mut session, &mut actor);
    assert_eq!(turn, Turn::Paused, "the running program froze");
    let ActorStatus::Paused(_) = &actor.status else {
        panic!("expected a paused actor, got {:?}", actor.status);
    };
    // Brokenness propagated from `draw` to `on_frame`, so `main` trapped at
    // the call boundary — before the frame performed a single effect. Nothing
    // was drawn half-way; the canvas still holds the last good frame.
    {
        let t = tk.lock().unwrap();
        assert_eq!(t.frames_cleared, cleared_at_break, "no half-executed frame");
        assert_eq!(t.shapes.len(), drawn_at_break, "nothing drawn while broken");
    }
    // Frozen means frozen: stepping again makes no further progress.
    session.engine.step(&mut actor);
    {
        let t = tk.lock().unwrap();
        assert_eq!(
            (t.frames_cleared, t.shapes.len()),
            (cleared_at_break, drawn_at_break),
            "a frozen program performs no further effects"
        );
    }

    // 6 — install the missing arm and thaw. It resumes at the trapping
    //     instruction: the particles it already drew this frame are not redrawn.
    session.eval(include_str!("../demo/edit_6_repair.lt")).unwrap();
    session.engine.thaw(&mut actor);
    let turn = frame(&mut session, &mut actor);
    assert_eq!(turn, Turn::Yielded, "it finished the frame it was frozen inside");
    let t = tk.lock().unwrap();
    assert_eq!(t.canvases_opened, 1, "never reopened across the whole scenario");
    // The suspended `main` made the call it had been unable to make: exactly
    // one frame's worth of effects, from the same actor, with no restart.
    assert_eq!(
        (t.frames_cleared - cleared_at_break, t.shapes.len() - drawn_at_break),
        (1, 18),
        "resumed into exactly one frame — not restarted, not replayed"
    );
    // The repaired arm calls a third foreign function, so the repair shows up
    // as bars appearing rather than as a size change.
    assert!(
        t.shapes[drawn_at_break..]
            .iter()
            .any(|s| s.x > 430 && s.kind == "bar" && (s.a, s.b) == (42, 14)),
        "the repaired arm is live"
    );
}

#[test]
fn the_actor_survives_a_live_edit_and_never_restarts() {
    let (mut session, mut actor, tk) = boot();
    for _ in 0..3 {
        frame(&mut session, &mut actor);
    }
    let before = tk.lock().unwrap().shapes.len();
    session
        .eval("fn radius(p: Particle) -> i64 { 30 }")
        .expect("edit installs");
    frame(&mut session, &mut actor);
    let t = tk.lock().unwrap();
    assert_eq!(t.canvases_opened, 1, "the canvas was never reopened");
    assert!(matches!(actor.status, ActorStatus::Runnable), "same running actor");
    assert!(
        t.shapes[before..].iter().all(|s| s.a == 30),
        "the new radius took effect on the very next frame"
    );
}
