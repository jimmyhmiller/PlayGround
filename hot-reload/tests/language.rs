//! The surface language's newer features — strings and enums/match — proven on
//! every configuration of the one engine, including the live-evolution stories
//! they exist to exercise: adding a variant statically invalidates stale
//! matches (repair live), and removing a variant gaps its objects until a
//! per-variant migration maps them forward.

use livetype::*;
use std::collections::BTreeMap;
use std::sync::Arc;

fn both_engines() -> Vec<(&'static str, Arc<Engine>)> {
    vec![("interp", Engine::interp()), ("jit", jit_engine(0))]
}

/// Compile + run `main` on one engine, returning (outcome, output).
fn run_program(engine: &Arc<Engine>, src: &str) -> (Outcome, Vec<Value>) {
    let compiled = livetype_core::compile_on(src, Arc::clone(engine)).expect("compile");
    let outcome = compiled.engine.run_call(compiled.functions["main"], vec![]);
    (outcome, engine.output())
}

// ───────────────────────────── strings ─────────────────────────────

#[test]
fn strings_concat_compare_and_flow_identically() {
    let src = r#"
        fn greet(name: str) -> str {
            "hello, " + name
        }
        fn main() -> i64 {
            let a = greet("world");
            emit(a);
            let b = "hello, " + "world";
            if a == b { emit(1); }
            if a != "goodbye" { emit(2); }
            let n = 0;
            let banner = "";
            while n < 3 {
                banner = banner + "!";
                n = n + 1;
            }
            emit(banner);
            0
        }
    "#;
    let mut results = Vec::new();
    for (name, engine) in both_engines() {
        let (outcome, output) = run_program(&engine, src);
        assert_eq!(outcome, Outcome::Complete(Value::I64(0)), "{name}");
        assert_eq!(output.len(), 4, "{name}: expected 4 effects");
        assert_eq!(
            livetype_core::strings::text(match output[0] {
                Value::Str(id) => id,
                other => panic!("{name}: expected a string, got {other:?}"),
            })
            .as_ref(),
            "hello, world"
        );
        assert_eq!(output[1], Value::I64(1), "{name}: == on equal strings");
        assert_eq!(output[2], Value::I64(2), "{name}: != on different strings");
        assert_eq!(
            livetype_core::strings::text(match output[3] {
                Value::Str(id) => id,
                other => panic!("{name}: expected a string, got {other:?}"),
            })
            .as_ref(),
            "!!!"
        );
        results.push(output);
    }
    assert_eq!(results[0], results[1], "configurations diverged on strings");
}

#[test]
fn string_fields_survive_migration_and_letonce() {
    // A struct with a string field evolves (gains a defaulted string field);
    // the auto-derived migration copies the string; a letonce string survives
    // the edit. Strings are scalars, so they cross the barrier untouched.
    let mut s = Session::with_engine(jit_engine(0));
    s.eval(
        r#"
        struct User { name: str }
        letonce motd = "welcome";
        fn describe(u: User) -> str { u.name + ": " + motd }
        fn main() -> i64 { emit(describe(User { name: "jimmy" })); 0 }
        "#,
    )
    .unwrap();
    assert_eq!(s.call("main", vec![]).unwrap(), Value::I64(0));

    // Live edit: User gains a defaulted role; describe uses it.
    s.eval(
        r#"
        struct User { name: str, role: str = "guest" }
        fn describe(u: User) -> str { u.name + " (" + u.role + "): " + motd }
        "#,
    )
    .unwrap();
    assert_eq!(s.call("main", vec![]).unwrap(), Value::I64(0));

    let out = s.engine.output();
    let text = |v: &Value| match v {
        Value::Str(id) => livetype_core::strings::text(*id).to_string(),
        other => panic!("expected a string, got {other:?}"),
    };
    assert_eq!(text(&out[0]), "jimmy: welcome");
    assert_eq!(text(&out[1]), "jimmy (guest): welcome");
}

// ───────────────────────────── enums + match ─────────────────────────────

const SHAPES: &str = r#"
    enum Shape {
        Circle { r: i64 },
        Rect { w: i64, h: i64 },
        Point,
    }
    fn area(s: Shape) -> i64 {
        let result = 0;
        match s {
            Circle { r } => { result = 3 * r * r; }
            Rect { w, h } => { result = w * h; }
            Point => { result = 0; }
        }
        result
    }
    fn main() -> i64 {
        emit(area(Shape::Circle { r: 2 }));
        emit(area(Shape::Rect { w: 3, h: 4 }));
        emit(area(Shape::Point));
        0
    }
"#;

#[test]
fn enums_construct_and_match_identically() {
    let mut results = Vec::new();
    for (name, engine) in both_engines() {
        let (outcome, output) = run_program(&engine, SHAPES);
        assert_eq!(outcome, Outcome::Complete(Value::I64(0)), "{name}");
        assert_eq!(
            output,
            vec![Value::I64(12), Value::I64(12), Value::I64(0)],
            "{name}: match arms computed wrong"
        );
        results.push(output);
    }
    assert_eq!(results[0], results[1]);
}

#[test]
fn non_exhaustive_match_is_rejected_at_install() {
    let src = r#"
        enum Shape { Circle { r: i64 }, Point }
        fn bad(s: Shape) -> i64 {
            match s {
                Circle { r } => { return r; }
            }
            0
        }
        fn main() -> i64 { 0 }
    "#;
    let err = livetype_core::compile(src).map(|_| ()).unwrap_err();
    assert!(
        err.contains("non-exhaustive"),
        "expected a non-exhaustiveness error, got: {err}"
    );
}

/// THE enum live-evolution story, part 1: adding a variant. The verifier's
/// exhaustiveness rule makes every stale match Broken at install (D7), so the
/// running program traps at its next call — never silently falls through an
/// arm — and a live re-eval of the match repairs and resumes it.
#[test]
fn adding_a_variant_invalidates_stale_matches_then_repair_resumes() {
    for (name, engine) in both_engines() {
        let mut s = Session::with_engine(engine);
        s.eval(
            r#"
            enum Shape { Circle { r: i64 }, Point }
            fn area(s: Shape) -> i64 {
                let result = 0;
                match s {
                    Circle { r } => { result = 3 * r * r; }
                    Point => { result = 0; }
                }
                result
            }
            fn main() -> i64 {
                let i = 0;
                while i < 4 {
                    emit(area(Shape::Circle { r: 1 }));
                    yield;
                    i = i + 1;
                }
                0
            }
            "#,
        )
        .unwrap();
        let main = s.fn_id("main").unwrap();
        let area = s.fn_id("area").unwrap();
        let mut actor = s.engine.spawn(main, vec![]).unwrap();

        // Run to the first yield: one area computed.
        loop {
            match s.engine.step(&mut actor) {
                Turn::Progress => {}
                _ => break,
            }
        }
        assert_eq!(s.engine.output(), vec![Value::I64(3)], "{name}");

        // LIVE EDIT: Shape gains a Rect variant. `area`'s match no longer
        // covers the enum, so the edit re-verifies it to Broken...
        s.eval(r#"enum Shape { Circle { r: i64 }, Rect { w: i64, h: i64 }, Point }"#)
            .unwrap();
        let area_broken = s.engine.with_world(|w| {
            let v = w.current_functions[&area];
            matches!(w.functions[&(area, v)], FunctionState::Broken { .. })
        });
        assert!(area_broken, "{name}: stale match must be invalidated");

        // ...and the RUNNING loop traps at its next call to `area`.
        let outcome = s.engine.run(&mut actor);
        assert!(
            matches!(outcome, Outcome::Paused(Condition::BrokenFunction { function, .. }) if function == area),
            "{name}: expected the stale match to trap, got {outcome:?}"
        );

        // REPAIR LIVE: re-eval `area` with the new arm; the same actor resumes
        // and finishes, computing with the repaired match.
        s.eval(
            r#"
            fn area(s: Shape) -> i64 {
                let result = 0;
                match s {
                    Circle { r } => { result = 3 * r * r; }
                    Rect { w, h } => { result = w * h; }
                    Point => { result = 0; }
                }
                result
            }
            "#,
        )
        .unwrap();
        assert_eq!(s.engine.resume(&mut actor), Outcome::Complete(Value::I64(0)), "{name}");
        assert_eq!(
            s.engine.output(),
            vec![Value::I64(3); 4],
            "{name}: the resumed loop finished its remaining iterations"
        );

        // The new variant is immediately constructible and matchable.
        s.eval(r#"fn rect_area() -> i64 { area(Shape::Rect { w: 5, h: 6 }) }"#)
            .unwrap();
        assert_eq!(s.call("rect_area", vec![]).unwrap(), Value::I64(30), "{name}");
    }
}

/// THE enum live-evolution story, part 2: removing a variant. Live objects of
/// the removed variant gap individually (MissingMigration at their next use —
/// sibling variants migrate transparently via the auto-derived identity
/// mappings), until an explicit per-variant migration maps them forward; the
/// merge semantics keep the auto-derived survivor mappings intact.
#[test]
fn removing_a_variant_gaps_its_objects_until_a_variant_migration() {
    for (name, engine) in both_engines() {
        let mut s = Session::with_engine(engine);
        s.eval(
            r#"
            enum Status {
                Active { level: i64 },
                Legacy { code: i64 },
            }
            letonce old_user = Status::Legacy { code: 7 };
            letonce new_user = Status::Active { level: 2 };
            fn rank(s: Status) -> i64 {
                let result = 0;
                match s {
                    Active { level } => { result = level; }
                    Legacy { code } => { result = 0 - code; }
                }
                result
            }
            fn check_old() -> i64 { rank(old_user) }
            fn check_new() -> i64 { rank(new_user) }
            fn main() -> i64 { 0 }
            "#,
        )
        .unwrap();
        assert_eq!(s.call("check_old", vec![]).unwrap(), Value::I64(-7), "{name}");
        assert_eq!(s.call("check_new", vec![]).unwrap(), Value::I64(2), "{name}");

        // LIVE EDIT: Legacy is removed; rank is re-evaled to match the new
        // shape in the same edit (otherwise it would be Broken).
        s.eval(
            r#"
            enum Status { Active { level: i64 } }
            fn rank(s: Status) -> i64 {
                let result = 0;
                match s {
                    Active { level } => { result = level; }
                }
                result
            }
            "#,
        )
        .unwrap();

        // The surviving variant's object migrated transparently (auto-derived
        // identity mapping)...
        assert_eq!(s.call("check_new", vec![]).unwrap(), Value::I64(2), "{name}");
        // ...but the Legacy object is GAPPED: using it traps MissingMigration.
        let err = s.call("check_old", vec![]).unwrap_err();
        assert!(
            matches!(err, Condition::MissingMigration { .. }),
            "{name}: a removed variant's object must gap, got {err:?}"
        );

        // REPAIR: map Legacy{code} forward to Active{level: 1} (a per-variant
        // migration; Copy is impossible since the field changed meaning, so
        // supply a value). Ids come from the world.
        let status = s.struct_id("Status").unwrap();
        let (from_version, legacy_id, active_id, level_field) = s.engine.with_world(|w| {
            let current = w.current_schemas[&status];
            let old_schema = &w.schemas[&(status, Version(current.0 - 1))];
            let new_schema = &w.schemas[&(status, current)];
            let legacy = old_schema
                .variants
                .iter()
                .find(|v| v.name == "Legacy")
                .expect("old schema has Legacy");
            let active = new_schema
                .variants
                .iter()
                .find(|v| v.name == "Active")
                .expect("new schema has Active");
            (
                Version(current.0 - 1),
                legacy.id,
                active.id,
                active.fields[0].id,
            )
        });
        s.engine
            .install_migration(Migration {
                type_id: status,
                from: from_version,
                to: Version(from_version.0 + 1),
                fields: BTreeMap::new(),
                variants: BTreeMap::from([(
                    legacy_id,
                    VariantMigration {
                        to_variant: active_id,
                        fields: BTreeMap::from([(level_field, MigrationSource::Value(Value::I64(1)))]),
                    },
                )]),
            })
            .unwrap();

        // The gapped object now migrates Legacy → Active{level: 1} at its next
        // use, and BOTH objects keep working (merge preserved the survivor
        // mapping).
        assert_eq!(s.call("check_old", vec![]).unwrap(), Value::I64(1), "{name}");
        assert_eq!(s.call("check_new", vec![]).unwrap(), Value::I64(2), "{name}");
    }
}

/// A promoted (JIT) match keeps working across a schema evolution that adds a
/// defaulted field to a variant: the object migrates lazily at the match
/// barrier (`lt_case_variant` runs the same `Heap::variant_case`), invisible
/// to the compiled code.
#[test]
fn variant_field_addition_is_transparent_to_a_hot_match() {
    let mut s = Session::with_engine(jit_engine(2));
    s.eval(
        r#"
        enum Msg { Ping { n: i64 } }
        letonce first = Msg::Ping { n: 41 };
        fn read(m: Msg) -> i64 {
            let result = 0;
            match m {
                Ping { n } => { result = n; }
            }
            result
        }
        fn main() -> i64 {
            let i = 0;
            let last = 0;
            while i < 10 {
                last = read(first);
                i = i + 1;
            }
            last
        }
        "#,
    )
    .unwrap();
    let read = s.fn_id("read").unwrap();
    assert_eq!(s.call("main", vec![]).unwrap(), Value::I64(41));
    assert!(s.engine.is_hot(read, Version(1)), "read must be promoted");

    // Evolve: Ping gains a defaulted field (auto-derived identity migration)
    // and read v2 uses it. The letonce object crosses the barrier lazily.
    s.eval(
        r#"
        enum Msg { Ping { n: i64, hops: i64 = 5 } }
        fn read(m: Msg) -> i64 {
            let result = 0;
            match m {
                Ping { n, hops } => { result = n + hops; }
            }
            result
        }
        "#,
    )
    .unwrap();
    assert_eq!(s.call("main", vec![]).unwrap(), Value::I64(46));
}

// ───────────────────── floats, division, unary minus ─────────────────────

#[test]
fn floats_division_and_negation_run_identically() {
    let src = r#"
        fn norm(x: f64, y: f64) -> f64 { (x * x + y * y) / 2.0 }
        fn main() -> i64 {
            let a = norm(3.0, 4.0);       // 12.5
            emit(a);
            if a > 12.0 { emit(1); }
            if a <= 12.5 { emit(2); }
            if -a < 0.0 { emit(3); }
            let q = 17 / 5;               // integer division
            emit(q);
            emit(-q);
            0
        }
    "#;
    let mut results = Vec::new();
    for (name, engine) in both_engines() {
        let (outcome, output) = run_program(&engine, src);
        assert_eq!(outcome, Outcome::Complete(Value::I64(0)), "{name}");
        assert_eq!(
            output,
            vec![
                Value::F64(12.5),
                Value::I64(1),
                Value::I64(2),
                Value::I64(3),
                Value::I64(3),
                Value::I64(-3),
            ],
            "{name}"
        );
        results.push(output);
    }
    assert_eq!(results[0], results[1]);
}

#[test]
fn division_by_zero_traps_and_resumes_identically() {
    let src = r#"
        fn main(d: i64) -> i64 { 10 / d }
    "#;
    for (name, engine) in both_engines() {
        let compiled = livetype_core::compile_on(src, Arc::clone(&engine)).unwrap();
        let main = compiled.functions["main"];
        let mut actor = engine.spawn(main, vec![Value::I64(0)]).unwrap();
        let outcome = engine.run(&mut actor);
        assert!(
            matches!(
                &outcome,
                Outcome::Paused(Condition::RuntimeTypeError { message, .. })
                    if message == "division by zero"
            ),
            "{name}: expected a division-by-zero trap, got {outcome:?}"
        );
        // The trap is a one-shot continuation: supply the quotient, resume.
        engine.resume_with(&mut actor, Value::I64(99)).unwrap();
        assert_eq!(engine.run(&mut actor), Outcome::Complete(Value::I64(99)), "{name}");
    }
}

// ───────────────────────────── arrays ─────────────────────────────

#[test]
fn arrays_build_index_mutate_grow_identically() {
    let src = r#"
        fn sum(xs: [i64]) -> i64 {
            let total = 0;
            let i = 0;
            while i < len(xs) {
                total = total + xs[i];
                i = i + 1;
            }
            total
        }
        fn main() -> i64 {
            let xs = [10, 20, 30];
            emit(sum(xs));            // 60
            xs[1] = 25;
            emit(sum(xs));            // 65
            push(xs, 5);
            emit(sum(xs));            // 70
            emit(len(xs));            // 4
            let empty: [i64] = [];
            emit(len(empty));         // 0
            let names = ["a", "b"];
            emit(names[0] + names[1]);
            sum(xs)
        }
    "#;
    let mut results = Vec::new();
    for (name, engine) in both_engines() {
        let (outcome, output) = run_program(&engine, src);
        assert_eq!(outcome, Outcome::Complete(Value::I64(70)), "{name}");
        assert_eq!(&output[..5], &[
            Value::I64(60),
            Value::I64(65),
            Value::I64(70),
            Value::I64(4),
            Value::I64(0),
        ], "{name}");
        assert_eq!(
            livetype_core::strings::text(match output[5] {
                Value::Str(id) => id,
                other => panic!("{name}: expected str, got {other:?}"),
            })
            .as_ref(),
            "ab"
        );
        results.push(output);
    }
    assert_eq!(results[0], results[1]);
}

#[test]
fn array_out_of_bounds_traps_and_resumes() {
    let src = "fn main(i: i64) -> i64 { let xs = [1, 2]; xs[i] }";
    for (name, engine) in both_engines() {
        let compiled = livetype_core::compile_on(src, Arc::clone(&engine)).unwrap();
        let main = compiled.functions["main"];
        let mut actor = engine.spawn(main, vec![Value::I64(7)]).unwrap();
        let outcome = engine.run(&mut actor);
        assert!(
            matches!(
                &outcome,
                Outcome::Paused(Condition::RuntimeTypeError { message, .. })
                    if message.contains("out of bounds")
            ),
            "{name}: expected a bounds trap, got {outcome:?}"
        );
    }
}

#[test]
fn arrays_of_structs_ride_the_migration_barrier() {
    // Array elements are references; the ARRAY never migrates, but the structs
    // it holds do — lazily, at their next field access, even from a compiled
    // frame.
    let mut s = Session::with_engine(jit_engine(0));
    s.eval(
        r#"
        struct Point { x: i64 }
        letonce points = [Point { x: 1 }, Point { x: 2 }];
        fn total() -> i64 {
            let t = 0;
            let i = 0;
            while i < len(points) {
                t = t + points[i].x;
                i = i + 1;
            }
            t
        }
        fn main() -> i64 { total() }
        "#,
    )
    .unwrap();
    assert_eq!(s.call("main", vec![]).unwrap(), Value::I64(3));

    // Evolve Point (defaulted field) and use it — the live elements migrate.
    s.eval(
        r#"
        struct Point { x: i64, weight: i64 = 10 }
        fn total() -> i64 {
            let t = 0;
            let i = 0;
            while i < len(points) {
                t = t + points[i].x * points[i].weight;
                i = i + 1;
            }
            t
        }
        "#,
    )
    .unwrap();
    assert_eq!(s.call("main", vec![]).unwrap(), Value::I64(30));
}

// ───────────────────────── match as an expression ─────────────────────────

#[test]
fn match_expression_yields_values_identically() {
    let src = r#"
        enum Shape { Circle { r: i64 }, Rect { w: i64, h: i64 }, Point }
        fn area(s: Shape) -> i64 {
            match s {
                Circle { r } => 3 * r * r,
                Rect { w, h } => w * h,
                Point => 0,
            }
        }
        fn main() -> i64 {
            let total = area(Shape::Circle { r: 2 }) + area(Shape::Rect { w: 2, h: 5 });
            emit(total);
            total
        }
    "#;
    for (name, engine) in both_engines() {
        let (outcome, output) = run_program(&engine, src);
        assert_eq!(outcome, Outcome::Complete(Value::I64(22)), "{name}");
        assert_eq!(output, vec![Value::I64(22)], "{name}");
    }
}

// ─────────────────────── first-class function values ───────────────────────

#[test]
fn function_values_call_indirectly_identically() {
    let src = r#"
        fn double(x: i64) -> i64 { x * 2 }
        fn triple(x: i64) -> i64 { x * 3 }
        fn apply(f: fn(i64) -> i64, x: i64) -> i64 { f(x) }
        fn main() -> i64 {
            let ops = [double, triple];
            let t = apply(ops[0], 10) + apply(ops[1], 10);
            emit(t);
            t
        }
    "#;
    for (name, engine) in both_engines() {
        let (outcome, output) = run_program(&engine, src);
        assert_eq!(outcome, Outcome::Complete(Value::I64(50)), "{name}");
        assert_eq!(output, vec![Value::I64(50)], "{name}");
    }
}

/// THE function-value hot-reload story: a function value is a NAME, not a code
/// pointer. A stored reference (here: living in a letonce array across edits)
/// picks up a live redefinition at its next call — no re-registration, no
/// stale code.
#[test]
fn stored_function_values_late_bind_to_live_edits() {
    for (name, engine) in both_engines() {
        let mut s = Session::with_engine(engine);
        s.eval(
            r#"
            fn handler(x: i64) -> i64 { x + 1 }
            letonce handlers = [handler];
            fn dispatch(x: i64) -> i64 {
                let f = handlers[0];
                f(x)
            }
            fn main() -> i64 { dispatch(10) }
            "#,
        )
        .unwrap();
        assert_eq!(s.call("main", vec![]).unwrap(), Value::I64(11), "{name}");

        // LIVE EDIT: redefine handler. The value stored in the letonce array
        // was never touched — but it names the function, so the next dispatch
        // runs the new version.
        s.eval("fn handler(x: i64) -> i64 { x + 100 }").unwrap();
        assert_eq!(s.call("main", vec![]).unwrap(), Value::I64(110), "{name}");

        // A BREAKING edit to the referenced function traps the indirect call
        // exactly like a direct one (late binding, same checks)...
        s.eval("fn handler(x: i64) -> bool { x == 0 }").unwrap();
        let err = s.call("main", vec![]).unwrap_err();
        assert!(
            matches!(err, Condition::RuntimeTypeError { .. } | Condition::BrokenFunction { .. }),
            "{name}: expected the indirect call to trap, got {err:?}"
        );
        // ...and repairing it repairs every stored reference at once.
        s.eval("fn handler(x: i64) -> i64 { x * 2 }").unwrap();
        assert_eq!(s.call("main", vec![]).unwrap(), Value::I64(20), "{name}");
    }
}
