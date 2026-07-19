//! Live editing: a running program is changed by `eval`-ing new source into the
//! same session, and the change takes effect on the in-flight computation.

use livetype_core::*;

/// Step the engine until the actor stops or has just crossed a `Yield`.
fn to_yield(engine: &Engine, actor: &mut Actor) {
    loop {
        match engine.step(actor) {
            Turn::Progress => {}
            Turn::Yielded | Turn::Done | Turn::Paused => break,
            Turn::Blocked => panic!("nothing sends to this actor"),
        }
    }
}

#[test]
fn a_live_edit_changes_a_running_program() {
    let mut s = Session::new();
    s.eval(
        r#"
        struct Account { balance: i64 }
        fn report(a: Account) -> i64 { a.balance }
        fn main() -> i64 {
            let a = Account { balance: 100 };
            let x = report(a); emit(x); yield;
            let y = report(a); emit(y); yield;
            report(a)
        }
    "#,
    )
    .unwrap();

    let main = s.fn_id("main").unwrap();
    let mut actor = s.engine.spawn(main, vec![]).unwrap();

    // Run to the first yield: reports the original balance.
    to_yield(&s.engine, &mut actor);
    assert_eq!(s.engine.output(), vec![Value::I64(100)]);

    // LIVE EDIT: give Account a defaulted `fee` and make `report` subtract it.
    // The schema change auto-derives its migration (copy balance, default fee);
    // the function change is picked up by the next call.
    s.eval(
        r#"
        struct Account { balance: i64, fee: i64 = 10 }
        fn report(a: Account) -> i64 { a.balance - a.fee }
    "#,
    )
    .unwrap();

    // Resume: the running `main` now calls the new `report` over the migrated
    // account — 100 - 10 = 90 — for the rest of its life.
    assert_eq!(s.engine.run(&mut actor), Outcome::Complete(Value::I64(90)));
    // Emitted the original 100, then 90 after the edit (the final `report(a)` is
    // the return value, not emitted).
    assert_eq!(s.engine.output(), vec![Value::I64(100), Value::I64(90)]);
}

#[test]
fn a_breaking_edit_traps_and_a_fix_resumes() {
    let mut s = Session::new();
    s.eval(
        r#"
        struct Account { balance: i64 }
        fn charge(a: Account, amt: i64) -> i64 { a.balance - amt }
        fn main() -> i64 {
            let a = Account { balance: 100 };
            emit(a.balance);
            yield;
            charge(a, 5)
        }
    "#,
    )
    .unwrap();
    let main = s.fn_id("main").unwrap();
    let mut actor = s.engine.spawn(main, vec![]).unwrap();
    to_yield(&s.engine, &mut actor);

    // BREAKING EDIT: `balance` becomes a `Money` struct. `charge` no longer
    // type-checks (`Money - i64`) and is republished Broken; reaching it traps.
    s.eval(
        r#"
        struct Money { cents: i64 }
        struct Account { balance: Money }
    "#,
    )
    .unwrap();
    let outcome = s.engine.run(&mut actor);
    assert!(
        matches!(outcome, Outcome::Paused(Condition::BrokenFunction { .. })),
        "reaching the broken charge should trap, got {outcome:?}"
    );

    // REPAIR (from source): a Money-aware `charge`. It type-checks, so the
    // pinned frame resumes into the repaired call...
    s.eval(r#" fn charge(a: Account, amt: i64) -> i64 { a.balance.cents - amt } "#)
        .unwrap();
    // ...and now it needs the Int→Money migration, which the developer supplies.
    assert!(matches!(
        s.engine.resume(&mut actor),
        Outcome::Paused(Condition::MissingMigration { .. })
    ));
    let money = s.struct_id("Money").unwrap();
    // Wrap the old Int balance into a Money{cents}. (Migration transformers are
    // supplied through the runtime API; a migration surface syntax is future.)
    let bal_field = field_id(&s, "Account", "balance");
    let cents_field = field_id(&s, "Money", "cents");
    s.engine
        .install_migration(Migration {
            type_id: s.struct_id("Account").unwrap(),
            from: Version(1),
            to: Version(2),
            fields: std::collections::BTreeMap::from([(
                bal_field,
                MigrationSource::Wrap { type_id: money, field: cents_field, source: bal_field },
            )]),
            variants: std::collections::BTreeMap::new(),
        })
        .unwrap();
    assert_eq!(s.engine.resume(&mut actor), Outcome::Complete(Value::I64(95)));
}

/// Read a field id out of a running schema by name (test helper).
fn field_id(s: &Session, struct_name: &str, field_name: &str) -> FieldId {
    let tid = s.struct_id(struct_name).unwrap();
    s.engine.with_world(|w| {
        let v = w.current_schemas[&tid];
        w.schemas[&(tid, v)]
            .fields
            .iter()
            .find(|f| f.name == field_name)
            .unwrap()
            .id
    })
}

#[test]
fn edit_a_tight_loop_with_no_yields_between_steps() {
    // A hot, active loop — NO `yield` anywhere in it.
    let mut s = Session::new();
    s.eval(
        r#"
        struct Sensor { reading: i64 }
        fn read(x: Sensor) -> i64 { x.reading }
        fn main() -> i64 {
            let x = Sensor { reading: 42 };
            let i = 0;
            while i < 8 {
                emit(read(x));
                i = i + 1;
            }
            0
        }
    "#,
    )
    .unwrap();
    let main = s.fn_id("main").unwrap();
    let mut actor = s.engine.spawn(main, vec![]).unwrap();

    // Advance the loop a few iterations by stepping — it is genuinely running,
    // not parked at any safe point.
    while s.engine.output().len() < 3 {
        s.engine.step(&mut actor);
    }

    // Live-edit `read` between two steps of the running loop. No yield needed.
    s.eval("fn read(x: Sensor) -> i64 { x.reading + 100 }").unwrap();

    // Let it finish. Subsequent iterations call the NEW `read`.
    s.engine.run(&mut actor);

    let out = s.engine.output();
    assert_eq!(out[0], Value::I64(42), "early iterations ran the old code");
    assert_eq!(*out.last().unwrap(), Value::I64(142), "later iterations ran the edit");
    assert!(out.contains(&Value::I64(42)) && out.contains(&Value::I64(142)), "switched mid-loop");
}

#[test]
fn a_repair_revives_transitive_callers_to_a_fixpoint() {
    let mut s = Session::new();
    s.eval(
        r#"
        enum Kind { A, B }
        fn leaf(k: Kind) -> i64 { match k { A => 1, B => 2 } }
        fn mid(k: Kind) -> i64 { leaf(k) + 10 }
        fn top() -> i64 {
            let k = Kind::A;
            mid(k) + 100
        }
    "#,
    )
    .unwrap();
    assert_eq!(s.call("top", vec![]).unwrap(), Value::I64(111));

    // Growing the enum breaks `leaf`'s match, and brokenness propagates up the
    // chain: `mid` and `top` go Broken too.
    s.eval("enum Kind { A, B, C }").unwrap();
    assert!(s.call("top", vec![]).is_err(), "the whole chain is broken");

    // Repair ONLY the root cause. `mid` and `top` are untouched — they were
    // never wrong — so they must come back on their own.
    s.eval("fn leaf(k: Kind) -> i64 { match k { A => 1, B => 2, C => 3 } }")
        .unwrap();
    assert_eq!(
        s.call("top", vec![]).unwrap(),
        Value::I64(111),
        "fixing the callee revived its transitive callers"
    );
}

#[test]
fn a_function_broken_on_its_own_merits_is_not_revived() {
    let mut s = Session::new();
    s.eval(
        r#"
        enum Kind { A, B }
        fn leaf(k: Kind) -> i64 { match k { A => 1, B => 2 } }
        fn top() -> i64 {
            let k = Kind::A;
            leaf(k)
        }
    "#,
    )
    .unwrap();
    s.eval("enum Kind { A, B, C }").unwrap();
    assert!(s.call("top", vec![]).is_err());

    // An unrelated good install triggers revalidation, but `leaf` still fails
    // verification on its own terms, so it — and its caller — stay broken.
    s.eval("fn unrelated() -> i64 { 7 }").unwrap();
    assert!(
        s.call("top", vec![]).is_err(),
        "a genuinely non-exhaustive match is not silently revived"
    );
    assert_eq!(s.call("unrelated", vec![]).unwrap(), Value::I64(7));
}
