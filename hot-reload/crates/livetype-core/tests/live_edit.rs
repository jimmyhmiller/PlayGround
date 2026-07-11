//! Live editing: a running program is changed by `eval`-ing new source into the
//! same session, and the change takes effect on the in-flight computation.

use livetype_core::*;

/// Step the interpreter until the actor stops or has just executed a `Yield`.
fn to_yield(rt: &mut Runtime, actor: ActorId) {
    while matches!(rt.actors[&actor].status, ActorStatus::Runnable) {
        let (key, pc) = {
            let f = rt.actors[&actor].frames.last().unwrap();
            (f.function, f.pc)
        };
        let was_yield = matches!(
            &rt.world.functions[&key],
            FunctionState::Ready(f) if matches!(f.code[pc], Instruction::Yield)
        );
        rt.step(actor);
        if was_yield {
            break;
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
    let actor = s.runtime.spawn(main, vec![]).unwrap();

    // Run to the first yield: reports the original balance.
    to_yield(&mut s.runtime, actor);
    assert_eq!(s.runtime.output, vec![Value::I64(100)]);

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
    s.runtime.run();
    assert_eq!(
        s.runtime.actors[&actor].status,
        ActorStatus::Complete(Value::I64(90))
    );
    // Emitted the original 100, then 90 after the edit (the final `report(a)` is
    // the return value, not emitted).
    assert_eq!(s.runtime.output, vec![Value::I64(100), Value::I64(90)]);
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
    let actor = s.runtime.spawn(main, vec![]).unwrap();
    to_yield(&mut s.runtime, actor);

    // BREAKING EDIT: `balance` becomes a `Money` struct. `charge` no longer
    // type-checks (`Money - i64`) and is republished Broken; reaching it traps.
    s.eval(
        r#"
        struct Money { cents: i64 }
        struct Account { balance: Money }
    "#,
    )
    .unwrap();
    s.runtime.run();
    assert!(
        matches!(
            s.runtime.actors[&actor].status,
            ActorStatus::Paused(Condition::BrokenFunction { .. })
        ),
        "reaching the broken charge should trap, got {:?}",
        s.runtime.actors[&actor].status
    );

    // REPAIR (from source): a Money-aware `charge`. It type-checks, so the
    // pinned frame resumes into the repaired call...
    s.eval(r#" fn charge(a: Account, amt: i64) -> i64 { a.balance.cents - amt } "#)
        .unwrap();
    s.runtime.run();
    // ...and now it needs the Int→Money migration, which the developer supplies.
    assert!(matches!(
        s.runtime.actors[&actor].status,
        ActorStatus::Paused(Condition::MissingMigration { .. })
    ));
    let account = ObjectId::from(1u64);
    let money = s.struct_id("Money").unwrap();
    // Wrap the old Int balance into a Money{cents}. (Migration transformers are
    // supplied through the runtime API; a migration surface syntax is future.)
    let bal_field = field_id(&s, "Account", "balance");
    let cents_field = field_id(&s, "Money", "cents");
    s.runtime
        .install_migration(Migration {
            type_id: s.struct_id("Account").unwrap(),
            from: Version(1),
            to: Version(2),
            fields: std::collections::BTreeMap::from([(
                bal_field,
                MigrationSource::Wrap { type_id: money, field: cents_field, source: bal_field },
            )]),
        })
        .unwrap();
    s.runtime.run();
    assert_eq!(
        s.runtime.actors[&actor].status,
        ActorStatus::Complete(Value::I64(95))
    );
    let _ = account;
}

/// Read a field id out of a running schema by name (test helper).
fn field_id(s: &Session, struct_name: &str, field_name: &str) -> FieldId {
    let tid = s.struct_id(struct_name).unwrap();
    let v = s.runtime.world.current_schemas[&tid];
    s.runtime.world.schemas[&(tid, v)]
        .fields
        .iter()
        .find(|f| f.name == field_name)
        .unwrap()
        .id
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
    let a = s.runtime.spawn(main, vec![]).unwrap();

    // Advance the loop a few iterations by stepping — it is genuinely running,
    // not parked at any safe point.
    while s.runtime.output.len() < 3 {
        s.runtime.step(a);
    }

    // Live-edit `read` between two steps of the running loop. No yield needed.
    s.eval("fn read(x: Sensor) -> i64 { x.reading + 100 }").unwrap();

    // Let it finish. Subsequent iterations call the NEW `read`.
    s.runtime.run();

    let out = &s.runtime.output;
    assert_eq!(out[0], Value::I64(42), "early iterations ran the old code");
    assert_eq!(*out.last().unwrap(), Value::I64(142), "later iterations ran the edit");
    assert!(out.contains(&Value::I64(42)) && out.contains(&Value::I64(142)), "switched mid-loop");
}
