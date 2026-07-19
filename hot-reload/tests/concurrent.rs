//! Multi-actor quarantine — the design's "highest-value next experiment"
//! (RUNTIME_DESIGN.md §7 corner 1). Several actors share one heap; the question
//! is whether the soundness invariant survives when a value is migrated
//! mid-flight while another actor holds it. These run *semantic* concurrency:
//! actors interleaved on one thread at `Yield` granularity (via `Engine::step`)
//! — the setting that stresses quarantine, without OS-thread data races to
//! confound it. The result: the invariant holds with no world-freeze and no
//! ownership rule, because soundness is enforced at each value *use* regardless
//! of which actor triggered the migration. Run on the always-JIT configuration,
//! so the checks proven here are the native ones.

use livetype::*;
use std::collections::BTreeMap;
use std::sync::Arc;

fn field(id: FieldId, name: &str, ty: Type) -> Field {
    Field {
        id,
        name: name.into(),
        ty,
        default: None,
    }
}

/// Advance one actor until it crosses a `Yield`, pauses, or completes — the
/// interleaving quantum.
fn to_yield(engine: &Engine, actor: &mut Actor) {
    loop {
        match engine.step(actor) {
            Turn::Progress => {}
            Turn::Yielded | Turn::Done | Turn::Paused => break,
            Turn::Blocked => panic!("nothing sends to these actors"),
        }
    }
}

/// Round-robin actors at `Yield` granularity until none is runnable.
fn run_interleaved(engine: &Engine, actors: &mut [Actor]) {
    loop {
        let mut progressed = false;
        for actor in actors.iter_mut() {
            if matches!(actor.status, ActorStatus::Runnable) {
                to_yield(engine, actor);
                progressed = true;
            }
        }
        if !progressed {
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// Baseline: two actors sharing one heap object, interleaved to completion.
// ---------------------------------------------------------------------------

const BOXT: DefId = 1;
const READER: DefId = 20;
const VAL: FieldId = 100;

#[test]
fn shared_heap_two_actors_both_complete() {
    let engine: Arc<Engine> = jit_engine(0);
    engine
        .install_schema(Schema {
            type_id: BOXT,
            version: Version(1),
            name: "Box".into(),
            fields: vec![field(VAL, "value", Type::I64)],
            variants: Vec::new(),
        })
        .unwrap();
    // reader(x) yields once (so the scheduler interleaves), then reads x.value.
    engine
        .install_function(Function {
            id: READER,
            version: Version(1),
            name: "reader".into(),
            params: vec![Type::Ref(BOXT)],
            result: Type::I64,
            registers: 2,
            code: vec![
                Instruction::Yield,
                Instruction::GetField {
                    dst: 1,
                    object: 0,
                    field: VAL,
                },
                Instruction::Return { value: 1 },
            ],
        })
        .unwrap();

    let shared = engine.shared().jit_new(BOXT, &[(VAL, Value::I64(7))]).unwrap();
    let mut actors = vec![
        engine.spawn(READER, vec![Value::Ref(shared)]).unwrap(),
        engine.spawn(READER, vec![Value::Ref(shared)]).unwrap(),
    ];

    run_interleaved(&engine, &mut actors);

    assert_eq!(actors[0].status, ActorStatus::Complete(Value::I64(7)));
    assert_eq!(actors[1].status, ActorStatus::Complete(Value::I64(7)));
    // One shared object, reachable from no live frame now (both actors done).
    assert_eq!(engine.shared().object_count(), 1);
    assert_eq!(engine.collect(&[&actors[0], &actors[1]]), 1);
    assert_eq!(engine.shared().object_count(), 0);
}

// ---------------------------------------------------------------------------
// The experiment: actor X migrates a SHARED object out from under actor Y,
// which is a pinned old reader. Does quarantine hold?
// ---------------------------------------------------------------------------

const ACCT: DefId = 1;
const MONEY: DefId = 2;
const PEEK_INT: DefId = 30; // pinned old reader: balance is Int
const PEEK_MONEY: DefId = 31; // new reader: balance is Money
const BAL: FieldId = 100;
const CENTS: FieldId = 200;

#[test]
fn concurrent_migration_quarantines_the_pinned_reader() {
    let engine: Arc<Engine> = jit_engine(0);
    engine
        .install_schema(Schema {
            type_id: ACCT,
            version: Version(1),
            name: "Account".into(),
            fields: vec![field(BAL, "balance", Type::I64)],
            variants: Vec::new(),
        })
        .unwrap();
    // The pinned reader: yields, then reads balance expecting an Int.
    engine
        .install_function(Function {
            id: PEEK_INT,
            version: Version(1),
            name: "peek_int".into(),
            params: vec![Type::Ref(ACCT)],
            result: Type::I64,
            registers: 2,
            code: vec![
                Instruction::Yield,
                Instruction::GetField {
                    dst: 1,
                    object: 0,
                    field: BAL,
                },
                Instruction::Return { value: 1 },
            ],
        })
        .unwrap();

    let shared = engine.shared().jit_new(ACCT, &[(BAL, Value::I64(100))]).unwrap();
    let mut y = engine.spawn(PEEK_INT, vec![Value::Ref(shared)]).unwrap();

    // Y runs up to its yield — pinned at peek_int@v1, before it reads `balance`.
    to_yield(&engine, &mut y);
    assert!(matches!(y.status, ActorStatus::Runnable));
    assert_eq!(engine.shared().object_schema(shared), Some(Version(1)));

    // Hot update: balance becomes Money. peek_int is now broken, but Y's frame
    // pins the old version. Install a new reader that speaks Money, and the
    // migration.
    engine
        .install_schema(Schema {
            type_id: MONEY,
            version: Version(1),
            name: "Money".into(),
            fields: vec![field(CENTS, "cents", Type::I64)],
            variants: Vec::new(),
        })
        .unwrap();
    engine
        .install_schema(Schema {
            type_id: ACCT,
            version: Version(2),
            name: "Account".into(),
            fields: vec![field(BAL, "balance", Type::Ref(MONEY))],
            variants: Vec::new(),
        })
        .unwrap();
    engine
        .install_migration(Migration {
            type_id: ACCT,
            from: Version(1),
            to: Version(2),
            fields: BTreeMap::from([(
                BAL,
                MigrationSource::Wrap {
                    type_id: MONEY,
                    field: CENTS,
                    source: BAL,
                },
            )]),
            variants: std::collections::BTreeMap::new(),
        })
        .unwrap();
    engine
        .install_function(Function {
            id: PEEK_MONEY,
            version: Version(1),
            name: "peek_money".into(),
            params: vec![Type::Ref(ACCT)],
            result: Type::I64,
            registers: 3,
            code: vec![
                Instruction::GetField {
                    dst: 1,
                    object: 0,
                    field: BAL,
                },
                Instruction::GetField {
                    dst: 2,
                    object: 1,
                    field: CENTS,
                },
                Instruction::Return { value: 2 },
            ],
        })
        .unwrap();

    // Actor X reads the SHARED object with the new code. Its first field access
    // migrates the object v1 -> v2 in the shared heap, and X finishes cleanly.
    let mut x = engine.spawn(PEEK_MONEY, vec![Value::Ref(shared)]).unwrap();
    assert_eq!(engine.run(&mut x), Outcome::Complete(Value::I64(100)));
    assert_eq!(
        engine.shared().object_schema(shared),
        Some(Version(2)),
        "X migrated the shared object"
    );

    // Now Y resumes — pinned old code, reading the object X migrated. It reads a
    // Money reference where it expects an Int and traps on use: quarantined,
    // never observing the ill-typed value as an integer. No world-freeze was
    // needed; the per-use soundness check did the whole job.
    engine.run(&mut y);
    assert!(
        matches!(
            &y.status,
            ActorStatus::Paused(Condition::RuntimeTypeError { function, pc, .. })
                if *function == PEEK_INT && *pc == 2
        ),
        "pinned reader should trap on the migrated value, got {:?}",
        y.status
    );
    // X's result stands; the invariant held for the running actor.
    assert_eq!(x.status, ActorStatus::Complete(Value::I64(100)));
}
