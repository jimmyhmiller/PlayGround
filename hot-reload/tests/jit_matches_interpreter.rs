//! Differential tests: the always-JIT configuration must be observationally
//! indistinguishable from the never-promote (interpreter) configuration of the
//! one engine. Each scenario is set up on two identical engines, driven through
//! the same hot updates at the same execution points, and compared at every
//! observable boundary — emitted effects, pause conditions, completion value,
//! and the whole heap. A final test proves the precise GC roots correctly from
//! native frame slots.

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

fn field_with_default(id: FieldId, name: &str, ty: Type, default: Value) -> Field {
    Field {
        id,
        name: name.into(),
        ty,
        default: Some(default),
    }
}

/// Step until the actor stops being runnable OR has just crossed a `Yield` —
/// identical code for both configurations (that is the point of the engine).
fn to_yield(engine: &Engine, actor: &mut Actor) {
    loop {
        match engine.step(actor) {
            Turn::Progress => {}
            Turn::Yielded | Turn::Done | Turn::Paused => break,
            Turn::Blocked => panic!("these scenarios have no message passing"),
        }
    }
}

/// Continue a stopped-or-running actor to its next stop: re-drives a paused
/// stack (post-repair) or just runs a runnable one.
fn advance(engine: &Engine, actor: &mut Actor) {
    if matches!(actor.status, ActorStatus::Paused(_)) {
        engine.resume(actor);
    } else {
        engine.run(actor);
    }
}

/// Assert the two configurations agree on everything observable.
fn assert_same(e_i: &Engine, a_i: &Actor, e_j: &Engine, a_j: &Actor, label: &str) {
    assert_eq!(e_i.output(), e_j.output(), "{label}: emitted effects diverged");
    assert_eq!(a_i.status, a_j.status, "{label}: actor status diverged");
    assert_eq!(
        *e_i.shared().heap(),
        *e_j.shared().heap(),
        "{label}: heap diverged"
    );
}

/// The two configurations under comparison.
fn both_engines() -> (Arc<Engine>, Arc<Engine>) {
    (Engine::interp(), jit_engine(0))
}

// ===========================================================================
// Scenario 1 — Box migrated to hold a Wrapper (the tests/live_update.rs story,
// with a Yield safe point so both configurations stop at the same place).
// ===========================================================================

const BOX: DefId = 1;
const WRAPPER: DefId = 2;
const READ: DefId = 10;
const S1_MAIN: DefId = 11;
const VALUE: FieldId = 100;
const INNER: FieldId = 200;

fn scenario1_setup(rt: &Engine) {
    rt.install_schema(Schema {
        type_id: BOX,
        version: Version(1),
        name: "Box".into(),
        fields: vec![field(VALUE, "value", Type::I64)],
    })
    .unwrap();
    rt.install_function(Function {
        id: READ,
        version: Version(1),
        name: "read".into(),
        params: vec![Type::Ref(BOX)],
        result: Type::I64,
        registers: 2,
        code: vec![
            Instruction::GetField {
                dst: 1,
                object: 0,
                field: VALUE,
            },
            Instruction::Return { value: 1 },
        ],
    })
    .unwrap();
    rt.install_function(Function {
        id: S1_MAIN,
        version: Version(1),
        name: "main".into(),
        params: vec![],
        result: Type::I64,
        registers: 3,
        code: vec![
            Instruction::Const {
                dst: 0,
                value: Value::I64(42),
            },
            Instruction::New {
                dst: 1,
                type_id: BOX,
                fields: vec![(VALUE, 0)],
            },
            Instruction::Emit { value: 0 },
            Instruction::Yield,
            Instruction::Call {
                dst: 2,
                function: READ,
                args: vec![1],
            },
            Instruction::Return { value: 2 },
        ],
    })
    .unwrap();
}

fn scenario1_break(rt: &Engine) {
    rt.install_schema(Schema {
        type_id: WRAPPER,
        version: Version(1),
        name: "Wrapper".into(),
        fields: vec![field(INNER, "inner", Type::I64)],
    })
    .unwrap();
    rt.install_schema(Schema {
        type_id: BOX,
        version: Version(2),
        name: "Box".into(),
        fields: vec![field(VALUE, "value", Type::Ref(WRAPPER))],
    })
    .unwrap();
}

fn scenario1_fix_read(rt: &Engine) {
    rt.install_function(Function {
        id: READ,
        version: Version(3),
        name: "read".into(),
        params: vec![Type::Ref(BOX)],
        result: Type::I64,
        registers: 3,
        code: vec![
            Instruction::GetField {
                dst: 1,
                object: 0,
                field: VALUE,
            },
            Instruction::GetField {
                dst: 2,
                object: 1,
                field: INNER,
            },
            Instruction::Return { value: 2 },
        ],
    })
    .unwrap();
}

fn scenario1_migration(rt: &Engine) {
    rt.install_migration(Migration {
        type_id: BOX,
        from: Version(1),
        to: Version(2),
        fields: BTreeMap::from([(
            VALUE,
            MigrationSource::Wrap {
                type_id: WRAPPER,
                field: INNER,
                source: VALUE,
            },
        )]),
    })
    .unwrap();
}

#[test]
fn scenario1_box_wrapper_migration_matches() {
    let (e_i, e_j) = both_engines();
    scenario1_setup(&e_i);
    scenario1_setup(&e_j);
    let mut a_i = e_i.spawn(S1_MAIN, vec![]).unwrap();
    let mut a_j = e_j.spawn(S1_MAIN, vec![]).unwrap();

    // Phase A: run to the yield — Box allocated, effect emitted, before Call.
    to_yield(&e_i, &mut a_i);
    to_yield(&e_j, &mut a_j);
    assert_eq!(e_i.output(), vec![Value::I64(42)]);
    assert_same(&e_i, &a_i, &e_j, &a_j, "A: at yield");

    // Update: Box now holds a Wrapper — this breaks `read`.
    scenario1_break(&e_i);
    scenario1_break(&e_j);

    // Phase B: reaching the broken `read` pauses.
    advance(&e_i, &mut a_i);
    advance(&e_j, &mut a_j);
    assert!(matches!(
        a_j.status,
        ActorStatus::Paused(Condition::BrokenFunction { function: READ, .. })
    ));
    assert_same(&e_i, &a_i, &e_j, &a_j, "B: broken function");

    // Repair `read`, then hit the un-migratable value.
    scenario1_fix_read(&e_i);
    scenario1_fix_read(&e_j);
    advance(&e_i, &mut a_i);
    advance(&e_j, &mut a_j);
    assert!(matches!(
        a_j.status,
        ActorStatus::Paused(Condition::MissingMigration { .. })
    ));
    assert_same(&e_i, &a_i, &e_j, &a_j, "C: missing migration");

    // Supply the migration and resume to completion.
    scenario1_migration(&e_i);
    scenario1_migration(&e_j);
    advance(&e_i, &mut a_i);
    advance(&e_j, &mut a_j);
    assert_eq!(a_j.status, ActorStatus::Complete(Value::I64(42)));
    assert_same(&e_i, &a_i, &e_j, &a_j, "D: complete");
}

// ===========================================================================
// Scenario 2 — Account.balance: Int → Money (the src/main.rs story).
// ===========================================================================

const ACCOUNT: DefId = 1;
const MONEY: DefId = 2;
const CHARGE: DefId = 10;
const S2_MAIN: DefId = 11;
const BALANCE: FieldId = 100;
const CENTS: FieldId = 200;

fn scenario2_setup(rt: &Engine) {
    rt.install_schema(Schema {
        type_id: ACCOUNT,
        version: Version(1),
        name: "Account".into(),
        fields: vec![field(BALANCE, "balance", Type::I64)],
    })
    .unwrap();
    rt.install_function(Function {
        id: CHARGE,
        version: Version(1),
        name: "charge".into(),
        params: vec![Type::Ref(ACCOUNT), Type::I64],
        result: Type::I64,
        registers: 4,
        code: vec![
            Instruction::GetField {
                dst: 2,
                object: 0,
                field: BALANCE,
            },
            Instruction::SubI64 {
                dst: 3,
                left: 2,
                right: 1,
            },
            Instruction::Return { value: 3 },
        ],
    })
    .unwrap();
    rt.install_function(Function {
        id: S2_MAIN,
        version: Version(1),
        name: "main".into(),
        params: vec![],
        result: Type::I64,
        registers: 4,
        code: vec![
            Instruction::Const {
                dst: 0,
                value: Value::I64(100),
            },
            Instruction::New {
                dst: 1,
                type_id: ACCOUNT,
                fields: vec![(BALANCE, 0)],
            },
            Instruction::Emit { value: 0 },
            Instruction::Yield,
            Instruction::Const {
                dst: 2,
                value: Value::I64(5),
            },
            Instruction::Call {
                dst: 3,
                function: CHARGE,
                args: vec![1, 2],
            },
            Instruction::Return { value: 3 },
        ],
    })
    .unwrap();
}

fn scenario2_break(rt: &Engine) {
    rt.install_schema(Schema {
        type_id: MONEY,
        version: Version(1),
        name: "Money".into(),
        fields: vec![field(CENTS, "cents", Type::I64)],
    })
    .unwrap();
    rt.install_schema(Schema {
        type_id: ACCOUNT,
        version: Version(2),
        name: "Account".into(),
        fields: vec![field(BALANCE, "balance", Type::Ref(MONEY))],
    })
    .unwrap();
}

fn scenario2_fix_charge(rt: &Engine) {
    rt.install_function(Function {
        id: CHARGE,
        version: Version(3),
        name: "charge".into(),
        params: vec![Type::Ref(ACCOUNT), Type::I64],
        result: Type::I64,
        registers: 5,
        code: vec![
            Instruction::GetField {
                dst: 2,
                object: 0,
                field: BALANCE,
            },
            Instruction::GetField {
                dst: 3,
                object: 2,
                field: CENTS,
            },
            Instruction::SubI64 {
                dst: 4,
                left: 3,
                right: 1,
            },
            Instruction::Return { value: 4 },
        ],
    })
    .unwrap();
}

fn scenario2_migration(rt: &Engine) {
    rt.install_migration(Migration {
        type_id: ACCOUNT,
        from: Version(1),
        to: Version(2),
        fields: BTreeMap::from([(
            BALANCE,
            MigrationSource::Wrap {
                type_id: MONEY,
                field: CENTS,
                source: BALANCE,
            },
        )]),
    })
    .unwrap();
}

#[test]
fn scenario2_account_money_matches() {
    let (e_i, e_j) = both_engines();
    scenario2_setup(&e_i);
    scenario2_setup(&e_j);
    let mut a_i = e_i.spawn(S2_MAIN, vec![]).unwrap();
    let mut a_j = e_j.spawn(S2_MAIN, vec![]).unwrap();

    to_yield(&e_i, &mut a_i);
    to_yield(&e_j, &mut a_j);
    assert_eq!(e_i.output(), vec![Value::I64(100)]);
    assert_same(&e_i, &a_i, &e_j, &a_j, "A: at yield");

    scenario2_break(&e_i);
    scenario2_break(&e_j);
    advance(&e_i, &mut a_i);
    advance(&e_j, &mut a_j);
    assert!(matches!(
        a_j.status,
        ActorStatus::Paused(Condition::BrokenFunction { function: CHARGE, .. })
    ));
    assert_same(&e_i, &a_i, &e_j, &a_j, "B: broken charge");

    scenario2_fix_charge(&e_i);
    scenario2_fix_charge(&e_j);
    advance(&e_i, &mut a_i);
    advance(&e_j, &mut a_j);
    assert!(matches!(
        a_j.status,
        ActorStatus::Paused(Condition::MissingMigration { .. })
    ));
    assert_same(&e_i, &a_i, &e_j, &a_j, "C: missing migration");

    scenario2_migration(&e_i);
    scenario2_migration(&e_j);
    advance(&e_i, &mut a_i);
    advance(&e_j, &mut a_j);
    assert_eq!(a_j.status, ActorStatus::Complete(Value::I64(95)));
    assert_same(&e_i, &a_i, &e_j, &a_j, "D: complete");
}

// ===========================================================================
// Scenario 3 — a loop with a Yield each iteration, hot-updated MID-LOOP.
// Exercises back-edges, the recurring safe point (T5), and a lazy migration
// landing between iterations, all identically on both configurations.
// ===========================================================================

const LOOP: DefId = 10;
const FEE: FieldId = 101;

fn scenario3_setup(rt: &Engine) -> ObjectId {
    rt.install_schema(Schema {
        type_id: ACCOUNT,
        version: Version(1),
        name: "Account".into(),
        fields: vec![field(BALANCE, "balance", Type::I64)],
    })
    .unwrap();
    rt.install_function(Function {
        id: LOOP,
        version: Version(1),
        name: "loop_balance".into(),
        params: vec![Type::Ref(ACCOUNT), Type::I64],
        result: Type::I64,
        registers: 6,
        code: vec![
            // r0 = account, r1 = count
            Instruction::Const {
                dst: 2,
                value: Value::I64(1),
            }, // 0: one
            Instruction::Const {
                dst: 3,
                value: Value::I64(0),
            }, // 1: zero
            Instruction::GetField {
                dst: 4,
                object: 0,
                field: BALANCE,
            }, // 2: header — migration barrier each iteration
            Instruction::Emit { value: 4 }, // 3
            Instruction::Yield,             // 4: recurring safe point
            Instruction::SubI64 {
                dst: 1,
                left: 1,
                right: 2,
            }, // 5: count -= 1
            Instruction::LtI64 {
                dst: 5,
                left: 3,
                right: 1,
            }, // 6: 0 < count
            Instruction::Branch {
                cond: 5,
                then_pc: 2,
                else_pc: 8,
            }, // 7
            Instruction::Return { value: 1 }, // 8
        ],
    })
    .unwrap();
    rt.shared().jit_new(ACCOUNT, &[(BALANCE, Value::I64(100))]).unwrap()
}

/// Additive change: `Account` gains a defaulted `fee`. No migration is
/// installed — the runtime auto-derives it (copy `balance`, default `fee`)
/// because the change is trivial. This does NOT break `loop_balance` (it still
/// reads `balance: Int`); the live account migrates lazily and transparently at
/// the next field access — the hot update landing between loop iterations.
fn scenario3_add_fee(rt: &Engine) {
    rt.install_schema(Schema {
        type_id: ACCOUNT,
        version: Version(2),
        name: "Account".into(),
        fields: vec![
            field(BALANCE, "balance", Type::I64),
            field_with_default(FEE, "fee", Type::I64, Value::I64(0)),
        ],
    })
    .unwrap();
}

#[test]
fn scenario3_loop_with_midloop_update_matches() {
    let (e_i, e_j) = both_engines();
    let acct_i = scenario3_setup(&e_i);
    let acct_j = scenario3_setup(&e_j);
    assert_eq!(acct_i, acct_j, "both engines allocate the account identically");
    let mut a_i = e_i.spawn(LOOP, vec![Value::Ref(acct_i), Value::I64(3)]).unwrap();
    let mut a_j = e_j.spawn(LOOP, vec![Value::Ref(acct_j), Value::I64(3)]).unwrap();

    // First iteration.
    to_yield(&e_i, &mut a_i);
    to_yield(&e_j, &mut a_j);
    assert_eq!(e_i.output(), vec![Value::I64(100)]);
    assert_same(&e_i, &a_i, &e_j, &a_j, "iter 1");

    // Hot update lands between iterations at the yield safe point.
    scenario3_add_fee(&e_i);
    scenario3_add_fee(&e_j);

    // Remaining iterations to completion — the account migrates transparently.
    let mut guard = 0;
    while matches!(a_i.status, ActorStatus::Runnable) {
        to_yield(&e_i, &mut a_i);
        to_yield(&e_j, &mut a_j);
        assert_same(&e_i, &a_i, &e_j, &a_j, "loop tail");
        guard += 1;
        assert!(guard < 100, "loop did not terminate");
    }

    // Balance was unchanged, so every emitted effect is identical...
    assert_eq!(
        e_i.output(),
        vec![Value::I64(100), Value::I64(100), Value::I64(100)]
    );
    assert_eq!(a_j.status, ActorStatus::Complete(Value::I64(0)));
    // ...and the mid-loop migration actually happened: the account is now v2.
    assert_eq!(
        e_j.shared().object_body(acct_j).unwrap().schema,
        Version(2),
        "account migrated mid-loop"
    );
    assert_same(&e_i, &a_i, &e_j, &a_j, "loop complete");
}

// ===========================================================================
// Precise GC over native frame slots — the design's "complete root map" claim.
// ===========================================================================

const GC_MAIN: DefId = 20;

#[test]
fn jit_gc_roots_from_frame_slots() {
    let rt = jit_engine(0);
    rt.install_schema(Schema {
        type_id: BOX,
        version: Version(1),
        name: "Box".into(),
        fields: vec![field(VALUE, "value", Type::I64)],
    })
    .unwrap();
    rt.install_function(Function {
        id: GC_MAIN,
        version: Version(1),
        name: "gc_main".into(),
        params: vec![],
        result: Type::I64,
        registers: 3,
        code: vec![
            Instruction::Const {
                dst: 0,
                value: Value::I64(7),
            },
            Instruction::New {
                dst: 1,
                type_id: BOX,
                fields: vec![(VALUE, 0)],
            },
            Instruction::Emit { value: 0 },
            Instruction::Yield, // stop with the box live in slot r1
            Instruction::GetField {
                dst: 2,
                object: 1,
                field: VALUE,
            },
            Instruction::Return { value: 2 },
        ],
    })
    .unwrap();

    let mut actor = rt.spawn(GC_MAIN, vec![]).unwrap();
    to_yield(&rt, &mut actor);
    assert!(actor.top_is_native(), "the frame under GC must be a JIT frame");

    // The box is allocated and rooted only by the native frame slot r1.
    assert_eq!(rt.shared().object_count(), 1);
    let roots = actor.roots();
    assert_eq!(roots.len(), 1, "the box is a frame-slot root");
    assert_eq!(rt.collect(&[&actor]), 0, "rooted box survives");
    assert_eq!(rt.shared().object_count(), 1);

    // Run to completion; the frame is popped, so the box is unreachable.
    rt.run(&mut actor);
    assert_eq!(actor.status, ActorStatus::Complete(Value::I64(7)));
    assert!(actor.stack.is_empty());
    assert_eq!(rt.collect(&[&actor]), 1, "box swept");
    assert_eq!(rt.shared().object_count(), 0);
}

// ===========================================================================
// Con-freeness trap (soundness). A frame paused at a Yield inside an OLD pinned
// function resumes AFTER a migration has changed a field's representation out
// from under it. The pinned code must not silently use the wrong-typed value:
// it traps, identically on both configurations. Before the soundness work the
// JIT read the migrated Ref's object id as an integer and diverged silently.
// ===========================================================================

const CELL: DefId = 5;
const WRAPT: DefId = 6;
const USE_CELL: DefId = 40;
const NF: FieldId = 500;
const WF: FieldId = 600;

/// Install `Cell{n:Int}` plus the pinned function `body`, allocate a cell, and
/// return the cell id.
fn confree_setup(rt: &Engine, body: Function) -> ObjectId {
    rt.install_schema(Schema {
        type_id: CELL,
        version: Version(1),
        name: "Cell".into(),
        fields: vec![field(NF, "n", Type::I64)],
    })
    .unwrap();
    rt.install_function(body).unwrap();
    rt.shared().jit_new(CELL, &[(NF, Value::I64(41))]).unwrap()
}

/// The hot update that pulls the representation out from under pinned code:
/// `Cell.n` becomes `Ref(Wrap)`, with a migration that wraps the old int. This
/// breaks the pinned function (re-verify fails) but its running frame keeps the
/// old version and later reads the now-`Ref` field.
fn confree_migrate(rt: &Engine) {
    rt.install_schema(Schema {
        type_id: WRAPT,
        version: Version(1),
        name: "Wrap".into(),
        fields: vec![field(WF, "w", Type::I64)],
    })
    .unwrap();
    rt.install_schema(Schema {
        type_id: CELL,
        version: Version(2),
        name: "Cell".into(),
        fields: vec![field(NF, "n", Type::Ref(WRAPT))],
    })
    .unwrap();
    rt.install_migration(Migration {
        type_id: CELL,
        from: Version(1),
        to: Version(2),
        fields: BTreeMap::from([(
            NF,
            MigrationSource::Wrap {
                type_id: WRAPT,
                field: WF,
                source: NF,
            },
        )]),
    })
    .unwrap();
}

fn run_confree(body: Function) -> (ActorStatus, ActorStatus) {
    let (e_i, e_j) = both_engines();
    let cell_i = confree_setup(&e_i, body.clone());
    let cell_j = confree_setup(&e_j, body);
    let mut a_i = e_i.spawn(USE_CELL, vec![Value::Ref(cell_i)]).unwrap();
    let mut a_j = e_j.spawn(USE_CELL, vec![Value::Ref(cell_j)]).unwrap();

    // Pause inside the pinned function at its Yield.
    to_yield(&e_i, &mut a_i);
    to_yield(&e_j, &mut a_j);
    assert_same(&e_i, &a_i, &e_j, &a_j, "confree: at yield");

    // Migrate the field's representation, then resume the pinned frame.
    confree_migrate(&e_i);
    confree_migrate(&e_j);
    advance(&e_i, &mut a_i);
    advance(&e_j, &mut a_j);
    (a_i.status.clone(), a_j.status.clone())
}

#[test]
fn confree_arithmetic_trap_matches() {
    // 0:Yield 1:n=GetField 2:one=1 3:SubI64(n,one) 4:Return
    let body = Function {
        id: USE_CELL,
        version: Version(1),
        name: "use_cell".into(),
        params: vec![Type::Ref(CELL)],
        result: Type::I64,
        registers: 4,
        code: vec![
            Instruction::Yield,
            Instruction::GetField {
                dst: 1,
                object: 0,
                field: NF,
            },
            Instruction::Const {
                dst: 2,
                value: Value::I64(1),
            },
            Instruction::SubI64 {
                dst: 3,
                left: 1,
                right: 2,
            },
            Instruction::Return { value: 3 },
        ],
    };
    let (interp, jit) = run_confree(body);
    // The pinned v1 code reads a now-Ref field and feeds it to SubI64: both
    // configurations trap with the identical operand-type condition (at pc 3).
    assert_eq!(interp, jit, "configurations diverged on the con-freeness trap");
    assert!(
        matches!(
            &jit,
            ActorStatus::Paused(Condition::RuntimeTypeError { function, pc, .. })
                if *function == USE_CELL && *pc == 3
        ),
        "expected an operand-type trap at the subtraction, got {jit:?}"
    );
}

#[test]
fn confree_return_trap_matches() {
    // 0:Yield 1:n=GetField 2:Return n  — returns a now-Ref where Int is declared
    let body = Function {
        id: USE_CELL,
        version: Version(1),
        name: "read_cell".into(),
        params: vec![Type::Ref(CELL)],
        result: Type::I64,
        registers: 2,
        code: vec![
            Instruction::Yield,
            Instruction::GetField {
                dst: 1,
                object: 0,
                field: NF,
            },
            Instruction::Return { value: 1 },
        ],
    };
    let (interp, jit) = run_confree(body);
    assert_eq!(interp, jit, "configurations diverged on the return-type trap");
    assert!(
        matches!(
            &jit,
            ActorStatus::Paused(Condition::RuntimeTypeError { function, pc, .. })
                if *function == USE_CELL && *pc == 2
        ),
        "expected a return-type trap, got {jit:?}"
    );
}

// ===========================================================================
// Auto-derived migrations (D6). A trivial schema change (add a defaulted field)
// migrates live values with NO hand-written migration; a genuine representation
// change is left as a gap for the developer.
// ===========================================================================

const AUTO: DefId = 7;
const AF: FieldId = 700;
const AF2: FieldId = 701;
const READ_AUTO: DefId = 50;

fn auto_setup(rt: &Engine) -> ObjectId {
    rt.install_schema(Schema {
        type_id: AUTO,
        version: Version(1),
        name: "Auto".into(),
        fields: vec![field(AF, "a", Type::I64)],
    })
    .unwrap();
    rt.install_function(Function {
        id: READ_AUTO,
        version: Version(1),
        name: "read_auto".into(),
        params: vec![Type::Ref(AUTO)],
        result: Type::I64,
        registers: 2,
        code: vec![
            Instruction::GetField {
                dst: 1,
                object: 0,
                field: AF,
            },
            Instruction::Return { value: 1 },
        ],
    })
    .unwrap();
    rt.shared().jit_new(AUTO, &[(AF, Value::I64(10))]).unwrap()
}

fn auto_add_field(rt: &Engine) {
    rt.install_schema(Schema {
        type_id: AUTO,
        version: Version(2),
        name: "Auto".into(),
        fields: vec![
            field(AF, "a", Type::I64),
            field_with_default(AF2, "b", Type::I64, Value::I64(99)),
        ],
    })
    .unwrap();
}

#[test]
fn auto_derived_migration_is_transparent_on_both_configurations() {
    let (e_i, e_j) = both_engines();
    let obj_i = auto_setup(&e_i);
    let obj_j = auto_setup(&e_j);

    auto_add_field(&e_i);
    auto_add_field(&e_j);
    // Installing the schema auto-derived the v1→v2 migration — no explicit one.
    assert!(
        e_i.with_world(|w| w.migrations.contains_key(&(AUTO, Version(1)))),
        "additive+defaulted change should auto-derive a migration"
    );

    let mut a_i = e_i.spawn(READ_AUTO, vec![Value::Ref(obj_i)]).unwrap();
    e_i.run(&mut a_i);
    let mut a_j = e_j.spawn(READ_AUTO, vec![Value::Ref(obj_j)]).unwrap();
    e_j.run(&mut a_j);

    // The read migrated the object transparently (no pause) and saw the old
    // value; the defaulted field was filled in.
    assert_same(&e_i, &a_i, &e_j, &a_j, "auto-derived");
    assert_eq!(a_j.status, ActorStatus::Complete(Value::I64(10)));
    assert_eq!(e_j.shared().object_body(obj_j).unwrap().schema, Version(2));
    assert_eq!(
        e_j.shared().object_body(obj_j).unwrap().fields[&AF2],
        Value::I64(99)
    );
}

#[test]
fn auto_derivation_abstains_on_representation_change() {
    // Retyping a field with no default is a gap: no migration is auto-installed,
    // so a cross traps `MissingMigration` until a developer supplies one. (This
    // is why the Box/Wrapper and Account/Money scenarios still pause.)
    let rt = Engine::interp();
    rt.install_schema(Schema {
        type_id: AUTO,
        version: Version(1),
        name: "Auto".into(),
        fields: vec![field(AF, "a", Type::I64)],
    })
    .unwrap();
    rt.install_schema(Schema {
        type_id: AUTO,
        version: Version(2),
        name: "Auto".into(),
        fields: vec![field(AF, "a", Type::Ref(AUTO))], // Int → Ref, no default
    })
    .unwrap();
    assert!(
        rt.with_world(|w| !w.migrations.contains_key(&(AUTO, Version(1)))),
        "a representation change must NOT be auto-derived"
    );
}
