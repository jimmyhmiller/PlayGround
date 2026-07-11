//! Differential tests: the native `step` backend must be observationally
//! indistinguishable from the reference interpreter. Each scenario is set up in
//! two identical `Runtime`s, driven by the two executors through the same hot
//! updates at the same execution points, and compared at every observable
//! boundary — emitted effects, pause conditions, completion value, and the
//! whole heap. A final test proves the precise GC roots correctly from native
//! frame slots.

use livetype::*;
use std::collections::BTreeMap;

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

/// The current instruction at an interpreter actor's top frame.
fn top_instr(rt: &Runtime, actor: ActorId) -> Instruction {
    let frame = rt.actors[&actor].frames.last().unwrap();
    let FunctionState::Ready(func) = &rt.world.functions[&frame.function] else {
        panic!("top frame pins non-ready code");
    };
    func.code[frame.pc].clone()
}

/// Step the interpreter until the actor stops being runnable OR has just
/// executed a `Yield` — the interpreter analogue of `drive(.., stop_on_yield)`.
fn interp_run_to_yield(rt: &mut Runtime, actor: ActorId) {
    while matches!(rt.actors[&actor].status, ActorStatus::Runnable) {
        let was_yield = matches!(top_instr(rt, actor), Instruction::Yield);
        rt.step(actor);
        if was_yield {
            break;
        }
    }
}

fn interp_runnable(rt: &Runtime, actor: ActorId) -> bool {
    matches!(rt.actors[&actor].status, ActorStatus::Runnable)
}

/// Assert the two executors agree on everything observable.
fn assert_same(rt_i: &Runtime, a_i: ActorId, rt_j: &Runtime, a_j: &JitActor, label: &str) {
    assert_eq!(rt_i.output, rt_j.output, "{label}: emitted effects diverged");
    assert_eq!(
        rt_i.actors[&a_i].status, a_j.status,
        "{label}: actor status diverged"
    );
    assert_eq!(rt_i.heap, rt_j.heap, "{label}: heap diverged");
}

// ===========================================================================
// Scenario 1 — Box migrated to hold a Wrapper (the tests/live_update.rs story,
// with a Yield safe point so both executors stop at the same place).
// ===========================================================================

const BOX: DefId = 1;
const WRAPPER: DefId = 2;
const READ: DefId = 10;
const S1_MAIN: DefId = 11;
const VALUE: FieldId = 100;
const INNER: FieldId = 200;

fn scenario1_setup() -> Runtime {
    let mut rt = Runtime::default();
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
    rt
}

fn scenario1_break(rt: &mut Runtime) {
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

fn scenario1_fix_read(rt: &mut Runtime) {
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

fn scenario1_migration(rt: &mut Runtime) {
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
    let mut rt_i = scenario1_setup();
    let a_i = rt_i.spawn(S1_MAIN, vec![]).unwrap();
    let mut rt_j = scenario1_setup();
    let mut a_j = JitActor::spawn(&rt_j, 1, S1_MAIN, vec![]).unwrap();

    // Phase A: run to the yield — Box allocated, effect emitted, before Call.
    interp_run_to_yield(&mut rt_i, a_i);
    drive(&mut rt_j, &mut a_j, true).unwrap();
    assert_eq!(rt_i.output, vec![Value::I64(42)]);
    assert_same(&rt_i, a_i, &rt_j, &a_j, "A: at yield");

    // Update: Box now holds a Wrapper — this breaks `read`.
    scenario1_break(&mut rt_i);
    scenario1_break(&mut rt_j);

    // Phase B: reaching the broken `read` pauses.
    rt_i.run();
    drive(&mut rt_j, &mut a_j, false).unwrap();
    assert!(matches!(
        a_j.status,
        ActorStatus::Paused(Condition::BrokenFunction { function: READ, .. })
    ));
    assert_same(&rt_i, a_i, &rt_j, &a_j, "B: broken function");

    // Repair `read`, then hit the un-migratable value.
    scenario1_fix_read(&mut rt_i);
    scenario1_fix_read(&mut rt_j);
    rt_i.run();
    drive(&mut rt_j, &mut a_j, false).unwrap();
    assert!(matches!(
        a_j.status,
        ActorStatus::Paused(Condition::MissingMigration { .. })
    ));
    assert_same(&rt_i, a_i, &rt_j, &a_j, "C: missing migration");

    // Supply the migration and resume to completion.
    scenario1_migration(&mut rt_i);
    scenario1_migration(&mut rt_j);
    rt_i.run();
    drive(&mut rt_j, &mut a_j, false).unwrap();
    assert_eq!(a_j.status, ActorStatus::Complete(Value::I64(42)));
    assert_same(&rt_i, a_i, &rt_j, &a_j, "D: complete");
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

fn scenario2_setup() -> Runtime {
    let mut rt = Runtime::default();
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
    rt
}

fn scenario2_break(rt: &mut Runtime) {
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

fn scenario2_fix_charge(rt: &mut Runtime) {
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

fn scenario2_migration(rt: &mut Runtime) {
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
    let mut rt_i = scenario2_setup();
    let a_i = rt_i.spawn(S2_MAIN, vec![]).unwrap();
    let mut rt_j = scenario2_setup();
    let mut a_j = JitActor::spawn(&rt_j, 1, S2_MAIN, vec![]).unwrap();

    interp_run_to_yield(&mut rt_i, a_i);
    drive(&mut rt_j, &mut a_j, true).unwrap();
    assert_eq!(rt_i.output, vec![Value::I64(100)]);
    assert_same(&rt_i, a_i, &rt_j, &a_j, "A: at yield");

    scenario2_break(&mut rt_i);
    scenario2_break(&mut rt_j);
    rt_i.run();
    drive(&mut rt_j, &mut a_j, false).unwrap();
    assert!(matches!(
        a_j.status,
        ActorStatus::Paused(Condition::BrokenFunction { function: CHARGE, .. })
    ));
    assert_same(&rt_i, a_i, &rt_j, &a_j, "B: broken charge");

    scenario2_fix_charge(&mut rt_i);
    scenario2_fix_charge(&mut rt_j);
    rt_i.run();
    drive(&mut rt_j, &mut a_j, false).unwrap();
    assert!(matches!(
        a_j.status,
        ActorStatus::Paused(Condition::MissingMigration { .. })
    ));
    assert_same(&rt_i, a_i, &rt_j, &a_j, "C: missing migration");

    scenario2_migration(&mut rt_i);
    scenario2_migration(&mut rt_j);
    rt_i.run();
    drive(&mut rt_j, &mut a_j, false).unwrap();
    assert_eq!(a_j.status, ActorStatus::Complete(Value::I64(95)));
    assert_same(&rt_i, a_i, &rt_j, &a_j, "D: complete");
}

// ===========================================================================
// Scenario 3 — a loop with a Yield each iteration, hot-updated MID-LOOP.
// Exercises back-edges, the recurring safe point (T5), and a lazy migration
// landing between iterations, all identically on both executors.
// ===========================================================================

const LOOP: DefId = 10;
const FEE: FieldId = 101;

fn scenario3_setup() -> (Runtime, ObjectId) {
    let mut rt = Runtime::default();
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
    let account = rt.jit_new(ACCOUNT, &[(BALANCE, Value::I64(100))]);
    (rt, account)
}

/// Additive change with its migration: `Account` gains a defaulted `fee`, and
/// the v1 → v2 migration copies `balance` and initializes `fee`. This does NOT
/// break `loop_balance` (it still reads `balance: Int`); the live account
/// migrates lazily and transparently at the next field access — the hot update
/// landing between loop iterations.
fn scenario3_add_fee(rt: &mut Runtime) {
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
    rt.install_migration(Migration {
        type_id: ACCOUNT,
        from: Version(1),
        to: Version(2),
        fields: BTreeMap::from([
            (BALANCE, MigrationSource::Copy(BALANCE)),
            (FEE, MigrationSource::Value(Value::I64(0))),
        ]),
    })
    .unwrap();
}

#[test]
fn scenario3_loop_with_midloop_update_matches() {
    let (mut rt_i, acct_i) = scenario3_setup();
    let a_i = rt_i.spawn(LOOP, vec![Value::Ref(acct_i), Value::I64(3)]).unwrap();
    let (mut rt_j, acct_j) = scenario3_setup();
    let mut a_j = JitActor::spawn(&rt_j, 1, LOOP, vec![Value::Ref(acct_j), Value::I64(3)]).unwrap();
    assert_eq!(acct_i, acct_j, "both runtimes allocate the account identically");

    // First iteration.
    interp_run_to_yield(&mut rt_i, a_i);
    drive(&mut rt_j, &mut a_j, true).unwrap();
    assert_eq!(rt_i.output, vec![Value::I64(100)]);
    assert_same(&rt_i, a_i, &rt_j, &a_j, "iter 1");

    // Hot update lands between iterations at the yield safe point.
    scenario3_add_fee(&mut rt_i);
    scenario3_add_fee(&mut rt_j);

    // Remaining iterations to completion — the account migrates transparently.
    let mut guard = 0;
    while interp_runnable(&rt_i, a_i) {
        interp_run_to_yield(&mut rt_i, a_i);
        drive(&mut rt_j, &mut a_j, true).unwrap();
        assert_same(&rt_i, a_i, &rt_j, &a_j, "loop tail");
        guard += 1;
        assert!(guard < 100, "loop did not terminate");
    }

    // Balance was unchanged, so every emitted effect is identical...
    assert_eq!(
        rt_i.output,
        vec![Value::I64(100), Value::I64(100), Value::I64(100)]
    );
    assert_eq!(a_j.status, ActorStatus::Complete(Value::I64(0)));
    // ...and the mid-loop migration actually happened: the account is now v2.
    assert_eq!(rt_j.heap[&acct_j].schema, Version(2), "account migrated mid-loop");
    assert_same(&rt_i, a_i, &rt_j, &a_j, "loop complete");
}

// ===========================================================================
// Precise GC over native frame slots — the design's "complete root map" claim.
// ===========================================================================

const GC_MAIN: DefId = 20;

#[test]
fn jit_gc_roots_from_frame_slots() {
    let mut rt = Runtime::default();
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

    let mut actor = JitActor::spawn(&rt, 1, GC_MAIN, vec![]).unwrap();
    drive(&mut rt, &mut actor, true).unwrap();

    // The box is allocated and rooted only by the native frame slot r1.
    assert_eq!(rt.heap.len(), 1);
    let roots = actor.roots();
    assert_eq!(roots.len(), 1, "the box is a frame-slot root");
    assert_eq!(rt.collect_garbage_with_roots(&roots), 0, "rooted box survives");
    assert_eq!(rt.heap.len(), 1);

    // Run to completion; the frame is popped, so the box is unreachable.
    drive(&mut rt, &mut actor, false).unwrap();
    assert_eq!(actor.status, ActorStatus::Complete(Value::I64(7)));
    assert!(actor.frames.is_empty());
    assert_eq!(rt.collect_garbage_with_roots(&actor.roots()), 1, "box swept");
    assert!(rt.heap.is_empty());
}
