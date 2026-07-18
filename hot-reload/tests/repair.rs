//! Repairable con-freeness traps — the trap-and-*repair* half of the thesis.
//! A frame that froze on a value a migration made wrong-typed is a one-shot
//! delimited continuation: you resume it, optionally supplying the value the
//! frozen instruction should have produced. The offering must be well-typed, so
//! a repair can never reintroduce an ill-typed value. Both configurations of
//! the one engine (interpreter-only and always-JIT) must agree.

use livetype::*;
use std::collections::BTreeMap;
use std::sync::Arc;

const CELL: DefId = 1;
const WRAP: DefId = 2;
const USE: DefId = 10;
const N: FieldId = 100;
const W: FieldId = 200;

fn field(id: FieldId, name: &str, ty: Type) -> Field {
    Field {
        id,
        name: name.into(),
        ty,
        default: None,
    }
}

/// `use_cell(c)` yields, reads `c.n` (an Int), subtracts, returns.
fn arithmetic_body() -> Function {
    Function {
        id: USE,
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
                field: N,
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
    }
}

/// `read_cell(c)` yields, reads `c.n`, returns it directly.
fn return_body() -> Function {
    Function {
        id: USE,
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
                field: N,
            },
            Instruction::Return { value: 1 },
        ],
    }
}

fn setup(engine: &Engine, body: Function) -> ObjectId {
    engine
        .install_schema(Schema {
            type_id: CELL,
            version: Version(1),
            name: "Cell".into(),
            fields: vec![field(N, "n", Type::I64)],
        })
        .unwrap();
    engine.install_function(body).unwrap();
    engine.shared().jit_new(CELL, &[(N, Value::I64(41))]).unwrap()
}

/// The update that pulls the representation out from under the pinned function:
/// `Cell.n` becomes `Ref(Wrap)`, with the wrapping migration supplied.
fn migrate(engine: &Engine) {
    engine
        .install_schema(Schema {
            type_id: WRAP,
            version: Version(1),
            name: "Wrap".into(),
            fields: vec![field(W, "w", Type::I64)],
        })
        .unwrap();
    engine
        .install_schema(Schema {
            type_id: CELL,
            version: Version(2),
            name: "Cell".into(),
            fields: vec![field(N, "n", Type::Ref(WRAP))],
        })
        .unwrap();
    engine
        .install_migration(Migration {
            type_id: CELL,
            from: Version(1),
            to: Version(2),
            fields: BTreeMap::from([(
                N,
                MigrationSource::Wrap {
                    type_id: WRAP,
                    field: W,
                    source: N,
                },
            )]),
        })
        .unwrap();
}

fn run_to_yield(engine: &Engine, actor: &mut Actor) {
    loop {
        match engine.step(actor) {
            Turn::Progress => {}
            Turn::Yielded | Turn::Done | Turn::Paused => break,
            Turn::Blocked => panic!("nothing sends to this actor"),
        }
    }
}

/// Drive one configuration to the con-freeness trap and return it paused there.
fn to_trap_on(engine: &Arc<Engine>, body: Function) -> Actor {
    let cell = setup(engine, body);
    let mut actor = engine.spawn(USE, vec![Value::Ref(cell)]).unwrap();
    run_to_yield(engine, &mut actor);
    migrate(engine);
    engine.run(&mut actor);
    actor
}

/// Both configurations, paused at the same trap.
fn to_trap(body: Function) -> (Arc<Engine>, Actor, Arc<Engine>, Actor) {
    let e_i = Engine::interp();
    let a_i = to_trap_on(&e_i, body.clone());
    let e_j = jit_engine(0);
    let a_j = to_trap_on(&e_j, body);
    (e_i, a_i, e_j, a_j)
}

#[test]
fn resume_arithmetic_trap_with_value_matches() {
    let (e_i, mut a_i, e_j, mut a_j) = to_trap(arithmetic_body());
    // Both froze at the subtraction.
    assert!(matches!(
        a_i.status,
        ActorStatus::Paused(Condition::RuntimeTypeError { pc: 3, .. })
    ));
    assert!(matches!(
        a_j.status,
        ActorStatus::Paused(Condition::RuntimeTypeError { pc: 3, .. })
    ));
    // The trap tells us it wants an Int (the subtraction's result).
    assert_eq!(e_i.pause_expected(&a_i), Some(Type::I64));
    assert_eq!(e_j.pause_expected(&a_j), Some(Type::I64));

    // Resume both by supplying the result the subtraction should have produced.
    e_i.resume_with(&mut a_i, Value::I64(777)).unwrap();
    assert_eq!(e_i.run(&mut a_i), Outcome::Complete(Value::I64(777)));
    e_j.resume_with(&mut a_j, Value::I64(777)).unwrap();
    assert_eq!(e_j.run(&mut a_j), Outcome::Complete(Value::I64(777)));
}

#[test]
fn resume_return_trap_with_value_matches() {
    let (e_i, mut a_i, e_j, mut a_j) = to_trap(return_body());
    assert!(matches!(
        a_i.status,
        ActorStatus::Paused(Condition::RuntimeTypeError { pc: 2, .. })
    ));
    // A Return trap expects the function's result type.
    assert_eq!(e_i.pause_expected(&a_i), Some(Type::I64));

    e_i.resume_with(&mut a_i, Value::I64(42)).unwrap();
    assert_eq!(e_i.run(&mut a_i), Outcome::Complete(Value::I64(42)));
    e_j.resume_with(&mut a_j, Value::I64(42)).unwrap();
    assert_eq!(e_j.run(&mut a_j), Outcome::Complete(Value::I64(42)));
}

#[test]
fn resume_rejects_an_ill_typed_offering() {
    // The subtraction wants an Int; offering a Bool must be refused, and the
    // actor must stay quarantined — repair cannot smuggle in an ill-typed value.
    let (e_i, mut a_i, e_j, mut a_j) = to_trap(arithmetic_body());

    let err_i = e_i.resume_with(&mut a_i, Value::Bool(true));
    assert!(err_i.is_err(), "interpreter accepted a wrong-typed resume");
    assert!(matches!(
        a_i.status,
        ActorStatus::Paused(Condition::RuntimeTypeError { .. })
    ));

    let err_j = e_j.resume_with(&mut a_j, Value::Bool(true));
    assert!(err_j.is_err(), "jit accepted a wrong-typed resume");
    assert!(matches!(
        a_j.status,
        ActorStatus::Paused(Condition::RuntimeTypeError { .. })
    ));

    // And a correct offering afterward still works — the trap wasn't consumed.
    e_i.resume_with(&mut a_i, Value::I64(5)).unwrap();
    assert_eq!(e_i.run(&mut a_i), Outcome::Complete(Value::I64(5)));
    e_j.resume_with(&mut a_j, Value::I64(5)).unwrap();
    assert_eq!(e_j.run(&mut a_j), Outcome::Complete(Value::I64(5)));
}
