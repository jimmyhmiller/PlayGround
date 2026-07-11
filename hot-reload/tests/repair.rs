//! Repairable con-freeness traps — the trap-and-*repair* half of the thesis.
//! A frame that froze on a value a migration made wrong-typed is a one-shot
//! delimited continuation: you resume it, optionally supplying the value the
//! frozen instruction should have produced. The offering must be well-typed, so
//! a repair can never reintroduce an ill-typed value. Both executors must agree.

use livetype::*;
use std::collections::BTreeMap;

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

fn setup(body: Function) -> (Runtime, ObjectId) {
    let mut rt = Runtime::default();
    rt.install_schema(Schema {
        type_id: CELL,
        version: Version(1),
        name: "Cell".into(),
        fields: vec![field(N, "n", Type::I64)],
    })
    .unwrap();
    rt.install_function(body).unwrap();
    let cell = rt.jit_new(CELL, &[(N, Value::I64(41))]).unwrap();
    (rt, cell)
}

/// The update that pulls the representation out from under the pinned function:
/// `Cell.n` becomes `Ref(Wrap)`, with the wrapping migration supplied.
fn migrate(rt: &mut Runtime) {
    rt.install_schema(Schema {
        type_id: WRAP,
        version: Version(1),
        name: "Wrap".into(),
        fields: vec![field(W, "w", Type::I64)],
    })
    .unwrap();
    rt.install_schema(Schema {
        type_id: CELL,
        version: Version(2),
        name: "Cell".into(),
        fields: vec![field(N, "n", Type::Ref(WRAP))],
    })
    .unwrap();
    rt.install_migration(Migration {
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

fn interp_run_to_yield(rt: &mut Runtime, actor: ActorId) {
    while matches!(rt.actors[&actor].status, ActorStatus::Runnable) {
        let (key, pc) = {
            let f = rt.actors[&actor].frames.last().unwrap();
            (f.function, f.pc)
        };
        let is_yield = matches!(
            &rt.world.functions[&key],
            FunctionState::Ready(f) if matches!(f.code[pc], Instruction::Yield)
        );
        rt.step(actor);
        if is_yield {
            break;
        }
    }
}

/// Drive both executors to the con-freeness trap and return them paused there.
fn to_trap(body: Function) -> (Runtime, ActorId, Runtime, JitActor) {
    let (mut rt_i, cell_i) = setup(body.clone());
    let a_i = rt_i.spawn(USE, vec![Value::Ref(cell_i)]).unwrap();
    interp_run_to_yield(&mut rt_i, a_i);
    migrate(&mut rt_i);
    rt_i.run();

    let (mut rt_j, cell_j) = setup(body);
    let mut a_j = JitActor::spawn(&rt_j, 1, USE, vec![Value::Ref(cell_j)]).unwrap();
    drive(&mut rt_j, &mut a_j, true).unwrap();
    migrate(&mut rt_j);
    drive(&mut rt_j, &mut a_j, false).unwrap();

    (rt_i, a_i, rt_j, a_j)
}

#[test]
fn resume_arithmetic_trap_with_value_matches() {
    let (mut rt_i, a_i, mut rt_j, mut a_j) = to_trap(arithmetic_body());
    // Both froze at the subtraction.
    assert!(matches!(
        rt_i.actors[&a_i].status,
        ActorStatus::Paused(Condition::RuntimeTypeError { pc: 3, .. })
    ));
    assert!(matches!(
        a_j.status,
        ActorStatus::Paused(Condition::RuntimeTypeError { pc: 3, .. })
    ));
    // The trap tells us it wants an Int (the subtraction's result).
    assert_eq!(rt_i.pause_expected(a_i), Some(Type::I64));

    // Resume both by supplying the result the subtraction should have produced.
    rt_i.resume_with(a_i, Value::I64(777)).unwrap();
    rt_i.run();
    resume_with(&rt_j, &mut a_j, Value::I64(777)).unwrap();
    drive(&mut rt_j, &mut a_j, false).unwrap();

    assert_eq!(rt_i.actors[&a_i].status, ActorStatus::Complete(Value::I64(777)));
    assert_eq!(a_j.status, ActorStatus::Complete(Value::I64(777)));
}

#[test]
fn resume_return_trap_with_value_matches() {
    let (mut rt_i, a_i, mut rt_j, mut a_j) = to_trap(return_body());
    assert!(matches!(
        rt_i.actors[&a_i].status,
        ActorStatus::Paused(Condition::RuntimeTypeError { pc: 2, .. })
    ));
    // A Return trap expects the function's result type.
    assert_eq!(rt_i.pause_expected(a_i), Some(Type::I64));

    rt_i.resume_with(a_i, Value::I64(42)).unwrap();
    rt_i.run();
    resume_with(&rt_j, &mut a_j, Value::I64(42)).unwrap();
    drive(&mut rt_j, &mut a_j, false).unwrap();

    assert_eq!(rt_i.actors[&a_i].status, ActorStatus::Complete(Value::I64(42)));
    assert_eq!(a_j.status, ActorStatus::Complete(Value::I64(42)));
}

#[test]
fn resume_rejects_an_ill_typed_offering() {
    // The subtraction wants an Int; offering a Bool must be refused, and the
    // actor must stay quarantined — repair cannot smuggle in an ill-typed value.
    let (mut rt_i, a_i, mut rt_j, mut a_j) = to_trap(arithmetic_body());

    let err_i = rt_i.resume_with(a_i, Value::Bool(true));
    assert!(err_i.is_err(), "interpreter accepted a wrong-typed resume");
    assert!(matches!(
        rt_i.actors[&a_i].status,
        ActorStatus::Paused(Condition::RuntimeTypeError { .. })
    ));

    let err_j = resume_with(&rt_j, &mut a_j, Value::Bool(true));
    assert!(err_j.is_err(), "jit accepted a wrong-typed resume");
    assert!(matches!(
        a_j.status,
        ActorStatus::Paused(Condition::RuntimeTypeError { .. })
    ));

    // And a correct offering afterward still works — the trap wasn't consumed.
    rt_i.resume_with(a_i, Value::I64(5)).unwrap();
    rt_i.run();
    resume_with(&rt_j, &mut a_j, Value::I64(5)).unwrap();
    drive(&mut rt_j, &mut a_j, false).unwrap();
    assert_eq!(rt_i.actors[&a_i].status, ActorStatus::Complete(Value::I64(5)));
    assert_eq!(a_j.status, ActorStatus::Complete(Value::I64(5)));
}
