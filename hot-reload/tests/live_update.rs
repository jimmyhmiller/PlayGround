use livetype::*;
use std::collections::BTreeMap;

const BOX: DefId = 1;
const WRAPPER: DefId = 2;
const READ: DefId = 10;
const MAIN: DefId = 11;
const VALUE: FieldId = 100;
const INNER: FieldId = 200;

fn field(id: FieldId, ty: Type) -> Field {
    Field {
        id,
        name: format!("f{id}"),
        ty,
        default: None,
    }
}

fn initial_runtime() -> Runtime {
    let mut runtime = Runtime::default();
    runtime
        .install_schema(Schema {
            type_id: BOX,
            version: Version(1),
            name: "Box".into(),
            fields: vec![field(VALUE, Type::I64)],
        })
        .unwrap();
    runtime
        .install_function(Function {
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
    runtime
        .install_function(Function {
            id: MAIN,
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
                Instruction::Call {
                    dst: 2,
                    function: READ,
                    args: vec![1],
                },
                Instruction::Return { value: 2 },
            ],
        })
        .unwrap();
    runtime
}

#[test]
fn repairs_resume_exactly_and_effects_are_not_replayed() {
    let mut runtime = initial_runtime();
    let actor = runtime.spawn(MAIN, vec![]).unwrap();
    for _ in 0..3 {
        runtime.step(actor);
    }
    assert_eq!(runtime.output, vec![Value::I64(42)]);

    runtime
        .install_schema(Schema {
            type_id: WRAPPER,
            version: Version(1),
            name: "Wrapper".into(),
            fields: vec![field(INNER, Type::I64)],
        })
        .unwrap();
    runtime
        .install_schema(Schema {
            type_id: BOX,
            version: Version(2),
            name: "Box".into(),
            fields: vec![field(VALUE, Type::Ref(WRAPPER))],
        })
        .unwrap();
    runtime.run();
    assert!(matches!(
        runtime.actors[&actor].status,
        ActorStatus::Paused(Condition::BrokenFunction { function: READ, .. })
    ));

    runtime
        .install_function(Function {
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
    runtime.run();
    assert!(matches!(
        runtime.actors[&actor].status,
        ActorStatus::Paused(Condition::MissingMigration { .. })
    ));

    runtime
        .install_migration(Migration {
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
    runtime.run();
    assert_eq!(
        runtime.actors[&actor].status,
        ActorStatus::Complete(Value::I64(42))
    );
    assert_eq!(runtime.output, vec![Value::I64(42)]);
}

#[test]
fn collection_traces_explicit_frame_roots() {
    let mut runtime = initial_runtime();
    let actor = runtime.spawn(MAIN, vec![]).unwrap();
    runtime.step(actor);
    runtime.step(actor);
    assert_eq!(runtime.heap.len(), 1);
    assert_eq!(runtime.collect_garbage(), 0);
    runtime.run();
    assert_eq!(runtime.collect_garbage(), 1);
    assert!(runtime.heap.is_empty());
}
