use livetype::*;
use std::collections::BTreeMap;

const ACCOUNT: DefId = 1;
const MONEY: DefId = 2;
const CHARGE: DefId = 10;
const MAIN: DefId = 11;
const BALANCE: FieldId = 100;
const CENTS: FieldId = 200;

fn field(id: FieldId, name: &str, ty: Type) -> Field {
    Field {
        id,
        name: name.into(),
        ty,
        default: None,
    }
}

fn main() {
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
        id: MAIN,
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
    let actor = rt.spawn(MAIN, vec![]).unwrap();
    for _ in 0..3 {
        rt.step(actor);
    }

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
    rt.run();
    println!("paused: {:?}", rt.actors[&actor].status);

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
    rt.run();
    println!("paused: {:?}", rt.actors[&actor].status);
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
    rt.run();
    println!(
        "complete: {:?}; effects: {:?}",
        rt.actors[&actor].status, rt.output
    );
}
