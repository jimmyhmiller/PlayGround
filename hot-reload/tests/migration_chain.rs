//! Multi-step migration chains. An object allocated at v1 and first touched
//! only after several schema updates must migrate across the whole chain at
//! once (v1 → v2 → … in one `GetField`), mixing auto-derived and explicit
//! steps. Both executors must agree. This exercises the "a type can change more
//! than once" concern from the design's open agenda.

use livetype::*;
use std::collections::BTreeMap;

const ACCT: DefId = 1;
const MONEY: DefId = 2;
const READ_INT: DefId = 10;
const READ_MONEY: DefId = 11;
const BAL: FieldId = 100;
const FEE: FieldId = 101;
const TAX: FieldId = 102;
const CENTS: FieldId = 200;

fn field(id: FieldId, name: &str, ty: Type) -> Field {
    Field {
        id,
        name: name.into(),
        ty,
        default: None,
    }
}

fn field_def(id: FieldId, name: &str, ty: Type, d: Value) -> Field {
    Field {
        id,
        name: name.into(),
        ty,
        default: Some(d),
    }
}

/// Run a build closure on a fresh interpreter-only engine and a fresh
/// always-JIT engine, then compare the object's migrated state and the result.
fn both(build: impl Fn(&Engine) -> ObjectId, entry: DefId) -> (Outcome, std::sync::Arc<Body>) {
    // interpreter configuration
    let e_i = Engine::interp();
    let obj_i = build(&e_i);
    let o_i = e_i.run_call(entry, vec![Value::Ref(obj_i)]);
    // always-JIT configuration
    let e_j = jit_engine(0);
    let obj_j = build(&e_j);
    let o_j = e_j.run_call(entry, vec![Value::Ref(obj_j)]);

    assert_eq!(obj_i, obj_j);
    assert_eq!(o_i, o_j, "configurations diverged on the migration chain");
    assert_eq!(
        e_i.shared().object_body(obj_i).unwrap(),
        e_j.shared().object_body(obj_j).unwrap(),
        "migrated object diverged"
    );
    (o_j, e_j.shared().object_body(obj_j).unwrap())
}

#[test]
fn auto_derived_chain_v1_v2_v3() {
    // v1{balance} → v2{+fee} → v3{+tax}, all auto-derived; balance stays Int so
    // read_int survives every version.
    let build = |rt: &Engine| {
        rt.install_schema(Schema {
            type_id: ACCT,
            version: Version(1),
            name: "Account".into(),
            fields: vec![field(BAL, "balance", Type::I64)],
        })
        .unwrap();
        rt.install_function(Function {
            id: READ_INT,
            version: Version(1),
            name: "read_int".into(),
            params: vec![Type::Ref(ACCT)],
            result: Type::I64,
            registers: 2,
            code: vec![
                Instruction::GetField {
                    dst: 1,
                    object: 0,
                    field: BAL,
                },
                Instruction::Return { value: 1 },
            ],
        })
        .unwrap();
        let obj = rt.shared().jit_new(ACCT, &[(BAL, Value::I64(100))]).unwrap();
        // Two further versions, each auto-derivable (additive + defaulted).
        rt.install_schema(Schema {
            type_id: ACCT,
            version: Version(2),
            name: "Account".into(),
            fields: vec![
                field(BAL, "balance", Type::I64),
                field_def(FEE, "fee", Type::I64, Value::I64(0)),
            ],
        })
        .unwrap();
        rt.install_schema(Schema {
            type_id: ACCT,
            version: Version(3),
            name: "Account".into(),
            fields: vec![
                field(BAL, "balance", Type::I64),
                field_def(FEE, "fee", Type::I64, Value::I64(0)),
                field_def(TAX, "tax", Type::I64, Value::I64(7)),
            ],
        })
        .unwrap();
        obj
    };

    let (status, obj) = both(build, READ_INT);
    assert_eq!(status, Outcome::Complete(Value::I64(100)));
    // The single field access migrated the object across the whole chain.
    assert_eq!(obj.schema, Version(3));
    assert_eq!(obj.fields[&BAL], Value::I64(100));
    assert_eq!(obj.fields[&FEE], Value::I64(0));
    assert_eq!(obj.fields[&TAX], Value::I64(7));
}

#[test]
fn mixed_auto_and_explicit_chain() {
    // v1{balance:Int} → v2{+fee} (auto) → v3{balance:Money} (explicit Wrap).
    // The object crosses an auto-derived step AND an explicit one in a single
    // migration, and a Money-aware reader reads through it.
    let build = |rt: &Engine| {
        rt.install_schema(Schema {
            type_id: ACCT,
            version: Version(1),
            name: "Account".into(),
            fields: vec![field(BAL, "balance", Type::I64)],
        })
        .unwrap();
        rt.install_schema(Schema {
            type_id: MONEY,
            version: Version(1),
            name: "Money".into(),
            fields: vec![field(CENTS, "cents", Type::I64)],
        })
        .unwrap();
        let obj = rt.shared().jit_new(ACCT, &[(BAL, Value::I64(100))]).unwrap();

        rt.install_schema(Schema {
            type_id: ACCT,
            version: Version(2),
            name: "Account".into(),
            fields: vec![
                field(BAL, "balance", Type::I64),
                field_def(FEE, "fee", Type::I64, Value::I64(0)),
            ],
        })
        .unwrap(); // auto-derives v1→v2
        rt.install_schema(Schema {
            type_id: ACCT,
            version: Version(3),
            name: "Account".into(),
            fields: vec![
                field(BAL, "balance", Type::Ref(MONEY)),
                field_def(FEE, "fee", Type::I64, Value::I64(0)),
            ],
        })
        .unwrap();
        // balance retyped Int→Money: not auto-derivable, supply it.
        rt.install_migration(Migration {
            type_id: ACCT,
            from: Version(2),
            to: Version(3),
            fields: BTreeMap::from([
                (
                    BAL,
                    MigrationSource::Wrap {
                        type_id: MONEY,
                        field: CENTS,
                        source: BAL,
                    },
                ),
                (FEE, MigrationSource::Copy(FEE)),
            ]),
        })
        .unwrap();
        // Reader installed against the final shape.
        rt.install_function(Function {
            id: READ_MONEY,
            version: Version(1),
            name: "read_money".into(),
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
        obj
    };

    let (status, obj) = both(build, READ_MONEY);
    assert_eq!(status, Outcome::Complete(Value::I64(100)));
    assert_eq!(obj.schema, Version(3));
    // balance is now a Money reference; fee carried through the chain.
    assert!(matches!(obj.fields[&BAL], Value::Ref(_)));
    assert_eq!(obj.fields[&FEE], Value::I64(0));
}
