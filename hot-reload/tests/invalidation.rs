//! Demand-driven invalidation (D7): a schema change re-verifies only the
//! functions it can reach — those whose type-dependency set contains the
//! changed type, plus (transitively) the callers of any function that thereby
//! breaks. Functions that reference neither are left untouched.

use livetype::*;

const TA: DefId = 8;
const TB: DefId = 9;
const G: DefId = 60; // uses TA internally (New + GetField)
const F: DefId = 61; // calls G, references no type of its own
const TOUCH_B: DefId = 62; // uses TB only — the control
const FA: FieldId = 800;
const FB: FieldId = 900;

fn field(id: FieldId, name: &str, ty: Type) -> Field {
    Field {
        id,
        name: name.into(),
        ty,
        default: None,
    }
}

fn current_version(rt: &Runtime, id: DefId) -> Version {
    rt.world.current_functions[&id]
}

fn is_broken(rt: &Runtime, id: DefId) -> bool {
    matches!(
        rt.world.functions[&(id, current_version(rt, id))],
        FunctionState::Broken { .. }
    )
}

fn setup() -> Runtime {
    let mut rt = Runtime::default();
    rt.install_schema(Schema {
        type_id: TA,
        version: Version(1),
        name: "A".into(),
        fields: vec![field(FA, "fa", Type::I64)],
    })
    .unwrap();
    rt.install_schema(Schema {
        type_id: TB,
        version: Version(1),
        name: "B".into(),
        fields: vec![field(FB, "fb", Type::I64)],
    })
    .unwrap();
    // g() constructs and reads an A — depends on TA, takes no arguments.
    rt.install_function(Function {
        id: G,
        version: Version(1),
        name: "g".into(),
        params: vec![],
        result: Type::I64,
        registers: 3,
        code: vec![
            Instruction::Const {
                dst: 0,
                value: Value::I64(5),
            },
            Instruction::New {
                dst: 1,
                type_id: TA,
                fields: vec![(FA, 0)],
            },
            Instruction::GetField {
                dst: 2,
                object: 1,
                field: FA,
            },
            Instruction::Return { value: 2 },
        ],
    })
    .unwrap();
    // f() just calls g — it references no type directly.
    rt.install_function(Function {
        id: F,
        version: Version(1),
        name: "f".into(),
        params: vec![],
        result: Type::I64,
        registers: 1,
        code: vec![
            Instruction::Call {
                dst: 0,
                function: G,
                args: vec![],
            },
            Instruction::Return { value: 0 },
        ],
    })
    .unwrap();
    // touch_b() reads a B — depends on TB, unrelated to TA.
    rt.install_function(Function {
        id: TOUCH_B,
        version: Version(1),
        name: "touch_b".into(),
        params: vec![Type::Ref(TB)],
        result: Type::I64,
        registers: 2,
        code: vec![
            Instruction::GetField {
                dst: 1,
                object: 0,
                field: FB,
            },
            Instruction::Return { value: 1 },
        ],
    })
    .unwrap();
    rt
}

#[test]
fn schema_change_reverifies_only_reachable_functions() {
    let mut rt = setup();
    assert!(!is_broken(&rt, G));
    assert!(!is_broken(&rt, F));
    let touch_b_before = current_version(&rt, TOUCH_B);

    // Retype A.fa Int -> Ref(A): g's `New A { fa: <i64> }` no longer type-checks.
    rt.install_schema(Schema {
        type_id: TA,
        version: Version(2),
        name: "A".into(),
        fields: vec![field(FA, "fa", Type::Ref(TA))],
    })
    .unwrap();

    // g depends on TA and breaks; f depends on no type but calls g, so it is
    // reached transitively and also breaks.
    assert!(is_broken(&rt, G), "g uses A and must be re-verified + broken");
    assert!(
        is_broken(&rt, F),
        "f calls the now-broken g and must break by propagation"
    );

    // touch_b references only B: it was never re-verified, so its version and
    // its Ready status are untouched, and no broken version was minted for it.
    assert_eq!(
        current_version(&rt, TOUCH_B),
        touch_b_before,
        "an unrelated function must not be re-versioned"
    );
    assert!(!is_broken(&rt, TOUCH_B));
    assert!(
        !rt.world
            .functions
            .contains_key(&(TOUCH_B, Version(touch_b_before.0 + 1))),
        "no work should have been done for the unrelated function"
    );
}

#[test]
fn unrelated_schema_change_touches_nothing() {
    let mut rt = setup();
    let g_before = current_version(&rt, G);
    let f_before = current_version(&rt, F);

    // Change B (add a field). Only touch_b depends on B, and the change is
    // additive so it stays well-typed — g and f are never even looked at.
    rt.install_schema(Schema {
        type_id: TB,
        version: Version(2),
        name: "B".into(),
        fields: vec![
            field(FB, "fb", Type::I64),
            Field {
                id: 901,
                name: "extra".into(),
                ty: Type::I64,
                default: Some(Value::I64(0)),
            },
        ],
    })
    .unwrap();

    assert_eq!(current_version(&rt, G), g_before);
    assert_eq!(current_version(&rt, F), f_before);
    assert!(!is_broken(&rt, TOUCH_B), "additive change keeps touch_b Ready");
}
