//! End-to-end demo: the Account/Money hot-update scenario driven by BOTH
//! configurations of the one engine — never-promote (interpreter) and
//! always-JIT — showing they produce identical effects, pauses, and results.
//! This is `RUNTIME_DESIGN.md` made real: LLVM accelerates execution while the
//! engine keeps pause/repair/resume.

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

/// Account.balance : Int, `charge` subtracts, `main` opens an account, emits its
/// balance, hits a Yield safe point, then charges 5. The Yield lets both
/// executors stop at the same place before the hot update lands.
fn setup(rt: &Engine) {
    rt.install_schema(Schema {
        type_id: ACCOUNT,
        version: Version(1),
        name: "Account".into(),
        fields: vec![field(BALANCE, "balance", Type::I64)],
        variants: Vec::new(),
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

// The hot update: balance becomes Money (breaks `charge`), then the fix, then
// the migration.
fn update_break(rt: &Engine) {
    rt.install_schema(Schema {
        type_id: MONEY,
        version: Version(1),
        name: "Money".into(),
        fields: vec![field(CENTS, "cents", Type::I64)],
        variants: Vec::new(),
    })
    .unwrap();
    rt.install_schema(Schema {
        type_id: ACCOUNT,
        version: Version(2),
        name: "Account".into(),
        fields: vec![field(BALANCE, "balance", Type::Ref(MONEY))],
        variants: Vec::new(),
    })
    .unwrap();
}

fn update_fix(rt: &Engine) {
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

fn update_migrate(rt: &Engine) {
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
        variants: std::collections::BTreeMap::new(),
    })
    .unwrap();
}

/// Drive the scenario on one engine configuration, narrating each phase.
fn run_config(label: &str, rt: std::sync::Arc<Engine>) -> (ActorStatus, Vec<Value>) {
    println!("── {label} ──");
    setup(&rt);
    let mut actor = rt.spawn(MAIN, vec![]).unwrap();
    // Run to the yield safe point (Const, New, Emit, Yield).
    loop {
        match rt.step(&mut actor) {
            Turn::Progress => {}
            _ => break,
        }
    }
    println!("  at yield, effects so far: {:?}", rt.output());

    update_break(&rt);
    rt.run(&mut actor);
    println!("  after update:   {:?}", actor.status);

    update_fix(&rt);
    rt.resume(&mut actor);
    println!("  after fix:      {:?}", actor.status);

    update_migrate(&rt);
    rt.resume(&mut actor);
    println!("  after migrate:  {:?}", actor.status);
    (actor.status.clone(), rt.output())
}

/// Compile a program written in the Rust-flavored surface syntax and run it.
fn run_frontend() {
    println!("── frontend (source → IR → run) ──");
    let source = r#"
        struct Account {
            balance: i64,
            fee: i64 = 0,
        }
        fn charge(a: Account, amt: i64) -> i64 {
            let b = a.balance;
            return b - amt;
        }
        fn main() -> i64 {
            let acct = Account { balance: 100 };
            emit(acct.balance);
            return charge(acct, 5);
        }
    "#;
    let compiled = livetype_core::compile(source).expect("compile");
    let main_id = compiled.functions["main"];
    let outcome = compiled.engine.run_call(main_id, vec![]);
    println!(
        "  result: {:?}; effects: {:?}",
        outcome,
        compiled.engine.output()
    );
}

fn main() {
    run_frontend();
    println!();
    let (interp_status, interp_out) = run_config("interpreter", Engine::interp());
    println!();
    let (jit_status, jit_out) = run_config("llvm jit", jit_engine(0));
    println!();
    let matched = interp_status == jit_status && interp_out == jit_out;
    println!(
        "interpreter and jit agree: {matched}  (status {:?}, effects {:?})",
        jit_status, jit_out
    );
    assert!(matched, "configurations diverged");
}
