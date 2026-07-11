//! End-to-end demo: the Account/Money hot-update scenario driven by BOTH
//! executors — the reference interpreter and the LLVM `step` backend — showing
//! they produce identical effects, pauses, and results. This is
//! `RUNTIME_DESIGN.md` made real: LLVM accelerates execution while the runtime
//! keeps pause/repair/resume.

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
fn setup() -> Runtime {
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

// The hot update: balance becomes Money (breaks `charge`), then the fix, then
// the migration.
fn update_break(rt: &mut Runtime) {
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

fn update_fix(rt: &mut Runtime) {
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

fn update_migrate(rt: &mut Runtime) {
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

/// Drive the scenario on the interpreter, narrating each phase.
fn run_interpreter() -> (ActorStatus, Vec<Value>) {
    println!("── interpreter ──");
    let mut rt = setup();
    let actor = rt.spawn(MAIN, vec![]).unwrap();
    // Run to the yield safe point (Const, New, Emit, Yield).
    for _ in 0..4 {
        rt.step(actor);
    }
    println!("  at yield, effects so far: {:?}", rt.output);

    update_break(&mut rt);
    rt.run();
    println!("  after update:   {:?}", rt.actors[&actor].status);

    update_fix(&mut rt);
    rt.run();
    println!("  after fix:      {:?}", rt.actors[&actor].status);

    update_migrate(&mut rt);
    rt.run();
    println!("  after migrate:  {:?}", rt.actors[&actor].status);
    (rt.actors[&actor].status.clone(), rt.output.clone())
}

/// Drive the same scenario on the LLVM `step` backend, narrating each phase.
fn run_jit() -> (ActorStatus, Vec<Value>) {
    println!("── llvm jit ──");
    let mut rt = setup();
    let mut actor = JitActor::spawn(&rt, 1, MAIN, vec![]).unwrap();
    drive(&mut rt, &mut actor, true).unwrap();
    println!("  at yield, effects so far: {:?}", rt.output);

    update_break(&mut rt);
    drive(&mut rt, &mut actor, false).unwrap();
    println!("  after update:   {:?}", actor.status);

    update_fix(&mut rt);
    drive(&mut rt, &mut actor, false).unwrap();
    println!("  after fix:      {:?}", actor.status);

    update_migrate(&mut rt);
    drive(&mut rt, &mut actor, false).unwrap();
    println!("  after migrate:  {:?}", actor.status);
    (actor.status.clone(), rt.output.clone())
}

/// Compile a program written in the Rust-flavored surface syntax and run it.
fn run_frontend() {
    println!("── frontend (source → IR → run) ──");
    let source = r#"
        struct Account {
            balance: i64,
            fee: i64 = 0,
        }
        fn charge(a: &Account, amt: i64) -> i64 {
            let b = a.balance;
            return b - amt;
        }
        fn main() -> i64 {
            let acct = Account { balance: 100 };
            emit(acct.balance);
            return charge(acct, 5);
        }
    "#;
    let mut compiled = livetype_core::compile(source).expect("compile");
    let main_id = compiled.functions["main"];
    let actor = compiled.runtime.spawn(main_id, vec![]).unwrap();
    compiled.runtime.run();
    println!(
        "  result: {:?}; effects: {:?}",
        compiled.runtime.actors[&actor].status, compiled.runtime.output
    );
}

fn main() {
    run_frontend();
    println!();
    let (interp_status, interp_out) = run_interpreter();
    println!();
    let (jit_status, jit_out) = run_jit();
    println!();
    let matched = interp_status == jit_status && interp_out == jit_out;
    println!(
        "interpreter and jit agree: {matched}  (status {:?}, effects {:?})",
        jit_status, jit_out
    );
    assert!(matched, "executors diverged");
}
