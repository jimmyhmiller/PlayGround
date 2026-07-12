//! Hot reloading composed with auto-tiering: a function runs, gets promoted to
//! JIT, and every live-edit flow still behaves *exactly* as it does on the pure
//! interpreter — the edit is picked up, a breaking edit traps the JIT frame
//! (soundness holds, no stale native execution), and trap → repair → resume
//! works with a JIT frame on the stack.

use livetype::*;

/// `helper()` returning the constant `n` at `version` (i64).
fn helper_i64(id: DefId, version: u64, n: i64) -> Function {
    Function {
        id,
        version: Version(version),
        name: "helper".into(),
        params: vec![],
        result: Type::I64,
        registers: 1,
        code: vec![
            Instruction::Const { dst: 0, value: Value::I64(n) },
            Instruction::Return { value: 0 },
        ],
    }
}

/// A broken `helper()` — declares i64 but returns a bool, so it installs Broken.
fn helper_broken(id: DefId, version: u64) -> Function {
    Function {
        id,
        version: Version(version),
        name: "helper".into(),
        params: vec![],
        result: Type::I64,
        registers: 1,
        code: vec![
            Instruction::Const { dst: 0, value: Value::Bool(true) },
            Instruction::Return { value: 0 },
        ],
    }
}

#[test]
fn editing_a_promoted_function_is_picked_up() {
    // `f` is called 50x (threshold 10) so it is promoted to JIT during the run.
    let mut s = Session::new();
    s.eval(
        "fn f(x: i64) -> i64 { x + 1 }
         fn main() -> i64 {
            let s = 0;
            let i = 0;
            while i < 50 { s = f(i); i = i + 1; }
            s
         }",
    )
    .unwrap();
    let f = s.fn_id("f").unwrap();
    let main = s.fn_id("main").unwrap();

    let mut tiered = Tiered::new(s.runtime, 10);
    assert_eq!(tiered.run(main, vec![]), Outcome::Complete(Value::I64(50))); // f(49)=50
    assert!(tiered.is_hot(f, Version(1)), "f must have been promoted");

    // Hot-reload f: x + 100. A running frame would keep the pinned version, but
    // the next run re-resolves the current one.
    let f_v2 = Function {
        id: f,
        version: Version(2),
        name: "f".into(),
        params: vec![Type::I64],
        result: Type::I64,
        registers: 3,
        code: vec![
            Instruction::Const { dst: 1, value: Value::I64(100) },
            Instruction::AddI64 { dst: 2, left: 0, right: 1 },
            Instruction::Return { value: 2 },
        ],
    };
    tiered.install_function(f_v2).unwrap();

    // Re-run: the edit is live (f(49) = 149), even though the old f was JITted.
    assert_eq!(tiered.run(main, vec![]), Outcome::Complete(Value::I64(149)));
}

#[test]
fn breaking_edit_traps_a_jit_caller_then_repair_resumes() {
    let mut s = Session::new();
    s.eval(
        "fn helper() -> i64 { 7 }
         fn compute() -> i64 { helper() }
         fn main() -> i64 {
            let s = 0;
            let i = 0;
            while i < 30 { s = compute(); i = i + 1; }
            s
         }",
    )
    .unwrap();
    let helper = s.fn_id("helper").unwrap();
    let compute = s.fn_id("compute").unwrap();
    let main = s.fn_id("main").unwrap();

    // Threshold 5: `compute` (and `helper`) are promoted during the first run.
    let mut tiered = Tiered::new(s.runtime, 5);
    assert_eq!(tiered.run(main, vec![]), Outcome::Complete(Value::I64(7)));
    assert!(tiered.is_hot(compute, Version(1)), "compute must be promoted");
    assert!(tiered.is_hot(helper, Version(1)), "helper must be promoted");

    // SOUNDNESS: break `helper`. On the next run the JIT-compiled `compute`
    // calls it — and traps, rather than running stale native code.
    tiered.install_function(helper_broken(helper, 2)).unwrap();
    let trap = tiered.run(main, vec![]);
    assert!(
        matches!(trap, Outcome::Paused(Condition::BrokenFunction { function, .. }) if function == helper),
        "a JIT caller of a now-broken function must trap; got {trap:?}"
    );

    // REPAIR + RESUME: redefine `helper`, then resume the *same* paused stack
    // (which has a JIT `compute` frame on it). It re-runs its call to the
    // repaired helper and completes — hot-reload's repair half, through the JIT.
    tiered.install_function(helper_i64(helper, 3, 7)).unwrap();
    assert_eq!(tiered.resume(), Outcome::Complete(Value::I64(7)));
}

#[test]
fn data_migration_is_transparent_to_a_jit_reader() {
    // A JIT-promoted reader of a struct field keeps working after the schema is
    // migrated live: its `GetField` goes through the same migration barrier the
    // interpreter uses (the `lt_get_field` extern), so the object migrates
    // lazily and the read stays correct.
    let src = "struct Account { balance: i64 }
               letonce acct = Account { balance: 5 };
               fn read() -> i64 { acct.balance }
               fn main() -> i64 {
                  let s = 0;
                  let i = 0;
                  while i < 30 { s = read(); i = i + 1; }
                  s
               }";
    let mut s = Session::new();
    s.eval(src).unwrap();
    let read = s.fn_id("read").unwrap();
    let main = s.fn_id("main").unwrap();
    let account = s.struct_id("Account").unwrap();

    let mut tiered = Tiered::new(s.runtime, 5);
    assert_eq!(tiered.run(main, vec![]), Outcome::Complete(Value::I64(5)));
    assert!(tiered.is_hot(read, Version(1)), "read must be promoted");

    // Build the migrated schema via a twin session that evals the same source
    // (so DefIds/field ids match) and then the struct edit, producing Account v2.
    let mut twin = Session::new();
    twin.eval(src).unwrap();
    twin.eval("struct Account { balance: i64, fee: i64 = 0 }").unwrap();
    let account_v2 = twin.runtime.world.schemas[&(account, Version(2))].clone();

    // Hot-reload the schema; the auto-derived migration is created on install.
    tiered.install_schema(account_v2).unwrap();

    // Re-run: the JIT-compiled `read` migrates `acct` on first access and still
    // returns the correct balance.
    assert_eq!(tiered.run(main, vec![]), Outcome::Complete(Value::I64(5)));
}
