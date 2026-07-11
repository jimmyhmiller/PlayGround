//! Real OS threads over one shared heap. The headline test drives many threads
//! into migrating the *same* object concurrently and asserts they all agree —
//! run over many iterations to shake out races. Because this crate is LLVM-free
//! it can be checked for data races under Miri's detector:
//! `cargo +nightly miri test -p livetype-core --test mt_threads`.
//! (ThreadSanitizer's runtime SIGSEGVs on aarch64-apple-darwin regardless of
//! the code, so Miri is the machine-checked race gate here.)

use livetype_core::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

// Miri interprets (and race-checks) the code, so it is far slower than native.
// Scale the stress loops down under Miri while keeping full counts otherwise.
const MIGRATE_ITERS: usize = if cfg!(miri) { 4 } else { 200 };
const MIGRATE_THREADS: usize = if cfg!(miri) { 3 } else { 8 };
const GC_ITERS: usize = if cfg!(miri) { 2 } else { 20 };
const GC_ACTORS: usize = if cfg!(miri) { 2 } else { 4 };
const CHURN_N: i64 = if cfg!(miri) { 12 } else { 200 };

const ACCT: DefId = 1;
const MONEY: DefId = 2;
const PEEK_INT: DefId = 10;
const PEEK_MONEY: DefId = 11;
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

/// `peek_int(a) -> Int`: read `a.balance` (an Int) and return it.
fn peek_int() -> Function {
    Function {
        id: PEEK_INT,
        version: Version(1),
        name: "peek_int".into(),
        params: vec![Type::Ref(ACCT)],
        result: Type::I64,
        registers: 2,
        code: vec![
            Instruction::GetField {
                dst: 1,
                object: 0,
                field: BALANCE,
            },
            Instruction::Return { value: 1 },
        ],
    }
}

/// `peek_money(a) -> Int`: read `a.balance` (a Money) then its cents.
fn peek_money() -> Function {
    Function {
        id: PEEK_MONEY,
        version: Version(1),
        name: "peek_money".into(),
        params: vec![Type::Ref(ACCT)],
        result: Type::I64,
        registers: 3,
        code: vec![
            Instruction::GetField {
                dst: 1,
                object: 0,
                field: BALANCE,
            },
            Instruction::GetField {
                dst: 2,
                object: 1,
                field: CENTS,
            },
            Instruction::Return { value: 2 },
        ],
    }
}

#[test]
fn threads_share_reads_of_one_object() {
    let mut rt = Runtime::default();
    rt.install_schema(Schema {
        type_id: ACCT,
        version: Version(1),
        name: "Account".into(),
        fields: vec![field(BALANCE, "balance", Type::I64)],
    })
    .unwrap();
    rt.install_function(peek_int()).unwrap();
    let obj = rt.jit_new(ACCT, &[(BALANCE, Value::I64(100))]).unwrap();

    let shared = Shared::from_runtime(rt);
    let outcomes = shared.run_threads(vec![(PEEK_INT, vec![Value::Ref(obj)]); 8]);
    for o in &outcomes {
        assert_eq!(*o, Outcome::Complete(Value::I64(100)));
    }
}

/// Set up a v1 account, allocate it, then move the world to v2 (balance becomes
/// Money) with a migration — leaving the live object lazily at v1. Threads that
/// read it will each trigger the v1→v2 migration concurrently.
fn migratable_shared() -> (Arc<Shared>, ObjectId) {
    let mut rt = Runtime::default();
    rt.install_schema(Schema {
        type_id: ACCT,
        version: Version(1),
        name: "Account".into(),
        fields: vec![field(BALANCE, "balance", Type::I64)],
    })
    .unwrap();
    let obj = rt.jit_new(ACCT, &[(BALANCE, Value::I64(100))]).unwrap();
    rt.install_schema(Schema {
        type_id: MONEY,
        version: Version(1),
        name: "Money".into(),
        fields: vec![field(CENTS, "cents", Type::I64)],
    })
    .unwrap();
    rt.install_schema(Schema {
        type_id: ACCT,
        version: Version(2),
        name: "Account".into(),
        fields: vec![field(BALANCE, "balance", Type::Ref(MONEY))],
    })
    .unwrap();
    rt.install_migration(Migration {
        type_id: ACCT,
        from: Version(1),
        to: Version(2),
        fields: std::collections::BTreeMap::from([(
            BALANCE,
            MigrationSource::Wrap {
                type_id: MONEY,
                field: CENTS,
                source: BALANCE,
            },
        )]),
    })
    .unwrap();
    rt.install_function(peek_money()).unwrap();
    (Shared::from_runtime(rt), obj)
}

#[test]
fn concurrent_migration_of_a_shared_object_is_race_free() {
    // Many iterations, each with 8 threads racing to migrate the same object.
    // A torn read or a lost/duplicated migration would show up as a wrong
    // result or a panic on some interleaving.
    for _ in 0..MIGRATE_ITERS {
        let (shared, obj) = migratable_shared();
        assert_eq!(shared.object_schema(obj), Some(Version(1)));

        let outcomes = shared.run_threads(vec![(PEEK_MONEY, vec![Value::Ref(obj)]); MIGRATE_THREADS]);
        for o in &outcomes {
            assert_eq!(
                *o,
                Outcome::Complete(Value::I64(100)),
                "a thread saw a torn or wrong migration"
            );
        }
        // Exactly one migration won: the object is at v2 and holds a Money.
        assert_eq!(shared.object_schema(obj), Some(Version(2)));
    }
}

#[test]
fn a_single_actor_runs_and_migrates() {
    let (shared, obj) = migratable_shared();
    let outcome = shared.run_actor(0, PEEK_MONEY, vec![Value::Ref(obj)]);
    assert_eq!(outcome, Outcome::Complete(Value::I64(100)));
    assert_eq!(shared.object_schema(obj), Some(Version(2)));
}

#[test]
fn stop_the_world_collect_reclaims_unreachable_objects() {
    let (shared, obj) = migratable_shared();
    // Run the migration (single actor) so the account now references a Money
    // wrapper; that wrapper is live only through the account.
    assert_eq!(
        shared.run_actor(0, PEEK_MONEY, vec![Value::Ref(obj)]),
        Outcome::Complete(Value::I64(100))
    );
    let after_migration = shared.object_count();
    assert!(after_migration >= 2, "account + its Money wrapper exist");

    // Collect from the account as the only root: the account and the Money it
    // reaches survive; any migration garbage is reclaimed.
    let reclaimed = shared.collect(&[obj]);
    let live = shared.object_count();
    assert_eq!(live, 2, "account and its reachable Money survive");
    assert_eq!(reclaimed, after_migration - live);

    // With no roots, everything goes.
    assert_eq!(shared.collect(&[]), 2);
    assert_eq!(shared.object_count(), 0);
}

// --- preemptive stop-the-world GC while threads run ------------------------

const SOME: DefId = 3;
const F: FieldId = 300;
const CHURN: DefId = 12;

/// `churn(n)` loops `n` times, allocating a throwaway object each iteration
/// (only the current one is live), then returns 0. Lots of garbage for a
/// concurrent collector to reclaim while the actor runs.
fn churn() -> Function {
    Function {
        id: CHURN,
        version: Version(1),
        name: "churn".into(),
        params: vec![Type::I64],
        result: Type::I64,
        registers: 6,
        code: vec![
            Instruction::Const {
                dst: 1,
                value: Value::I64(1),
            }, // one
            Instruction::Const {
                dst: 2,
                value: Value::I64(0),
            }, // zero
            Instruction::Const {
                dst: 3,
                value: Value::I64(7),
            }, // 2: loop header
            Instruction::New {
                dst: 4,
                type_id: SOME,
                fields: vec![(F, 3)],
            },
            Instruction::Yield,
            Instruction::SubI64 {
                dst: 0,
                left: 0,
                right: 1,
            }, // n -= 1
            Instruction::LtI64 {
                dst: 5,
                left: 2,
                right: 0,
            }, // 0 < n
            Instruction::Branch {
                cond: 5,
                then_pc: 2,
                else_pc: 8,
            },
            Instruction::Return { value: 0 },
        ],
    }
}

#[test]
fn preemptive_gc_runs_while_actors_churn() {
    for _ in 0..GC_ITERS {
        let mut rt = Runtime::default();
        rt.install_schema(Schema {
            type_id: SOME,
            version: Version(1),
            name: "Some".into(),
            fields: vec![field(F, "f", Type::I64)],
        })
        .unwrap();
        rt.install_function(churn()).unwrap();
        let shared = Shared::from_runtime(rt);

        // A collector thread hammering GC requests while the actors run.
        let done = Arc::new(AtomicBool::new(false));
        let collector = {
            let shared = Arc::clone(&shared);
            let done = Arc::clone(&done);
            std::thread::spawn(move || {
                while !done.load(Ordering::Acquire) {
                    shared.request_gc();
                    std::thread::yield_now();
                }
            })
        };

        // Four actors, each churning 200 allocations, preempted throughout.
        let outcomes = shared.run_threads(vec![(CHURN, vec![Value::I64(CHURN_N)]); GC_ACTORS]);
        done.store(true, Ordering::Release);
        collector.join().unwrap();

        // Every actor finished correctly despite being repeatedly paused and
        // having its garbage swept out from under it.
        for o in &outcomes {
            assert_eq!(*o, Outcome::Complete(Value::I64(0)));
        }
        // Collections actually happened, and nothing leaked once idle.
        assert!(shared.collections() > 0, "no preemptive collection occurred");
        let remaining = shared.object_count();
        assert_eq!(shared.collect(&[]), remaining);
        assert_eq!(shared.object_count(), 0);
    }
}

// --- multi-frame actors (Call/Return) --------------------------------------

const CALLER: DefId = 13;
const SUB: DefId = 14;

fn sub_fn() -> Function {
    Function {
        id: SUB,
        version: Version(1),
        name: "sub".into(),
        params: vec![Type::I64, Type::I64],
        result: Type::I64,
        registers: 3,
        code: vec![
            Instruction::SubI64 { dst: 2, left: 0, right: 1 },
            Instruction::Return { value: 2 },
        ],
    }
}

fn caller_fn() -> Function {
    Function {
        id: CALLER,
        version: Version(1),
        name: "caller".into(),
        params: vec![],
        result: Type::I64,
        registers: 3,
        code: vec![
            Instruction::Const { dst: 0, value: Value::I64(10) },
            Instruction::Const { dst: 1, value: Value::I64(3) },
            Instruction::Call { dst: 2, function: SUB, args: vec![0, 1] },
            Instruction::Return { value: 2 },
        ],
    }
}

#[test]
fn multi_frame_call_in_the_concurrent_tier() {
    let mut rt = Runtime::default();
    rt.install_function(sub_fn()).unwrap();
    rt.install_function(caller_fn()).unwrap();
    let shared = Shared::from_runtime(rt);
    // Several threads each push a call frame and pop it — 10 - 3 = 7.
    let outcomes = shared.run_threads(vec![(CALLER, vec![]); 4]);
    for o in &outcomes {
        assert_eq!(*o, Outcome::Complete(Value::I64(7)));
    }
}

// --- message passing --------------------------------------------------------

const CONSUMER: DefId = 15;
const PRODUCER: DefId = 16;
const CONSUMER_REF: DefId = 17;
const PRODUCER_REF: DefId = 18;
const BOXT: DefId = 6;
const VAL: FieldId = 600;

fn consumer_int() -> Function {
    Function {
        id: CONSUMER,
        version: Version(1),
        name: "consumer".into(),
        params: vec![],
        result: Type::I64,
        registers: 1,
        code: vec![
            Instruction::Recv { dst: 0, ty: Type::I64 },
            Instruction::Return { value: 0 },
        ],
    }
}

/// `producer(target)` sends 42 to actor `target`, then returns 0.
fn producer_int() -> Function {
    Function {
        id: PRODUCER,
        version: Version(1),
        name: "producer".into(),
        params: vec![Type::I64],
        result: Type::I64,
        registers: 3,
        code: vec![
            Instruction::Const { dst: 1, value: Value::I64(42) },
            Instruction::Send { target: 0, value: 1 },
            Instruction::Const { dst: 2, value: Value::I64(0) },
            Instruction::Return { value: 2 },
        ],
    }
}

#[test]
fn message_passing_between_actors() {
    let mut rt = Runtime::default();
    rt.install_function(consumer_int()).unwrap();
    rt.install_function(producer_int()).unwrap();
    let shared = Shared::from_runtime(rt);
    // Actor 0 waits for a message; actor 1 sends it 42.
    let outcomes = shared.run_threads(vec![(CONSUMER, vec![]), (PRODUCER, vec![Value::I64(0)])]);
    assert_eq!(outcomes[0], Outcome::Complete(Value::I64(42)));
    assert_eq!(outcomes[1], Outcome::Complete(Value::I64(0)));
}

fn consumer_ref() -> Function {
    Function {
        id: CONSUMER_REF,
        version: Version(1),
        name: "consumer_ref".into(),
        params: vec![],
        result: Type::I64,
        registers: 2,
        code: vec![
            Instruction::Recv { dst: 0, ty: Type::Ref(BOXT) },
            Instruction::GetField { dst: 1, object: 0, field: VAL },
            Instruction::Return { value: 1 },
        ],
    }
}

/// `producer_ref(boxref, target)` hands a shared object to another actor.
fn producer_ref() -> Function {
    Function {
        id: PRODUCER_REF,
        version: Version(1),
        name: "producer_ref".into(),
        params: vec![Type::Ref(BOXT), Type::I64],
        result: Type::I64,
        registers: 3,
        code: vec![
            Instruction::Send { target: 1, value: 0 },
            Instruction::Const { dst: 2, value: Value::I64(0) },
            Instruction::Return { value: 2 },
        ],
    }
}

#[test]
fn message_passing_shares_a_heap_reference() {
    let mut rt = Runtime::default();
    rt.install_schema(Schema {
        type_id: BOXT,
        version: Version(1),
        name: "Box".into(),
        fields: vec![field(VAL, "value", Type::I64)],
    })
    .unwrap();
    let boxid = rt.jit_new(BOXT, &[(VAL, Value::I64(99))]).unwrap();
    rt.install_function(consumer_ref()).unwrap();
    rt.install_function(producer_ref()).unwrap();
    let shared = Shared::from_runtime(rt);
    // Actor 1 sends the shared Box to actor 0, which reads its field.
    let outcomes = shared.run_threads(vec![
        (CONSUMER_REF, vec![]),
        (PRODUCER_REF, vec![Value::Ref(boxid), Value::I64(0)]),
    ]);
    assert_eq!(outcomes[0], Outcome::Complete(Value::I64(99)));
    assert_eq!(outcomes[1], Outcome::Complete(Value::I64(0)));
}

#[test]
fn message_type_mismatch_traps_the_receiver() {
    // The consumer's mailbox contract says Int, but a Ref is sent: receiving it
    // traps, exactly like any other con-freeness violation.
    let mut rt = Runtime::default();
    rt.install_schema(Schema {
        type_id: BOXT,
        version: Version(1),
        name: "Box".into(),
        fields: vec![field(VAL, "value", Type::I64)],
    })
    .unwrap();
    let boxid = rt.jit_new(BOXT, &[(VAL, Value::I64(1))]).unwrap();
    rt.install_function(consumer_int()).unwrap(); // expects an Int message
    rt.install_function(producer_ref()).unwrap(); // sends a Ref
    let shared = Shared::from_runtime(rt);
    let outcomes = shared.run_threads(vec![
        (CONSUMER, vec![]),
        (PRODUCER_REF, vec![Value::Ref(boxid), Value::I64(0)]),
    ]);
    assert!(
        matches!(outcomes[0], Outcome::Paused(Condition::RuntimeTypeError { .. })),
        "receiving a wrong-typed message should trap, got {:?}",
        outcomes[0]
    );
}
