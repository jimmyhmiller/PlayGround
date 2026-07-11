//! A live-programming session, narrated. A bank service runs; we edit its code
//! and types *from source while it is running* — adding a field (data migrates
//! transparently), then making a breaking change (the running computation traps
//! at the point of use, not crashes), then repairing it live so it resumes.
//!
//! Run with: `cargo run --bin livetype-poc`

use livetype::*;
use livetype_core::Session;

fn main() {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  Live & Typed — editing a running program from source        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    let mut s = Session::new();

    // ── v1: the program we start with ──────────────────────────────────────
    edit(
        &mut s,
        "boot",
        r#"
struct Account { balance: i64 }

fn charge(a: Account, amt: i64) -> i64 {
    a.balance - amt
}

fn main() -> i64 {
    let acct = Account { balance: 100 };
    let i = 0;
    while i < 6 {
        emit(charge(acct, 5));   // what the customer would be charged down to
        yield;                   // a safe point between requests
        i = i + 1;
    }
    0
}
"#,
    );

    let main = s.fn_id("main").unwrap();
    let actor = s.runtime.spawn(main, vec![]).unwrap();
    let mut seen = 0;

    narrate("start the service; let two requests go by");
    run_to_yield(&mut s, actor, &mut seen);
    run_to_yield(&mut s, actor, &mut seen);

    // ── live edit 1: additive change, auto-migrated ────────────────────────
    edit(
        &mut s,
        "live edit #1 — add a fee, change the charge",
        r#"
struct Account { balance: i64, fee: i64 = 3 }

fn charge(a: Account, amt: i64) -> i64 {
    a.balance - amt - a.fee
}
"#,
    );
    narrate("resume — the SAME running loop now charges the new way, and the");
    narrate("live account grew a `fee` field with no restart (auto-migrated)");
    run_to_yield(&mut s, actor, &mut seen);
    run_to_yield(&mut s, actor, &mut seen);

    // ── live edit 2: a breaking change ─────────────────────────────────────
    edit(
        &mut s,
        "live edit #2 — balance becomes Money (a breaking change)",
        r#"
struct Money { cents: i64 }
struct Account { balance: Money }
"#,
    );
    narrate("resume — `charge` now computes `Money - i64`, which is inconsistent.");
    narrate("the running loop does not crash: it FREEZES at the point of use.");
    run_to_yield(&mut s, actor, &mut seen);
    report_status(&s, actor);

    // ── repair, live ───────────────────────────────────────────────────────
    edit(
        &mut s,
        "repair — teach `charge` about Money",
        r#"
fn charge(a: Account, amt: i64) -> i64 {
    a.balance.cents - amt
}
"#,
    );
    narrate("resume — the fixed `charge` type-checks, so the frozen frame thaws...");
    run_to_yield(&mut s, actor, &mut seen);
    report_status(&s, actor);
    narrate("...and now the live account still holds an old-shaped balance, so it");
    narrate("pauses for a migration. We supply the Int→Money transformer:");
    supply_money_migration(&mut s);
    narrate("resume — migrated, and the service runs on. Never restarted.");
    run_to_yield(&mut s, actor, &mut seen);
    run_to_yield(&mut s, actor, &mut seen);
    report_status(&s, actor);

    println!("\nAll of that — behavior change, data migration, a breaking edit, a");
    println!("trap, and a live repair — happened to ONE running program, from source.\n");
}

// ── helpers ────────────────────────────────────────────────────────────────

fn edit(s: &mut Session, title: &str, src: &str) {
    println!("\n┏━ {title} ");
    for line in src.trim_matches('\n').lines() {
        println!("┃   {line}");
    }
    match s.eval(src) {
        Ok(()) => println!("┗━ installed."),
        Err(e) => println!("┗━ compile error: {e}"),
    }
}

fn narrate(msg: &str) {
    println!("   ▸ {msg}");
}

/// Step the running actor to its next `Yield` (or a stop), printing any newly
/// emitted values.
fn run_to_yield(s: &mut Session, actor: ActorId, seen: &mut usize) {
    while matches!(s.runtime.actors[&actor].status, ActorStatus::Runnable) {
        let (key, pc) = {
            let f = s.runtime.actors[&actor].frames.last().unwrap();
            (f.function, f.pc)
        };
        let was_yield = matches!(
            &s.runtime.world.functions[&key],
            FunctionState::Ready(f) if matches!(f.code[pc], Instruction::Yield)
        );
        s.runtime.step(actor);
        if was_yield {
            break;
        }
    }
    let out = &s.runtime.output;
    for v in &out[*seen..] {
        if let Value::I64(n) = v {
            println!("     · charged down to {n}");
        }
    }
    *seen = out.len();
}

fn report_status(s: &Session, actor: ActorId) {
    match &s.runtime.actors[&actor].status {
        ActorStatus::Complete(v) => println!("     ⏹ finished: {v:?}"),
        ActorStatus::Runnable => {}
        ActorStatus::Paused(Condition::BrokenFunction { function, .. }) => {
            println!("     ⏸ FROZEN: function #{function} no longer type-checks")
        }
        ActorStatus::Paused(Condition::MissingMigration { type_id, from, to, .. }) => {
            println!("     ⏸ FROZEN: type #{type_id} needs a v{}→v{} migration", from.0, to.0)
        }
        ActorStatus::Paused(Condition::RuntimeTypeError { message, .. }) => {
            println!("     ⏸ FROZEN: {message}")
        }
    }
}

/// Supply the Int→Money migration for `Account.balance` through the runtime API
/// (migration transformers aren't in the surface syntax yet).
fn supply_money_migration(s: &mut Session) {
    let account = s.struct_id("Account").unwrap();
    let money = s.struct_id("Money").unwrap();
    let bal = field_id(s, "Account", "balance");
    let cents = field_id(s, "Money", "cents");
    let to = s.runtime.world.current_schemas[&account];
    s.runtime
        .install_migration(Migration {
            type_id: account,
            from: Version(to.0 - 1),
            to,
            fields: std::collections::BTreeMap::from([(
                bal,
                MigrationSource::Wrap { type_id: money, field: cents, source: bal },
            )]),
        })
        .unwrap();
    println!("     ✎ migration installed: wrap the old balance into Money{{cents}}");
}

fn field_id(s: &Session, struct_name: &str, field_name: &str) -> FieldId {
    let tid = s.struct_id(struct_name).unwrap();
    let v = s.runtime.world.current_schemas[&tid];
    s.runtime.world.schemas[&(tid, v)]
        .fields
        .iter()
        .find(|f| f.name == field_name)
        .unwrap()
        .id
}
