//! An interactive live-programming REPL. Type struct/function definitions to
//! edit the world *while a program is running*; drive the running actor with
//! `:go`; repair with more definitions (and `:migrate`) when it freezes.
//!
//! Run with: `cargo run --bin livetype-repl`

use livetype::*;
use livetype_core::Session;
use std::collections::BTreeMap;
use std::io::{self, BufRead, Write};

fn main() {
    banner();
    let mut s = Session::new();
    let mut actor: Option<ActorId> = None;
    let mut seen = 0usize; // emitted values already shown for `actor`

    let mut buf = String::new();
    prompt(false);
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };

        // A command only when we're not mid-definition.
        if buf.trim().is_empty() && line.trim_start().starts_with(':') {
            buf.clear();
            if !command(line.trim(), &mut s, &mut actor, &mut seen) {
                break;
            }
            prompt(false);
            continue;
        }

        buf.push_str(&line);
        buf.push('\n');
        if braces_balanced(&buf) && !buf.trim().is_empty() {
            match s.eval(&buf) {
                Ok(()) => {
                    println!("  ✓ installed");
                    note_resumable(&s, actor);
                }
                Err(e) => println!("  ✗ {e}"),
            }
            buf.clear();
            prompt(false);
        } else {
            prompt(true);
        }
    }
    println!();
}

fn banner() {
    println!("Live & Typed — interactive live editing");
    println!("  · type `struct`/`fn` definitions to edit the world (live)");
    println!("  · `:run main`   start a function running     `:go`   advance to the next yield / pause");
    println!("  · `:status`     where the running actor is    `:out`  everything it has emitted");
    println!("  · `:migrate T.f = wrap M.c | copy g | 42`     supply a migration when one is needed");
    println!("  · `:defs`  `:reset`  `:help`  `:quit`");
    println!("Tip: give `main` a `while` loop with a `yield` inside, then edit between `:go`s.\n");
}

fn prompt(cont: bool) {
    print!("{}", if cont { "  ... " } else { "lt> " });
    let _ = io::stdout().flush();
}

/// Naive brace balance (good enough — the only nesting is `{ }`).
fn braces_balanced(s: &str) -> bool {
    let mut depth = 0i32;
    for c in s.chars() {
        match c {
            '{' => depth += 1,
            '}' => depth -= 1,
            _ => {}
        }
    }
    depth <= 0
}

/// Returns false to quit.
fn command(cmd: &str, s: &mut Session, actor: &mut Option<ActorId>, seen: &mut usize) -> bool {
    let mut parts = cmd.split_whitespace();
    let head = parts.next().unwrap_or("");
    let rest: Vec<&str> = parts.collect();
    match head {
        ":quit" | ":q" => return false,
        ":help" | ":h" => banner(),
        ":defs" => show_defs(s),
        ":reset" => {
            *s = Session::new();
            *actor = None;
            *seen = 0;
            println!("  (fresh session)");
        }
        ":run" => run_actor(s, actor, seen, &rest),
        ":go" => go(s, actor, seen),
        ":status" => match actor {
            Some(a) => print_status(s, *a),
            None => println!("  (no actor — `:run <fn>` first)"),
        },
        ":out" => match actor {
            Some(_) => println!("  emitted: {:?}", s.runtime.output),
            None => println!("  (nothing running)"),
        },
        ":migrate" => migrate(s, &cmd[":migrate".len()..]),
        other => println!("  unknown command `{other}` — try `:help`"),
    }
    true
}

fn show_defs(s: &Session) {
    let mut structs: Vec<_> = s.runtime.world.current_schemas.keys().copied().collect();
    structs.sort_unstable();
    for tid in structs {
        let v = s.runtime.world.current_schemas[&tid];
        let sc = &s.runtime.world.schemas[&(tid, v)];
        let fields: Vec<String> = sc
            .fields
            .iter()
            .map(|f| format!("{}: {}", f.name, ty_name(&f.ty, s)))
            .collect();
        println!("  struct {} {{ {} }}  (v{})", sc.name, fields.join(", "), v.0);
    }
    let mut fns: Vec<_> = s.runtime.world.current_functions.keys().copied().collect();
    fns.sort_unstable();
    for fid in fns {
        let v = s.runtime.world.current_functions[&fid];
        match &s.runtime.world.functions[&(fid, v)] {
            FunctionState::Ready(f) => println!("  fn {}  (v{}, ready)", f.name, v.0),
            FunctionState::Broken { name, .. } => println!("  fn {}  (v{}, BROKEN)", name, v.0),
        }
    }
}

fn run_actor(s: &mut Session, actor: &mut Option<ActorId>, seen: &mut usize, rest: &[&str]) {
    let Some(&name) = rest.first() else {
        println!("  usage: :run <fn> [i64 args...]");
        return;
    };
    let Some(id) = s.fn_id(name) else {
        println!("  no function `{name}`");
        return;
    };
    let args: Vec<Value> = rest[1..]
        .iter()
        .filter_map(|a| a.parse::<i64>().ok().map(Value::I64))
        .collect();
    match s.runtime.spawn(id, args) {
        Ok(a) => {
            *actor = Some(a);
            *seen = 0;
            println!("  ▸ running `{name}`");
            go(s, actor, seen);
        }
        Err(c) => println!("  cannot start: {c:?}"),
    }
}

/// Advance the current actor to its next `Yield`, pause, or completion.
fn go(s: &mut Session, actor: &mut Option<ActorId>, seen: &mut usize) {
    let Some(a) = *actor else {
        println!("  (no actor — `:run <fn>` first)");
        return;
    };
    // A repairing edit (a fixed function, or an installed migration) already
    // flipped a paused actor back to Runnable; if it's still paused, it hasn't
    // been repaired yet.
    if !matches!(s.runtime.actors[&a].status, ActorStatus::Runnable) {
        print_status(s, a);
        return;
    }
    while matches!(s.runtime.actors[&a].status, ActorStatus::Runnable) {
        let (key, pc) = {
            let f = s.runtime.actors[&a].frames.last().unwrap();
            (f.function, f.pc)
        };
        let was_yield = matches!(
            &s.runtime.world.functions[&key],
            FunctionState::Ready(f) if matches!(f.code[pc], Instruction::Yield)
        );
        s.runtime.step(a);
        if was_yield {
            break;
        }
    }
    for v in &s.runtime.output[*seen..] {
        println!("     → {}", show(v));
    }
    *seen = s.runtime.output.len();
    print_status(s, a);
}

fn print_status(s: &Session, a: ActorId) {
    match &s.runtime.actors[&a].status {
        ActorStatus::Runnable => println!("  ⏵ at a yield (safe point) — edit, then `:go`"),
        ActorStatus::Complete(v) => println!("  ⏹ finished: {}", show(v)),
        ActorStatus::Paused(Condition::BrokenFunction { function, diagnostics }) => {
            println!("  ⏸ FROZEN — {} no longer type-checks:", fn_name(s, *function));
            for d in diagnostics {
                println!("       {d}");
            }
            println!("     (redefine it to repair, then `:go`)");
        }
        ActorStatus::Paused(Condition::MissingMigration { type_id, from, to, .. }) => {
            println!(
                "  ⏸ FROZEN — {} needs a v{}→v{} migration",
                struct_name(s, *type_id),
                from.0,
                to.0
            );
            println!("     (supply it: `:migrate {}.<field> = wrap <T>.<f>`, then `:go`)", struct_name(s, *type_id));
        }
        ActorStatus::Paused(Condition::RuntimeTypeError { message, .. }) => {
            println!("  ⏸ FROZEN — {message}");
        }
    }
}

fn note_resumable(s: &Session, actor: Option<ActorId>) {
    if let Some(a) = actor {
        if matches!(s.runtime.actors[&a].status, ActorStatus::Runnable) {
            println!("  ↻ the running actor can resume — `:go`");
        }
    }
}

/// `:migrate Account.balance = wrap Money.cents` (or `copy old` or `42`). Fills
/// the rest of the target type's fields by auto-derivation.
fn migrate(s: &mut Session, arg: &str) {
    let Some((lhs, rhs)) = arg.split_once('=') else {
        println!("  usage: :migrate Type.field = wrap T.f | copy g | <int>");
        return;
    };
    let lhs = lhs.trim();
    let Some((ty_name, field_name)) = lhs.split_once('.') else {
        println!("  usage: :migrate Type.field = ...");
        return;
    };
    let (ty_name, field_name) = (ty_name.trim(), field_name.trim());
    let Some(type_id) = s.struct_id(ty_name) else {
        println!("  no struct `{ty_name}`");
        return;
    };
    let to = s.runtime.world.current_schemas[&type_id];
    if to.0 < 2 {
        println!("  `{ty_name}` has only one version — nothing to migrate");
        return;
    }
    let from = Version(to.0 - 1);
    let old = s.runtime.world.schemas[&(type_id, from)].clone();
    let new = s.runtime.world.schemas[&(type_id, to)].clone();

    // Parse the user's source for the named field.
    let src_tokens: Vec<&str> = rhs.trim().split_whitespace().collect();
    let user_field_id = new.fields.iter().find(|f| f.name == field_name).map(|f| f.id);
    let Some(user_field_id) = user_field_id else {
        println!("  `{ty_name}` (new version) has no field `{field_name}`");
        return;
    };
    let user_source = match parse_source(s, &old, &src_tokens) {
        Ok(src) => src,
        Err(e) => {
            println!("  {e}");
            return;
        }
    };

    // Build the full migration: user's field explicit, the rest auto-derived.
    let mut fields = BTreeMap::new();
    for f in &new.fields {
        if f.id == user_field_id {
            fields.insert(f.id, user_source.clone());
            continue;
        }
        match old.field(f.id) {
            Some(of) if of.ty == f.ty => {
                fields.insert(f.id, MigrationSource::Copy(f.id));
            }
            _ => match &f.default {
                Some(d) => {
                    fields.insert(f.id, MigrationSource::Value(d.clone()));
                }
                None => {
                    println!("  field `{}` also needs a migration; add another `:migrate`", f.name);
                    return;
                }
            },
        }
    }
    match s.runtime.install_migration(Migration { type_id, from, to, fields }) {
        Ok(()) => println!("  ✎ migration installed for {ty_name} v{}→v{}", from.0, to.0),
        Err(e) => println!("  ✗ {e:?}"),
    }
}

fn parse_source(s: &Session, old: &Schema, tokens: &[&str]) -> Result<MigrationSource, String> {
    match tokens {
        ["wrap", spec] => {
            let (wt, wf) = spec.split_once('.').ok_or("wrap needs `Type.field`")?;
            let wtid = s.struct_id(wt).ok_or_else(|| format!("no struct `{wt}`"))?;
            let wv = s.runtime.world.current_schemas[&wtid];
            let wfid = s.runtime.world.schemas[&(wtid, wv)]
                .fields
                .iter()
                .find(|f| f.name == wf)
                .ok_or_else(|| format!("`{wt}` has no field `{wf}`"))?
                .id;
            // Wrap the type's single old field that is being migrated: use the
            // old field of the same name as the target (its value carries over).
            let src = old
                .fields
                .first()
                .map(|f| f.id)
                .ok_or("nothing to wrap")?;
            Ok(MigrationSource::Wrap { type_id: wtid, field: wfid, source: src })
        }
        ["copy", name] => {
            let id = old
                .fields
                .iter()
                .find(|f| f.name == *name)
                .ok_or_else(|| format!("old version has no field `{name}`"))?
                .id;
            Ok(MigrationSource::Copy(id))
        }
        [lit] => lit
            .parse::<i64>()
            .map(|n| MigrationSource::Value(Value::I64(n)))
            .map_err(|_| format!("don't understand `{lit}` (use `wrap T.f`, `copy g`, or an int)")),
        _ => Err("usage: = wrap T.f | copy g | <int>".into()),
    }
}

// ── small display helpers ───────────────────────────────────────────────────

fn show(v: &Value) -> String {
    match v {
        Value::I64(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Unit => "()".into(),
        Value::Ref(id) => format!("<ref #{id}>"),
    }
}

fn ty_name(t: &Type, s: &Session) -> String {
    match t {
        Type::I64 => "i64".into(),
        Type::Bool => "bool".into(),
        Type::Unit => "()".into(),
        Type::Ref(id) => struct_name(s, *id),
    }
}

fn struct_name(s: &Session, id: DefId) -> String {
    s.runtime
        .world
        .current_schemas
        .get(&id)
        .and_then(|v| s.runtime.world.schemas.get(&(id, *v)))
        .map(|sc| sc.name.clone())
        .unwrap_or_else(|| format!("#{id}"))
}

fn fn_name(s: &Session, id: DefId) -> String {
    s.runtime
        .world
        .current_functions
        .get(&id)
        .map(|v| match &s.runtime.world.functions[&(id, *v)] {
            FunctionState::Ready(f) => f.name.clone(),
            FunctionState::Broken { name, .. } => name.clone(),
        })
        .unwrap_or_else(|| format!("#{id}"))
}
