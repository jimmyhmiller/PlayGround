//! An interactive live-programming REPL. Type struct/function definitions to
//! edit the world *while a program is running*; drive the running actor with
//! `:go`; repair with more definitions (and `:migrate`) when it freezes.
//!
//! Run with: `cargo run --bin livetype-repl`

use livetype::*;
use livetype_core::Session;
use std::collections::BTreeMap;
use std::io::{self, BufRead, Write};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// A message to the background driver during a `:live` run.
enum ToDriver {
    Edit(String),
    Stop,
}

/// A running `:live` session: the driver thread owns the `Session` and returns
/// it when stopped.
struct Live {
    tx: Sender<ToDriver>,
    handle: JoinHandle<Session>,
}

fn main() {
    banner();
    // The session is owned by the main thread while idle, and moved to the
    // driver thread during a `:live` run (reclaimed on `:stop`).
    let mut s: Option<Session> = Some(Session::new());
    let mut actor: Option<Actor> = None;
    let mut seen = 0usize;
    let mut live: Option<Live> = None;

    let mut buf = String::new();
    prompt(false);
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };

        // ── while a program is running live, input goes to it ──────────────
        if let Some(l) = &live {
            if buf.trim().is_empty() && line.trim_start().starts_with(':') {
                if line.trim() == ":stop" {
                    l.tx.send(ToDriver::Stop).ok();
                    let session = live.take().unwrap().handle.join().unwrap();
                    s = Some(session);
                    actor = None;
                    println!("  ⏹ stopped; back to the prompt");
                } else {
                    println!("  (running live — type a definition to edit it, or `:stop`)");
                }
                prompt(false);
                continue;
            }
            buf.push_str(&line);
            buf.push('\n');
            if braces_balanced(&buf) && !buf.trim().is_empty() {
                l.tx.send(ToDriver::Edit(buf.clone())).ok();
                buf.clear();
                prompt(false);
            } else {
                prompt(true);
            }
            continue;
        }

        // ── idle: commands + definitions ───────────────────────────────────
        if buf.trim().is_empty() && line.trim_start().starts_with(':') {
            buf.clear();
            let cmd = line.trim();
            if let Some(name) = cmd.strip_prefix(":live ") {
                live = start_live(&mut s, name.trim());
            } else if !command(cmd, s.as_mut().unwrap(), &mut actor, &mut seen) {
                break;
            }
            prompt(false);
            continue;
        }

        buf.push_str(&line);
        buf.push('\n');
        if braces_balanced(&buf) && !buf.trim().is_empty() {
            match s.as_mut().unwrap().eval(&buf) {
                Ok(()) => {
                    println!("  ✓ installed");
                    note_resumable(s.as_ref().unwrap(), actor.as_ref());
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

/// Move the session onto a background thread that runs `name` in a loop,
/// applying edits between steps and printing what it emits — the program keeps
/// running while you type edits at the prompt.
fn start_live(s: &mut Option<Session>, name: &str) -> Option<Live> {
    let session = s.as_ref().unwrap();
    let Some(id) = session.fn_id(name) else {
        println!("  no function `{name}`");
        return None;
    };
    let _ = id;
    let mut session = s.take().unwrap();
    let (tx, rx) = channel::<ToDriver>();
    println!("  ▸ live: `{name}` is running — type definitions to edit it, `:stop` to end");
    prompt(false);
    let name = name.to_string();
    let handle = thread::spawn(move || {
        live_driver(&mut session, &name, rx);
        session
    });
    Some(Live { tx, handle })
}

/// The background loop: step the actor continuously, draining and applying
/// edits between steps; block for input when it finishes or freezes.
fn live_driver(s: &mut Session, name: &str, rx: Receiver<ToDriver>) {
    let id = s.fn_id(name).unwrap();
    let mut actor = match s.engine.spawn(id, vec![]) {
        Ok(a) => a,
        Err(c) => {
            println!("\r  cannot start `{name}`: {c:?}");
            return;
        }
    };
    let mut seen = 0usize;
    loop {
        // Apply any edits the user typed — between steps of the running loop.
        loop {
            match rx.try_recv() {
                Ok(ToDriver::Edit(src)) => match s.eval(&src) {
                    Ok(()) => println!("\r  ✎ live edit applied"),
                    Err(e) => println!("\r  ✗ {e}"),
                },
                Ok(ToDriver::Stop) => return,
                Err(_) => break,
            }
        }
        match &actor.status {
            ActorStatus::Runnable => {
                s.engine.step(&mut actor);
                let out = s.engine.output();
                if out.len() > seen {
                    seen = out.len();
                    if let Value::I64(n) = out[seen - 1] {
                        println!("\r  → {n}");
                    }
                    thread::sleep(Duration::from_millis(350));
                }
            }
            // Finished or frozen: nothing to step, so block for the next input
            // (an edit that repairs it, or `:stop`).
            other => {
                match other {
                    ActorStatus::Complete(v) => println!("\r  ⏹ finished: {}", show(v)),
                    ActorStatus::Paused(c) => println!("\r  ⏸ frozen: {}", cond_line(c)),
                    _ => {}
                }
                match rx.recv() {
                    Ok(ToDriver::Edit(src)) => {
                        if let Err(e) = s.eval(&src) {
                            println!("\r  ✗ {e}");
                        }
                        // If that edit repaired the freeze, the next step
                        // resumes; if not, it re-traps in place.
                        s.engine.thaw(&mut actor);
                    }
                    _ => return,
                }
            }
        }
    }
}

fn cond_line(c: &Condition) -> String {
    match c {
        Condition::BrokenFunction { .. } => "a function no longer type-checks (redefine it)".into(),
        Condition::MissingMigration { .. } => "needs a migration".into(),
        Condition::RuntimeTypeError { message, .. } => message.clone(),
    }
}

fn banner() {
    println!("Live & Typed — interactive live editing");
    println!("  · type `struct`/`fn` definitions to edit the world (live)");
    println!("  · `:run main`   step it turn-by-turn (`:go`)   `:live main`  run it continuously");
    println!("  · while `:live`, type edits and watch the loop change; `:stop` to end");
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
fn command(cmd: &str, s: &mut Session, actor: &mut Option<Actor>, seen: &mut usize) -> bool {
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
            Some(a) => print_status(s, a),
            None => println!("  (no actor — `:run <fn>` first)"),
        },
        ":out" => match actor {
            Some(_) => println!("  emitted: {:?}", s.engine.output()),
            None => println!("  (nothing running)"),
        },
        ":migrate" => migrate(s, &cmd[":migrate".len()..]),
        other => println!("  unknown command `{other}` — try `:help`"),
    }
    true
}

fn show_defs(s: &Session) {
    let lines = s.engine.with_world(|w| {
        let mut lines = Vec::new();
        let mut structs: Vec<_> = w.current_schemas.keys().copied().collect();
        structs.sort_unstable();
        for tid in structs {
            let v = w.current_schemas[&tid];
            let sc = &w.schemas[&(tid, v)];
            if sc.is_enum() {
                let variants: Vec<String> = sc
                    .variants
                    .iter()
                    .map(|var| {
                        if var.fields.is_empty() {
                            var.name.clone()
                        } else {
                            let fields: Vec<String> = var
                                .fields
                                .iter()
                                .map(|f| format!("{}: {}", f.name, ty_name_in(&f.ty, w)))
                                .collect();
                            format!("{} {{ {} }}", var.name, fields.join(", "))
                        }
                    })
                    .collect();
                lines.push(format!("  enum {} {{ {} }}  (v{})", sc.name, variants.join(", "), v.0));
            } else {
                let fields: Vec<String> = sc
                    .fields
                    .iter()
                    .map(|f| format!("{}: {}", f.name, ty_name_in(&f.ty, w)))
                    .collect();
                lines.push(format!("  struct {} {{ {} }}  (v{})", sc.name, fields.join(", "), v.0));
            }
        }
        let mut fns: Vec<_> = w.current_functions.keys().copied().collect();
        fns.sort_unstable();
        for fid in fns {
            let v = w.current_functions[&fid];
            match &w.functions[&(fid, v)] {
                FunctionState::Ready(f) => lines.push(format!("  fn {}  (v{}, ready)", f.name, v.0)),
                FunctionState::Broken { name, .. } => {
                    lines.push(format!("  fn {}  (v{}, BROKEN)", name, v.0))
                }
            }
        }
        lines
    });
    for line in lines {
        println!("{line}");
    }
}

fn run_actor(s: &mut Session, actor: &mut Option<Actor>, seen: &mut usize, rest: &[&str]) {
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
    match s.engine.spawn(id, args) {
        Ok(a) => {
            *actor = Some(a);
            *seen = 0;
            println!("  ▸ running `{name}`");
            go(s, actor, seen);
        }
        Err(c) => println!("  cannot start: {c:?}"),
    }
}

/// Advance the current actor to its next `Yield`, pause, or completion. A
/// paused actor is thawed first — if the last edit repaired it, it resumes; if
/// not, it re-traps at the same spot.
fn go(s: &mut Session, actor: &mut Option<Actor>, seen: &mut usize) {
    let Some(a) = actor.as_mut() else {
        println!("  (no actor — `:run <fn>` first)");
        return;
    };
    if matches!(a.status, ActorStatus::Complete(_)) {
        print_status(s, a);
        return;
    }
    s.engine.thaw(a);
    loop {
        match s.engine.step(a) {
            Turn::Progress => {}
            Turn::Yielded | Turn::Done | Turn::Paused => break,
            Turn::Blocked => {
                println!("  (blocked on a message no one will send)");
                break;
            }
        }
    }
    let out = s.engine.output();
    for v in &out[*seen..] {
        println!("     → {}", show(v));
    }
    *seen = out.len();
    print_status(s, actor.as_ref().unwrap());
}

fn print_status(s: &Session, a: &Actor) {
    match &a.status {
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

fn note_resumable(_s: &Session, actor: Option<&Actor>) {
    if let Some(a) = actor {
        if matches!(a.status, ActorStatus::Paused(_)) {
            println!("  ↻ if that repaired the freeze, `:go` resumes the actor");
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
    let to = s.engine.with_world(|w| w.current_schemas[&type_id]);
    if to.0 < 2 {
        println!("  `{ty_name}` has only one version — nothing to migrate");
        return;
    }
    let from = Version(to.0 - 1);
    let (old, new) = s.engine.with_world(|w| {
        (
            w.schemas[&(type_id, from)].clone(),
            w.schemas[&(type_id, to)].clone(),
        )
    });

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
    match s.engine.install_migration(Migration {
        type_id,
        from,
        to,
        fields,
        variants: std::collections::BTreeMap::new(),
    }) {
        Ok(()) => println!("  ✎ migration installed for {ty_name} v{}→v{}", from.0, to.0),
        Err(e) => println!("  ✗ {e:?}"),
    }
}

fn parse_source(s: &Session, old: &Schema, tokens: &[&str]) -> Result<MigrationSource, String> {
    match tokens {
        ["wrap", spec] => {
            let (wt, wf) = spec.split_once('.').ok_or("wrap needs `Type.field`")?;
            let wtid = s.struct_id(wt).ok_or_else(|| format!("no struct `{wt}`"))?;
            let wfid = s.engine.with_world(|w| {
                let wv = w.current_schemas[&wtid];
                w.schemas[&(wtid, wv)]
                    .fields
                    .iter()
                    .find(|f| f.name == wf)
                    .map(|f| f.id)
            });
            let wfid = wfid.ok_or_else(|| format!("`{wt}` has no field `{wf}`"))?;
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
        Value::Str(id) => format!("{:?}", &*livetype_core::strings::text(*id)),
        Value::Ref(id) => format!("<ref #{id}>"),
        Value::Foreign { kind, ptr } => format!("<foreign k{kind} @{ptr:#x}>"),
    }
}

fn ty_name_in(t: &Type, w: &World) -> String {
    match t {
        Type::I64 => "i64".into(),
        Type::Bool => "bool".into(),
        Type::Str => "str".into(),
        Type::Unit => "()".into(),
        Type::Ref(id) => struct_name_in(w, *id),
        Type::Foreign(kind) => format!("foreign#{kind}"),
    }
}

fn struct_name_in(w: &World, id: DefId) -> String {
    w.current_schemas
        .get(&id)
        .and_then(|v| w.schemas.get(&(id, *v)))
        .map(|sc| sc.name.clone())
        .unwrap_or_else(|| format!("#{id}"))
}

fn struct_name(s: &Session, id: DefId) -> String {
    s.engine.with_world(|w| struct_name_in(w, id))
}

fn fn_name(s: &Session, id: DefId) -> String {
    s.engine.with_world(|w| {
        w.current_functions
            .get(&id)
            .map(|v| match &w.functions[&(id, *v)] {
                FunctionState::Ready(f) => f.name.clone(),
                FunctionState::Broken { name, .. } => name.clone(),
            })
            .unwrap_or_else(|| format!("#{id}"))
    })
}
