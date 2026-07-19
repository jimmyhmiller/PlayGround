//! Differential fuzzer: generate random *valid* programs and random schema
//! evolutions, then run the same scenario on several *configurations of the one
//! engine* and assert they agree on everything observable. Any divergence is a
//! bug in the JIT backend, the tiering seam, the verifier, migration, or the
//! soundness checks. Deterministic (seeded), so a failing seed reproduces
//! exactly.
//!
//! There are no separate executors left to compare — the configurations are
//! never-promote (the oracle), always-promote, and auto-tiering, plus a
//! worker-thread run of the same loop.

use livetype::*;
use std::sync::Arc;

/// xorshift64 — a tiny deterministic PRNG (no external crates).
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed | 1) // never zero
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn chance(&mut self, num: usize, den: usize) -> bool {
        self.below(den) < num
    }
}

const ENTRY: DefId = 1000;

/// A generated program: the initial schemas, one param-less entry function that
/// returns an Int and contains a single `Yield`, and a sequence of schema
/// updates to apply at that yield.
struct Program {
    schemas: Vec<Schema>,
    entry: Function,
    updates: Vec<Schema>,
}

/// Generation-time bookkeeping for a type.
#[derive(Clone)]
struct TypeState {
    id: DefId,
    version: u64,
    fields: Vec<Field>,
    next_field: FieldId,
}

fn gen_program(seed: u64) -> Program {
    let mut rng = Rng::new(seed);
    let n_types = 1 + rng.below(3); // 1..=3
    let mut types: Vec<TypeState> = Vec::new();
    for t in 0..n_types {
        let id = (t + 1) as DefId;
        let n_fields = 1 + rng.below(3);
        let mut fields = Vec::new();
        let base = id * 100;
        for f in 0..n_fields {
            let fid = base + f as FieldId;
            // Ref fields only to earlier types (keeps schemas acyclic + valid).
            let ty = if !types.is_empty() && rng.chance(1, 3) {
                Type::Ref(types[rng.below(types.len())].id)
            } else {
                Type::I64
            };
            // Give Int fields a default sometimes, so later auto-migration works.
            let default = if ty == Type::I64 && rng.chance(1, 2) {
                Some(Value::I64(0))
            } else {
                None
            };
            fields.push(Field {
                id: fid,
                name: format!("f{fid}"),
                ty,
                default,
            });
        }
        types.push(TypeState {
            id,
            version: 1,
            fields,
            next_field: base + 10,
        });
    }
    let schemas = types
        .iter()
        .map(|t| Schema {
            type_id: t.id,
            version: Version(t.version),
            name: format!("T{}", t.id),
            fields: t.fields.clone(),
            variants: Vec::new(),
        })
        .collect();

    let entry = gen_entry(&mut rng, &types);
    let updates = gen_updates(&mut rng, &mut types);
    Program {
        schemas,
        entry,
        updates,
    }
}

/// Generate a valid, param-less entry body. Tracks each register's type so
/// every emitted instruction is well-typed by construction; inserts one Yield.
fn gen_entry(rng: &mut Rng, types: &[TypeState]) -> Function {
    let mut regs: Vec<Type> = Vec::new();
    let mut code: Vec<Instruction> = Vec::new();

    let ints = |regs: &[Type]| -> Vec<usize> {
        regs.iter()
            .enumerate()
            .filter(|(_, t)| **t == Type::I64)
            .map(|(i, _)| i)
            .collect()
    };
    let refs_of = |regs: &[Type], id: DefId| -> Vec<usize> {
        regs.iter()
            .enumerate()
            .filter(|(_, t)| **t == Type::Ref(id))
            .map(|(i, _)| i)
            .collect()
    };

    // Always start with an int constant so a Return is always possible.
    code.push(Instruction::Const {
        dst: regs.len(),
        value: Value::I64((rng.next() % 50) as i64),
    });
    regs.push(Type::I64);

    let steps = 5 + rng.below(8);
    let yield_at = 1 + rng.below(steps.max(1));
    for step in 0..steps {
        if step == yield_at {
            code.push(Instruction::Yield);
        }
        match rng.below(6) {
            0 => {
                code.push(Instruction::Const {
                    dst: regs.len(),
                    value: Value::I64((rng.next() % 50) as i64),
                });
                regs.push(Type::I64);
            }
            1 => {
                let ii = ints(&regs);
                if ii.len() >= 2 {
                    let a = ii[rng.below(ii.len())];
                    let b = ii[rng.below(ii.len())];
                    code.push(Instruction::SubI64 {
                        dst: regs.len(),
                        left: a,
                        right: b,
                    });
                    regs.push(Type::I64);
                }
            }
            2 => {
                // New a random type, if all its fields can be supplied.
                let t = &types[rng.below(types.len())];
                let mut supplied = Vec::new();
                let mut ok = true;
                for f in &t.fields {
                    let candidates = match &f.ty {
                        Type::I64 => ints(&regs),
                        Type::Ref(id) => refs_of(&regs, *id),
                        _ => Vec::new(),
                    };
                    if let Some(&r) = candidates.get(rng.below(candidates.len().max(1))) {
                        if !candidates.is_empty() {
                            supplied.push((f.id, r));
                            continue;
                        }
                    }
                    if f.default.is_none() {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    code.push(Instruction::New {
                        dst: regs.len(),
                        type_id: t.id,
                        fields: supplied,
                    });
                    regs.push(Type::Ref(t.id));
                }
            }
            3 => {
                // GetField from any available reference.
                let with_refs: Vec<DefId> = types
                    .iter()
                    .map(|t| t.id)
                    .filter(|id| !refs_of(&regs, *id).is_empty())
                    .collect();
                if !with_refs.is_empty() {
                    let id = with_refs[rng.below(with_refs.len())];
                    let obj = {
                        let r = refs_of(&regs, id);
                        r[rng.below(r.len())]
                    };
                    let t = types.iter().find(|t| t.id == id).unwrap();
                    let f = &t.fields[rng.below(t.fields.len())];
                    code.push(Instruction::GetField {
                        dst: regs.len(),
                        object: obj,
                        field: f.id,
                    });
                    regs.push(f.ty.clone());
                }
            }
            4 => {
                if !regs.is_empty() {
                    code.push(Instruction::Emit {
                        value: rng.below(regs.len()),
                    });
                }
            }
            _ => {
                let ii = ints(&regs);
                if ii.len() >= 2 {
                    let a = ii[rng.below(ii.len())];
                    let b = ii[rng.below(ii.len())];
                    code.push(Instruction::LtI64 {
                        dst: regs.len(),
                        left: a,
                        right: b,
                    });
                    regs.push(Type::Bool);
                }
            }
        }
    }
    // Return the first int register (guaranteed to exist).
    let ret = ints(&regs)[0];
    code.push(Instruction::Return { value: ret });

    Function {
        id: ENTRY,
        version: Version(1),
        name: "entry".into(),
        params: vec![],
        result: Type::I64,
        registers: regs.len(),
        code,
    }
}

/// Generate a sequence of schema updates: additive defaulted fields
/// (auto-derivable) and Int→Ref retypes with no migration (a gap that traps).
fn gen_updates(rng: &mut Rng, types: &mut [TypeState]) -> Vec<Schema> {
    let mut updates = Vec::new();
    let n = rng.below(4);
    for _ in 0..n {
        let t_idx = rng.below(types.len());
        let existing_ids: Vec<DefId> = types.iter().map(|t| t.id).collect();
        let t = &mut types[t_idx];
        if rng.chance(2, 3) {
            // Add a defaulted Int field (auto-derivable).
            let fid = t.next_field;
            t.next_field += 1;
            t.fields.push(Field {
                id: fid,
                name: format!("f{fid}"),
                ty: Type::I64,
                default: Some(Value::I64((rng.next() % 20) as i64)),
            });
        } else {
            // Retype an Int field to a Ref with no default → gap. Pick a target
            // type (any existing) and an Int field to retype.
            let target = existing_ids[rng.below(existing_ids.len())];
            let int_field = t.fields.iter().position(|f| f.ty == Type::I64);
            if let Some(pos) = int_field {
                t.fields[pos].ty = Type::Ref(target);
                t.fields[pos].default = None;
            } else {
                continue; // nothing to retype
            }
        }
        t.version += 1;
        updates.push(Schema {
            type_id: t.id,
            version: Version(t.version),
            name: format!("T{}", t.id),
            fields: t.fields.clone(),
            variants: Vec::new(),
        });
    }
    updates
}

fn install_program(engine: &Engine, program: &Program) {
    for schema in &program.schemas {
        engine.install_schema(schema.clone()).unwrap();
    }
    // The entry may or may not verify against the initial schemas; either way it
    // installs (Ready or Broken) and every configuration treats it identically.
    let _ = engine.install_function(program.entry.clone());
}

/// Step the engine until the actor is no longer runnable or has crossed a
/// `Yield` — identical code for every configuration, which is the point.
fn to_yield(engine: &Engine, actor: &mut Actor) {
    loop {
        match engine.step(actor) {
            Turn::Progress => {}
            Turn::Yielded | Turn::Done | Turn::Paused => break,
            Turn::Blocked => panic!("fuzzed programs have no message passing"),
        }
    }
}

/// The full edit scenario on one engine configuration: run to the yield, apply
/// the schema updates, run to a stop.
fn run_scenario(engine: &Arc<Engine>, program: &Program) -> Result<Outcome, Condition> {
    install_program(engine, program);
    let mut actor = engine.spawn(ENTRY, vec![])?;
    to_yield(engine, &mut actor);
    for schema in &program.updates {
        let _ = engine.install_schema(schema.clone());
    }
    Ok(engine.run(&mut actor))
}

/// Compare two outcomes by what "observable" means here: completion values
/// exactly, pause conditions by variant.
fn outcome_eq(a: &Outcome, b: &Outcome) -> bool {
    match (a, b) {
        (Outcome::Complete(x), Outcome::Complete(y)) => x == y,
        (Outcome::Paused(x), Outcome::Paused(y)) => {
            std::mem::discriminant(x) == std::mem::discriminant(y)
        }
        _ => false,
    }
}

fn run_seed(seed: u64) {
    let program = gen_program(seed);

    let configs: Vec<(&str, Arc<Engine>)> = vec![
        ("interp", Engine::interp()), // the oracle: never promotes
        ("jit", jit_engine(0)),       // promotes everything on first entry
        ("tiered", jit_engine(3)),    // auto-tiering mid-scenario
    ];
    let results: Vec<(&str, &Arc<Engine>, Result<Outcome, Condition>)> = configs
        .iter()
        .map(|(name, e)| (*name, e, run_scenario(e, &program)))
        .collect();

    let (oracle_name, oracle_engine, oracle) = &results[0];
    for (name, engine, result) in &results[1..] {
        match (oracle, result) {
            (Ok(a), Ok(b)) => {
                assert!(
                    outcome_eq(a, b),
                    "seed {seed}: outcome diverged\n  {oracle_name}: {a:?}\n  {name}: {b:?}"
                );
                assert_eq!(
                    oracle_engine.output(),
                    engine.output(),
                    "seed {seed}: effects diverged ({oracle_name} vs {name})"
                );
                assert_eq!(
                    *oracle_engine.shared().heap(),
                    *engine.shared().heap(),
                    "seed {seed}: heap diverged ({oracle_name} vs {name})"
                );
            }
            (Err(_), Err(_)) => {} // both refused to spawn a broken entry
            (a, b) => panic!(
                "seed {seed}: spawn behavior diverged\n  {oracle_name}: {a:?}\n  {name}: {b:?}"
            ),
        }
    }
}

#[test]
fn engine_configurations_agree_on_random_programs() {
    // Over 600 seeds this reaches completion, migration-gap traps, and
    // con-freeness traps — every configuration of the one engine agrees on
    // every one.
    for seed in 1..=600u64 {
        run_seed(seed);
    }
}

/// Run the *base* program (no schema updates) to completion on the calling
/// thread (interp oracle) and on a worker thread running the always-JIT
/// configuration, and assert they agree — pinning that "on a thread" is the
/// same loop with the same observable behavior.
fn run_seed_threaded(seed: u64) {
    let program = gen_program(seed);

    let oracle = Engine::interp();
    install_program(&oracle, &program);
    let Ok(mut actor) = oracle.spawn(ENTRY, vec![]) else {
        return; // a broken entry cannot spawn; nothing to compare
    };
    let oracle_outcome = oracle.run(&mut actor);

    let jit = jit_engine(0);
    install_program(&jit, &program);
    let outcomes = jit.run_threads(vec![(ENTRY, vec![])]);

    assert!(
        outcome_eq(&oracle_outcome, &outcomes[0]),
        "seed {seed}: threaded outcome diverged\n  oracle: {oracle_outcome:?}\n  thread: {:?}",
        outcomes[0]
    );
    assert_eq!(oracle.output(), jit.output(), "seed {seed}: threaded effects diverged");
    assert_eq!(
        oracle.shared().object_count(),
        jit.shared().object_count(),
        "seed {seed}: threaded object count diverged"
    );
}

#[test]
fn worker_thread_matches_the_calling_thread() {
    for seed in 1..=600u64 {
        run_seed_threaded(seed);
    }
}
