//! Differential fuzzer: generate random *valid* programs and random schema
//! evolutions, then run the same scenario on both executors and assert they
//! agree on everything observable. Any divergence is a bug in the JIT backend,
//! the verifier, migration, or the soundness checks. Deterministic (seeded), so
//! a failing seed reproduces exactly.

use livetype::*;

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
        });
    }
    updates
}

fn install_program(program: &Program) -> Runtime {
    let mut rt = Runtime::default();
    for schema in &program.schemas {
        rt.install_schema(schema.clone()).unwrap();
    }
    // The entry may or may not verify against the initial schemas; either way it
    // installs (Ready or Broken) and both executors treat it identically.
    let _ = rt.install_function(program.entry.clone());
    rt
}

/// Step the interpreter until the actor is no longer runnable or has executed a
/// `Yield`.
fn interp_to_yield(rt: &mut Runtime, actor: ActorId) {
    while matches!(rt.actors[&actor].status, ActorStatus::Runnable) {
        let (key, pc) = {
            let f = rt.actors[&actor].frames.last().unwrap();
            (f.function, f.pc)
        };
        let is_yield = matches!(
            &rt.world.functions[&key],
            FunctionState::Ready(func) if matches!(func.code[pc], Instruction::Yield)
        );
        rt.step(actor);
        if is_yield {
            break;
        }
    }
}

/// Compare the two statuses by kind (and value/condition variant), which is what
/// "observable" means here — the two runtimes allocate ids identically, so
/// object references compare directly too.
fn status_eq(a: &ActorStatus, b: &ActorStatus) -> bool {
    match (a, b) {
        (ActorStatus::Complete(x), ActorStatus::Complete(y)) => x == y,
        (ActorStatus::Runnable, ActorStatus::Runnable) => true,
        (ActorStatus::Paused(x), ActorStatus::Paused(y)) => {
            std::mem::discriminant(x) == std::mem::discriminant(y)
        }
        _ => false,
    }
}

fn run_seed(seed: u64) {
    let program = gen_program(seed);

    // Interpreter.
    let mut rt_i = install_program(&program);
    let spawn_i = rt_i.spawn(ENTRY, vec![]);
    // JIT.
    let mut rt_j = install_program(&program);
    let spawn_j = JitActor::spawn(&rt_j, 1, ENTRY, vec![]);

    // A broken entry cannot be spawned; both must agree on that.
    match (spawn_i, spawn_j) {
        (Ok(a_i), Ok(mut a_j)) => {
            interp_to_yield(&mut rt_i, a_i);
            drive(&mut rt_j, &mut a_j, true).unwrap();

            for schema in &program.updates {
                let _ = rt_i.install_schema(schema.clone());
                let _ = rt_j.install_schema(schema.clone());
            }

            rt_i.run();
            drive(&mut rt_j, &mut a_j, false).unwrap();

            let si = &rt_i.actors[&a_i].status;
            assert!(
                status_eq(si, &a_j.status),
                "seed {seed}: status diverged\n  interp: {si:?}\n  jit:    {:?}",
                a_j.status
            );
            assert_eq!(rt_i.output, rt_j.output, "seed {seed}: effects diverged");
            assert_eq!(rt_i.heap, rt_j.heap, "seed {seed}: heap diverged");
        }
        (Err(_), Err(_)) => {}
        (Ok(_), Err(e)) => panic!("seed {seed}: interpreter spawned but JIT did not: {e:?}"),
        (Err(e), Ok(_)) => panic!("seed {seed}: JIT spawned but interpreter did not: {e:?}"),
    }
}

#[test]
fn jit_matches_interpreter_on_random_programs() {
    // Over 600 seeds this reaches completion, migration-gap traps, and
    // con-freeness traps (roughly 526 / 22 / 52 by outcome) — the JIT and
    // interpreter agree on every one.
    for seed in 1..=600u64 {
        run_seed(seed);
    }
}

/// Run the *base* program (no schema updates) to completion on the interpreter
/// and on the `Shared` concurrent tier, and assert they agree. The Shared tier
/// freezes the world and cannot take mid-run edits (that is what Phase 4 adds),
/// so the three-way net compares the edit scenario on interp↔JIT and the
/// steady-state run on interp↔Shared. Together they pin every executor's shared
/// machinery (heap, migration, step semantics, soundness) during the
/// unification refactor.
fn run_seed_shared(seed: u64) {
    let program = gen_program(seed);

    let mut rt_i = install_program(&program);
    let Ok(a_i) = rt_i.spawn(ENTRY, vec![]) else {
        return; // a broken entry cannot spawn; nothing to compare
    };
    rt_i.run();
    let interp_status = rt_i.actors[&a_i].status.clone();
    let interp_out = rt_i.output.clone();
    let interp_objs = rt_i.heap.len();

    let rt_s = install_program(&program);
    let shared = Shared::from_runtime(rt_s);
    let outcomes = shared.run_threads(vec![(ENTRY, vec![])]);

    match (&interp_status, &outcomes[0]) {
        (ActorStatus::Complete(x), Outcome::Complete(y)) => {
            assert_eq!(x, y, "seed {seed}: shared completion value diverged");
        }
        (ActorStatus::Paused(x), Outcome::Paused(y)) => {
            assert_eq!(
                std::mem::discriminant(x),
                std::mem::discriminant(y),
                "seed {seed}: shared pause condition diverged"
            );
        }
        (i, s) => panic!("seed {seed}: shared outcome diverged\n  interp: {i:?}\n  shared: {s:?}"),
    }
    assert_eq!(interp_out, shared.output(), "seed {seed}: shared effects diverged");
    assert_eq!(
        interp_objs,
        shared.object_count(),
        "seed {seed}: shared object count diverged"
    );
}

#[test]
fn shared_matches_interpreter_on_random_programs() {
    for seed in 1..=600u64 {
        run_seed_shared(seed);
    }
}
