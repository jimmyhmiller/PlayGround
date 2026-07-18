use crate::{DefId, Function, Instruction, Schema, Type, Value, World};
use std::collections::{BTreeMap, BTreeSet};

fn value_type(value: &Value) -> Result<Type, String> {
    match value {
        Value::Unit => Ok(Type::Unit),
        Value::I64(_) => Ok(Type::I64),
        Value::Bool(_) => Ok(Type::Bool),
        Value::Str(_) => Ok(Type::Str),
        Value::Ref(_) => Err("object literals are runtime values, not code constants".into()),
        Value::Foreign { .. } => Err("foreign handles are runtime values, not code constants".into()),
    }
}

/// The constructor rule shared by `New` and `NewVariant`: every declared field
/// is supplied with the right type or has a default.
fn check_supplied_fields(
    declared: &[crate::Field],
    supplied: &[(crate::FieldId, usize)],
    regs: &[Option<Type>],
) -> Result<(), String> {
    let supplied: BTreeMap<_, _> = supplied.iter().copied().collect();
    for field in declared {
        match supplied.get(&field.id) {
            Some(reg) if read(regs, *reg)? == field.ty => {}
            Some(_) => return Err(format!("field '{}' has the wrong type", field.name)),
            None if field.default.is_some() => {}
            None => return Err(format!("missing field '{}'", field.name)),
        }
    }
    Ok(())
}

fn read(regs: &[Option<Type>], register: usize) -> Result<Type, String> {
    regs.get(register)
        .ok_or_else(|| format!("register r{register} is out of bounds"))?
        .clone()
        .ok_or_else(|| format!("register r{register} is read before assignment"))
}

pub fn verify_schema(schema: &Schema, world: &World) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    if schema.is_enum() && !schema.fields.is_empty() {
        errors.push("a schema is a struct (fields) or an enum (variants), not both".into());
    }
    let mut ids = BTreeSet::new();
    let mut check_field = |field: &crate::Field, errors: &mut Vec<String>| {
        if !ids.insert(field.id) {
            errors.push(format!("duplicate field id {}", field.id));
        }
        if let Type::Ref(id) = field.ty
            && !world.current_schemas.contains_key(&id)
            && id != schema.type_id
        {
            errors.push(format!("field '{}' refers to unknown type {id}", field.name));
        }
        if let Some(default) = &field.default {
            match value_type(default) {
                Ok(ty) if ty == field.ty => {}
                _ => errors.push(format!("default for '{}' has the wrong type", field.name)),
            }
        }
    };
    for field in &schema.fields {
        check_field(field, &mut errors);
    }
    let mut variant_ids = BTreeSet::new();
    for variant in &schema.variants {
        if !variant_ids.insert(variant.id) {
            errors.push(format!("duplicate variant id {}", variant.id));
        }
        // Field ids are unique across the WHOLE enum (the same `ids` set), so
        // field reads and migrations name a field unambiguously.
        for field in &variant.fields {
            check_field(field, &mut errors);
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Type-check one instruction against the incoming register environment,
/// returning the register it writes (if any) and that register's new type.
/// Control-flow instructions (`Jump`/`Branch`/`Yield`) and effects write no
/// register and return `None`; successor wiring is handled by the caller.
fn check_instruction(
    instruction: &Instruction,
    function: &Function,
    world: &World,
    extra: &BTreeMap<DefId, (Vec<Type>, Type)>,
    regs: &[Option<Type>],
) -> Result<Option<(usize, Type)>, String> {
    match instruction {
        Instruction::Const { dst, value } => Ok(Some((*dst, value_type(value)?))),
        Instruction::New {
            dst,
            type_id,
            fields,
        } => {
                let version = world
                    .current_schemas
                    .get(type_id)
                    .ok_or_else(|| format!("unknown type {type_id}"))?;
                let schema = world
                    .schemas
                    .get(&(*type_id, *version))
                    .ok_or_else(|| "missing current schema".to_string())?;
                if schema.is_enum() {
                    return Err(format!(
                        "type {} is an enum — construct a variant",
                        schema.name
                    ));
                }
                check_supplied_fields(&schema.fields, fields, regs)?;
                Ok(Some((*dst, Type::Ref(*type_id))))
            }
            Instruction::NewVariant {
                dst,
                type_id,
                variant,
                fields,
            } => {
                let version = world
                    .current_schemas
                    .get(type_id)
                    .ok_or_else(|| format!("unknown type {type_id}"))?;
                let schema = world
                    .schemas
                    .get(&(*type_id, *version))
                    .ok_or_else(|| "missing current schema".to_string())?;
                let v = schema
                    .variant(*variant)
                    .ok_or_else(|| format!("type {} has no variant {variant}", schema.name))?;
                check_supplied_fields(&v.fields, fields, regs)?;
                Ok(Some((*dst, Type::Ref(*type_id))))
            }
            Instruction::CaseVariant { object, arms } => {
                let Type::Ref(type_id) = read(regs, *object)? else {
                    return Err("match needs an enum reference".into());
                };
                let version = world
                    .current_schemas
                    .get(&type_id)
                    .ok_or_else(|| "unknown object type".to_string())?;
                let schema = &world.schemas[&(type_id, *version)];
                if !schema.is_enum() {
                    return Err(format!("type {} is not an enum", schema.name));
                }
                // EXACT coverage of the current variants: a missing arm is a
                // verification error, so adding a variant statically breaks
                // (invalidates, D7) every match that doesn't handle it — the
                // developer repairs the match live instead of a runtime surprise.
                let mut seen = BTreeSet::new();
                for (variant, _) in arms {
                    if schema.variant(*variant).is_none() {
                        return Err(format!(
                            "match arm names variant {variant}, which {} does not have",
                            schema.name
                        ));
                    }
                    if !seen.insert(*variant) {
                        return Err(format!("duplicate match arm for variant {variant}"));
                    }
                }
                for variant in &schema.variants {
                    if !seen.contains(&variant.id) {
                        return Err(format!(
                            "non-exhaustive match: missing arm for variant '{}'",
                            variant.name
                        ));
                    }
                }
                Ok(None)
            }
            Instruction::GetField { dst, object, field } => {
                let Type::Ref(type_id) = read(regs, *object)? else {
                    return Err("field access needs a reference".into());
                };
                let version = world
                    .current_schemas
                    .get(&type_id)
                    .ok_or_else(|| "unknown object type".to_string())?;
                let schema = &world.schemas[&(type_id, *version)];
                let ty = schema
                    .field(*field)
                    .ok_or_else(|| format!("unknown field {field}"))?
                    .ty
                    .clone();
                Ok(Some((*dst, ty)))
            }
            Instruction::Copy { dst, src } => Ok(Some((*dst, read(regs, *src)?))),
            Instruction::AddI64 { dst, left, right } => {
                if read(regs, *left)? != Type::I64 || read(regs, *right)? != Type::I64 {
                    return Err("addition needs i64 operands".into());
                }
                Ok(Some((*dst, Type::I64)))
            }
            Instruction::SubI64 { dst, left, right } => {
                if read(regs, *left)? != Type::I64 || read(regs, *right)? != Type::I64 {
                    return Err("subtraction needs i64 operands".into());
                }
                Ok(Some((*dst, Type::I64)))
            }
            Instruction::MulI64 { dst, left, right } => {
                if read(regs, *left)? != Type::I64 || read(regs, *right)? != Type::I64 {
                    return Err("multiplication needs i64 operands".into());
                }
                Ok(Some((*dst, Type::I64)))
            }
            Instruction::EqI64 { dst, left, right } => {
                if read(regs, *left)? != Type::I64 || read(regs, *right)? != Type::I64 {
                    return Err("equality needs i64 operands".into());
                }
                Ok(Some((*dst, Type::Bool)))
            }
            Instruction::Not { dst, src } => {
                if read(regs, *src)? != Type::Bool {
                    return Err("negation needs a bool operand".into());
                }
                Ok(Some((*dst, Type::Bool)))
            }
            Instruction::ConcatStr { dst, left, right } => {
                if read(regs, *left)? != Type::Str || read(regs, *right)? != Type::Str {
                    return Err("concatenation needs string operands".into());
                }
                Ok(Some((*dst, Type::Str)))
            }
            Instruction::EqStr { dst, left, right } => {
                if read(regs, *left)? != Type::Str || read(regs, *right)? != Type::Str {
                    return Err("string equality needs string operands".into());
                }
                Ok(Some((*dst, Type::Bool)))
            }
            Instruction::Call {
                dst,
                function: callee,
                args,
            } => {
                // Resolve the callee's signature from the world (its current
                // Ready version) or, failing that, from the batch of functions
                // being installed together (`extra`) — which is how a call to a
                // not-yet-installed function, including recursion, type-checks.
                let (params, result) = if let Some(version) = world.current_functions.get(callee) {
                    match &world.functions[&(*callee, *version)] {
                        crate::FunctionState::Ready(f) => (f.params.clone(), f.result.clone()),
                        _ => return Err(format!("callee {callee} is broken")),
                    }
                } else if let Some(sig) = extra.get(callee) {
                    sig.clone()
                } else {
                    return Err(format!("unknown function {callee}"));
                };
                if args.len() != params.len() {
                    return Err("wrong argument count".into());
                }
                for (arg, expected) in args.iter().zip(&params) {
                    if read(regs, *arg)? != *expected {
                        return Err(format!("argument r{arg} has the wrong type"));
                    }
                }
                Ok(Some((*dst, result)))
            }
        Instruction::CallForeign { dst, foreign, args } => {
            let (params, result) = world
                .foreign_sigs
                .get(foreign)
                .ok_or_else(|| format!("unknown foreign fn {foreign}"))?;
            if args.len() != params.len() {
                return Err("wrong argument count to foreign fn".into());
            }
            for (arg, expected) in args.iter().zip(params) {
                if read(regs, *arg)? != *expected {
                    return Err(format!("foreign argument r{arg} has the wrong type"));
                }
            }
            Ok(Some((*dst, result.clone())))
        }
        Instruction::LoadGlobal { dst, global } => {
            let ty = world
                .global_types
                .get(global)
                .ok_or_else(|| format!("unknown global {global}"))?
                .clone();
            Ok(Some((*dst, ty)))
        }
        Instruction::LtI64 { dst, left, right } => {
            if read(regs, *left)? != Type::I64 || read(regs, *right)? != Type::I64 {
                return Err("comparison needs i64 operands".into());
            }
            Ok(Some((*dst, Type::Bool)))
        }
        Instruction::Branch { cond, .. } => {
            if read(regs, *cond)? != Type::Bool {
                return Err("branch condition must be a bool".into());
            }
            Ok(None)
        }
        Instruction::Jump { .. } | Instruction::Yield => Ok(None),
        Instruction::Emit { value } => {
            read(regs, *value)?;
            Ok(None)
        }
        Instruction::Send { target, value } => {
            if read(regs, *target)? != Type::I64 {
                return Err("send target must be an actor id (i64)".into());
            }
            read(regs, *value)?;
            Ok(None)
        }
        Instruction::Recv { dst, ty } => Ok(Some((*dst, ty.clone()))),
        Instruction::Return { value } => {
            if read(regs, *value)? != function.result {
                return Err("return value has the wrong type".into());
            }
            Ok(None)
        }
    }
}

/// Successor program counters of the instruction at `pc`. `Return` has none;
/// `Jump`/`Branch` name their targets; everything else falls through to `pc+1`.
fn successors(instruction: &Instruction, pc: usize) -> Vec<usize> {
    match instruction {
        Instruction::Return { .. } => vec![],
        Instruction::Jump { target } => vec![*target],
        Instruction::Branch {
            then_pc, else_pc, ..
        } => vec![*then_pc, *else_pc],
        Instruction::CaseVariant { arms, .. } => arms.iter().map(|(_, pc)| *pc).collect(),
        _ => vec![pc + 1],
    }
}

/// Verify the resumable IR as a control-flow graph. Every instruction is a
/// potential pause boundary, so all live values are named registers and the
/// register-type environment must be consistent on entry to each pc regardless
/// of the path taken there (this is what makes a loop back-edge or a branch
/// merge type-safe). A fixpoint worklist propagates environments from entry;
/// a pc reached with two different environments, an out-of-range target, or a
/// fall-through off the end is a type error.
/// On success returns the set of nominal types this function references (its
/// dependency set, D7), so a schema change re-verifies only the functions that
/// could be affected by it.
pub fn verify_function(function: &Function, world: &World) -> Result<BTreeSet<DefId>, Vec<String>> {
    verify_function_with(function, world, &BTreeMap::new())
}

/// Like [`verify_function`], but calls may also resolve against `extra` — the
/// signatures of a batch of functions being installed together but not yet in
/// the world. This is what lets a batch verify recursive and mutually-recursive
/// calls (and removes any need to install callees before callers).
pub fn verify_function_with(
    function: &Function,
    world: &World,
    extra: &BTreeMap<DefId, (Vec<Type>, Type)>,
) -> Result<BTreeSet<DefId>, Vec<String>> {
    let mut errors = Vec::new();
    if function.registers < function.params.len() {
        return Err(vec!["not enough registers for parameters".into()]);
    }
    if function.code.is_empty() {
        return Err(vec!["function has no code".into()]);
    }

    let mut entry = vec![Some(vec![None; function.registers])];
    for (slot, ty) in function.params.iter().enumerate() {
        entry[0].as_mut().unwrap()[slot] = Some(ty.clone());
    }
    let mut envs: Vec<Option<Vec<Option<Type>>>> = vec![None; function.code.len()];
    envs[0] = entry.pop().unwrap();
    let mut work = vec![0usize];
    let mut any_return = false;

    while let Some(pc) = work.pop() {
        let regs = envs[pc].clone().expect("worklist only enqueues assigned pcs");
        let instruction = &function.code[pc];
        if matches!(instruction, Instruction::Return { .. }) {
            any_return = true;
        }
        let mut out = regs.clone();
        match check_instruction(instruction, function, world, extra, &regs) {
            Ok(Some((dst, ty))) if dst < out.len() => out[dst] = Some(ty),
            Ok(Some((dst, _))) => {
                errors.push(format!("pc {pc}: destination r{dst} is out of bounds"));
                continue;
            }
            Ok(None) => {}
            Err(error) => {
                errors.push(format!("pc {pc}: {error}"));
                continue;
            }
        }
        for succ in successors(instruction, pc) {
            if succ >= function.code.len() {
                errors.push(format!("pc {pc}: control flows to invalid pc {succ}"));
                continue;
            }
            match &mut envs[succ] {
                None => {
                    envs[succ] = Some(out.clone());
                    work.push(succ);
                }
                Some(existing) => {
                    // Join incoming paths per register: a register keeps a type
                    // only if every path agrees; defined-on-some-paths weakens to
                    // undefined (reading it later is then the error), and two
                    // different concrete types is an outright conflict. Weakening
                    // is monotone, so the fixpoint terminates; a loop back-edge
                    // that reassigns its induction registers before use verifies
                    // cleanly this way.
                    let mut changed = false;
                    for (reg, slot) in existing.iter_mut().enumerate() {
                        match (slot.clone(), &out[reg]) {
                            (Some(a), Some(b)) if a == *b => {}
                            (Some(a), Some(b)) => {
                                errors.push(format!(
                                    "pc {succ}: register r{reg} merges incompatible types \
                                     {a:?} and {b:?}"
                                ));
                                *slot = None;
                                changed = true;
                            }
                            (Some(_), None) => {
                                *slot = None;
                                changed = true;
                            }
                            (None, _) => {}
                        }
                    }
                    if changed {
                        work.push(succ);
                    }
                }
            }
        }
    }

    // Only worth reporting when nothing else went wrong; an earlier error
    // aborts a path before its return and would make this misleading noise.
    if !any_return && errors.is_empty() {
        errors.push("function has no reachable return".into());
    }
    if !errors.is_empty() {
        return Err(errors);
    }

    // Collect the dependency set: every nominal type reachable through this
    // function's signature, its registers (which cover `GetField` object and
    // field types), and its `New` sites. A superset is fine — it only re-checks
    // a few extra functions on a schema change, never too few.
    let mut deps = BTreeSet::new();
    let mut add = |ty: &Type| {
        if let Type::Ref(t) = ty {
            deps.insert(*t);
        }
    };
    function.params.iter().for_each(&mut add);
    add(&function.result);
    for env in envs.iter().flatten() {
        env.iter().flatten().for_each(&mut add);
    }
    for instruction in &function.code {
        match instruction {
            Instruction::New { type_id, .. } | Instruction::NewVariant { type_id, .. } => {
                deps.insert(*type_id);
            }
            _ => {}
        }
    }
    Ok(deps)
}
