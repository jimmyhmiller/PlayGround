use crate::{Function, Instruction, Schema, Type, Value, World};
use std::collections::{BTreeMap, BTreeSet};

fn value_type(value: &Value) -> Result<Type, String> {
    match value {
        Value::Unit => Ok(Type::Unit),
        Value::I64(_) => Ok(Type::I64),
        Value::Ref(_) => Err("object literals are runtime values, not code constants".into()),
    }
}

fn read(regs: &[Option<Type>], register: usize) -> Result<Type, String> {
    regs.get(register)
        .ok_or_else(|| format!("register r{register} is out of bounds"))?
        .clone()
        .ok_or_else(|| format!("register r{register} is read before assignment"))
}

pub fn verify_schema(schema: &Schema, world: &World) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    let mut ids = BTreeSet::new();
    for field in &schema.fields {
        if !ids.insert(field.id) {
            errors.push(format!("duplicate field id {}", field.id));
        }
        if let Type::Ref(id) = field.ty
            && !world.current_schemas.contains_key(&id)
            && id != schema.type_id
        {
            errors.push(format!(
                "field '{}' refers to unknown type {id}",
                field.name
            ));
        }
        if let Some(default) = &field.default {
            match value_type(default) {
                Ok(ty) if ty == field.ty => {}
                _ => errors.push(format!("default for '{}' has the wrong type", field.name)),
            }
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Verify the deliberately straight-line resumable IR. Each instruction is a
/// potential pause boundary; all live values are therefore named registers.
pub fn verify_function(function: &Function, world: &World) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    if function.registers < function.params.len() {
        return Err(vec!["not enough registers for parameters".into()]);
    }
    let mut regs = vec![None; function.registers];
    for (slot, ty) in function.params.iter().enumerate() {
        regs[slot] = Some(ty.clone());
    }
    let mut returned = false;
    for (pc, instruction) in function.code.iter().enumerate() {
        let result: Result<Option<(usize, Type)>, String> = (|| match instruction {
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
                let supplied: BTreeMap<_, _> = fields.iter().copied().collect();
                for field in &schema.fields {
                    match supplied.get(&field.id) {
                        Some(reg) if read(&regs, *reg)? == field.ty => {}
                        Some(_) => {
                            return Err(format!("field '{}' has the wrong type", field.name));
                        }
                        None if field.default.is_some() => {}
                        None => return Err(format!("missing field '{}'", field.name)),
                    }
                }
                Ok(Some((*dst, Type::Ref(*type_id))))
            }
            Instruction::GetField { dst, object, field } => {
                let Type::Ref(type_id) = read(&regs, *object)? else {
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
            Instruction::SubI64 { dst, left, right } => {
                if read(&regs, *left)? != Type::I64 || read(&regs, *right)? != Type::I64 {
                    return Err("subtraction needs i64 operands".into());
                }
                Ok(Some((*dst, Type::I64)))
            }
            Instruction::Call {
                dst,
                function: callee,
                args,
            } => {
                let version = world
                    .current_functions
                    .get(callee)
                    .ok_or_else(|| format!("unknown function {callee}"))?;
                let state = &world.functions[&(*callee, *version)];
                let crate::FunctionState::Ready(callee) = state else {
                    return Err(format!("callee {callee} is broken"));
                };
                if args.len() != callee.params.len() {
                    return Err("wrong argument count".into());
                }
                for (arg, expected) in args.iter().zip(&callee.params) {
                    if read(&regs, *arg)? != *expected {
                        return Err(format!("argument r{arg} has the wrong type"));
                    }
                }
                Ok(Some((*dst, callee.result.clone())))
            }
            Instruction::Emit { value } => {
                read(&regs, *value)?;
                Ok(None)
            }
            Instruction::Return { value } => {
                if read(&regs, *value)? != function.result {
                    return Err("return value has the wrong type".into());
                }
                returned = true;
                Ok(None)
            }
        })();
        match result {
            Ok(Some((dst, ty))) if dst < regs.len() => regs[dst] = Some(ty),
            Ok(Some((dst, _))) => {
                errors.push(format!("pc {pc}: destination r{dst} is out of bounds"))
            }
            Ok(None) => {}
            Err(error) => errors.push(format!("pc {pc}: {error}")),
        }
    }
    if !returned {
        errors.push("function has no return".into());
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}
