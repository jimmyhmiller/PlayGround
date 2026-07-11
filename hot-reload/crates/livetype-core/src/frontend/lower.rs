//! Lowering: AST → runtime IR. Structs become `Schema`s, functions become
//! register-machine `Function`s. Locals get fixed registers (this IR has no
//! phi), sub-expressions get fresh temporaries, and control flow is emitted
//! with symbolic labels that are patched to program counters at the end.

use super::ast::*;
use crate::{DefId, Field, FieldId, Function, Instruction, Schema, Type, Value, Version};
use std::collections::HashMap;

struct StructInfo {
    id: DefId,
    fields: Vec<(String, FieldId, Type)>,
}

struct FnSig {
    id: DefId,
    params: Vec<Type>,
    result: Type,
}

/// The result of lowering a whole program.
pub struct Lowered {
    pub schemas: Vec<Schema>,
    pub functions: Vec<Function>,
    pub struct_ids: HashMap<String, DefId>,
    pub fn_ids: HashMap<String, DefId>,
}

pub fn lower(program: &Program) -> Result<Lowered, String> {
    let structs: Vec<&StructDef> = program
        .items
        .iter()
        .filter_map(|i| match i {
            Item::Struct(s) => Some(s),
            _ => None,
        })
        .collect();
    let fns: Vec<&FnDef> = program
        .items
        .iter()
        .filter_map(|i| match i {
            Item::Fn(f) => Some(f),
            _ => None,
        })
        .collect();

    // Assign disjoint id ranges: structs 1..=S, functions S+1.. .
    let mut struct_table: HashMap<String, StructInfo> = HashMap::new();
    for (idx, s) in structs.iter().enumerate() {
        let id = (idx + 1) as DefId;
        let mut fields = Vec::new();
        for (fidx, f) in s.fields.iter().enumerate() {
            let fid = id * 1000 + fidx as FieldId;
            // Field type is resolved against the whole struct table below; a
            // temporary placeholder is fine here since we re-resolve.
            fields.push((f.name.clone(), fid, Type::Unit));
        }
        if struct_table
            .insert(s.name.clone(), StructInfo { id, fields })
            .is_some()
        {
            return Err(format!("duplicate struct `{}`", s.name));
        }
    }
    // Resolve field types now that all struct names are known.
    let resolve = |te: &TypeExpr, table: &HashMap<String, StructInfo>| -> Result<Type, String> {
        Ok(match te {
            TypeExpr::I64 => Type::I64,
            TypeExpr::Bool => Type::Bool,
            TypeExpr::Unit => Type::Unit,
            TypeExpr::Ref(name) => {
                Type::Ref(table.get(name).ok_or_else(|| format!("unknown struct `{name}`"))?.id)
            }
        })
    };
    let mut schemas = Vec::new();
    for s in &structs {
        let info_id = struct_table[&s.name].id;
        let mut fields = Vec::new();
        let mut resolved = Vec::new();
        for (fidx, f) in s.fields.iter().enumerate() {
            let fid = info_id * 1000 + fidx as FieldId;
            let ty = resolve(&f.ty, &struct_table)?;
            let default = match &f.default {
                None => None,
                Some(e) => Some(const_value(e, &ty)?),
            };
            fields.push(Field {
                id: fid,
                name: f.name.clone(),
                ty: ty.clone(),
                default,
            });
            resolved.push((f.name.clone(), fid, ty));
        }
        struct_table.get_mut(&s.name).unwrap().fields = resolved;
        schemas.push(Schema {
            type_id: info_id,
            version: Version(1),
            name: s.name.clone(),
            fields,
        });
    }

    // Function signatures.
    let mut fn_table: HashMap<String, FnSig> = HashMap::new();
    let base = structs.len() as DefId;
    for (idx, f) in fns.iter().enumerate() {
        let id = base + 1 + idx as DefId;
        let params = f
            .params
            .iter()
            .map(|p| resolve(&p.ty, &struct_table))
            .collect::<Result<_, _>>()?;
        let result = resolve(&f.ret, &struct_table)?;
        if fn_table
            .insert(f.name.clone(), FnSig { id, params, result })
            .is_some()
        {
            return Err(format!("duplicate function `{}`", f.name));
        }
    }

    let mut functions = Vec::new();
    for f in &fns {
        functions.push(lower_fn(f, &struct_table, &fn_table)?);
    }

    let struct_ids = struct_table.iter().map(|(k, v)| (k.clone(), v.id)).collect();
    let fn_ids = fn_table.iter().map(|(k, v)| (k.clone(), v.id)).collect();
    Ok(Lowered {
        schemas,
        functions,
        struct_ids,
        fn_ids,
    })
}

fn const_value(e: &Expr, ty: &Type) -> Result<Value, String> {
    let v = match e {
        Expr::Int(n) => Value::I64(*n),
        Expr::Bool(b) => Value::Bool(*b),
        Expr::Unit => Value::Unit,
        _ => return Err("field default must be a literal".into()),
    };
    if v.shallow_type(&std::collections::BTreeMap::new()).as_ref() != Some(ty) {
        return Err("field default has the wrong type".into());
    }
    Ok(v)
}

struct Lower<'a> {
    structs: &'a HashMap<String, StructInfo>,
    fns: &'a HashMap<String, FnSig>,
    code: Vec<Instruction>,
    labels: Vec<usize>,
    next_reg: usize,
    scopes: Vec<HashMap<String, (usize, Type)>>,
    result_ty: Type,
}

fn lower_fn(
    f: &FnDef,
    structs: &HashMap<String, StructInfo>,
    fns: &HashMap<String, FnSig>,
) -> Result<Function, String> {
    let sig = &fns[&f.name];
    let mut lo = Lower {
        structs,
        fns,
        code: Vec::new(),
        labels: Vec::new(),
        next_reg: 0,
        scopes: vec![HashMap::new()],
        result_ty: sig.result.clone(),
    };
    for (p, ty) in f.params.iter().zip(&sig.params) {
        let r = lo.fresh_reg();
        lo.bind(&p.name, r, ty.clone());
    }
    for s in &f.body {
        lo.stmt(s)?;
    }
    lo.patch_labels()?;
    Ok(Function {
        id: sig.id,
        version: Version(1),
        name: f.name.clone(),
        params: sig.params.clone(),
        result: sig.result.clone(),
        registers: lo.next_reg,
        code: lo.code,
    })
}

impl<'a> Lower<'a> {
    fn fresh_reg(&mut self) -> usize {
        let r = self.next_reg;
        self.next_reg += 1;
        r
    }
    fn new_label(&mut self) -> usize {
        self.labels.push(usize::MAX);
        self.labels.len() - 1
    }
    fn place(&mut self, label: usize) {
        self.labels[label] = self.code.len();
    }
    fn bind(&mut self, name: &str, reg: usize, ty: Type) {
        self.scopes.last_mut().unwrap().insert(name.to_string(), (reg, ty));
    }
    fn lookup(&self, name: &str) -> Result<(usize, Type), String> {
        for scope in self.scopes.iter().rev() {
            if let Some((r, t)) = scope.get(name) {
                return Ok((*r, t.clone()));
            }
        }
        Err(format!("unknown variable `{name}`"))
    }

    fn patch_labels(&mut self) -> Result<(), String> {
        for instr in &mut self.code {
            match instr {
                Instruction::Jump { target } => *target = self.labels[*target],
                Instruction::Branch { then_pc, else_pc, .. } => {
                    *then_pc = self.labels[*then_pc];
                    *else_pc = self.labels[*else_pc];
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn stmt(&mut self, s: &Stmt) -> Result<(), String> {
        match s {
            Stmt::Let { name, value } => {
                let dst = self.fresh_reg();
                let ty = self.expr_into(value, dst)?;
                self.bind(name, dst, ty);
            }
            Stmt::Assign { name, value } => {
                let (dst, _) = self.lookup(name)?;
                self.expr_into(value, dst)?;
            }
            Stmt::Return(e) => {
                let (r, _) = self.expr(e)?;
                self.code.push(Instruction::Return { value: r });
            }
            Stmt::Emit(e) => {
                let (r, _) = self.expr(e)?;
                self.code.push(Instruction::Emit { value: r });
            }
            Stmt::Yield => self.code.push(Instruction::Yield),
            Stmt::Expr(e) => {
                let _ = self.expr(e)?;
            }
            Stmt::If { cond, then, els } => {
                let (cr, _) = self.expr(cond)?;
                let then_l = self.new_label();
                let else_l = self.new_label();
                let end_l = self.new_label();
                self.code.push(Instruction::Branch {
                    cond: cr,
                    then_pc: then_l,
                    else_pc: else_l,
                });
                self.place(then_l);
                self.scoped(|lo| lo.stmts(then))?;
                self.code.push(Instruction::Jump { target: end_l });
                self.place(else_l);
                self.scoped(|lo| lo.stmts(els))?;
                self.code.push(Instruction::Jump { target: end_l });
                self.place(end_l);
            }
            Stmt::While { cond, body } => {
                let head_l = self.new_label();
                let body_l = self.new_label();
                let exit_l = self.new_label();
                self.place(head_l);
                let (cr, _) = self.expr(cond)?;
                self.code.push(Instruction::Branch {
                    cond: cr,
                    then_pc: body_l,
                    else_pc: exit_l,
                });
                self.place(body_l);
                self.scoped(|lo| lo.stmts(body))?;
                self.code.push(Instruction::Jump { target: head_l });
                self.place(exit_l);
            }
        }
        Ok(())
    }

    fn stmts(&mut self, body: &[Stmt]) -> Result<(), String> {
        for s in body {
            self.stmt(s)?;
        }
        Ok(())
    }

    fn scoped<R>(&mut self, f: impl FnOnce(&mut Self) -> R) -> R {
        self.scopes.push(HashMap::new());
        let r = f(self);
        self.scopes.pop();
        r
    }

    /// Evaluate `e` into a fresh temporary (or, for a variable, reuse its
    /// register directly — reads don't need a copy).
    fn expr(&mut self, e: &Expr) -> Result<(usize, Type), String> {
        if let Expr::Var(name) = e {
            return self.lookup(name);
        }
        let dst = self.fresh_reg();
        let ty = self.expr_into(e, dst)?;
        Ok((dst, ty))
    }

    fn expr_into(&mut self, e: &Expr, dst: usize) -> Result<Type, String> {
        Ok(match e {
            Expr::Int(n) => {
                self.code.push(Instruction::Const { dst, value: Value::I64(*n) });
                Type::I64
            }
            Expr::Bool(b) => {
                self.code.push(Instruction::Const { dst, value: Value::Bool(*b) });
                Type::Bool
            }
            Expr::Unit => {
                self.code.push(Instruction::Const { dst, value: Value::Unit });
                Type::Unit
            }
            Expr::Var(name) => {
                let (src, ty) = self.lookup(name)?;
                self.code.push(Instruction::Copy { dst, src });
                ty
            }
            Expr::Binary { op, left, right } => {
                let (lr, _) = self.expr(left)?;
                let (rr, _) = self.expr(right)?;
                match op {
                    BinOp::Add => {
                        self.code.push(Instruction::AddI64 { dst, left: lr, right: rr });
                        Type::I64
                    }
                    BinOp::Sub => {
                        self.code.push(Instruction::SubI64 { dst, left: lr, right: rr });
                        Type::I64
                    }
                    BinOp::Lt => {
                        self.code.push(Instruction::LtI64 { dst, left: lr, right: rr });
                        Type::Bool
                    }
                    // `a > b` is `b < a`.
                    BinOp::Gt => {
                        self.code.push(Instruction::LtI64 { dst, left: rr, right: lr });
                        Type::Bool
                    }
                }
            }
            Expr::Field { object, field } => {
                let (obj, ty) = self.expr(object)?;
                let Type::Ref(type_id) = ty else {
                    return Err(format!("`.{field}` on a non-struct value"));
                };
                let info = self
                    .structs
                    .values()
                    .find(|s| s.id == type_id)
                    .ok_or("field access on an unknown struct")?;
                let (_, fid, fty) = info
                    .fields
                    .iter()
                    .find(|(n, _, _)| n == field)
                    .ok_or_else(|| format!("no field `{field}` on that struct"))?;
                self.code.push(Instruction::GetField {
                    dst,
                    object: obj,
                    field: *fid,
                });
                fty.clone()
            }
            Expr::StructLit { name, fields } => {
                let info = self
                    .structs
                    .get(name)
                    .ok_or_else(|| format!("unknown struct `{name}`"))?;
                let type_id = info.id;
                let field_ids: HashMap<&str, FieldId> =
                    info.fields.iter().map(|(n, id, _)| (n.as_str(), *id)).collect();
                let mut supplied = Vec::new();
                for (fname, fexpr) in fields {
                    let fid = *field_ids
                        .get(fname.as_str())
                        .ok_or_else(|| format!("struct `{name}` has no field `{fname}`"))?;
                    let (r, _) = self.expr(fexpr)?;
                    supplied.push((fid, r));
                }
                self.code.push(Instruction::New {
                    dst,
                    type_id,
                    fields: supplied,
                });
                Type::Ref(type_id)
            }
            Expr::Call { name, args } => {
                let sig = self
                    .fns
                    .get(name)
                    .ok_or_else(|| format!("unknown function `{name}`"))?;
                let (callee, result) = (sig.id, sig.result.clone());
                let mut arg_regs = Vec::new();
                for a in args {
                    let (r, _) = self.expr(a)?;
                    arg_regs.push(r);
                }
                self.code.push(Instruction::Call {
                    dst,
                    function: callee,
                    args: arg_regs,
                });
                result
            }
        })
    }
}
