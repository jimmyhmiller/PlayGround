//! Lowering: AST → runtime IR. Structs become `Schema`s, functions become
//! register-machine `Function`s. Locals get fixed registers (this IR has no
//! phi), sub-expressions get fresh temporaries, and control flow is emitted
//! with symbolic labels that are patched to program counters at the end.

use super::ast::*;
use crate::{DefId, Field, FieldId, Function, Instruction, Schema, Type, Value, Version};
use std::collections::HashMap;

/// The persistent symbol table of a compilation. It survives across live edits
/// (a [`crate::Session`] owns one), so a redefined struct keeps its `DefId` and
/// each field keeps its `FieldId` *by name* — which is exactly what lets an
/// auto-derived migration copy the fields that stayed the same. It also holds
/// the current layout of every struct and signature of every function, so a
/// later edit can reference definitions from an earlier one.
#[derive(Default)]
pub struct IdEnv {
    struct_ids: HashMap<String, DefId>,
    fn_ids: HashMap<String, DefId>,
    field_ids: HashMap<(DefId, String), FieldId>,
    struct_fields: HashMap<DefId, Vec<(String, FieldId, Type)>>,
    fn_sigs: HashMap<DefId, (Vec<Type>, Type)>,
    next_struct: DefId,
    next_fn: DefId,
    next_field: FieldId,
}

impl IdEnv {
    pub fn new() -> IdEnv {
        // Disjoint id ranges so a type id and a function id never collide.
        IdEnv {
            next_struct: 1,
            next_fn: 1_000_000,
            next_field: 1,
            ..Default::default()
        }
    }
    fn struct_id(&mut self, name: &str) -> DefId {
        if let Some(id) = self.struct_ids.get(name) {
            return *id;
        }
        let id = self.next_struct;
        self.next_struct += 1;
        self.struct_ids.insert(name.to_string(), id);
        id
    }
    fn fn_id(&mut self, name: &str) -> DefId {
        if let Some(id) = self.fn_ids.get(name) {
            return *id;
        }
        let id = self.next_fn;
        self.next_fn += 1;
        self.fn_ids.insert(name.to_string(), id);
        id
    }
    fn field_id(&mut self, struct_id: DefId, name: &str) -> FieldId {
        if let Some(id) = self.field_ids.get(&(struct_id, name.to_string())) {
            return *id;
        }
        let id = self.next_field;
        self.next_field += 1;
        self.field_ids.insert((struct_id, name.to_string()), id);
        id
    }
    pub fn struct_of(&self, name: &str) -> Option<DefId> {
        self.struct_ids.get(name).copied()
    }
    pub fn fn_of(&self, name: &str) -> Option<DefId> {
        self.fn_ids.get(name).copied()
    }
    pub fn struct_map(&self) -> HashMap<String, DefId> {
        self.struct_ids.clone()
    }
    pub fn fn_map(&self) -> HashMap<String, DefId> {
        self.fn_ids.clone()
    }
}

fn resolve(te: &TypeExpr, ids: &IdEnv) -> Result<Type, String> {
    Ok(match te {
        TypeExpr::I64 => Type::I64,
        TypeExpr::Bool => Type::Bool,
        TypeExpr::Unit => Type::Unit,
        TypeExpr::Ref(name) => {
            Type::Ref(ids.struct_of(name).ok_or_else(|| format!("unknown struct `{name}`"))?)
        }
    })
}

/// The result of lowering one program (or one live edit).
pub struct Lowered {
    pub schemas: Vec<Schema>,
    pub functions: Vec<Function>,
}

pub fn lower(program: &Program, ids: &mut IdEnv) -> Result<Lowered, String> {
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

    // Register every struct name first (so fields that reference each other
    // resolve), then build each schema and record its current layout.
    for s in &structs {
        ids.struct_id(&s.name);
    }
    let mut schemas = Vec::new();
    for s in &structs {
        let sid = ids.struct_of(&s.name).unwrap();
        let mut fields = Vec::new();
        let mut layout = Vec::new();
        for f in &s.fields {
            let fid = ids.field_id(sid, &f.name);
            let ty = resolve(&f.ty, ids)?;
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
            layout.push((f.name.clone(), fid, ty));
        }
        ids.struct_fields.insert(sid, layout);
        schemas.push(Schema {
            type_id: sid,
            version: Version(1), // the session rewrites this to the next version
            name: s.name.clone(),
            fields,
        });
    }

    // Register function signatures (so calls — including recursion — resolve),
    // then lower the bodies.
    for f in &fns {
        let id = ids.fn_id(&f.name);
        let params: Vec<Type> = f
            .params
            .iter()
            .map(|p| resolve(&p.ty, ids))
            .collect::<Result<_, _>>()?;
        let result = resolve(&f.ret, ids)?;
        ids.fn_sigs.insert(id, (params, result));
    }
    let mut functions = Vec::new();
    for f in &fns {
        functions.push(lower_fn(f, ids)?);
    }

    Ok(Lowered { schemas, functions })
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
    ids: &'a IdEnv,
    code: Vec<Instruction>,
    labels: Vec<usize>,
    next_reg: usize,
    scopes: Vec<HashMap<String, (usize, Type)>>,
}

fn lower_fn(f: &FnDef, ids: &IdEnv) -> Result<Function, String> {
    let id = ids.fn_of(&f.name).unwrap();
    let (params, result) = ids.fn_sigs[&id].clone();
    let mut lo = Lower {
        ids,
        code: Vec::new(),
        labels: Vec::new(),
        next_reg: 0,
        scopes: vec![HashMap::new()],
    };
    for (p, ty) in f.params.iter().zip(&params) {
        let r = lo.fresh_reg();
        lo.bind(&p.name, r, ty.clone());
    }
    for s in &f.body {
        lo.stmt(s)?;
    }
    lo.patch_labels()?;
    Ok(Function {
        id,
        version: Version(1),
        name: f.name.clone(),
        params,
        result,
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
                    BinOp::Mul => {
                        self.code.push(Instruction::MulI64 { dst, left: lr, right: rr });
                        Type::I64
                    }
                    BinOp::Lt => {
                        self.code.push(Instruction::LtI64 { dst, left: lr, right: rr });
                        Type::Bool
                    }
                    BinOp::Gt => {
                        // `a > b` is `b < a`.
                        self.code.push(Instruction::LtI64 { dst, left: rr, right: lr });
                        Type::Bool
                    }
                    BinOp::Eq => {
                        self.code.push(Instruction::EqI64 { dst, left: lr, right: rr });
                        Type::Bool
                    }
                    // The rest compose from `<`/`==` and `!`.
                    BinOp::Ne => {
                        let t = self.fresh_reg();
                        self.code.push(Instruction::EqI64 { dst: t, left: lr, right: rr });
                        self.code.push(Instruction::Not { dst, src: t });
                        Type::Bool
                    }
                    BinOp::Le => {
                        // `a <= b` is `!(b < a)`.
                        let t = self.fresh_reg();
                        self.code.push(Instruction::LtI64 { dst: t, left: rr, right: lr });
                        self.code.push(Instruction::Not { dst, src: t });
                        Type::Bool
                    }
                    BinOp::Ge => {
                        // `a >= b` is `!(a < b)`.
                        let t = self.fresh_reg();
                        self.code.push(Instruction::LtI64 { dst: t, left: lr, right: rr });
                        self.code.push(Instruction::Not { dst, src: t });
                        Type::Bool
                    }
                }
            }
            Expr::Not(inner) => {
                let (r, _) = self.expr(inner)?;
                self.code.push(Instruction::Not { dst, src: r });
                Type::Bool
            }
            Expr::Field { object, field } => {
                let (obj, ty) = self.expr(object)?;
                let Type::Ref(type_id) = ty else {
                    return Err(format!("`.{field}` on a non-struct value"));
                };
                let layout = self
                    .ids
                    .struct_fields
                    .get(&type_id)
                    .ok_or("field access on an unknown struct")?;
                let (_, fid, fty) = layout
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
                let type_id = self
                    .ids
                    .struct_of(name)
                    .ok_or_else(|| format!("unknown struct `{name}`"))?;
                let layout = self
                    .ids
                    .struct_fields
                    .get(&type_id)
                    .ok_or_else(|| format!("unknown struct `{name}`"))?;
                let field_ids: HashMap<&str, FieldId> =
                    layout.iter().map(|(n, id, _)| (n.as_str(), *id)).collect();
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
                let callee = self
                    .ids
                    .fn_of(name)
                    .ok_or_else(|| format!("unknown function `{name}`"))?;
                let result = self.ids.fn_sigs[&callee].1.clone();
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
