//! Lowering: AST → runtime IR. Structs become `Schema`s, functions become
//! register-machine `Function`s. Locals get fixed registers (this IR has no
//! phi), sub-expressions get fresh temporaries, and control flow is emitted
//! with symbolic labels that are patched to program counters at the end.

use super::ast::*;
use crate::{
    DefId, Field, FieldId, ForeignFnId, ForeignKind, Function, Instruction, Schema, Type, Value,
    Version,
};
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
    // Foreign world: opaque resource kinds, native fn ids, and their signatures.
    foreign_type_ids: HashMap<String, ForeignKind>,
    foreign_fn_ids: HashMap<String, ForeignFnId>,
    foreign_sigs: HashMap<ForeignFnId, (Vec<Type>, Type)>,
    // Top-level `letonce` globals: their def ids and declared types.
    global_ids: HashMap<String, DefId>,
    global_types: HashMap<DefId, Type>,
    next_struct: DefId,
    next_fn: DefId,
    next_field: FieldId,
    next_foreign_type: ForeignKind,
    next_foreign_fn: ForeignFnId,
    next_global: DefId,
}

impl IdEnv {
    pub fn new() -> IdEnv {
        // Disjoint id ranges so a type id, a function id, and a global id never
        // collide (globals share the DefId space with structs/functions).
        IdEnv {
            next_struct: 1,
            next_fn: 1_000_000,
            next_field: 1,
            next_foreign_type: 1,
            next_foreign_fn: 1,
            next_global: 2_000_000,
            ..Default::default()
        }
    }

    fn foreign_type_id(&mut self, name: &str) -> ForeignKind {
        if let Some(id) = self.foreign_type_ids.get(name) {
            return *id;
        }
        let id = self.next_foreign_type;
        self.next_foreign_type += 1;
        self.foreign_type_ids.insert(name.to_string(), id);
        id
    }
    fn foreign_fn_id(&mut self, name: &str) -> ForeignFnId {
        if let Some(id) = self.foreign_fn_ids.get(name) {
            return *id;
        }
        let id = self.next_foreign_fn;
        self.next_foreign_fn += 1;
        self.foreign_fn_ids.insert(name.to_string(), id);
        id
    }
    fn global_id(&mut self, name: &str) -> DefId {
        if let Some(id) = self.global_ids.get(name) {
            return *id;
        }
        let id = self.next_global;
        self.next_global += 1;
        self.global_ids.insert(name.to_string(), id);
        id
    }
    /// The foreign-fn id and signature bound to `name`, if it is a `foreign fn`.
    fn foreign_fn_of(&self, name: &str) -> Option<(ForeignFnId, Vec<Type>, Type)> {
        let id = *self.foreign_fn_ids.get(name)?;
        let (params, result) = self.foreign_sigs.get(&id)?;
        Some((id, params.clone(), result.clone()))
    }
    /// The global def id and type bound to `name`, if it is a `letonce`.
    fn global_of(&self, name: &str) -> Option<(DefId, Type)> {
        let id = *self.global_ids.get(name)?;
        Some((id, self.global_types.get(&id)?.clone()))
    }
    /// A foreign-fn id resolved by name — for a host registering native impls.
    pub fn foreign_fn_id_of(&self, name: &str) -> Option<ForeignFnId> {
        self.foreign_fn_ids.get(name).copied()
    }
    /// A foreign-type kind resolved by name — so a native constructor can tag
    /// the handle it returns with the kind the declared signature expects.
    pub fn foreign_kind_of(&self, name: &str) -> Option<ForeignKind> {
        self.foreign_type_ids.get(name).copied()
    }
    /// The accumulated foreign signatures, for installing into the `World`.
    pub fn foreign_sigs(&self) -> HashMap<ForeignFnId, (Vec<Type>, Type)> {
        self.foreign_sigs.clone()
    }
    /// The accumulated global types, for installing into the `World`.
    pub fn global_types(&self) -> HashMap<DefId, Type> {
        self.global_types.clone()
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
            // A written name is a foreign resource type if declared as one,
            // otherwise a struct reference.
            if let Some(kind) = ids.foreign_type_ids.get(name) {
                Type::Foreign(*kind)
            } else {
                Type::Ref(ids.struct_of(name).ok_or_else(|| format!("unknown type `{name}`"))?)
            }
        }
    })
}

/// How a `letonce` global is initialized: the synthetic zero-arg function that
/// computes its first value, plus the global's def id and type. The session
/// installs `init_fn` and, if the global is not yet set, runs it once.
pub struct GlobalInit {
    pub global_id: DefId,
    pub init_fn: DefId,
}

/// The result of lowering one program (or one live edit).
pub struct Lowered {
    pub schemas: Vec<Schema>,
    pub functions: Vec<Function>,
    pub global_inits: Vec<GlobalInit>,
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
    let globals: Vec<&GlobalDef> = program
        .items
        .iter()
        .filter_map(|i| match i {
            Item::Global(g) => Some(g),
            _ => None,
        })
        .collect();

    // Register foreign types first — a struct field or signature may reference
    // one, and `resolve` needs the name→kind mapping to exist.
    for item in &program.items {
        if let Item::ForeignType(name) = item {
            ids.foreign_type_id(name);
        }
    }
    // Then foreign fn signatures (a body may call one).
    for item in &program.items {
        if let Item::ForeignFn(ff) = item {
            let id = ids.foreign_fn_id(&ff.name);
            let params: Vec<Type> = ff
                .params
                .iter()
                .map(|p| resolve(&p.ty, ids))
                .collect::<Result<_, _>>()?;
            let result = resolve(&ff.ret, ids)?;
            ids.foreign_sigs.insert(id, (params, result));
        }
    }

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

    // Register function signatures (so calls — including recursion — resolve).
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

    // Lower each `letonce` global into a synthetic zero-arg init function that
    // returns its value. Processed in source order so a later global's
    // initializer can read an earlier one; the global's type is published to
    // `ids` before any managed body is lowered, so a `LoadGlobal` types.
    let mut global_inits = Vec::new();
    for g in &globals {
        let global_id = ids.global_id(&g.name);
        let (init_fn, ty) = lower_init_fn(&g.name, &g.init, ids)?;
        ids.global_types.insert(global_id, ty);
        let init_fn_id = init_fn.id;
        functions.push(init_fn);
        global_inits.push(GlobalInit {
            global_id,
            init_fn: init_fn_id,
        });
    }

    // Lower the managed function bodies (now that globals resolve).
    for f in &fns {
        functions.push(lower_fn(f, ids)?);
    }

    Ok(Lowered {
        schemas,
        functions,
        global_inits,
    })
}

/// Lower a `letonce` initializer into a synthetic `__init_<name>` function that
/// computes and returns the value. Returns the function and its result type
/// (the global's inferred type).
fn lower_init_fn(name: &str, init: &Expr, ids: &mut IdEnv) -> Result<(Function, Type), String> {
    let fn_name = format!("__init_{name}");
    let id = ids.fn_id(&fn_name);
    let mut lo = Lower {
        ids,
        code: Vec::new(),
        labels: Vec::new(),
        next_reg: 0,
        scopes: vec![HashMap::new()],
    };
    let (r, ty) = lo.expr(init)?;
    lo.code.push(Instruction::Return { value: r });
    lo.patch_labels()?;
    let registers = lo.next_reg;
    let code = lo.code;
    ids.fn_sigs.insert(id, (Vec::new(), ty.clone()));
    Ok((
        Function {
            id,
            version: Version(1),
            name: fn_name,
            params: Vec::new(),
            result: ty.clone(),
            registers,
            code,
        },
        ty,
    ))
}

fn const_value(e: &Expr, ty: &Type) -> Result<Value, String> {
    let v = match e {
        Expr::Int(n) => Value::I64(*n),
        Expr::Bool(b) => Value::Bool(*b),
        Expr::Unit => Value::Unit,
        _ => return Err("field default must be a literal".into()),
    };
    if v.scalar_type().as_ref() != Some(ty) {
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
            // A local read reuses its register; a global read loads into a fresh
            // temporary (globals have no fixed frame slot).
            if let Ok(hit) = self.lookup(name) {
                return Ok(hit);
            }
            if let Some((gid, ty)) = self.ids.global_of(name) {
                let dst = self.fresh_reg();
                self.code.push(Instruction::LoadGlobal { dst, global: gid });
                return Ok((dst, ty));
            }
            return Err(format!("unknown variable `{name}`"));
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
                if let Ok((src, ty)) = self.lookup(name) {
                    self.code.push(Instruction::Copy { dst, src });
                    ty
                } else if let Some((gid, ty)) = self.ids.global_of(name) {
                    self.code.push(Instruction::LoadGlobal { dst, global: gid });
                    ty
                } else {
                    return Err(format!("unknown variable `{name}`"));
                }
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
                // A foreign fn lowers to a native call; anything else is a
                // managed call. Foreign names are checked first so a `foreign fn`
                // and a managed `fn` can't be confused.
                if let Some((foreign, _params, result)) = self.ids.foreign_fn_of(name) {
                    let mut arg_regs = Vec::new();
                    for a in args {
                        let (r, _) = self.expr(a)?;
                        arg_regs.push(r);
                    }
                    self.code.push(Instruction::CallForeign {
                        dst,
                        foreign,
                        args: arg_regs,
                    });
                    result
                } else {
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
            }
        })
    }
}
