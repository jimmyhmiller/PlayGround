//! Monomorphization.
//!
//! Generics are erased before type-checking by stamping out a concrete copy of
//! each generic function/struct per set of concrete type arguments — the
//! AOT-friendly approach (Rust/C++/Zig style), so the checker and codegen stay
//! monomorphic. Type arguments are explicit: `(id [i64] x)`, `(Pair i64 i64)`.
//!
//! Pipeline position: `read → expand → parse → ►monomorphize◄ → check → codegen`.

use std::collections::{HashMap, HashSet};

use crate::ast::*;

pub fn monomorphize(program: Program) -> Result<Program, String> {
    let mut m = Mono {
        gfuncs: program
            .funcs
            .iter()
            .filter(|f| !f.type_params.is_empty())
            .map(|f| (f.name.clone(), f))
            .collect(),
        gstructs: program
            .structs
            .iter()
            .filter(|s| !s.type_params.is_empty())
            .map(|s| (s.name.clone(), s))
            .collect(),
        out_structs: HashMap::new(),
        out_funcs: HashMap::new(),
        pending_structs: Vec::new(),
        pending_funcs: Vec::new(),
        queued_structs: HashSet::new(),
        queued_funcs: HashSet::new(),
    };
    let empty = HashMap::new();

    // seed from concrete (non-generic) definitions
    for sd in program.structs.iter().filter(|s| s.type_params.is_empty()) {
        let fields = sd
            .fields
            .iter()
            .map(|(n, t)| Ok((n.clone(), m.resolve_ty(t, &empty)?)))
            .collect::<Result<Vec<_>, String>>()?;
        m.out_structs.insert(
            sd.name.clone(),
            StructDef {
                name: sd.name.clone(),
                type_params: vec![],
                fields,
            },
        );
    }
    for f in program.funcs.iter().filter(|f| f.type_params.is_empty()) {
        let nf = m.resolve_func(f, &empty, f.name.clone())?;
        m.out_funcs.insert(f.name.clone(), nf);
    }

    m.drain()?;

    Ok(Program {
        conventions: program.conventions,
        externs: program.externs,
        structs: m.out_structs.into_values().collect(),
        funcs: m.out_funcs.into_values().collect(),
    })
}

type Subst = HashMap<String, Type>;

struct Mono<'a> {
    gfuncs: HashMap<String, &'a Func>,
    gstructs: HashMap<String, &'a StructDef>,
    out_structs: HashMap<String, StructDef>,
    out_funcs: HashMap<String, Func>,
    pending_structs: Vec<(String, Vec<Type>)>,
    pending_funcs: Vec<(String, Vec<Type>)>,
    queued_structs: HashSet<String>,
    queued_funcs: HashSet<String>,
}

impl<'a> Mono<'a> {
    fn drain(&mut self) -> Result<(), String> {
        loop {
            if let Some((name, args)) = self.pending_structs.pop() {
                let gs: &'a StructDef = self.gstructs[&name];
                let map = subst_map(&gs.type_params, &args);
                let mangled = mangle(&name, &args);
                let fields = gs
                    .fields
                    .iter()
                    .map(|(n, t)| Ok((n.clone(), self.resolve_ty(t, &map)?)))
                    .collect::<Result<Vec<_>, String>>()?;
                self.out_structs.insert(
                    mangled.clone(),
                    StructDef {
                        name: mangled,
                        type_params: vec![],
                        fields,
                    },
                );
                continue;
            }
            if let Some((name, args)) = self.pending_funcs.pop() {
                let gf: &'a Func = self.gfuncs[&name];
                let map = subst_map(&gf.type_params, &args);
                let mangled = mangle(&name, &args);
                let nf = self.resolve_func(gf, &map, mangled.clone())?;
                self.out_funcs.insert(mangled, nf);
                continue;
            }
            return Ok(());
        }
    }

    fn queue_struct(&mut self, name: String, args: Vec<Type>) {
        if self.queued_structs.insert(mangle(&name, &args)) {
            self.pending_structs.push((name, args));
        }
    }

    fn queue_func(&mut self, name: String, args: Vec<Type>) {
        if self.queued_funcs.insert(mangle(&name, &args)) {
            self.pending_funcs.push((name, args));
        }
    }

    /// Substitute type parameters and resolve generic applications to concrete
    /// (mangled) struct names, queueing any newly-needed instantiations.
    fn resolve_ty(&mut self, t: &Type, map: &Subst) -> Result<Type, String> {
        Ok(match t {
            Type::Int(w) => Type::Int(*w),
            Type::Ptr(p) => Type::Ptr(Box::new(self.resolve_ty(p, map)?)),
            Type::Array(e, n) => Type::Array(Box::new(self.resolve_ty(e, map)?), *n),
            Type::Fn(cc, ps, r) => Type::Fn(
                cc.clone(),
                ps.iter().map(|p| self.resolve_ty(p, map)).collect::<Result<_, _>>()?,
                Box::new(self.resolve_ty(r, map)?),
            ),
            Type::Struct(name) => match map.get(name) {
                Some(concrete) => concrete.clone(), // a type parameter
                None => Type::Struct(name.clone()),  // a concrete struct
            },
            Type::App(name, args) => {
                let rargs = args
                    .iter()
                    .map(|a| self.resolve_ty(a, map))
                    .collect::<Result<Vec<_>, _>>()?;
                let gs = self
                    .gstructs
                    .get(name)
                    .ok_or_else(|| format!("unknown generic type '{name}'"))?;
                if gs.type_params.len() != rargs.len() {
                    return Err(format!(
                        "generic type '{name}' expects {} type arguments, got {}",
                        gs.type_params.len(),
                        rargs.len()
                    ));
                }
                let mangled = mangle(name, &rargs);
                self.queue_struct(name.clone(), rargs);
                Type::Struct(mangled)
            }
        })
    }

    fn resolve_func(&mut self, f: &Func, map: &Subst, name: String) -> Result<Func, String> {
        let params = f
            .params
            .iter()
            .map(|p| {
                Ok(Param {
                    name: p.name.clone(),
                    ty: self.resolve_ty(&p.ty, map)?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let ret = self.resolve_ty(&f.ret, map)?;
        let body = f
            .body
            .iter()
            .map(|e| self.resolve_expr(e, map))
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Func {
            name,
            type_params: vec![],
            cc: f.cc.clone(),
            params,
            ret,
            body,
        })
    }

    fn resolve_expr(&mut self, e: &Expr, map: &Subst) -> Result<Expr, String> {
        let go = |m: &mut Self, e: &Expr| m.resolve_expr(e, map);
        Ok(match e {
            Expr::Int(n) => Expr::Int(*n),
            Expr::Var(s) => Expr::Var(s.clone()),
            Expr::Bin { op, lhs, rhs } => Expr::Bin {
                op: *op,
                lhs: Box::new(go(self, lhs)?),
                rhs: Box::new(go(self, rhs)?),
            },
            Expr::Cmp { op, lhs, rhs } => Expr::Cmp {
                op: *op,
                lhs: Box::new(go(self, lhs)?),
                rhs: Box::new(go(self, rhs)?),
            },
            Expr::If { cond, then, els } => Expr::If {
                cond: Box::new(go(self, cond)?),
                then: Box::new(go(self, then)?),
                els: Box::new(go(self, els)?),
            },
            Expr::Do(es) => Expr::Do(es.iter().map(|e| go(self, e)).collect::<Result<_, _>>()?),
            Expr::Let { binds, body } => Expr::Let {
                binds: binds
                    .iter()
                    .map(|(n, e)| Ok((n.clone(), go(self, e)?)))
                    .collect::<Result<_, String>>()?,
                body: body.iter().map(|e| go(self, e)).collect::<Result<_, _>>()?,
            },
            Expr::Call { func, type_args, args } => {
                let args = args.iter().map(|a| go(self, a)).collect::<Result<Vec<_>, _>>()?;
                if let Some(gf) = self.gfuncs.get(func).copied() {
                    if type_args.is_empty() {
                        return Err(format!(
                            "generic function '{func}' needs explicit type arguments: ({func} [<types>] ...)"
                        ));
                    }
                    let rtargs = type_args
                        .iter()
                        .map(|t| self.resolve_ty(t, map))
                        .collect::<Result<Vec<_>, _>>()?;
                    if rtargs.len() != gf.type_params.len() {
                        return Err(format!(
                            "generic function '{func}' expects {} type arguments, got {}",
                            gf.type_params.len(),
                            rtargs.len()
                        ));
                    }
                    let mangled = mangle(func, &rtargs);
                    self.queue_func(func.clone(), rtargs);
                    Expr::Call { func: mangled, type_args: vec![], args }
                } else {
                    Expr::Call { func: func.clone(), type_args: vec![], args }
                }
            }
            Expr::Alloc { storage, ty } => Expr::Alloc {
                storage: *storage,
                ty: self.resolve_ty(ty, map)?,
            },
            Expr::Field { ptr, field } => Expr::Field {
                ptr: Box::new(go(self, ptr)?),
                field: field.clone(),
            },
            Expr::Load(p) => Expr::Load(Box::new(go(self, p)?)),
            Expr::Store { ptr, val } => Expr::Store {
                ptr: Box::new(go(self, ptr)?),
                val: Box::new(go(self, val)?),
            },
            Expr::Index { ptr, idx } => Expr::Index {
                ptr: Box::new(go(self, ptr)?),
                idx: Box::new(go(self, idx)?),
            },
            Expr::Cast { ty, expr } => Expr::Cast {
                ty: self.resolve_ty(ty, map)?,
                expr: Box::new(go(self, expr)?),
            },
            Expr::SizeOf(ty) => Expr::SizeOf(self.resolve_ty(ty, map)?),
            Expr::Free(p) => Expr::Free(Box::new(go(self, p)?)),
            Expr::FnPtrOf(name) => {
                if self.gfuncs.contains_key(name) {
                    return Err(format!(
                        "fnptr-of '{name}': cannot take a function pointer to a generic function"
                    ));
                }
                Expr::FnPtrOf(name.clone())
            }
            Expr::CallPtr { fp, args } => Expr::CallPtr {
                fp: Box::new(go(self, fp)?),
                args: args.iter().map(|a| go(self, a)).collect::<Result<_, _>>()?,
            },
        })
    }
}

fn subst_map(params: &[String], args: &[Type]) -> Subst {
    params.iter().cloned().zip(args.iter().cloned()).collect()
}

/// A unique concrete name for an instantiation, e.g. `Pair__i64__i64`.
fn mangle(name: &str, args: &[Type]) -> String {
    let mut s = name.to_string();
    for a in args {
        s.push_str("__");
        s.push_str(&type_key(a));
    }
    s
}

fn type_key(t: &Type) -> String {
    match t {
        Type::Int(w) => format!("i{w}"),
        Type::Ptr(p) => format!("ptr_{}", type_key(p)),
        Type::Struct(s) => s.clone(),
        Type::Array(e, n) => format!("arr{n}_{}", type_key(e)),
        Type::Fn(..) => "fn".to_string(),
        Type::App(n, a) => mangle(n, a),
    }
}
