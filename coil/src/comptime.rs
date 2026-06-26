//! Compile-time evaluation. `(comptime E)` runs `E` in the *real* language during
//! compilation and splices the resulting literal — the same `defn`s, the same
//! `=`/arithmetic, the same `match`, executed by interpretation instead of being
//! lowered to machine code. The bridge toward "the whole language at compile
//! time": runtime code becomes available at compile time by running it.
//!
//! Stage 1b adds a compile-time **memory model**: mutable `let`s, `loop`/`break`/
//! `continue`, and aggregates (structs/arrays/sums) with `field`/`index`/`load`/
//! `store!`/`zeroed`/`alloc` and by-reference arguments. Memory is modelled with
//! reference-counted cells; aggregate *values* are references into them, with
//! value-semantics copies where the language copies. A `comptime` form must still
//! produce a SCALAR result (returning an aggregate is the next increment). FFI/
//! `llvm-ir`/function pointers/generic calls remain unsupported and error clearly.

use crate::ast::*;
use crate::span::Span;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

/// A mutable storage cell. Holds *content* (a scalar or an aggregate) — never a
/// `Ref` (references point AT cells).
type Cell = Rc<RefCell<CtVal>>;

#[derive(Clone, Debug)]
enum CtVal {
    Int(i64),
    Bool(bool),
    Float(f64),
    /// A `(slice u8)` string value (e.g. a reflected field name). Treated as a
    /// scalar (immutable); lowers back to a string literal.
    Str(String),
    /// Quoted code (Stage 3): raw syntax manipulated at comptime. Has no runtime
    /// representation, so it can't be a `comptime` result.
    Code(crate::reader::Sexp),
    /// A struct's fields, by declaration order, each in its own cell (so a `field`
    /// place is a real reference). `name` resolves field names to indices.
    Struct { name: String, fields: Vec<Cell> },
    /// Array elements, each in a cell, plus the element type (needed to rebuild
    /// the array as a result via `alloc-stack`).
    Array { elem: Type, cells: Vec<Cell> },
    Sum { variant: String, fields: Vec<Cell> },
    /// A pointer / place: a handle to a cell. Aggregate *values* are `Ref`s.
    Ref(Cell),
}

/// Control-flow outcome of evaluating an expression.
enum Flow {
    Val(CtVal),
    Brk(Option<CtVal>, Option<String>),
    Cont(Option<String>),
}

/// Extract a value, propagating a `break`/`continue` up to the enclosing loop.
macro_rules! val {
    ($e:expr) => {{
        match $e {
            Flow::Val(v) => v,
            ctrl => return Ok(ctrl),
        }
    }};
}

struct Ctx<'a> {
    fns: &'a HashMap<String, Func>,
    structs: &'a HashMap<String, StructDef>,
    sums: &'a HashMap<String, SumDef>,
    variants: &'a HashSet<String>,
    fuel: std::cell::Cell<u64>,
    gensym: std::cell::Cell<u64>,
}

const FUEL: u64 = 50_000_000;

/// Replace every `(comptime …)` in the checked program with its evaluated literal.
pub fn fold_program(
    funcs: &mut Vec<Func>,
    asserts: &mut [StaticAssert],
    structs: &[StructDef],
    sums: &[SumDef],
) -> Result<(), String> {
    let table: HashMap<String, Func> = funcs.iter().map(|f| (f.name.clone(), f.clone())).collect();
    let smap: HashMap<String, StructDef> = structs.iter().map(|s| (s.name.clone(), s.clone())).collect();
    let smap_sums: HashMap<String, SumDef> = sums.iter().map(|s| (s.name.clone(), s.clone())).collect();
    let variants: HashSet<String> =
        sums.iter().flat_map(|s| s.variants.iter().map(|v| v.name.clone())).collect();
    let ctx = Ctx {
        fns: &table,
        structs: &smap,
        sums: &smap_sums,
        variants: &variants,
        fuel: std::cell::Cell::new(FUEL),
        gensym: std::cell::Cell::new(0),
    };
    for f in funcs.iter_mut() {
        let body = std::mem::take(&mut f.body);
        f.body = body.iter().map(|e| fold_expr(e, &ctx)).collect::<Result<_, _>>()?;
    }
    // Static-assert conditions can use `(comptime …)` too (the assertion then runs
    // real code at compile time; codegen's const-eval sees the folded literal).
    for a in asserts.iter_mut() {
        a.cond = fold_expr(&a.cond, &ctx)?;
    }
    Ok(())
}

/// Evaluate every `(meta …)` form to the top-level syntax it generates. Each meta
/// expression must produce a `Code` value; a `(do …)` result splices into several
/// top-level forms. Run by the elaboration loop after the generator functions are
/// checked.
pub fn run_metas(program: &Program) -> Result<Vec<crate::reader::Sexp>, String> {
    use crate::reader::SexpKind;
    let table: HashMap<String, Func> = program.funcs.iter().map(|f| (f.name.clone(), f.clone())).collect();
    let smap: HashMap<String, StructDef> = program.structs.iter().map(|s| (s.name.clone(), s.clone())).collect();
    let smap_sums: HashMap<String, SumDef> = program.sums.iter().map(|s| (s.name.clone(), s.clone())).collect();
    let variants: HashSet<String> =
        program.sums.iter().flat_map(|s| s.variants.iter().map(|v| v.name.clone())).collect();
    let ctx = Ctx {
        fns: &table,
        structs: &smap,
        sums: &smap_sums,
        variants: &variants,
        fuel: std::cell::Cell::new(FUEL),
        gensym: std::cell::Cell::new(0),
    };
    let mut out = Vec::new();
    for meta in &program.metas {
        let v = match eval(meta, &mut HashMap::new(), &ctx)? {
            Flow::Val(v) => v,
            _ => return Err("meta: stray control flow at top level".to_string()),
        };
        let sexp = match v {
            CtVal::Code(s) => s,
            _ => return Err("meta: expression must produce Code (a quoted form)".to_string()),
        };
        // A `(do f1 f2 …)` result generates several top-level forms.
        match &sexp.kind {
            SexpKind::List(items) if matches!(items.first().map(|s| &s.kind), Some(SexpKind::Sym(h)) if h == "do") => {
                out.extend(items[1..].iter().cloned());
            }
            _ => out.push(sexp),
        }
    }
    Ok(out)
}

/// Expand one expression-macro call: run the macro function `fn_name` (a checked
/// `[Code…] -> Code` function in `program`) with its arguments bound to the quoted
/// call-site forms, returning the generated syntax.
pub fn expand_macro(
    program: &Program,
    fn_name: &str,
    args: Vec<crate::reader::Sexp>,
) -> Result<crate::reader::Sexp, String> {
    let table: HashMap<String, Func> = program.funcs.iter().map(|f| (f.name.clone(), f.clone())).collect();
    let smap: HashMap<String, StructDef> = program.structs.iter().map(|s| (s.name.clone(), s.clone())).collect();
    let smap_sums: HashMap<String, SumDef> = program.sums.iter().map(|s| (s.name.clone(), s.clone())).collect();
    let variants: HashSet<String> =
        program.sums.iter().flat_map(|s| s.variants.iter().map(|v| v.name.clone())).collect();
    let ctx = Ctx {
        fns: &table,
        structs: &smap,
        sums: &smap_sums,
        variants: &variants,
        fuel: std::cell::Cell::new(FUEL),
        gensym: std::cell::Cell::new(0),
    };
    let callee = table.get(fn_name).ok_or_else(|| format!("macro '{fn_name}' not found"))?;
    if callee.params.len() != args.len() {
        return Err(format!(
            "macro '{fn_name}': expects {} args, got {}",
            callee.params.len(),
            args.len()
        ));
    }
    let mut env: Env = HashMap::new();
    for (p, a) in callee.params.iter().zip(args) {
        env.insert(p.name.clone(), CtVal::Code(a));
    }
    match eval_seq(&callee.body, &mut env, &ctx)? {
        Flow::Val(CtVal::Code(s)) => Ok(s),
        Flow::Val(_) => Err(format!("macro '{fn_name}' did not return Code")),
        _ => Err(format!("macro '{fn_name}': stray control flow")),
    }
}

/// Evaluate a const's value expression to a constant-initializer tree, for an
/// aggregate const lowered to a static global.
pub fn eval_const(
    value: &Expr,
    funcs: &[Func],
    structs: &[StructDef],
    sums: &[SumDef],
) -> Result<ConstInit, String> {
    let table: HashMap<String, Func> = funcs.iter().map(|f| (f.name.clone(), f.clone())).collect();
    let smap: HashMap<String, StructDef> = structs.iter().map(|s| (s.name.clone(), s.clone())).collect();
    let smap_sums: HashMap<String, SumDef> = sums.iter().map(|s| (s.name.clone(), s.clone())).collect();
    let variants: HashSet<String> =
        sums.iter().flat_map(|s| s.variants.iter().map(|v| v.name.clone())).collect();
    let ctx = Ctx {
        fns: &table,
        structs: &smap,
        sums: &smap_sums,
        variants: &variants,
        fuel: std::cell::Cell::new(FUEL),
        gensym: std::cell::Cell::new(0),
    };
    let v = match eval(value, &mut HashMap::new(), &ctx)? {
        Flow::Val(v) => v,
        _ => return Err("comptime: stray break/continue in const value".to_string()),
    };
    ctval_to_const_init(&v)
}

fn ctval_to_const_init(v: &CtVal) -> Result<ConstInit, String> {
    match v {
        CtVal::Int(n) => Ok(ConstInit::Int(*n)),
        CtVal::Bool(b) => Ok(ConstInit::Bool(*b)),
        CtVal::Float(x) => Ok(ConstInit::Float(*x)),
        CtVal::Str(s) => Ok(ConstInit::Str(s.clone())),
        CtVal::Ref(cell) => ctval_to_const_init(&cell.borrow()),
        CtVal::Struct { fields, .. } => Ok(ConstInit::Struct(
            fields.iter().map(|c| ctval_to_const_init(&c.borrow())).collect::<Result<_, _>>()?,
        )),
        CtVal::Array { cells, .. } => Ok(ConstInit::Array(
            cells.iter().map(|c| ctval_to_const_init(&c.borrow())).collect::<Result<_, _>>()?,
        )),
        CtVal::Sum { .. } => Err(
            "comptime: a sum value can't initialize a static const yet (scalar/struct/array only)"
                .to_string(),
        ),
        CtVal::Code(_) => Err("comptime: a Code value can't initialize a const".to_string()),
    }
}

// ---- the comptime fold (find Comptime nodes, evaluate, splice literals) -------

/// Fold the unquote holes of a quasiquote template (literal syntax is untouched).
fn fold_quasi(q: &Quasi, ctx: &Ctx) -> Result<Quasi, String> {
    Ok(match q {
        Quasi::Lit(s) => Quasi::Lit(s.clone()),
        Quasi::Unquote(e) => Quasi::Unquote(Box::new(fold_expr(e, ctx)?)),
        Quasi::List(items) => Quasi::List(items.iter().map(|i| fold_quasi(i, ctx)).collect::<Result<_, _>>()?),
        Quasi::Vector(items) => Quasi::Vector(items.iter().map(|i| fold_quasi(i, ctx)).collect::<Result<_, _>>()?),
    })
}

fn fold_expr(e: &Expr, ctx: &Ctx) -> Result<Expr, String> {
    let go = |e: &Expr| fold_expr(e, ctx);
    let gb = |e: &Expr| -> Result<Box<Expr>, String> { Ok(Box::new(fold_expr(e, ctx)?)) };
    let kind = match &e.kind {
        ExprKind::Comptime(inner) => {
            // Evaluate directly — `eval` handles nested comptime/reflection with the
            // proper environment. (Don't `fold_expr` into the inner: that would
            // eagerly evaluate reflection nodes whose index is a loop variable not
            // yet bound, with an empty environment.)
            let v = match eval(inner, &mut HashMap::new(), ctx)? {
                Flow::Val(v) => v,
                _ => return Err("comptime: stray break/continue at top level".to_string()),
            };
            return build_value(&v, e.span, ctx);
        }
        // Type-reflection queries fold to a literal everywhere (like sizeof).
        ExprKind::TypeQuery { q, ty } => {
            let v = type_query(*q, ty, ctx)?;
            return build_value(&v, e.span, ctx);
        }
        // Per-field reflection folds when its index is a compile-time constant
        // (a literal, or evaluated inside a `comptime`); a runtime index errors.
        ExprKind::FieldMeta { .. } | ExprKind::FieldIndex { .. } => {
            let v = match eval(e, &mut HashMap::new(), ctx)? {
                Flow::Val(v) => v,
                _ => return Err("comptime: stray control flow in field reflection".to_string()),
            };
            return build_value(&v, e.span, ctx);
        }
        // Code-as-data nodes are NOT folded here: inside a generator function they
        // are the function's computation (run when it's called at comptime), and
        // inside `(comptime …)` the Comptime arm evaluates them. Just recurse.
        ExprKind::Quote(_) => e.kind.clone(),
        ExprKind::CodeOp { op, args } => ExprKind::CodeOp {
            op: *op,
            args: args.iter().map(go).collect::<Result<_, _>>()?,
        },
        ExprKind::Quasi(q) => ExprKind::Quasi(fold_quasi(q, ctx)?),
        ExprKind::Int(_)
        | ExprKind::Float(_)
        | ExprKind::Bool(_)
        | ExprKind::Str(_)
        | ExprKind::CStr(_)
        | ExprKind::Var(_)
        | ExprKind::Zeroed(_)
        | ExprKind::SizeOf(_)
        | ExprKind::AlignOf(_)
        | ExprKind::OffsetOf(_, _)
        | ExprKind::FnPtrOf(_)
        | ExprKind::StaticRef(_)
        | ExprKind::Continue { .. } => e.kind.clone(),
        ExprKind::Borrow { mutable, place } => ExprKind::Borrow { mutable: *mutable, place: gb(place)? },
        ExprKind::SpillRef(x) => ExprKind::SpillRef(gb(x)?),
        ExprKind::Not(x) => ExprKind::Not(gb(x)?),
        ExprKind::Load(x) => ExprKind::Load(gb(x)?),
        ExprKind::Free(x) => ExprKind::Free(gb(x)?),
        ExprKind::Cast { ty, expr } => ExprKind::Cast { ty: ty.clone(), expr: gb(expr)? },
        ExprKind::Bin { op, lhs, rhs } => ExprKind::Bin { op: *op, lhs: gb(lhs)?, rhs: gb(rhs)? },
        ExprKind::Cmp { op, lhs, rhs } => ExprKind::Cmp { op: *op, lhs: gb(lhs)?, rhs: gb(rhs)? },
        ExprKind::If { cond, then, els } => ExprKind::If { cond: gb(cond)?, then: gb(then)?, els: gb(els)? },
        ExprKind::Do(es) => ExprKind::Do(es.iter().map(go).collect::<Result<_, _>>()?),
        ExprKind::Loop { label, body } => ExprKind::Loop {
            label: label.clone(),
            body: body.iter().map(go).collect::<Result<_, _>>()?,
        },
        ExprKind::Break { label, value } => ExprKind::Break {
            label: label.clone(),
            value: match value { Some(v) => Some(gb(v)?), None => None },
        },
        ExprKind::Let { binds, body } => ExprKind::Let {
            binds: binds.iter().map(|(n, m, e)| Ok((n.clone(), *m, fold_expr(e, ctx)?))).collect::<Result<_, String>>()?,
            body: body.iter().map(go).collect::<Result<_, _>>()?,
        },
        ExprKind::Call { func, type_args, args } => ExprKind::Call {
            func: func.clone(),
            type_args: type_args.clone(),
            args: args.iter().map(go).collect::<Result<_, _>>()?,
        },
        ExprKind::Alloc { storage, ty } => ExprKind::Alloc { storage: *storage, ty: ty.clone() },
        ExprKind::Field { ptr, field } => ExprKind::Field { ptr: gb(ptr)?, field: field.clone() },
        ExprKind::Store { ptr, val } => ExprKind::Store { ptr: gb(ptr)?, val: gb(val)? },
        ExprKind::Index { ptr, idx } => ExprKind::Index { ptr: gb(ptr)?, idx: gb(idx)? },
        ExprKind::BitGet { ptr, field } => ExprKind::BitGet { ptr: gb(ptr)?, field: field.clone() },
        ExprKind::BitSet { ptr, field, val } => ExprKind::BitSet { ptr: gb(ptr)?, field: field.clone(), val: gb(val)? },
        ExprKind::Construct { sum, variant, args } => ExprKind::Construct {
            sum: sum.clone(),
            variant: variant.clone(),
            args: args.iter().map(go).collect::<Result<_, _>>()?,
        },
        ExprKind::Match { scrut, arms } => ExprKind::Match {
            scrut: gb(scrut)?,
            arms: arms.iter().map(|a| Ok(Arm { variant: a.variant.clone(), binds: a.binds.clone(), body: fold_expr(&a.body, ctx)? })).collect::<Result<_, String>>()?,
        },
        ExprKind::CallPtr { fp, args } => ExprKind::CallPtr {
            fp: gb(fp)?,
            args: args.iter().map(go).collect::<Result<_, _>>()?,
        },
        ExprKind::LlvmIr { result, args, body } => ExprKind::LlvmIr {
            result: result.clone(),
            args: args.iter().map(go).collect::<Result<_, _>>()?,
            body: body.clone(),
        },
        ExprKind::TraitCall { .. } => e.kind.clone(),
    };
    Ok(Expr::new(kind, e.span))
}

/// Synthesize an (already-elaborated) expression that *reconstructs* a comptime
/// value — the "literal-back". Scalars are literals; a sum is a variant call; a
/// struct is built imperatively (`let (mut t) (zeroed S)` + a `store!` per field +
/// `(load t)`), recursing for nested aggregates. Arrays aren't supported yet
/// (no element type recorded on the value).
fn build_value(v: &CtVal, span: Span, ctx: &Ctx) -> Result<Expr, String> {
    let e = |k| Expr::new(k, span);
    Ok(match v {
        CtVal::Int(n) => e(ExprKind::Int(*n)),
        CtVal::Bool(b) => e(ExprKind::Bool(*b)),
        CtVal::Float(x) => e(ExprKind::Float(*x)),
        CtVal::Str(s) => e(ExprKind::Str(s.clone())),
        CtVal::Code(_) => {
            return Err("comptime: a Code value can't be used as a runtime value \
                        (it's only meaningful at compile time)"
                .to_string())
        }
        CtVal::Ref(cell) => build_content(&cell.borrow(), span, ctx)?,
        CtVal::Struct { .. } | CtVal::Array { .. } | CtVal::Sum { .. } => {
            // aggregates only appear behind a Ref as values
            build_content(v, span, ctx)?
        }
    })
}

fn build_content(c: &CtVal, span: Span, ctx: &Ctx) -> Result<Expr, String> {
    let e = |k| Expr::new(k, span);
    match c {
        CtVal::Int(_) | CtVal::Bool(_) | CtVal::Float(_) | CtVal::Str(_) | CtVal::Code(_) => {
            build_value(c, span, ctx)
        }
        CtVal::Sum { variant, fields } => {
            // pre-mono, sum construction is a call to the variant name
            let args = fields
                .iter()
                .map(|f| build_value(&cell_value(f), span, ctx))
                .collect::<Result<_, _>>()?;
            Ok(e(ExprKind::Call { func: variant.clone(), type_args: vec![], args }))
        }
        CtVal::Struct { name, fields } => {
            // `(let [t (alloc-stack S)]  (store! (field t f) v)…  (load t))`.
            // `t` is an immutable binding holding a real `(ptr S)`, so `field`/
            // `load` see a pointer (a mutable local would bind the struct *value*).
            let sd = ctx.structs.get(name).ok_or_else(|| format!("comptime: unknown struct '{name}'"))?;
            let n = ctx.gensym.get();
            ctx.gensym.set(n + 1);
            let t = format!("$ct{n}");
            let var = || Expr::new(ExprKind::Var(t.clone()), span);
            let mut body = Vec::with_capacity(sd.fields.len() + 1);
            for ((fname, _), fcell) in sd.fields.iter().zip(fields) {
                let val = build_value(&cell_value(fcell), span, ctx)?;
                body.push(e(ExprKind::Store {
                    ptr: Box::new(e(ExprKind::Field { ptr: Box::new(var()), field: fname.clone() })),
                    val: Box::new(val),
                }));
            }
            body.push(e(ExprKind::Load(Box::new(var()))));
            Ok(e(ExprKind::Let {
                binds: vec![(
                    t.clone(),
                    false,
                    e(ExprKind::Alloc { storage: Storage::Stack, ty: Type::Struct(name.clone()) }),
                )],
                body,
            }))
        }
        CtVal::Array { elem, cells } => {
            // `(let [t (alloc-stack (array elem N))] (store! (index t i) v)… (load t))`
            let n = ctx.gensym.get();
            ctx.gensym.set(n + 1);
            let t = format!("$ct{n}");
            let var = || Expr::new(ExprKind::Var(t.clone()), span);
            let arr_ty = Type::Array(Box::new(elem.clone()), cells.len() as u32);
            let mut body = Vec::with_capacity(cells.len() + 1);
            for (i, cell) in cells.iter().enumerate() {
                let val = build_value(&cell_value(cell), span, ctx)?;
                body.push(e(ExprKind::Store {
                    ptr: Box::new(e(ExprKind::Index {
                        ptr: Box::new(var()),
                        idx: Box::new(e(ExprKind::Int(i as i64))),
                    })),
                    val: Box::new(val),
                }));
            }
            body.push(e(ExprKind::Load(Box::new(var()))));
            Ok(e(ExprKind::Let {
                binds: vec![(
                    t.clone(),
                    false,
                    e(ExprKind::Alloc { storage: Storage::Stack, ty: arr_ty }),
                )],
                body,
            }))
        }
        CtVal::Ref(cell) => build_content(&cell.borrow(), span, ctx),
    }
}

// ---- the interpreter ---------------------------------------------------------

type Env = HashMap<String, CtVal>;

fn new_cell(c: CtVal) -> Cell {
    Rc::new(RefCell::new(c))
}

fn is_aggregate(c: &CtVal) -> bool {
    matches!(c, CtVal::Struct { .. } | CtVal::Array { .. } | CtVal::Sum { .. })
}

/// Deep-copy cell *content* (scalars copy; aggregates get fresh cells), giving
/// value semantics where the language copies.
fn deep_clone(c: &CtVal) -> CtVal {
    match c {
        CtVal::Struct { name, fields } => CtVal::Struct {
            name: name.clone(),
            fields: fields.iter().map(|f| new_cell(deep_clone(&f.borrow()))).collect(),
        },
        CtVal::Array { elem, cells } => CtVal::Array {
            elem: elem.clone(),
            cells: cells.iter().map(|e| new_cell(deep_clone(&e.borrow()))).collect(),
        },
        CtVal::Sum { variant, fields } => CtVal::Sum {
            variant: variant.clone(),
            fields: fields.iter().map(|f| new_cell(deep_clone(&f.borrow()))).collect(),
        },
        scalar => scalar.clone(),
    }
}

/// Turn an evaluated value into cell *content*: a reference's pointee is copied in
/// (so the cell never holds a `Ref`).
fn to_content(v: CtVal) -> CtVal {
    match v {
        CtVal::Ref(cell) => deep_clone(&cell.borrow()),
        other => other,
    }
}

/// Read a cell as a value: a scalar is returned directly; an aggregate is returned
/// as a reference to the cell (the aggregate-value representation).
fn cell_value(cell: &Cell) -> CtVal {
    if is_aggregate(&cell.borrow()) {
        CtVal::Ref(cell.clone())
    } else {
        cell.borrow().clone()
    }
}

/// The zero *content* of a type (`Int(0)`, a struct of zeroed fields, …).
fn zeroed_content(ty: &Type, ctx: &Ctx) -> Result<CtVal, String> {
    Ok(match ty {
        Type::Int(..) => CtVal::Int(0),
        Type::Bool => CtVal::Bool(false),
        Type::Float(_) => CtVal::Float(0.0),
        // a zeroed `(slice u8)` string (the empty string); usually overwritten.
        Type::Slice(el) if matches!(**el, Type::Int(8, _)) => CtVal::Str(String::new()),
        Type::Struct(n) => {
            let sd = ctx.structs.get(n).ok_or_else(|| format!("comptime: unknown struct '{n}'"))?;
            let mut fields = Vec::with_capacity(sd.fields.len());
            for (_, ft) in &sd.fields {
                fields.push(new_cell(zeroed_content(ft, ctx)?));
            }
            CtVal::Struct { name: n.clone(), fields }
        }
        Type::Array(el, k) => {
            let mut cells = Vec::with_capacity(*k as usize);
            for _ in 0..*k {
                cells.push(new_cell(zeroed_content(el, ctx)?));
            }
            CtVal::Array { elem: (**el).clone(), cells }
        }
        other => return Err(format!("comptime: `zeroed` of {other:?} isn't supported yet")),
    })
}

/// The zero *value* of a type (aggregates become a fresh `Ref`).
fn zeroed_value(ty: &Type, ctx: &Ctx) -> Result<CtVal, String> {
    let c = zeroed_content(ty, ctx)?;
    Ok(if is_aggregate(&c) { CtVal::Ref(new_cell(c)) } else { c })
}

/// Field index of `f` in struct `name`.
fn field_index(name: &str, f: &str, ctx: &Ctx) -> Result<usize, String> {
    let sd = ctx.structs.get(name).ok_or_else(|| format!("comptime: unknown struct '{name}'"))?;
    sd.fields
        .iter()
        .position(|(fname, _)| fname == f)
        .ok_or_else(|| format!("comptime: struct '{name}' has no field '{f}'"))
}

fn eval(e: &Expr, env: &mut Env, ctx: &Ctx) -> Result<Flow, String> {
    let f = ctx.fuel.get();
    if f == 0 {
        return Err("comptime: evaluation budget exhausted (infinite loop/recursion?)".to_string());
    }
    ctx.fuel.set(f - 1);
    let v = match &e.kind {
        ExprKind::Int(n) => CtVal::Int(*n),
        ExprKind::Bool(b) => CtVal::Bool(*b),
        ExprKind::Float(x) => CtVal::Float(*x),
        ExprKind::Comptime(inner) => return eval(inner, env, ctx),
        ExprKind::Str(s) => CtVal::Str(s.clone()),
        ExprKind::TypeQuery { q, ty } => type_query(*q, ty, ctx)?,
        ExprKind::FieldMeta { meta, ty, idx } => {
            let i = match val!(eval(idx, env, ctx)?) {
                CtVal::Int(n) => n,
                _ => return Err("comptime: field index must be an integer".to_string()),
            };
            field_meta(*meta, ty, i, ctx)?
        }
        ExprKind::FieldIndex { ty, name } => {
            let want = match val!(eval(name, env, ctx)?) {
                CtVal::Str(s) => s,
                _ => return Err("comptime: field-index name must be a string".to_string()),
            };
            find_field_index(ty, &want, ctx)?
        }
        ExprKind::Quote(s) => CtVal::Code((**s).clone()),
        ExprKind::Quasi(q) => CtVal::Code(eval_quasi(q, env, ctx)?),
        ExprKind::CodeOp { op, args } => {
            let mut argv = Vec::with_capacity(args.len());
            for a in args {
                argv.push(val!(eval(a, env, ctx)?));
            }
            code_op(*op, &argv)?
        }
        ExprKind::Var(name) => env
            .get(name)
            .cloned()
            .ok_or_else(|| format!("comptime: unbound variable '{name}'"))?,
        ExprKind::Not(x) => match val!(eval(x, env, ctx)?) {
            CtVal::Int(n) => CtVal::Int(!n),
            _ => return Err("comptime: `inot` needs an integer".to_string()),
        },
        ExprKind::Bin { op, lhs, rhs } => {
            let a = val!(eval(lhs, env, ctx)?);
            let b = val!(eval(rhs, env, ctx)?);
            bin(*op, a, b)?
        }
        ExprKind::Cmp { op, lhs, rhs } => {
            let a = val!(eval(lhs, env, ctx)?);
            let b = val!(eval(rhs, env, ctx)?);
            cmp(*op, a, b)?
        }
        ExprKind::If { cond, then, els } => {
            if truthy(&val!(eval(cond, env, ctx)?))? {
                return eval(then, env, ctx);
            } else {
                return eval(els, env, ctx);
            }
        }
        ExprKind::Do(es) => return eval_seq(es, env, ctx),
        ExprKind::Let { binds, body } => {
            let mut scope = env.clone();
            for (n, mutable, ve) in binds {
                let v = val!(eval(ve, &mut scope, ctx)?);
                let bound = if *mutable {
                    // a mutable place. An aggregate already IS a place (Ref); a
                    // scalar gets a fresh cell so it can be written through.
                    match v {
                        CtVal::Ref(_) => v,
                        scalar => CtVal::Ref(new_cell(scalar)),
                    }
                } else {
                    v
                };
                scope.insert(n.clone(), bound);
            }
            return eval_seq(body, &mut scope, ctx);
        }
        ExprKind::Cast { ty, expr } => match (val!(eval(expr, env, ctx)?), ty) {
            (CtVal::Int(n), Type::Int(bits, signed)) => CtVal::Int(trunc(n, *bits, *signed)),
            (other, _) => other,
        },
        ExprKind::Zeroed(ty) => zeroed_value(ty, ctx)?,
        ExprKind::Alloc { ty, .. } => {
            let content = zeroed_content(ty, ctx)?;
            CtVal::Ref(new_cell(content))
        }
        ExprKind::Borrow { place, .. } => place_ref(place, env, ctx)?,
        ExprKind::SpillRef(inner) => {
            let v = val!(eval(inner, env, ctx)?);
            match v {
                CtVal::Ref(_) => v, // already a place
                scalar => CtVal::Ref(new_cell(scalar)),
            }
        }
        ExprKind::Load(ptr) => {
            let r = val!(eval(ptr, env, ctx)?);
            match r {
                CtVal::Ref(cell) => cell_value(&cell),
                other => other, // load of a non-pointer value: identity
            }
        }
        ExprKind::Store { ptr, val } => {
            let r = val!(eval(ptr, env, ctx)?);
            let v = val!(eval(val, env, ctx)?);
            match r {
                CtVal::Ref(cell) => {
                    *cell.borrow_mut() = to_content(v.clone());
                    v
                }
                _ => return Err("comptime: `store!` target isn't a place".to_string()),
            }
        }
        ExprKind::Field { ptr, field } => place_ref(&Expr::new(ExprKind::Field { ptr: ptr.clone(), field: field.clone() }, e.span), env, ctx)?,
        ExprKind::Index { ptr, idx } => place_ref(&Expr::new(ExprKind::Index { ptr: ptr.clone(), idx: idx.clone() }, e.span), env, ctx)?,
        ExprKind::Construct { variant, args, .. } => {
            let mut fields = Vec::with_capacity(args.len());
            for a in args {
                let v = val!(eval(a, env, ctx)?);
                fields.push(new_cell(to_content(v)));
            }
            CtVal::Ref(new_cell(CtVal::Sum { variant: variant.clone(), fields }))
        }
        ExprKind::Match { scrut, arms } => {
            let s = val!(eval(scrut, env, ctx)?);
            let cell = match s {
                CtVal::Ref(c) => c,
                _ => return Err("comptime: `match` scrutinee must be a sum value".to_string()),
            };
            let (variant, fields) = match &*cell.borrow() {
                CtVal::Sum { variant, fields } => (variant.clone(), fields.clone()),
                _ => return Err("comptime: `match` scrutinee must be a sum value".to_string()),
            };
            let arm = arms
                .iter()
                .find(|a| a.variant == variant)
                .ok_or_else(|| format!("comptime: no match arm for variant '{variant}'"))?;
            let mut scope = env.clone();
            for (binder, fcell) in arm.binds.iter().zip(fields.iter()) {
                scope.insert(binder.clone(), cell_value(fcell));
            }
            return eval(&arm.body, &mut scope, ctx);
        }
        ExprKind::Call { func, type_args, args } => {
            let mut argv = Vec::with_capacity(args.len());
            for a in args {
                argv.push(val!(eval(a, env, ctx)?));
            }
            if ctx.variants.contains(func) {
                let fields = argv.into_iter().map(|v| new_cell(to_content(v))).collect();
                CtVal::Ref(new_cell(CtVal::Sum { variant: func.clone(), fields }))
            } else {
                if !type_args.is_empty() {
                    return Err(format!("comptime: generic call to '{func}' isn't supported yet"));
                }
                let callee = ctx.fns.get(func).ok_or_else(|| {
                    format!("comptime: call to '{func}' (not a comptime-evaluable function — extern/FFI?)")
                })?;
                if callee.params.len() != argv.len() {
                    return Err(format!("comptime: '{func}' arity mismatch"));
                }
                let mut scope: Env = HashMap::new();
                for (p, v) in callee.params.iter().zip(argv) {
                    scope.insert(p.name.clone(), v);
                }
                return eval_seq(&callee.body, &mut scope, ctx);
            }
        }
        ExprKind::Loop { label, body } => {
            loop {
                let mut broke: Option<Option<CtVal>> = None;
                for e in body {
                    match eval(e, env, ctx)? {
                        Flow::Val(_) => {}
                        Flow::Cont(l) if l.is_none() || l == *label => break,
                        Flow::Cont(l) => return Ok(Flow::Cont(l)),
                        Flow::Brk(v, l) if l.is_none() || l == *label => {
                            broke = Some(v);
                            break;
                        }
                        Flow::Brk(v, l) => return Ok(Flow::Brk(v, l)),
                    }
                }
                if let Some(v) = broke {
                    return Ok(Flow::Val(v.unwrap_or(CtVal::Int(0))));
                }
                // fell off the end / continue → iterate again (fuel bounds it)
                let f = ctx.fuel.get();
                if f == 0 {
                    return Err("comptime: loop budget exhausted".to_string());
                }
                ctx.fuel.set(f - 1);
            }
        }
        ExprKind::Break { label, value } => {
            let v = match value {
                Some(ve) => Some(val!(eval(ve, env, ctx)?)),
                None => None,
            };
            return Ok(Flow::Brk(v, label.clone()));
        }
        ExprKind::Continue { label } => return Ok(Flow::Cont(label.clone())),
        other => return Err(format!("comptime: {} isn't supported yet", kind_name(other))),
    };
    Ok(Flow::Val(v))
}

/// Evaluate an expression that denotes a *place*, yielding a `Ref` to its cell.
fn place_ref(e: &Expr, env: &mut Env, ctx: &Ctx) -> Result<CtVal, String> {
    match &e.kind {
        ExprKind::Field { ptr, field } => {
            let base = match eval(ptr, env, ctx)? {
                Flow::Val(v) => v,
                _ => return Err("comptime: control flow in a place".to_string()),
            };
            let cell = match base {
                CtVal::Ref(c) => c,
                _ => return Err("comptime: `field` base isn't a place".to_string()),
            };
            let (name, fcell) = {
                let b = cell.borrow();
                match &*b {
                    CtVal::Struct { name, fields } => (name.clone(), fields.clone()),
                    _ => return Err("comptime: `field` of a non-struct".to_string()),
                }
            };
            let idx = field_index(&name, field, ctx)?;
            Ok(CtVal::Ref(fcell[idx].clone()))
        }
        ExprKind::Index { ptr, idx } => {
            let base = match eval(ptr, env, ctx)? {
                Flow::Val(v) => v,
                _ => return Err("comptime: control flow in a place".to_string()),
            };
            let i = match eval(idx, env, ctx)? {
                Flow::Val(CtVal::Int(n)) => n as usize,
                _ => return Err("comptime: array index must be an integer".to_string()),
            };
            let cell = match base {
                CtVal::Ref(c) => c,
                _ => return Err("comptime: `index` base isn't a place".to_string()),
            };
            let elems = match &*cell.borrow() {
                CtVal::Array { cells, .. } => cells.clone(),
                _ => return Err("comptime: `index` of a non-array".to_string()),
            };
            let el = elems.get(i).ok_or_else(|| format!("comptime: array index {i} out of bounds"))?;
            Ok(CtVal::Ref(el.clone()))
        }
        // A bare place (a variable bound to a Ref, etc.) is already a reference.
        _ => match eval(e, env, ctx)? {
            Flow::Val(v @ CtVal::Ref(_)) => Ok(v),
            Flow::Val(scalar) => Ok(CtVal::Ref(new_cell(scalar))),
            _ => Err("comptime: control flow where a place was expected".to_string()),
        },
    }
}

fn eval_seq(es: &[Expr], env: &mut Env, ctx: &Ctx) -> Result<Flow, String> {
    let mut last = Flow::Val(CtVal::Int(0));
    for e in es {
        match eval(e, env, ctx)? {
            Flow::Val(v) => last = Flow::Val(v),
            other => return Ok(other),
        }
    }
    Ok(last)
}

/// Evaluate a compile-time type-reflection query against the type tables.
fn type_query(q: TypeQuery, ty: &Type, ctx: &Ctx) -> Result<CtVal, String> {
    let name = match ty {
        Type::Struct(n) | Type::App(n, _) => Some(n.as_str()),
        _ => None,
    };
    Ok(match q {
        TypeQuery::FieldCount => {
            let sd = name
                .and_then(|n| ctx.structs.get(n))
                .ok_or_else(|| format!("field-count: {ty:?} is not a struct"))?;
            CtVal::Int(sd.fields.len() as i64)
        }
        TypeQuery::VariantCount => {
            let sd = name
                .and_then(|n| ctx.sums.get(n))
                .ok_or_else(|| format!("variant-count: {ty:?} is not a sum"))?;
            CtVal::Int(sd.variants.len() as i64)
        }
        TypeQuery::IsStruct => CtVal::Bool(name.is_some_and(|n| ctx.structs.contains_key(n))),
        TypeQuery::IsSum => CtVal::Bool(name.is_some_and(|n| ctx.sums.contains_key(n))),
        TypeQuery::IsInt => CtVal::Bool(matches!(ty, Type::Int(..))),
        TypeQuery::IsFloat => CtVal::Bool(matches!(ty, Type::Float(_))),
        TypeQuery::IsPtr => CtVal::Bool(matches!(ty, Type::Ptr(_) | Type::Ref(..))),
        TypeQuery::IsArray => CtVal::Bool(matches!(ty, Type::Array(..))),
    })
}

/// Per-field reflection: the i-th field's name (a string) or its type's kind tag.
fn field_meta(meta: FieldMeta, ty: &Type, i: i64, ctx: &Ctx) -> Result<CtVal, String> {
    let name = match ty {
        Type::Struct(n) | Type::App(n, _) => n.as_str(),
        _ => return Err(format!("field reflection: {ty:?} is not a struct")),
    };
    let sd = ctx.structs.get(name).ok_or_else(|| format!("field reflection: {name} is not a struct"))?;
    let (fname, fty) = sd
        .fields
        .get(i as usize)
        .ok_or_else(|| format!("field reflection: index {i} out of range for {name} ({} fields)", sd.fields.len()))?;
    Ok(match meta {
        FieldMeta::Name => CtVal::Str(fname.clone()),
        FieldMeta::TypeKind => CtVal::Int(type_kind_tag(fty, ctx)),
        FieldMeta::TypeName => CtVal::Str(type_name_str(fty)),
    })
}

/// Find the index of the field named `want` in struct `ty` (for `field-index`).
fn find_field_index(ty: &Type, want: &str, ctx: &Ctx) -> Result<CtVal, String> {
    let name = match ty {
        Type::Struct(n) | Type::App(n, _) => n.as_str(),
        _ => return Err(format!("field-index: {ty:?} is not a struct")),
    };
    let sd = ctx.structs.get(name).ok_or_else(|| format!("field-index: {name} is not a struct"))?;
    match sd.fields.iter().position(|(fname, _)| fname == want) {
        Some(i) => Ok(CtVal::Int(i as i64)),
        None => Err(format!("field-index: struct {name} has no field '{want}'")),
    }
}

/// Build the `Sexp` of a quasiquote template, splicing unquote holes.
fn eval_quasi(q: &Quasi, env: &mut Env, ctx: &Ctx) -> Result<crate::reader::Sexp, String> {
    use crate::reader::{Sexp, SexpKind};
    use crate::span::Span;
    match q {
        Quasi::Lit(s) => Ok(s.clone()),
        Quasi::Unquote(e) => match eval(e, env, ctx)? {
            Flow::Val(v) => ctval_to_sexp(&v),
            _ => Err("comptime: control flow inside an unquote".to_string()),
        },
        Quasi::List(items) => {
            let mut out = Vec::with_capacity(items.len());
            for it in items {
                out.push(eval_quasi(it, env, ctx)?);
            }
            Ok(Sexp::new(SexpKind::List(out), Span::DUMMY))
        }
        Quasi::Vector(items) => {
            let mut out = Vec::with_capacity(items.len());
            for it in items {
                out.push(eval_quasi(it, env, ctx)?);
            }
            Ok(Sexp::new(SexpKind::Vector(out), Span::DUMMY))
        }
    }
}

/// Convert a comptime value to spliceable syntax (for an unquote).
fn ctval_to_sexp(v: &CtVal) -> Result<crate::reader::Sexp, String> {
    use crate::reader::{Sexp, SexpKind};
    use crate::span::Span;
    let s = |k| Sexp::new(k, Span::DUMMY);
    match v {
        CtVal::Code(c) => Ok(c.clone()),
        CtVal::Int(n) => Ok(s(SexpKind::Int(*n))),
        CtVal::Float(x) => Ok(s(SexpKind::Float(*x))),
        CtVal::Bool(b) => Ok(s(SexpKind::Sym(if *b { "true" } else { "false" }.to_string()))),
        CtVal::Str(st) => Ok(s(SexpKind::Str(st.clone()))),
        _ => Err("comptime: can't unquote an aggregate value into code".to_string()),
    }
}

/// Evaluate a `Code` operation. `op`'s first argument is the `Code` value.
fn code_op(op: CodeOp, args: &[CtVal]) -> Result<CtVal, String> {
    use crate::reader::SexpKind as K;
    let code = match args.first() {
        Some(CtVal::Code(s)) => s,
        _ => return Err("comptime: code op expects a Code value".to_string()),
    };
    Ok(match op {
        CodeOp::IsList => CtVal::Bool(matches!(code.kind, K::List(_))),
        CodeOp::IsSym => CtVal::Bool(matches!(code.kind, K::Sym(_))),
        CodeOp::IsInt => CtVal::Bool(matches!(code.kind, K::Int(_))),
        CodeOp::Count => match &code.kind {
            K::List(items) => CtVal::Int(items.len() as i64),
            _ => return Err("comptime: code-count expects a list".to_string()),
        },
        CodeOp::Nth => {
            let i = match args.get(1) {
                Some(CtVal::Int(n)) => *n as usize,
                _ => return Err("comptime: code-nth index must be an integer".to_string()),
            };
            match &code.kind {
                K::List(items) => match items.get(i) {
                    Some(s) => CtVal::Code(s.clone()),
                    None => return Err(format!("comptime: code-nth index {i} out of range")),
                },
                _ => return Err("comptime: code-nth expects a list".to_string()),
            }
        }
        CodeOp::Sym => match &code.kind {
            K::Sym(s) => CtVal::Str(s.clone()),
            _ => return Err("comptime: code-sym expects a symbol".to_string()),
        },
        CodeOp::Int => match &code.kind {
            K::Int(n) => CtVal::Int(*n),
            _ => return Err("comptime: code-int expects an integer".to_string()),
        },
    })
}

/// A type's display name (module prefix stripped): `i64`, `Point`, `ptr`, …
fn type_name_str(ty: &Type) -> String {
    let short = |n: &str| n.rsplit('.').next().unwrap_or(n).to_string();
    match ty {
        Type::Int(b, true) => format!("i{b}"),
        Type::Int(b, false) => format!("u{b}"),
        Type::Float(w) => format!("f{w}"),
        Type::Bool => "bool".to_string(),
        Type::Struct(n) | Type::App(n, _) => short(n),
        Type::Ptr(_) | Type::Ref(..) => "ptr".to_string(),
        Type::Array(..) => "array".to_string(),
        Type::Slice(_) => "slice".to_string(),
        other => format!("{other:?}"),
    }
}

/// The kind tag of a type (see `ast::FieldMeta::TypeKind` for the scheme).
fn type_kind_tag(ty: &Type, ctx: &Ctx) -> i64 {
    match ty {
        Type::Int(..) => 0,
        Type::Float(_) => 1,
        Type::Bool => 2,
        Type::Struct(n) | Type::App(n, _) => {
            if ctx.structs.contains_key(n) {
                3
            } else if ctx.sums.contains_key(n) {
                4
            } else {
                8
            }
        }
        Type::Ptr(_) | Type::Ref(..) => 5,
        Type::Array(..) => 6,
        Type::Slice(_) => 7,
        _ => 8,
    }
}

fn truthy(v: &CtVal) -> Result<bool, String> {
    match v {
        CtVal::Bool(b) => Ok(*b),
        CtVal::Int(n) => Ok(*n != 0),
        _ => Err("comptime: condition must be a bool or integer".to_string()),
    }
}

fn bin(op: BinOp, a: CtVal, b: CtVal) -> Result<CtVal, String> {
    if let (CtVal::Float(x), CtVal::Float(y)) = (&a, &b) {
        return match op {
            BinOp::Add => Ok(CtVal::Float(x + y)),
            BinOp::Sub => Ok(CtVal::Float(x - y)),
            BinOp::Mul => Ok(CtVal::Float(x * y)),
            BinOp::Div => Ok(CtVal::Float(x / y)),
            _ => Err("comptime: unsupported float operation".to_string()),
        };
    }
    let (x, y) = match (a, b) {
        (CtVal::Int(x), CtVal::Int(y)) => (x, y),
        _ => return Err("comptime: arithmetic needs two integers (or two floats)".to_string()),
    };
    let r = match op {
        BinOp::Add => x.wrapping_add(y),
        BinOp::Sub => x.wrapping_sub(y),
        BinOp::Mul => x.wrapping_mul(y),
        BinOp::Div => {
            if y == 0 { return Err("comptime: division by zero".to_string()); }
            x.wrapping_div(y)
        }
        BinOp::Rem => {
            if y == 0 { return Err("comptime: remainder by zero".to_string()); }
            x.wrapping_rem(y)
        }
        BinOp::UDiv => {
            if y == 0 { return Err("comptime: division by zero".to_string()); }
            ((x as u64) / (y as u64)) as i64
        }
        BinOp::URem => {
            if y == 0 { return Err("comptime: remainder by zero".to_string()); }
            ((x as u64) % (y as u64)) as i64
        }
        BinOp::And => x & y,
        BinOp::Or => x | y,
        BinOp::Xor => x ^ y,
        BinOp::Shl => x.wrapping_shl(y as u32),
        BinOp::Shr => x.wrapping_shr(y as u32),
    };
    Ok(CtVal::Int(r))
}

fn cmp(op: CmpOp, a: CtVal, b: CtVal) -> Result<CtVal, String> {
    use std::cmp::Ordering::*;
    let ord = match (&a, &b) {
        (CtVal::Int(x), CtVal::Int(y)) => x.cmp(y),
        (CtVal::Float(x), CtVal::Float(y)) => x.partial_cmp(y).unwrap_or(Less),
        (CtVal::Bool(x), CtVal::Bool(y)) => x.cmp(y),
        _ => return Err("comptime: comparison needs two values of the same scalar kind".to_string()),
    };
    let r = match op {
        CmpOp::Lt => ord == Less,
        CmpOp::Le => ord != Greater,
        CmpOp::Gt => ord == Greater,
        CmpOp::Ge => ord != Less,
        CmpOp::Eq => ord == Equal,
        CmpOp::Ne => ord != Equal,
    };
    Ok(CtVal::Bool(r))
}

fn trunc(n: i64, bits: u32, signed: bool) -> i64 {
    if bits >= 64 {
        return n;
    }
    let mask = (1i128 << bits) - 1;
    let masked = (n as i128) & mask;
    if signed && (masked & (1i128 << (bits - 1))) != 0 {
        (masked - (1i128 << bits)) as i64
    } else {
        masked as i64
    }
}

fn kind_name(k: &ExprKind) -> String {
    match k {
        ExprKind::Str(_) | ExprKind::CStr(_) => "a string (comptime strings aren't supported yet)".to_string(),
        ExprKind::SizeOf(_) | ExprKind::AlignOf(_) | ExprKind::OffsetOf(_, _) => "sizeof/alignof/offsetof".to_string(),
        ExprKind::FnPtrOf(_) | ExprKind::CallPtr { .. } => "a function pointer".to_string(),
        ExprKind::Free(_) => "free".to_string(),
        ExprKind::LlvmIr { .. } => "raw llvm-ir".to_string(),
        ExprKind::BitGet { .. } | ExprKind::BitSet { .. } => "a bitfield op".to_string(),
        ExprKind::TraitCall { .. } => "an unresolved trait call".to_string(),
        ExprKind::StaticRef(_) => "a reference to an aggregate const (a static global)".to_string(),
        _ => "this expression".to_string(),
    }
}
