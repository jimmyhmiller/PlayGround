//! Canonical, lossless, deterministic dump of a parsed `Program` — the
//! differential-oracle target for the self-hosted parser (`coil dump-ast`).
//!
//! Discipline mirrors `reader::dump_canonical` / `lib::dump_read`: numbers are
//! explicit, strings use a fixed per-byte escape, spans render as `@lo:hi`
//! (`@D:D` for a dummy), and floats are dumped as raw IEEE bits — so *formatting*
//! can never be the axis along which two faithful parsers diverge. Every field of
//! every AST node is emitted; nothing is omitted, so the gate sees everything.
//!
//! This module is ADDITIVE (like `dump-read`): it does not change any existing
//! compiler behavior.

use std::fmt::Write;

use crate::ast::*;
use crate::convention::{Convention, Lowering, NativeCc};
use crate::reader::Sexp;
use crate::span::Span;

/// Dump a whole program canonically. Sections appear in a fixed order; the only
/// HashMap (`conventions`) is sorted by name for determinism.
pub fn dump_program(p: &Program) -> String {
    let mut o = String::new();
    // conventions — sorted by name (the one unordered field).
    let mut names: Vec<&String> = p.conventions.keys().collect();
    names.sort();
    o.push_str("(conventions");
    for n in names {
        o.push(' ');
        dump_convention(&p.conventions[n], &mut o);
    }
    o.push(')');

    o.push_str("\n(structs");
    for s in &p.structs {
        o.push(' ');
        dump_struct(s, &mut o);
    }
    o.push(')');

    o.push_str("\n(sums");
    for s in &p.sums {
        o.push(' ');
        dump_sum(s, &mut o);
    }
    o.push(')');

    o.push_str("\n(externs");
    for e in &p.externs {
        o.push(' ');
        dump_extern(e, &mut o);
    }
    o.push(')');

    o.push_str("\n(funcs");
    for f in &p.funcs {
        o.push(' ');
        dump_func(f, &mut o);
    }
    o.push(')');

    o.push_str("\n(asserts");
    for a in &p.asserts {
        o.push(' ');
        dump_assert(a, &mut o);
    }
    o.push(')');

    o.push_str("\n(consts");
    for c in &p.consts {
        o.push(' ');
        dump_const(c, &mut o);
    }
    o.push(')');

    o.push_str("\n(traits");
    for t in &p.traits {
        o.push(' ');
        dump_trait(t, &mut o);
    }
    o.push(')');

    o.push_str("\n(impls");
    for im in &p.impls {
        o.push(' ');
        dump_impl(im, &mut o);
    }
    o.push(')');

    o.push_str("\n(statics");
    for s in &p.statics {
        o.push(' ');
        dump_static(s, &mut o);
    }
    o.push(')');

    o.push_str("\n(metas");
    for m in &p.metas {
        o.push(' ');
        dump_expr(m, &mut o);
    }
    o.push(')');

    o.push_str("\n(exports");
    for e in &p.exports {
        o.push(' ');
        dump_export(e, &mut o);
    }
    o.push(')');

    o
}

// ---- primitive encoders ---------------------------------------------------

/// Per-byte canonical escape — identical to `reader::esc`.
fn esc(text: &str, out: &mut String) {
    for &b in text.as_bytes() {
        match b {
            b'\\' => out.push_str("\\\\"),
            b'"' => out.push_str("\\\""),
            b'\n' => out.push_str("\\n"),
            b'\t' => out.push_str("\\t"),
            b'\r' => out.push_str("\\r"),
            0x20..=0x7e => out.push(b as char),
            _ => write!(out, "\\x{b:02x}").unwrap(),
        }
    }
}

fn dstr(s: &str, o: &mut String) {
    o.push('"');
    esc(s, o);
    o.push('"');
}

fn dump_span(span: Span, o: &mut String) {
    if span.lo == u32::MAX {
        o.push_str("@D:");
    } else {
        write!(o, "@{}:", span.lo).unwrap();
    }
    if span.hi == u32::MAX {
        o.push('D');
    } else {
        write!(o, "{}", span.hi).unwrap();
    }
}

fn dbool(b: bool, o: &mut String) {
    o.push(if b { '1' } else { '0' });
}

/// `["a" "b"]` (a vector of strings).
fn dstrs(v: &[String], o: &mut String) {
    o.push('[');
    for (i, s) in v.iter().enumerate() {
        if i > 0 {
            o.push(' ');
        }
        dstr(s, o);
    }
    o.push(']');
}

/// `[T1 T2]` (a vector of types).
fn dtypes(v: &[Type], o: &mut String) {
    o.push('[');
    for (i, t) in v.iter().enumerate() {
        if i > 0 {
            o.push(' ');
        }
        dump_type(t, o);
    }
    o.push(']');
}

/// `[E1 E2]` (a vector of exprs).
fn dexprs(v: &[Expr], o: &mut String) {
    o.push('[');
    for (i, e) in v.iter().enumerate() {
        if i > 0 {
            o.push(' ');
        }
        dump_expr(e, o);
    }
    o.push(']');
}

/// `(some "x")` / `(none)` for an optional string.
fn dopt_str(v: &Option<String>, o: &mut String) {
    match v {
        Some(s) => {
            o.push_str("(some ");
            dstr(s, o);
            o.push(')');
        }
        None => o.push_str("(none)"),
    }
}

// ---- types ----------------------------------------------------------------

fn dump_type(t: &Type, o: &mut String) {
    match t {
        Type::Int(b, s) => {
            write!(o, "(int {b} ").unwrap();
            dbool(*s, o);
            o.push(')');
        }
        Type::Float(b) => write!(o, "(float {b})").unwrap(),
        Type::Bool => o.push_str("bool"),
        Type::Void => o.push_str("void"),
        Type::Ptr(t) => {
            o.push_str("(ptr ");
            dump_type(t, o);
            o.push(')');
        }
        Type::Ref(m, t) => {
            o.push_str("(ref ");
            dbool(*m, o);
            o.push(' ');
            dump_type(t, o);
            o.push(')');
        }
        Type::Struct(n) => {
            o.push_str("(struct ");
            dstr(n, o);
            o.push(')');
        }
        Type::Array(t, n) => {
            o.push_str("(array ");
            dump_type(t, o);
            write!(o, " {n})").unwrap();
        }
        Type::Slice(t) => {
            o.push_str("(slice ");
            dump_type(t, o);
            o.push(')');
        }
        Type::Vec(t, n) => {
            o.push_str("(vec ");
            dump_type(t, o);
            write!(o, " {n})").unwrap();
        }
        Type::Fn(cc, ps, ret) => {
            o.push_str("(fn ");
            dstr(cc, o);
            o.push(' ');
            dtypes(ps, o);
            o.push(' ');
            dump_type(ret, o);
            o.push(')');
        }
        Type::App(n, args) => {
            o.push_str("(app ");
            dstr(n, o);
            o.push(' ');
            dtypes(args, o);
            o.push(')');
        }
        Type::Never => o.push_str("never"),
        Type::Code => o.push_str("code"),
    }
}

// ---- enum tags ------------------------------------------------------------

fn binop_tag(op: &BinOp) -> &'static str {
    match op {
        BinOp::Add => "add",
        BinOp::Sub => "sub",
        BinOp::Mul => "mul",
        BinOp::Div => "div",
        BinOp::Rem => "rem",
        BinOp::UDiv => "udiv",
        BinOp::URem => "urem",
        BinOp::And => "and",
        BinOp::Or => "or",
        BinOp::Xor => "xor",
        BinOp::Shl => "shl",
        BinOp::Shr => "shr",
    }
}

fn cmpop_tag(op: &CmpOp) -> &'static str {
    match op {
        CmpOp::Lt => "lt",
        CmpOp::Le => "le",
        CmpOp::Gt => "gt",
        CmpOp::Ge => "ge",
        CmpOp::Eq => "eq",
        CmpOp::Ne => "ne",
    }
}

fn storage_tag(s: &Storage) -> &'static str {
    match s {
        Storage::Stack => "stack",
        Storage::Static => "static",
        Storage::Heap => "heap",
    }
}

fn typequery_tag(q: &TypeQuery) -> &'static str {
    match q {
        TypeQuery::FieldCount => "fieldcount",
        TypeQuery::VariantCount => "variantcount",
        TypeQuery::IsStruct => "isstruct",
        TypeQuery::IsSum => "issum",
        TypeQuery::IsInt => "isint",
        TypeQuery::IsFloat => "isfloat",
        TypeQuery::IsPtr => "isptr",
        TypeQuery::IsArray => "isarray",
    }
}

fn fieldmeta_tag(m: &FieldMeta) -> &'static str {
    match m {
        FieldMeta::Name => "name",
        FieldMeta::TypeKind => "typekind",
        FieldMeta::TypeName => "typename",
    }
}

fn codeop_tag(op: &CodeOp) -> &'static str {
    match op {
        CodeOp::IsList => "islist",
        CodeOp::IsSym => "issym",
        CodeOp::IsInt => "isint",
        CodeOp::Count => "count",
        CodeOp::Nth => "nth",
        CodeOp::Sym => "sym",
        CodeOp::Int => "int",
        CodeOp::Rest => "rest",
        CodeOp::Gensym => "gensym",
        CodeOp::Error => "error",
        CodeOp::Eq => "eq",
        CodeOp::IsKeyword => "iskeyword",
        CodeOp::Symbol => "symbol",
        CodeOp::Str => "str",
        CodeOp::CFieldCount => "cfieldcount",
        CodeOp::CFieldName => "cfieldname",
        CodeOp::CFieldKind => "cfieldkind",
        CodeOp::CFieldType => "cfieldtype",
        CodeOp::CVariantSum => "cvariantsum",
        CodeOp::CVariantCount => "cvariantcount",
        CodeOp::CVariantName => "cvariantname",
        CodeOp::CVariantFields => "cvariantfields",
        CodeOp::CTraitMethodCount => "ctraitmethodcount",
        CodeOp::CTraitMethodName => "ctraitmethodname",
        CodeOp::CTraitArity => "ctraitarity",
        CodeOp::CTraitParamName => "ctraitparamname",
        CodeOp::CTraitParamType => "ctraitparamtype",
        CodeOp::CTraitRetType => "ctraitrettype",
        CodeOp::StrBytes => "strbytes",
        CodeOp::BytesToStr => "bytestostr",
        CodeOp::TargetArch => "targetarch",
    }
}

// ---- expressions ----------------------------------------------------------

fn dump_expr(e: &Expr, o: &mut String) {
    o.push('(');
    match &e.kind {
        ExprKind::Int(n) => {
            o.push_str("eint");
            dump_span(e.span, o);
            write!(o, " {n}").unwrap();
        }
        ExprKind::Float(x) => {
            o.push_str("efloat");
            dump_span(e.span, o);
            write!(o, " 0x{:016x}", x.to_bits()).unwrap();
        }
        ExprKind::Bool(b) => {
            o.push_str("ebool");
            dump_span(e.span, o);
            o.push(' ');
            dbool(*b, o);
        }
        ExprKind::Str(s) => {
            o.push_str("estr");
            dump_span(e.span, o);
            o.push(' ');
            dstr(s, o);
        }
        ExprKind::CStr(s) => {
            o.push_str("ecstr");
            dump_span(e.span, o);
            o.push(' ');
            dstr(s, o);
        }
        ExprKind::Var(s) => {
            o.push_str("var");
            dump_span(e.span, o);
            o.push(' ');
            dstr(s, o);
        }
        ExprKind::Zeroed(t) => {
            o.push_str("zeroed");
            dump_span(e.span, o);
            o.push(' ');
            dump_type(t, o);
        }
        ExprKind::Borrow { mutable, place } => {
            o.push_str("borrow");
            dump_span(e.span, o);
            o.push(' ');
            dbool(*mutable, o);
            o.push(' ');
            dump_expr(place, o);
        }
        ExprKind::SpillRef(inner) => {
            o.push_str("spillref");
            dump_span(e.span, o);
            o.push(' ');
            dump_expr(inner, o);
        }
        ExprKind::Let { binds, body } => {
            o.push_str("let");
            dump_span(e.span, o);
            o.push_str(" [");
            for (i, (n, m, v)) in binds.iter().enumerate() {
                if i > 0 {
                    o.push(' ');
                }
                o.push_str("(bind ");
                dstr(n, o);
                o.push(' ');
                dbool(*m, o);
                o.push(' ');
                dump_expr(v, o);
                o.push(')');
            }
            o.push(']');
            o.push(' ');
            dexprs(body, o);
        }
        ExprKind::Bin { op, lhs, rhs } => {
            o.push_str("bin");
            dump_span(e.span, o);
            write!(o, " {} ", binop_tag(op)).unwrap();
            dump_expr(lhs, o);
            o.push(' ');
            dump_expr(rhs, o);
        }
        ExprKind::Not(inner) => {
            o.push_str("inot");
            dump_span(e.span, o);
            o.push(' ');
            dump_expr(inner, o);
        }
        ExprKind::Cmp { op, lhs, rhs } => {
            o.push_str("cmp");
            dump_span(e.span, o);
            write!(o, " {} ", cmpop_tag(op)).unwrap();
            dump_expr(lhs, o);
            o.push(' ');
            dump_expr(rhs, o);
        }
        ExprKind::If { cond, then, els } => {
            o.push_str("if");
            dump_span(e.span, o);
            o.push(' ');
            dump_expr(cond, o);
            o.push(' ');
            dump_expr(then, o);
            o.push(' ');
            dump_expr(els, o);
        }
        ExprKind::Do(body) => {
            o.push_str("do");
            dump_span(e.span, o);
            o.push(' ');
            dexprs(body, o);
        }
        ExprKind::Call { func, type_args, args } => {
            o.push_str("call");
            dump_span(e.span, o);
            o.push(' ');
            dstr(func, o);
            o.push(' ');
            dtypes(type_args, o);
            o.push(' ');
            dexprs(args, o);
        }
        ExprKind::Alloc { storage, ty } => {
            o.push_str("alloc");
            dump_span(e.span, o);
            write!(o, " {} ", storage_tag(storage)).unwrap();
            dump_type(ty, o);
        }
        ExprKind::Field { ptr, field } => {
            o.push_str("field");
            dump_span(e.span, o);
            o.push(' ');
            dump_expr(ptr, o);
            o.push(' ');
            dstr(field, o);
        }
        ExprKind::Load(inner) => {
            o.push_str("load");
            dump_span(e.span, o);
            o.push(' ');
            dump_expr(inner, o);
        }
        ExprKind::Store { ptr, val } => {
            o.push_str("store");
            dump_span(e.span, o);
            o.push(' ');
            dump_expr(ptr, o);
            o.push(' ');
            dump_expr(val, o);
        }
        ExprKind::Index { ptr, idx } => {
            o.push_str("index");
            dump_span(e.span, o);
            o.push(' ');
            dump_expr(ptr, o);
            o.push(' ');
            dump_expr(idx, o);
        }
        ExprKind::Cast { ty, expr } => {
            o.push_str("cast");
            dump_span(e.span, o);
            o.push(' ');
            dump_type(ty, o);
            o.push(' ');
            dump_expr(expr, o);
        }
        ExprKind::SizeOf(t) => {
            o.push_str("sizeof");
            dump_span(e.span, o);
            o.push(' ');
            dump_type(t, o);
        }
        ExprKind::AlignOf(t) => {
            o.push_str("alignof");
            dump_span(e.span, o);
            o.push(' ');
            dump_type(t, o);
        }
        ExprKind::OffsetOf(t, field) => {
            o.push_str("offsetof");
            dump_span(e.span, o);
            o.push(' ');
            dump_type(t, o);
            o.push(' ');
            dstr(field, o);
        }
        ExprKind::BitGet { ptr, field } => {
            o.push_str("bitget");
            dump_span(e.span, o);
            o.push(' ');
            dump_expr(ptr, o);
            o.push(' ');
            dstr(field, o);
        }
        ExprKind::BitSet { ptr, field, val } => {
            o.push_str("bitset");
            dump_span(e.span, o);
            o.push(' ');
            dump_expr(ptr, o);
            o.push(' ');
            dstr(field, o);
            o.push(' ');
            dump_expr(val, o);
        }
        ExprKind::Free(inner) => {
            o.push_str("free");
            dump_span(e.span, o);
            o.push(' ');
            dump_expr(inner, o);
        }
        ExprKind::Erase { trait_name, inner } => {
            o.push_str("erase");
            dump_span(e.span, o);
            o.push(' ');
            dstr(trait_name, o);
            o.push(' ');
            dump_expr(inner, o);
        }
        ExprKind::DynDispatch { dyn_struct, vtable_struct, method_index, cc, params, ret, recv, args } => {
            o.push_str("dyndispatch");
            dump_span(e.span, o);
            o.push(' ');
            dstr(dyn_struct, o);
            o.push(' ');
            dstr(vtable_struct, o);
            write!(o, " {method_index} ").unwrap();
            dstr(cc, o);
            o.push(' ');
            dtypes(params, o);
            o.push(' ');
            dump_type(ret, o);
            o.push(' ');
            dump_expr(recv, o);
            o.push(' ');
            dexprs(args, o);
        }
        ExprKind::MakeDyn { dyn_struct, vtable_struct, methods, inner } => {
            o.push_str("makedyn");
            dump_span(e.span, o);
            o.push(' ');
            dstr(dyn_struct, o);
            o.push(' ');
            dstr(vtable_struct, o);
            o.push(' ');
            dstrs(methods, o);
            o.push(' ');
            dump_expr(inner, o);
        }
        ExprKind::Construct { sum, variant, args } => {
            o.push_str("construct");
            dump_span(e.span, o);
            o.push(' ');
            dstr(sum, o);
            o.push(' ');
            dstr(variant, o);
            o.push(' ');
            dexprs(args, o);
        }
        ExprKind::Match { scrut, arms } => {
            o.push_str("match");
            dump_span(e.span, o);
            o.push(' ');
            dump_expr(scrut, o);
            o.push_str(" [");
            for (i, a) in arms.iter().enumerate() {
                if i > 0 {
                    o.push(' ');
                }
                o.push_str("(arm ");
                dstr(&a.variant, o);
                o.push(' ');
                dstrs(&a.binds, o);
                o.push(' ');
                dump_expr(&a.body, o);
                o.push(')');
            }
            o.push(']');
        }
        ExprKind::FnPtrOf(s) => {
            o.push_str("fnptrof");
            dump_span(e.span, o);
            o.push(' ');
            dstr(s, o);
        }
        ExprKind::CallPtr { fp, args } => {
            o.push_str("callptr");
            dump_span(e.span, o);
            o.push(' ');
            dump_expr(fp, o);
            o.push(' ');
            dexprs(args, o);
        }
        ExprKind::Loop { label, body } => {
            o.push_str("loop");
            dump_span(e.span, o);
            o.push(' ');
            dopt_str(label, o);
            o.push(' ');
            dexprs(body, o);
        }
        ExprKind::Break { label, value } => {
            o.push_str("break");
            dump_span(e.span, o);
            o.push(' ');
            dopt_str(label, o);
            o.push(' ');
            match value {
                Some(v) => {
                    o.push_str("(some ");
                    dump_expr(v, o);
                    o.push(')');
                }
                None => o.push_str("(none)"),
            }
        }
        ExprKind::Continue { label } => {
            o.push_str("continue");
            dump_span(e.span, o);
            o.push(' ');
            dopt_str(label, o);
        }
        ExprKind::LlvmIr { result, args, body } => {
            o.push_str("llvmir");
            dump_span(e.span, o);
            o.push(' ');
            dump_type(result, o);
            o.push(' ');
            dexprs(args, o);
            o.push(' ');
            dstr(body, o);
        }
        ExprKind::StaticRef(s) => {
            o.push_str("staticref");
            dump_span(e.span, o);
            o.push(' ');
            dstr(s, o);
        }
        ExprKind::TypeQuery { q, ty } => {
            o.push_str("typequery");
            dump_span(e.span, o);
            write!(o, " {} ", typequery_tag(q)).unwrap();
            dump_type(ty, o);
        }
        ExprKind::FieldMeta { meta, ty, idx } => {
            o.push_str("fieldmeta");
            dump_span(e.span, o);
            write!(o, " {} ", fieldmeta_tag(meta)).unwrap();
            dump_type(ty, o);
            o.push(' ');
            dump_expr(idx, o);
        }
        ExprKind::FieldIndex { ty, name } => {
            o.push_str("fieldindex");
            dump_span(e.span, o);
            o.push(' ');
            dump_type(ty, o);
            o.push(' ');
            dump_expr(name, o);
        }
        ExprKind::Quote(s) => {
            o.push_str("quote");
            dump_span(e.span, o);
            o.push(' ');
            dump_sexp(s, o);
        }
        ExprKind::CodeOp { op, args } => {
            o.push_str("codeop");
            dump_span(e.span, o);
            write!(o, " {} ", codeop_tag(op)).unwrap();
            dexprs(args, o);
        }
        ExprKind::Quasi(q) => {
            o.push_str("quasi");
            dump_span(e.span, o);
            o.push(' ');
            dump_quasi(q, o);
        }
        ExprKind::Comptime(inner) => {
            o.push_str("comptime");
            dump_span(e.span, o);
            o.push(' ');
            dump_expr(inner, o);
        }
        ExprKind::TraitCall { trait_name, method, self_tp, args } => {
            o.push_str("traitcall");
            dump_span(e.span, o);
            o.push(' ');
            dstr(trait_name, o);
            o.push(' ');
            dstr(method, o);
            o.push(' ');
            dstr(self_tp, o);
            o.push(' ');
            dexprs(args, o);
        }
    }
    o.push(')');
}

// ---- quasiquote templates + quoted sexp -----------------------------------

/// A quoted `Sexp` — reuse the reader's canonical node dump so quote/quasiquote
/// literals are encoded exactly as `dump-read` would.
fn dump_sexp(s: &Sexp, o: &mut String) {
    o.push_str(&crate::reader::dump_canonical(std::slice::from_ref(s)));
}

fn dump_quasi(q: &Quasi, o: &mut String) {
    match q {
        Quasi::Lit(s) => {
            o.push_str("(qlit ");
            dump_sexp(s, o);
            o.push(')');
        }
        Quasi::Unquote(e) => {
            o.push_str("(qunq ");
            dump_expr(e, o);
            o.push(')');
        }
        Quasi::Splice(e) => {
            o.push_str("(qspl ");
            dump_expr(e, o);
            o.push(')');
        }
        Quasi::List(items) => {
            o.push_str("(qlist");
            for it in items {
                o.push(' ');
                dump_quasi(it, o);
            }
            o.push(')');
        }
        Quasi::Vector(items) => {
            o.push_str("(qvec");
            for it in items {
                o.push(' ');
                dump_quasi(it, o);
            }
            o.push(')');
        }
    }
}

// ---- top-level declarations -----------------------------------------------

fn dump_param(p: &Param, o: &mut String) {
    o.push_str("(param ");
    dstr(&p.name, o);
    o.push(' ');
    dump_type(&p.ty, o);
    o.push(')');
}

fn dump_field(name: &str, ty: &Type, o: &mut String) {
    o.push_str("(fld ");
    dstr(name, o);
    o.push(' ');
    dump_type(ty, o);
    o.push(')');
}

fn dump_bounds(bounds: &[(String, Vec<String>)], o: &mut String) {
    o.push('[');
    for (i, (n, traits)) in bounds.iter().enumerate() {
        if i > 0 {
            o.push(' ');
        }
        o.push_str("(bound ");
        dstr(n, o);
        o.push(' ');
        dstrs(traits, o);
        o.push(')');
    }
    o.push(']');
}

fn dump_func(f: &Func, o: &mut String) {
    o.push_str("(func ");
    dstr(&f.name, o);
    o.push(' ');
    dstrs(&f.type_params, o);
    o.push(' ');
    dump_bounds(&f.bounds, o);
    o.push(' ');
    dstr(&f.cc, o);
    o.push_str(" [");
    for (i, p) in f.params.iter().enumerate() {
        if i > 0 {
            o.push(' ');
        }
        dump_param(p, o);
    }
    o.push(']');
    o.push(' ');
    dump_type(&f.ret, o);
    o.push(' ');
    dexprs(&f.body, o);
    o.push(' ');
    dbool(f.macro_variadic, o);
    o.push(' ');
    dump_span(f.span, o);
    o.push(')');
}

fn dump_convention(c: &Convention, o: &mut String) {
    o.push_str("(cc ");
    dstr(&c.name, o);
    o.push(' ');
    dstrs(&c.params, o);
    o.push(' ');
    dopt_str(&c.ret, o);
    o.push(' ');
    dstrs(&c.clobber, o);
    o.push(' ');
    dstrs(&c.preserve, o);
    o.push(' ');
    match &c.lowering {
        Lowering::Native(cc) => {
            let n = match cc {
                NativeCc::C => "c",
                NativeCc::Fast => "fast",
                NativeCc::Cold => "cold",
            };
            write!(o, "(native {n})").unwrap();
        }
        Lowering::Shim => o.push_str("(shim)"),
    }
    o.push(')');
}

fn dump_struct(s: &StructDef, o: &mut String) {
    o.push_str("(struct ");
    dstr(&s.name, o);
    o.push(' ');
    dstrs(&s.type_params, o);
    o.push(' ');
    dump_layout(&s.layout, o);
    o.push_str(" [");
    for (i, (n, t)) in s.fields.iter().enumerate() {
        if i > 0 {
            o.push(' ');
        }
        dump_field(n, t, o);
    }
    o.push(']');
    o.push(')');
}

fn dump_layout(l: &Layout, o: &mut String) {
    match l {
        Layout::C => o.push_str("(c)"),
        Layout::Packed => o.push_str("(packed)"),
        Layout::Aligned(n) => write!(o, "(aligned {n})").unwrap(),
        Layout::Explicit(e) => {
            o.push_str("(explicit [");
            for (i, off) in e.offsets.iter().enumerate() {
                if i > 0 {
                    o.push(' ');
                }
                write!(o, "{off}").unwrap();
            }
            o.push(']');
            o.push(' ');
            match e.size {
                Some(n) => write!(o, "(some {n})").unwrap(),
                None => o.push_str("(none)"),
            }
            write!(o, " {})", e.align).unwrap();
        }
        Layout::Bits(b) => {
            write!(o, "(bits {} [", b.backing).unwrap();
            for (i, off) in b.offsets.iter().enumerate() {
                if i > 0 {
                    o.push(' ');
                }
                write!(o, "{off}").unwrap();
            }
            o.push_str("])");
        }
    }
}

fn dump_sum(s: &SumDef, o: &mut String) {
    o.push_str("(sum ");
    dstr(&s.name, o);
    o.push(' ');
    dstrs(&s.type_params, o);
    o.push_str(" [");
    for (i, v) in s.variants.iter().enumerate() {
        if i > 0 {
            o.push(' ');
        }
        o.push_str("(variant ");
        dstr(&v.name, o);
        o.push_str(" [");
        for (j, (n, t)) in v.fields.iter().enumerate() {
            if j > 0 {
                o.push(' ');
            }
            dump_field(n, t, o);
        }
        o.push(']');
        o.push(')');
    }
    o.push(']');
    o.push(')');
}

fn dump_extern(e: &Extern, o: &mut String) {
    o.push_str("(extern ");
    dstr(&e.name, o);
    o.push(' ');
    dstr(&e.cc, o);
    o.push(' ');
    dtypes(&e.params, o);
    o.push(' ');
    dbool(e.variadic, o);
    o.push(' ');
    dump_type(&e.ret, o);
    o.push(')');
}

fn dump_assert(a: &StaticAssert, o: &mut String) {
    o.push_str("(assert ");
    dump_expr(&a.cond, o);
    o.push(' ');
    dstr(&a.msg, o);
    o.push(')');
}

fn dump_const(c: &Const, o: &mut String) {
    o.push_str("(const ");
    dstr(&c.name, o);
    o.push(' ');
    match &c.ty {
        Some(t) => {
            o.push_str("(some ");
            dump_type(t, o);
            o.push(')');
        }
        None => o.push_str("(none)"),
    }
    o.push(' ');
    dump_expr(&c.value, o);
    o.push(')');
}

fn dump_trait(t: &TraitDef, o: &mut String) {
    o.push_str("(trait ");
    dstr(&t.name, o);
    o.push(' ');
    dstr(&t.self_param, o);
    o.push_str(" [");
    for (i, m) in t.methods.iter().enumerate() {
        if i > 0 {
            o.push(' ');
        }
        o.push_str("(method ");
        dstr(&m.name, o);
        o.push_str(" [");
        for (j, p) in m.params.iter().enumerate() {
            if j > 0 {
                o.push(' ');
            }
            dump_param(p, o);
        }
        o.push(']');
        o.push(' ');
        dump_type(&m.ret, o);
        o.push(')');
    }
    o.push(']');
    o.push(')');
}

fn dump_impl(im: &ImplDef, o: &mut String) {
    o.push_str("(impl ");
    dstr(&im.trait_name, o);
    o.push(' ');
    dstr(&im.for_type, o);
    o.push_str(" [");
    for (i, f) in im.methods.iter().enumerate() {
        if i > 0 {
            o.push(' ');
        }
        dump_func(f, o);
    }
    o.push(']');
    o.push(')');
}

fn dump_static(s: &StaticDef, o: &mut String) {
    o.push_str("(static ");
    dstr(&s.name, o);
    o.push(' ');
    dump_type(&s.ty, o);
    o.push(' ');
    dump_constinit(&s.init, o);
    o.push(')');
}

fn dump_constinit(c: &ConstInit, o: &mut String) {
    match c {
        ConstInit::Int(n) => write!(o, "(ci-int {n})").unwrap(),
        ConstInit::Float(x) => write!(o, "(ci-float 0x{:016x})", x.to_bits()).unwrap(),
        ConstInit::Bool(b) => {
            o.push_str("(ci-bool ");
            dbool(*b, o);
            o.push(')');
        }
        ConstInit::Str(s) => {
            o.push_str("(ci-str ");
            dstr(s, o);
            o.push(')');
        }
        ConstInit::Array(items) => {
            o.push_str("(ci-array");
            for it in items {
                o.push(' ');
                dump_constinit(it, o);
            }
            o.push(')');
        }
        ConstInit::Struct(items) => {
            o.push_str("(ci-struct");
            for it in items {
                o.push(' ');
                dump_constinit(it, o);
            }
            o.push(')');
        }
    }
}

fn dump_export(e: &ExportC, o: &mut String) {
    o.push_str("(export ");
    dstr(&e.name, o);
    o.push(' ');
    dopt_str(&e.symbol, o);
    o.push(' ');
    dtypes(&e.params, o);
    o.push(' ');
    dump_span(e.span, o);
    o.push(')');
}
