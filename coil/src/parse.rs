//! Surface s-expressions -> core AST.
//!
//! Diagnostics: every form carries a `Span` (from the reader). The recursive
//! entry points (`parse_expr`, `parse_type`, and the top-level loop) attach the
//! span of the form they're parsing to any error that bubbles up without one, so
//! a parse error points at the most specific offending sub-form. The small
//! matcher helpers (`as_list`, `sym`, …) attach the span of the node they were
//! handed directly.

use std::collections::HashMap;

use crate::ast::*;
use crate::convention::{Convention, Lowering, NativeCc};
use crate::reader::{Sexp, SexpKind};
use crate::span::{Diag, Span};

pub fn parse_program(forms: &[Sexp]) -> Result<Program, Diag> {
    let mut conventions = HashMap::new();
    conventions.insert("c".to_string(), Convention::default_c());
    let mut funcs = Vec::new();
    let mut externs = Vec::new();
    let mut structs = Vec::new();
    let mut sums = Vec::new();
    let mut asserts = Vec::new();
    let mut consts = Vec::new();
    let mut traits = Vec::new();
    let mut impls = Vec::new();
    let mut metas = Vec::new();

    for form in forms {
        // Any error from a top-level form gets that form's span as a fallback
        // location (a deeper frame may already have set a more precise one).
        let r = (|| -> Result<(), Diag> {
            let items = as_list(form, "top-level form must be a list")?;
            let head = head_sym(items)?;
            match head.as_str() {
                "defcc" => {
                    let c = parse_defcc(&items[1..])?;
                    if conventions.insert(c.name.clone(), c.clone()).is_some() && c.name != "c" {
                        return Err(Diag::new(format!("convention '{}' defined twice", c.name)));
                    }
                }
                "defstruct" => structs.push(parse_defstruct(&items[1..])?),
                "defsum" => sums.push(parse_defsum(&items[1..])?),
                "defn" => {
                    let mut f = parse_defn(&items[1..])?;
                    f.span = form.span; // the (defn …) form's location, for DWARF
                    if funcs.iter().any(|g: &Func| g.name == f.name) {
                        return Err(Diag::new(format!("function '{}' defined twice", f.name)));
                    }
                    funcs.push(f);
                }
                "extern" => externs.push(parse_extern(&items[1..])?),
                "static-assert" => {
                    let cond = parse_expr(items.get(1).ok_or("static-assert: missing condition")?)?;
                    let msg = match items.get(2).map(|s| &s.kind) {
                        Some(SexpKind::Str(s)) => s.clone(),
                        None => "static assertion failed".to_string(),
                        _ => return Err(Diag::new("static-assert: message must be a string")),
                    };
                    asserts.push(StaticAssert { cond, msg });
                }
                "const" => consts.push(parse_const(&items[1..])?),
                "deftrait" => traits.push(parse_deftrait(&items[1..])?),
                "impl" => impls.push(parse_impl(&items[1..], form.span)?),
                "meta" => {
                    if items.len() != 2 {
                        return Err(Diag::new("meta: expected (meta EXPR)"));
                    }
                    metas.push(parse_expr(&items[1])?);
                }
                other => return Err(Diag::new(format!("unknown top-level form '{other}'"))),
            }
            Ok(())
        })();
        r.map_err(|d| d.with_span(form.span))?;
    }

    Ok(Program {
        conventions,
        structs,
        sums,
        externs,
        funcs,
        asserts,
        consts,
        traits,
        impls,
        statics: vec![], // produced by checking
        metas,
    })
}

/// `(deftrait Name [Self] (method [(a Self) …] (-> ret)) …)` — a trait: the
/// implementing-type parameter `Self`, then one signature per method.
fn parse_deftrait(rest: &[Sexp]) -> Result<TraitDef, Diag> {
    let name = sym(rest.first().ok_or("deftrait: missing name")?, "trait name")?;
    let self_param = match rest.get(1).map(|s| &s.kind) {
        Some(SexpKind::Vector(v)) if v.len() == 1 => sym(&v[0], "Self parameter")?,
        _ => return Err(Diag::new("deftrait: expected [Self] after the name")),
    };
    let mut methods = Vec::new();
    for m in &rest[2..] {
        let ml = as_list(m, "trait method must be (name [params] (-> ret))")?;
        let mname = sym(ml.first().ok_or("trait method: missing name")?, "method name")?;
        let params_v = match ml.get(1).map(|s| &s.kind) {
            Some(SexpKind::Vector(v)) => v,
            _ => return Err(Diag::at(m.span, format!("trait method '{mname}': expected a param vector"))),
        };
        let mut params = Vec::new();
        for p in params_v {
            let pl = as_list(p, "method param must be (name :type)")?;
            params.push(Param {
                name: sym(&pl[0], "param name")?,
                ty: parse_type(pl.get(1).ok_or("method param missing type")?)?,
            });
        }
        let ret_l = as_list(ml.get(2).ok_or("trait method: missing (-> ret)")?, "expected (-> ret)")?;
        if head_sym(ret_l)? != "->" {
            return Err(Diag::at(m.span, format!("trait method '{mname}': expected (-> ret)")));
        }
        let ret = parse_type(ret_l.get(1).ok_or("(-> ) missing type")?)?;
        methods.push(TraitMethod { name: mname, params, ret });
    }
    Ok(TraitDef { name, self_param, methods })
}

/// `(impl Trait Type (method [params] (-> ret) body…) …)` — implement `Trait`
/// for `Type`. Each method is parsed as a `defn` (concrete `Type`/`Self` types in
/// its signature); naming/`Self`-checking happens in the checker.
fn parse_impl(rest: &[Sexp], span: Span) -> Result<ImplDef, Diag> {
    let trait_name = sym(rest.first().ok_or("impl: missing trait name")?, "trait name")?;
    let for_type = sym(rest.get(1).ok_or("impl: missing type name")?, "impl type")?;
    let mut methods = Vec::new();
    for m in &rest[2..] {
        let ml = as_list(m, "impl method must be (name [params] (-> ret) body…)")?;
        let mut f = parse_defn(ml)?; // (name [params] (-> ret) body…) parses like a defn body
        f.span = span;
        methods.push(f);
    }
    Ok(ImplDef { trait_name, for_type, methods })
}

/// `(const NAME VALUE)` or `(const NAME TYPE VALUE)` — a named scalar constant.
/// The value must be a literal (int / float / `true` / `false`), possibly with a
/// leading type to pin its width; anything else is refused (no const-expression
/// evaluation here — keep it minimal).
fn parse_const(rest: &[Sexp]) -> Result<Const, Diag> {
    let name = sym(rest.first().ok_or("const: missing name")?, "const name")?;
    let (ty, val_sexp) = match rest.len() {
        2 => (None, &rest[1]),
        3 => (Some(parse_type(&rest[1])?), &rest[2]),
        _ => {
            return Err(Diag::new(
                "const: expected (const NAME VALUE) or (const NAME TYPE VALUE)",
            ))
        }
    };
    // A const's value is any expression. A bare literal is inlined as before; a
    // richer expression is evaluated at compile time (the comptime interpreter).
    let value = parse_expr(val_sexp)?;
    Ok(Const { name, ty, value })
}

pub fn parse_defsum(rest: &[Sexp]) -> Result<SumDef, Diag> {
    let name = sym(rest.first().ok_or("defsum: missing name")?, "sum name")?;
    let mut i = 1;
    let mut type_params = vec![];
    if let Some(SexpKind::Vector(tp)) = rest.get(i).map(|s| &s.kind) {
        type_params = tp.iter().map(|s| sym(s, "type parameter")).collect::<Result<_, _>>()?;
        i += 1;
    }
    let mut variants = Vec::new();
    for v in &rest[i..] {
        let vl = as_list(v, "variant must be (Name [fields])")?;
        let vname = sym(vl.first().ok_or("variant: missing name")?, "variant name")?;
        let fields = match vl.get(1).map(|s| &s.kind) {
            Some(SexpKind::Vector(fv)) => {
                let mut fs = Vec::new();
                for f in fv {
                    let fl = as_list(f, "field must be (name :type)")?;
                    fs.push((sym(&fl[0], "field name")?, parse_type(fl.get(1).ok_or("field type")?)?));
                }
                fs
            }
            None => vec![],
            _ => return Err(Diag::at(v.span, format!("variant '{vname}': fields must be a vector"))),
        };
        variants.push(SumVariant { name: vname, fields });
    }
    Ok(SumDef {
        name,
        type_params,
        variants,
    })
}

pub fn parse_defstruct(rest: &[Sexp]) -> Result<StructDef, Diag> {
    let name = sym(rest.first().ok_or("defstruct: missing name")?, "struct name")?;
    let mut i = 1;
    // struct-level options: :layout c|packed|explicit|(align N), :size N, :align N.
    let mut packed = false;
    let mut explicit = false;
    let mut bits = false;
    let mut aligned: Option<u32> = None;
    let mut exp_size: Option<u64> = None;
    let mut exp_align: u32 = 0;
    let mut backing: Option<u32> = None;
    while let Some(SexpKind::Keyword(k)) = rest.get(i).map(|s| &s.kind) {
        let val = rest.get(i + 1).ok_or_else(|| format!("defstruct: ':{k}' missing value"))?;
        match k.as_str() {
            "layout" => match &val.kind {
                SexpKind::Sym(s) if s == "c" => {}
                SexpKind::Sym(s) if s == "packed" => packed = true,
                SexpKind::Sym(s) if s == "explicit" => explicit = true,
                SexpKind::Sym(s) if s == "bits" => bits = true,
                SexpKind::List(items) if head_sym(items).ok().as_deref() == Some("align") => {
                    aligned = Some(pow2(items.get(1))?);
                }
                _ => return Err(Diag::at(val.span, "layout must be c, packed, explicit, bits, or (align N)")),
            },
            "size" => exp_size = Some(int_val(val, "size")? as u64),
            "align" => exp_align = pow2(Some(val))?,
            "backing" => match parse_type(val)? {
                Type::Int(b, _) => backing = Some(b),
                _ => return Err(Diag::at(val.span, ":backing must be an integer type")),
            },
            _ => break,
        }
        i += 2;
    }

    // structs don't carry trait bounds (v1) — take the names, ignore any bounds.
    let (type_params, _bounds) = parse_type_params(rest, &mut i)?;
    let fields_v = match rest.get(i).map(|s| &s.kind) {
        Some(SexpKind::Vector(v)) => v,
        _ => return Err(Diag::new(format!("defstruct '{name}': expected a field vector"))),
    };

    if bits {
        let mut fields = Vec::new();
        let mut offsets = Vec::new();
        let mut acc: u32 = 0;
        for f in fields_v {
            let fl = as_list(f, "bitfield must be (name :bits N)")?;
            let fname = sym(&fl[0], "field name")?;
            let width = match (fl.get(1).map(|s| &s.kind), fl.get(2).map(|s| &s.kind)) {
                (Some(SexpKind::Keyword(k)), Some(SexpKind::Int(n))) if k == "bits" && *n > 0 => *n as u32,
                _ => return Err(Diag::at(f.span, format!("field '{fname}' must be (name :bits N)"))),
            };
            fields.push((fname, Type::Int(width, false))); // a bitfield's value type is uN
            offsets.push(acc);
            acc += width;
        }
        let backing = match backing {
            Some(b) if b >= acc => b,
            Some(b) => return Err(Diag::new(format!("defstruct '{name}': :backing i{b} too small for {acc} bits"))),
            None if acc <= 8 => 8,
            None if acc <= 16 => 16,
            None if acc <= 32 => 32,
            None if acc <= 64 => 64,
            None => return Err(Diag::new(format!("defstruct '{name}': {acc} bits needs an explicit :backing"))),
        };
        return Ok(StructDef {
            name,
            type_params,
            layout: Layout::Bits(BitsLayout { backing, offsets }),
            fields,
        });
    }

    let mut fields = Vec::new();
    let mut at_offsets: Vec<Option<u64>> = Vec::new();
    for f in fields_v {
        let fl = as_list(f, "field must be (name :type [:at N])")?;
        let fname = sym(&fl[0], "field name")?;
        let fty = parse_type(fl.get(1).ok_or("field missing type")?)?;
        // optional `:at N`
        let mut at = None;
        let mut j = 2;
        while j < fl.len() {
            match &fl[j].kind {
                SexpKind::Keyword(k) if k == "at" => {
                    at = Some(int_val(fl.get(j + 1).ok_or(":at missing value")?, "at")? as u64);
                    j += 2;
                }
                _ => return Err(Diag::at(fl[j].span, format!("unknown field option {}", fl[j]))),
            }
        }
        fields.push((fname, fty));
        at_offsets.push(at);
    }

    let layout = if let Some(n) = aligned {
        Layout::Aligned(n)
    } else if explicit || at_offsets.iter().any(Option::is_some) {
        let offsets = at_offsets
            .iter()
            .map(|o| o.ok_or_else(|| format!("defstruct '{name}': explicit layout needs :at on every field")))
            .collect::<Result<Vec<_>, _>>()?;
        Layout::Explicit(ExplicitLayout {
            offsets,
            size: exp_size,
            align: exp_align,
        })
    } else if packed {
        Layout::Packed
    } else {
        Layout::C
    };

    Ok(StructDef {
        name,
        type_params,
        layout,
        fields,
    })
}

fn int_val(s: &Sexp, what: &str) -> Result<i64, Diag> {
    match &s.kind {
        SexpKind::Int(n) => Ok(*n),
        _ => Err(Diag::at(s.span, format!("{what}: expected an integer, got {s}"))),
    }
}

fn pow2(s: Option<&Sexp>) -> Result<u32, Diag> {
    match s.map(|s| &s.kind) {
        Some(SexpKind::Int(n)) if *n > 0 && (*n as u64).is_power_of_two() => Ok(*n as u32),
        _ => Err(Diag::new("alignment must be a positive power of two")),
    }
}

/// If `rest[i]` is a vector immediately followed by another vector, the first is
/// a generic type-parameter list (bare symbols); consume it and advance `i`.
/// Parse the optional generic type-parameter vector. Each entry is either a bare
/// symbol `T` (unbounded) or a list `(T Trait1 Trait2 …)` (T bounded by those
/// traits). Returns the param names and the non-empty bounds.
fn parse_type_params(
    rest: &[Sexp],
    i: &mut usize,
) -> Result<(Vec<String>, Vec<(String, Vec<String>)>), Diag> {
    if let (Some(SexpKind::Vector(tp)), Some(SexpKind::Vector(_))) =
        (rest.get(*i).map(|s| &s.kind), rest.get(*i + 1).map(|s| &s.kind))
    {
        let mut params = Vec::new();
        let mut bounds = Vec::new();
        for entry in tp {
            match &entry.kind {
                SexpKind::Sym(name) => params.push(name.clone()),
                SexpKind::List(items) => {
                    let pname = sym(
                        items.first().ok_or("type parameter: empty () bound")?,
                        "type parameter",
                    )?;
                    let traits = items[1..]
                        .iter()
                        .map(|s| sym(s, "bound trait"))
                        .collect::<Result<Vec<_>, _>>()?;
                    if !traits.is_empty() {
                        bounds.push((pname.clone(), traits));
                    }
                    params.push(pname);
                }
                _ => return Err(Diag::at(entry.span, "type parameter must be a name or (name Trait…)")),
            }
        }
        *i += 1;
        Ok((params, bounds))
    } else {
        Ok((vec![], vec![]))
    }
}

/// Build a quasiquote template: `(unquote E)` becomes a hole carrying the parsed
/// expression `E`; everything else is literal syntax (recursing into lists/vectors).
fn build_quasi(s: &Sexp) -> Result<Quasi, Diag> {
    match &s.kind {
        SexpKind::List(items) => {
            if items.len() == 2 {
                if let SexpKind::Sym(h) = &items[0].kind {
                    if h == "unquote" {
                        return Ok(Quasi::Unquote(Box::new(parse_expr(&items[1])?)));
                    }
                    if h == "unquote-splicing" {
                        return Ok(Quasi::Splice(Box::new(parse_expr(&items[1])?)));
                    }
                }
            }
            Ok(Quasi::List(items.iter().map(build_quasi).collect::<Result<_, _>>()?))
        }
        SexpKind::Vector(items) => {
            Ok(Quasi::Vector(items.iter().map(build_quasi).collect::<Result<_, _>>()?))
        }
        _ => Ok(Quasi::Lit(s.clone())),
    }
}

fn parse_extern(rest: &[Sexp]) -> Result<Extern, Diag> {
    let mut i = 0;
    let name = sym(rest.get(i).ok_or("extern: missing name")?, "extern name")?;
    i += 1;

    let mut cc = "c".to_string();
    if let Some(SexpKind::Keyword(k)) = rest.get(i).map(|s| &s.kind) {
        if k == "cc" {
            cc = sym(
                rest.get(i + 1).ok_or("extern: ':cc' missing convention")?,
                "cc name",
            )?;
            i += 2;
        }
    }

    let params_v = match rest.get(i).map(|s| &s.kind) {
        Some(SexpKind::Vector(v)) => v,
        _ => return Err(Diag::new(format!("extern '{name}': expected a vector of parameter types"))),
    };
    i += 1;
    // A trailing `...` in the parameter vector marks a C variadic function.
    let variadic = matches!(params_v.last().map(|s| &s.kind), Some(SexpKind::Sym(s)) if s == "...");
    let fixed = if variadic { &params_v[..params_v.len() - 1] } else { &params_v[..] };
    let params = fixed.iter().map(parse_type).collect::<Result<_, _>>()?;

    let ret_l = as_list(
        rest.get(i).ok_or("extern: missing (-> :type)")?,
        "expected (-> :type)",
    )?;
    if head_sym(ret_l)? != "->" {
        return Err(Diag::new(format!("extern '{name}': expected (-> :type)")));
    }
    let ret = parse_type(ret_l.get(1).ok_or("(-> ) missing type")?)?;

    Ok(Extern {
        name,
        cc,
        params,
        variadic,
        ret,
    })
}

fn parse_defcc(rest: &[Sexp]) -> Result<Convention, Diag> {
    let name = sym(rest.first().ok_or("defcc: missing name")?, "defcc name")?;
    let mut params = vec![];
    let mut ret = None;
    let mut clobber = vec![];
    let mut preserve = vec![];
    let mut native: Option<NativeCc> = None;
    let mut shim = false;

    let mut i = 1;
    while i < rest.len() {
        let kw = keyword(&rest[i], "defcc option")?;
        let val = rest
            .get(i + 1)
            .ok_or_else(|| format!("defcc: option ':{kw}' missing value"))?;
        match kw.as_str() {
            "params" => params = sym_vec(val, "params")?,
            "ret" => ret = Some(sym(val, "ret")?),
            "clobber" => clobber = sym_vec(val, "clobber")?,
            "preserve" => preserve = sym_vec(val, "preserve")?,
            "native" => {
                let n = sym(val, "native")?;
                native = Some(
                    NativeCc::parse(&n)
                        .ok_or_else(|| Diag::at(val.span, format!("defcc: unknown native cc '{n}'")))?,
                );
            }
            "lower" => {
                // :lower shim  — declares an exotic convention (M2)
                if sym(val, "lower")? == "shim" {
                    shim = true;
                }
            }
            other => return Err(Diag::at(rest[i].span, format!("defcc: unknown option ':{other}'"))),
        }
        i += 2;
    }

    let lowering = match (native, shim) {
        (Some(cc), _) => Lowering::Native(cc),
        (None, true) => Lowering::Shim,
        (None, false) => {
            return Err(Diag::new(format!(
                "defcc '{name}': needs a lowering (:native <c|fast|cold> or :lower shim)"
            )))
        }
    };

    Ok(Convention {
        name,
        params,
        ret,
        clobber,
        preserve,
        lowering,
    })
}

fn parse_defn(rest: &[Sexp]) -> Result<Func, Diag> {
    let mut i = 0;
    let name = sym(rest.get(i).ok_or("defn: missing name")?, "defn name")?;
    i += 1;

    // optional `:cc <name>`
    let mut cc = "c".to_string();
    if let Some(SexpKind::Keyword(k)) = rest.get(i).map(|s| &s.kind) {
        if k == "cc" {
            cc = sym(
                rest.get(i + 1).ok_or("defn: ':cc' missing convention name")?,
                "cc name",
            )?;
            i += 2;
        }
    }

    // optional generic params: a vector immediately followed by the param vector.
    let (type_params, bounds) = parse_type_params(rest, &mut i)?;

    // params vector
    let params_v = match rest.get(i).map(|s| &s.kind) {
        Some(SexpKind::Vector(v)) => v,
        _ => return Err(Diag::new(format!("defn '{name}': expected parameter vector"))),
    };
    i += 1;
    let mut params = Vec::new();
    // `&` before the last param marks a variadic macro: that param soaks up all
    // remaining call args as one Code list.
    let mut macro_variadic = false;
    for p in params_v {
        if matches!(&p.kind, SexpKind::Sym(s) if s == "&") {
            macro_variadic = true;
            continue;
        }
        let pl = as_list(p, "parameter must be (name :type)")?;
        let pname = sym(&pl[0], "param name")?;
        let pty = parse_type(pl.get(1).ok_or("param missing type")?)?;
        params.push(Param { name: pname, ty: pty });
    }

    // return type `(-> :T)`
    let ret_l = as_list(
        rest.get(i).ok_or("defn: missing (-> :type)")?,
        "expected (-> :type)",
    )?;
    if head_sym(ret_l)? != "->" {
        return Err(Diag::new("defn: expected (-> :type) after params"));
    }
    let ret = parse_type(ret_l.get(1).ok_or("(-> ) missing type")?)?;
    i += 1;

    // body
    let body: Vec<Expr> = rest[i..]
        .iter()
        .map(parse_expr)
        .collect::<Result<_, _>>()?;
    if body.is_empty() {
        return Err(Diag::new(format!("defn '{name}': empty body")));
    }

    Ok(Func {
        name,
        type_params,
        bounds,
        cc,
        params,
        ret,
        body,
        macro_variadic,
        // The caller (parse_program) stamps the real `(defn …)` form span; a Func
        // built in isolation has none.
        span: Span::DUMMY,
    })
}

/// Parse a type, attaching the form's span to any spanless error from within.
fn parse_type(s: &Sexp) -> Result<Type, Diag> {
    parse_type_inner(s).map_err(|d| d.with_span(s.span))
}

fn parse_type_inner(s: &Sexp) -> Result<Type, Diag> {
    match &s.kind {
        // `:i32` (keyword) must be an int. A bare symbol is an int name, or
        // otherwise a struct name (so `(ptr c i8)` and `Point` both read well).
        SexpKind::Keyword(k) => prim_type(k).map_err(Diag::from),
        // `Code` is the comptime quoted-code type (Stage 3); otherwise a primitive
        // int name, else a struct name.
        SexpKind::Sym(k) if k == "Code" => Ok(Type::Code),
        SexpKind::Sym(k) => Ok(prim_type(k).unwrap_or_else(|_| Type::Struct(k.clone()))),
        SexpKind::List(items) => match head_sym(items)?.as_str() {
            // (mut TYPE) -> a mutable reference to TYPE.
            "mut" => {
                let pointee = parse_type(items.get(1).ok_or("mut: expects (mut TYPE)")?)?;
                Ok(Type::Ref(true, Box::new(pointee)))
            }
            // (ref TYPE) -> an immutable reference to TYPE. The dual of (mut T):
            // it spells the by-immutable-reference parameter that an aggregate
            // takes implicitly, and lets a non-aggregate param be taken by
            // reference too.
            "ref" => {
                let pointee = parse_type(items.get(1).ok_or("ref: expects (ref TYPE)")?)?;
                Ok(Type::Ref(false, Box::new(pointee)))
            }
            // (ptr TYPE) -> pointer to TYPE; (ptr) defaults the pointee to i64.
            "ptr" => {
                let pointee = match items.get(1) {
                    Some(t) => parse_type(t)?,
                    None => Type::Int(64, true),
                };
                Ok(Type::Ptr(Box::new(pointee)))
            }
            // (array TYPE N)
            "array" => {
                let elem = parse_type(items.get(1).ok_or("array type: missing element type")?)?;
                let n = match items.get(2).map(|s| &s.kind) {
                    Some(SexpKind::Int(n)) if *n > 0 => *n as u32,
                    _ => return Err(Diag::new("array type: expected a positive length")),
                };
                Ok(Type::Array(Box::new(elem), n))
            }
            // (slice TYPE) -> a fat-pointer view {ptr, len} over elements of TYPE.
            "slice" => {
                let elem = parse_type(items.get(1).ok_or("slice type: missing element type")?)?;
                Ok(Type::Slice(Box::new(elem)))
            }
            // (vec TYPE N) -> a SIMD vector <N x TYPE>; element must be a scalar.
            "vec" => {
                let elem = parse_type(items.get(1).ok_or("vec type: missing element type")?)?;
                let n = match items.get(2).map(|s| &s.kind) {
                    Some(SexpKind::Int(n)) if *n > 0 => *n as u32,
                    _ => return Err(Diag::new("vec type: expected a positive lane count")),
                };
                Ok(Type::Vec(Box::new(elem), n))
            }
            // (struct Name) — explicit form; bare `Name` also works.
            "struct" => {
                let n = sym(items.get(1).ok_or("struct type: missing name")?, "struct name")?;
                Ok(Type::Struct(n))
            }
            // (fnptr CC [param-types] ret-type)
            "fnptr" => {
                let cc = sym(items.get(1).ok_or("fnptr type: missing convention")?, "fnptr cc")?;
                let params_v = match items.get(2).map(|s| &s.kind) {
                    Some(SexpKind::Vector(v)) => v,
                    _ => return Err(Diag::new("fnptr type: expected a vector of parameter types")),
                };
                let params = params_v.iter().map(parse_type).collect::<Result<_, _>>()?;
                let ret = parse_type(items.get(3).ok_or("fnptr type: missing return type")?)?;
                Ok(Type::Fn(cc, params, Box::new(ret)))
            }
            // (Name TARG...) -> a generic type application, e.g. (Pair i64 i64).
            other => {
                let args = items[1..].iter().map(parse_type).collect::<Result<_, _>>()?;
                Ok(Type::App(other.to_string(), args))
            }
        },
        _ => Err(Diag::at(s.span, format!("unsupported type: {s}"))),
    }
}

fn alloc_form(args: &[Sexp], storage: Storage) -> Result<ExprKind, Diag> {
    let ty = match args.first() {
        Some(t) => parse_type(t)?,
        None => Type::Int(64, true),
    };
    Ok(ExprKind::Alloc { storage, ty })
}

/// A primitive scalar type name: `f32`/`f64`, or an integer `iN`/`uN`.
pub fn prim_type(name: &str) -> Result<Type, String> {
    match name {
        "f32" => Ok(Type::Float(32)),
        "f64" => Ok(Type::Float(64)),
        "bool" => Ok(Type::Bool),
        "void" => Ok(Type::Void),   // valid only as a return type (checked later)
        _ => int_type(name),
    }
}

/// Parse a Zig-style integer type name: `i<N>` (signed) or `u<N>` (unsigned),
/// for any width N in 1..=65535 (e.g. `i64`, `u2`, `i7`, `u23`).
fn int_type(name: &str) -> Result<Type, String> {
    let signed = match name.as_bytes().first() {
        Some(b'i') => true,
        Some(b'u') => false,
        _ => return Err(format!("not an int type '{name}'")),
    };
    let digits = &name[1..];
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return Err(format!("not an int type '{name}'"));
    }
    let bits: u32 = digits.parse().map_err(|_| format!("int width too large in '{name}'"))?;
    if bits == 0 || bits > 65535 {
        return Err(format!("int width out of range in '{name}' (1..=65535)"));
    }
    Ok(Type::Int(bits, signed))
}

/// Parse an expression, attaching the form's span to any spanless error from
/// within (so the diagnostic points at the most specific offending sub-form).
fn parse_expr(s: &Sexp) -> Result<Expr, Diag> {
    let kind = parse_expr_inner(s).map_err(|d| d.with_span(s.span))?;
    Ok(Expr::new(kind, s.span))
}

fn parse_expr_inner(s: &Sexp) -> Result<ExprKind, Diag> {
    match &s.kind {
        SexpKind::Int(n) => Ok(ExprKind::Int(*n)),
        SexpKind::Float(x) => Ok(ExprKind::Float(*x)),
        SexpKind::Sym(name) if name == "true" => Ok(ExprKind::Bool(true)),
        SexpKind::Sym(name) if name == "false" => Ok(ExprKind::Bool(false)),
        SexpKind::Sym(name) => Ok(ExprKind::Var(name.clone())),
        SexpKind::Keyword(k) => Err(Diag::at(s.span, format!("unexpected keyword :{k} in expression"))),
        SexpKind::Str(s) => Ok(ExprKind::Str(s.clone())),
        SexpKind::CStr(s) => Ok(ExprKind::CStr(s.clone())),
        SexpKind::Vector(_) => Err(Diag::at(s.span, "unexpected vector in expression")),
        SexpKind::List(items) => parse_list_expr(items, s.span),
    }
}

fn parse_list_expr(items: &[Sexp], span: Span) -> Result<ExprKind, Diag> {
    let head = head_sym(items)?;
    let args = &items[1..];
    let bin = |op: BinOp| -> Result<ExprKind, Diag> {
        let (l, r) = two(args, &head)?;
        Ok(ExprKind::Bin {
            op,
            lhs: Box::new(parse_expr(l)?),
            rhs: Box::new(parse_expr(r)?),
        })
    };
    let cmp = |op: CmpOp| -> Result<ExprKind, Diag> {
        let (l, r) = two(args, &head)?;
        Ok(ExprKind::Cmp {
            op,
            lhs: Box::new(parse_expr(l)?),
            rhs: Box::new(parse_expr(r)?),
        })
    };
    match head.as_str() {
        "iadd" => bin(BinOp::Add),
        "isub" => bin(BinOp::Sub),
        "imul" => bin(BinOp::Mul),
        "idiv" => bin(BinOp::Div),
        "irem" => bin(BinOp::Rem),
        "udiv" => bin(BinOp::UDiv),
        "urem" => bin(BinOp::URem),
        "iand" => bin(BinOp::And),
        "ior" => bin(BinOp::Or),
        "ixor" => bin(BinOp::Xor),
        "ishl" => bin(BinOp::Shl),
        "ishr" => bin(BinOp::Shr),
        // floating-point arithmetic & comparison (dispatched on operand type).
        "fadd" => bin(BinOp::Add),
        "fsub" => bin(BinOp::Sub),
        "fmul" => bin(BinOp::Mul),
        "fdiv" => bin(BinOp::Div),
        "frem" => bin(BinOp::Rem),
        "fcmp-lt" => cmp(CmpOp::Lt),
        "fcmp-le" => cmp(CmpOp::Le),
        "fcmp-gt" => cmp(CmpOp::Gt),
        "fcmp-ge" => cmp(CmpOp::Ge),
        "fcmp-eq" => cmp(CmpOp::Eq),
        "fcmp-ne" => cmp(CmpOp::Ne),
        "inot" => {
            if args.len() != 1 {
                return Err(Diag::at(span, "inot: expects (inot x)"));
            }
            Ok(ExprKind::Not(Box::new(parse_expr(&args[0])?)))
        }
        "icmp-lt" => cmp(CmpOp::Lt),
        "icmp-le" => cmp(CmpOp::Le),
        "icmp-gt" => cmp(CmpOp::Gt),
        "icmp-ge" => cmp(CmpOp::Ge),
        "icmp-eq" => cmp(CmpOp::Eq),
        "icmp-ne" => cmp(CmpOp::Ne),
        // short-circuiting logical operators desugar to `if` over booleans.
        "and" => {
            let (a, b) = two(args, "and")?;
            Ok(ExprKind::If {
                cond: Box::new(parse_expr(a)?),
                then: Box::new(parse_expr(b)?),
                els: Box::new(Expr::new(ExprKind::Bool(false), span)),
            })
        }
        "or" => {
            let (a, b) = two(args, "or")?;
            Ok(ExprKind::If {
                cond: Box::new(parse_expr(a)?),
                then: Box::new(Expr::new(ExprKind::Bool(true), span)),
                els: Box::new(parse_expr(b)?),
            })
        }
        "not" => {
            if args.len() != 1 {
                return Err(Diag::at(span, "not: expects (not x)"));
            }
            Ok(ExprKind::If {
                cond: Box::new(parse_expr(&args[0])?),
                then: Box::new(Expr::new(ExprKind::Bool(false), span)),
                els: Box::new(Expr::new(ExprKind::Bool(true), span)),
            })
        }
        "if" => {
            if args.len() != 3 {
                return Err(Diag::at(span, "if: expects (if cond then else)"));
            }
            Ok(ExprKind::If {
                cond: Box::new(parse_expr(&args[0])?),
                then: Box::new(parse_expr(&args[1])?),
                els: Box::new(parse_expr(&args[2])?),
            })
        }
        "do" => Ok(ExprKind::Do(
            args.iter().map(parse_expr).collect::<Result<_, _>>()?,
        )),
        "comptime" => {
            if args.len() != 1 {
                return Err(Diag::at(span, "comptime: expects (comptime expr)"));
            }
            Ok(ExprKind::Comptime(Box::new(parse_expr(&args[0])?)))
        }
        "let" => {
            let binds_v = match args.first().map(|s| &s.kind) {
                Some(SexpKind::Vector(v)) => v,
                _ => return Err(Diag::at(span, "let: expected binding vector")),
            };
            if binds_v.len() % 2 != 0 {
                return Err(Diag::at(span, "let: bindings must be name/value pairs"));
            }
            let mut binds = Vec::new();
            for pair in binds_v.chunks(2) {
                // A binding name is either `name` (immutable) or `(mut name)`
                // (a mutable stack place).
                let (name, mutable) = match &pair[0].kind {
                    SexpKind::Sym(s) => (s.clone(), false),
                    SexpKind::List(items)
                        if head_sym(items).ok().as_deref() == Some("mut") && items.len() == 2 =>
                    {
                        (sym(&items[1], "let binding name")?, true)
                    }
                    _ => {
                        return Err(Diag::at(
                            pair[0].span,
                            format!("let binding name must be `name` or `(mut name)`, got {}", pair[0]),
                        ))
                    }
                };
                binds.push((name, mutable, parse_expr(&pair[1])?));
            }
            let body: Vec<Expr> = args[1..]
                .iter()
                .map(parse_expr)
                .collect::<Result<_, _>>()?;
            if body.is_empty() {
                return Err(Diag::at(span, "let: empty body"));
            }
            Ok(ExprKind::Let { binds, body })
        }
        "call" => {
            let f = sym(args.first().ok_or("call: missing function")?, "call target")?;
            let cargs = args[1..]
                .iter()
                .map(parse_expr)
                .collect::<Result<_, _>>()?;
            Ok(ExprKind::Call {
                func: f,
                type_args: vec![],
                args: cargs,
            })
        }
        // `(zeroed T)` — the zero value of T, for initializing a fresh local.
        "zeroed" => {
            if args.len() != 1 {
                return Err(Diag::at(span, "zeroed: expects (zeroed TYPE)"));
            }
            Ok(ExprKind::Zeroed(parse_type(&args[0])?))
        }
        // `(mut x)` — borrow a place mutably (e.g. a call argument that may be
        // written through).
        "mut" => {
            if args.len() != 1 {
                return Err(Diag::at(span, "mut: expects (mut PLACE)"));
            }
            Ok(ExprKind::Borrow {
                mutable: true,
                place: Box::new(parse_expr(&args[0])?),
            })
        }
        "alloc-stack" => alloc_form(args, Storage::Stack),
        "alloc-static" => alloc_form(args, Storage::Static),
        "alloc-heap" => alloc_form(args, Storage::Heap),
        "field" => {
            let p = parse_expr(args.first().ok_or("field: missing pointer")?)?;
            let name = sym(args.get(1).ok_or("field: missing field name")?, "field name")?;
            Ok(ExprKind::Field {
                ptr: Box::new(p),
                field: name,
            })
        }
        "get" => {
            let p = parse_expr(args.first().ok_or("get: missing pointer")?)?;
            let name = sym(args.get(1).ok_or("get: missing field name")?, "field name")?;
            Ok(ExprKind::BitGet {
                ptr: Box::new(p),
                field: name,
            })
        }
        "set!" => {
            let p = parse_expr(args.first().ok_or("set!: missing pointer")?)?;
            let name = sym(args.get(1).ok_or("set!: missing field name")?, "field name")?;
            let v = parse_expr(args.get(2).ok_or("set!: missing value")?)?;
            Ok(ExprKind::BitSet {
                ptr: Box::new(p),
                field: name,
                val: Box::new(v),
            })
        }
        "fnptr-of" => {
            if args.len() != 1 {
                return Err(Diag::at(span, "fnptr-of: expects (fnptr-of name)"));
            }
            Ok(ExprKind::FnPtrOf(sym(&args[0], "function name")?))
        }
        "call-ptr" => {
            let fp = parse_expr(args.first().ok_or("call-ptr: missing function pointer")?)?;
            let cargs = args[1..].iter().map(parse_expr).collect::<Result<_, _>>()?;
            Ok(ExprKind::CallPtr {
                fp: Box::new(fp),
                args: cargs,
            })
        }
        "load" => {
            if args.len() != 1 {
                return Err(Diag::at(span, "load: expects (load ptr)"));
            }
            Ok(ExprKind::Load(Box::new(parse_expr(&args[0])?)))
        }
        "store!" => {
            let (p, v) = two(args, "store!")?;
            Ok(ExprKind::Store {
                ptr: Box::new(parse_expr(p)?),
                val: Box::new(parse_expr(v)?),
            })
        }
        "free" => {
            if args.len() != 1 {
                return Err(Diag::at(span, "free: expects (free ptr)"));
            }
            Ok(ExprKind::Free(Box::new(parse_expr(&args[0])?)))
        }
        "index" => {
            let (p, i) = two(args, "index")?;
            Ok(ExprKind::Index {
                ptr: Box::new(parse_expr(p)?),
                idx: Box::new(parse_expr(i)?),
            })
        }
        "cast" => {
            if args.len() != 2 {
                return Err(Diag::at(span, "cast: expects (cast :type expr)"));
            }
            Ok(ExprKind::Cast {
                ty: parse_type(&args[0])?,
                expr: Box::new(parse_expr(&args[1])?),
            })
        }
        "match" => {
            let scrut = parse_expr(args.first().ok_or("match: missing scrutinee")?)?;
            let mut arms = Vec::new();
            for a in &args[1..] {
                let al = as_list(a, "match arm must be (Variant [binds] body)")?;
                let variant = sym(al.first().ok_or("arm: missing variant")?, "variant")?;
                let binds = match al.get(1).map(|s| &s.kind) {
                    Some(SexpKind::Vector(bv)) => {
                        bv.iter().map(|s| sym(s, "bind")).collect::<Result<_, _>>()?
                    }
                    _ => return Err(Diag::at(a.span, format!("arm '{variant}': expected a bind vector"))),
                };
                // ALL forms after the bind vector are the arm body — sequence
                // them in a `do` (a single form stays bare). Keeping only the
                // first would silently drop the rest (an arm's tail statements),
                // which silently breaks any multi-statement arm.
                let body_forms: Vec<Expr> =
                    al[2..].iter().map(parse_expr).collect::<Result<_, _>>()?;
                let body = match body_forms.len() {
                    // Reachable: an arm with a variant + bind vector but no body
                    // forms, e.g. `(Variant [])` (the `al.get(1)` Vector guard
                    // above ensures al[1] is the binds, so al[2..] is the body).
                    0 => return Err(Diag::at(a.span, format!("arm '{variant}': missing body"))),
                    1 => body_forms.into_iter().next().unwrap(),
                    _ => Expr::new(ExprKind::Do(body_forms), a.span),
                };
                arms.push(Arm { variant, binds, body });
            }
            Ok(ExprKind::Match {
                scrut: Box::new(scrut),
                arms,
            })
        }
        "sizeof" => {
            if args.len() != 1 {
                return Err(Diag::at(span, "sizeof: expects (sizeof TYPE)"));
            }
            Ok(ExprKind::SizeOf(parse_type(&args[0])?))
        }
        "alignof" => {
            if args.len() != 1 {
                return Err(Diag::at(span, "alignof: expects (alignof TYPE)"));
            }
            Ok(ExprKind::AlignOf(parse_type(&args[0])?))
        }
        "offsetof" => {
            if args.len() != 2 {
                return Err(Diag::at(span, "offsetof: expects (offsetof TYPE field)"));
            }
            Ok(ExprKind::OffsetOf(parse_type(&args[0])?, sym(&args[1], "field")?))
        }
        // compile-time type reflection (evaluated by the comptime interpreter)
        "field-count" | "variant-count" | "struct?" | "sum?" | "int?" | "float?"
        | "ptr?" | "array?" => {
            if args.len() != 1 {
                return Err(Diag::at(span, format!("{head}: expects ({head} TYPE)")));
            }
            let q = match head.as_str() {
                "field-count" => TypeQuery::FieldCount,
                "variant-count" => TypeQuery::VariantCount,
                "struct?" => TypeQuery::IsStruct,
                "sum?" => TypeQuery::IsSum,
                "int?" => TypeQuery::IsInt,
                "float?" => TypeQuery::IsFloat,
                "ptr?" => TypeQuery::IsPtr,
                _ => TypeQuery::IsArray,
            };
            Ok(ExprKind::TypeQuery { q, ty: parse_type(&args[0])? })
        }
        // per-field reflection: (field-name T i) / (field-type-kind T i) / (field-type-name T i)
        "field-name" | "field-type-kind" | "field-type-name" => {
            if args.len() != 2 {
                return Err(Diag::at(span, format!("{head}: expects ({head} TYPE index)")));
            }
            let meta = match head.as_str() {
                "field-name" => FieldMeta::Name,
                "field-type-kind" => FieldMeta::TypeKind,
                _ => FieldMeta::TypeName,
            };
            Ok(ExprKind::FieldMeta {
                meta,
                ty: parse_type(&args[0])?,
                idx: Box::new(parse_expr(&args[1])?),
            })
        }
        // (quote FORM) — quoted code as a comptime Code value (the raw syntax).
        "quote" => {
            if args.len() != 1 {
                return Err(Diag::at(span, "quote: expects (quote FORM)"));
            }
            Ok(ExprKind::Quote(Box::new(args[0].clone())))
        }
        // `template (quasiquote) — build Code, splicing ~E (unquote) holes.
        "quasiquote" => {
            if args.len() != 1 {
                return Err(Diag::at(span, "quasiquote: expects one template"));
            }
            Ok(ExprKind::Quasi(build_quasi(&args[0])?))
        }
        "unquote" => Err(Diag::at(span, "unquote (~) is only valid inside a quasiquote (`)")),
        // (gensym) — a fresh Code symbol (hygiene)
        "gensym" => {
            if !args.is_empty() {
                return Err(Diag::at(span, "gensym: expects (gensym)"));
            }
            Ok(ExprKind::CodeOp { op: CodeOp::Gensym, args: vec![] })
        }
        // operations on Code values
        "code-list?" | "code-sym?" | "code-int?" | "code-count" | "code-nth" | "code-sym"
        | "code-int" | "code-rest" => {
            let op = match head.as_str() {
                "code-list?" => CodeOp::IsList,
                "code-sym?" => CodeOp::IsSym,
                "code-int?" => CodeOp::IsInt,
                "code-count" => CodeOp::Count,
                "code-nth" => CodeOp::Nth,
                "code-sym" => CodeOp::Sym,
                "code-rest" => CodeOp::Rest,
                _ => CodeOp::Int,
            };
            Ok(ExprKind::CodeOp {
                op,
                args: args.iter().map(parse_expr).collect::<Result<_, _>>()?,
            })
        }
        // (field-index T name) — index of the field named `name`
        "field-index" => {
            if args.len() != 2 {
                return Err(Diag::at(span, "field-index: expects (field-index TYPE name)"));
            }
            Ok(ExprKind::FieldIndex {
                ty: parse_type(&args[0])?,
                name: Box::new(parse_expr(&args[1])?),
            })
        }
        // (loop [:label] body...) — the structured-loop primitive.
        "loop" => {
            let (label, rest) = parse_label(args);
            let body: Vec<Expr> = rest.iter().map(parse_expr).collect::<Result<_, _>>()?;
            if body.is_empty() {
                return Err(Diag::at(span, "loop: expects (loop [:label] body...)"));
            }
            Ok(ExprKind::Loop { label, body })
        }
        // (break [:label] [value]) — exit the (named) loop, optionally with a value.
        "break" => {
            let (label, rest) = parse_label(args);
            if rest.len() > 1 {
                return Err(Diag::at(span, "break: expects (break [:label] [value])"));
            }
            let value = match rest.first() {
                Some(e) => Some(Box::new(parse_expr(e)?)),
                None => None,
            };
            Ok(ExprKind::Break { label, value })
        }
        // (continue [:label]) — restart the (named) loop's body.
        "continue" => {
            let (label, rest) = parse_label(args);
            if !rest.is_empty() {
                return Err(Diag::at(span, "continue: expects (continue [:label])"));
            }
            Ok(ExprKind::Continue { label })
        }
        // (llvm-ir RESULT-TYPE [operand...] "BODY") — raw LLVM IR escape hatch.
        "llvm-ir" => {
            if args.len() != 3 {
                return Err(Diag::at(span, "llvm-ir: expects (llvm-ir RESULT-TYPE [operands...] \"BODY\")"));
            }
            let result = parse_type(&args[0])?;
            let operands = match &args[1].kind {
                SexpKind::Vector(v) => v.iter().map(parse_expr).collect::<Result<Vec<_>, _>>()?,
                _ => return Err(Diag::at(args[1].span, "llvm-ir: expected a [operand...] vector")),
            };
            let body = match &args[2].kind {
                SexpKind::Str(s) => s.clone(),
                _ => return Err(Diag::at(args[2].span, "llvm-ir: body must be a string literal")),
            };
            Ok(ExprKind::LlvmIr { result, args: operands, body })
        }
        // direct application: (fib n) == (call fib n). A leading vector is an
        // explicit type-argument list for a generic call: (id [i64] 5).
        other => {
            let (type_args, value_args): (Vec<Type>, &[Sexp]) = match args.first().map(|s| &s.kind) {
                Some(SexpKind::Vector(tv)) => {
                    (tv.iter().map(parse_type).collect::<Result<_, _>>()?, &args[1..])
                }
                _ => (vec![], args),
            };
            let cargs = value_args.iter().map(parse_expr).collect::<Result<_, _>>()?;
            Ok(ExprKind::Call {
                func: other.to_string(),
                type_args,
                args: cargs,
            })
        }
    }
}

// ---- small helpers -------------------------------------------------------

fn as_list<'a>(s: &'a Sexp, msg: &str) -> Result<&'a [Sexp], Diag> {
    match &s.kind {
        SexpKind::List(items) => Ok(items),
        _ => Err(Diag::at(s.span, msg)),
    }
}

fn head_sym(items: &[Sexp]) -> Result<String, Diag> {
    match items.first().map(|s| (&s.kind, s.span)) {
        Some((SexpKind::Sym(s), _)) => Ok(s.clone()),
        Some((_, span)) => Err(Diag::at(span, format!("expected a symbol head, got {}", items[0]))),
        None => Err(Diag::new("empty list")),
    }
}

fn sym(s: &Sexp, what: &str) -> Result<String, Diag> {
    match &s.kind {
        SexpKind::Sym(s) => Ok(s.clone()),
        _ => Err(Diag::at(s.span, format!("{what}: expected symbol, got {s}"))),
    }
}

fn keyword(s: &Sexp, what: &str) -> Result<String, Diag> {
    match &s.kind {
        SexpKind::Keyword(k) => Ok(k.clone()),
        _ => Err(Diag::at(s.span, format!("{what}: expected keyword, got {s}"))),
    }
}

fn sym_vec(s: &Sexp, what: &str) -> Result<Vec<String>, Diag> {
    match &s.kind {
        SexpKind::Vector(v) => v.iter().map(|x| sym(x, what)).collect(),
        _ => Err(Diag::at(s.span, format!("{what}: expected vector, got {s}"))),
    }
}

/// A leading `:label` keyword names a loop for `loop`/`break`/`continue`. Returns
/// the label (if present) and the remaining forms.
fn parse_label(args: &[Sexp]) -> (Option<String>, &[Sexp]) {
    match args.first().map(|s| &s.kind) {
        Some(SexpKind::Keyword(k)) => (Some(k.clone()), &args[1..]),
        _ => (None, args),
    }
}

fn two<'a>(args: &'a [Sexp], head: &str) -> Result<(&'a Sexp, &'a Sexp), Diag> {
    if args.len() != 2 {
        return Err(Diag::new(format!("{head}: expects exactly 2 arguments")));
    }
    Ok((&args[0], &args[1]))
}
