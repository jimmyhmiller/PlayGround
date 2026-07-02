//! A NATIVE backend for the dependent core (`dep.rs`), behind the `llvm` feature.
//!
//! This is the bridge that makes the dependent+linear front end run end to end:
//! a checked `dep::Term` is lowered to LLVM and JIT-executed, with no
//! intermediate normalization — eliminators become real loops / recursive
//! functions, so the recursion happens in native code, not in the type checker's
//! evaluator.
//!
//! Representations:
//!   * `Nat` (and any `Nat`-like family: one nullary "zero" constructor and one
//!     single-recursive-argument "successor") stays an UNBOXED `i64`. Its
//!     eliminator is a native counting loop.
//!   * Every other inductive family is BOXED: a constructor becomes a `malloc`'d
//!     block of `i64` slots. Slot 0 is an integer constructor TAG (the
//!     constructor's index in declaration order); the remaining slots hold the
//!     constructor's NON-ERASED arguments, in declaration order. Its eliminator
//!     becomes a recursive native function that switches on the tag and recurses
//!     on the recursive fields (the induction hypotheses).
//!
//! ERASURE is the zero-overhead guarantee: multiplicity-0 arguments (types,
//! indices, proofs) get NO runtime slot and NO instruction — they are never
//! stored, never loaded, and the de Bruijn binder for them carries `None` in the
//! runtime environment, so any attempt to read one at runtime is a hard error
//! (the QTT kernel guarantees this never happens for a checked term).

use crate::dep::{Signature, Term};
use crate::mult::Mult;
use inkwell::context::Context;
use inkwell::execution_engine::JitFunction;
use inkwell::module::Module;
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine,
};
use inkwell::types::IntType;
use inkwell::values::{FunctionValue, IntValue, PointerValue};
use inkwell::{AddressSpace, OptimizationLevel};
use std::cell::RefCell;
use std::path::Path;

/// Is `data` a `Nat`-like family (no params/indices; one nullary constructor and
/// one constructor with a single recursive argument)? If so, return
/// `(zero_ctor, succ_ctor)` names.
fn nat_like(sig: &Signature, data: &str) -> Option<(String, String)> {
    let d = sig.data(data)?;
    if !d.params.is_empty() || !d.indices.is_empty() || d.ctors.len() != 2 {
        return None;
    }
    let mut zero = None;
    let mut succ = None;
    for c in &d.ctors {
        if c.args.is_empty() {
            zero = Some(c.name.clone());
        } else if c.args.len() == 1 && matches!(&c.args[0].1, Term::Data(dn, _) if dn == data) {
            succ = Some(c.name.clone());
        }
    }
    Some((zero?, succ?))
}

/// The runtime layout of one boxed constructor: which of its *arguments* occupy
/// runtime slots (the non-erased ones), how WIDE each is (a flat-record field
/// spans several consecutive slots), and which are recursive occurrences of the
/// family. `args` are the constructor's own arguments (NOT the family params);
/// the index space here is the constructor's argument index `0..ctor.args.len()`.
struct CtorLayout {
    /// constructor tag = its position in `decl.ctors`.
    tag: u64,
    /// for each constructor argument: its first runtime slot (1-based; slot 0 is
    /// the tag), or `None` if it is erased (multiplicity 0) and therefore unstored.
    arg_slot: Vec<Option<u32>>,
    /// for each constructor argument: how many consecutive slots it occupies
    /// (1 for scalars/pointers/generic fields; a flat record's width inline).
    arg_width: Vec<u32>,
    /// for each constructor argument: is its WIDTH instantiation-dependent (a
    /// parameter-typed field, or a record containing one)? A flex field of a
    /// FLAT record takes its actual width from the value; a flex field of a
    /// BOXED datatype has a fixed one-slot cell layout, so a record cannot be
    /// stored there (hard error — box explicitly or use a concrete container).
    arg_flex: Vec<bool>,
    /// for each constructor argument: is it a recursive occurrence of the family?
    arg_recursive: Vec<bool>,
    /// number of runtime field slots (excluding the tag).
    nfields: u32,
}

fn ctor_layout(sig: &Signature, data: &str, cname: &str) -> Result<CtorLayout, String> {
    let decl = sig
        .data(data)
        .ok_or_else(|| format!("unknown datatype `{data}`"))?;
    let tag = decl
        .ctors
        .iter()
        .position(|c| c.name == cname)
        .ok_or_else(|| format!("`{cname}` is not a constructor of `{data}`"))? as u64;
    let ctor = &decl.ctors[tag as usize];
    let np = decl.params.len();
    let mut arg_slot = Vec::with_capacity(ctor.args.len());
    let mut arg_width = Vec::with_capacity(ctor.args.len());
    let mut arg_flex = Vec::with_capacity(ctor.args.len());
    let mut arg_recursive = Vec::with_capacity(ctor.args.len());
    let mut next: u32 = 1; // slot 0 is the tag
    for (ai, (mult, aty)) in ctor.args.iter().enumerate() {
        let recursive = matches!(aty, Term::Data(dn, _) if dn == data);
        // a recursive field is always ONE slot (a pointer — a record cannot be
        // recursive, so this only guards belt-and-suspenders).
        let (w, fx) = if recursive {
            (1, false)
        } else {
            field_width(sig, aty, ai, np, &[], &mut std::collections::HashSet::new())
        };
        if *mult == Mult::Zero {
            arg_slot.push(None);
            arg_width.push(0);
            arg_flex.push(false);
        } else {
            arg_slot.push(Some(next));
            arg_width.push(w);
            arg_flex.push(fx);
            next += w;
        }
        arg_recursive.push(recursive);
    }
    Ok(CtorLayout {
        tag,
        arg_slot,
        arg_width,
        arg_flex,
        arg_recursive,
        nfields: next - 1,
    })
}

/// Is `data` a FLAT RECORD — a datatype whose values live in REGISTERS, never a
/// heap cell? Criteria: exactly one constructor, at least TWO non-erased fields
/// (one is the transparent-newtype path, zero the shared-constant path), no
/// recursive field (recursion needs indirection). Returns the total width in
/// scalar slots (fields of other record types flatten recursively; generic /
/// pointer / scalar fields are one slot). This is Phase B2's core: constructing
/// one is `insertvalue`-free SSA, matching is projection — zero malloc, no tag.
fn record_width(sig: &Signature, data: &str) -> Option<u32> {
    match record_shape(sig, data, &[], &mut std::collections::HashSet::new()) {
        Some((w, _flex)) => Some(w),
        None => None,
    }
}

/// The SHAPE of a flat record instantiated at `dargs` (the datatype's type
/// arguments, possibly empty): `Some((width, flex))` where `width` is the total
/// scalar-slot count with every parameter-typed field resolved through `dargs`,
/// and `flex` is true when some field's width could not be resolved (an
/// abstract `Var` survives) — the record's layout then varies by instantiation
/// and only the VALUE knows its true width. `None` = not a flat record.
fn record_shape(
    sig: &Signature,
    data: &str,
    dargs: &[Term],
    seen: &mut std::collections::HashSet<String>,
) -> Option<(u32, bool)> {
    if !seen.insert(data.to_string()) {
        return None; // a cycle through record types needs indirection: boxed.
    }
    let d = sig.data(data)?;
    if sig.boxed_types.contains(data) || nat_like(sig, data).is_some() || d.ctors.len() != 1 {
        seen.remove(data);
        return None;
    }
    let np = d.params.len();
    let ctor = &d.ctors[0];
    let mut nruntime = 0usize;
    let mut width = 0u32;
    let mut flex = false;
    for (ai, (mult, aty)) in ctor.args.iter().enumerate() {
        if matches!(aty, Term::Data(dn, _) if dn == data) {
            seen.remove(data);
            return None; // recursive: needs indirection.
        }
        if *mult == Mult::Zero {
            continue;
        }
        nruntime += 1;
        let (w, fx) = field_width(sig, aty, ai, np, dargs, seen);
        width += w;
        flex |= fx;
    }
    seen.remove(data);
    if nruntime < 2 {
        return None; // nullary → shared constant; single → transparent newtype.
    }
    Some((width, flex))
}

/// Width of constructor-argument `ai`'s type, resolving a reference to the
/// datatype's own parameters through the instantiation `dargs`. The context of
/// a ctor arg type is `[params…, earlier ctor args…]`: `Var(j)` with
/// `j >= ai` points into the params (param index `np - 1 - (j - ai)`).
fn field_width(
    sig: &Signature,
    aty: &Term,
    ai: usize,
    np: usize,
    dargs: &[Term],
    seen: &mut std::collections::HashSet<String>,
) -> (u32, bool) {
    match strip_ann(aty) {
        Term::Var(j) if *j >= ai && (*j - ai) < np => {
            let pidx = np - 1 - (*j - ai);
            match dargs.get(pidx) {
                Some(arg) => type_shape(sig, arg, seen),
                None => (1, true), // uninstantiated parameter: flexible.
            }
        }
        Term::Var(_) => (1, true), // a dependent (non-param) type variable.
        other => type_shape(sig, other, seen),
    }
}

/// The shape of a VALUE ENUM instantiated at `dargs`: a multi-constructor,
/// non-recursive, non-`boxed` datatype is a flat TAGGED UNION in registers —
/// `[tag, payload…, zero-padding]`, width `1 + max(payload widths)`. No value
/// of it ever allocates. Returns per-constructor payload widths + flex flags
/// alongside the total. `None` = not a value enum (single-ctor, Nat-like,
/// recursive, or declared `boxed`).
struct EnumShape {
    /// total width: 1 (tag) + the widest constructor payload — or 1 for a
    /// NULL-NICHE enum (see `niche`).
    width: u32,
    /// per constructor: (payload width at this instantiation, has-flex-field).
    payloads: Vec<(u32, bool)>,
    /// THE NULL-POINTER NICHE: `(nullary_ci, ptr_ci)` when the enum is exactly
    /// {one nullary ctor, one single-`Own`-pointer ctor} at this instantiation —
    /// the C nullable-pointer pattern. The value is then ONE slot: the nullary
    /// ctor is 0, the pointer ctor is the (non-null, malloc'd) pointer itself.
    /// `Opt (Own Node)` costs nothing over C's `struct node *`.
    niche: Option<(usize, usize)>,
}

fn enum_value_shape(
    sig: &Signature,
    data: &str,
    dargs: &[Term],
    seen: &mut std::collections::HashSet<String>,
) -> Option<EnumShape> {
    if sig.boxed_types.contains(data) || nat_like(sig, data).is_some() {
        return None;
    }
    let d = sig.data(data)?;
    if d.ctors.len() < 2 {
        return None;
    }
    if !seen.insert(data.to_string()) {
        return None; // a cycle needs indirection: not a value.
    }
    let np = d.params.len();
    let mut payloads = Vec::with_capacity(d.ctors.len());
    let mut maxw = 0u32;
    for ctor in &d.ctors {
        let mut w = 0u32;
        let mut flex = false;
        for (ai, (mult, aty)) in ctor.args.iter().enumerate() {
            if matches!(aty, Term::Data(dn, _) if dn == data) {
                seen.remove(data);
                return None; // recursive: needs indirection.
            }
            if *mult == Mult::Zero {
                continue;
            }
            let (fw, fx) = field_width(sig, aty, ai, np, dargs, seen);
            w += fw;
            flex |= fx;
        }
        maxw = maxw.max(w);
        payloads.push((w, flex));
    }
    seen.remove(data);
    // null-niche detection: exactly two constructors — one nullary, one with a
    // single runtime field that RESOLVES (through the instantiation) to an
    // `Own …` pointer.
    let mut niche = None;
    if d.ctors.len() == 2 {
        let nullary = payloads.iter().position(|(w, _)| *w == 0);
        if let Some(nc) = nullary {
            let pc = 1 - nc;
            let ctor = &d.ctors[pc];
            let runtime: Vec<usize> = ctor
                .args
                .iter()
                .enumerate()
                .filter(|(_, (m, _))| *m != Mult::Zero)
                .map(|(i, _)| i)
                .collect();
            if runtime.len() == 1 {
                let ai = runtime[0];
                let aty = &ctor.args[ai].1;
                // resolve a parameter-typed field through dargs.
                let resolved: Option<&Term> = match strip_ann(aty) {
                    Term::Var(j) if *j >= ai && (*j - ai) < np => {
                        dargs.get(np - 1 - (*j - ai))
                    }
                    other => Some(other),
                };
                let is_own = resolved.is_some_and(|t| {
                    let (head, _) = flatten_app(strip_ann(t));
                    matches!(strip_ann(head), Term::Const(c) if c == "Own")
                });
                if is_own {
                    niche = Some((nc, pc));
                }
            }
        }
    }
    let width = if niche.is_some() { 1 } else { 1 + maxw };
    Some(EnumShape { width, payloads, niche })
}

/// Is `data` a {one nullary ctor, one single-runtime-field ctor} enum — the
/// null-niche CANDIDATE shape? Returns (nullary_ci, ptr_ci). Whether a given
/// VALUE actually uses the niche representation is decided by its width
/// (niche = 1 slot; the tagged form of this shape is always ≥ 2 slots).
fn d_nullary_ptr_shape(sig: &Signature, data: &str) -> Option<(usize, usize)> {
    let d = sig.data(data)?;
    if d.ctors.len() != 2 {
        return None;
    }
    let nruntime = |ci: usize| {
        d.ctors[ci]
            .args
            .iter()
            .filter(|(m, _)| *m != Mult::Zero)
            .count()
    };
    match (nruntime(0), nruntime(1)) {
        (0, 1) => Some((0, 1)),
        (1, 0) => Some((1, 0)),
        _ => None,
    }
}

/// LAYOUT VALIDATION — the zero-implicit-allocation gate, run on every checked
/// program: a datatype that is neither `%builtin Nat`-shaped nor declared
/// `boxed enum` must have a finite VALUE layout. A recursive occurrence (direct
/// or through other value types) is rejected with the two explicit fixes.
pub fn validate_layouts(sig: &Signature) -> Result<(), String> {
    for decl in &sig.datas {
        if sig.boxed_types.contains(&decl.name) || nat_like(sig, &decl.name).is_some() {
            continue;
        }
        let mut path = vec![decl.name.clone()];
        if inline_reaches(sig, &decl.name, &decl.name, &mut path) {
            let chain = path.join(" → ");
            return Err(format!(
                "`{0}` is RECURSIVE without indirection ({chain}): a VALUE of it would be infinitely sized, and tallyc constructors never allocate implicitly. Either add explicit indirection — an `Own {0}` (or `Opt (Own {0})`) field, so every node is an `alloc` you write — or declare it `boxed enum {0}` to opt into the heap-cell representation (its constructors then allocate one cell per value, visibly, by declaration).",
                decl.name
            ));
        }
    }
    Ok(())
}

/// Does `data`'s INLINE layout (fields stored by value, following value enums,
/// records, and transparent wrappers, resolving parameter instantiations, but
/// stopping at pointers — `Own`/postulate applications — and at `boxed`/
/// Nat-like datatypes) reach `target`? `path` records the chain for the error.
fn inline_reaches(sig: &Signature, data: &str, target: &str, path: &mut Vec<String>) -> bool {
    let Some(d) = sig.data(data) else { return false };
    let np = d.params.len();
    for ctor in &d.ctors {
        for (ai, (mult, aty)) in ctor.args.iter().enumerate() {
            if *mult == Mult::Zero {
                continue; // erased: never stored.
            }
            if inline_ty_reaches(sig, aty, ai, np, &[], target, path) {
                return true;
            }
        }
    }
    false
}

fn inline_ty_reaches(
    sig: &Signature,
    t: &Term,
    ai: usize,
    np: usize,
    dargs: &[Term],
    target: &str,
    path: &mut Vec<String>,
) -> bool {
    match strip_ann(t) {
        // a parameter-typed field: resolve through the instantiation if known.
        Term::Var(j) if *j >= ai && (*j - ai) < np => {
            let pidx = np - 1 - (*j - ai);
            match dargs.get(pidx) {
                Some(arg) => inline_ty_reaches(sig, arg, 0, 0, &[], target, path),
                None => false, // uninstantiated: nothing stored yet.
            }
        }
        Term::Data(dn, da) => {
            if dn == target {
                return true;
            }
            // boxed / Nat-like datatypes are pointers/scalars: indirection.
            if sig.boxed_types.contains(dn) || nat_like(sig, dn).is_some() {
                return false;
            }
            if path.contains(dn) {
                return false; // some other cycle; it gets its own diagnosis.
            }
            path.push(dn.clone());
            let Some(d2) = sig.data(dn) else {
                path.pop();
                return false;
            };
            let np2 = d2.params.len();
            for ctor in &d2.ctors {
                for (ai2, (mult, aty)) in ctor.args.iter().enumerate() {
                    if *mult == Mult::Zero {
                        continue;
                    }
                    if inline_ty_reaches(sig, aty, ai2, np2, da, target, path) {
                        return true;
                    }
                }
            }
            path.pop();
            false
        }
        // Own/PtsTo/other postulate applications, Π, Σ, … : one slot, a pointer
        // or opaque scalar — an INDIRECTION boundary, recursion behind it is the
        // explicit kind.
        _ => false,
    }
}

/// The runtime WIDTH (in i64 slots) of a value of this type term, plus a FLEX
/// bit: flat records span their (instantiated) field widths; a transparent
/// newtype is exactly its field; every other type — scalars, boxed pointers —
/// is one slot. An abstract (`Var`-headed, or unresolved-parameter) width is
/// `(1, flex = true)`: the layout varies by instantiation, and tallyc NEVER
/// papers over that with a hidden box — width-sensitive consumers of a flex
/// type are hard errors pointing at the explicit alternative.
fn type_width(sig: &Signature, t: &Term) -> u32 {
    type_shape(sig, t, &mut std::collections::HashSet::new()).0
}

fn type_is_flex(sig: &Signature, t: &Term) -> bool {
    type_shape(sig, t, &mut std::collections::HashSet::new()).1
}

fn type_shape(
    sig: &Signature,
    t: &Term,
    seen: &mut std::collections::HashSet<String>,
) -> (u32, bool) {
    let t = strip_ann(t);
    let (name, dargs): (&str, Vec<Term>) = match t {
        Term::Data(dn, da) => (dn.as_str(), da.clone()),
        Term::Var(_) => return (1, true),
        // an APPLIED head (quoted spines): `((Data …) a) n` or a Var head.
        Term::App(_, _) => {
            let (head, args) = flatten_app(t);
            match strip_ann(head) {
                Term::Data(dn, da) => {
                    let mut all = da.clone();
                    all.extend(args.iter().map(|a| (*a).clone()));
                    (dn.as_str(), all)
                }
                Term::Var(_) => return (1, true),
                // A VIEW is a ZERO-WIDTH value (Phase C): `PtsTo l a` / `Loan l a`
                // are linear PERMISSIONS — the checker's accounting is their whole
                // existence; at runtime they are NOTHING (the address lives in the
                // ω `Ptr l`). This is the §4.4 erasure bar: a view leaves no trace.
                Term::Const(c) if c == "PtsTo" || c == "Loan" => return (0, false),
                // `Hole a` (the taken/uninitialized typestate) has `a`'s SIZE —
                // the take-then-refill discipline must remember how big the slot
                // is, or a wider refill would overflow the allocation.
                Term::Const(c) if c == "Hole" => {
                    return match args.first() {
                        Some(a) => type_shape(sig, a, seen),
                        None => (1, false),
                    }
                }
                _ => return (1, false),
            }
        }
        _ => return (1, false),
    };
    if let Some(fi) = transparent_field(sig, name) {
        // the wrapper IS its field: same width, resolved through the args.
        if let Some(d) = sig.data(name) {
            let np = d.params.len();
            let aty = d.ctors[0].args[fi].1.clone();
            return field_width(sig, &aty, fi, np, &dargs, seen);
        }
        return (1, false);
    }
    // avoid infinite recursion through record cycles.
    if seen.contains(name) {
        return (1, false);
    }
    if let Some(shape) = record_shape(sig, name, &dargs, seen) {
        return shape;
    }
    if let Some(es) = enum_value_shape(sig, name, &dargs, seen) {
        let flex = es.payloads.iter().any(|(_, fx)| *fx);
        return (es.width, flex);
    }
    (1, false)
}

/// A runtime VALUE: one scalar `i64` (an integer or a pointer disguised as
/// one — the representation of everything before Phase B2), or a FLAT RECORD
/// by value — its fields as SSA scalars, in registers, never allocated.
#[derive(Clone, Debug)]
enum Val<'c> {
    Int(IntValue<'c>),
    Agg(Vec<IntValue<'c>>),
}

impl<'c> Val<'c> {
    fn width(&self) -> u32 {
        match self {
            Val::Int(_) => 1,
            Val::Agg(vs) => vs.len() as u32,
        }
    }
    fn as_int(&self, what: &str) -> Result<IntValue<'c>, String> {
        match self {
            Val::Int(v) => Ok(*v),
            Val::Agg(vs) => Err(format!(
                "{what}: a {}-field flat record flowed where a scalar was required \
                 (records are not yet supported in this position)",
                vs.len()
            )),
        }
    }
    fn components(&self) -> Vec<IntValue<'c>> {
        match self {
            Val::Int(v) => vec![*v],
            Val::Agg(vs) => vs.clone(),
        }
    }
}

/// A TRANSPARENT (newtype) datatype: exactly one constructor with exactly one
/// non-erased field, and that field not recursive. Its runtime value IS the
/// field — no cell, no tag, no allocation, no load (Phase B slice 1: a
/// zero-cost wrapper, the first departure from the everything-is-a-boxed-cell
/// representation). Returns the field's argument index.
fn transparent_field(sig: &Signature, data: &str) -> Option<usize> {
    let decl = sig.data(data)?;
    if decl.ctors.len() != 1 || sig.boxed_types.contains(data) {
        return None;
    }
    let ctor = &decl.ctors[0];
    let mut field = None;
    for (i, (mult, aty)) in ctor.args.iter().enumerate() {
        if *mult == Mult::Zero {
            continue;
        }
        if field.is_some() {
            return None; // two runtime fields: a real record, boxed
        }
        if matches!(aty, Term::Data(dn, _) if dn == data) {
            return None; // recursive: needs indirection
        }
        field = Some(i);
    }
    field
}

struct DepCg<'c, 'a> {
    ctx: &'c Context,
    i64t: IntType<'c>,
    ptr: inkwell::types::PointerType<'c>,
    builder: &'a inkwell::builder::Builder<'c>,
    module: &'a Module<'c>,
    malloc: FunctionValue<'c>,
    free: FunctionValue<'c>,
    sig: &'a Signature,
    /// monotonically-increasing counter for unique generated-function names.
    next_id: RefCell<u32>,
    /// one compiled native function per `Fix` term (memoized by its address), so a
    /// recursive function shared across call sites is emitted once.
    fix_cache: RefCell<std::collections::HashMap<*const Term, FunctionValue<'c>>>,
    /// the stack of `Fix` self-bindings currently in scope: a reference to the
    /// self variable (at de Bruijn LEVEL `level`) compiles to a call to `func`,
    /// passing only the non-erased arguments (`erased[i]` flags multiplicity-0).
    fix_selves: RefCell<Vec<FixSelf<'c>>>,
    /// the stack of accumulator-fold induction-hypothesis bindings in scope (Phase
    /// 1a′ native codegen). Inside the successor method of a function-typed-motive
    /// `NatElim`, the IH (at de Bruijn LEVEL `level`) is the recursive fold on the
    /// predecessor `k`; `App(ih, accs…)` compiles to a call `func(k, accs…)`.
    acc_ih_selves: RefCell<Vec<AccIhSelf<'c>>>,
    elim_fold_ihs: RefCell<Vec<ElimFoldIh<'c>>>,
}

struct FixSelf<'c> {
    level: usize,
    func: FunctionValue<'c>,
    /// one entry per surface argument: `None` = erased (not passed), `Some(w)` =
    /// a runtime argument of width `w` scalar slots (a flat record flattens into
    /// `w` consecutive i64 parameters — for a small record this is exactly the
    /// two-register C ABI).
    params: Vec<Option<u32>>,
    /// width of the return value (a `w > 1` fn returns an LLVM `{i64 × w}`).
    ret_width: u32,
}

/// An accumulator-fold induction-hypothesis binding (Phase 1a′ native codegen): the
/// IH variable (at de Bruijn LEVEL `ih_level`) stands for "the fold on the
/// predecessor `k`", so `App(ih, accs…)` compiles to `call func(k, accs…)`. The
/// predecessor `k` is identified by its env LEVEL (`k_level`), NOT a raw value — so
/// it flows correctly through a boxed-match helper (which re-captures the env by
/// level into its own parameters). `func` is the native function for the fold.
struct AccIhSelf<'c> {
    ih_level: usize,
    func: FunctionValue<'c>,
    k_level: usize,
}

/// A CONVOY-FOLD induction hypothesis (a boxed eliminator whose motive
/// Π-abstracts the index-dependent deps): `ih(deps…)` compiles to
/// `func(field, deps…, live…)`. The recursive FIELD and the captured live
/// slots are identified by env LEVEL so they thread through nested helpers.
struct ElimFoldIh<'c> {
    ih_level: usize,
    func: FunctionValue<'c>,
    field_level: usize,
    live_idx: Vec<usize>,
    ndep: usize,
}

/// A runtime environment slot: `Some(v)` for a live runtime value, `None` for an
/// ERASED binder (a multiplicity-0 variable that has no runtime witness). Reading
/// a `None` is a hard error — the kernel guarantees a checked term never does.
type Slot<'c> = Option<Val<'c>>;

impl<'c, 'a> DepCg<'c, 'a> {
    /// Require a value to occupy exactly `w` scalar slots. There is NO implicit
    /// representation change here — tallyc NEVER boxes behind your back. A flat
    /// record reaching a one-slot position (a generic field of a boxed
    /// datatype, an abstract-typed `%partial` parameter, a fold accumulator the
    /// lowering cannot widen) is a HARD ERROR naming the explicit alternatives;
    /// any other mismatch is a compiler bug surfaced loudly.
    fn expect_width(&self, v: Val<'c>, w: u32, what: &str) -> Result<Val<'c>, String> {
        if v.width() == w {
            return Ok(v);
        }
        match &v {
            Val::Agg(vs) if w == 1 => Err(format!(
                "{what}: a {}-field flat record cannot flow into this ONE-slot \
                 position — its layout is instantiation-dependent there, and tallyc \
                 never boxes implicitly. Box it explicitly (`alloc(…) : Own T`) or \
                 make the container/function concrete (a monomorphic field/parameter \
                 stores the record flat).",
                vs.len()
            )),
            _ => Err(format!(
                "{what}: layout width mismatch (value {} vs expected {w} slots) — \
                 this should be impossible for a kernel-checked term",
                v.width()
            )),
        }
    }
    fn fresh(&self, base: &str) -> String {
        let mut id = self.next_id.borrow_mut();
        let s = format!("{base}.{}", *id);
        *id += 1;
        s
    }

    fn gep(&self, p: PointerValue<'c>, idx: u32, name: &str) -> PointerValue<'c> {
        unsafe {
            self.builder
                .build_gep(self.i64t, p, &[self.i64t.const_int(idx as u64, false)], name)
                .unwrap()
        }
    }
    fn load(&self, p: PointerValue<'c>, idx: u32, name: &str) -> IntValue<'c> {
        self.builder
            .build_load(self.i64t, self.gep(p, idx, name), name)
            .unwrap()
            .into_int_value()
    }

    /// Load one constructor field: a scalar slot, or a flat record's `w`
    /// consecutive slots (loaded into registers — the record leaves the cell by
    /// value).
    fn load_field(&self, p: PointerValue<'c>, idx: u32, w: u32) -> Val<'c> {
        if w == 0 {
            return Val::Agg(Vec::new()); // a zero-width view: nothing to load.
        }
        if w == 1 {
            return Val::Int(self.load(p, idx, "field"));
        }
        let mut comps = Vec::with_capacity(w as usize);
        for k in 0..w {
            comps.push(self.load(p, idx + k, "field"));
        }
        Val::Agg(comps)
    }

    /// Read a runtime variable, erroring if it is erased (no runtime witness).
    fn read_var(&self, env: &[Slot<'c>], i: usize) -> Result<Val<'c>, String> {
        let pos = env
            .len()
            .checked_sub(1)
            .and_then(|n| n.checked_sub(i))
            .ok_or_else(|| format!("unbound runtime variable #{i}"))?;
        match env.get(pos) {
            Some(Some(v)) => Ok(v.clone()),
            Some(None) => Err(format!(
                "runtime variable #{i} is ERASED (multiplicity 0): it has no runtime \
                 representation, yet codegen reached it in a value position \
                 (this should be impossible for a kernel-checked term)"
            )),
            None => Err(format!("unbound runtime variable #{i}")),
        }
    }

    /// Compile a term to a SCALAR `i64` — the pre-B2 interface, used everywhere
    /// a scalar is structurally required (arithmetic, tags, indices, pointers).
    /// A flat record reaching such a position is a hard error, not a silent box.
    fn compile(
        &self,
        f: FunctionValue<'c>,
        env: &[Slot<'c>],
        t: &Term,
    ) -> Result<IntValue<'c>, String> {
        self.compile_v(f, env, t)?.as_int("compile")
    }

    /// Compile a term to a runtime `Val` — a scalar `i64`, or a flat record's
    /// fields as SSA scalars (Phase B2: records by VALUE, in registers). `env`
    /// gives each de Bruijn var's value, innermost last; `None` = erased.
    fn compile_v(
        &self,
        f: FunctionValue<'c>,
        env: &[Slot<'c>],
        t: &Term,
    ) -> Result<Val<'c>, String> {
        match t {
            Term::NatLit(n) => Ok(Val::Int(self.i64t.const_int(*n, false))),
            Term::Zero => Ok(Val::Int(self.i64t.const_int(0, false))),
            Term::Suc(x) => {
                let v = self.compile(f, env, x)?;
                Ok(Val::Int(
                    self.builder
                        .build_int_add(v, self.i64t.const_int(1, false), "suc")
                        .unwrap(),
                ))
            }
            Term::Add(a, b) => {
                let x = self.compile(f, env, a)?;
                let y = self.compile(f, env, b)?;
                Ok(Val::Int(self.builder.build_int_add(x, y, "add").unwrap()))
            }
            Term::Var(i) => self.read_var(env, *i),
            // a string literal: the address of one shared NUL-terminated global.
            Term::StrLit(x) => {
                let id = {
                    let mut n = self.next_id.borrow_mut();
                    *n += 1;
                    *n
                };
                let g = self
                    .builder
                    .build_global_string_ptr(x, &format!("tally.str.{id}"))
                    .map_err(|e| format!("string literal: {e}"))?;
                Ok(Val::Int(
                    self.builder
                        .build_ptr_to_int(g.as_pointer_value(), self.i64t, "strlit")
                        .unwrap(),
                ))
            }
            // `J(P, b, e)`: at runtime every closed equality proof is `refl`
            // (the kernel admits no equality axioms), so path induction ERASES
            // to its base case — the motive and the proof cost nothing.
            Term::J(_, b, _) => self.compile_v(f, env, b),
            Term::Ann(e, _) => self.compile_v(f, env, e),
            // CALL-BY-VALUE let: compile `e` ONCE (so its effects — e.g. `free` — run
            // exactly once), bind it, compile the body. (`ty` is 0-erased.)
            Term::Let(_sigma, _ty, e, body) => {
                let ev = self.compile_v(f, env, e)?;
                let mut env2 = env.to_vec();
                env2.push(Some(ev));
                self.compile_v(f, &env2, body)
            }
            Term::NatElim(_p, z, s, scrut) => self.compile_fold(f, env, z, s, scrut),
            Term::NatCase(_p, z, s, scrut) => self.compile_natcase(f, env, z, s, scrut),
            // a fully-applied `Fix` is handled in the `App` spine below; a bare one
            // (a function value with no call) has no runtime representation here.
            Term::Fix(_, _) => Err("a recursive function must be applied to its arguments".into()),
            Term::Constr(name, args) => self.compile_constr(f, env, name, args),
            Term::Elim(data, _motive, methods, scrut) => {
                self.compile_elim(f, env, data, methods, scrut)
            }
            Term::Case(data, _motive, methods, scrut) => {
                self.compile_case(f, env, data, methods, scrut)
            }
            Term::App(_, _) => {
                // β-reduce a fully-applied spine: (λ…λ. body) a₁ … aₙ
                let (head, args) = flatten_app(t);
                // A function-typed-motive `NatElim` applied to its accumulators is an
                // ACCUMULATOR FOLD (Phase 1a′): compile it to a native recursive
                // function threading the accumulators (the IH is the fold on `k`).
                if let Term::NatElim(p, z, s, scrut) = strip_ann(head) {
                    if !args.is_empty() {
                        return Ok(Val::Int(
                            self.compile_acc_fold(f, env, p, z, s, scrut, &args)?,
                        ));
                    }
                }
                // A postulate at the head of a spine (e.g. `alloc`, `free`,
                // `insert`, `remove`) is a memory primitive: dispatch to its
                // native implementation, erasing its multiplicity-0 arguments.
                if let Term::Const(c) = strip_ann(head) {
                    return self.compile_postulate(f, env, c, &args);
                }
                // A reference to an enclosing `Fix`'s self variable is a recursive
                // CALL (not an inline): emit `call self_fn(non-erased args)`. An
                // accumulator-fold IH self is the same idea — a recursive fold call
                // on the predecessor `k`, with `k` prepended to the accumulators.
                if let Term::Var(i) = strip_ann(head) {
                    let lvl = env.len() - 1 - *i;
                    // a CONVOY-FOLD IH applied to its deps: recurse on the field.
                    let ef = self
                        .elim_fold_ihs
                        .borrow()
                        .iter()
                        .rev()
                        .find(|e| e.ih_level == lvl)
                        .map(|e| (e.func, e.field_level, e.live_idx.clone(), e.ndep));
                    if let Some((func, field_level, live_idx, ndep)) = ef {
                        if args.len() != ndep {
                            return Err(format!(
                                "convoy-fold recursive call: expected {ndep} varying argument(s), got {}",
                                args.len()
                            ));
                        }
                        let fieldv = env[field_level]
                            .clone()
                            .ok_or("convoy-fold: the recursive field was erased")?
                            .as_int("convoy-fold recursive field")?;
                        let mut ca: Vec<inkwell::values::BasicMetadataValueEnum> =
                            vec![fieldv.into()];
                        for a in &args {
                            // dep positions are one-slot (box a record crossing in).
                            let v = self.compile_v(f, env, a)?;
                            ca.push(
                                self.expect_width(v, 1, "a convoy-fold dependent")?
                                    .as_int("convoy-fold dep")?
                                    .into(),
                            );
                        }
                        for li in live_idx {
                            for c in env[li]
                                .clone()
                                .ok_or("convoy-fold: a captured value was erased")?
                                .components()
                            {
                                ca.push(c.into());
                            }
                        }
                        return Ok(Val::Int(
                            self.builder
                                .build_call(func, &ca, "foldih.call")
                                .unwrap()
                                .try_as_basic_value()
                                .left()
                                .unwrap()
                                .into_int_value(),
                        ));
                    }
                    // copy the lookup OUT of the borrow before compiling args (which may
                    // re-enter and borrow `acc_ih_selves` again).
                    let ih_hit = self
                        .acc_ih_selves
                        .borrow()
                        .iter()
                        .rev()
                        .find(|a| a.ih_level == lvl)
                        .map(|a| (a.func, a.k_level));
                    if let Some((func, k_level)) = ih_hit {
                        // read the predecessor `k` from the CURRENT env by level — inside
                        // a boxed-match helper this is the helper's captured parameter.
                        let k = env[k_level].clone().ok_or_else(|| {
                            format!("recursive fold call to `{}`: predecessor is erased", func.get_name().to_str().unwrap_or("?"))
                        })?.as_int("accumulator-fold predecessor")?;
                        let mut call_args: Vec<inkwell::values::BasicMetadataValueEnum> =
                            vec![k.into()];
                        for a in &args {
                            // accumulator slots are one-slot (box a record crossing in).
                            let v = self.compile_v(f, env, a)?;
                            call_args.push(
                                self.expect_width(v, 1, "a total-fold accumulator (make the \
                                     function %partial — Fix supports records by value)")?
                                    .as_int("accumulator arg")?
                                    .into(),
                            );
                        }
                        return Ok(Val::Int(
                            self.builder
                                .build_call(func, &call_args, "ih.call")
                                .unwrap()
                                .try_as_basic_value()
                                .left()
                                .unwrap()
                                .into_int_value(),
                        ));
                    }
                    let hit = self
                        .fix_selves
                        .borrow()
                        .iter()
                        .rev()
                        .find(|fs| fs.level == lvl)
                        .map(|fs| (fs.func, fs.params.clone(), fs.ret_width));
                    if let Some((func, params, retw)) = hit {
                        return self.compile_call(f, env, func, &params, retw, &args);
                    }
                }
                // The head is itself a `Fix`: build (memoized) its function and call.
                if let Term::Fix(ty, body) = strip_ann(head) {
                    let (func, params, retw) = self.build_fix(strip_ann(head), ty, body)?;
                    return self.compile_call(f, env, func, &params, retw, &args);
                }
                // The CONVOY (a dependent match whose motive Π-abstracts the
                // index-dependent context variables — rust_surface's
                // `elab_nested_match`): the checked term is `(Case/Elim …) dep₁ … dep_k`,
                // each method ending in k dep-lambdas. COMMUTE the application into
                // the arms (the case-commuting conversion): substitute the deps into
                // each method body and compile the plain `Case`/`Elim`, the switch
                // shape this backend already knows. Sound because exactly one arm
                // runs, and the elaborator only applies context VARIABLES here
                // (already-computed values — nothing is re-evaluated or duplicated).
                match strip_ann(head) {
                    Term::Case(d, m, methods, sc) => {
                        let new = self.commute_apply_into_methods(d, methods, &args, false)?;
                        return self.compile_v(
                            f,
                            env,
                            &Term::Case(d.clone(), m.clone(), new, sc.clone()),
                        );
                    }
                    Term::Elim(d, m, methods, sc) => {
                        // methods that USE their induction hypotheses are a real
                        // recursive fold — deps vary per level, so they must be
                        // threaded as parameters, not substituted in.
                        if self.elim_methods_use_ih(d, methods, args.len())? {
                            return Ok(Val::Int(
                                self.compile_elim_dep_fold(f, env, d, methods, sc, &args)?,
                            ));
                        }
                        let new = self.commute_apply_into_methods(d, methods, &args, true)?;
                        return self.compile_v(
                            f,
                            env,
                            &Term::Elim(d.clone(), m.clone(), new, sc.clone()),
                        );
                    }
                    // The NAT CONVOY: `(NatCase …) dep₁ … dep_k` — same conversion.
                    // The zero method has no telescope binders; the successor method
                    // keeps its one predecessor binder.
                    Term::NatCase(p, z, s, sc) => {
                        let z2 = self.commute_apply_into_natcase_method(z, 0, &args)?;
                        let s2 = self.commute_apply_into_natcase_method(s, 1, &args)?;
                        return self.compile_v(
                            f,
                            env,
                            &Term::NatCase(p.clone(), Box::new(z2), Box::new(s2), sc.clone()),
                        );
                    }
                    _ => {}
                }
                // The head is an annotated function `(λ…. body : Π…. _)` (defs are
                // inlined this way). Walk the body's lambdas and the type's Pi
                // telescope in lockstep so we know each argument's MULTIPLICITY: a
                // `Π[0]` binder erases its argument — it is never compiled, never
                // stored, and binds `None`. This is the erasure that keeps indices
                // and proofs out of the runtime (zero instructions, zero slots).
                let mut head_ty: Option<&Term> = match head {
                    Term::Ann(_, ty) => Some(strip_ann(ty)),
                    _ => None,
                };
                let mut body = strip_ann(head);
                let mut env2 = env.to_vec();
                for a in &args {
                    match body {
                        Term::Lam(inner) => {
                            // multiplicity of this binder, read off the Pi type.
                            let mult = match head_ty {
                                Some(Term::Pi(m, _dom, _cod)) => Some(*m),
                                Some(_) => {
                                    return Err(
                                        "function applied to more arguments than its type's \
                                         Pi telescope allows"
                                            .into(),
                                    )
                                }
                                None => None,
                            };
                            match mult {
                                Some(Mult::Zero) => {
                                    // ERASED: do not compile the argument at all.
                                    env2.push(None);
                                }
                                _ => {
                                    // INLINE binding — no function boundary, so a
                                    // flat record stays in registers even through
                                    // generic (inlined) code.
                                    let v = self.compile_v(f, env, a)?;
                                    env2.push(Some(v));
                                }
                            }
                            // advance the type's telescope alongside the body.
                            head_ty = match head_ty {
                                Some(Term::Pi(_, _dom, cod)) => Some(strip_ann(cod)),
                                other => other,
                            };
                            body = strip_ann(inner);
                        }
                        _ => return Err("application of a non-function in runtime code".into()),
                    }
                }
                self.compile_v(f, &env2, body)
            }
            Term::Const(c) => self.compile_postulate(f, env, c, &[]),
            other => Err(format!("not a runtime value: {other:?}")),
        }
    }

    /// Does any method of this eliminator REFERENCE one of its own induction-
    /// hypothesis binders? (Method shape: `λ^nargs. λ^nrec. λ^ndep. body`; the IH
    /// binders sit between the fields and the deps.)
    fn elim_methods_use_ih(&self, data: &str, methods: &[Term], ndep: usize) -> Result<bool, String> {
        let decl = self
            .sig
            .data(data)
            .ok_or_else(|| format!("elim on unknown datatype `{data}`"))?;
        for (ci, ctor) in decl.ctors.iter().enumerate() {
            let lay = ctor_layout(self.sig, data, &ctor.name)?;
            let nrec = lay.arg_recursive.iter().filter(|b| **b).count();
            if nrec == 0 {
                continue;
            }
            let nargs = ctor.args.len();
            let mut body = strip_ann(&methods[ci]);
            let mut peeled = 0;
            while peeled < nargs + nrec + ndep {
                match body {
                    Term::Lam(inner) => {
                        body = strip_ann(inner);
                        peeled += 1;
                    }
                    _ => break,
                }
            }
            if peeled < nargs + nrec {
                continue; // a refuted-arm sentinel — no IHs bound
            }
            let kdep = peeled - nargs - nrec;
            // IH i has de Bruijn index kdep + (nrec-1-i) at the body's top.
            let hit = std::cell::Cell::new(false);
            crate::dep::map_vars(body, 0, &|i, depth| {
                if i >= depth {
                    let rel = i - depth;
                    if rel >= kdep && rel < kdep + nrec {
                        hit.set(true);
                    }
                }
                Term::Var(i)
            });
            if hit.get() {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// A CONVOY FOLD `(Elim …) dep₁ … dep_d` whose methods use their IHs: compile
    /// to a native recursive helper `g(node, deps…, live…)` — a tag-switch whose
    /// recursive calls (`ih(newdeps)`) descend on the matched field with the NEW
    /// deps, threading the captured live environment unchanged. This is the exact
    /// recursion a C programmer writes for the same dependent fold; the deps are
    /// parameters because they legitimately VARY per level (substituting them, as
    /// the commute does for non-recursive matches, would freeze them).
    fn compile_elim_dep_fold(
        &self,
        f: FunctionValue<'c>,
        env: &[Slot<'c>],
        data: &str,
        methods: &[Term],
        scrut: &Term,
        deps: &[&Term],
    ) -> Result<IntValue<'c>, String> {
        let decl = self
            .sig
            .data(data)
            .ok_or_else(|| format!("elim on unknown datatype `{data}`"))?
            .clone();
        let nd = deps.len();
        let scrut_val = self.compile(f, env, scrut)?;
        // dep slots are one scalar each (index-refined Nat-family values); a
        // record dependent is a guided hard error, never an implicit box.
        let dep_vals: Vec<IntValue<'c>> = deps
            .iter()
            .map(|a| {
                let v = self.compile_v(f, env, a)?;
                self.expect_width(v, 1, "a convoy-fold dependent")?
                    .as_int("convoy-fold dep")
            })
            .collect::<Result<_, _>>()?;
        let live: Vec<(usize, Val<'c>)> =
            env.iter().enumerate().filter_map(|(i, s)| s.clone().map(|v| (i, v))).collect();
        let live_idx: Vec<usize> = live.iter().map(|(i, _)| *i).collect();
        let flat_live: usize = live.iter().map(|(_, v)| v.width() as usize).sum();

        let nparams = 1 + nd + flat_live;
        let param_tys: Vec<inkwell::types::BasicMetadataTypeEnum> =
            vec![self.i64t.into(); nparams];
        let fname = self.fresh(&format!("tally_fold_{data}"));
        let helper = self
            .module
            .add_function(&fname, self.i64t.fn_type(&param_tys, false), None);

        let saved = self.builder.get_insert_block().unwrap();
        // inner env: the outer SHAPE with live slots bound to helper params, so
        // every level-identified reference (markers, captured vars) threads.
        let mut inner_env: Vec<Slot<'c>> = env.iter().map(|_| None).collect();
        let mut pk = (1 + nd) as u32;
        for (i, v) in &live {
            let w = v.width();
            if w == 1 {
                inner_env[*i] =
                    Some(Val::Int(helper.get_nth_param(pk).unwrap().into_int_value()));
            } else {
                let comps: Vec<IntValue<'c>> = (0..w)
                    .map(|k| helper.get_nth_param(pk + k).unwrap().into_int_value())
                    .collect();
                inner_env[*i] = Some(Val::Agg(comps));
            }
            pk += w;
        }
        let node = helper.get_nth_param(0).unwrap().into_int_value();

        let entry = self.ctx.append_basic_block(helper, "entry");
        self.builder.position_at_end(entry);
        let node_ptr = self.builder.build_int_to_ptr(node, self.ptr, "fold.node").unwrap();
        let tag = self.load(node_ptr, 0, "fold.tag");
        let default = self.ctx.append_basic_block(helper, "fold.default");
        let arm_blocks: Vec<_> = decl
            .ctors
            .iter()
            .map(|c| self.ctx.append_basic_block(helper, &format!("fold.{}", c.name)))
            .collect();
        let cases: Vec<(IntValue<'c>, inkwell::basic_block::BasicBlock<'c>)> = decl
            .ctors
            .iter()
            .enumerate()
            .map(|(ci, _)| (self.i64t.const_int(ci as u64, false), arm_blocks[ci]))
            .collect();
        self.builder.build_switch(tag, default, &cases).unwrap();
        self.builder.position_at_end(default);
        self.builder.build_unreachable().unwrap();

        for (ci, ctor) in decl.ctors.iter().enumerate() {
            self.builder.position_at_end(arm_blocks[ci]);
            let lay = ctor_layout(self.sig, data, &ctor.name)?;
            let mut menv = inner_env.clone();
            for ai in 0..ctor.args.len() {
                let v: Slot<'c> =
                    lay.arg_slot[ai].map(|sl| self.load_field(node_ptr, sl, lay.arg_width[ai]));
                menv.push(v);
            }
            // IH markers: level-identified recursion points (no eager fold).
            let mut nmarks = 0;
            for ai in 0..ctor.args.len() {
                if lay.arg_recursive[ai] {
                    let field_level = inner_env.len() + ai;
                    self.elim_fold_ihs.borrow_mut().push(ElimFoldIh {
                        ih_level: menv.len(),
                        func: helper,
                        field_level,
                        live_idx: live_idx.clone(),
                        ndep: nd,
                    });
                    menv.push(None);
                    nmarks += 1;
                }
            }
            let nrec = nmarks;
            // peel the method: fields + IHs strictly, then up to nd dep binders
            // (a refuted-arm sentinel binds none).
            let mut body = strip_ann(&methods[ci]);
            for _ in 0..(ctor.args.len() + nrec) {
                match body {
                    Term::Lam(inner) => body = strip_ann(inner),
                    _ => {
                        return Err(format!(
                            "convoy fold[{data}]: method for `{}` is not a {}-binder function",
                            ctor.name,
                            ctor.args.len() + nrec
                        ))
                    }
                }
            }
            let mut kdep = 0;
            while kdep < nd {
                match body {
                    Term::Lam(inner) => {
                        body = strip_ann(inner);
                        kdep += 1;
                    }
                    _ => break,
                }
            }
            for j in 0..kdep {
                menv.push(Some(Val::Int(
                    helper.get_nth_param((1 + j) as u32).unwrap().into_int_value(),
                )));
            }
            let result = self
                .compile_v(helper, &menv, body)
                .and_then(|v| {
                    self.expect_width(
                        v,
                        1,
                        "a convoy fold's result (make the function %partial — Fix \
                         returns records by value)",
                    )
                })
                .and_then(|v| v.as_int("convoy-fold result"));
            for _ in 0..nmarks {
                self.elim_fold_ihs.borrow_mut().pop();
            }
            let result = result?;
            self.builder.build_return(Some(&result)).unwrap();
        }

        self.builder.position_at_end(saved);
        let mut ca: Vec<inkwell::values::BasicMetadataValueEnum> = vec![scrut_val.into()];
        for v in &dep_vals {
            ca.push((*v).into());
        }
        for (_, v) in &live {
            for c in v.components() {
                ca.push(c.into());
            }
        }
        Ok(self
            .builder
            .build_call(helper, &ca, "fold.call")
            .unwrap()
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_int_value())
    }

    /// Case-commuting conversion for a CONVOY application `(Case/Elim …) a₁ … a_k`:
    /// rewrite each method `λ(telescope). λdep₁…λdep_k. body` to
    /// `λ(telescope). body[depⱼ := aⱼ]` so the result is a plain `Case`/`Elim`.
    /// A REFUTED arm's method (the convoy's `Nat`-sentinel dead code) binds NO
    /// deps — it is kept as-is; the scrutinee's type guarantees it never runs.
    fn commute_apply_into_methods(
        &self,
        data: &str,
        methods: &[Term],
        args: &[&Term],
        with_ih: bool,
    ) -> Result<Vec<Term>, String> {
        let decl = self
            .sig
            .data(data)
            .ok_or_else(|| format!("case on unknown datatype `{data}`"))?;
        let k = args.len();
        let mut out = Vec::with_capacity(methods.len());
        for (ci, ctor) in decl.ctors.iter().enumerate() {
            let lay = ctor_layout(self.sig, data, &ctor.name)?;
            let nb = ctor.args.len()
                + if with_ih {
                    lay.arg_recursive.iter().filter(|b| **b).count()
                } else {
                    0
                };
            let mut body = strip_ann(&methods[ci]);
            for _ in 0..nb {
                match body {
                    Term::Lam(inner) => body = strip_ann(inner),
                    _ => {
                        return Err(format!(
                            "case[{data}]: method for `{}` is not a {nb}-argument function",
                            ctor.name
                        ))
                    }
                }
            }
            // peel up to k dep-lambdas (a refuted arm's sentinel has none).
            let mut kp = 0;
            while kp < k {
                match body {
                    Term::Lam(inner) => {
                        body = strip_ann(inner);
                        kp += 1;
                    }
                    _ => break,
                }
            }
            // `body` sits under `nb` telescope binders + `kp` dep binders
            // (innermost). Substitute the dep binders with the (shifted) applied
            // arguments and close the `kp`-binder gap for everything else.
            let new_body = crate::dep::map_vars(body, 0, &|i, depth| {
                if i < depth {
                    return Term::Var(i);
                }
                let rel = i - depth;
                if rel < kp {
                    crate::dep::shift_term(nb + depth, args[kp - 1 - rel])
                } else {
                    Term::Var(i - kp)
                }
            });
            let mut m = new_body;
            for _ in 0..nb {
                m = Term::Lam(Box::new(m));
            }
            out.push(m);
        }
        Ok(out)
    }

    /// Commute an application into one `NatCase` method (the Nat convoy — see
    /// `commute_apply_into_methods` for the datatype version): peel the method's
    /// `nb` telescope binders (0 for zero, 1 — the predecessor — for successor)
    /// and its dep-lambdas, substitute the applied dep VARIABLES into the body,
    /// and re-wrap the telescope binders. Sound because exactly one arm runs and
    /// the deps are already-computed values.
    fn commute_apply_into_natcase_method(
        &self,
        method: &Term,
        nb: usize,
        args: &[&Term],
    ) -> Result<Term, String> {
        let mut body = strip_ann(method);
        for _ in 0..nb {
            match body {
                Term::Lam(inner) => body = strip_ann(inner),
                _ => {
                    return Err(format!(
                        "NatCase convoy: method is not a {nb}-argument function"
                    ))
                }
            }
        }
        let k = args.len();
        let mut kp = 0;
        while kp < k {
            match body {
                Term::Lam(inner) => {
                    body = strip_ann(inner);
                    kp += 1;
                }
                _ => break,
            }
        }
        let new_body = crate::dep::map_vars(body, 0, &|i, depth| {
            if i < depth {
                return Term::Var(i);
            }
            let rel = i - depth;
            if rel < kp {
                crate::dep::shift_term(nb + depth, args[kp - 1 - rel])
            } else {
                Term::Var(i - kp)
            }
        });
        let mut m = new_body;
        for _ in 0..nb {
            m = Term::Lam(Box::new(m));
        }
        Ok(m)
    }

    /// `malloc(nslots * 8)` and return the raw pointer.
    fn malloc_cell(&self, nslots: u32, name: &str) -> PointerValue<'c> {
        let sz = self.i64t.const_int(nslots as u64 * 8, false);
        self.builder
            .build_call(self.malloc, &[sz.into()], name)
            .unwrap()
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_pointer_value()
    }
    fn store(&self, p: PointerValue<'c>, idx: u32, v: IntValue<'c>) {
        self.builder
            .build_store(self.gep(p, idx, "st"), v)
            .unwrap();
    }
    fn toptr(&self, v: IntValue<'c>, name: &str) -> PointerValue<'c> {
        self.builder.build_int_to_ptr(v, self.ptr, name).unwrap()
    }
    fn toint(&self, p: PointerValue<'c>, name: &str) -> IntValue<'c> {
        self.builder.build_ptr_to_int(p, self.i64t, name).unwrap()
    }
    fn free_int(&self, v: IntValue<'c>) {
        let p = self.toptr(v, "fp");
        self.builder.build_call(self.free, &[p.into()], "").unwrap();
    }

    /// Build a boxed cell for a single-constructor datatype `data`/`cname`
    /// (e.g. the linear pairs `CL`/`VL`): slot 0 = tag, then the non-erased
    /// constructor arguments in declaration order. `field_vals` must line up
    /// with the constructor's NON-erased arguments (its runtime fields).
    fn box_single_ctor(
        &self,
        data: &str,
        cname: &str,
        field_vals: &[IntValue<'c>],
    ) -> Result<Val<'c>, String> {
        let lay = ctor_layout(self.sig, data, cname)?;
        if field_vals.len() as u32 != lay.nfields {
            return Err(format!(
                "{data}.{cname}: native impl produced {} field(s), layout expects {}",
                field_vals.len(),
                lay.nfields
            ));
        }
        // A FLAT RECORD (all these prelude pairs are): no cell, no tag — the
        // fields ARE the value, in registers. The read-back pairs (`ARead`,
        // `Cell`, `CL`, `VL`, `Taken`) stop costing a malloc even at -O0.
        if record_width(self.sig, data).is_some() {
            return Ok(Val::Agg(field_vals.to_vec()));
        }
        let raw = self.malloc_cell(1 + lay.nfields, "box");
        self.store(raw, 0, self.i64t.const_int(lay.tag, false));
        // arg_slot lists slots for ALL ctor args (None = erased); the i-th
        // non-erased slot gets field_vals[i].
        let mut fi = 0usize;
        for slot in &lay.arg_slot {
            if let Some(s) = slot {
                self.store(raw, *s, field_vals[fi]);
                fi += 1;
            }
        }
        Ok(Val::Int(self.toint(raw, "boxint")))
    }

    /// Collect the RUNTIME (non-erased) arguments of a postulate spine, in
    /// order, by walking the postulate's Pi telescope: a `Π[0]` binder erases
    /// its argument (no slot, not even evaluated); `Π[1]`/`Π[ω]` keep it. This
    /// is the erasure that makes the memory layer zero-overhead — type/region
    /// indices and proofs never become instructions.
    fn runtime_args(
        &self,
        f: FunctionValue<'c>,
        env: &[Slot<'c>],
        cname: &str,
        args: &[&Term],
    ) -> Result<Vec<IntValue<'c>>, String> {
        // the SCALAR collection, for primitives whose runtime arguments are all
        // one slot (pointers, views, chars, list elements). A record reaching
        // one is a guided hard error — never an implicit box.
        self.runtime_args_v(f, env, cname, args)?
            .into_iter()
            .map(|v| {
                self.expect_width(v, 1, &format!("`{cname}`'s argument"))?
                    .as_int(cname)
            })
            .collect()
    }

    /// The runtime (non-erased) arguments as `Val`s, widths untouched — the
    /// width-aware primitives (alloc/unbox, the view ops, the array ops) decide
    /// their own layouts from the ERASED TYPE ARGUMENTS in the spine.
    fn runtime_args_v(
        &self,
        f: FunctionValue<'c>,
        env: &[Slot<'c>],
        cname: &str,
        args: &[&Term],
    ) -> Result<Vec<Val<'c>>, String> {
        let ty = self
            .sig
            .postulate(cname)
            .ok_or_else(|| format!("unknown postulate `{cname}`"))?;
        let mut cur = strip_ann(ty);
        let mut out = Vec::new();
        for a in args {
            match cur {
                Term::Pi(mult, _dom, cod) => {
                    if *mult != Mult::Zero {
                        out.push(self.compile_v(f, env, a)?);
                    }
                    // else: erased argument — never compiled, never stored.
                    cur = strip_ann(cod);
                }
                _ => {
                    return Err(format!(
                        "postulate `{cname}` applied to more arguments than its type allows"
                    ))
                }
            }
        }
        Ok(out)
    }

    /// The width of the erased TYPE argument at spine position `i` for a
    /// width-sensitive primitive. A FLEX (instantiation-dependent) type here is
    /// a hard error: the primitive is compiled once per call site, and inside a
    /// generic `%partial` function the layout genuinely is not known — tallyc
    /// will not guess and will not box behind your back.
    fn spine_type_width(&self, cname: &str, args: &[&Term], i: usize) -> Result<u32, String> {
        let t = args.get(i).ok_or_else(|| {
            format!("`{cname}`: missing its erased type argument (spine too short)")
        })?;
        if type_is_flex(self.sig, t) {
            return Err(format!(
                "`{cname}`: the payload/element type here is ABSTRACT (a type \
                 variable of a generic `%partial` function), so its layout is \
                 unknown at compile time — and tallyc never boxes implicitly. \
                 Specialize the function to a concrete payload type (a generic \
                 fn that is NOT `%partial` is inlined per call site, where the \
                 type is concrete, and works as-is).",
            ));
        }
        Ok(type_width(self.sig, t))
    }

    /// Native implementations of the known memory postulates. Erased
    /// (multiplicity-0) arguments are dropped here; only the genuine runtime
    /// values survive. Any postulate without a native impl that is reached in a
    /// runtime position is a hard error (the abstract postulate cannot run).
    fn compile_postulate(
        &self,
        f: FunctionValue<'c>,
        env: &[Slot<'c>],
        cname: &str,
        args: &[&Term],
    ) -> Result<Val<'c>, String> {
        // DLL node layout: an intrusive CIRCULAR doubly-linked list with a
        // sentinel (reused from the non-dependent backend). slot 0 = next,
        // 1 = prev, 2 = elem. The list value IS the sentinel; a cursor IS a
        // real node pointer.
        const NEXT: u32 = 0;
        const PREV: u32 = 1;
        const ELEM: u32 = 2;

        match cname {
            // alloc : {0 a} -> a -> Own a  — Own is erased to a bare pointer:
            // malloc one slot, store the payload, hand back the pointer.
            "alloc" => {
                // `alloc` IS the explicit box: the Own cell holds the payload
                // INLINE at its real width — `Own Point` is one 16-byte cell
                // holding {x, y}, not a pointer to a second cell.
                let rt = self.runtime_args_v(f, env, cname, args)?;
                let [payload] = rt.as_slice() else {
                    return Err(format!(
                        "alloc: expected 1 runtime argument (the payload), got {}",
                        rt.len()
                    ));
                };
                let comps = payload.components();
                let cell = self.malloc_cell(comps.len() as u32, "own");
                for (i, c) in comps.iter().enumerate() {
                    self.store(cell, i as u32, *c);
                }
                Ok(Val::Int(self.toint(cell, "ownint")))
            }
            // free : ... -> (1 o : Own a / List r) -> Unit  — libc free of the
            // (single) linear pointer argument; Unit is represented as 0.
            "free" => {
                let rt = self.runtime_args(f, env, cname, args)?;
                let [ptr] = rt.as_slice() else {
                    return Err(format!(
                        "free: expected 1 runtime argument (the owned pointer), got {}",
                        rt.len()
                    ));
                };
                self.free_int(*ptr);
                Ok(Val::Int(self.i64t.const_zero()))
            }
            // unbox : {0 a} -> (1 o : Own a) -> a  — deref-and-CONSUME: load the
            // pointee from the cell, FREE the cell, return the pointee (moved out). The
            // linear `o` is consumed exactly once; its linear fields (e.g. a `tail`
            // Own) are moved to the caller (threaded onward by the eliminator). This is
            // the sound primitive for a consuming free-traversal of a linked structure.
            "unbox" => {
                // the payload's width comes from the erased type argument (the
                // spine's `a`); an abstract `a` is a hard error — the cell's
                // layout is instantiation-dependent (`spine_type_width`).
                let w = self.spine_type_width(cname, args, 0)?;
                let rt = self.runtime_args(f, env, cname, args)?;
                let [ptr] = rt.as_slice() else {
                    return Err(format!(
                        "unbox: expected 1 runtime argument (the owned pointer), got {}",
                        rt.len()
                    ));
                };
                let cell = self.toptr(*ptr, "ownp");
                let payload = self.load_field(cell, 0, w);
                self.free_int(*ptr); // free the Own cell; the payload was moved out
                Ok(payload)
            }
            // print : Nat -> Unit  — the one observable effect: write the number
            // and a newline to stdout (`printf("%lld\n", n)`). Sequenced by the
            // CBV `let` like any other effect; the kernel sees an opaque postulate
            // (it can never reduce in a type). Unit is represented as 0.
            "print" => {
                let rt = self.runtime_args(f, env, cname, args)?;
                let [x] = rt.as_slice() else {
                    return Err(format!(
                        "print: expected 1 runtime argument (the Nat), got {}",
                        rt.len()
                    ));
                };
                let printf = self.module.get_function("printf").unwrap_or_else(|| {
                    self.module.add_function(
                        "printf",
                        self.ctx.i32_type().fn_type(&[self.ptr.into()], true),
                        None,
                    )
                });
                let fmt = match self.module.get_global("tally.print.fmt") {
                    Some(g) => g.as_pointer_value(),
                    None => self
                        .builder
                        .build_global_string_ptr("%lld\n", "tally.print.fmt")
                        .map_err(|e| format!("print: cannot emit format string: {e}"))?
                        .as_pointer_value(),
                };
                self.builder
                    .build_call(printf, &[fmt.into(), (*x).into()], "print")
                    .unwrap();
                Ok(Val::Int(self.i64t.const_zero()))
            }
            // prints : Str -> Unit — write the string and a newline to stdout.
            "prints" => {
                let rt = self.runtime_args(f, env, cname, args)?;
                let [x] = rt.as_slice() else {
                    return Err(format!(
                        "prints: expected 1 runtime argument (the Str), got {}",
                        rt.len()
                    ));
                };
                let printf = self.module.get_function("printf").unwrap_or_else(|| {
                    self.module.add_function(
                        "printf",
                        self.ctx.i32_type().fn_type(&[self.ptr.into()], true),
                        None,
                    )
                });
                let fmt = match self.module.get_global("tally.prints.fmt") {
                    Some(g) => g.as_pointer_value(),
                    None => self
                        .builder
                        .build_global_string_ptr("%s\n", "tally.prints.fmt")
                        .map_err(|e| format!("prints: cannot emit format string: {e}"))?
                        .as_pointer_value(),
                };
                let sptr = self.builder.build_int_to_ptr(*x, self.ptr, "prints.p").unwrap();
                self.builder
                    .build_call(printf, &[fmt.into(), sptr.into()], "prints")
                    .unwrap();
                Ok(Val::Int(self.i64t.const_zero()))
            }
            // putc : Nat -> Unit — write ONE byte to stdout (libc putchar; the
            // Nat is truncated to the C int, i.e. the low byte of a char code).
            "putc" => {
                let rt = self.runtime_args(f, env, cname, args)?;
                let [c] = rt.as_slice() else {
                    return Err(format!(
                        "putc: expected 1 runtime argument (the char code), got {}",
                        rt.len()
                    ));
                };
                let i32t = self.ctx.i32_type();
                let putchar = self.module.get_function("putchar").unwrap_or_else(|| {
                    self.module
                        .add_function("putchar", i32t.fn_type(&[i32t.into()], false), None)
                });
                let c32 = self
                    .builder
                    .build_int_truncate(*c, i32t, "putc.c")
                    .unwrap();
                self.builder
                    .build_call(putchar, &[c32.into()], "putc")
                    .unwrap();
                Ok(Val::Int(self.i64t.const_zero()))
            }
            // getc : Unit -> Nat — read ONE byte from stdin (libc getchar) and
            // return `Succ(byte)`, or `Zero` at EOF. Nat cannot be negative, so
            // C's `-1` sentinel becomes the `Zero` constructor and the byte is
            // the predecessor: `match getc(U) { Zero => done, Succ(c) => … }`
            // both tests for EOF and binds the character, in one native branch.
            "getc" => {
                let rt = self.runtime_args(f, env, cname, args)?;
                let [_u] = rt.as_slice() else {
                    return Err(format!(
                        "getc: expected 1 runtime argument (the Unit), got {}",
                        rt.len()
                    ));
                };
                let i32t = self.ctx.i32_type();
                let getchar = self.module.get_function("getchar").unwrap_or_else(|| {
                    self.module.add_function("getchar", i32t.fn_type(&[], false), None)
                });
                let r = self
                    .builder
                    .build_call(getchar, &[], "getc")
                    .unwrap()
                    .try_as_basic_value()
                    .left()
                    .unwrap()
                    .into_int_value();
                let r64 = self
                    .builder
                    .build_int_s_extend(r, self.i64t, "getc.w")
                    .unwrap();
                let eof = self
                    .builder
                    .build_int_compare(
                        inkwell::IntPredicate::SLT,
                        r64,
                        self.i64t.const_zero(),
                        "getc.eof",
                    )
                    .unwrap();
                let succ = self
                    .builder
                    .build_int_add(r64, self.i64t.const_int(1, false), "getc.succ")
                    .unwrap();
                let out = self
                    .builder
                    .build_select(eof, self.i64t.const_zero(), succ, "getc.out")
                    .unwrap()
                    .into_int_value();
                Ok(Val::Int(out))
            }
            // new : {0 r} -> List r  — create an empty circular sentinel list.
            "new" => {
                let s = self.malloc_cell(3, "sentinel");
                let si = self.toint(s, "si");
                self.store(s, NEXT, si); // s.next = s
                self.store(s, PREV, si); // s.prev = s
                Ok(Val::Int(si))
            }
            // insert : {0 r} -> (1 l : List r) -> Nat -> CL r
            // append a node at the tail; return the boxed pair (cursor, list).
            "insert" => {
                let rt = self.runtime_args(f, env, cname, args)?;
                let [li, x] = rt.as_slice() else {
                    return Err(format!(
                        "insert: expected 2 runtime arguments (list, value), got {}",
                        rt.len()
                    ));
                };
                let s = self.toptr(*li, "s");
                let node = self.malloc_cell(3, "node");
                let ni = self.toint(node, "ni");
                let prev = self.load(s, PREV, "sp"); // old tail
                let prevp = self.toptr(prev, "pp");
                self.store(node, ELEM, *x);
                self.store(node, PREV, prev); // node.prev = old tail
                self.store(node, NEXT, *li); // node.next = sentinel
                self.store(prevp, NEXT, ni); // old_tail.next = node
                self.store(s, PREV, ni); // sentinel.prev = node
                self.box_single_ctor("CL", "MkCL", &[ni, *li])
            }
            // remove : {0 r} -> (1 c : Cursor r) -> (1 l : List r) -> VL r
            // O(1) unlink of the node by its handle; return (value, list).
            "remove" => {
                let rt = self.runtime_args(f, env, cname, args)?;
                let [ci, li] = rt.as_slice() else {
                    return Err(format!(
                        "remove: expected 2 runtime arguments (cursor, list), got {}",
                        rt.len()
                    ));
                };
                let node = self.toptr(*ci, "c");
                let prev = self.load(node, PREV, "np");
                let next = self.load(node, NEXT, "nx");
                let prevp = self.toptr(prev, "pp");
                let nextp = self.toptr(next, "xp");
                self.store(prevp, NEXT, next); // prev.next = next
                self.store(nextp, PREV, prev); // next.prev = prev
                let elem = self.load(node, ELEM, "elem");
                self.free_int(*ci); // O(1) free of the node
                self.box_single_ctor("VL", "MkVL", &[elem, *li])
            }
            // ---- the view layer (docs/02): the L3 address/permission split ----
            // A location is erased; `Ptr l` and the linear view `PtsTo l a` are
            // BOTH represented as the raw cell address (linearity keeps the view
            // single-use, so no stale one can exist). See docs/VIEW_LAYER_PLAN.md.
            //
            // valloc : {0 a} -> a -> Cell a  = ∃l. (Ptr l ⊗ a@l). malloc one slot,
            // store the payload, pack the address as BOTH the pointer and the view.
            "valloc" => {
                let rt = self.runtime_args_v(f, env, cname, args)?;
                let [payload] = rt.as_slice() else {
                    return Err(format!(
                        "valloc: expected 1 runtime argument (the payload), got {}",
                        rt.len()
                    ));
                };
                let comps = payload.components();
                let cell = self.malloc_cell(comps.len().max(1) as u32, "cell");
                for (i, c) in comps.iter().enumerate() {
                    self.store(cell, i as u32, *c);
                }
                // `Cell a` = { Ptr l (the address), PtsTo l a (ZERO-WIDTH) }:
                // the record collapses to the bare address. The view exists only
                // in the checker.
                Ok(Val::Int(self.toint(cell, "addr")))
            }
            // vwrite : ... -> Ptr l -> (1 v : PtsTo l a) -> b -> PtsTo l b
            // STRONG UPDATE: store the new payload at the cell, hand back the
            // (same) address retyped as `b@l`. The old view is consumed.
            "vwrite" => {
                // strong update at real widths. The allocation's size is the
                // ORIGINAL valloc width, which the view types don't carry — so a
                // width-CHANGING strong update is rejected (it could overflow
                // the cell); same-width retyping (incl. record → record) is fine.
                let w_a = self.spine_type_width(cname, args, 0)?;
                let w_b = self.spine_type_width(cname, args, 1)?;
                if w_a != w_b {
                    return Err(format!(
                        "vwrite: a strong update must preserve the slot's LAYOUT \
                         SIZE (old {w_a} slot(s), new {w_b}) — the cell was \
                         allocated at the old width. valloc a new cell instead."
                    ));
                }
                let rt = self.runtime_args_v(f, env, cname, args)?;
                let [p, _v, new] = rt.as_slice() else {
                    return Err(format!(
                        "vwrite: expected 3 runtime arguments (ptr, view, value), got {}",
                        rt.len()
                    ));
                };
                let pv = p.as_int("vwrite ptr")?;
                let newv = self.expect_width(new.clone(), w_b, "vwrite payload")?;
                let cell = self.toptr(pv, "cellp");
                for (i, c) in newv.components().iter().enumerate() {
                    self.store(cell, i as u32, *c);
                }
                // the new view: zero-width — the store above IS the whole runtime.
                Ok(Val::Agg(Vec::new()))
            }
            // vtake : ... -> Ptr l -> (1 v : PtsTo l a) -> Taken a l
            // MOVE the payload out, retyping the slot as `Hole` (the cell is NOT
            // freed and NOT read again as `a` — the returned view is `PtsTo l Hole`,
            // which must be refilled by `vwrite` or reclaimed by `vfree`). Sound for
            // ANY payload — unlike a copying borrow, which would need `a` copyable.
            "vtake" => {
                let w = self.spine_type_width(cname, args, 0)?;
                let rt = self.runtime_args_v(f, env, cname, args)?;
                let [p, _v] = rt.as_slice() else {
                    return Err(format!(
                        "vtake: expected 2 runtime arguments (ptr, view), got {}",
                        rt.len()
                    ));
                };
                let cell = self.toptr(p.as_int("vtake ptr")?, "cellp");
                // the slot's bits are now stale but UNTYPEABLE as `a` (the view is
                // `PtsTo l (Hole a)`); no store needed. `Taken` = { payload,
                // zero-width hole view } — collapses to the moved payload.
                let payload = self.load_field(cell, 0, w);
                Ok(payload)
            }
            // vread : ... -> Ptr l -> (1 v : PtsTo l a) -> a
            // Destructive read: load the payload, CONSUME the view, and reclaim
            // the cell. (A borrowing read that returns the view is a later slice.)
            "vread" => {
                let w = self.spine_type_width(cname, args, 0)?;
                let rt = self.runtime_args_v(f, env, cname, args)?;
                let [p, _v] = rt.as_slice() else {
                    return Err(format!(
                        "vread: expected 2 runtime arguments (ptr, view), got {}",
                        rt.len()
                    ));
                };
                let pv = p.as_int("vread ptr")?;
                let cell = self.toptr(pv, "cellp");
                let payload = self.load_field(cell, 0, w);
                self.free_int(pv);
                Ok(payload)
            }
            // vfree : ... -> Ptr l -> (1 v : PtsTo l a) -> Unit
            // consume the view and reclaim the cell; Unit is represented as 0.
            "vfree" => {
                let rt = self.runtime_args_v(f, env, cname, args)?;
                let [p, _v] = rt.as_slice() else {
                    return Err(format!(
                        "vfree: expected 2 runtime arguments (ptr, view), got {}",
                        rt.len()
                    ));
                };
                self.free_int(p.as_int("vfree ptr")?);
                Ok(Val::Int(self.i64t.const_zero()))
            }
            // ---- contiguous arrays (the C `a[i]` on a flat buffer) ----
            // `Arr a n` IS one flat `n`-slot block: `malloc(n*8)`, no header, no
            // per-element cell, no stored length (the length is the erased index).
            //
            // anew : {0 a} -> (n : Nat) -> a -> Arr a n — one malloc, filled with
            // the initial element by a native counting loop.
            "anew" => {
                // the element width `w` comes from the erased type argument: an
                // `Arr Point n` is one flat block of `n * w` slots — REAL AoS,
                // records stored inline at stride `w`. (An abstract element
                // type is a hard error — `spine_type_width`.)
                let w = self.spine_type_width(cname, args, 0)?;
                let rt = self.runtime_args_v(f, env, cname, args)?;
                let [n, init] = rt.as_slice() else {
                    return Err(format!(
                        "anew: expected 2 runtime arguments (length, initial element), got {}",
                        rt.len()
                    ));
                };
                let n = n.as_int("anew length")?;
                let n = &n;
                let init = self.expect_width(init.clone(), w, "anew initial element")?;
                let sz = self
                    .builder
                    .build_int_mul(*n, self.i64t.const_int(8 * w as u64, false), "arr.sz")
                    .unwrap();
                let raw = self
                    .builder
                    .build_call(self.malloc, &[sz.into()], "arr")
                    .unwrap()
                    .try_as_basic_value()
                    .left()
                    .unwrap()
                    .into_pointer_value();
                // for i in 0..n { arr[i] = init }
                let entry = self.builder.get_insert_block().unwrap();
                let head = self.ctx.append_basic_block(f, "arr.fill");
                let body = self.ctx.append_basic_block(f, "arr.fill.body");
                let done = self.ctx.append_basic_block(f, "arr.fill.done");
                self.builder.build_unconditional_branch(head).unwrap();
                self.builder.position_at_end(head);
                let i = self.builder.build_phi(self.i64t, "arr.i").unwrap();
                i.add_incoming(&[(&self.i64t.const_zero(), entry)]);
                let iv = i.as_basic_value().into_int_value();
                let more = self
                    .builder
                    .build_int_compare(inkwell::IntPredicate::ULT, iv, *n, "arr.more")
                    .unwrap();
                self.builder.build_conditional_branch(more, body, done).unwrap();
                self.builder.position_at_end(body);
                let base = self
                    .builder
                    .build_int_mul(iv, self.i64t.const_int(w as u64, false), "arr.base")
                    .unwrap();
                for (k, c) in init.components().iter().enumerate() {
                    let off = self
                        .builder
                        .build_int_add(base, self.i64t.const_int(k as u64, false), "arr.off")
                        .unwrap();
                    let slot = unsafe {
                        self.builder
                            .build_gep(self.i64t, raw, &[off], "arr.slot")
                            .unwrap()
                    };
                    self.builder.build_store(slot, *c).unwrap();
                }
                let inext = self
                    .builder
                    .build_int_add(iv, self.i64t.const_int(1, false), "arr.inext")
                    .unwrap();
                i.add_incoming(&[(&inext, body)]);
                self.builder.build_unconditional_branch(head).unwrap();
                self.builder.position_at_end(done);
                Ok(Val::Int(self.toint(raw, "arrint")))
            }
            // aget : … -> (i : Nat) -> (0 p : Lt i n) -> (1 arr : Arr a n) -> ARead a n
            // THE C-level read: a bare indexed load. The bound is an ERASED proof
            // (multiplicity 0 — `runtime_args` never compiles it), so there is no
            // bounds compare and no branch. The `ARead` pair threads the linear
            // array back to the caller; -O2 scalarizes it away.
            "aget" => {
                let w = self.spine_type_width(cname, args, 0)?;
                let rt = self.runtime_args(f, env, cname, args)?;
                let [i, arr] = rt.as_slice() else {
                    return Err(format!(
                        "aget: expected 2 runtime arguments (index, array), got {}",
                        rt.len()
                    ));
                };
                let base = self.toptr(*arr, "arr.p");
                let off = self
                    .builder
                    .build_int_mul(*i, self.i64t.const_int(w as u64, false), "arr.off")
                    .unwrap();
                let mut comps: Vec<IntValue<'c>> = Vec::with_capacity(w as usize + 1);
                for k in 0..w {
                    let ko = self
                        .builder
                        .build_int_add(off, self.i64t.const_int(k as u64, false), "arr.ko")
                        .unwrap();
                    let slot = unsafe {
                        self.builder
                            .build_gep(self.i64t, base, &[ko], "arr.slot")
                            .unwrap()
                    };
                    comps.push(
                        self.builder
                            .build_load(self.i64t, slot, "arr.elem")
                            .unwrap()
                            .into_int_value(),
                    );
                }
                // ARead is a FLEX flat record [element…, array]: build it at the
                // element's true width; the match splits it back flexibly.
                comps.push(*arr);
                Ok(Val::Agg(comps))
            }
            // aset : … -> (i : Nat) -> (0 p : Lt i n) -> a -> (1 arr : Arr a n) -> Arr a n
            // a bare indexed store; the array value (the base pointer) is returned
            // unchanged — in-place mutation, justified by linearity (the input
            // array was consumed; this result is its sole continuation).
            "aset" => {
                let w = self.spine_type_width(cname, args, 0)?;
                let rt = self.runtime_args_v(f, env, cname, args)?;
                let [i, v, arr] = rt.as_slice() else {
                    return Err(format!(
                        "aset: expected 3 runtime arguments (index, element, array), got {}",
                        rt.len()
                    ));
                };
                let i = i.as_int("aset index")?;
                let arr = arr.as_int("aset array")?;
                let v = self.expect_width(v.clone(), w, "aset element")?;
                let base = self.toptr(arr, "arr.p");
                let off = self
                    .builder
                    .build_int_mul(i, self.i64t.const_int(w as u64, false), "arr.off")
                    .unwrap();
                for (k, c) in v.components().iter().enumerate() {
                    let ko = self
                        .builder
                        .build_int_add(off, self.i64t.const_int(k as u64, false), "arr.ko")
                        .unwrap();
                    let slot = unsafe {
                        self.builder
                            .build_gep(self.i64t, base, &[ko], "arr.slot")
                            .unwrap()
                    };
                    self.builder.build_store(slot, *c).unwrap();
                }
                Ok(Val::Int(arr))
            }
            // afree : … -> (1 arr : Arr a n) -> Unit — ONE libc free of the block.
            "afree" => {
                let rt = self.runtime_args(f, env, cname, args)?;
                let [arr] = rt.as_slice() else {
                    return Err(format!(
                        "afree: expected 1 runtime argument (the array), got {}",
                        rt.len()
                    ));
                };
                self.free_int(*arr);
                Ok(Val::Int(self.i64t.const_zero()))
            }
            // `%foreign` — an extern C symbol at the i64 ABI: declare it with one
            // i64 parameter per RUNTIME argument (erased Π[0] binders are dropped,
            // as everywhere) returning i64, and emit a direct call. The C side of
            // the boundary is trusted (the audited escape hatch); the kernel has
            // already checked every use against the declared tally type.
            other if self.sig.foreigns.contains_key(other) => {
                let sym = self.sig.foreigns.get(other).unwrap().clone();
                // records FLATTEN into consecutive i64 arguments — on AArch64 and
                // SysV x86-64 that is exactly the by-value ABI for a small
                // (≤ 2-register) integer-class C struct, so `%foreign` functions
                // taking such structs by value work directly.
                let ty = self
                    .sig
                    .postulate(other)
                    .ok_or_else(|| format!("unknown postulate `{other}`"))?;
                let mut cur = strip_ann(ty);
                let mut rt: Vec<IntValue<'c>> = Vec::new();
                for a in args {
                    match cur {
                        Term::Pi(mult, _dom, cod) => {
                            if *mult != Mult::Zero {
                                let v = self.compile_v(f, env, a)?;
                                rt.extend(v.components());
                            }
                            cur = strip_ann(cod);
                        }
                        _ => {
                            return Err(format!(
                                "postulate `{other}` applied to more arguments than its type allows"
                            ))
                        }
                    }
                }
                let func = match self.module.get_function(&sym) {
                    Some(g) => g,
                    None => {
                        let params: Vec<inkwell::types::BasicMetadataTypeEnum> =
                            rt.iter().map(|_| self.i64t.into()).collect();
                        self.module
                            .add_function(&sym, self.i64t.fn_type(&params, false), None)
                    }
                };
                if func.count_params() as usize != rt.len() {
                    return Err(format!(
                        "%foreign `{other}` (symbol `{sym}`): called with {} runtime \
                         argument(s) but the symbol was already declared with {} — one \
                         extern declaration per symbol",
                        rt.len(),
                        func.count_params()
                    ));
                }
                let call_args: Vec<inkwell::values::BasicMetadataValueEnum> =
                    rt.iter().map(|v| (*v).into()).collect();
                let ret = self
                    .builder
                    .build_call(func, &call_args, "ffi")
                    .unwrap()
                    .try_as_basic_value()
                    .left()
                    .ok_or_else(|| format!("%foreign `{other}`: call produced no value"))?
                    .into_int_value();
                Ok(Val::Int(ret))
            }
            // borrow : {0 a} -> (1 o : Own a) -> Borrowed a — split an Own into
            // {address, view, loan}. The view and loan are ZERO-WIDTH, so
            // `Borrowed a` collapses to the address: borrow is the IDENTITY at
            // runtime. The loan pins the obligation to `restore` (getting the
            // Own back) — dropping it is a leak, and freeing the cell while
            // borrowed strands the loan, so no accepted program can do either.
            "borrow" => {
                let rt = self.runtime_args(f, env, cname, args)?;
                let [o] = rt.as_slice() else {
                    return Err(format!(
                        "borrow: expected 1 runtime argument (the Own), got {}",
                        rt.len()
                    ));
                };
                Ok(Val::Int(*o))
            }
            // restore : {0 a} {0 l} -> Ptr l -> (1 v : PtsTo l a) -> (1 ln : Loan l a)
            //         -> Own a — reunite the pieces: IDENTITY on the address.
            "restore" => {
                let rt = self.runtime_args_v(f, env, cname, args)?;
                let [p, _v, _ln] = rt.as_slice() else {
                    return Err(format!(
                        "restore: expected 3 runtime arguments (ptr, view, loan), got {}",
                        rt.len()
                    ));
                };
                Ok(Val::Int(p.as_int("restore ptr")?))
            }
            // type-level postulates (Own/Region/List/Cursor/Arr/Loc/Ptr/PtsTo):
            // these only ever appear inside ERASED type annotations, so they must
            // never reach a runtime value position. If one does, that is a real bug.
            other => Err(format!(
                "cannot run the abstract postulate `{other}` (no native impl yet)"
            )),
        }
    }

    /// Lower a constructor application. `args` is `[params.., ctor-args..]` (the
    /// kernel stores params first). Nat-like families stay unboxed `i64`; every
    /// other family is a boxed heap cell tagged in slot 0.
    fn compile_constr(
        &self,
        f: FunctionValue<'c>,
        env: &[Slot<'c>],
        name: &str,
        args: &[Term],
    ) -> Result<Val<'c>, String> {
        let (decl, ctor) = self
            .sig
            .ctor(name)
            .ok_or_else(|| format!("unknown constructor `{name}`"))?;
        let data = decl.name.clone();
        let np = decl.params.len();
        let ctor_args = ctor.args.clone();

        // Nat-like: keep the unboxed i64 representation.
        if let Some((zero, _succ)) = nat_like(self.sig, &data) {
            if name == zero {
                return Ok(Val::Int(self.i64t.const_int(0, false)));
            } else {
                // the successor's single argument is the predecessor.
                let pred = self.compile(f, env, &args[args.len() - 1])?;
                return Ok(Val::Int(
                    self.builder
                        .build_int_add(pred, self.i64t.const_int(1, false), "succ")
                        .unwrap(),
                ));
            }
        }

        // TRANSPARENT newtype: constructing the wrapper IS the field value.
        if let Some(fi) = transparent_field(self.sig, &data) {
            return self.compile_v(f, env, &args[np + fi]);
        }

        let lay = ctor_layout(self.sig, &data, name)?;

        // A FLAT RECORD: no malloc, no tag — the value is its fields' scalars,
        // concatenated in declaration order (each coerced to its declared
        // width; erased fields contribute nothing). This is Phase B2's
        // by-value construction: `MkPoint(3, 4)` is two SSA values.
        if record_width(self.sig, &data).is_some() {
            let mut comps: Vec<IntValue<'c>> = Vec::with_capacity(lay.nfields as usize);
            for (ai, slot) in lay.arg_slot.iter().enumerate() {
                if slot.is_some() {
                    let v = self.compile_v(f, env, &args[np + ai])?;
                    // a FLEX (parameter-typed) field of a flat record takes the
                    // value's ACTUAL width — the record's layout is whatever its
                    // instantiation makes it, no boxing; a concrete field must
                    // match its declared width exactly.
                    let v = if lay.arg_flex[ai] {
                        v
                    } else {
                        self.expect_width(v, lay.arg_width[ai], "record field")?
                    };
                    comps.extend(v.components());
                }
            }
            return Ok(match comps.len() {
                1 => Val::Int(comps[0]),
                _ => Val::Agg(comps),
            });
        }

        // A VALUE ENUM: a flat tagged union in registers — [tag, payload…,
        // zero-padding to the enum's width]. The width comes from the
        // INSTANTIATED parameters (args[..np] are the param terms), so
        // `Opt Point` is [tag, x, y] and `Opt Nat` is [tag, n]. NO malloc.
        if let Some(es) = enum_value_shape(
            self.sig,
            &data,
            &args[..np].to_vec(),
            &mut std::collections::HashSet::new(),
        ) {
            // THE NULL-POINTER NICHE: the nullary constructor IS 0; the pointer
            // constructor IS its (non-null, malloc'd) pointer. One slot, zero
            // instructions of overhead — C's nullable pointer.
            if let Some((nc, pc)) = es.niche {
                let lay0 = ctor_layout(self.sig, &data, name)?;
                let ci = lay0.tag as usize;
                if ci == nc {
                    return Ok(Val::Int(self.i64t.const_zero()));
                }
                if ci == pc {
                    // the single runtime argument is the pointer.
                    let lay = ctor_layout(self.sig, &data, name)?;
                    for (ai, slot) in lay.arg_slot.iter().enumerate() {
                        if slot.is_some() {
                            let v = self.compile_v(f, env, &args[np + ai])?;
                            return Ok(Val::Int(v.as_int("niche pointer payload")?));
                        }
                    }
                    return Err(format!("`{name}`: niche constructor has no runtime field"));
                }
            }
            // an abstract instantiation cannot know its layout — guided error.
            if args[..np].iter().any(|a| type_is_flex(self.sig, a) )
                || es.payloads.iter().any(|(_, fx)| *fx)
            {
                return Err(format!(
                    "`{name}`: constructing `{data}` at an ABSTRACT type \
                     instantiation — its tagged-union layout is unknown at compile \
                     time, and tallyc never boxes implicitly. Specialize the \
                     function (non-%partial generics inline and work as-is), or \
                     declare `boxed enum {data}`."
                ));
            }
            // THE FLEX-RECOVERABILITY INVARIANT: a match compiles without the
            // instantiation, so an arm recovers a parameter-typed field's width
            // as (total − 1 − its concrete siblings) — valid only if that
            // constructor's payload is the WIDEST at this instantiation.
            // Enforced here, at construction, where the instantiation is known.
            let decl_es = enum_value_shape(
                self.sig,
                &data,
                &[],
                &mut std::collections::HashSet::new(),
            )
            .ok_or_else(|| format!("`{data}`: lost its value-enum shape"))?;
            for (ci, (dw, dfx)) in decl_es.payloads.iter().enumerate() {
                let _ = dw;
                if *dfx && es.payloads[ci].0 != es.width - 1 {
                    return Err(format!(
                        "`{data}` at this instantiation gives constructor \
                         `{}` a parameter-typed payload NARROWER than the widest \
                         constructor — a match could not recover its layout from \
                         the value, and tallyc never boxes implicitly. Make the \
                         payload concrete, box it (`Own …`), or declare \
                         `boxed enum {data}`.",
                        self.sig.data(&data).unwrap().ctors[ci].name
                    ));
                }
            }
            let lay = ctor_layout(self.sig, &data, name)?;
            let mut comps: Vec<IntValue<'c>> = vec![self.i64t.const_int(lay.tag, false)];
            for (ai, slot) in lay.arg_slot.iter().enumerate() {
                if slot.is_some() {
                    let v = self.compile_v(f, env, &args[np + ai])?;
                    let v = if lay.arg_flex[ai] {
                        v
                    } else {
                        self.expect_width(v, lay.arg_width[ai], "enum payload field")?
                    };
                    comps.extend(v.components());
                }
            }
            while (comps.len() as u32) < es.width {
                comps.push(self.i64t.const_zero());
            }
            if comps.len() as u32 != es.width {
                return Err(format!(
                    "`{name}`: payload wider than `{data}`'s tagged-union layout \
                     ({} vs {} slots) — impossible for a kernel-checked term",
                    comps.len(),
                    es.width
                ));
            }
            return Ok(if es.width == 1 {
                Val::Int(comps[0])
            } else {
                Val::Agg(comps)
            });
        }
        let _ = &ctor_args;

        // A NULLARY constructor (no runtime fields, e.g. `Leaf`/`Nil`) is an
        // immutable, field-less value: it needs no per-use allocation. Represent it
        // as the address of ONE shared, module-level constant cell `{tag}` — so a
        // tree's 2^d leaves cost zero `malloc`s (matching C's NULL-for-leaf), while
        // the eliminator still reads the tag the same way.
        if lay.nfields == 0 {
            let gname = format!("tally_nullary_{name}");
            let global = self.module.get_global(&gname).unwrap_or_else(|| {
                let arr_ty = self.i64t.array_type(1);
                let g = self.module.add_global(arr_ty, None, &gname);
                g.set_initializer(
                    &self.i64t.const_array(&[self.i64t.const_int(lay.tag, false)]),
                );
                g.set_constant(true);
                g
            });
            return Ok(Val::Int(
                self.builder
                    .build_ptr_to_int(global.as_pointer_value(), self.i64t, "nullary")
                    .unwrap(),
            ));
        }

        // Boxed: malloc(8 * (1 + nfields)); store tag in slot 0; store each
        // non-erased ctor argument in its slot. Params and erased args store
        // NOTHING (erasure = zero overhead).
        let nslots = 1 + lay.nfields;
        let sz = self.i64t.const_int(nslots as u64 * 8, false);
        let raw = self
            .builder
            .build_call(self.malloc, &[sz.into()], "cell")
            .unwrap()
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_pointer_value();
        // slot 0: constructor tag
        self.builder
            .build_store(self.gep(raw, 0, "tag"), self.i64t.const_int(lay.tag, false))
            .unwrap();
        // ctor args are args[np..]; a flat-record field spans consecutive slots.
        // A GENERIC (parameter-typed) field of a boxed datatype is one slot in a
        // FIXED cell layout — a record cannot go there, and tallyc never boxes
        // implicitly: `expect_width` errors with the explicit alternatives
        // (store an `Own T`, or declare a concrete container, which inlines it).
        for (ai, slot) in lay.arg_slot.iter().enumerate() {
            if let Some(s) = slot {
                let v = self.compile_v(f, env, &args[np + ai])?;
                let what = if lay.arg_flex[ai] {
                    format!("`{name}`'s generic field")
                } else {
                    format!("`{name}`'s field")
                };
                let v = self.expect_width(v, lay.arg_width[ai], &what)?;
                for (k, c) in v.components().into_iter().enumerate() {
                    self.builder
                        .build_store(self.gep(raw, *s + k as u32, "fld"), c)
                        .unwrap();
                }
            }
            // erased argument: not evaluated, not stored.
        }
        Ok(Val::Int(
            self.builder
                .build_ptr_to_int(raw, self.i64t, "cellint")
                .unwrap(),
        ))
    }

    /// Lower a dependent eliminator. Nat-like families fold with a native loop;
    /// every other (boxed) family lowers to a recursive native function that
    /// switches on the constructor tag and recurses on the recursive fields.
    fn compile_elim(
        &self,
        f: FunctionValue<'c>,
        env: &[Slot<'c>],
        data: &str,
        methods: &[Term],
        scrut: &Term,
    ) -> Result<Val<'c>, String> {
        if let Some((zero, _succ)) = nat_like(self.sig, data) {
            let decl = self.sig.data(data).unwrap();
            let zidx = decl.ctors.iter().position(|c| c.name == zero).unwrap();
            let sidx = 1 - zidx;
            return self.compile_fold(f, env, &methods[zidx], &methods[sidx], scrut);
        }
        // TRANSPARENT newtype: no recursion possible (transparency excludes
        // recursive fields ⇒ no IHs), so the eliminator is the same in-place
        // bind-and-go as a `Case`.
        if let Some(fi) = transparent_field(self.sig, data) {
            return self.compile_transparent_match(f, env, data, fi, &methods[0], scrut);
        }
        // A FLAT RECORD has no recursive fields ⇒ its eliminator has no IHs: it
        // is exactly the single-method in-place match.
        if record_width(self.sig, data).is_some() {
            return self.compile_case(f, env, data, methods, scrut);
        }
        // A VALUE ENUM is non-recursive ⇒ no IHs either: same in-place switch.
        if enum_value_shape(self.sig, data, &[], &mut std::collections::HashSet::new())
            .is_some()
        {
            return self.compile_case(f, env, data, methods, scrut);
        }

        // Boxed eliminator: build a recursive helper that captures the live outer
        // environment (the non-erased slots), then call it on the scrutinee.
        let scrut_val = self.compile(f, env, scrut)?;

        // Which outer env slots are live (non-erased)? They become helper params
        // (in addition to the scrutinee) — a flat-record slot FLATTENS into its
        // components. We thread them through every recursive call unchanged —
        // only the scrutinee shrinks.
        let live: Vec<(usize, Val<'c>)> = env
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.clone().map(|v| (i, v)))
            .collect();
        let flat_live: u32 = live.iter().map(|(_, v)| v.width()).sum();

        let nparams = 1 + flat_live as usize;
        let param_tys: Vec<inkwell::types::BasicMetadataTypeEnum> =
            std::iter::repeat(self.i64t.into()).take(nparams).collect();
        let fname = self.fresh(&format!("tally_elim_{data}"));
        let helper = self
            .module
            .add_function(&fname, self.i64t.fn_type(&param_tys, false), None);

        // Reconstruct, INSIDE the helper, the env that the method bodies expect:
        // the same shape as the outer env (Nones preserved), but every live slot
        // bound to the corresponding helper parameter(s) — an Agg slot rebuilt
        // from its consecutive params.
        let mut inner_env: Vec<Slot<'c>> = env.iter().map(|_| None).collect();
        let mut pk = 1u32;
        for (i, v) in &live {
            let w = v.width();
            if w == 1 {
                inner_env[*i] =
                    Some(Val::Int(helper.get_nth_param(pk).unwrap().into_int_value()));
            } else {
                let comps: Vec<IntValue<'c>> = (0..w)
                    .map(|k| helper.get_nth_param(pk + k).unwrap().into_int_value())
                    .collect();
                inner_env[*i] = Some(Val::Agg(comps));
            }
            pk += w;
        }
        let helper_scrut = helper.get_nth_param(0).unwrap().into_int_value();

        // Save the current insert position, build the helper body, then restore.
        let saved = self.builder.get_insert_block().unwrap();
        self.build_elim_helper(helper, data, methods, &inner_env, helper_scrut)?;
        self.builder.position_at_end(saved);

        // call the helper on the scrutinee + captured live env (Aggs flattened).
        let mut call_args: Vec<inkwell::values::BasicMetadataValueEnum> = Vec::with_capacity(nparams);
        call_args.push(scrut_val.into());
        for (_, v) in &live {
            for c in v.components() {
                call_args.push(c.into());
            }
        }
        Ok(Val::Int(
            self.builder
                .build_call(helper, &call_args, "elim")
                .unwrap()
                .try_as_basic_value()
                .left()
                .unwrap()
                .into_int_value(),
        ))
    }

    /// Emit the body of a boxed-eliminator helper: switch on the scrutinee's tag,
    /// and in each arm bind the constructor's runtime fields plus one induction
    /// hypothesis per recursive field (a recursive `self` call), then compile the
    /// corresponding method.
    fn build_elim_helper(
        &self,
        helper: FunctionValue<'c>,
        data: &str,
        methods: &[Term],
        env: &[Slot<'c>],
        scrut_val: IntValue<'c>,
    ) -> Result<(), String> {
        // The captured env values, as seen INSIDE the helper, are exactly the
        // helper's own parameters #1.. (param #0 is the scrutinee; flat-record
        // slots were flattened into consecutive params by the caller). A
        // recursive call must forward THESE, not the outer caller's values.
        let captured: Vec<IntValue<'c>> = helper
            .get_params()
            .iter()
            .skip(1)
            .map(|p| p.into_int_value())
            .collect();
        let decl = self.sig.data(data).unwrap();
        if methods.len() != decl.ctors.len() {
            return Err(format!(
                "elim[{data}]: expected {} method(s), got {}",
                decl.ctors.len(),
                methods.len()
            ));
        }

        let entry = self.ctx.append_basic_block(helper, "entry");
        self.builder.position_at_end(entry);
        let scrut_ptr = self
            .builder
            .build_int_to_ptr(scrut_val, self.ptr, "scrut")
            .unwrap();
        let tag = self.load(scrut_ptr, 0, "tag");

        // one block per constructor + an unreachable default.
        let default = self.ctx.append_basic_block(helper, "elim.default");
        let arm_blocks: Vec<_> = decl
            .ctors
            .iter()
            .map(|c| self.ctx.append_basic_block(helper, &format!("elim.{}", c.name)))
            .collect();
        let cases: Vec<(IntValue<'c>, inkwell::basic_block::BasicBlock<'c>)> = decl
            .ctors
            .iter()
            .enumerate()
            .map(|(ci, _)| (self.i64t.const_int(ci as u64, false), arm_blocks[ci]))
            .collect();
        self.builder.build_switch(tag, default, &cases).unwrap();

        self.builder.position_at_end(default);
        self.builder.build_unreachable().unwrap();

        for (ci, ctor) in decl.ctors.iter().enumerate() {
            self.builder.position_at_end(arm_blocks[ci]);
            let lay = ctor_layout(self.sig, data, &ctor.name)?;

            // Bind the method's lambda parameters in the standard order: ALL the
            // constructor's arguments first (each loaded from its slot, or `None`
            // if erased), THEN one induction hypothesis per recursive argument (a
            // recursive `self` call on that field). This args-then-IHs order
            // matches `method_ty_tm` and the kernel `velim`.
            let mut menv = env.to_vec();
            let mut arg_vals: Vec<Slot<'c>> = Vec::with_capacity(ctor.args.len());
            for ai in 0..ctor.args.len() {
                let arg_val: Slot<'c> = match lay.arg_slot[ai] {
                    Some(sl) => Some(self.load_field(scrut_ptr, sl, lay.arg_width[ai])),
                    None => None, // erased argument: no runtime witness.
                };
                menv.push(arg_val.clone());
                arg_vals.push(arg_val);
            }
            for ai in 0..ctor.args.len() {
                if lay.arg_recursive[ai] {
                    // induction hypothesis: elim recursively on the (boxed) field.
                    let field_ptr = arg_vals[ai]
                        .clone()
                        .ok_or_else(|| {
                            format!(
                                "{}.{}: a recursive argument was erased — impossible for a \
                             strictly-positive family",
                                data, ctor.name
                            )
                        })?
                        .as_int("recursive field")?;
                    let mut rec_args: Vec<inkwell::values::BasicMetadataValueEnum> =
                        Vec::with_capacity(1 + captured.len());
                    rec_args.push(field_ptr.into());
                    for v in &captured {
                        rec_args.push((*v).into());
                    }
                    let ih = self
                        .builder
                        .build_call(helper, &rec_args, "ih")
                        .unwrap()
                        .try_as_basic_value()
                        .left()
                        .unwrap()
                        .into_int_value();
                    menv.push(Some(Val::Int(ih)));
                }
            }

            // the method is a nested-Lam term; its body is reached by stripping
            // exactly `#args + #recursive-args` lambdas — which is what `menv`
            // bound. β-reduce by walking the lambdas.
            let mut body = strip_ann(&methods[ci]);
            let nbind = ctor.args.len() + lay.arg_recursive.iter().filter(|b| **b).count();
            for _ in 0..nbind {
                match body {
                    Term::Lam(inner) => body = strip_ann(inner),
                    _ => {
                        return Err(format!(
                            "elim[{data}] method for `{}` is not a {nbind}-argument function",
                            ctor.name
                        ))
                    }
                }
            }
            // the helper's return slot is one scalar. A total fold over a boxed
            // family producing a RECORD is not widened yet — and tallyc never
            // boxes implicitly, so it is a guided hard error (`%partial` Fix
            // recursion returns records by value today).
            let result = self.compile_v(helper, &menv, body)?;
            let result = self
                .expect_width(
                    result,
                    1,
                    "a total boxed-family fold's result (make the function \
                     %partial — Fix returns records by value)",
                )?
                .as_int("eliminator result")?;
            self.builder.build_return(Some(&result)).unwrap();
        }
        Ok(())
    }

    /// A match on a TRANSPARENT newtype: the scrutinee IS the single runtime
    /// field — bind it (erased siblings bind nothing) and compile the sole
    /// method body in place. No switch, no loads, no helper.
    fn compile_transparent_match(
        &self,
        f: FunctionValue<'c>,
        env: &[Slot<'c>],
        data: &str,
        fi: usize,
        method: &Term,
        scrut: &Term,
    ) -> Result<Val<'c>, String> {
        let decl = self
            .sig
            .data(data)
            .ok_or_else(|| format!("match on unknown datatype `{data}`"))?;
        let ctor = &decl.ctors[0];
        let scrut_val = self.compile_v(f, env, scrut)?;
        let mut menv = env.to_vec();
        for ai in 0..ctor.args.len() {
            menv.push(if ai == fi { Some(scrut_val.clone()) } else { None });
        }
        let mut body = strip_ann(method);
        for _ in 0..ctor.args.len() {
            match body {
                Term::Lam(inner) => body = strip_ann(inner),
                _ => {
                    return Err(format!(
                        "match[{data}]: method is not a {}-argument function",
                        ctor.args.len()
                    ))
                }
            }
        }
        self.compile_v(f, &menv, body)
    }

    /// Compile `Case` (non-recursive general case-split) IN PLACE: switch on the boxed
    /// scrutinee's tag, bind each constructor's runtime fields, β-reduce + compile the
    /// matched method (its args ONLY — NO induction hypotheses, NO recursive helper, NO
    /// recursion), and phi-join the arm results. Recursion, if any, is the enclosing
    /// `Fix`'s self-call — never here. The general-datatype counterpart of
    /// `compile_natcase`, and what makes a `Fix` body dispatch on a heap structure
    /// WITHOUT the eliminator's implicit-IH exponential blow-up.
    fn compile_case(
        &self,
        f: FunctionValue<'c>,
        env: &[Slot<'c>],
        data: &str,
        methods: &[Term],
        scrut: &Term,
    ) -> Result<Val<'c>, String> {
        let decl = self
            .sig
            .data(data)
            .ok_or_else(|| format!("case on unknown datatype `{data}`"))?;
        if methods.len() != decl.ctors.len() {
            return Err(format!(
                "case[{data}]: expected {} method(s), got {}",
                decl.ctors.len(),
                methods.len()
            ));
        }
        if let Some(fi) = transparent_field(self.sig, data) {
            return self.compile_transparent_match(f, env, data, fi, &methods[0], scrut);
        }
        // A FLAT RECORD match: no tag, no switch, no loads — the scrutinee's
        // components (unboxed here if it arrived as a boxed pointer from a
        // generic boundary) ARE the fields; bind them and compile the single
        // method in place.
        if record_width(self.sig, data).is_some() {
            let sv = self.compile_v(f, env, scrut)?;
            let comps = sv.components();
            let lay = ctor_layout(self.sig, data, &decl.ctors[0].name)?;
            let nargs = decl.ctors[0].args.len();
            // FLEXIBLE split: concrete fields take their declared widths; when
            // exactly ONE runtime field is flex (parameter-typed), it absorbs
            // the value's surplus width — the layout is read off the VALUE, so
            // generic flat records (`ARead a n`, `Taken a l`) work at every
            // instantiation with no boxing. Ambiguity (two flex fields and a
            // surplus) is a hard error, not a guess.
            let total: u32 = comps.len() as u32;
            let declared: u32 = lay.nfields;
            let nflex = (0..nargs)
                .filter(|&ai| lay.arg_slot[ai].is_some() && lay.arg_flex[ai])
                .count();
            let surplus = total.checked_sub(declared).ok_or_else(|| {
                format!(
                    "match[{data}]: this {total}-slot value cannot be a flat \
                     `{data}` ({declared} slots) — it reached here through a \
                     ONE-slot generic position (a generic container field or an \
                     abstract-typed parameter), where a record's layout cannot \
                     live and tallyc never boxes implicitly. Box it explicitly \
                     (store `Own {data}` in the container and `unbox` it), or \
                     declare a concrete container ({data}-typed fields store the \
                     record flat)."
                )
            })?;
            if surplus > 0 && nflex != 1 {
                return Err(format!(
                    "match[{data}]: cannot split a {total}-slot record over \
                     {nflex} generic field(s) (declared width {declared}) — the \
                     layout is ambiguous; make the record's fields concrete"
                ));
            }
            let mut menv = env.to_vec();
            let mut off = 0usize;
            for ai in 0..nargs {
                if lay.arg_slot[ai].is_some() {
                    let wi = if lay.arg_flex[ai] && surplus > 0 {
                        (lay.arg_width[ai] + surplus) as usize
                    } else {
                        lay.arg_width[ai] as usize
                    };
                    let v = if wi == 1 {
                        Val::Int(comps[off])
                    } else {
                        Val::Agg(comps[off..off + wi].to_vec())
                    };
                    off += wi;
                    menv.push(Some(v));
                } else {
                    menv.push(None);
                }
            }
            let mut body = strip_ann(&methods[0]);
            for _ in 0..nargs {
                match body {
                    Term::Lam(inner) => body = strip_ann(inner),
                    _ => {
                        return Err(format!(
                            "case[{data}]: method is not a {nargs}-argument function"
                        ))
                    }
                }
            }
            return self.compile_v(f, &menv, body);
        }

        // A VALUE ENUM match: switch on the TAG COMPONENT (register, not a
        // load), each arm binding its payload from the value's components —
        // concrete fields at their declared widths, one parameter-typed field
        // absorbing the surplus (sound by the construction-side invariant).
        // No cell, no loads, no free. A refuted (absurd) arm's method is a
        // binderless sentinel: its block is simply unreachable.
        if let Some(decl_es) =
            enum_value_shape(self.sig, data, &[], &mut std::collections::HashSet::new())
        {
            let sv = self.compile_v(f, env, scrut)?;
            let comps = sv.components();
            let total = comps.len() as u32;

            // THE NULL-POINTER NICHE at the match: a one-slot value of a
            // {nullary, single-field} enum is the niche representation (the
            // tagged form of such an enum is always ≥ 2 slots) — compare
            // against 0 and bind the pointer in the non-null arm.
            let niche_shape = decl_es.payloads.len() == 2
                && decl_es.width > 1
                && d_nullary_ptr_shape(self.sig, data).is_some();
            if total == 1 && niche_shape {
                let (nc, pc) = d_nullary_ptr_shape(self.sig, data).unwrap();
                let pv = comps[0];
                let null_bb = self.ctx.append_basic_block(f, "niche.null");
                let ptr_bb = self.ctx.append_basic_block(f, "niche.ptr");
                let join_bb = self.ctx.append_basic_block(f, "niche.join");
                let is_null = self
                    .builder
                    .build_int_compare(
                        inkwell::IntPredicate::EQ,
                        pv,
                        self.i64t.const_zero(),
                        "niche.isnull",
                    )
                    .unwrap();
                self.builder
                    .build_conditional_branch(is_null, null_bb, ptr_bb)
                    .unwrap();

                self.builder.position_at_end(null_bb);
                let nz = peel_n_lams(&methods[nc], decl.ctors[nc].args.len())
                    .ok_or_else(|| format!("match[{data}]: nullary method shape"))?;
                let zv = self.compile_v(f, env, nz)?;
                let null_end = self.builder.get_insert_block().unwrap();

                self.builder.position_at_end(ptr_bb);
                let lay = ctor_layout(self.sig, data, &decl.ctors[pc].name)?;
                let mut menv = env.to_vec();
                for ai in 0..decl.ctors[pc].args.len() {
                    if lay.arg_slot[ai].is_some() {
                        menv.push(Some(Val::Int(pv)));
                    } else {
                        menv.push(None);
                    }
                }
                let pb = peel_n_lams(&methods[pc], decl.ctors[pc].args.len())
                    .ok_or_else(|| format!("match[{data}]: pointer method shape"))?;
                let pvr = self.compile_v(f, &menv, pb)?;
                let ptr_end = self.builder.get_insert_block().unwrap();

                let w = zv.width().max(pvr.width());
                self.builder.position_at_end(null_end);
                let zc = self.expect_width(zv, w, "niche null arm")?.components();
                let null_end = self.builder.get_insert_block().unwrap();
                self.builder.build_unconditional_branch(join_bb).unwrap();
                self.builder.position_at_end(ptr_end);
                let pc2 = self.expect_width(pvr, w, "niche pointer arm")?.components();
                let ptr_end = self.builder.get_insert_block().unwrap();
                self.builder.build_unconditional_branch(join_bb).unwrap();

                self.builder.position_at_end(join_bb);
                let mut out: Vec<IntValue<'c>> = Vec::with_capacity(w as usize);
                for k in 0..w as usize {
                    let phi = self.builder.build_phi(self.i64t, "niche.result").unwrap();
                    phi.add_incoming(&[
                        (&zc[k] as &dyn inkwell::values::BasicValue, null_end),
                        (&pc2[k] as &dyn inkwell::values::BasicValue, ptr_end),
                    ]);
                    out.push(phi.as_basic_value().into_int_value());
                }
                return Ok(if w == 1 { Val::Int(out[0]) } else { Val::Agg(out) });
            }
            let tag = comps[0];

            let default = self.ctx.append_basic_block(f, "vcase.default");
            let join = self.ctx.append_basic_block(f, "vcase.join");
            let arm_blocks: Vec<_> = decl
                .ctors
                .iter()
                .map(|c| self.ctx.append_basic_block(f, &format!("vcase.{}", c.name)))
                .collect();
            let cases: Vec<(IntValue<'c>, inkwell::basic_block::BasicBlock<'c>)> = decl
                .ctors
                .iter()
                .enumerate()
                .map(|(ci, _)| (self.i64t.const_int(ci as u64, false), arm_blocks[ci]))
                .collect();
            self.builder.build_switch(tag, default, &cases).unwrap();
            self.builder.position_at_end(default);
            self.builder.build_unreachable().unwrap();

            let mut arm_results: Vec<(Val<'c>, inkwell::basic_block::BasicBlock<'c>)> =
                Vec::new();
            for (ci, ctor) in decl.ctors.iter().enumerate() {
                self.builder.position_at_end(arm_blocks[ci]);
                let lay = ctor_layout(self.sig, data, &ctor.name)?;
                // a refuted arm: the elaborator's sentinel has no binders for an
                // argful constructor — the scrutinee's type already rules this
                // tag out, so the block is dead.
                let peeled = peel_n_lams(&methods[ci], ctor.args.len());
                let Some(body) = peeled else {
                    self.builder.build_unreachable().unwrap();
                    continue;
                };
                let concrete: u32 = (0..ctor.args.len())
                    .filter(|&ai| lay.arg_slot[ai].is_some() && !lay.arg_flex[ai])
                    .map(|ai| lay.arg_width[ai])
                    .sum();
                let nflex = (0..ctor.args.len())
                    .filter(|&ai| lay.arg_slot[ai].is_some() && lay.arg_flex[ai])
                    .count();
                let mut menv = env.to_vec();
                let mut off = 1usize;
                for ai in 0..ctor.args.len() {
                    if lay.arg_slot[ai].is_none() {
                        menv.push(None);
                        continue;
                    }
                    let wi = if lay.arg_flex[ai] {
                        if nflex != 1 {
                            return Err(format!(
                                "match[{data}]: constructor `{}` has {nflex} \
                                 parameter-typed payload fields — the layout is \
                                 ambiguous at the match; make them concrete or \
                                 declare `boxed enum {data}`",
                                ctor.name
                            ));
                        }
                        (total - 1 - concrete) as usize
                    } else {
                        lay.arg_width[ai] as usize
                    };
                    let v = if wi == 1 {
                        Val::Int(comps[off])
                    } else {
                        Val::Agg(comps[off..off + wi].to_vec())
                    };
                    off += wi;
                    menv.push(Some(v));
                }
                let result = self.compile_v(f, &menv, body)?;
                let pred = self.builder.get_insert_block().unwrap();
                arm_results.push((result, pred));
            }

            if arm_results.is_empty() {
                return Err(format!(
                    "match[{data}]: every arm is refuted — an uninhabited match \
                     should have been discharged upstream"
                ));
            }
            let w = arm_results.iter().map(|(v, _)| v.width()).max().unwrap_or(1);
            let mut incoming: Vec<(Vec<IntValue<'c>>, inkwell::basic_block::BasicBlock<'c>)> =
                Vec::new();
            for (v, bb) in arm_results {
                self.builder.position_at_end(bb);
                let cs = self
                    .expect_width(v, w, &format!("case[{data}] arm result"))?
                    .components();
                let pred = self.builder.get_insert_block().unwrap();
                self.builder.build_unconditional_branch(join).unwrap();
                incoming.push((cs, pred));
            }
            self.builder.position_at_end(join);
            let mut out: Vec<IntValue<'c>> = Vec::with_capacity(w as usize);
            for k in 0..w as usize {
                let phi = self.builder.build_phi(self.i64t, "vcase.result").unwrap();
                for (cs, bb) in &incoming {
                    phi.add_incoming(&[(&cs[k] as &dyn inkwell::values::BasicValue, *bb)]);
                }
                out.push(phi.as_basic_value().into_int_value());
            }
            return Ok(if w == 1 { Val::Int(out[0]) } else { Val::Agg(out) });
        }
        let scrut_val = self.compile(f, env, scrut)?;
        let scrut_ptr = self
            .builder
            .build_int_to_ptr(scrut_val, self.ptr, "case.scrut")
            .unwrap();
        let tag = self.load(scrut_ptr, 0, "case.tag");

        let default = self.ctx.append_basic_block(f, "case.default");
        let join = self.ctx.append_basic_block(f, "case.join");
        let arm_blocks: Vec<_> = decl
            .ctors
            .iter()
            .map(|c| self.ctx.append_basic_block(f, &format!("case.{}", c.name)))
            .collect();
        let cases: Vec<(IntValue<'c>, inkwell::basic_block::BasicBlock<'c>)> = decl
            .ctors
            .iter()
            .enumerate()
            .map(|(ci, _)| (self.i64t.const_int(ci as u64, false), arm_blocks[ci]))
            .collect();
        self.builder.build_switch(tag, default, &cases).unwrap();

        self.builder.position_at_end(default);
        self.builder.build_unreachable().unwrap();

        // each arm binds the ctor's fields (a record field loads its consecutive
        // slots) and compiles the method; the branch to `join` is deferred until
        // every arm's result WIDTH is known, so mixed representations coerce to
        // one width before the phi-join.
        let mut arm_results: Vec<(Val<'c>, inkwell::basic_block::BasicBlock<'c>)> =
            Vec::with_capacity(decl.ctors.len());
        for (ci, ctor) in decl.ctors.iter().enumerate() {
            self.builder.position_at_end(arm_blocks[ci]);
            let lay = ctor_layout(self.sig, data, &ctor.name)?;
            let mut menv = env.to_vec();
            for ai in 0..ctor.args.len() {
                let arg_val: Slot<'c> = match lay.arg_slot[ai] {
                    Some(sl) => Some(self.load_field(scrut_ptr, sl, lay.arg_width[ai])),
                    None => None, // erased argument: no runtime witness.
                };
                menv.push(arg_val);
            }
            // β-reduce the method: strip exactly `#args` lambdas — NO IH binders.
            let mut body = strip_ann(&methods[ci]);
            for _ in 0..ctor.args.len() {
                match body {
                    Term::Lam(inner) => body = strip_ann(inner),
                    _ => {
                        return Err(format!(
                            "case[{data}] method for `{}` is not a {}-argument function",
                            ctor.name,
                            ctor.args.len()
                        ))
                    }
                }
            }
            let result = self.compile_v(f, &menv, body)?;
            // the method body may have emitted its own control flow, so the phi's
            // predecessor is the CURRENT block, not necessarily `arm_blocks[ci]`.
            let pred = self.builder.get_insert_block().unwrap();
            arm_results.push((result, pred));
        }

        // every arm has the same kernel type, and each type has exactly ONE
        // representation (records are always flat) — so the arm widths must
        // agree. A disagreement is a compiler bug, not something to paper over.
        let w = arm_results.iter().map(|(v, _)| v.width()).max().unwrap_or(1);
        let mut incoming: Vec<(Vec<IntValue<'c>>, inkwell::basic_block::BasicBlock<'c>)> =
            Vec::with_capacity(arm_results.len());
        for (v, bb) in arm_results {
            self.builder.position_at_end(bb);
            let comps = self
                .expect_width(v, w, &format!("case[{data}] arm result"))?
                .components();
            let pred = self.builder.get_insert_block().unwrap();
            self.builder.build_unconditional_branch(join).unwrap();
            incoming.push((comps, pred));
        }

        self.builder.position_at_end(join);
        let mut out: Vec<IntValue<'c>> = Vec::with_capacity(w as usize);
        for k in 0..w as usize {
            let phi = self.builder.build_phi(self.i64t, "case.result").unwrap();
            for (comps, bb) in &incoming {
                phi.add_incoming(&[(&comps[k] as &dyn inkwell::values::BasicValue, *bb)]);
            }
            out.push(phi.as_basic_value().into_int_value());
        }
        Ok(if w == 1 { Val::Int(out[0]) } else { Val::Agg(out) })
    }

    /// `elim z s n` (Nat-like) as a native loop:
    ///   acc = z; for k in 0..n { acc = s k acc }.
    /// The accumulator may be a flat record — one phi per component.
    fn compile_fold(
        &self,
        f: FunctionValue<'c>,
        env: &[Slot<'c>],
        z: &Term,
        s: &Term,
        scrut: &Term,
    ) -> Result<Val<'c>, String> {
        // the loop-carried accumulator is as wide as its value: a flat record
        // accumulates in registers (one phi per component) — never boxed.
        let zv = self.compile_v(f, env, z)?;
        let aw = zv.width();
        let nv = self.compile(f, env, scrut)?;

        // s = λk. λih. body
        let s_body = match strip_ann(s) {
            Term::Lam(b1) => match strip_ann(b1) {
                Term::Lam(b2) => &**b2,
                _ => return Err("eliminator step is not a 2-argument function".into()),
            },
            _ => return Err("eliminator step is not a function".into()),
        };

        let entry = self.builder.get_insert_block().unwrap();
        let cond = self.ctx.append_basic_block(f, "fold.cond");
        let body = self.ctx.append_basic_block(f, "fold.body");
        let exit = self.ctx.append_basic_block(f, "fold.exit");
        self.builder.build_unconditional_branch(cond).unwrap();

        self.builder.position_at_end(cond);
        let k_phi = self.builder.build_phi(self.i64t, "k").unwrap();
        let acc_phis: Vec<_> = (0..aw)
            .map(|_| self.builder.build_phi(self.i64t, "acc").unwrap())
            .collect();
        k_phi.add_incoming(&[(&self.i64t.const_int(0, false), entry)]);
        for (phi, c) in acc_phis.iter().zip(zv.components()) {
            phi.add_incoming(&[(&c, entry)]);
        }
        let k_val = k_phi.as_basic_value().into_int_value();
        let acc_comps: Vec<IntValue<'c>> = acc_phis
            .iter()
            .map(|p| p.as_basic_value().into_int_value())
            .collect();
        let acc_val = if aw == 1 {
            Val::Int(acc_comps[0])
        } else {
            Val::Agg(acc_comps.clone())
        };
        let more = self
            .builder
            .build_int_compare(inkwell::IntPredicate::ULT, k_val, nv, "more")
            .unwrap();
        self.builder
            .build_conditional_branch(more, body, exit)
            .unwrap();

        self.builder.position_at_end(body);
        let mut env2 = env.to_vec();
        env2.push(Some(Val::Int(k_val))); // k
        env2.push(Some(acc_val.clone())); // ih
        let next_acc0 = self.compile_v(f, &env2, s_body)?;
        let next_acc = self.expect_width(next_acc0, aw, "fold accumulator")?;
        let next_k = self
            .builder
            .build_int_add(k_val, self.i64t.const_int(1, false), "k.next")
            .unwrap();
        let body_end = self.builder.get_insert_block().unwrap();
        k_phi.add_incoming(&[(&next_k, body_end)]);
        for (phi, c) in acc_phis.iter().zip(next_acc.components()) {
            phi.add_incoming(&[(&c, body_end)]);
        }
        self.builder.build_unconditional_branch(cond).unwrap();

        self.builder.position_at_end(exit);
        Ok(acc_val)
    }

    /// `natCase z s n` as a single native branch: `if n == 0 then z else s (n-1)`.
    /// (No loop, no induction hypothesis — used for case-splits inside `Fix`.)
    fn compile_natcase(
        &self,
        f: FunctionValue<'c>,
        env: &[Slot<'c>],
        z: &Term,
        s: &Term,
        scrut: &Term,
    ) -> Result<Val<'c>, String> {
        let nv = self.compile(f, env, scrut)?;
        let s_body = match strip_ann(s) {
            Term::Lam(b) => &**b,
            _ => return Err("natCase successor is not a function".into()),
        };
        let zero_bb = self.ctx.append_basic_block(f, "case.zero");
        let succ_bb = self.ctx.append_basic_block(f, "case.succ");
        let join_bb = self.ctx.append_basic_block(f, "case.join");
        let is_zero = self
            .builder
            .build_int_compare(inkwell::IntPredicate::EQ, nv, self.i64t.const_zero(), "iszero")
            .unwrap();
        self.builder
            .build_conditional_branch(is_zero, zero_bb, succ_bb)
            .unwrap();

        // compile both arms first (branches deferred), then coerce to one common
        // width — a record arm may be flat (Agg) or boxed (Int); both join.
        self.builder.position_at_end(zero_bb);
        let zv = self.compile_v(f, env, z)?;
        let zero_end = self.builder.get_insert_block().unwrap();

        self.builder.position_at_end(succ_bb);
        let k = self
            .builder
            .build_int_sub(nv, self.i64t.const_int(1, false), "k")
            .unwrap();
        let mut env2 = env.to_vec();
        env2.push(Some(Val::Int(k)));
        let sv = self.compile_v(f, &env2, s_body)?;
        let succ_end = self.builder.get_insert_block().unwrap();

        let w = zv.width().max(sv.width());
        self.builder.position_at_end(zero_end);
        let zc = self.expect_width(zv, w, "natCase zero arm")?.components();
        let zero_end = self.builder.get_insert_block().unwrap();
        self.builder.build_unconditional_branch(join_bb).unwrap();
        self.builder.position_at_end(succ_end);
        let sc = self.expect_width(sv, w, "natCase successor arm")?.components();
        let succ_end = self.builder.get_insert_block().unwrap();
        self.builder.build_unconditional_branch(join_bb).unwrap();

        self.builder.position_at_end(join_bb);
        let mut out: Vec<IntValue<'c>> = Vec::with_capacity(w as usize);
        for kk in 0..w as usize {
            let phi = self.builder.build_phi(self.i64t, "case").unwrap();
            phi.add_incoming(&[(&zc[kk], zero_end), (&sc[kk], succ_end)]);
            out.push(phi.as_basic_value().into_int_value());
        }
        Ok(if w == 1 { Val::Int(out[0]) } else { Val::Agg(out) })
    }

    /// Compile a call to a `Fix` function: evaluate the NON-erased arguments —
    /// each coerced to its DECLARED width and flattened (a flat record becomes
    /// consecutive i64 arguments; erased — multiplicity-0 — arguments are not
    /// passed) — and emit the call. A `ret_width > 1` callee returns an LLVM
    /// `{i64 × w}` aggregate, unpacked back into registers here.
    fn compile_call(
        &self,
        f: FunctionValue<'c>,
        env: &[Slot<'c>],
        func: FunctionValue<'c>,
        params: &[Option<u32>],
        ret_width: u32,
        args: &[&Term],
    ) -> Result<Val<'c>, String> {
        let mut call_args: Vec<inkwell::values::BasicMetadataValueEnum> = Vec::new();
        for (i, a) in args.iter().enumerate() {
            match params.get(i).copied().unwrap_or(Some(1)) {
                None => {} // erased: never compiled, never passed.
                Some(w) => {
                    let v = self.compile_v(f, env, a)?;
                    // an abstract-typed (`Var`-domain) parameter is one slot: a
                    // record argument here is the guided hard error — a single
                    // compiled body cannot serve two layouts.
                    let v = self.expect_width(v, w, "a %partial function argument")?;
                    for c in v.components() {
                        call_args.push(c.into());
                    }
                }
            }
        }
        let ret = self
            .builder
            .build_call(func, &call_args, "call")
            .unwrap()
            .try_as_basic_value()
            .left()
            .unwrap();
        if ret_width == 0 {
            // a zero-width (view-typed) result: the callee returned a dummy.
            return Ok(Val::Agg(Vec::new()));
        }
        if ret_width == 1 {
            return Ok(Val::Int(ret.into_int_value()));
        }
        let agg = ret.into_struct_value();
        let mut comps = Vec::with_capacity(ret_width as usize);
        for k in 0..ret_width {
            comps.push(
                self.builder
                    .build_extract_value(agg, k, "ret.fld")
                    .unwrap()
                    .into_int_value(),
            );
        }
        Ok(Val::Agg(comps))
    }

    /// Build (once, memoized by term identity) the native function for a `Fix`.
    /// `Fix(ty, body)` with `body = λp₁…λpₙ. inner` becomes a recursive function of
    /// the non-erased parameters; self-references in `inner` compile to calls to it.
    /// The `Fix` carries its full Π telescope, so each parameter's WIDTH is known:
    /// a flat-record parameter flattens into consecutive i64 params (registers —
    /// the by-value calling convention), and a record RETURN is an `{i64 × w}`.
    fn build_fix(
        &self,
        fix_term: &Term,
        ty: &Term,
        body: &Term,
    ) -> Result<(FunctionValue<'c>, Vec<Option<u32>>, u32), String> {
        // per-parameter layout: pair each body lambda with its Π binder — `None`
        // for an erased (multiplicity-0) binder, else the domain type's width.
        let mut params: Vec<Option<u32>> = Vec::new();
        let mut t_ty = strip_ann(ty);
        let mut t_body = strip_ann(body);
        while let (Term::Lam(inner), Term::Pi(m, d, cod)) = (t_body, t_ty) {
            if *m == Mult::Zero {
                params.push(None);
            } else {
                params.push(Some(type_width(self.sig, d)));
            }
            t_body = strip_ann(inner);
            t_ty = strip_ann(cod);
        }
        let inner = t_body;
        let ret_width = type_width(self.sig, t_ty);

        let key = fix_term as *const Term;
        if let Some(&func) = self.fix_cache.borrow().get(&key) {
            return Ok((func, params, ret_width));
        }

        let n_scalar: u32 = params.iter().filter_map(|p| *p).sum();
        let param_tys: Vec<inkwell::types::BasicMetadataTypeEnum> =
            std::iter::repeat(self.i64t.into()).take(n_scalar as usize).collect();
        let fname = self.fresh("tally_fix");
        let ret_struct = if ret_width > 1 {
            Some(self.ctx.struct_type(
                &vec![self.i64t.into(); ret_width as usize],
                false,
            ))
        } else {
            None
        };
        let fn_ty = match ret_struct {
            Some(st) => st.fn_type(&param_tys, false),
            None => self.i64t.fn_type(&param_tys, false),
        };
        let func = self.module.add_function(&fname, fn_ty, None);
        self.fix_cache.borrow_mut().insert(key, func);

        // Fresh body env: [self, p₁, …, pₙ] (self at level 0; erased params are
        // None; a width-w param rebuilds its Agg from w consecutive LLVM params).
        let saved = self.builder.get_insert_block();
        let entry = self.ctx.append_basic_block(func, "entry");
        self.builder.position_at_end(entry);
        let mut body_env: Vec<Slot<'c>> = vec![None]; // self
        let mut rt = 0u32;
        for pw in &params {
            match pw {
                None => body_env.push(None),
                Some(1) => {
                    body_env
                        .push(Some(Val::Int(func.get_nth_param(rt).unwrap().into_int_value())));
                    rt += 1;
                }
                Some(w) => {
                    let comps: Vec<IntValue<'c>> = (0..*w)
                        .map(|k| func.get_nth_param(rt + k).unwrap().into_int_value())
                        .collect();
                    body_env.push(Some(Val::Agg(comps)));
                    rt += *w;
                }
            }
        }
        self.fix_selves.borrow_mut().push(FixSelf {
            level: 0,
            func,
            params: params.clone(),
            ret_width,
        });
        let result = self.compile_v(func, &body_env, inner);
        self.fix_selves.borrow_mut().pop();
        let result = self.expect_width(result?, ret_width, "this function's result")?;
        match (&result, ret_struct) {
            (Val::Int(v), None) => {
                self.builder.build_return(Some(v)).unwrap();
            }
            (Val::Agg(comps), None) if comps.is_empty() => {
                // zero-width (view-typed) result: return a dummy scalar; the
                // call site reconstructs the (empty) value.
                self.builder
                    .build_return(Some(&self.i64t.const_zero()))
                    .unwrap();
            }
            (Val::Agg(comps), Some(st)) => {
                let mut agg = st.get_undef();
                for (k, c) in comps.iter().enumerate() {
                    agg = self
                        .builder
                        .build_insert_value(agg, *c, k as u32, "ret.agg")
                        .unwrap()
                        .into_struct_value();
                }
                self.builder.build_return(Some(&agg)).unwrap();
            }
            _ => return Err("fix: return width mismatch (impossible)".into()),
        }
        if let Some(bb) = saved {
            self.builder.position_at_end(bb);
        }
        Ok((func, params, ret_width))
    }

    /// Compile a function-typed-motive `NatElim` applied to its accumulators (Phase
    /// 1a′) as a native recursive function `g(i, a₁…a_K)`:
    ///   g(0,   a…) = z a…
    ///   g(i+1, a…) = s i ih a…       where the IH `ih(a'…)` recurses as `g(i, a'…)`.
    /// The IH is NOT a heap closure: a call `ih(a'…)` in `s`'s body compiles to
    /// `call g(i, a'…)` via the `acc_ih_selves` registry. No heap, no closures — the
    /// exact loop/recursion a C programmer writes for the same accumulator fold, so
    /// an accumulator fold the totality checker certifies total RUNS natively.
    #[allow(clippy::too_many_arguments)]
    fn compile_acc_fold(
        &self,
        f: FunctionValue<'c>,
        env: &[Slot<'c>],
        _motive: &Term,
        z: &Term,
        s: &Term,
        scrut: &Term,
        args: &[&Term],
    ) -> Result<IntValue<'c>, String> {
        // K = number of accumulators = the count of args the NatElim is applied to.
        // The zero method is `λa₁…λa_K. z_body`; the step `λk.λih.λa₁…λa_K. s_body`.
        let k_count = args.len();
        let z_body = peel_n_lams(z, k_count)
            .ok_or("accumulator fold: zero method has too few binders")?;
        let s_body = peel_n_lams(s, k_count + 2)
            .ok_or("accumulator fold: step method has too few binders (k, ih, accumulators)")?;

        // build the native recursive function once, memoized by the step's identity.
        // (Copy the cache hit out FIRST so the immutable borrow is released before the
        // `else` branch takes a mutable borrow to insert.)
        let key = s as *const Term;
        let cached = self.fix_cache.borrow().get(&key).copied();
        let func = if let Some(func) = cached {
            func
        } else {
            let n_params = k_count + 1; // the Nat `i`, then the K accumulators
            let param_tys: Vec<inkwell::types::BasicMetadataTypeEnum> =
                vec![self.i64t.into(); n_params];
            let fname = self.fresh("tally_accfold");
            let func =
                self.module.add_function(&fname, self.i64t.fn_type(&param_tys, false), None);
            self.fix_cache.borrow_mut().insert(key, func);

            let saved = self.builder.get_insert_block();
            // g's body is a FRESH scope with no free references to enclosing Fix/IH
            // selves (no closure capture), so save+clear the registries while building.
            let saved_fix = std::mem::take(&mut *self.fix_selves.borrow_mut());
            let saved_ih = std::mem::take(&mut *self.acc_ih_selves.borrow_mut());

            let i_param = func.get_nth_param(0).unwrap().into_int_value();
            let acc_params: Vec<IntValue<'c>> = (0..k_count)
                .map(|j| func.get_nth_param((j + 1) as u32).unwrap().into_int_value())
                .collect();

            let entry = self.ctx.append_basic_block(func, "entry");
            let zero_bb = self.ctx.append_basic_block(func, "accfold.zero");
            let succ_bb = self.ctx.append_basic_block(func, "accfold.succ");
            self.builder.position_at_end(entry);
            let is_zero = self
                .builder
                .build_int_compare(inkwell::IntPredicate::EQ, i_param, self.i64t.const_zero(), "iszero")
                .unwrap();
            self.builder.build_conditional_branch(is_zero, zero_bb, succ_bb).unwrap();

            // zero: `z a₁…a_K` — env = [a₁ … a_K] (de Bruijn 0 = a_K, the innermost).
            self.builder.position_at_end(zero_bb);
            let zero_env: Vec<Slot<'c>> =
                acc_params.iter().map(|v| Some(Val::Int(*v))).collect();
            let zr0 = self.compile_v(func, &zero_env, z_body)?;
            let zr = self
                .expect_width(
                    zr0,
                    1,
                    "a total accumulator-fold's result (make the function %partial)",
                )?
                .as_int("accumulator-fold result")?;
            self.builder.build_return(Some(&zr)).unwrap();

            // succ: k = i - 1; env = [k, ih, a₁ … a_K]; the IH self sits at level 1.
            self.builder.position_at_end(succ_bb);
            let k_val = self
                .builder
                .build_int_sub(i_param, self.i64t.const_int(1, false), "k")
                .unwrap();
            let mut succ_env: Vec<Slot<'c>> = vec![Some(Val::Int(k_val)), None]; // k (level 0), ih (level 1)
            succ_env.extend(acc_params.iter().map(|v| Some(Val::Int(*v))));
            self.acc_ih_selves.borrow_mut().push(AccIhSelf { ih_level: 1, func, k_level: 0 });
            let sr = self
                .compile_v(func, &succ_env, s_body)
                .and_then(|v| {
                    self.expect_width(
                        v,
                        1,
                        "a total accumulator-fold's result (make the function %partial)",
                    )
                })
                .and_then(|v| v.as_int("accumulator-fold result"));
            self.acc_ih_selves.borrow_mut().pop();
            let sr = sr?;
            self.builder.build_return(Some(&sr)).unwrap();

            *self.fix_selves.borrow_mut() = saved_fix;
            *self.acc_ih_selves.borrow_mut() = saved_ih;
            if let Some(bb) = saved {
                self.builder.position_at_end(bb);
            }
            func
        };

        // call `g(scrut, args…)` with the accumulators evaluated in the CURRENT env
        // (accumulator slots are one scalar; a record acc crosses boxed).
        let scrut_v = self.compile(f, env, scrut)?;
        let mut call_args: Vec<inkwell::values::BasicMetadataValueEnum> = vec![scrut_v.into()];
        for a in args {
            let v = self.compile_v(f, env, a)?;
            call_args.push(
                self.expect_width(
                    v,
                    1,
                    "a total-fold accumulator (make the function %partial — Fix \
                     takes records by value)",
                )?
                .as_int("accumulator arg")?
                .into(),
            );
        }
        Ok(self
            .builder
            .build_call(func, &call_args, "accfold.call")
            .unwrap()
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_int_value())
    }
}

/// Strip up to `n` leading `Lam`s (through `Ann`s); `None` if there are fewer.
fn peel_n_lams(t: &Term, n: usize) -> Option<&Term> {
    let mut t = strip_ann(t);
    for _ in 0..n {
        match t {
            Term::Lam(inner) => t = strip_ann(inner),
            _ => return None,
        }
    }
    Some(t)
}

fn strip_ann(t: &Term) -> &Term {
    match t {
        Term::Ann(e, _) => strip_ann(e),
        other => other,
    }
}

fn flatten_app(t: &Term) -> (&Term, Vec<&Term>) {
    let mut args = Vec::new();
    let mut head = t;
    while let Term::App(f, a) = head {
        args.push(&**a);
        head = f;
    }
    args.reverse();
    (head, args)
}

/// Build the LLVM module for a closed term `main` that reduces to an `i64`,
/// lowering `main` into `tally_dep_main` and running module verification. The
/// caller owns the `Context`; the returned `Module` borrows it. Shared by both
/// the JIT runner (`run_main`) and the IR emitter (`emit_ir`).
fn build_module<'c>(
    ctx: &'c Context,
    sig: &Signature,
    main: &Term,
) -> Result<Module<'c>, String> {
    let module = ctx.create_module("tally_dep");
    let builder = ctx.create_builder();
    let i64t = ctx.i64_type();
    let ptr = ctx.ptr_type(AddressSpace::default());

    let malloc = module.add_function("malloc", ptr.fn_type(&[i64t.into()], false), None);
    let free = module.add_function("free", ctx.void_type().fn_type(&[ptr.into()], false), None);

    let f = module.add_function("tally_dep_main", i64t.fn_type(&[], false), None);
    let bb = ctx.append_basic_block(f, "entry");
    builder.position_at_end(bb);

    let cg = DepCg {
        ctx,
        i64t,
        ptr,
        builder: &builder,
        module: &module,
        malloc,
        free,
        sig,
        next_id: RefCell::new(0),
        fix_cache: RefCell::new(std::collections::HashMap::new()),
        fix_selves: RefCell::new(Vec::new()),
        acc_ih_selves: RefCell::new(Vec::new()),
        elim_fold_ihs: RefCell::new(Vec::new()),
    };
    let result = cg
        .compile_v(f, &[], main)?
        .as_int("main (a program's result must be a scalar)")?;
    builder.build_return(Some(&result)).unwrap();

    if let Err(e) = module.verify() {
        return Err(format!(
            "generated LLVM module failed verification:\n{}\n{}",
            e.to_string(),
            module.print_to_string().to_string()
        ));
    }
    Ok(module)
}

/// Compile a closed term `main` to native and return its LLVM IR as text,
/// WITHOUT executing it. Used to prove the zero-overhead property: the IR is the
/// ground truth that no erased index/proof/region is ever materialized.
pub fn emit_ir(sig: &Signature, main: &Term) -> Result<String, String> {
    let ctx = Context::create();
    let module = build_module(&ctx, sig, main)?;
    Ok(module.print_to_string().to_string())
}

/// JIT-compile and run a closed term that reduces to an `i64`, returning that
/// `i64`. Nat eliminators run as native loops; boxed (Vec/Fin/…) eliminators run
/// as recursive native functions over `malloc`'d cells — no pre-normalization.
pub fn run_main(sig: &Signature, main: &Term) -> Result<i64, String> {
    let ctx = Context::create();
    let module = build_module(&ctx, sig, main)?;
    let ee = module
        .create_jit_execution_engine(OptimizationLevel::None)
        .map_err(|e| e.to_string())?;
    unsafe {
        let func: JitFunction<unsafe extern "C" fn() -> i64> =
            ee.get_function("tally_dep_main").map_err(|e| e.to_string())?;
        Ok(func.call())
    }
}

/// Add a C-ABI `int main()` that runs the compiled program once and prints its
/// `i64` result to stdout, so the module links into a standalone native
/// executable. This is just normal program entry: run `main`, print, exit 0.
fn add_c_main<'c>(ctx: &'c Context, module: &Module<'c>, print_result: bool) {
    let i32t = ctx.i32_type();
    let ptr = ctx.ptr_type(AddressSpace::default());
    let builder = ctx.create_builder();

    // `printf` may already be declared (the `print` postulate) — REUSE it, or a
    // second `add_function` would silently create a renamed `printf.1` that fails
    // to link.
    let printf = module.get_function("printf").unwrap_or_else(|| {
        module.add_function("printf", i32t.fn_type(&[ptr.into()], true), None)
    });
    let dep_main = module
        .get_function("tally_dep_main")
        .expect("tally_dep_main must be built before add_c_main");

    let main = module.add_function("main", i32t.fn_type(&[], false), None);
    let entry = ctx.append_basic_block(main, "entry");
    builder.position_at_end(entry);
    let res = builder
        .build_call(dep_main, &[], "res")
        .unwrap()
        .try_as_basic_value()
        .left()
        .unwrap();
    if print_result {
        let fmt = builder.build_global_string_ptr("%lld\n", "fmt").unwrap();
        builder
            .build_call(printf, &[fmt.as_pointer_value().into(), res.into()], "_")
            .unwrap();
    }
    builder.build_return(Some(&i32t.const_zero())).unwrap();
}

/// Host `TargetMachine` at the requested optimization level. Initializes the
/// native target the first time it is called.
fn host_machine(opt: OptimizationLevel) -> Result<TargetMachine, String> {
    Target::initialize_native(&InitializationConfig::default())
        .map_err(|e| format!("cannot initialize native target: {e}"))?;
    let triple = TargetMachine::get_default_triple();
    let target = Target::from_triple(&triple).map_err(|e| e.to_string())?;
    target
        .create_target_machine(
            &triple,
            &TargetMachine::get_host_cpu_name().to_string(),
            &TargetMachine::get_host_cpu_features().to_string(),
            opt,
            RelocMode::PIC,
            CodeModel::Default,
        )
        .ok_or_else(|| "cannot create host target machine".to_string())
}

/// AOT-compile a closed term `main` to a native object file at `obj_path` with a
/// normal C-ABI entry point (run `main`, print its result, exit 0). Also writes
/// the textual IR alongside (`<obj>.ll`) for inspection, and returns it. `opt`
/// selects the LLVM optimization level.
pub fn build_object(
    sig: &Signature,
    main: &Term,
    obj_path: &Path,
    opt: OptimizationLevel,
) -> Result<String, String> {
    build_object_opts(sig, main, obj_path, opt, true)
}

/// `print_result: false` suppresses the C `main`'s result `printf` — used when
/// the program's `main : Unit` (its output IS its `print`/`prints` effects; a
/// trailing `0` would be noise).
pub fn build_object_opts(
    sig: &Signature,
    main: &Term,
    obj_path: &Path,
    opt: OptimizationLevel,
    print_result: bool,
) -> Result<String, String> {
    let ctx = Context::create();
    let module = build_module(&ctx, sig, main)?;
    add_c_main(&ctx, &module, print_result);

    let machine = host_machine(opt)?;
    module.set_triple(&machine.get_triple());
    module.set_data_layout(&machine.get_target_data().get_data_layout());

    // run a standard optimization pipeline so erasure-clean IR is actually
    // optimized to the same shape a C compiler would produce.
    let opt_str = match opt {
        OptimizationLevel::None => "default<O0>",
        OptimizationLevel::Less => "default<O1>",
        OptimizationLevel::Default => "default<O2>",
        OptimizationLevel::Aggressive => "default<O3>",
    };
    module
        .run_passes(opt_str, &machine, inkwell::passes::PassBuilderOptions::create())
        .map_err(|e| e.to_string())?;

    if let Err(e) = module.verify() {
        return Err(format!("optimized module failed verification:\n{}", e.to_string()));
    }

    let ir = module.print_to_string().to_string();
    if let Some(ll) = obj_path.to_str().map(|s| format!("{s}.ll")) {
        let _ = std::fs::write(&ll, &ir);
    }
    machine
        .write_to_file(&module, FileType::Object, obj_path)
        .map_err(|e| e.to_string())?;
    Ok(ir)
}

#[cfg(test)]
mod tests {
    use crate::rust_surface;

    fn run(src: &str) -> i64 {
        let prog = rust_surface::check_program(src).unwrap_or_else(|e| panic!("{e:?}"));
        let (_, _, body) = prog.defs.iter().find(|(n, _, _)| n == "main").expect("no main");
        super::run_main(&prog.sig, body).unwrap_or_else(|e| panic!("{e}"))
    }

    fn ir(src: &str) -> String {
        let prog = rust_surface::check_program(src).unwrap_or_else(|e| panic!("{e:?}"));
        let (_, _, body) = prog.defs.iter().find(|(n, _, _)| n == "main").expect("no main");
        super::emit_ir(&prog.sig, body).unwrap_or_else(|e| panic!("{e}"))
    }

    #[test]
    #[ignore]
    fn dump_ir() {
        let vec_src = format!(
            "{VEC}\n\
             v3 : Vec Nat (Succ (Succ (Succ Zero)))\nfn v3() {{ Cons(Succ(Succ(Zero)), Cons(Succ(Zero), Cons(Zero, Nil))) }}\n\
             main : Nat\nfn main() {{ vsum(v3) }}\n"
        );
        eprintln!("==== VEC IR ====\n{}", ir(&vec_src));
        let dll = std::fs::read_to_string("examples/dll.rs.tal").unwrap();
        eprintln!("==== DLL IR ====\n{}", ir(&dll));
    }

    // ---- MILESTONE 3: zero-overhead, proven in the emitted LLVM IR ----
    //
    // These tests read the actual generated IR and assert the erasure property
    // directly: the multiplicity-0 length index / proof / region / cursor-identity
    // never become a runtime slot, value, or instruction. The IR is the ground
    // truth — if erasure ever regressed, the byte sizes / call shapes below change.

    /// Count `malloc(i64 N)` calls in the IR, grouped by the byte size N.
    fn malloc_sizes(ir: &str) -> std::collections::BTreeMap<u64, usize> {
        let mut m = std::collections::BTreeMap::new();
        for line in ir.lines() {
            // e.g. "  %cell = call ptr @malloc(i64 24)"
            if let Some(rest) = line.split("@malloc(i64 ").nth(1) {
                if let Some(num) = rest.split(')').next() {
                    if let Ok(n) = num.trim().parse::<u64>() {
                        *m.entry(n).or_insert(0) += 1;
                    }
                }
            }
        }
        m
    }

    #[test]
    fn vec_ir_has_zero_overhead() {
        // A Vec whose constructor carries an ERASED length index `{0 k : Nat}`.
        // If the index were materialized, each Cons cell would need a 4th slot
        // (32 bytes); erasure keeps it at 3 slots = 24 bytes (tag, element, tail).
        let vec_src = format!(
            "{VEC}\n\
             v3 : Vec Nat (Succ (Succ (Succ Zero)))\nfn v3() {{ Cons(Succ(Succ(Zero)), Cons(Succ(Zero), Cons(Zero, Nil))) }}\n\
             main : Nat\nfn main() {{ vsum(v3) }}\n"
        );
        let ir = ir(&vec_src);

        // (1) The ONLY heap traffic is the three Cons cells (24 bytes each: tag +
        // element + tail pointer). `Nil` is nullary, so it is a shared module-level
        // constant — zero allocation — not a per-use cell. No 32-byte cell exists;
        // that would mean a stored (erased) length index.
        let sizes = malloc_sizes(&ir);
        assert_eq!(
            sizes,
            [(24u64, 3usize)].into_iter().collect(),
            "Vec heap traffic must be exactly 3×Cons(24B), with Nil a shared constant; got {sizes:?}\n{ir}"
        );
        assert!(
            ir.contains("@tally_nullary_Nil = constant"),
            "Nil must be a shared module-level constant, not a malloc'd cell\n{ir}"
        );
        assert!(
            !sizes.contains_key(&32),
            "a 32-byte cell means the erased length index leaked into the layout\n{ir}"
        );

        // (2) The erased length-index Nat is never built into the cells: every
        // store into a Cons field is the element or the tail pointer — there is
        // no extra `store i64 3` / `store i64 2` index alongside them. We check
        // the structural fact that the eliminator helper takes exactly the
        // scrutinee plus the captured live env, never a separate "length"
        // argument: the fold counts cons cells at runtime, it is not handed `n`.
        // `vsum` over a 3-element vec must NOT contain the literal length 3 as a
        // materialized index store.
        assert!(
            !ir.contains("store i64 3,"),
            "the length index 3 was materialized into the heap (not erased)\n{ir}"
        );

        // (3) alloc/free of the boxed data go through real libc malloc; there is
        // no bespoke allocator and no boxing of the (erased) type parameter `a`.
        assert!(ir.contains("call ptr @malloc("), "Vec data must use libc malloc\n{ir}");

        // (4) Sanity: it still computes the right answer when actually run.
        assert_eq!(run(&vec_src), 3);
    }

    #[test]
    fn dll_ir_has_zero_overhead() {
        // examples/dll.rs.tal: build an empty circular sentinel DLL, insert `1`,
        // remove it by its cursor in O(1), free the list, return the value.
        let dll = std::fs::read_to_string("examples/dll.rs.tal").unwrap();
        let ir = ir(&dll);

        // (1) The ghost region machinery is multiplicity-0 and must leave NO
        // trace in the runtime IR: no Region/R0/Cursor value is allocated,
        // loaded, stored, or named anywhere. (These are postulate names; if any
        // appeared as a runtime symbol it would be an abstract value at runtime.)
        for ghost in ["Region", "@R0", "Cursor", "tally_elim_Region", "tally_elim_Cursor"] {
            assert!(
                !ir.contains(ghost),
                "ghost `{ghost}` leaked into the runtime IR — it must be fully erased\n{ir}"
            );
        }

        // (2) alloc/free are DIRECT libc calls — the linear memory layer is not
        // simulated; `free` is a real `call void @free(ptr ...)` and the only
        // allocations are real `call ptr @malloc(...)`.
        assert!(
            ir.contains("call void @free(ptr "),
            "remove/free must lower to a direct libc free\n{ir}"
        );
        assert!(
            ir.contains("call ptr @malloc(i64 "),
            "new/insert must lower to a direct libc malloc\n{ir}"
        );

        // (3) The only heap cells are real nodes (24 bytes: next, prev, elem),
        // the sentinel (24 bytes), and the transient linear-pair boxes (24 bytes:
        // tag + two fields). There is NO region cell and NO cursor-identity cell:
        // every malloc is 24 bytes; not one carries an erased region/proof slot.
        let sizes = malloc_sizes(&ir);
        assert!(
            sizes.keys().all(|&n| n == 24),
            "every DLL heap cell must be 24 bytes (node/sentinel/pair); got {sizes:?}\n{ir}"
        );

        // (4) The cursor is just a node pointer reused for the O(1) unlink: the
        // SAME ssa value that the remove loads as the cursor field is the pointer
        // passed to free — no region check, no identity comparison, no extra
        // indirection. We verify there is exactly one free per removed node and
        // one free per freed list (two frees total: the node, then the list).
        let nfree = ir.matches("call void @free(").count();
        assert_eq!(
            nfree, 2,
            "expected exactly 2 frees (the removed node + the freed list); got {nfree}\n{ir}"
        );

        // (5) Sanity: it still runs to the removed value 1.
        assert_eq!(run(&dll), 1);
    }

    // ---- AOT: build_object emits a real native executable ----

    /// Compile `src`'s `main` to an object via `build_object`, link it with `cc`,
    /// run it, and return its stdout's first line as an i64 — exactly what a user
    /// gets from `tally build` then running the executable.
    fn aot_run(src: &str, label: &str) -> i64 {
        use inkwell::OptimizationLevel;
        let prog = rust_surface::check_program(src).unwrap_or_else(|e| panic!("{e:?}"));
        let (_, _, body) = prog.defs.iter().find(|(n, _, _)| n == "main").expect("no main");
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let stem = format!("tally_aot_{pid}_{label}");
        let obj = dir.join(format!("{stem}.o"));
        let exe = dir.join(&stem);
        super::build_object(&prog.sig, body, &obj, OptimizationLevel::Default)
            .unwrap_or_else(|e| panic!("build_object: {e}"));
        let link = std::process::Command::new("cc")
            .arg(&obj)
            .arg("-o")
            .arg(&exe)
            .status()
            .expect("invoke cc");
        assert!(link.success(), "cc failed");
        let out = std::process::Command::new(&exe).output().expect("run exe");
        let _ = std::fs::remove_file(&obj);
        let _ = std::fs::remove_file(exe.with_extension("o.ll"));
        let _ = std::fs::remove_file(&exe);
        assert!(out.status.success(), "exe exited non-zero");
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .next()
            .unwrap()
            .trim()
            .parse()
            .unwrap()
    }

    /// AOT-build + run, returning the FULL stdout (for `print`-effect tests).
    fn aot_stdout(src: &str, label: &str) -> String {
        use inkwell::OptimizationLevel;
        let prog = rust_surface::check_program(src).unwrap_or_else(|e| panic!("{e:?}"));
        let (_, _, body) = prog.defs.iter().find(|(n, _, _)| n == "main").expect("no main");
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let stem = format!("tally_aot_{pid}_{label}");
        let obj = dir.join(format!("{stem}.o"));
        let exe = dir.join(&stem);
        super::build_object(&prog.sig, body, &obj, OptimizationLevel::Default)
            .unwrap_or_else(|e| panic!("build_object: {e}"));
        let link = std::process::Command::new("cc")
            .arg(&obj)
            .arg("-o")
            .arg(&exe)
            .status()
            .expect("invoke cc");
        assert!(link.success(), "cc failed");
        let out = std::process::Command::new(&exe).output().expect("run exe");
        let _ = std::fs::remove_file(&obj);
        let _ = std::fs::remove_file(exe.with_extension("o.ll"));
        let _ = std::fs::remove_file(&exe);
        assert!(out.status.success(), "exe exited non-zero");
        String::from_utf8_lossy(&out.stdout).to_string()
    }

    #[test]
    fn print_effect_writes_to_stdout() {
        // `print : Nat -> Unit` — the first observable effect. CBV-`let`
        // sequencing runs each print exactly once, in order; the process's
        // stdout carries the numbers before the harness's own result line.
        let src = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
            main : Nat\nfn main() { let a = print(1); let b = print(2); let c = print(42); 0 }\n";
        let out = aot_stdout(src, "print");
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(&lines[..3], &["1", "2", "42"], "full stdout: {out:?}");
    }

    #[test]
    fn aot_dll_transaction_executable() {
        // examples/dll.rs.tal AOT-linked to a native exe: the transaction removes
        // and returns the value 1.
        let dll = std::fs::read_to_string("examples/dll.rs.tal").unwrap();
        assert_eq!(aot_run(&dll, "dll"), 1);
    }

    #[test]
    fn aot_memory_executable() {
        // examples/memory.rs.tal: alloc then free, returns Unit (represented 0).
        let src = std::fs::read_to_string("examples/memory.rs.tal").unwrap();
        assert_eq!(aot_run(&src, "memory"), 0);
    }

    const NAT: &str = r#"
enum Nat { Zero : Nat, Succ : Nat -> Nat }
add : Nat -> Nat -> Nat
fn add(m, n) { match m { Zero => n, Succ(k) => Succ(add(k, n)) } }
mul : Nat -> Nat -> Nat
fn mul(m, n) { match m { Zero => Zero, Succ(k) => add(n, mul(k, n)) } }
"#;

    #[test]
    fn nat_add_runs_natively() {
        // add 2 3 — the eliminator runs as a native loop, returning 5
        let src = format!("{NAT}\nmain : Nat\nfn main() {{ add(Succ(Succ(Zero)), Succ(Succ(Succ(Zero)))) }}\n");
        assert_eq!(run(&src), 5);
    }

    #[test]
    fn nat_mul_runs_natively() {
        // mul 3 4 = 12 — nested eliminators (mul calls add), all native
        let src = format!(
            "{NAT}\nmain : Nat\nfn main() {{ mul(Succ(Succ(Succ(Zero))), Succ(Succ(Succ(Succ(Zero))))) }}\n"
        );
        assert_eq!(run(&src), 12);
    }

    // ---- `%builtin Nat`: the packed (Idris-2-style) integer representation ----

    const BNAT: &str = r#"
%builtin Nat Nat
enum Nat { Zero : Nat, Succ : Nat -> Nat }
add : Nat -> Nat -> Nat
fn add(m, n) { match m { Zero => n, Succ(k) => Succ(add(k, n)) } }
mul : Nat -> Nat -> Nat
fn mul(m, n) { match m { Zero => Zero, Succ(k) => add(n, mul(k, n)) } }
"#;

    #[test]
    fn builtin_nat_literals_and_add() {
        // literals and `+` are packed Nat; `2 + 3 + 4 = 9`.
        let src = format!("{BNAT}\nmain : Nat\nfn main() {{ 2 + 3 + 4 }}\n");
        assert_eq!(run(&src), 9);
    }

    #[test]
    fn builtin_nat_no_overflow_at_scale() {
        // `mul(1000, 1000)` = 1_000_000. With a UNARY Nat this overflows the
        // checker's evaluator; with `%builtin Nat` it normalizes on machine ints.
        let src = format!("{BNAT}\nmain : Nat\nfn main() {{ mul(1000, 1000) }}\n");
        assert_eq!(run(&src), 1_000_000);
    }

    #[test]
    fn builtin_nat_match_is_native() {
        // `match` on a `%builtin Nat` lowers to NatElim (a native loop). A literal
        // pattern-driven recursion summing 0..n via `add`.
        let src = format!(
            "{BNAT}\n\
             sumto : Nat -> Nat\n\
             fn sumto(n) {{ match n {{ Zero => 0, Succ(k) => add(Succ(k), sumto(k)) }} }}\n\
             main : Nat\nfn main() {{ sumto(100) }}\n"
        );
        assert_eq!(run(&src), 5050);
    }

    // ---- boxed inductive with an INTERLEAVED recursive arg (binary tree) ----

    const TREE: &str = r#"
%builtin Nat Nat
enum Nat { Zero : Nat, Succ : Nat -> Nat }
boxed enum Tree { Leaf : Tree, Node : Tree -> Nat -> Tree -> Tree }
build : Nat -> Nat -> Tree
fn build(d, x) { match d { Zero => Leaf, Succ(k) => Node(build(k, x), x, build(k, x)) } }
tsum : Tree -> Nat
fn tsum(t) { match t { Leaf => 0, Node(l, x, r) => tsum(l) + x + tsum(r) } }
"#;

    #[test]
    fn general_recursion_builds_distinct_tree() {
        // GENERAL recursion (a `Fix`, not a fold): `build` recurses with a DIFFERENT
        // label on each side, so the two subtrees are distinct and 2^d - 1 separate
        // nodes are allocated. (A fold would reuse one induction hypothesis for both
        // children — a shared DAG.) Sum of the distinct labels: d=1→1, d=2→1+2+3=6,
        // d=3→28, d=4→120.
        const G: &str = r#"
%builtin Nat Nat
enum Nat { Zero : Nat, Succ : Nat -> Nat }
boxed enum Tree { Leaf : Tree, Node : Tree -> Nat -> Tree -> Tree }
build : Nat -> Nat -> Tree
fn build(d, label) { match d { Zero => Leaf, Succ(k) => Node(build(k, label + label), label, build(k, label + label + 1)) } }
tsum : Tree -> Nat
fn tsum(t) { match t { Leaf => 0, Node(l, x, r) => tsum(l) + x + tsum(r) } }
"#;
        for (d, expect) in [(1u64, 1i64), (2, 6), (3, 28), (4, 120)] {
            let src = format!("{G}\nmain : Nat\nfn main() {{ tsum(build({d}, 1)) }}\n");
            assert_eq!(run(&src), expect, "depth {d}");
        }
    }

    #[test]
    fn interpreter_mvp_arithmetic_runs_natively() {
        // THE DOGFOOD, MVP-0: a real program — an interpreter for an arithmetic expression
        // language — written in surface Tally. The AST is an OWNED tree (`Own Expr`
        // children, the (a)-positivity shape); `eval` is `%partial` heap recursion (the (A)
        // capability) that UNBOX-CONSUMES the tree — freeing each node as it evaluates —
        // with the recursive results `let`-sequenced. Evaluating `1 + (2 * 3)` builds the
        // tree on the heap, walks + frees it, and returns 7. Memory-safe, zero-GC, the
        // differentiator applied to a genuinely complicated program.
        let src = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
            boxed enum Expr { lit : Nat -> Expr, add : Own Expr -> Own Expr -> Expr, mul : Own Expr -> Own Expr -> Expr }\n\
            plus : Nat -> Nat -> Nat\nfn plus(m, n) { match m { Zero => n, Succ(k) => Succ(plus(k, n)) } }\n\
            times : Nat -> Nat -> Nat\nfn times(m, n) { match m { Zero => Zero, Succ(k) => plus(n, times(k, n)) } }\n\
            eval : Own Expr -> Nat\n\
            fn eval(e) { match unbox(e) { lit(n) => n, add(a, b) => let va = eval(a); let vb = eval(b); plus(va, vb), mul(a, b) => let va = eval(a); let vb = eval(b); times(va, vb) } }\n\
            mkexpr : Own Expr\n\
            fn mkexpr() { alloc(add(alloc(lit(Succ(Zero))), alloc(mul(alloc(lit(Succ(Succ(Zero)))), alloc(lit(Succ(Succ(Succ(Zero))))))))) }\n\
            main : Nat\nfn main() { eval(mkexpr) }\n";
        assert_eq!(run(src), 7);
    }

    #[test]
    fn owned_list_traversal_runs_natively() {
        // (A) the full interpreter-relevant shape: arbitrary-length recursion over a
        // LINEAR OWNED heap structure, BUILDING + TRAVERSING + FREEING it. `sumFree`
        // recurses on the unbox'd tail (a `%partial` `Fix` whose boxed matches are
        // `Term::Case`), freeing each node via `unbox` and summing the heads. The
        // recursive result is `let`-SEQUENCED (`let s = sumFree(t); add(h, s)`): passing
        // `sumFree(t)` directly to `add`'s `ω` Nat argument would `ω`-scale the linear
        // `t`'s consumption (`ω⋢1`) — the CBV-`let` counts `e` once, the correct
        // discipline for sequencing a linear consumption into an unrestricted position.
        // Builds [1,2], frees both nodes, sums → 3. Memory-safe, zero-GC, `%partial`.
        let src = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
            enum Opt (a : Type) { none : Opt a, some : a -> Opt a }\n\
            struct Node { head : Nat, tail : Opt (Own Node) }\n\
            add : Nat -> Nat -> Nat\nfn add(m, n) { match m { Zero => n, Succ(k) => Succ(add(k, n)) } }\n\
            sumFree : Opt (Own Node) -> Nat\n\
            fn sumFree(l) { match l { none => Zero, some(o) => match unbox(o) { Node(h, t) => let s = sumFree(t); add(h, s) } } }\n\
            mklist : Opt (Own Node)\n\
            fn mklist() { some(alloc(Node(Succ(Zero), some(alloc(Node(Succ(Succ(Zero)), none)))))) }\n\
            main : Nat\nfn main() { sumFree(mklist) }\n";
        assert_eq!(run(src), 3);
    }

    #[test]
    fn dependent_partial_recursion_with_implicit_runs_natively() {
        // (A)-extension (piece 2 of the running dependent eval): a %partial recursive
        // function over a LENGTH-INDEXED Vec, with an INFERRABLE implicit `{0 n}` and a
        // SELF-CALL. `vlast` recurses on the tail (varying the accumulator) ⇒ %partial ⇒
        // lowers to a Fix; the self-call `vlast(t, h)` has the implicit `n` (inferred from
        // `t : Vec Nat k`) and lands in CHECK position — previously it hit
        // def_has_implicits → solve_fn_call → panic (the Fix self-binder isn't in `defs`).
        // Now solve_fn_call resolves a not-in-defs callee as the in-scope (Fix self) binder
        // and infers the implicit. `vlast v3 0` = the last element = 3.
        let src = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
            boxed enum Vec (a : Type) : Nat -> Type { Nil : Vec a Zero, Cons : {0 k : Nat} -> a -> Vec a k -> Vec a (Succ k) }\n\
            vlast : {0 n : Nat} -> Vec Nat n -> Nat -> Nat\n\
            fn vlast(v, acc) { match v { Nil => acc, Cons(h, t) => vlast(t, h) } }\n\
            v3 : Vec Nat (Succ (Succ (Succ Zero)))\nfn v3() { Cons(Succ(Zero), Cons(Succ(Succ(Zero)), Cons(Succ(Succ(Succ(Zero))), Nil))) }\n\
            main : Nat\nfn main() { vlast(v3, Zero) }\n";
        assert_eq!(run(src), 3);
    }

    #[test]
    fn heap_general_recursion_runs_natively() {
        // (A) HEAP RECURSION: a `%partial` fn recursing on a BOXED/heap structure — here a
        // boxed accumulator fold (the accumulator VARIES, so it is general recursion, not a
        // structural fold) — now lowers (previously "general recursion only on a %builtin
        // Nat scrutinee" hard-errored). It compiles to an opaque `Fix` whose `match`
        // dispatches via the NON-recursive `Term::Case` (NOT the eliminator, whose implicit
        // IHs would be exponential), with recursion as the `Fix` self-call. Runs natively:
        // sumAcc [1,2] 0 = 3. This is the unblock that lets the interpreter (eval over an
        // AST) run as `%partial`.
        let src = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
            boxed enum List { nil : List, cons : Nat -> List -> List }\n\
            add : Nat -> Nat -> Nat\nfn add(m, n) { match m { Zero => n, Succ(k) => Succ(add(k, n)) } }\n\
            sumAcc : List -> Nat -> Nat\n\
            fn sumAcc(l, acc) { match l { nil => acc, cons(h, t) => sumAcc(t, add(acc, h)) } }\n\
            mk : List\nfn mk() { cons(Succ(Zero), cons(Succ(Succ(Zero)), nil)) }\n\
            main : Nat\nfn main() { sumAcc(mk, Zero) }\n";
        assert_eq!(run(src), 3);
    }

    #[test]
    fn differentiator_demo_owned_linked_list_runs_natively() {
        // THE DIFFERENTIATOR DEMO — the first safe manual-memory linked structure in
        // surface Tally. A 2-node OWNED linked list `[1, 2]` (`struct Node { head : Nat,
        // tail : Opt (Own Node) }`, recursion through the (a)-verified positivity):
        //   • BUILT on the heap with `alloc` (box) — `Node(1, some(alloc(Node(2, none))))`,
        //   • TRAVERSED + FREED with `unbox` (deref-and-consume: load the pointee, free
        //     the cell, move the contents out), summing the heads,
        //   • TYPE-CHECKED memory-safe: every `Own` is consumed EXACTLY ONCE on EVERY
        //     path — including the dead match arms, which must still `free` their owned
        //     binders (that obligation IS the safety guarantee; omitting it is a leak,
        //     `0⋢1`; reusing one is a double-free, `ω⋢1` — both rejected, see the
        //     rust_surface red-team tests),
        //   • RUNS natively to `1 + 2 = 3`, ZERO GC, fully manual reclamation.
        // This proves the thesis: the linear-dependent TYPE SYSTEM enforces manual-memory
        // obligations soundly, end-to-end, running. (Byte-level zero-leak of the inner
        // boxed struct-cells awaits Phase B's inline value-layout — the all-boxed codegen
        // double-boxes `Own`-of-struct today; the TYPE-LEVEL safety + correct run is the
        // milestone proven here.)
        let src = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
            enum Opt (a : Type) { none : Opt a, some : a -> Opt a }\n\
            struct Node { head : Nat, tail : Opt (Own Node) }\n\
            add : Nat -> Nat -> Nat\nfn add(m, n) { match m { Zero => n, Succ(k) => Succ(add(k, n)) } }\n\
            main : Nat\n\
            fn main() { \
              match unbox(alloc(Node(Succ(Zero), some(alloc(Node(Succ(Succ(Zero)), none)))))) { \
                Node(h1, t1) => match t1 { \
                  none => Zero, \
                  some(o2) => match unbox(o2) { Node(h2, t2) => match t2 { \
                    none => add(h1, h2), \
                    some(o3) => match free(o3) { U => Zero } \
                  } } \
                } \
              } \
            }\n";
        assert_eq!(run(src), 3);
    }

    #[test]
    fn cbv_let_multi_owner_free_runs_natively() {
        // The CALL-BY-VALUE `let` compiles to the LLVM backend: `e` runs ONCE (its
        // effects happen exactly once), then the body. Multi-owner sequencing — free
        // TWO heap cells, then return 5 — runs natively (proves `free` actually fires
        // once per owner and the sequencing is real, not just kernel-evaluator typing).
        let src = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
            g : Own Nat -> Own Nat -> Nat\n\
            fn g(x, y) { let u = free(x); let v = free(y); Succ(Succ(Succ(Succ(Succ(Zero))))) }\n\
            main : Nat\nfn main() { g(alloc(Zero), alloc(Zero)) }\n";
        assert_eq!(run(src), 5);
    }

    #[test]
    fn accumulator_folds_run_natively() {
        // PHASE 1a′ NATIVE BACKEND: a `%total` accumulator fold (descends on the
        // scrutinee, VARIES another argument) lowers to a function-typed-motive
        // `NatElim` that now compiles to a native recursive function (no closures) —
        // so what the totality checker certifies total RUNS on the LLVM backend, not
        // only in the kernel evaluator.
        const NB: &str = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n";

        // single accumulator: addacc(m,n) = m+n, sumacc(m,acc) = acc+m.
        let src = format!(
            "{NB}\n%total fn addacc(m, n) {{ match m {{ Zero => n, Succ(k) => addacc(k, Succ(n)) }} }}\n\
             addacc : Nat -> Nat -> Nat\nmain : Nat\nfn main() {{ addacc(2, 3) }}\n"
        );
        assert_eq!(run(&src), 5, "addacc(2,3)");

        let src = format!(
            "{NB}\n%total fn sumacc(m, acc) {{ match m {{ Zero => acc, Succ(k) => sumacc(k, Succ(acc)) }} }}\n\
             sumacc : Nat -> Nat -> Nat\nmain : Nat\nfn main() {{ sumacc(7, 0) }}\n"
        );
        assert_eq!(run(&src), 7, "sumacc(7,0)");

        // two accumulators, both varying: twoacc(2,0,10) = 1.
        let src = format!(
            "{NB}\n%total fn twoacc(m, a, b) {{ match m {{ Zero => a, Succ(k) => twoacc(k, b, Succ(a)) }} }}\n\
             twoacc : Nat -> Nat -> Nat -> Nat\nmain : Nat\nfn main() {{ twoacc(2, 0, 10) }}\n"
        );
        assert_eq!(run(&src), 1, "twoacc(2,0,10)");

        // nested-match accumulator: sub(m,n) = m - n (truncated). sub(5,2)=3, sub(2,5)=0.
        const SUB: &str = "%total fn sub(m, n) { match m { \
              Zero => Zero, Succ(j) => match n { Zero => Succ(j), Succ(k) => sub(j, k) } } }\n\
              sub : Nat -> Nat -> Nat\n";
        assert_eq!(run(&format!("{NB}\n{SUB}main : Nat\nfn main() {{ sub(5, 2) }}\n")), 3, "sub(5,2)");
        assert_eq!(run(&format!("{NB}\n{SUB}main : Nat\nfn main() {{ sub(2, 5) }}\n")), 0, "sub(2,5)");
    }

    #[test]
    fn fuel_div_runs_natively() {
        // THE 1a′ PROOF TARGET, now ACTUALLY on the LLVM backend: `%total fuel-div`
        // composing the accumulator folds `lt`/`sub` and the fuel-driven `div`.
        // div(10,7,2) = 3 (7/2 with enough fuel). lt returns a boxed Bool.
        const PRE: &str = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
                           enum Bool { False : Bool, True : Bool }\n";
        let src = format!(
            "{PRE}\n\
             %total fn lt(m, n) {{ match m {{ \
                 Zero => match n {{ Zero => False, Succ(x) => True }}, \
                 Succ(j) => match n {{ Zero => False, Succ(k) => lt(j, k) }} }} }}\nlt : Nat -> Nat -> Bool\n\
             %total fn sub(m, n) {{ match m {{ \
                 Zero => Zero, Succ(j) => match n {{ Zero => Succ(j), Succ(k) => sub(j, k) }} }} }}\nsub : Nat -> Nat -> Nat\n\
             %total fn div(fuel, n, d) {{ match fuel {{ \
                 Zero => Zero, Succ(f) => match lt(n, d) {{ True => Zero, False => Succ(div(f, sub(n, d), d)) }} }} }}\ndiv : Nat -> Nat -> Nat -> Nat\n\
             main : Nat\nfn main() {{ div(10, 7, 2) }}\n"
        );
        assert_eq!(run(&src), 3, "div(10,7,2)");
    }

    #[test]
    fn tree_build_and_traverse() {
        // `Node : Tree -> Nat -> Tree -> Tree` has a recursive arg that is NOT last,
        // so its eliminator method is `λl. λx. λr. λih_l. λih_r. …` (args-then-IHs).
        // This used to be miscompiled (the backend/`velim` interleaved the IHs while
        // the type said args-then-IHs), reading the value field as a pointer. Build a
        // depth-d binary tree of 1s and sum every node: the result is 2^d - 1.
        for (d, expect) in [(0u64, 0i64), (1, 1), (5, 31), (10, 1023)] {
            let src = format!("{TREE}\nmain : Nat\nfn main() {{ tsum(build({d}, 1)) }}\n");
            assert_eq!(run(&src), expect, "depth {d}");
        }
    }

    // ---- boxed inductive families: Vec and Fin ----

    const VEC: &str = r#"
enum Nat { Zero : Nat, Succ : Nat -> Nat }
add : Nat -> Nat -> Nat
fn add(m, n) { match m { Zero => n, Succ(k) => Succ(add(k, n)) } }
boxed enum Vec (a : Type) : Nat -> Type {
    Nil  : Vec a Zero,
    Cons : {0 k : Nat} -> a -> Vec a k -> Vec a (Succ k),
}
vsum : {0 n : Nat} -> Vec Nat n -> Nat
fn vsum(xs) { match xs { Nil => Zero, Cons(h, t) => add(h, vsum(t)) } }
"#;

    #[test]
    fn dependent_vec_sum_runs() {
        // build the boxed Vec [10, 20, 30] and fold it to the Nat 60.
        let src = format!(
            "{VEC}\n\
             ten : Nat\nfn ten() {{ Succ(Succ(Succ(Succ(Succ(Succ(Succ(Succ(Succ(Succ(Zero)))))))))) }}\n\
             twenty : Nat\nfn twenty() {{ add(ten, ten) }}\n\
             thirty : Nat\nfn thirty() {{ add(ten, twenty) }}\n\
             v3 : Vec Nat (Succ (Succ (Succ Zero)))\nfn v3() {{ Cons(ten, Cons(twenty, Cons(thirty, Nil))) }}\n\
             main : Nat\nfn main() {{ vsum(v3) }}\n"
        );
        assert_eq!(run(&src), 60);
    }

    #[test]
    fn dependent_vec_length_runs() {
        // a length fold: count the cons cells of a boxed Vec, ignoring contents.
        let src = format!(
            "{VEC}\n\
             vlen : {{0 n : Nat}} -> Vec Nat n -> Nat\n\
             fn vlen(xs) {{ match xs {{ Nil => Zero, Cons(h, t) => Succ(vlen(t)) }} }}\n\
             v3 : Vec Nat (Succ (Succ (Succ Zero)))\n\
             fn v3() {{ Cons(Zero, Cons(Zero, Cons(Zero, Nil))) }}\n\
             main : Nat\nfn main() {{ vlen(v3) }}\n"
        );
        assert_eq!(run(&src), 3);
    }

    #[test]
    fn dependent_vec_head_runs() {
        // vhead: read the FIRST runtime field of a Cons cell (a non-recursive use
        // of the eliminator that ignores the tail / IH). `Nil` returns a default.
        let src = format!(
            "{VEC}\n\
             vhead : {{0 n : Nat}} -> Vec Nat n -> Nat\n\
             fn vhead(xs) {{ match xs {{ Nil => Zero, Cons(h, t) => h }} }}\n\
             v3 : Vec Nat (Succ (Succ (Succ Zero)))\n\
             fn v3() {{ Cons(Succ(Succ(Succ(Succ(Succ(Succ(Succ(Zero))))))), Cons(Succ(Succ(Zero)), Cons(Zero, Nil))) }}\n\
             main : Nat\nfn main() {{ vhead(v3) }}\n"
        );
        assert_eq!(run(&src), 7);
    }

    const FIN: &str = r#"
enum Nat { Zero : Nat, Succ : Nat -> Nat }
boxed enum Fin : Nat -> Type {
    FZ : {0 n : Nat} -> Fin (Succ n),
    FS : {0 n : Nat} -> Fin n -> Fin (Succ n),
}
fin2nat : {0 n : Nat} -> Fin n -> Nat
fn fin2nat(i) { match i { FZ => Zero, FS(prev) => Succ(fin2nat(prev)) } }
"#;

    #[test]
    fn fin_to_nat_runs() {
        // the element "2" of Fin 3 (FS (FS FZ)) forgets its bound to the Nat 2.
        let src = format!(
            "{FIN}\n\
             two : Fin (Succ (Succ (Succ Zero)))\nfn two() {{ FS(FS(FZ)) }}\n\
             main : Nat\nfn main() {{ fin2nat(two) }}\n"
        );
        assert_eq!(run(&src), 2);
    }

    // ---- the LINEAR memory layer: postulates with native implementations ----

    #[test]
    fn linear_alloc_free_runs() {
        // examples/memory.rs.tal: main = free(alloc(Zero)). `alloc` mallocs a
        // cell and stores the payload; `free` is libc free; `Own` is erased to a
        // bare pointer. The whole thing returns Unit (== 0).
        let src = std::fs::read_to_string("examples/memory.rs.tal").unwrap();
        assert_eq!(run(&src), 0);
    }

    #[test]
    fn dependent_dll_remove_runs() {
        // examples/dll.rs.tal: build an empty intrusive circular DLL, insert the
        // Nat `Succ(Zero)` (== 1), remove that node by its cursor in O(1), free
        // the list, and return the removed value. The region and cursor's
        // list-identity are erased (multiplicity 0); only the node pointers
        // exist at runtime.
        let src = std::fs::read_to_string("examples/dll.rs.tal").unwrap();
        assert_eq!(run(&src), 1);
    }

    #[test]
    fn fin_zero_runs() {
        // FZ : Fin (Succ n) forgets to 0.
        let src = format!(
            "{FIN}\n\
             z3 : Fin (Succ (Succ (Succ Zero)))\nfn z3() {{ FZ }}\n\
             main : Nat\nfn main() {{ fin2nat(z3) }}\n"
        );
        assert_eq!(run(&src), 0);
    }

    // ---- the VIEW LAYER (docs/02): L3 address/permission split, running natively.
    // `valloc`/`vwrite`/`vread`/`vfree` are prelude built-ins with real
    // malloc/store/free lowering (see `compile_postulate`). ----

    #[test]
    fn view_alloc_write_read_runs() {
        // valloc a cell holding 0, STRONG-UPDATE it to 2 in place, then read the
        // value back (reclaiming the cell). Native malloc → store → load → free.
        let src = "enum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
                   main : Nat\n\
                   fn main() { match valloc(Zero) { MkCell(p, v) => vread(p, vwrite(p, v, Succ(Succ(Zero)))), } }\n";
        assert_eq!(run(src), 2);
    }

    #[test]
    fn view_strong_update_changes_type_runs() {
        // a Bool cell strong-updated to a Nat 5, then read — the stored TYPE
        // changes in place (Bool@l → Nat@l), sound because the view is linear.
        let src = "enum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
                   enum Bool { False : Bool, True : Bool }\n\
                   main : Nat\n\
                   fn main() { match valloc(False) { MkCell(p, v) => vread(p, vwrite(p, v, Succ(Succ(Succ(Succ(Succ(Zero))))))), } }\n";
        assert_eq!(run(src), 5);
    }

    #[test]
    fn view_take_modify_write_read_runs() {
        // READ-MODIFY-WRITE via `vtake`: move 2 out (slot → Hole), write 3 back,
        // read it. `vtake` is a move, so it is sound for any payload. → 3.
        let src = "enum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
                   main : Nat\n\
                   fn main() { match valloc(Succ(Succ(Zero))) { MkCell(p, v) => \
                       let (x, vh) = vtake(p, v); let v2 = vwrite(p, vh, Succ(x)); vread(p, v2), } }\n";
        assert_eq!(run(src), 3);
    }

    #[test]
    fn view_take_then_free_hole_runs() {
        // move the value out, then reclaim the (Hole) cell directly; return the
        // moved value. → 1.
        let src = "enum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
                   main : Nat\n\
                   fn main() { match valloc(Succ(Zero)) { MkCell(p, v) => \
                       let (x, vh) = vtake(p, v); let u = vfree(p, vh); x, } }\n";
        assert_eq!(run(src), 1);
    }

    #[test]
    fn convoy_dependent_lookup_runs_natively() {
        // THE CONVOY end-to-end (docs/CONVOY_HANDOFF.md): total-coverage,
        // bounds-check-free `lookup : {0 n} -> Fin n -> Vec Nat n -> Nat`,
        // compiled and RUN. The typed term is `(Case …) applied to the deps`;
        // codegen commutes the application into the arms and emits the ordinary
        // tag-switch — `n`, the `Fin` bound, and the motive all erase.
        // lookup [1,2,3] at index FS(FZ) = 2. (See examples/lookup.tal.)
        let src = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
            boxed enum Vec (a : Type) : Nat -> Type { Nil : Vec a Zero, Cons : {0 k : Nat} -> a -> Vec a k -> Vec a (Succ k) }\n\
            boxed enum Fin : Nat -> Type { FZ : {0 n : Nat} -> Fin (Succ n), FS : {0 n : Nat} -> Fin n -> Fin (Succ n) }\n\
            enum Void { }\n\
            exfalso : {0 a : Type} -> Void -> a\nfn exfalso(v) { match v { } }\n\
            fzv : Fin Zero -> Void\nfn fzv(f) { match f { } }\n\
            lookup : {0 n : Nat} -> Fin n -> Vec Nat n -> Nat\n\
            fn lookup(i, env) { match env { Nil => exfalso(fzv(i)), Cons(v, rest) => match i { FZ => v, FS(j) => lookup(j, rest) } } }\n\
            env3 : Vec Nat (Succ (Succ (Succ Zero)))\nfn env3() { Cons(1, Cons(2, Cons(3, Nil))) }\n\
            i1 : Fin (Succ (Succ (Succ Zero)))\nfn i1() { FS(FZ) }\n\
            main : Nat\nfn main() { lookup(i1, env3) }\n";
        assert_eq!(run(src), 2);
    }

    #[test]
    fn dependent_eval_runs_natively() {
        // THE COMPLETE RUNNING DEPENDENT EVAL (examples/scoped_eval.tal): a
        // well-scoped-by-typing interpreter — depth-indexed AST (`Expr d`,
        // `var : Fin d -> Expr d` makes out-of-scope UNREPRESENTABLE), a
        // length-indexed environment, bounds-check-free total `lookup` (absurd
        // `Nil` case discharged via the convoy), linear `Own` children freed
        // exactly once as `eval` walks. env = [10, 20];
        // prog = x0 + (x1 + 12) → 42. All indices/permissions erased.
        let src = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/scoped_eval.tal"),
        )
        .unwrap();
        assert_eq!(run(&src), 42);
    }

    #[test]
    fn nested_patterns_run_natively() {
        // the pattern-matrix desugar end-to-end: merged arms + inner matches
        // (`second`), nested Nat patterns (`pred2`), constructor patterns in the
        // first position (`swaps`); 9 + 3 + 2 = 14.
        let src = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
            boxed enum LList (a : Type) { LNil : LList a, LCons : a -> LList a -> LList a }\n\
            second : LList Nat -> Nat\n\
            fn second(xs) { match xs { LCons(h, LCons(h2, t)) => h2, LCons(h, LNil) => h, LNil => 0 } }\n\
            pred2 : Nat -> Nat\n\
            fn pred2(n) { match n { Succ(Succ(k)) => k, Succ(Zero) => 0, Zero => 0 } }\n\
            swaps : LList Nat -> Nat\n\
            fn swaps(xs) { match xs { LCons(Zero, r) => 100, LCons(Succ(k), r) => k, LNil => 7 } }\n\
            main : Nat\n\
            fn main() { let a = second(LCons(4, LCons(9, LNil))); let b = pred2(5); let c = swaps(LCons(3, LNil)); a + b + c }\n";
        assert_eq!(run(src), 14);
    }

    #[test]
    fn nested_patterns_over_linear_data_run_natively() {
        // a two-deep pattern MOVES both Owns out; each freed exactly once → 7.
        // (Also exercises the Case-not-Elim lowering for IH-free matches: an
        // eager eliminator IH would have re-traversed the linear tail.)
        let src = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
            boxed enum OL { ONil : OL, OCons : Own Nat -> OL -> OL }\n\
            freeRest : (1 r : OL) -> Nat\n\
            fn freeRest(r) { match r { ONil => Zero, OCons(a, t) => let u = free(a); freeRest(t) } }\n\
            sum2 : (1 xs : OL) -> Nat\n\
            fn sum2(xs) { match xs { OCons(a, OCons(b, ONil)) => let x = unbox(a); let y = unbox(b); x + y, OCons(a, r) => let u = free(a); freeRest(r), ONil => Zero } }\n\
            main : Nat\n\
            fn main() { sum2(OCons(alloc(3), OCons(alloc(4), ONil))) }\n";
        assert_eq!(run(src), 7);
    }

    #[test]
    fn convoy_vec_head_tail_run_natively() {
        // The index PROJECTIONS the convoy's Succ-inversion types: `vhead`/`vtail`
        // over `Vec Nat (Succ k)` with the impossible `Nil` arm OMITTED (refuted
        // by the index — real dependent coverage). vhead(vtail [1,2,3]) = 2.
        let src = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
            boxed enum Vec (a : Type) : Nat -> Type { Nil : Vec a Zero, Cons : {0 k : Nat} -> a -> Vec a k -> Vec a (Succ k) }\n\
            vhead : {0 k : Nat} -> Vec Nat (Succ k) -> Nat\n\
            fn vhead(v) { match v { Cons(h, t) => h } }\n\
            vtail : {0 k : Nat} -> Vec Nat (Succ k) -> Vec Nat k\n\
            fn vtail(v) { match v { Cons(h, t) => t } }\n\
            v3 : Vec Nat (Succ (Succ (Succ Zero)))\nfn v3() { Cons(1, Cons(2, Cons(3, Nil))) }\n\
            main : Nat\nfn main() { vhead(vtail(v3)) }\n";
        assert_eq!(run(src), 2);
    }

    #[test]
    fn transparent_newtype_zero_alloc() {
        // Phase B slice 1: a single-field wrapper (`struct Meters { v : Nat }`)
        // is TRANSPARENT — constructing it is the field, matching it is a bind.
        // Proven in the IR: the whole program compiles with ZERO `malloc`s
        // (previously: one cell + tag per construction).
        let src = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
            struct Meters { v : Nat }\n\
            mk : Nat -> Meters\nfn mk(n) { Meters(n + 1) }\n\
            get : Meters -> Nat\nfn get(m) { match m { Meters(v) => v } }\n\
            main : Nat\nfn main() { get(mk(41)) }\n";
        assert_eq!(run(src), 42);
        let prog = rust_surface::check_program(src).unwrap();
        let (_, _, body) = prog.defs.iter().find(|(n, _, _)| n == "main").unwrap();
        let ir = super::emit_ir(&prog.sig, body).unwrap();
        // (the module always DECLARES malloc; what must be absent is a CALL)
        assert!(!ir.contains("call ptr @malloc"), "a transparent newtype must not allocate:\n{ir}");
    }

    #[test]
    fn transparent_newtype_over_linear_payload() {
        // transparency is representation-only: a wrapper around an `Own` is
        // still linearity-checked (the wrapper IS the pointer at runtime).
        let src = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
            struct Handle { h : Own Nat }\n\
            open2 : Nat -> Handle\nfn open2(n) { Handle(alloc(n)) }\n\
            close2 : (1 x : Handle) -> Nat\nfn close2(x) { match x { Handle(o) => unbox(o) } }\n\
            main : Nat\nfn main() { close2(open2(9)) }\n";
        assert_eq!(run(src), 9);
        // dropping the wrapper leaks the Own inside — still rejected.
        let leak = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
            struct Handle { h : Own Nat }\n\
            bad : (1 x : Handle) -> Nat\nfn bad(x) { 0 }\n\
            main : Nat\nfn main() { 0 }\n";
        assert!(rust_surface::check_program(leak).is_err(), "dropping a linear wrapper must reject");
    }

    #[test]
    fn strings_print_and_flow_through_data() {
        // string literals are Str values: passed to fns, stored in datatypes,
        // printed via `prints`. main : Unit suppresses the result line, so the
        // program's stdout IS its effects, in order. `printall` is `%partial`
        // per the effect-order rule: a total fold is a CBV eliminator and would
        // emit bottom-up; `%partial` forces direct recursion (source order).
        let src = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
            boxed enum SList { SNil : SList, SCons : Str -> SList -> SList }\n\
            printall : SList -> Unit\n\
            %partial fn printall(xs) { match xs { SNil => U, SCons(h, t) => let u = prints(h); printall(t) } }\n\
            main : Unit\n\
            fn main() { printall(SCons(\"alpha\", SCons(\"beta\", SNil))) }\n";
        let prog = rust_surface::check_program(src).unwrap_or_else(|e| panic!("{e:?}"));
        let (_, _, body) = prog.defs.iter().find(|(n, _, _)| n == "main").unwrap();
        let dir = std::env::temp_dir();
        let obj = dir.join(format!("tally_str_{}.o", std::process::id()));
        let exe = dir.join(format!("tally_str_{}", std::process::id()));
        super::build_object_opts(&prog.sig, body, &obj, inkwell::OptimizationLevel::Default, false)
            .unwrap();
        assert!(std::process::Command::new("cc").arg(&obj).arg("-o").arg(&exe).status().unwrap().success());
        let out = std::process::Command::new(&exe).output().unwrap();
        let _ = std::fs::remove_file(&obj);
        let _ = std::fs::remove_file(exe.with_extension("o.ll"));
        let _ = std::fs::remove_file(&exe);
        assert_eq!(String::from_utf8_lossy(&out.stdout), "alpha\nbeta\n");
    }

    // ---- Phase C: zero-width views + &mut borrows (zero runtime trace) ----

    #[test]
    fn borrows_mutate_in_place_with_zero_trace() {
        // the §4.4 acceptance bar, verified in the IR: `borrow`/`restore` are
        // identity on the address, views/loans are ZERO-WIDTH, and a mutation
        // through a borrow is a bare load/store into the ORIGINAL cell. The
        // whole program — four in-place mutations over two owned cells — is
        // exactly two user mallocs and two user frees.
        let src = std::fs::read_to_string("examples/borrow.tal").unwrap();
        assert_eq!(run(&src), 51);
        let ir = ir(&src);
        assert_eq!(
            ir.matches("call ptr @malloc").count(),
            2,
            "exactly the two allocs\n{ir}"
        );
        assert_eq!(
            ir.matches("call void @free").count(),
            2,
            "exactly the two unboxes\n{ir}"
        );
    }

    // ---- Phase A: value enums, explicit indirection, zero implicit cells ----

    #[test]
    fn value_enums_are_flat_tagged_unions() {
        // a non-recursive multi-ctor enum is a VALUE: [tag, payload…, padding]
        // in registers — constructing, passing, and matching allocate NOTHING.
        let src = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
            enum Opt (a : Type) { None : Opt a, Some : a -> Opt a }\n\
            enum Shape { Circle : Nat -> Shape, Rect : Nat -> Nat -> Shape }\n\
            area : Shape -> Nat\n\
            fn area(s) { match s { Circle(r) => r + r, Rect(w, h) => w + h } }\n\
            grab : Opt Nat -> Nat\n\
            fn grab(o) { match o { None => 0, Some(x) => x } }\n\
            main : Nat\n\
            fn main() { area(Rect(3, 4)) + area(Circle(5)) + grab(Some(25)) }\n";
        assert_eq!(run(src), 42);
        let ir = ir(src);
        assert_eq!(
            ir.matches("call ptr @malloc").count(),
            0,
            "value enums must never allocate\n{ir}"
        );
    }

    #[test]
    fn recursive_enum_requires_explicit_boxing() {
        // a recursive enum without indirection would be an infinitely-sized
        // value — and its cells used to be allocated implicitly. Now: a guided
        // declaration error; `boxed enum` is the explicit opt-in.
        let bare = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
            enum L { LNil : L, LCons : Nat -> L -> L }\n\
            main : Nat\nfn main() { 0 }\n";
        let err = rust_surface::check_program(bare)
            .err()
            .expect("recursive enum without `boxed` must be rejected");
        assert!(
            format!("{err:?}").contains("RECURSIVE without indirection"),
            "expected the layout guidance, got {err:?}"
        );
        let boxed = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
            boxed enum L { LNil : L, LCons : Nat -> L -> L }\n\
            sum : L -> Nat\n\
            %partial fn sum(l) { match l { LNil => 0, LCons(x, t) => x + sum(t) } }\n\
            main : Nat\nfn main() { sum(LCons(40, LCons(2, LNil))) }\n";
        assert_eq!(run(boxed), 42);
    }

    #[test]
    fn own_linked_list_allocates_only_what_you_wrote() {
        // THE C linked structure, made safe: recursion through an explicit
        // `Own`, `Opt (Own Node)` stored via the NULL-POINTER NICHE — one slot,
        // None IS 0, Some IS the pointer, exactly C's nullable `struct node *`.
        // Every node is an `alloc` the program wrote: the IR contains exactly
        // the three user mallocs, 16 bytes each (v + the niched pointer), and
        // nothing else — byte-identical to the C struct.
        let src = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
            enum Opt (a : Type) { None : Opt a, Some : a -> Opt a }\n\
            struct Node { v : Nat, next : Opt (Own Node) }\n\
            sum : (1 o : Opt (Own Node)) -> Nat\n\
            %partial fn sum(o) { match o { None => 0, Some(p) => \
                let n = unbox(p); match n { Node(x, t) => x + sum(t) }, } }\n\
            main : Nat\n\
            fn main() { sum(Some(alloc(Node(1, Some(alloc(Node(2, Some(alloc(Node(39, None)))))))))) }\n";
        assert_eq!(run(src), 42);
        let ir = ir(&src);
        let sizes = malloc_sizes(&ir);
        assert_eq!(
            sizes,
            [(16u64, 3usize)].into_iter().collect(),
            "exactly the three user-written 16B node allocs (niched pointer); got {sizes:?}\n{ir}"
        );
    }

    // ---- Phase B2: flat multi-field structs BY VALUE ----

    const POINT: &str = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
        struct Point { x : Nat, y : Nat }\n";

    #[test]
    fn flat_record_by_value_runs() {
        // a two-field struct crossing a REAL function boundary (%partial ⇒ a
        // native Fix function) by value, both as parameter and as return.
        let src = format!(
            "{POINT}\
             step : Point -> Point\n\
             %partial fn step(p) {{ match p {{ Point(a, b) => Point(a + b, b + 1) }} }}\n\
             norm1 : Point -> Nat\n\
             fn norm1(p) {{ match p {{ Point(a, b) => a + b }} }}\n\
             main : Nat\n\
             fn main() {{ let p0 = Point(3, 4); let p1 = step(p0); norm1(p1) }}\n"
        );
        assert_eq!(run(&src), 12);
    }

    #[test]
    fn flat_record_ir_is_registers_no_malloc() {
        let src = format!(
            "{POINT}\
             step : Point -> Point\n\
             %partial fn step(p) {{ match p {{ Point(a, b) => Point(a + b, b + 1) }} }}\n\
             norm1 : Point -> Nat\n\
             fn norm1(p) {{ match p {{ Point(a, b) => a + b }} }}\n\
             main : Nat\n\
             fn main() {{ let p0 = Point(3, 4); let p1 = step(p0); norm1(p1) }}\n"
        );
        let ir = ir(&src);
        // (1) constructing, passing, returning, and matching the record touches
        // the heap NOWHERE — this is the whole point of Phase B2.
        assert_eq!(
            ir.matches("call ptr @malloc").count(),
            0,
            "a by-value record program must not allocate\n{ir}"
        );
        // (2) the function boundary is the two-register C convention: two i64
        // params in, a {{ i64, i64 }} aggregate out.
        assert!(
            ir.contains("define { i64, i64 }"),
            "the record-returning Fix must return an {{ i64, i64 }}\n{ir}"
        );
    }

    /// run a program EXPECTING codegen to reject it; returns the error text.
    fn run_err(src: &str) -> String {
        let prog = rust_surface::check_program(src).unwrap_or_else(|e| panic!("{e:?}"));
        let (_, _, body) = prog.defs.iter().find(|(n, _, _)| n == "main").expect("no main");
        super::run_main(&prog.sig, body).expect_err("program was expected to be rejected")
    }

    #[test]
    fn records_never_box_implicitly() {
        // ZERO NON-EXPLICIT BOXING. A record flowing into a GENERIC container
        // slot is a guided HARD ERROR — the compiler never inserts a hidden
        // malloc. The two explicit alternatives both work (below).
        let implicit = format!(
            "{POINT}\
             boxed enum PList (a : Type) {{ PNil : PList a, PCons : a -> PList a -> PList a }}\n\
             norm1 : Point -> Nat\n\
             fn norm1(p) {{ match p {{ Point(a, b) => a + b }} }}\n\
             sumlist : PList Point -> Nat\n\
             %partial fn sumlist(xs) {{ match xs {{ PNil => 0, PCons(p, t) => norm1(p) + sumlist(t), }} }}\n\
             main : Nat\n\
             fn main() {{ sumlist(PCons(Point(1, 2), PCons(Point(3, 4), PNil))) }}\n"
        );
        let err = run_err(&implicit);
        assert!(
            err.contains("never boxes implicitly"),
            "expected the guided zero-implicit-boxing error, got: {err}"
        );

        // explicit alternative 1: a CONCRETE container — the record is stored
        // FLAT INLINE in the cell ([tag, x, y, tail]), exactly the C struct-in-
        // node layout, still no per-element box.
        let concrete = format!(
            "{POINT}\
             boxed enum PointList {{ PNil : PointList, PCons : Point -> PointList -> PointList }}\n\
             norm1 : Point -> Nat\n\
             fn norm1(p) {{ match p {{ Point(a, b) => a + b }} }}\n\
             sumlist : PointList -> Nat\n\
             %partial fn sumlist(xs) {{ match xs {{ PNil => 0, PCons(p, t) => norm1(p) + sumlist(t), }} }}\n\
             main : Nat\n\
             fn main() {{ sumlist(PCons(Point(1, 2), PCons(Point(3, 4), PNil))) }}\n"
        );
        assert_eq!(run(&concrete), 10);

        // explicit alternative 2: the user writes the box — `Own Point`
        // elements, `alloc` at insertion, `unbox` at use. Every malloc in this
        // program is one the programmer wrote.
        let explicit = format!(
            "{POINT}\
             boxed enum PList (a : Type) {{ PNil : PList a, PCons : a -> PList a -> PList a }}\n\
             norm1 : Point -> Nat\n\
             fn norm1(p) {{ match p {{ Point(a, b) => a + b }} }}\n\
             sumlist : (1 xs : PList (Own Point)) -> Nat\n\
             %partial fn sumlist(xs) {{ match xs {{ PNil => 0, PCons(o, t) => let v = unbox(o); norm1(v) + sumlist(t), }} }}\n\
             main : Nat\n\
             fn main() {{ sumlist(PCons(alloc(Point(1, 2)), PCons(alloc(Point(3, 4)), PNil))) }}\n"
        );
        assert_eq!(run(&explicit), 10);
    }

    #[test]
    fn nested_records_and_transparent_wrappers_stay_flat() {
        // nesting still composes with zero allocation: a record field of record
        // type flattens inline (width 3), and a transparent single-field
        // wrapper of a record IS the record (identity).
        let src = format!(
            "{POINT}\
             enum PBox (a : Type) {{ MkBox : a -> PBox a }}\n\
             struct Seg {{ from : Point, len : Nat }}\n\
             norm1 : Point -> Nat\n\
             fn norm1(p) {{ match p {{ Point(a, b) => a + b }} }}\n\
             unbox1 : PBox Point -> Nat\n\
             fn unbox1(b) {{ match b {{ MkBox(p) => norm1(p) }} }}\n\
             seglen : Seg -> Nat\n\
             fn seglen(s) {{ match s {{ Seg(p, n) => norm1(p) + n }} }}\n\
             main : Nat\n\
             fn main() {{\n\
                 let v1 = unbox1(MkBox(Point(10, 20)));\n\
                 let v3 = seglen(Seg(Point(5, 6), 7));\n\
                 v1 + v3\n\
             }}\n"
        );
        assert_eq!(run(&src), 48);
        let ir = ir(&src);
        assert_eq!(
            ir.matches("call ptr @malloc").count(),
            0,
            "nested/wrapped records must not allocate\n{ir}"
        );
    }

    #[test]
    fn abstract_layout_ops_are_rejected_not_guessed() {
        // an `unbox` inside a GENERIC %partial function compiles once, but
        // `Own Point` and `Own Nat` cells have different layouts — reading one
        // slot of a two-slot payload would be silent corruption. tallyc
        // rejects it with guidance instead of guessing or boxing.
        let src = format!(
            "{POINT}\
             peel : {{0 a : Type}} -> (n : Nat) -> (1 o : Own a) -> Nat\n\
             %partial fn peel(n, o) {{ let v = unbox(o); n }}\n\
             main : Nat\n\
             fn main() {{ peel(1, alloc(Point(3, 4))) }}\n"
        );
        let err = run_err(&src);
        assert!(
            err.contains("ABSTRACT") && err.contains("never boxes implicitly"),
            "expected the abstract-layout guidance, got: {err}"
        );
    }

    #[test]
    fn aos_array_of_records_is_flat() {
        // Arr Point n is REAL AoS: one flat buffer at stride 2 (48 bytes for 3
        // Points), elements read/written by value; Own Point is ONE 16-byte
        // cell holding the record INLINE. Every allocation is user-written.
        let src = format!(
            "{POINT}\
             norm1 : Point -> Nat\n\
             fn norm1(p) {{ match p {{ Point(a, b) => a + b }} }}\n\
             main : Nat\n\
             fn main() {{\n\
                 let a0 = anew(3, Point(1, 2));\n\
                 let a1 = aset(1, lt(1, 3), Point(30, 40), a0);\n\
                 let (p, a2) = aget(1, lt(1, 3), a1);\n\
                 let u = afree(a2);\n\
                 let o = alloc(Point(5, 6));\n\
                 let q = unbox(o);\n\
                 norm1(p) + norm1(q)\n\
             }}\n"
        );
        assert_eq!(run(&src), 81); // (30+40) + (5+6)
        let ir = ir(&src);
        let sizes = malloc_sizes(&ir);
        assert_eq!(
            sizes,
            [(48u64, 1usize), (16u64, 1usize)].into_iter().collect(),
            "expected exactly the 3×16B AoS buffer and the inline Own cell; got {sizes:?}\n{ir}"
        );
    }

    #[test]
    fn record_by_value_crosses_c_ffi() {
        // THE C-interop payoff: a flat record flattens into consecutive i64
        // arguments, which on AArch64/SysV is exactly the by-value ABI for a
        // ≤ 2-register integer C struct — so a real C function receives a
        // `struct { long long x, y; }` BY VALUE from tally, no glue.
        let src = format!(
            "{POINT}\
             %foreign sumpt : Point -> Nat\n\
             main : Nat\n\
             fn main() {{ sumpt(Point(6, 7)) }}\n"
        );
        let prog = rust_surface::check_program(&src).unwrap_or_else(|e| panic!("{e:?}"));
        let (_, _, body) = prog.defs.iter().find(|(n, _, _)| n == "main").unwrap();
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let cfile = dir.join(format!("tally_sumpt_{pid}.c"));
        let cobj = dir.join(format!("tally_sumpt_{pid}_c.o"));
        let obj = dir.join(format!("tally_sumpt_{pid}.o"));
        let exe = dir.join(format!("tally_sumpt_{pid}"));
        std::fs::write(
            &cfile,
            "struct P { long long x, y; };\nlong long sumpt(struct P p) { return p.x * p.y + 1; }\n",
        )
        .unwrap();
        assert!(std::process::Command::new("cc")
            .args(["-O2", "-c"])
            .arg(&cfile)
            .arg("-o")
            .arg(&cobj)
            .status()
            .unwrap()
            .success());
        super::build_object_opts(&prog.sig, body, &obj, inkwell::OptimizationLevel::Default, true)
            .unwrap();
        assert!(std::process::Command::new("cc")
            .arg(&obj)
            .arg(&cobj)
            .arg("-o")
            .arg(&exe)
            .status()
            .unwrap()
            .success());
        let out = std::process::Command::new(&exe).output().unwrap();
        for fpath in [&cfile, &cobj, &obj, &exe] {
            let _ = std::fs::remove_file(fpath);
        }
        let _ = std::fs::remove_file(exe.with_extension("o.ll"));
        assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "43"); // 6·7 + 1
    }

    // ---- char I/O: getc/putc — a real program talks to the world ----

    #[test]
    fn char_io_cat_roundtrips_stdin() {
        // THE identity filter: read every byte of stdin, write it to stdout.
        // `getc` returns Succ(byte) / Zero-at-EOF, so one `match` is the whole
        // control flow. AOT-built and run with a piped stdin; the output must be
        // byte-exact.
        let src = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
            cat : Unit -> Unit\n\
            %partial fn cat(u) { match getc(u) { Zero => U, Succ(c) => let v = putc(c); cat(v), } }\n\
            main : Unit\nfn main() { cat(U) }\n";
        let prog = rust_surface::check_program(src).unwrap_or_else(|e| panic!("{e:?}"));
        let (_, _, body) = prog.defs.iter().find(|(n, _, _)| n == "main").unwrap();
        let dir = std::env::temp_dir();
        let obj = dir.join(format!("tally_cat_{}.o", std::process::id()));
        let exe = dir.join(format!("tally_cat_{}", std::process::id()));
        super::build_object_opts(&prog.sig, body, &obj, inkwell::OptimizationLevel::Default, false)
            .unwrap();
        assert!(std::process::Command::new("cc").arg(&obj).arg("-o").arg(&exe).status().unwrap().success());
        let input = "every byte, in order — π ≈ 3.14159\nsecond line\n";
        let mut child = std::process::Command::new(&exe)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        use std::io::Write as _;
        child.stdin.take().unwrap().write_all(input.as_bytes()).unwrap();
        let out = child.wait_with_output().unwrap();
        let _ = std::fs::remove_file(&obj);
        let _ = std::fs::remove_file(exe.with_extension("o.ll"));
        let _ = std::fs::remove_file(&exe);
        assert_eq!(String::from_utf8_lossy(&out.stdout), input);
    }

    // ---- %foreign: FFI to arbitrary C functions at the i64 ABI ----

    #[test]
    fn foreign_ffi_calls_libc() {
        // a real libc call, both the plain and the aliased ("C symbol") form. A
        // Str literal is a NUL-terminated pointer, so `strlen` sees exactly a
        // C string. 5 + 9 = 14.
        let src = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
            %foreign strlen : Str -> Nat\n\
            %foreign \"strlen\" clen : Str -> Nat\n\
            main : Nat\nfn main() { strlen(\"hello\") + clen(\"FFI works\") }\n";
        assert_eq!(run(src), 14);
    }

    // ---- CONTIGUOUS ARRAYS: the C `a[i]` on a flat buffer, bounds erased ----

    const BNAT_ARR: &str = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n";

    #[test]
    fn contiguous_array_runs() {
        // anew/aset/aget/afree on a flat 3-slot buffer; the `Lt` bounds are
        // solver-discharged, erased proofs.
        let src = format!(
            "{BNAT_ARR}\
             main : Nat\n\
             fn main() {{\n\
                 let a0 = anew(3, 0);\n\
                 let a1 = aset(1, lt(1, 3), 41, a0);\n\
                 let (x, a2) = aget(1, lt(1, 3), a1);\n\
                 let u = afree(a2);\n\
                 x + 1\n\
             }}\n"
        );
        assert_eq!(run(&src), 42);
    }

    #[test]
    fn contiguous_array_sum_loop_runs() {
        // THE C loop over an array, written safely: structural descent on `b`
        // with the array typed `Arr Nat (b + m)` — every index bound is a CLOSED
        // universal (`m < Succ k + m`) the stratum-(A) solver discharges, so no
        // proof is ever threaded. The `match` on `b` needs the NAT CONVOY: `arr`'s
        // type mentions the scrutinee (refined to `Succ k + m` in the arm) and
        // `arr` is linear (consumed exactly once per arm, joined — not ω-scaled).
        let src = format!(
            "{BNAT_ARR}\
             sum : (m : Nat) -> (b : Nat) -> (acc : Nat) -> (1 arr : Arr Nat (b + m)) -> Nat\n\
             %partial fn sum(m, b, acc, arr) {{\n\
                 match b {{\n\
                     Zero => let u = afree(arr); acc,\n\
                     Succ(k) =>\n\
                         let (x, arr2) = aget(m, lt(m, Succ(k) + m), arr);\n\
                         sum(Succ(m), k, acc + x, arr2),\n\
                 }}\n\
             }}\n\
             main : Nat\n\
             fn main() {{\n\
                 let a0 = anew(5, 7);\n\
                 let a1 = aset(2, lt(2, 5), 100, a0);\n\
                 sum(0, 5, 0, a1)\n\
             }}\n"
        );
        assert_eq!(run(&src), 128); // 4×7 + 100
    }

    #[test]
    fn soa_dot_product_threads_two_linear_arrays() {
        // the structure-of-arrays pattern (examples/arr_soa.tal): TWO linear
        // arrays abstracted by the same Nat convoy, each refined and consumed
        // exactly once per arm.
        let src = std::fs::read_to_string("examples/arr_soa.tal").unwrap();
        assert_eq!(run(&src), 75);
    }

    #[test]
    fn arr_ir_is_bare_indexed_load_no_bounds_check() {
        let src = format!(
            "{BNAT_ARR}\
             main : Nat\n\
             fn main() {{\n\
                 let a0 = anew(3, 0);\n\
                 let a1 = aset(1, lt(1, 3), 41, a0);\n\
                 let (x, a2) = aget(1, lt(1, 3), a1);\n\
                 let u = afree(a2);\n\
                 x + 1\n\
             }}\n"
        );
        let ir = ir(&src);

        // (1) The 3-element buffer is the ONLY malloc in the program — exactly
        // 3×8 = 24 bytes: no header, no stored length, no per-element cell (the
        // length lives only in the erased index), and the `ARead` pair that
        // threads the linear array back is a FLAT RECORD — registers, not a
        // cell, even at -O0 (Phase B2).
        let sizes = malloc_sizes(&ir);
        assert_eq!(
            sizes,
            [(24u64, 1usize)].into_iter().collect(),
            "array heap traffic must be exactly the flat buffer; got {sizes:?}\n{ir}"
        );

        // (2) The ONLY compare/branch in the whole program is `anew`'s fill
        // loop. `aget`/`aset` carry their bounds as ERASED `Lt` proofs, so the
        // emitted access is a bare gep+load / gep+store — no bounds check.
        assert_eq!(
            ir.matches("icmp").count(),
            1,
            "expected exactly one compare (the fill loop) — a second would be a \
             bounds check that should not exist\n{ir}"
        );
        assert_eq!(
            ir.matches("br i1").count(),
            1,
            "expected exactly one conditional branch (the fill loop)\n{ir}"
        );
    }
}
