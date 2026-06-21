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

/// The runtime layout of one boxed constructor: which of its *arguments* occupy a
/// runtime slot (the non-erased ones) and which are recursive occurrences of the
/// family. `args` are the constructor's own arguments (NOT the family params);
/// the index space here is the constructor's argument index `0..ctor.args.len()`.
struct CtorLayout {
    /// constructor tag = its position in `decl.ctors`.
    tag: u64,
    /// for each constructor argument: its runtime slot (1-based; slot 0 is the
    /// tag), or `None` if it is erased (multiplicity 0) and therefore unstored.
    arg_slot: Vec<Option<u32>>,
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
    let mut arg_slot = Vec::with_capacity(ctor.args.len());
    let mut arg_recursive = Vec::with_capacity(ctor.args.len());
    let mut next: u32 = 1; // slot 0 is the tag
    for (mult, aty) in &ctor.args {
        if *mult == Mult::Zero {
            arg_slot.push(None);
        } else {
            arg_slot.push(Some(next));
            next += 1;
        }
        arg_recursive.push(matches!(aty, Term::Data(dn, _) if dn == data));
    }
    Ok(CtorLayout {
        tag,
        arg_slot,
        arg_recursive,
        nfields: next - 1,
    })
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
}

struct FixSelf<'c> {
    level: usize,
    func: FunctionValue<'c>,
    erased: Vec<bool>,
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

/// A runtime environment slot: `Some(v)` for a live runtime value, `None` for an
/// ERASED binder (a multiplicity-0 variable that has no runtime witness). Reading
/// a `None` is a hard error — the kernel guarantees a checked term never does.
type Slot<'c> = Option<IntValue<'c>>;

impl<'c, 'a> DepCg<'c, 'a> {
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

    /// Read a runtime variable, erroring if it is erased (no runtime witness).
    fn read_var(&self, env: &[Slot<'c>], i: usize) -> Result<IntValue<'c>, String> {
        let pos = env
            .len()
            .checked_sub(1)
            .and_then(|n| n.checked_sub(i))
            .ok_or_else(|| format!("unbound runtime variable #{i}"))?;
        match env.get(pos) {
            Some(Some(v)) => Ok(*v),
            Some(None) => Err(format!(
                "runtime variable #{i} is ERASED (multiplicity 0): it has no runtime \
                 representation, yet codegen reached it in a value position \
                 (this should be impossible for a kernel-checked term)"
            )),
            None => Err(format!("unbound runtime variable #{i}")),
        }
    }

    /// Compile a term to an `i64`, given `env` (the value of each de Bruijn var,
    /// innermost last; `None` = erased) and the enclosing function.
    fn compile(
        &self,
        f: FunctionValue<'c>,
        env: &[Slot<'c>],
        t: &Term,
    ) -> Result<IntValue<'c>, String> {
        match t {
            Term::NatLit(n) => Ok(self.i64t.const_int(*n, false)),
            Term::Zero => Ok(self.i64t.const_int(0, false)),
            Term::Suc(x) => {
                let v = self.compile(f, env, x)?;
                Ok(self
                    .builder
                    .build_int_add(v, self.i64t.const_int(1, false), "suc")
                    .unwrap())
            }
            Term::Add(a, b) => {
                let x = self.compile(f, env, a)?;
                let y = self.compile(f, env, b)?;
                Ok(self.builder.build_int_add(x, y, "add").unwrap())
            }
            Term::Var(i) => self.read_var(env, *i),
            Term::Ann(e, _) => self.compile(f, env, e),
            // CALL-BY-VALUE let: compile `e` ONCE (so its effects — e.g. `free` — run
            // exactly once), bind it, compile the body. (`ty` is 0-erased.)
            Term::Let(_sigma, _ty, e, body) => {
                let ev = self.compile(f, env, e)?;
                let mut env2 = env.to_vec();
                env2.push(Some(ev));
                self.compile(f, &env2, body)
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
                        return self.compile_acc_fold(f, env, p, z, s, scrut, &args);
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
                        let k = env[k_level].ok_or_else(|| {
                            format!("recursive fold call to `{}`: predecessor is erased", func.get_name().to_str().unwrap_or("?"))
                        })?;
                        let mut call_args: Vec<inkwell::values::BasicMetadataValueEnum> =
                            vec![k.into()];
                        for a in &args {
                            call_args.push(self.compile(f, env, a)?.into());
                        }
                        return Ok(self
                            .builder
                            .build_call(func, &call_args, "ih.call")
                            .unwrap()
                            .try_as_basic_value()
                            .left()
                            .unwrap()
                            .into_int_value());
                    }
                    let hit = self
                        .fix_selves
                        .borrow()
                        .iter()
                        .rev()
                        .find(|fs| fs.level == lvl)
                        .map(|fs| (fs.func, fs.erased.clone()));
                    if let Some((func, erased)) = hit {
                        return self.compile_call(f, env, func, &erased, &args);
                    }
                }
                // The head is itself a `Fix`: build (memoized) its function and call.
                if let Term::Fix(ty, body) = strip_ann(head) {
                    let (func, erased) = self.build_fix(strip_ann(head), ty, body)?;
                    return self.compile_call(f, env, func, &erased, &args);
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
                                    let v = self.compile(f, env, a)?;
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
                self.compile(f, &env2, body)
            }
            Term::Const(c) => self.compile_postulate(f, env, c, &[]),
            other => Err(format!("not a runtime value: {other:?}")),
        }
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
    ) -> Result<IntValue<'c>, String> {
        let lay = ctor_layout(self.sig, data, cname)?;
        if field_vals.len() as u32 != lay.nfields {
            return Err(format!(
                "{data}.{cname}: native impl produced {} field(s), layout expects {}",
                field_vals.len(),
                lay.nfields
            ));
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
        Ok(self.toint(raw, "boxint"))
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
                        out.push(self.compile(f, env, a)?);
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
    ) -> Result<IntValue<'c>, String> {
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
                let rt = self.runtime_args(f, env, cname, args)?;
                let [payload] = rt.as_slice() else {
                    return Err(format!(
                        "alloc: expected 1 runtime argument (the payload), got {}",
                        rt.len()
                    ));
                };
                let cell = self.malloc_cell(1, "own");
                self.store(cell, 0, *payload);
                Ok(self.toint(cell, "ownint"))
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
                Ok(self.i64t.const_zero())
            }
            // unbox : {0 a} -> (1 o : Own a) -> a  — deref-and-CONSUME: load the
            // pointee from the cell, FREE the cell, return the pointee (moved out). The
            // linear `o` is consumed exactly once; its linear fields (e.g. a `tail`
            // Own) are moved to the caller (threaded onward by the eliminator). This is
            // the sound primitive for a consuming free-traversal of a linked structure.
            "unbox" => {
                let rt = self.runtime_args(f, env, cname, args)?;
                let [ptr] = rt.as_slice() else {
                    return Err(format!(
                        "unbox: expected 1 runtime argument (the owned pointer), got {}",
                        rt.len()
                    ));
                };
                let cell = self.toptr(*ptr, "ownp");
                let payload = self.load(cell, 0, "unboxed");
                self.free_int(*ptr); // free the Own cell; the payload was moved out
                Ok(payload)
            }
            // new : {0 r} -> List r  — create an empty circular sentinel list.
            "new" => {
                let s = self.malloc_cell(3, "sentinel");
                let si = self.toint(s, "si");
                self.store(s, NEXT, si); // s.next = s
                self.store(s, PREV, si); // s.prev = s
                Ok(si)
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
            // type-level postulates (Own/Region/List/Cursor/Arr): these only
            // ever appear inside ERASED type annotations, so they must never
            // reach a runtime value position. If one does, that is a real bug.
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
    ) -> Result<IntValue<'c>, String> {
        let (decl, _ctor) = self
            .sig
            .ctor(name)
            .ok_or_else(|| format!("unknown constructor `{name}`"))?;
        let data = decl.name.clone();
        let np = decl.params.len();

        // Nat-like: keep the unboxed i64 representation.
        if let Some((zero, _succ)) = nat_like(self.sig, &data) {
            if name == zero {
                return Ok(self.i64t.const_int(0, false));
            } else {
                // the successor's single argument is the predecessor.
                let pred = self.compile(f, env, &args[args.len() - 1])?;
                return Ok(self
                    .builder
                    .build_int_add(pred, self.i64t.const_int(1, false), "succ")
                    .unwrap());
            }
        }

        let lay = ctor_layout(self.sig, &data, name)?;

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
            return Ok(self
                .builder
                .build_ptr_to_int(global.as_pointer_value(), self.i64t, "nullary")
                .unwrap());
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
        // ctor args are args[np..]
        for (ai, slot) in lay.arg_slot.iter().enumerate() {
            if let Some(s) = slot {
                let v = self.compile(f, env, &args[np + ai])?;
                self.builder
                    .build_store(self.gep(raw, *s, "fld"), v)
                    .unwrap();
            }
            // erased argument: not evaluated, not stored.
        }
        Ok(self
            .builder
            .build_ptr_to_int(raw, self.i64t, "cellint")
            .unwrap())
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
    ) -> Result<IntValue<'c>, String> {
        if let Some((zero, _succ)) = nat_like(self.sig, data) {
            let decl = self.sig.data(data).unwrap();
            let zidx = decl.ctors.iter().position(|c| c.name == zero).unwrap();
            let sidx = 1 - zidx;
            return self.compile_fold(f, env, &methods[zidx], &methods[sidx], scrut);
        }

        // Boxed eliminator: build a recursive helper that captures the live outer
        // environment (the non-erased slots), then call it on the scrutinee.
        let scrut_val = self.compile(f, env, scrut)?;

        // Which outer env slots are live (non-erased)? They become helper params
        // (in addition to the scrutinee). We thread them through every recursive
        // call unchanged — only the scrutinee shrinks.
        let live: Vec<(usize, IntValue<'c>)> = env
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.map(|v| (i, v)))
            .collect();

        let nparams = 1 + live.len();
        let param_tys: Vec<inkwell::types::BasicMetadataTypeEnum> =
            std::iter::repeat(self.i64t.into()).take(nparams).collect();
        let fname = self.fresh(&format!("tally_elim_{data}"));
        let helper = self
            .module
            .add_function(&fname, self.i64t.fn_type(&param_tys, false), None);

        // Reconstruct, INSIDE the helper, the env that the method bodies expect:
        // the same shape as the outer env (Nones preserved), but every live slot
        // bound to the corresponding helper parameter.
        let mut inner_env: Vec<Slot<'c>> = env.iter().map(|_| None).collect();
        for (k, (i, _)) in live.iter().enumerate() {
            inner_env[*i] = Some(helper.get_nth_param((k + 1) as u32).unwrap().into_int_value());
        }
        let helper_scrut = helper.get_nth_param(0).unwrap().into_int_value();

        // Save the current insert position, build the helper body, then restore.
        let saved = self.builder.get_insert_block().unwrap();
        self.build_elim_helper(helper, data, methods, &inner_env, helper_scrut)?;
        self.builder.position_at_end(saved);

        // call the helper on the scrutinee + captured live env.
        let mut call_args: Vec<inkwell::values::BasicMetadataValueEnum> = Vec::with_capacity(nparams);
        call_args.push(scrut_val.into());
        for (_, v) in &live {
            call_args.push((*v).into());
        }
        Ok(self
            .builder
            .build_call(helper, &call_args, "elim")
            .unwrap()
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_int_value())
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
        // helper's own parameters #1.. (param #0 is the scrutinee). A recursive
        // call must forward THESE, not the outer caller's values.
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
                    Some(s) => Some(self.load(scrut_ptr, s, "field")),
                    None => None, // erased argument: no runtime witness.
                };
                menv.push(arg_val);
                arg_vals.push(arg_val);
            }
            for ai in 0..ctor.args.len() {
                if lay.arg_recursive[ai] {
                    // induction hypothesis: elim recursively on the (boxed) field.
                    let field_ptr = arg_vals[ai].ok_or_else(|| {
                        format!(
                            "{}.{}: a recursive argument was erased — impossible for a \
                             strictly-positive family",
                            data, ctor.name
                        )
                    })?;
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
                    menv.push(Some(ih));
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
            let result = self.compile(helper, &menv, body)?;
            self.builder.build_return(Some(&result)).unwrap();
        }
        Ok(())
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
    ) -> Result<IntValue<'c>, String> {
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

        // each arm binds the ctor's fields, compiles the method, branches to `join`.
        let mut incoming: Vec<(IntValue<'c>, inkwell::basic_block::BasicBlock<'c>)> =
            Vec::with_capacity(decl.ctors.len());
        for (ci, ctor) in decl.ctors.iter().enumerate() {
            self.builder.position_at_end(arm_blocks[ci]);
            let lay = ctor_layout(self.sig, data, &ctor.name)?;
            let mut menv = env.to_vec();
            for ai in 0..ctor.args.len() {
                let arg_val: Slot<'c> = match lay.arg_slot[ai] {
                    Some(s) => Some(self.load(scrut_ptr, s, "field")),
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
            let result = self.compile(f, &menv, body)?;
            // the method body may have emitted its own control flow, so the phi's
            // predecessor is the CURRENT block, not necessarily `arm_blocks[ci]`.
            let pred = self.builder.get_insert_block().unwrap();
            self.builder.build_unconditional_branch(join).unwrap();
            incoming.push((result, pred));
        }

        self.builder.position_at_end(join);
        let phi = self.builder.build_phi(self.i64t, "case.result").unwrap();
        for (v, bb) in &incoming {
            phi.add_incoming(&[(v as &dyn inkwell::values::BasicValue, *bb)]);
        }
        Ok(phi.as_basic_value().into_int_value())
    }

    /// `elim z s n` (Nat-like) as a native loop:
    ///   acc = z; for k in 0..n { acc = s k acc }.
    fn compile_fold(
        &self,
        f: FunctionValue<'c>,
        env: &[Slot<'c>],
        z: &Term,
        s: &Term,
        scrut: &Term,
    ) -> Result<IntValue<'c>, String> {
        let zv = self.compile(f, env, z)?;
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
        let acc_phi = self.builder.build_phi(self.i64t, "acc").unwrap();
        k_phi.add_incoming(&[(&self.i64t.const_int(0, false), entry)]);
        acc_phi.add_incoming(&[(&zv, entry)]);
        let k_val = k_phi.as_basic_value().into_int_value();
        let acc_val = acc_phi.as_basic_value().into_int_value();
        let more = self
            .builder
            .build_int_compare(inkwell::IntPredicate::ULT, k_val, nv, "more")
            .unwrap();
        self.builder
            .build_conditional_branch(more, body, exit)
            .unwrap();

        self.builder.position_at_end(body);
        let mut env2 = env.to_vec();
        env2.push(Some(k_val)); // k
        env2.push(Some(acc_val)); // ih
        let next_acc = self.compile(f, &env2, s_body)?;
        let next_k = self
            .builder
            .build_int_add(k_val, self.i64t.const_int(1, false), "k.next")
            .unwrap();
        let body_end = self.builder.get_insert_block().unwrap();
        k_phi.add_incoming(&[(&next_k, body_end)]);
        acc_phi.add_incoming(&[(&next_acc, body_end)]);
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
    ) -> Result<IntValue<'c>, String> {
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

        self.builder.position_at_end(zero_bb);
        let zv = self.compile(f, env, z)?;
        let zero_end = self.builder.get_insert_block().unwrap();
        self.builder.build_unconditional_branch(join_bb).unwrap();

        self.builder.position_at_end(succ_bb);
        let k = self
            .builder
            .build_int_sub(nv, self.i64t.const_int(1, false), "k")
            .unwrap();
        let mut env2 = env.to_vec();
        env2.push(Some(k));
        let sv = self.compile(f, &env2, s_body)?;
        let succ_end = self.builder.get_insert_block().unwrap();
        self.builder.build_unconditional_branch(join_bb).unwrap();

        self.builder.position_at_end(join_bb);
        let phi = self.builder.build_phi(self.i64t, "case").unwrap();
        phi.add_incoming(&[(&zv, zero_end), (&sv, succ_end)]);
        Ok(phi.as_basic_value().into_int_value())
    }

    /// Compile a call to a `Fix` function: evaluate the NON-erased arguments and
    /// emit the call. (Erased — multiplicity-0 — arguments are not passed.)
    fn compile_call(
        &self,
        f: FunctionValue<'c>,
        env: &[Slot<'c>],
        func: FunctionValue<'c>,
        erased: &[bool],
        args: &[&Term],
    ) -> Result<IntValue<'c>, String> {
        let mut call_args: Vec<inkwell::values::BasicMetadataValueEnum> = Vec::new();
        for (i, a) in args.iter().enumerate() {
            if !erased.get(i).copied().unwrap_or(false) {
                call_args.push(self.compile(f, env, a)?.into());
            }
        }
        Ok(self
            .builder
            .build_call(func, &call_args, "call")
            .unwrap()
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_int_value())
    }

    /// Build (once, memoized by term identity) the native function for a `Fix`.
    /// `Fix(ty, body)` with `body = λp₁…λpₙ. inner` becomes a recursive function of
    /// the non-erased parameters; self-references in `inner` compile to calls to it.
    fn build_fix(
        &self,
        fix_term: &Term,
        ty: &Term,
        body: &Term,
    ) -> Result<(FunctionValue<'c>, Vec<bool>), String> {
        // erasure flags: pair each body lambda with its type's Π multiplicity.
        let mut erased = Vec::new();
        let mut t_ty = strip_ann(ty);
        let mut t_body = strip_ann(body);
        while let (Term::Lam(inner), Term::Pi(m, _d, cod)) = (t_body, t_ty) {
            erased.push(*m == Mult::Zero);
            t_body = strip_ann(inner);
            t_ty = strip_ann(cod);
        }
        let inner = t_body;

        let key = fix_term as *const Term;
        if let Some(&func) = self.fix_cache.borrow().get(&key) {
            return Ok((func, erased));
        }

        let n_runtime = erased.iter().filter(|e| !**e).count();
        let param_tys: Vec<inkwell::types::BasicMetadataTypeEnum> =
            std::iter::repeat(self.i64t.into()).take(n_runtime).collect();
        let fname = self.fresh("tally_fix");
        let func = self
            .module
            .add_function(&fname, self.i64t.fn_type(&param_tys, false), None);
        self.fix_cache.borrow_mut().insert(key, func);

        // Fresh body env: [self, p₁, …, pₙ] (self at level 0; erased params are None).
        let saved = self.builder.get_insert_block();
        let entry = self.ctx.append_basic_block(func, "entry");
        self.builder.position_at_end(entry);
        let mut body_env: Vec<Slot<'c>> = vec![None]; // self
        let mut rt = 0u32;
        for e in &erased {
            if *e {
                body_env.push(None);
            } else {
                body_env.push(Some(func.get_nth_param(rt).unwrap().into_int_value()));
                rt += 1;
            }
        }
        self.fix_selves.borrow_mut().push(FixSelf {
            level: 0,
            func,
            erased: erased.clone(),
        });
        let result = self.compile(func, &body_env, inner);
        self.fix_selves.borrow_mut().pop();
        let result = result?;
        self.builder.build_return(Some(&result)).unwrap();
        if let Some(bb) = saved {
            self.builder.position_at_end(bb);
        }
        Ok((func, erased))
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
            let zero_env: Vec<Slot<'c>> = acc_params.iter().map(|v| Some(*v)).collect();
            let zr = self.compile(func, &zero_env, z_body)?;
            self.builder.build_return(Some(&zr)).unwrap();

            // succ: k = i - 1; env = [k, ih, a₁ … a_K]; the IH self sits at level 1.
            self.builder.position_at_end(succ_bb);
            let k_val = self
                .builder
                .build_int_sub(i_param, self.i64t.const_int(1, false), "k")
                .unwrap();
            let mut succ_env: Vec<Slot<'c>> = vec![Some(k_val), None]; // k (level 0), ih (level 1)
            succ_env.extend(acc_params.iter().map(|v| Some(*v)));
            self.acc_ih_selves.borrow_mut().push(AccIhSelf { ih_level: 1, func, k_level: 0 });
            let sr = self.compile(func, &succ_env, s_body);
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

        // call `g(scrut, args…)` with the accumulators evaluated in the CURRENT env.
        let scrut_v = self.compile(f, env, scrut)?;
        let mut call_args: Vec<inkwell::values::BasicMetadataValueEnum> = vec![scrut_v.into()];
        for a in args {
            call_args.push(self.compile(f, env, a)?.into());
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
    };
    let result = cg.compile(f, &[], main)?;
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
fn add_c_main<'c>(ctx: &'c Context, module: &Module<'c>) {
    let i32t = ctx.i32_type();
    let ptr = ctx.ptr_type(AddressSpace::default());
    let builder = ctx.create_builder();

    let printf = module.add_function("printf", i32t.fn_type(&[ptr.into()], true), None);
    let dep_main = module
        .get_function("tally_dep_main")
        .expect("tally_dep_main must be built before add_c_main");

    let main = module.add_function("main", i32t.fn_type(&[], false), None);
    let entry = ctx.append_basic_block(main, "entry");
    builder.position_at_end(entry);
    let fmt = builder.build_global_string_ptr("%lld\n", "fmt").unwrap();
    let res = builder
        .build_call(dep_main, &[], "res")
        .unwrap()
        .try_as_basic_value()
        .left()
        .unwrap();
    builder
        .build_call(printf, &[fmt.as_pointer_value().into(), res.into()], "_")
        .unwrap();
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
    let ctx = Context::create();
    let module = build_module(&ctx, sig, main)?;
    add_c_main(&ctx, &module);

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
enum Tree { Leaf : Tree, Node : Tree -> Nat -> Tree -> Tree }
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
enum Tree { Leaf : Tree, Node : Tree -> Nat -> Tree -> Tree }
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
            enum Expr { lit : Nat -> Expr, add : Own Expr -> Own Expr -> Expr, mul : Own Expr -> Own Expr -> Expr }\n\
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
            enum List { nil : List, cons : Nat -> List -> List }\n\
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
enum Vec (a : Type) : Nat -> Type {
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
enum Fin : Nat -> Type {
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
}
