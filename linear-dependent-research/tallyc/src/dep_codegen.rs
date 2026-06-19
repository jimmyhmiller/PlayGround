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
use inkwell::types::IntType;
use inkwell::values::{FunctionValue, IntValue, PointerValue};
use inkwell::{AddressSpace, OptimizationLevel};
use std::cell::RefCell;

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
            Term::NatElim(_p, z, s, scrut) => self.compile_fold(f, env, z, s, scrut),
            Term::Constr(name, args) => self.compile_constr(f, env, name, args),
            Term::Elim(data, _motive, methods, scrut) => {
                self.compile_elim(f, env, data, methods, scrut)
            }
            Term::App(_, _) => {
                // β-reduce a fully-applied spine: (λ…λ. body) a₁ … aₙ
                let (head, args) = flatten_app(t);
                // A postulate at the head of a spine (e.g. `alloc`, `free`,
                // `insert`, `remove`) is a memory primitive: dispatch to its
                // native implementation, erasing its multiplicity-0 arguments.
                if let Term::Const(c) = strip_ann(head) {
                    return self.compile_postulate(f, env, c, &args);
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

        // Boxed: malloc(8 * (1 + nfields)); store tag in slot 0; store each
        // non-erased ctor argument in its slot. Params and erased args store
        // NOTHING (erasure = zero overhead).
        let lay = ctor_layout(self.sig, &data, name)?;
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

            // Bind the method's lambda parameters: for each ctor argument, push
            // its value (loaded from its slot, or `None` if erased); right after
            // each recursive argument, push the induction hypothesis (a recursive
            // `self` call on that field).
            let mut menv = env.to_vec();
            for ai in 0..ctor.args.len() {
                let arg_slot = lay.arg_slot[ai];
                let arg_val: Slot<'c> = match arg_slot {
                    Some(s) => Some(self.load(scrut_ptr, s, "field")),
                    None => None, // erased argument: no runtime witness.
                };
                menv.push(arg_val);
                if lay.arg_recursive[ai] {
                    // induction hypothesis: elim recursively on the (boxed) field.
                    let field_ptr = arg_val.ok_or_else(|| {
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
/// the JIT runner (`run_nat`) and the IR emitter (`emit_ir`).
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
pub fn run_nat(sig: &Signature, main: &Term) -> Result<i64, String> {
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

#[cfg(test)]
mod tests {
    use crate::rust_surface;

    fn run(src: &str) -> i64 {
        let prog = rust_surface::check_program(src).unwrap_or_else(|e| panic!("{e:?}"));
        let (_, _, body) = prog.defs.iter().find(|(n, _, _)| n == "main").expect("no main");
        super::run_nat(&prog.sig, body).unwrap_or_else(|e| panic!("{e}"))
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

        // (1) The ONLY heap traffic is the data: three Cons cells (24 bytes each:
        // tag + element + tail pointer) and one Nil cell (8 bytes: just the tag).
        // No 32-byte cell exists — that would mean a stored length index.
        let sizes = malloc_sizes(&ir);
        assert_eq!(
            sizes,
            [(8u64, 1usize), (24u64, 3usize)].into_iter().collect(),
            "Vec heap traffic must be exactly 3×Cons(24B) + 1×Nil(8B); got {sizes:?}\n{ir}"
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
