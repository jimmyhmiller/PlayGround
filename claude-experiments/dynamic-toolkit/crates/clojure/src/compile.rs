//! Compiler: read values → dynir IR.
//!
//! Checkpoint 3 supports:
//!   - Self-evaluating literals
//!   - Local symbol lookup (params, `let`-bound vars)
//!   - Top-level fn lookup (everything in `func_refs`)
//!   - Special forms: `def`, `fn`, `if`, `let`, `do`, `quote`
//!   - Function calls (extern primitives + user-defined fns)
//!
//! `def` is recognized at top level and shaped specifically for
//! `(def NAME (fn [args...] body...))`. First-class fn values arrive
//! in a later checkpoint along with closures.

use std::collections::HashMap;

use dynir::builder::{FunctionBuilder, ModuleBuilder};
use dynir::ir::{BlockId, CmpOp, FuncRef, LiteralRef, Value};
use dynir::types::Type;
use dynlower::LiteralPool;

use crate::symbols::SymbolTable;
use crate::value as v;

/// One frame of the loop/recur target stack maintained while
/// lowering a function body. `recur` lowers to a jump to the most
/// recent target (a `loop` header, or — when no `loop` is open —
/// the function's entry block, which is what gives us tail-call
/// semantics for plain self-recursion).
///
/// `prepend` is non-empty for closure entry blocks: a closure's
/// entry block takes `(self_fn, p0, p1, …)`, and `recur` from inside
/// a closure body must pass the same self_fn through unchanged.
pub struct LoopTarget {
    pub block: BlockId,
    /// Number of user-visible recur arguments (matches the loop's
    /// binding count, or the function's user arity).
    pub arity: usize,
    /// Values to prepend to the recur args when emitting the jump.
    pub prepend: Vec<Value>,
}

/// Metadata for everything looked up by name in `func_refs`.
///
/// Two ABIs coexist:
///   - `Extern` — a C extern called with its natural N-argument
///     signature. Numeric primitives, the reader-collection
///     accessor bridges, and other host-supplied helpers all live
///     in this category.
///   - `User` — a user-defined fn (`def` or `fn`-expression). All
///     user-defined fns share a uniform single-list ABI:
///     `(self_fn, args_list)` where `self_fn` is the callee's own
///     heap pointer (used to read closure captures and as the
///     receiver for self-recursion via `recur`); `args_list` is a
///     Clojure list of every supplied argument. The body's
///     prologue walks `args_list` to bind individual params and (if
///     variadic) collect the rest. This makes both static and
///     indirect dispatch a single call shape, and gives us
///     multi-arity / variadic for free at the cost of one list
///     allocation per call.
#[derive(Clone, Copy)]
pub enum FnEntry {
    Extern { fref: FuncRef, arity: usize },
    User { fref: FuncRef },
}

impl FnEntry {
    pub fn fref(self) -> FuncRef {
        match self {
            FnEntry::Extern { fref, .. } => fref,
            FnEntry::User { fref } => fref,
        }
    }
    pub fn is_extern(self) -> bool {
        matches!(self, FnEntry::Extern { .. })
    }
}

/// Parsed parameter list. `(fn [a b & rs] …)` produces
/// `Params { fixed: [a, b], rest: Some(rs) }`. `min_arity` is the
/// number of fixed params; the user must supply at least that many
/// at the call site, with any extras packed into the rest list.
#[derive(Default)]
pub struct Params {
    pub fixed: Vec<u32>,
    pub rest: Option<u32>,
}

impl Params {
    pub fn min_arity(&self) -> usize {
        self.fixed.len()
    }
    pub fn is_variadic(&self) -> bool {
        self.rest.is_some()
    }
}

/// Parse a `[a b & rs]`-style parameter vector (or list) into a
/// `Params` struct. Panics on malformed input — every element must
/// be a symbol; `&` may appear at most once and must be followed by
/// exactly one symbol (the rest binding).
fn parse_params(arg_vec: u64, sym: &crate::symbols::SymbolTable) -> Params {
    let mut p = Params::default();
    let mut seen_amp = false;
    for elem in crate::collections::seq_iter(arg_vec) {
        if !v::is_sym_id(elem) {
            panic!("fn: parameter must be a symbol");
        }
        let id = v::as_sym_id(elem);
        let name = sym.name(id);
        if name == "&" {
            if seen_amp {
                panic!("fn: only one `&` allowed in parameter list");
            }
            seen_amp = true;
            continue;
        }
        if seen_amp {
            if p.rest.is_some() {
                panic!("fn: only one binding allowed after `&`");
            }
            p.rest = Some(id);
        } else {
            p.fixed.push(id);
        }
    }
    if seen_amp && p.rest.is_none() {
        panic!("fn: `&` must be followed by a rest-binding symbol");
    }
    p
}

/// Encode (min_arity, is_variadic) into the single Raw64 word stored
/// in `Fn.arity`. Bit 63 is the variadic flag; the low 63 bits are
/// `min_arity`. Decoded by the runtime arity-check / dispatch path.
const ARITY_VARIADIC_BIT: u64 = 1 << 63;
pub fn encode_arity(min_arity: usize, is_variadic: bool) -> u64 {
    let n = min_arity as u64;
    if is_variadic {
        n | ARITY_VARIADIC_BIT
    } else {
        n
    }
}

#[derive(Debug)]
pub enum TopResult {
    /// Anonymous top-level expression compiled to a 0-arg function
    /// returning the result bits.
    Expr(FuncRef),
    /// `(def name ...)` (or `defmacro`) declared a new top-level
    /// function. There's no expression to run; the driver interns a
    /// Var in `clojure.core` whose root holds an `Fn` heap object
    /// pointing at this `FuncRef`. When `is_macro` is true, the
    /// driver also sets the `:macro` flag bit on the Var.
    Define {
        name: String,
        fref: FuncRef,
        arity: usize,
        is_macro: bool,
    },
    /// Pure no-op (e.g. comment-only input).
    None,
}

pub struct Compiler<'a> {
    pub mb: &'a mut ModuleBuilder,
    pub func_refs: &'a mut HashMap<String, FnEntry>,
    pub sym: &'a mut SymbolTable,
    pub anon_counter: &'a mut u32,
    /// Literal pool we push GC-managed compile-time constants into.
    /// Each push returns a slot index that becomes a `LiteralRef` in
    /// emitted IR; the JIT lowers `Inst::GcLiteral(idx)` to a load
    /// from `pool.base() + idx*8`. The pool is registered with the
    /// GC as a root source so moving collections rewrite the slots in
    /// place.
    pub literal_pool: &'a LiteralPool,
    /// Stable base address of the JitModule's `call_table`. Used by
    /// the indirect-call path to translate a runtime `FuncRef` index
    /// (read from a heap `Fn`) into the code pointer to call:
    /// `code_ptr = *(call_table_base + fr * 8)`.
    pub call_table_base: u64,
    /// Stack of recur targets. Pushed on entry to a `loop` form or a
    /// function body; popped on exit. The compiler is reset between
    /// top-level forms so this only ever grows during a single
    /// function compilation.
    pub loop_targets: Vec<LoopTarget>,
}

impl<'a> Compiler<'a> {
    pub fn compile_top(&mut self, form: u64) -> TopResult {
        // Recognize `(def NAME EXPR)` and `(defmacro NAME [args] body)`
        // at top level. Only lists carry a head — vectors/maps/strings
        // would read garbage through `v::first` since they share the
        // ptr tag but not the list layout.
        if v::is_ptr(form) && crate::collections::is_list(form) {
            let head = v::first(form);
            if v::is_sym_id(head) {
                let head_name = self.sym.name(v::as_sym_id(head)).to_string();
                match head_name.as_str() {
                    "def" => return self.compile_def(form),
                    "defmacro" => return self.compile_defmacro(form),
                    _ => {}
                }
            }
        }
        // Otherwise: anonymous top-level expression.
        *self.anon_counter += 1;
        let name = format!("__top_{}", *self.anon_counter);
        let fref = self.mb.declare_func(&name, &[], Some(Type::I64));
        self.func_refs.insert(name.clone(), fref);

        let mut fb = self.mb.define_func(fref);
        let mut env = Env::new();
        let result = self.lower_expr(&mut fb, &mut env, form);
        fb.ret(result);
        self.mb.finish_func(fref, fb);

        TopResult::Expr(fref)
    }

    fn compile_def(&mut self, form: u64) -> TopResult {
        let (name, fref, arity) = self.compile_def_like(form, "def", /*expect_fn=*/ true);
        TopResult::Define {
            name,
            fref,
            arity,
            is_macro: false,
        }
    }

    fn compile_defmacro(&mut self, form: u64) -> TopResult {
        // `(defmacro NAME [args...] body...)` — same shape as
        // `(def NAME (fn [args] body))` but inline (without the
        // explicit `fn`). The result Var gets the `:macro` flag.
        let rest = v::rest(form);
        let name_v = v::first(rest);
        if !v::is_sym_id(name_v) {
            panic!("defmacro: name must be a symbol");
        }
        let name = self.sym.name(v::as_sym_id(name_v)).to_string();
        let after_name = v::rest(rest);
        let arg_vec = v::first(after_name);
        let body_forms = v::rest(after_name);

        let mut param_ids: Vec<u32> = Vec::new();
        for p in crate::collections::seq_iter(arg_vec) {
            if !v::is_sym_id(p) {
                panic!("defmacro: parameter must be a symbol");
            }
            param_ids.push(v::as_sym_id(p));
        }
        let param_tys = vec![Type::I64; param_ids.len()];
        let fref = self.mb.declare_func(&name, &param_tys, Some(Type::I64));
        self.func_refs.insert(name.clone(), fref);

        let arity = param_ids.len();
        self.compile_function_body(fref, &param_ids, body_forms);
        TopResult::Define {
            name,
            fref,
            arity,
            is_macro: true,
        }
    }

    /// Shared backbone for `def` and (eventually) other defining
    /// forms. Returns (name, FuncRef, arity).
    fn compile_def_like(&mut self, form: u64, kw: &str, expect_fn: bool) -> (String, FuncRef, usize) {
        let rest = v::rest(form);
        let name_v = v::first(rest);
        if !v::is_sym_id(name_v) {
            panic!("{}: name must be a symbol", kw);
        }
        let name = self.sym.name(v::as_sym_id(name_v)).to_string();
        let expr = v::first(v::rest(rest));

        if expect_fn {
            if !v::is_ptr(expr) {
                panic!(
                    "{}: only `({} NAME (fn [args] body))` is supported in this checkpoint; \
                     got {} of {} bound to non-fn",
                    kw, kw, kw, name
                );
            }
            let fn_head = v::first(expr);
            if !v::is_sym_id(fn_head)
                || self.sym.name(v::as_sym_id(fn_head)) != "fn"
            {
                panic!(
                    "{}: only `({} NAME (fn [args] body))` is supported in this checkpoint; \
                     got {} of {} bound to non-fn",
                    kw, kw, kw, name
                );
            }
            let fn_rest = v::rest(expr);
            let arg_vec = v::first(fn_rest);
            let body_forms = v::rest(fn_rest);

            let mut param_ids: Vec<u32> = Vec::new();
            for p in crate::collections::seq_iter(arg_vec) {
                if !v::is_sym_id(p) {
                    panic!("{}: parameter must be a symbol", kw);
                }
                param_ids.push(v::as_sym_id(p));
            }
            let param_tys = vec![Type::I64; param_ids.len()];
            let fref = self.mb.declare_func(&name, &param_tys, Some(Type::I64));
            self.func_refs.insert(name.clone(), fref);

            let arity = param_ids.len();
            self.compile_function_body(fref, &param_ids, body_forms);
            (name, fref, arity)
        } else {
            unreachable!("non-fn def shape not yet supported")
        }
    }

    /// Lower a function whose body is a list of forms `(e1 e2 ... eN)`.
    /// Forms run in sequence; the last produces the return value. An
    /// empty body returns nil.
    ///
    /// The function's entry block immediately jumps to a fresh
    /// `loop_header` block whose params mirror the user's params.
    /// `recur` targets `loop_header`, not `entry`. This indirection
    /// keeps `entry` predecessor-free (the regalloc treats entry
    /// specially because the calling convention pre-loads its params
    /// into specific registers; back-edges into entry would require
    /// re-establishing that ABI on every iteration).
    fn compile_function_body(&mut self, fref: FuncRef, params: &[u32], body_forms: u64) {
        let mut fb = self.mb.define_func(fref);
        let mut env = Env::new();
        let entry = fb.entry_block();
        let entry_param_vals: Vec<Value> = (0..params.len())
            .map(|i| fb.block_param(entry, i))
            .collect();

        let loop_header = fb.create_block(&vec![Type::I64; params.len()]);
        fb.jump(loop_header, &entry_param_vals);
        fb.switch_to_block(loop_header);

        for (i, &pid) in params.iter().enumerate() {
            env.bind(pid, fb.block_param(loop_header, i));
        }
        self.loop_targets.push(LoopTarget {
            block: loop_header,
            arity: params.len(),
            prepend: Vec::new(),
        });
        let result = self.lower_do(&mut fb, &mut env, body_forms);
        fb.ret(result);
        self.mb.finish_func(fref, fb);
        self.loop_targets.pop();
    }

    fn lower_expr(&mut self, fb: &mut FunctionBuilder, env: &mut Env, form: u64) -> Value {
        // Self-evaluating immediates.
        if v::is_number(form) || v::is_nil(form) || form == v::TRUE || form == v::FALSE {
            return fb.iconst(Type::I64, form as i64);
        }
        // Bare symbol: try local env first, then top-level func_refs
        // for 0-arg "vars".
        if v::is_sym_id(form) {
            let id = v::as_sym_id(form);
            if let Some(val) = env.lookup(id) {
                return val;
            }
            let name = self.sym.name(id);
            panic!("undefined variable: {}", name);
        }
        if !v::is_ptr(form) {
            panic!("unrecognized form: 0x{:016x}", form);
        }
        // Self-evaluating compound literals (string / keyword / vector
        // / map / set). Pin into the literal pool; emit a load.
        if !crate::collections::is_list(form) {
            let idx = self.literal_pool.push(form) as u32;
            return fb.gc_literal(LiteralRef::from_u32(idx));
        }
        // List form: head dispatches on special form vs. call.
        let head = v::first(form);
        if v::is_sym_id(head) {
            let head_name = self.sym.name(v::as_sym_id(head)).to_string();
            match head_name.as_str() {
                "if" => return self.lower_if(fb, env, v::rest(form)),
                "do" => return self.lower_do(fb, env, v::rest(form)),
                "let" => return self.lower_let(fb, env, v::rest(form)),
                "loop" => return self.lower_loop(fb, env, v::rest(form)),
                "recur" => return self.lower_recur(fb, env, v::rest(form)),
                "quote" => return self.lower_quote(fb, v::rest(form)),
                "def" => panic!("def is only valid at top level"),
                "fn" => return self.lower_fn_expr(fb, env, v::rest(form)),
                _ => {}
            }
        }
        self.lower_call(fb, env, form)
    }

    fn lower_if(&mut self, fb: &mut FunctionBuilder, env: &mut Env, args: u64) -> Value {
        let cond_form = v::first(args);
        let after_cond = v::rest(args);
        let then_form = v::first(after_cond);
        let after_then = v::rest(after_cond);
        let else_form = if v::is_ptr(after_then) {
            v::first(after_then)
        } else {
            v::NIL
        };

        let cond = self.lower_expr(fb, env, cond_form);
        let cond_bool = compute_truthy(fb, cond);

        let then_bb = fb.create_block(&[]);
        let else_bb = fb.create_block(&[]);
        let merge_bb = fb.create_block(&[Type::I64]);
        fb.br_if(cond_bool, then_bb, &[], else_bb, &[]);

        fb.switch_to_block(then_bb);
        let tv = self.lower_expr(fb, env, then_form);
        fb.jump(merge_bb, &[tv]);

        fb.switch_to_block(else_bb);
        let ev = self.lower_expr(fb, env, else_form);
        fb.jump(merge_bb, &[ev]);

        fb.switch_to_block(merge_bb);
        fb.block_param(merge_bb, 0)
    }

    fn lower_do(&mut self, fb: &mut FunctionBuilder, env: &mut Env, args: u64) -> Value {
        if v::is_nil(args) {
            return fb.iconst(Type::I64, v::NIL as i64);
        }
        let mut last = None;
        for x in v::list_iter(args) {
            last = Some(self.lower_expr(fb, env, x));
        }
        last.unwrap()
    }

    fn lower_let(&mut self, fb: &mut FunctionBuilder, env: &mut Env, args: u64) -> Value {
        // Clojure syntax: (let [name val name val ...] body...)
        // The bindings come in pairs flat in a vector.
        let bindings = v::first(args);
        let body_forms = v::rest(args);
        let pairs: Vec<u64> = crate::collections::seq_iter(bindings).collect();
        if pairs.len() % 2 != 0 {
            panic!("let: bindings must come in name/value pairs");
        }
        env.push();
        for chunk in pairs.chunks(2) {
            let name_v = chunk[0];
            let val_form = chunk[1];
            if !v::is_sym_id(name_v) {
                panic!("let: binding name must be a symbol");
            }
            let val = self.lower_expr(fb, env, val_form);
            env.bind(v::as_sym_id(name_v), val);
        }
        let result = self.lower_do(fb, env, body_forms);
        env.pop();
        result
    }

    fn lower_quote(&mut self, fb: &mut FunctionBuilder, args: u64) -> Value {
        let datum = v::first(args);
        if v::is_ptr(datum) {
            // Heap datum: pin into the literal pool. The pool is a GC
            // root source, so a moving collection updates the slot in
            // place; emitted code re-reads the (possibly relocated)
            // pointer on every load.
            let idx = self.literal_pool.push(datum) as u32;
            return fb.gc_literal(LiteralRef::from_u32(idx));
        }
        fb.iconst(Type::I64, datum as i64)
    }

    fn lower_call(&mut self, fb: &mut FunctionBuilder, env: &mut Env, form: u64) -> Value {
        let head = v::first(form);
        let arg_forms: Vec<u64> = v::list_iter(v::rest(form)).collect();
        let n = arg_forms.len();

        // Resolve head: prefer local binding (which is necessarily an
        // Fn value, so dispatch indirectly) over the static func_refs
        // table. Non-symbol heads always go indirect: we evaluate the
        // head expression to a value and dispatch through it.
        if v::is_sym_id(head) {
            let id = v::as_sym_id(head);
            if env.lookup(id).is_none() {
                if let Some(&fref) = self.func_refs.get(self.sym.name(id)) {
                    let arg_vals: Vec<Value> = arg_forms
                        .iter()
                        .map(|a| self.lower_expr(fb, env, *a))
                        .collect();
                    let mut live = env.live_values();
                    live.extend_from_slice(&arg_vals);
                    fb.safepoint(&live);
                    return fb
                        .call(fref, &arg_vals)
                        .expect("call returns a value (Type::I64)");
                }
            }
        }

        // Indirect dispatch through the callee's `Fn.func_ref` field.
        //
        //   callee_tagged := <head expr>           ; NanBox-tagged Fn ptr
        //   arity_check_extern(callee_tagged, n)   ; runtime arity check
        //   fn_obj  := payload(callee_tagged)      ; raw heap pointer
        //   fr      := load (fn_obj + FN_FUNCREF) ; runtime u32 FuncRef
        //   addr    := call_table_base + fr * 8
        //   code_ptr:= load addr
        //   result  := call_indirect code_ptr (callee_tagged, args...)
        //
        // No arity cap. Whatever arity the runtime fn was compiled
        // for, the indirect call passes through unchanged.
        let callee = self.lower_expr(fb, env, head);
        let mut arg_vals: Vec<Value> = Vec::with_capacity(n + 1);
        arg_vals.push(callee);
        for a in &arg_forms {
            arg_vals.push(self.lower_expr(fb, env, *a));
        }

        // Runtime arity check via extern. Static-dispatch sites get
        // their arity check at IR build time (call signature mismatch
        // panics there); the indirect path needs a runtime version.
        let arity_check = *self
            .func_refs
            .get("__arity_check")
            .expect("__arity_check extern not registered");
        let n_const = fb.iconst(Type::I64, n as i64);
        let mut live_for_check = env.live_values();
        live_for_check.extend_from_slice(&arg_vals);
        fb.safepoint(&live_for_check);
        fb.call(arity_check, &[callee, n_const]);

        // Untag the Fn ptr (clears the NanBox high bits) so it can be
        // used as a base for raw loads.
        let fn_obj = fb.payload(callee);
        let fn_obj_i64 = fb.bitcast(fn_obj, Type::I64);

        // Load the FuncRef index. It's stored as a Raw64 word
        // (zero-extended from u32) at FN_FUNCREF_OFFSET.
        let fr_i64 = fb.load(
            Type::I64,
            fn_obj_i64,
            crate::namespace::fn_func_ref_offset(),
        );

        // Compute call_table_base + fr * 8.
        let table_base = fb.iconst(Type::I64, self.call_table_base as i64);
        let three = fb.iconst(Type::I64, 3);
        let fr_x8 = fb.shl(fr_i64, three);
        let addr = fb.add(table_base, fr_x8);

        // Load the code pointer.
        let code_ptr = fb.load(Type::I64, addr, 0);

        // Indirect call. Safepoint covers the live captures + args.
        let mut live = env.live_values();
        live.extend_from_slice(&arg_vals);
        fb.safepoint(&live);
        fb.call_indirect(code_ptr, &arg_vals, Some(Type::I64))
            .expect("call_indirect returns a value")
    }

    /// Compile `(fn [args...] body...)` as a value.
    ///
    /// Steps:
    ///   1. Free-var analyze the body to find captures (symbols that
    ///      are bound in the enclosing env but not by the fn's own
    ///      params).
    ///   2. Declare an anonymous inner function whose first parameter
    ///      is the receiver `self_fn` (the Fn obj being invoked, used
    ///      to load captures); subsequent params are the user's args.
    ///   3. Compile the inner body with captures pre-bound to loads
    ///      from `self_fn`'s captures area.
    ///   4. Emit IR in the OUTER function that allocates an `Fn` obj
    ///      via either `__alloc_fn` (no captures) or `__alloc_closure`
    ///      (captures spilled to a stack slot, then copied in).
    fn lower_fn_expr(
        &mut self,
        fb: &mut FunctionBuilder,
        env: &mut Env,
        rest: u64,
    ) -> Value {
        let arg_vec = v::first(rest);
        let body_forms = v::rest(rest);

        let mut param_ids: Vec<u32> = Vec::new();
        for p in crate::collections::seq_iter(arg_vec) {
            if !v::is_sym_id(p) {
                panic!("fn: parameter must be a symbol");
            }
            param_ids.push(v::as_sym_id(p));
        }
        let arity = param_ids.len();

        // Free-var analysis. Capture only the names the OUTER env
        // actually has bindings for; everything else is either a
        // top-level fn (resolved statically inside the inner body)
        // or genuinely unbound (caught when the inner body compiles).
        let captures: Vec<u32> = {
            let raw = {
                let sym_ref = &*self.sym;
                crate::freevars::free_vars(body_forms, &param_ids, &|id| {
                    sym_ref.name(id).to_string()
                })
            };
            raw.into_iter().filter(|id| env.lookup(*id).is_some()).collect()
        };

        // Declare the inner function: signature is (self_fn, p0..pn-1).
        *self.anon_counter += 1;
        let name = format!("__lambda_{}", *self.anon_counter);
        let mut all_params = vec![Type::I64; 1 + arity];
        let _ = &mut all_params; // silence
        let inner_fref = self.mb.declare_func(&name, &vec![Type::I64; 1 + arity], Some(Type::I64));
        self.func_refs.insert(name, inner_fref);
        self.compile_closure_body(inner_fref, &param_ids, &captures, body_forms);

        // Emit allocation in the OUTER function.
        let live_pre = env.live_values();
        let n_caps = captures.len();
        if n_caps == 0 {
            let alloc_fref = *self
                .func_refs
                .get("__alloc_fn")
                .expect("__alloc_fn extern not registered");
            let fr_const = fb.iconst(Type::I64, inner_fref.as_u32() as i64);
            let arity_const = fb.iconst(Type::I64, arity as i64);
            fb.safepoint(&live_pre);
            return fb
                .call(alloc_fref, &[fr_const, arity_const])
                .expect("__alloc_fn returns a value");
        }

        // Spill captures to a single 8*N-byte GC-rooted stack slot,
        // then pass its address to `clj_alloc_closure(fr, arity,
        // n_caps, ptr)`. The slot's `size` is honored by the
        // toolkit's frame layout (see `alloc_root_slot_bytes`).
        let buf_slot = fb.create_stack_slot((n_caps * 8) as u32, /*is_gc_root=*/ true);
        let buf_addr = fb.stack_addr(buf_slot);
        for (i, &cap_id) in captures.iter().enumerate() {
            let val = env
                .lookup(cap_id)
                .expect("capture not in outer env");
            fb.store(val, buf_addr, (i * 8) as i32);
        }
        let slot_addr_i64 = fb.bitcast(buf_addr, Type::I64);
        let alloc_fref = *self
            .func_refs
            .get("__alloc_closure")
            .expect("__alloc_closure extern not registered");
        let fr_const = fb.iconst(Type::I64, inner_fref.as_u32() as i64);
        let arity_const = fb.iconst(Type::I64, arity as i64);
        let n_caps_const = fb.iconst(Type::I64, n_caps as i64);
        fb.safepoint(&live_pre);
        fb.call(
            alloc_fref,
            &[fr_const, arity_const, n_caps_const, slot_addr_i64],
        )
        .expect("__alloc_closure returns a value")
    }

    /// Compile a closure body. The generated function takes
    /// `self_fn` as parameter 0; user params follow. Captures are
    /// loaded from `self_fn` at the start of the body and bound to
    /// their original names in the inner env so the body's normal
    /// symbol-lookup path picks them up.
    fn compile_closure_body(
        &mut self,
        fref: FuncRef,
        params: &[u32],
        captures: &[u32],
        body_forms: u64,
    ) {
        let mut fb = self.mb.define_func(fref);
        let mut env = Env::new();
        let entry = fb.entry_block();

        // Read the entry params: self_fn at index 0, user params at 1..n.
        let self_fn_at_entry = fb.block_param(entry, 0);
        let user_params_at_entry: Vec<Value> = (0..params.len())
            .map(|i| fb.block_param(entry, 1 + i))
            .collect();

        // Trampoline through a `loop_header` so `recur` has a normal
        // (non-entry) jump target. See compile_function_body for the
        // reason — entry blocks have ABI constraints from the calling
        // convention that don't accommodate back-edges.
        let mut header_param_tys = vec![Type::I64; 1 + params.len()];
        let _ = &mut header_param_tys;
        let loop_header = fb.create_block(&vec![Type::I64; 1 + params.len()]);
        let mut to_header: Vec<Value> = Vec::with_capacity(1 + params.len());
        to_header.push(self_fn_at_entry);
        to_header.extend_from_slice(&user_params_at_entry);
        fb.jump(loop_header, &to_header);
        fb.switch_to_block(loop_header);

        // Re-bind from the loop header's params.
        let self_fn_tagged = fb.block_param(loop_header, 0);
        let self_fn = if !captures.is_empty() {
            fb.payload(self_fn_tagged)
        } else {
            self_fn_tagged
        };

        for (i, &pid) in params.iter().enumerate() {
            let val = fb.block_param(loop_header, 1 + i);
            env.bind(pid, val);
        }

        // Load each capture from self_fn at the right offset and
        // bind it as if it were a local.
        for (i, &cap_id) in captures.iter().enumerate() {
            let off = crate::namespace::fn_capture_offset(i);
            let v = fb.load(Type::I64, self_fn, off);
            env.bind(cap_id, v);
        }

        self.loop_targets.push(LoopTarget {
            block: loop_header,
            arity: params.len(),
            prepend: vec![self_fn_tagged],
        });
        let result = self.lower_do(&mut fb, &mut env, body_forms);
        fb.ret(result);
        self.mb.finish_func(fref, fb);
        self.loop_targets.pop();
    }

    fn lower_loop(&mut self, fb: &mut FunctionBuilder, env: &mut Env, args: u64) -> Value {
        // (loop [n0 v0 n1 v1 …] body…) — like let, but the binding
        // names are the targets of any enclosed `recur`. At the IR
        // level we open a fresh basic block whose block params are the
        // bindings, jump to it with the init values, then lower the
        // body in a scope that binds each name to its block param.
        let bindings_form = v::first(args);
        let body_forms = v::rest(args);
        let pairs: Vec<u64> = crate::collections::seq_iter(bindings_form).collect();
        if pairs.len() % 2 != 0 {
            panic!("loop: bindings must come in name/value pairs");
        }
        let mut names: Vec<u32> = Vec::new();
        let mut init_vals: Vec<Value> = Vec::new();
        for chunk in pairs.chunks(2) {
            let name_v = chunk[0];
            if !v::is_sym_id(name_v) {
                panic!("loop: binding name must be a symbol");
            }
            names.push(v::as_sym_id(name_v));
            // Init values evaluate in the OUTER env (sequential let
            // semantics: later inits see earlier names — rebound on
            // each push to env so the second loop test below holds).
            let val = self.lower_expr(fb, env, chunk[1]);
            env.push();
            env.bind(v::as_sym_id(name_v), val);
            init_vals.push(val);
        }
        // Pop the temporary inner scopes we used for sequential let
        // semantics during init.
        for _ in 0..names.len() {
            env.pop();
        }

        let header = fb.create_block(&vec![Type::I64; names.len()]);
        fb.jump(header, &init_vals);
        fb.switch_to_block(header);

        // Bind each name to the corresponding block_param.
        env.push();
        for (i, &nid) in names.iter().enumerate() {
            env.bind(nid, fb.block_param(header, i));
        }

        // Push as the new innermost recur target (no prepend — `loop`
        // doesn't pass any hidden self argument).
        self.loop_targets.push(LoopTarget {
            block: header,
            arity: names.len(),
            prepend: Vec::new(),
        });
        let result = self.lower_do(fb, env, body_forms);
        self.loop_targets.pop();
        env.pop();
        result
    }

    fn lower_recur(&mut self, fb: &mut FunctionBuilder, env: &mut Env, args: u64) -> Value {
        // Lower each arg in the current env.
        let arg_forms: Vec<u64> = v::list_iter(args).collect();
        let arg_vals: Vec<Value> = arg_forms
            .iter()
            .map(|f| self.lower_expr(fb, env, *f))
            .collect();

        let target = self
            .loop_targets
            .last()
            .expect("recur: no enclosing loop or fn body");
        if arg_vals.len() != target.arity {
            panic!(
                "recur: expected {} arg(s), got {}",
                target.arity,
                arg_vals.len()
            );
        }

        // Build the full jump-arg list: prepend (e.g. self_fn) then
        // the user-supplied recur args.
        let mut full_args: Vec<Value> = target.prepend.clone();
        full_args.extend(arg_vals.iter().copied());

        fb.jump(target.block, &full_args);

        // The current block is now terminated. lower_expr's caller
        // expects a Value, but anything after the recur is dead code.
        // Switch to a fresh unreachable block so subsequent IR can be
        // emitted without panicking, and return a dummy const.
        let dead = fb.create_block(&[]);
        fb.switch_to_block(dead);
        fb.iconst(Type::I64, 0)
    }
}

// ── Truthiness ──────────────────────────────────────────────────────

/// Truthy = (val != NIL) && (val != FALSE).
fn compute_truthy(fb: &mut FunctionBuilder, val: Value) -> Value {
    let nil_c = fb.iconst(Type::I64, v::NIL as i64);
    let false_c = fb.iconst(Type::I64, v::FALSE as i64);
    let ne_nil = fb.icmp(CmpOp::Ne, val, nil_c);
    let ne_false = fb.icmp(CmpOp::Ne, val, false_c);
    fb.and(ne_nil, ne_false)
}

// ── Environment ─────────────────────────────────────────────────────

pub struct Env {
    scopes: Vec<HashMap<u32, Value>>,
}

impl Env {
    pub fn new() -> Self {
        Env {
            scopes: vec![HashMap::new()],
        }
    }
    pub fn push(&mut self) {
        self.scopes.push(HashMap::new());
    }
    pub fn pop(&mut self) {
        self.scopes.pop();
    }
    pub fn bind(&mut self, id: u32, v: Value) {
        self.scopes.last_mut().unwrap().insert(id, v);
    }
    pub fn lookup(&self, id: u32) -> Option<Value> {
        for s in self.scopes.iter().rev() {
            if let Some(&v) = s.get(&id) {
                return Some(v);
            }
        }
        None
    }

    /// Snapshot of every IR `Value` currently bound in any scope.
    /// Over-approximates the live set at any point — a few false
    /// positives (e.g. shadowed bindings) are harmless. Used to feed
    /// the live-arg list of `fb.safepoint(...)` so the GC's stack-map
    /// updates the spill slots of these values when it relocates.
    pub fn live_values(&self) -> Vec<Value> {
        let mut out = Vec::new();
        for scope in &self.scopes {
            for &v in scope.values() {
                out.push(v);
            }
        }
        out
    }
}

impl Default for Env {
    fn default() -> Self {
        Self::new()
    }
}
