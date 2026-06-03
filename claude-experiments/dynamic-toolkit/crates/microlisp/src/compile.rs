//! Microlisp compiler: cons-tree → dynir.
//!
//! Each top-level form may declare zero or more functions in the
//! ModuleBuilder. The driver (in `lib.rs`) snapshots the builder, extends
//! the JitModule, and runs the entry function (if any).

use std::collections::HashMap;

use dynir::builder::{FunctionBuilder, ModuleBuilder};
use dynir::ir::{CmpOp, FuncRef, LiteralRef, Value};
use dynir::types::{Signature, Type};
use dynlower::LiteralPool;

use crate::symbols::SymbolTable;
use crate::value::{self as v, *};

pub struct Compiler<'a> {
    pub mb: &'a mut ModuleBuilder,
    pub func_refs: &'a mut HashMap<String, FuncRef>,
    pub sym: &'a mut SymbolTable,
    /// Counter for synthesized anonymous top-level expression names.
    pub anon_counter: &'a mut u32,
    /// JitModule's literal pool. Quote-literals whose payload is a heap
    /// pointer (cons cells, vectors, strings) are pushed here and loaded at
    /// runtime via `Inst::GcLiteral` so a moving GC can rewrite the slot
    /// when it relocates the underlying object.
    pub literal_pool: &'a LiteralPool,
    /// Stack of "in-flight" IR values: results computed by `lower_expr`
    /// that are about to be passed as arguments to a parent call but
    /// haven't been consumed yet. Pushed when a value is produced as an
    /// arg, popped after the parent's call completes.
    ///
    /// At every safepoint the compiler emits, the `live` set is:
    ///     env.live_values() ∪ live_stack
    /// This is necessary because nested allocator calls each emit their
    /// own safepoint; without listing the OUTER call's already-evaluated
    /// args, the GC would relocate them out from under the IR's spill
    /// slots and the outer call's `Inst::Call` would receive stale bits.
    /// Concrete example: in `(cons A (cons B C))`, the inner cons's
    /// safepoint must list `A`'s value as live — otherwise its spill
    /// slot isn't included in the stack map and a moving GC corrupts it.
    pub live_stack: Vec<Value>,
}

/// Names of primitive externs that allocate on the GC heap. Every IR call
/// to one of these MUST be preceded by an `Inst::Safepoint` listing the
/// caller's live values — `Module::validate_safepoints` enforces this.
pub const ALLOCATING_PRIMITIVES: &[&str] = &["cons", "append"];

/// Look up the [`FuncRef`]s of every allocating primitive currently
/// declared. Used at extend-time to feed `Module::validate_safepoints`.
pub fn allocator_frefs(func_refs: &HashMap<String, FuncRef>) -> Vec<FuncRef> {
    ALLOCATING_PRIMITIVES
        .iter()
        .filter_map(|name| func_refs.get(*name).copied())
        .collect()
}

#[derive(Debug)]
pub enum TopResult {
    /// `(define (name ...) body)` — a function was declared. Caller will
    /// snapshot+extend; no entry to run.
    Define { name: String, fref: FuncRef },
    /// `(defmacro name pattern body)` — function was compiled and must be
    /// registered in macro_env after extend.
    Defmacro { name: String, fref: FuncRef },
    /// Anonymous top-level expression compiled into a 0-arg function.
    Expr(FuncRef),
    /// Pure side-effect / no-op (e.g. blank line).
    None,
}

impl<'a> Compiler<'a> {
    pub fn compile_top(&mut self, form: u64) -> TopResult {
        // (define ...) and (defmacro ...) are recognized at the top level only.
        if is_cons(form) {
            let head = car(form);
            if is_symbol(head) {
                let head_name = self.sym.name(as_symbol_id(head)).to_string();
                match head_name.as_str() {
                    "define" => return self.compile_define(form),
                    "defmacro" => return self.compile_defmacro(form),
                    _ => {}
                }
            }
        }
        // Otherwise wrap in an anonymous 0-arg function.
        *self.anon_counter += 1;
        let name = format!("__top_{}", *self.anon_counter);
        let fref = self.mb.declare_func(&name, &[], Some(Type::I64));
        self.func_refs.insert(name.clone(), fref);
        self.compile_function_body(fref, &[], form);
        TopResult::Expr(fref)
    }

    fn compile_define(&mut self, form: u64) -> TopResult {
        // (define (name args...) body...)  OR  (define name expr)
        let rest = cdr(form);
        let target = car(rest);
        let body_forms = cdr(rest);
        if is_cons(target) {
            // function form
            let name_sym = car(target);
            let params_list = cdr(target);
            assert!(
                is_symbol(name_sym),
                "define: function name must be a symbol"
            );
            let name = self.sym.name(as_symbol_id(name_sym)).to_string();
            let mut params: Vec<u32> = Vec::new();
            for p in list_iter(params_list) {
                assert!(is_symbol(p), "define: param must be a symbol");
                params.push(as_symbol_id(p));
            }
            let param_tys = vec![Type::I64; params.len()];
            let fref = self.mb.declare_func(&name, &param_tys, Some(Type::I64));
            self.func_refs.insert(name.clone(), fref);
            // Body is a list of expressions — wrap in begin if more than one.
            let body = wrap_body(body_forms, self.sym);
            self.compile_function_body(fref, &params, body);
            TopResult::Define { name, fref }
        } else {
            // Variable form — represent as a 0-arg getter for v0. (We don't
            // have first-class globals; this lets `(define x 5)` followed by
            // `(x)` work.)
            assert!(
                is_symbol(target),
                "define: target must be symbol or function form"
            );
            let name = self.sym.name(as_symbol_id(target)).to_string();
            let init = car(body_forms);
            let fref = self.mb.declare_func(&name, &[], Some(Type::I64));
            self.func_refs.insert(name.clone(), fref);
            self.compile_function_body(fref, &[], init);
            TopResult::Define { name, fref }
        }
    }

    fn compile_defmacro(&mut self, form: u64) -> TopResult {
        // (defmacro name pattern body...)
        // pattern is either a symbol (whole args list) or a list (a b . rest)
        // for sugar destructuring.
        let rest = cdr(form);
        let name_sym = car(rest);
        assert!(is_symbol(name_sym), "defmacro: name must be a symbol");
        let name = self.sym.name(as_symbol_id(name_sym)).to_string();
        let after_name = cdr(rest);
        let pattern = car(after_name);
        let body_forms = cdr(after_name);

        // Unique internal name so the macro fn can't collide with a regular fn.
        let internal = format!("__macro__{name}");
        let fref = self
            .mb
            .declare_func(&internal, &[Type::I64], Some(Type::I64));
        self.func_refs.insert(internal.clone(), fref);

        // Synthesize a body that destructures `args` according to `pattern`.
        let args_id = self.sym.intern("__macro_args__");
        let body = if is_symbol(pattern) {
            // (defmacro name args body...)  — bind whole list to the pattern symbol.
            let pat_id = as_symbol_id(pattern);
            let raw_body = wrap_body(body_forms, self.sym);
            // (let ((<pat> __macro_args__)) raw_body)
            wrap_let_single(pat_id, encode_sym(args_id), raw_body, self.sym)
        } else if is_cons(pattern) || is_nil(pattern) {
            // List pattern with optional dotted tail: bind each fixed element
            // via car/cdr chains, last variable becomes the rest.
            let raw_body = wrap_body(body_forms, self.sym);
            wrap_destructure(pattern, args_id, raw_body, self.sym)
        } else {
            panic!(
                "defmacro: pattern must be a symbol or list, got 0x{:016x}",
                pattern
            );
        };

        self.compile_function_body(fref, &[args_id], body);
        TopResult::Defmacro { name, fref }
    }

    fn compile_function_body(&mut self, fref: FuncRef, params: &[u32], body: u64) {
        let mut fb = self.mb.define_func(fref);
        let mut env = Env::new();
        let entry = fb.entry_block();
        for (i, &p) in params.iter().enumerate() {
            let val = fb.block_param(entry, i);
            env.bind(p, val);
        }
        let result = self.lower_expr(&mut fb, &mut env, body);
        fb.ret(result);
        self.mb.finish_func(fref, fb);
    }

    fn lower_expr(&mut self, fb: &mut FunctionBuilder, env: &mut Env, form: u64) -> Value {
        // Self-evaluating literals: numbers, nil, true/false.
        if v::is_number(form) || is_nil(form) || form == v::TRUE || form == v::FALSE {
            return fb.iconst(Type::I64, form as i64);
        }
        // Bare symbol: variable lookup.
        if is_symbol(form) {
            let id = as_symbol_id(form);
            if let Some(val) = env.lookup(id) {
                return val;
            }
            // Top-level zero-arg "variable" defined via (define x v) — call it.
            let name = self.sym.name(id).to_string();
            if let Some(&fref) = self.func_refs.get(&name) {
                return fb.call(fref, &[]).expect("zero-arg fn returns a value");
            }
            panic!("undefined variable: {}", name);
        }
        if !is_cons(form) {
            panic!("unknown literal form: 0x{:016x}", form);
        }

        // Cons form: special form or call.
        let head = car(form);
        if is_symbol(head) {
            let head_name = self.sym.name(as_symbol_id(head)).to_string();
            match head_name.as_str() {
                "quote" => return self.lower_quote(fb, cdr(form)),
                "if" => return self.lower_if(fb, env, cdr(form)),
                "begin" => return self.lower_begin(fb, env, cdr(form)),
                "let" => return self.lower_let(fb, env, cdr(form)),
                "set!" => return self.lower_set(fb, env, cdr(form)),
                "list" => return self.lower_list(fb, env, cdr(form)),
                _ => {}
            }
        }
        self.lower_call(fb, env, form)
    }

    fn lower_quote(&self, fb: &mut FunctionBuilder, args: u64) -> Value {
        let datum = car(args);
        if datum_needs_gc_pool(datum) {
            // Heap-pointer payload (cons / vector / string). Push to the
            // GC-traced literal pool so a moving collector can rewrite the
            // slot in place, and emit a load through the pool.
            let idx = self.literal_pool.push(datum);
            fb.gc_literal(LiteralRef::from_u32(idx as u32))
        } else {
            // Immortal / non-pointer (number, nil, true, false, symbol id).
            // Safe to bake the bits directly into the instruction stream.
            fb.iconst(Type::I64, datum as i64)
        }
    }

    fn lower_if(&mut self, fb: &mut FunctionBuilder, env: &mut Env, args: u64) -> Value {
        let cond_form = car(args);
        let then_form = car(cdr(args));
        let else_form = if is_cons(cdr(cdr(args))) {
            car(cdr(cdr(args)))
        } else {
            v::NIL
        };

        let cond = self.lower_expr(fb, env, cond_form);
        // Truthy = !nil && !#false. Build a boolean by comparing against both.
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

    fn lower_begin(&mut self, fb: &mut FunctionBuilder, env: &mut Env, args: u64) -> Value {
        if is_nil(args) {
            return fb.iconst(Type::I64, v::NIL as i64);
        }
        let mut last: Option<Value> = None;
        let items: Vec<u64> = list_iter(args).collect();
        for x in items {
            last = Some(self.lower_expr(fb, env, x));
        }
        last.unwrap()
    }

    fn lower_let(&mut self, fb: &mut FunctionBuilder, env: &mut Env, args: u64) -> Value {
        // (let ((x v) (y w)) body...)
        let bindings = car(args);
        let body_forms = cdr(args);
        env.push();
        for binding in list_iter(bindings) {
            let name_v = car(binding);
            let val_form = car(cdr(binding));
            assert!(is_symbol(name_v), "let: binding name must be a symbol");
            let val = self.lower_expr(fb, env, val_form);
            env.bind(as_symbol_id(name_v), val);
        }
        let result = self.lower_begin(fb, env, body_forms);
        env.pop();
        result
    }

    fn lower_set(&mut self, fb: &mut FunctionBuilder, env: &mut Env, args: u64) -> Value {
        let name_v = car(args);
        let val_form = car(cdr(args));
        assert!(is_symbol(name_v), "set!: name must be a symbol");
        let new_val = self.lower_expr(fb, env, val_form);
        env.set(as_symbol_id(name_v), new_val);
        // set! returns nil
        fb.iconst(Type::I64, v::NIL as i64)
    }

    fn lower_list(&mut self, fb: &mut FunctionBuilder, env: &mut Env, args: u64) -> Value {
        // Build cons chain from the right. Each cons is an allocation
        // and emits its own safepoint via `lower_call`-style discipline:
        // evaluate args, push them onto `live_stack`, emit safepoint with
        // env + full live_stack, do the call, pop.
        let items: Vec<u64> = list_iter(args).collect();
        let cons_fref = *self.func_refs.get("cons").expect("cons primitive missing");
        // Evaluate items left-to-right; each result pushed onto live_stack.
        let mut item_vals: Vec<Value> = Vec::with_capacity(items.len());
        for x in items {
            let v = self.lower_expr(fb, env, x);
            self.live_stack.push(v);
            item_vals.push(v);
        }
        let mut tail = fb.iconst(Type::I64, v::NIL as i64);
        // Track tail on live_stack across each cons call.
        self.live_stack.push(tail);
        for &xv in item_vals.iter().rev() {
            // Live = env bindings ∪ everything in live_stack.
            let mut live = env.live_values();
            live.extend_from_slice(&self.live_stack);
            fb.safepoint(&live);
            tail = fb.call(cons_fref, &[xv, tail]).expect("cons returns value");
            // Update tail's slot on the stack.
            *self.live_stack.last_mut().unwrap() = tail;
        }
        // Pop tail and all item_vals.
        self.live_stack.pop(); // tail
        for _ in 0..item_vals.len() {
            self.live_stack.pop();
        }
        tail
    }

    fn lower_call(&mut self, fb: &mut FunctionBuilder, env: &mut Env, form: u64) -> Value {
        let head = car(form);
        let args = cdr(form);
        assert!(
            is_symbol(head),
            "call head must be a symbol; got 0x{:016x}",
            head
        );
        let head_name = self.sym.name(as_symbol_id(head)).to_string();
        // Evaluate args left-to-right; push each onto live_stack so a
        // safepoint emitted by a nested allocator call sees this arg as
        // live. (Without this, the inner safepoint's stack map omits our
        // already-evaluated args, and a moving GC fired by the inner call
        // relocates their cons cells without updating our spill slots.)
        let mut arg_vals: Vec<Value> = Vec::new();
        for a in list_iter(args).collect::<Vec<_>>() {
            let v = self.lower_expr(fb, env, a);
            self.live_stack.push(v);
            arg_vals.push(v);
        }
        let fref = *self
            .func_refs
            .get(&head_name)
            .unwrap_or_else(|| panic!("undefined function: {}", head_name));
        // Allocator calls require a preceding safepoint listing every
        // live IR value: env bindings + the entire live_stack (which
        // covers our args plus any partially-evaluated parent call's args).
        // `Module::validate_safepoints` enforces the safepoint's presence.
        if ALLOCATING_PRIMITIVES.contains(&head_name.as_str()) {
            let mut live = env.live_values();
            live.extend_from_slice(&self.live_stack);
            fb.safepoint(&live);
        }
        let result = fb.call(fref, &arg_vals).unwrap_or_else(|| {
            // Void return — produce nil so callers have a value.
            fb.iconst(Type::I64, v::NIL as i64)
        });
        // Pop our args from the live stack (they're consumed by the call).
        for _ in 0..arg_vals.len() {
            self.live_stack.pop();
        }
        result
    }
}

/// True when a NanBox literal carries a pointer into the GC heap and so
/// must go through the LiteralPool rather than being baked as an immediate.
/// Numbers, nil/true/false, and interned symbol ids are immortal and stay
/// inline.
fn datum_needs_gc_pool(datum: u64) -> bool {
    is_cons(datum)
}

// ── Helpers ─────────────────────────────────────────────────────────

/// Wrap a body list `(e1 e2 ... eN)` into a single `begin` form when there's
/// more than one expression. A single-expression body is unwrapped.
pub fn wrap_body(body_list: u64, sym: &mut SymbolTable) -> u64 {
    if is_nil(body_list) {
        return v::NIL;
    }
    if is_nil(cdr(body_list)) {
        return car(body_list);
    }
    let begin_id = sym.intern("begin");
    dynobj::roots::with_scope(2, |scope| {
        let head = scope.root::<NanBoxTag>(encode_sym(begin_id));
        let tail = scope.root::<NanBoxTag>(body_list);
        alloc_cons(scope, &head, &tail).get()
    })
}

/// `(let ((<name> <value-form>)) <body>)`.
pub fn wrap_let_single(name: u32, value_form: u64, body: u64, sym: &mut SymbolTable) -> u64 {
    let let_sym_bits = encode_sym(sym.intern("let"));
    // 5 input roots + 1 result per alloc_cons * 6 alloc_cons calls = 11.
    // Round up generously.
    dynobj::roots::with_scope(16, |scope| {
        let let_sym = scope.root::<NanBoxTag>(let_sym_bits);
        let name_sym = scope.root::<NanBoxTag>(encode_sym(name));
        let value = scope.root::<NanBoxTag>(value_form);
        let body_root = scope.root::<NanBoxTag>(body);
        let nil_root = scope.root::<NanBoxTag>(v::NIL);
        // (value . NIL)
        let value_list = alloc_cons(scope, &value, &nil_root);
        // (name value)
        let binding = alloc_cons(scope, &name_sym, &value_list);
        // ((name value))
        let bindings = alloc_cons(scope, &binding, &nil_root);
        // (body . NIL)
        let body_list = alloc_cons(scope, &body_root, &nil_root);
        // (bindings body)
        let bindings_body = alloc_cons(scope, &bindings, &body_list);
        // (let bindings body)
        alloc_cons(scope, &let_sym, &bindings_body).get()
    })
}

/// Build a `let` that destructures the args list according to `pattern`.
/// Pattern can be:
///   - nil   → no bindings
///   - (a b c)            → fixed arity
///   - (a b . rest)       → fixed prefix + rest
///   - (a b c . rest)     → same
fn wrap_destructure(pattern: u64, args_sym: u32, body: u64, sym: &mut SymbolTable) -> u64 {
    // Walk the pattern, peeling off head/tail with car/cdr.
    let car_sym = sym.intern("car");
    let cdr_sym = sym.intern("cdr");
    let car_v = encode_sym(car_sym);
    let cdr_v = encode_sym(cdr_sym);
    let _ = (car_v, cdr_v);

    // Build bindings list. Each call to `cons2` allocates via the safe
    // rooted path and immediately returns the bits; we hold them in
    // `bindings` (a Rust Vec) — safe because compile-time helpers don't
    // trigger GC (the only safe-by-construction context where Rust-side
    // raw u64 cons handles are OK).
    let mut current_cursor = args_sym;
    let mut bindings: Vec<(u32, u64)> = Vec::new();
    let mut p = pattern;
    let mut cursor_idx = 0u32;
    loop {
        if is_nil(p) {
            break;
        }
        if is_symbol(p) {
            let pat_id = as_symbol_id(p);
            bindings.push((pat_id, encode_sym(current_cursor)));
            break;
        }
        if !is_cons(p) {
            panic!("destructure pattern must be cons / symbol / nil");
        }
        let elem = car(p);
        assert!(
            is_symbol(elem),
            "destructure: each pattern element must be a symbol"
        );
        let elem_id = as_symbol_id(elem);
        let car_call = cons2(
            encode_sym(car_sym),
            cons2(encode_sym(current_cursor), v::NIL),
        );
        bindings.push((elem_id, car_call));
        let next_p = cdr(p);
        if is_nil(next_p) {
            break;
        }
        cursor_idx += 1;
        let next_cursor_id = sym.intern(&format!("__cursor_{cursor_idx}__"));
        let cdr_call = cons2(
            encode_sym(cdr_sym),
            cons2(encode_sym(current_cursor), v::NIL),
        );
        bindings.push((next_cursor_id, cdr_call));
        current_cursor = next_cursor_id;
        p = next_p;
    }

    let let_sym = encode_sym(sym.intern("let"));
    let mut result = body;
    for (name, val_form) in bindings.into_iter().rev() {
        let binding = cons2(encode_sym(name), cons2(val_form, v::NIL));
        let bindings_list = cons2(binding, v::NIL);
        result = cons2(let_sym, cons2(bindings_list, cons2(result, v::NIL)));
    }
    result
}

/// Compile-time-only cons builder. Wraps `alloc_cons_from_raw` and
/// returns raw `u64`; only safe in code paths that don't trigger GC
/// (currently: the macroexpansion-time IR-construction helpers in this
/// module). The `with_scope` per call is overhead but keeps the rooting
/// discipline honest and prevents any future change to `gc.alloc`'s
/// auto-collection behaviour from silently breaking these helpers.
fn cons2(car: u64, cdr: u64) -> u64 {
    // 3 slots: alloc_cons_from_raw needs car + cdr + result.
    dynobj::roots::with_scope(3, |scope| alloc_cons_from_raw(scope, car, cdr).get())
}

/// Truthy = (val != NIL) && (val != FALSE).
fn compute_truthy(fb: &mut FunctionBuilder, val: Value) -> Value {
    let nil_c = fb.iconst(Type::I64, v::NIL as i64);
    let false_c = fb.iconst(Type::I64, v::FALSE as i64);
    let ne_nil = fb.icmp(CmpOp::Ne, val, nil_c);
    let ne_false = fb.icmp(CmpOp::Ne, val, false_c);
    // Both must be true → AND. Use bit-AND on the booleans (Type::I8).
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
    pub fn set(&mut self, id: u32, v: Value) {
        for s in self.scopes.iter_mut().rev() {
            if s.contains_key(&id) {
                s.insert(id, v);
                return;
            }
        }
        panic!("set! on undefined variable id={}", id);
    }

    /// Snapshot of every Value bound in any scope. Over-approximates the
    /// liveness set at a given point — the GC scans these as roots so they
    /// stay valid across an allocator-triggered collection. False positives
    /// (binding shadowed in an inner scope) are harmless: the GC just sees
    /// each value via more than one reference.
    pub fn live_values(&self) -> Vec<Value> {
        let mut out = Vec::new();
        for scope in &self.scopes {
            for v in scope.values() {
                out.push(*v);
            }
        }
        out
    }
}

// ── Primitive declaration ───────────────────────────────────────────

/// Declare every primitive as an extern in the ModuleBuilder.
/// Returns the externs slice (in declaration order) and updates `func_refs`.
pub fn declare_primitives(
    mb: &mut ModuleBuilder,
    func_refs: &mut HashMap<String, FuncRef>,
) -> Vec<*const u8> {
    use crate::prims::{PrimSig, all_prims};
    let mut externs: Vec<*const u8> = Vec::new();
    for prim in all_prims() {
        let params = match prim.sig {
            PrimSig::Unary => vec![Type::I64],
            PrimSig::Binary => vec![Type::I64, Type::I64],
        };
        let sig = Signature {
            params,
            ret: Some(Type::I64),
        };
        let fref = mb.declare_extern(prim.name, sig);
        func_refs.insert(prim.name.to_string(), fref);
        externs.push(prim.ptr);
    }
    externs
}
