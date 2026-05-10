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
use dynir::ir::{CmpOp, FuncRef, LiteralRef, Value};
use dynir::types::Type;
use dynlower::LiteralPool;

use crate::symbols::SymbolTable;
use crate::value as v;

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
    pub func_refs: &'a mut HashMap<String, FuncRef>,
    pub sym: &'a mut SymbolTable,
    pub anon_counter: &'a mut u32,
    /// Literal pool we push GC-managed compile-time constants into.
    /// Each push returns a slot index that becomes a `LiteralRef` in
    /// emitted IR; the JIT lowers `Inst::GcLiteral(idx)` to a load
    /// from `pool.base() + idx*8`. The pool is registered with the
    /// GC as a root source so moving collections rewrite the slots in
    /// place.
    pub literal_pool: &'a LiteralPool,
}

impl<'a> Compiler<'a> {
    pub fn compile_top(&mut self, form: u64) -> TopResult {
        // Recognize `(def NAME EXPR)` and `(defmacro NAME [args] body)`
        // at top level.
        if v::is_ptr(form) {
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
    fn compile_function_body(&mut self, fref: FuncRef, params: &[u32], body_forms: u64) {
        let mut fb = self.mb.define_func(fref);
        let mut env = Env::new();
        let entry = fb.entry_block();
        for (i, &pid) in params.iter().enumerate() {
            let val = fb.block_param(entry, i);
            env.bind(pid, val);
        }
        let result = self.lower_do(&mut fb, &mut env, body_forms);
        fb.ret(result);
        self.mb.finish_func(fref, fb);
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
        let is_l = crate::collections::is_list(form);
        let tid = unsafe { v::read_type_id(v::as_ptr(form)) };
        let (sid, kid, str_id, list_id) = crate::host::with_host(|h| {
            (h.types.symbol.0, h.types.keyword.0, h.types.string.0, h.types.list.0)
        });
        eprintln!("[debug] form=0x{:016x} read_tid={} sym={} kw={} str={} list={} is_list={}",
                  form, tid, sid, kid, str_id, list_id, is_l);
        if !is_l {
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
                "quote" => return self.lower_quote(fb, v::rest(form)),
                "def" => panic!("def is only valid at top level"),
                "fn" => panic!(
                    "fn as an expression (first-class fn value) is not yet supported; \
                     use `(def NAME (fn [args] body))` at top level"
                ),
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
        if !v::is_sym_id(head) {
            panic!(
                "call head must be a symbol; got 0x{:016x} \
                 (higher-order calls arrive in a later checkpoint)",
                head
            );
        }
        let head_name = self.sym.name(v::as_sym_id(head)).to_string();
        let fref = *self
            .func_refs
            .get(&head_name)
            .unwrap_or_else(|| panic!("undefined function: {}", head_name));

        let arg_vals: Vec<Value> = v::list_iter(v::rest(form))
            .map(|a| self.lower_expr(fb, env, a))
            .collect();

        // Emit a safepoint BEFORE every call. This gives pure-compute
        // functions (fib, fact, etc. — no allocating primitives) the
        // poll points the multi-threaded GC needs: at each recursive
        // descent, the running thread checks `gc_requested` and
        // parks if a concurrent thread is requesting STW. Without
        // this, deep recursion races against a concurrent extend's
        // GC and reads relocated objects (SIGBUS).
        //
        // Live set is the union of env bindings and our already-
        // evaluated args. Both must be marked live so a relocating
        // GC at this point updates their spill slots.
        let mut live = env.live_values();
        live.extend_from_slice(&arg_vals);
        fb.safepoint(&live);

        fb.call(fref, &arg_vals)
            .expect("call returns a value (Type::I64)")
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
