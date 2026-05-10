//! Macroexpansion pass.
//!
//! Walks a form tree, replacing macro invocations with the result of
//! JIT-calling the macro's body. The macro body is regular compiled
//! code in the same `JitModule` — same-image macros, per the design
//! in `docs/microlisp-plan.md`.
//!
//! The expander runs ONCE before each top-level form is compiled.
//! It expands the head position to fixed point, then recurses into
//! sub-forms (the args of a non-macro call). Special-form heads
//! (`def`, `defmacro`, `if`, `let`, `do`, `fn`, `quote`) are NOT
//! expanded as macros even if a homonymous macro exists.

use dynlang::{GcPolicy, gc::DynGcRuntime};
use dynlower::JitModule;
use dynir::ir::FuncRef;

use crate::namespace::{
    fn_func_ref, ns_lookup, var_is_macro, var_root,
};
use crate::symbols::SymbolTable;
use crate::value as v;

/// Special form heads that the expander must NOT treat as macros.
const SPECIAL_FORMS: &[&str] = &[
    "def", "defmacro", "fn", "if", "let", "do", "quote",
];

pub struct ExpandCtx<'a> {
    pub core_ns: u64,
    pub sym: &'a mut SymbolTable,
    pub gc: &'a DynGcRuntime,
    pub jit: &'a JitModule,
    pub jit_gc_policy: GcPolicy,
    /// Bound on rounds of head-expansion to catch infinite loops in
    /// macros that always re-emit themselves.
    pub max_iters: usize,
}

impl<'a> ExpandCtx<'a> {
    pub fn expand_all(&mut self, form: u64) -> u64 {
        // Head expansion to fixed point, then recurse into subforms.
        let mut current = form;
        for _ in 0..self.max_iters {
            let (next, expanded) = self.expand_one(current);
            current = next;
            if !expanded {
                break;
            }
        }
        // Recurse into subforms. We only walk lists; atoms have no
        // sub-structure. Special forms get partial recursion (we
        // don't expand inside `quote`, etc.).
        if v::is_ptr(current) {
            current = self.walk_subforms(current);
        }
        current
    }

    /// One step of head-position expansion. Returns (form, did_expand).
    fn expand_one(&mut self, form: u64) -> (u64, bool) {
        if !v::is_ptr(form) {
            return (form, false);
        }
        let head = v::first(form);
        if !v::is_sym_id(head) {
            return (form, false);
        }
        // Skip special forms.
        let head_name = self.sym.name(v::as_sym_id(head)).to_string();
        if SPECIAL_FORMS.contains(&head_name.as_str()) {
            return (form, false);
        }
        // Look up Var in clojure.core; check :macro flag.
        let sym_val = head;
        let var = ns_lookup(self.core_ns, sym_val);
        if !v::is_ptr(var) || !var_is_macro(var) {
            return (form, false);
        }
        // It's a macro. Collect args (the rest of the form), JIT-call
        // the macro fn, and replace the form with its result.
        let args: Vec<u64> = v::list_iter(v::rest(form)).collect();
        let fn_obj = var_root(var);
        if !v::is_ptr(fn_obj) {
            panic!("macro Var.root is not an Fn: {}", head_name);
        }
        let fref = FuncRef::from_u32(fn_func_ref(fn_obj));
        match self.gc.run_jit(self.jit, fref, &args, self.jit_gc_policy) {
            dynlower::JitOutcome::Value(result) => (result, true),
            dynlower::JitOutcome::Void => (v::NIL, true),
            other => panic!(
                "unexpected JIT outcome from macro `{}`: {:?}",
                head_name, other
            ),
        }
    }

    /// Recursively expand subforms.
    ///
    /// For non-special-form lists `(f arg1 arg2 ...)`, expand each
    /// arg. For special forms, currently we punt: we leave the
    /// special-form structure as-is. (More precise per-special-form
    /// handling — only expanding the bodies, not the binding-name
    /// lists — is a refinement; the current behavior happens to work
    /// for our tests because macros only appear in expression
    /// positions.)
    fn walk_subforms(&mut self, form: u64) -> u64 {
        let head = v::first(form);
        if v::is_sym_id(head) {
            let head_name = self.sym.name(v::as_sym_id(head)).to_string();
            if SPECIAL_FORMS.contains(&head_name.as_str()) {
                // For now, don't recurse into special-form bodies.
                // This is conservative — it means a macro inside
                // (let [x ...] (macro-call ...)) won't expand. We'll
                // refine when needed.
                return form;
            }
        }
        // Generic call: expand each argument.
        let mut new_args: Vec<u64> = Vec::new();
        for a in v::list_iter(v::rest(form)) {
            new_args.push(self.expand_all(a));
        }
        // Rebuild the list: (head . expanded-args).
        // Use cons via the host so the new cells are properly rooted.
        dynobj::roots::with_scope(2 + new_args.len(), |scope| {
            let acc = scope.root::<v::NanBoxTag>(v::NIL);
            for x in new_args.into_iter().rev() {
                let new_bits = dynobj::roots::with_scope(3, |inner| {
                    v::alloc_list_cell_from_raw(inner, x, acc.get()).get()
                });
                acc.set(new_bits);
            }
            // Prepend the original head.
            let with_head = dynobj::roots::with_scope(3, |inner| {
                v::alloc_list_cell_from_raw(inner, head, acc.get()).get()
            });
            with_head
        })
    }
}
