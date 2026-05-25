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
    "def", "defmacro", "fn", "if", "let", "loop", "recur", "do", "quote",
    "quasiquote", "unquote", "unquote-splicing",
    "deftype*", "defprotocol", "extend-type",
    "try", "throw", "catch",
];

pub struct ExpandCtx<'a> {
    pub core_ns: u64,
    pub sym: &'a SymbolTable,
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
        // Macro fns can return any seqable shape — most commonly a
        // PList/Cons spine that core.clj's `(defn cons …)` builds,
        // but in principle any deftype that satisfies ISeq. After
        // each expansion step, normalize the result to a built-in
        // __ReaderList by walking through the seq protocol. The
        // rest of the walker (and the compiler downstream) only
        // needs to handle one shape.
        let mut current = self.maybe_normalize(form);
        for _ in 0..self.max_iters {
            let (next, expanded) = self.expand_one(current);
            current = next;
            if !expanded {
                break;
            }
            current = self.maybe_normalize(current);
        }
        if v::is_ptr(current) && crate::collections::is_list(current) {
            current = self.walk_subforms(current);
        }
        current
    }

    /// If `form` is a record-shaped LIST (claims `IList`), walk it
    /// via the protocol and rebuild as a built-in __ReaderList so
    /// the rest of the expander/compiler — which uses
    /// `v::first`/`v::rest` and only understands built-in lists —
    /// sees one shape.
    ///
    /// We gate on IList specifically (not just ISeq) because the
    /// compiler must distinguish "code-as-data list" from "data
    /// value." PersistentVector implements ISeq but is a vector
    /// literal: turning `(macro)` ⇒ `[1 2 3]` into `(1 2 3)` would
    /// flip the meaning.
    fn maybe_normalize(&self, form: u64) -> u64 {
        if !v::is_ptr(form) || crate::collections::is_list(form) {
            return form;
        }
        if !crate::collections::is_record(form) {
            return form;
        }
        let ilist_sym = crate::host::with_host(|h| h.ilist_sym);
        if !crate::protocol::type_satisfies(ilist_sym, form) {
            return form;
        }
        let items: Vec<u64> = crate::collections::seq_iter(form).collect();
        list_from_vec(&items)
    }

    /// One step of head-position expansion. Returns (form, did_expand).
    fn expand_one(&mut self, form: u64) -> (u64, bool) {
        // Only Lists carry a head/rest spine. Record-shaped lists
        // already got normalized to __ReaderList in expand_all, so we
        // can use the cheap v::first / v::rest accessors here.
        if !v::is_ptr(form) || !crate::collections::is_list(form) {
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
        // It's a macro. Collect args (the rest of the form), prepend
        // the &form/&env Clojure-style implicit parameters, pack the
        // whole thing into a list to match the def-fn single-list ABI,
        // JIT-call the macro fn, and replace the form with the result.
        //
        // &form is the entire (head args…) call form; &env is the
        // local-binding map at the call site (we don't track that
        // yet, so pass nil — same as `clojure.core/macroexpand-1`
        // when called outside a let-context).
        let args: Vec<u64> = v::list_iter(v::rest(form)).collect();
        let fn_obj = var_root(var);
        if !v::is_ptr(fn_obj) {
            panic!("macro Var.root is not an Fn: {}", head_name);
        }
        let fref = FuncRef::from_u32(fn_func_ref(fn_obj));
        let args_list = dynobj::roots::with_scope(args.len() + 8, |scope| {
            let acc = scope.root::<v::NanBoxTag>(v::NIL);
            // Walk user args back-to-front, then prepend env, then form.
            for &x in args.iter().rev() {
                let cell = dynobj::roots::with_scope(3, |inner| {
                    v::alloc_list_cell_from_raw(inner, x, acc.get()).get()
                });
                acc.set(cell);
            }
            let env_cell = dynobj::roots::with_scope(3, |inner| {
                v::alloc_list_cell_from_raw(inner, v::NIL, acc.get()).get()
            });
            acc.set(env_cell);
            let form_cell = dynobj::roots::with_scope(3, |inner| {
                v::alloc_list_cell_from_raw(inner, form, acc.get()).get()
            });
            acc.set(form_cell);
            acc.get()
        });
        match self.gc.run_jit(self.jit, fref, &[fn_obj, args_list], self.jit_gc_policy) {
            dynlower::JitOutcome::Value(result) => {
                if std::env::var("CLJ_DEBUG_MACRO").is_ok() {
                    let s = crate::printer::print(result, self.sym);
                    eprintln!("MACRO {} =>\n  {}", head_name, s);
                }
                (result, true)
            }
            dynlower::JitOutcome::Void => (v::NIL, true),
            other => panic!(
                "unexpected JIT outcome from macro `{}`: {:?}",
                head_name, other
            ),
        }
    }

    /// Recursively expand subforms. Each special form has its own
    /// sub-walk that respects what positions are EXPRESSION positions
    /// (where a macro call could legitimately appear) vs. NAME /
    /// PATTERN positions (where the form's literal shape is part of
    /// its meaning and must be preserved).
    fn walk_subforms(&mut self, form: u64) -> u64 {
        let head = v::first(form);
        let rest = v::rest(form);
        if v::is_sym_id(head) {
            let head_name = self.sym.name(v::as_sym_id(head)).to_string();
            match head_name.as_str() {
                // ── Skip-entirely: quoted data is literal. ─────────
                "quote" => return form,

                // ── Bindings: leave name slots untouched, expand the
                //    value forms and the body. ──────────────────────
                "let" | "loop" => {
                    let new_tail = self.walk_let_like(rest);
                    return rebuild_with_head(head, new_tail);
                }

                // ── (fn [params] body…) — preserve the param vector
                //    verbatim, expand each body form. Multi-arity
                //    `(fn ([a] …) ([a b] …))` recurses into each
                //    body block. ────────────────────────────────────
                "fn" => {
                    let new_tail = self.walk_fn(rest);
                    return rebuild_with_head(head, new_tail);
                }

                // ── Definers: name is literal, value form expands. ─
                "def" => {
                    let new_tail = self.walk_def(rest);
                    return rebuild_with_head(head, new_tail);
                }
                "defmacro" => {
                    // (defmacro NAME [params] body…) — same shape as
                    // a fn with a leading name slot.
                    let new_tail = self.walk_defmacro(rest);
                    return rebuild_with_head(head, new_tail);
                }

                // ── Sequencing/branching: every subform is an expr. ─
                "if" | "do" | "recur" => {
                    let new_tail = self.expand_each_form(rest);
                    return rebuild_with_head(head, new_tail);
                }

                // ── Quasiquote: rewrite to (cons/__concat/quote …)
                //    using the quasiquote pass. The result is itself
                //    an ordinary form that gets re-walked, so any
                //    macros inside an unquote do get expanded. ─────
                "quasiquote" => {
                    let inner = v::first(rest);
                    let mut qq = crate::quasiquote::QqCtx { sym: self.sym };
                    let rewritten = qq.expand_top(inner);
                    return self.expand_all(rewritten);
                }
                "unquote" | "unquote-splicing" => {
                    panic!("`unquote`/`unquote-splicing` outside a quasiquote");
                }

                _ => {}
            }
        }
        // Generic call: expand each argument.
        let head = v::first(form); // re-read in case of GC
        let new_args = self.expand_each_form(v::rest(form));
        rebuild_with_head(head, new_args)
    }

    /// `(let [n0 v0 n1 v1 ...] body…)` — names stay literal, values
    /// get expanded, body gets expanded.
    fn walk_let_like(&mut self, rest: u64) -> u64 {
        let bindings_form = v::first(rest);
        let body = v::rest(rest);

        // Walk bindings: name slot literal, value slot expanded.
        let pairs: Vec<u64> = crate::collections::seq_iter(bindings_form).collect();
        let mut new_pairs: Vec<u64> = Vec::with_capacity(pairs.len());
        let mut i = 0;
        while i < pairs.len() {
            new_pairs.push(pairs[i]);
            if i + 1 < pairs.len() {
                new_pairs.push(self.expand_all(pairs[i + 1]));
            }
            i += 2;
        }
        // Rebuild bindings as a Vector (matching reader output).
        let new_bindings = dynobj::roots::with_scope(new_pairs.len() + 8, |scope| {
            crate::collections::alloc_vector(scope, &new_pairs).get()
        });
        let new_body = self.expand_each_form(body);
        // Build the rest list: (new_bindings . new_body)
        dynobj::roots::with_scope(8, |scope| {
            v::alloc_list_cell_from_raw(scope, new_bindings, new_body).get()
        })
    }

    /// `(fn [params] body…)` or `(fn ([params] body…) ([params] body…))`.
    fn walk_fn(&mut self, rest: u64) -> u64 {
        let head_of_rest = v::first(rest);
        if v::is_ptr(head_of_rest) && crate::collections::is_list(head_of_rest) {
            // Multi-arity: each item in `rest` is a `([params] body…)` list.
            let arities: Vec<u64> = v::list_iter(rest).collect();
            let mut walked: Vec<u64> = Vec::with_capacity(arities.len());
            for arity_form in arities {
                let params = v::first(arity_form);
                let body = v::rest(arity_form);
                let expanded_body = self.expand_each_form(body);
                walked.push(dynobj::roots::with_scope(8, |scope| {
                    v::alloc_list_cell_from_raw(scope, params, expanded_body).get()
                }));
            }
            // Rebuild a plain list from `walked`.
            return list_from_vec(&walked);
        }
        // Single-arity: rest = ([params] body…). params literal, body expands.
        let params = v::first(rest);
        let body = v::rest(rest);
        let new_body = self.expand_each_form(body);
        dynobj::roots::with_scope(8, |scope| {
            v::alloc_list_cell_from_raw(scope, params, new_body).get()
        })
    }

    /// `(def NAME value)` — name literal, value expanded.
    fn walk_def(&mut self, rest: u64) -> u64 {
        let name = v::first(rest);
        let after_name = v::rest(rest);
        let value_form = v::first(after_name);
        let new_value = self.expand_all(value_form);
        // Rebuild as (NAME new_value).
        let with_value = dynobj::roots::with_scope(8, |scope| {
            v::alloc_list_cell_from_raw(scope, new_value, v::NIL).get()
        });
        dynobj::roots::with_scope(8, |scope| {
            v::alloc_list_cell_from_raw(scope, name, with_value).get()
        })
    }

    /// `(defmacro NAME [params] body…)` — name + params literal,
    /// body expands.
    fn walk_defmacro(&mut self, rest: u64) -> u64 {
        let name = v::first(rest);
        let after_name = v::rest(rest);
        let params = v::first(after_name);
        let body = v::rest(after_name);
        let new_body = self.expand_each_form(body);
        let with_params = dynobj::roots::with_scope(8, |scope| {
            v::alloc_list_cell_from_raw(scope, params, new_body).get()
        });
        dynobj::roots::with_scope(8, |scope| {
            v::alloc_list_cell_from_raw(scope, name, with_params).get()
        })
    }

    /// Expand each form in a list, returning a new list of expanded
    /// forms (preserving order, terminated by nil).
    fn expand_each_form(&mut self, list: u64) -> u64 {
        let items: Vec<u64> = v::list_iter(list).collect();
        let expanded: Vec<u64> = items.into_iter().map(|f| self.expand_all(f)).collect();
        list_from_vec(&expanded)
    }

}

/// Build `(head . tail-list)` where tail-list is already a list.
/// Free function (not a method) so it doesn't take `&mut self` —
/// avoids overlapping borrows when the tail is itself produced from
/// a `&mut self` call.
fn rebuild_with_head(head: u64, tail: u64) -> u64 {
    dynobj::roots::with_scope(8, |scope| {
        v::alloc_list_cell_from_raw(scope, head, tail).get()
    })
}

/// Convert a record-shaped list (PList / Cons spine produced by
/// `(defn cons …)`) into a built-in __ReaderList of the same
/// elements. Recurses into elements that are themselves record-lists
/// Build a Clojure list from a Vec, returning the head.
fn list_from_vec(items: &[u64]) -> u64 {
    dynobj::roots::with_scope(items.len() + 4, |scope| {
        let acc = scope.root::<v::NanBoxTag>(v::NIL);
        for x in items.iter().rev() {
            let cell = dynobj::roots::with_scope(3, |inner| {
                v::alloc_list_cell_from_raw(inner, *x, acc.get()).get()
            });
            acc.set(cell);
        }
        acc.get()
    })
}
