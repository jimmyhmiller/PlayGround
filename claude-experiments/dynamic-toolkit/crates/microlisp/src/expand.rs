//! Macroexpander.
//!
//! `macroexpand_all(form)` rewrites every macro call (head-position symbol
//! that's a key in `macro_env`) by invoking the JIT-compiled macro body and
//! substituting the result, recursively. Quasiquote is also rewritten here:
//! `\`(...)` becomes explicit `cons` / `list` / `append` calls before the
//! compiler sees it.

use std::collections::HashMap;

use dynir::ir::FuncRef;
use dynlang::GcPolicy;
use dynlang::gc::DynGcRuntime;
use dynlower::{JitModule, JitOutcome};
use dynobj::roots::with_scope;

use crate::host::Host;
use crate::symbols::SymbolTable;
use crate::value::*;

pub struct ExpandCtx<'a> {
    pub host: &'a Host,
    /// Cloned at construction so we don't have to borrow `host.macro_env`
    /// across JIT calls (the macro body might modify it via runtime defmacro
    /// in a future v0.5).
    pub macro_env: HashMap<u32, FuncRef>,
    pub jit: &'a JitModule,
    pub gc: &'a DynGcRuntime,
    pub max_iters: usize,
    /// Policy passed to every `gc.run_jit` invocation when invoking
    /// macro bodies. Set by the caller (typically the engine) so test
    /// harnesses can flip to `EveryPoint` for root-coverage validation.
    pub jit_gc_policy: GcPolicy,
}

impl<'a> ExpandCtx<'a> {
    pub fn expand_all(&mut self, form: u64) -> u64 {
        // First, rewrite quasiquotes (lexical, doesn't depend on macro env).
        let form = {
            let mut sym = self.host.sym.borrow_mut();
            quasiquote_rewrite(form, &mut sym)
        };
        self.expand_walk(form)
    }

    /// Each method below takes `u64` "bits" by value and returns `u64`.
    /// Inside, anything that crosses a potential GC point (a JIT call via
    /// `expand_one`, a recursive walker call, or any future GC-firing
    /// allocator) gets rooted in a `RootScope`.
    ///
    /// The scope's slot count is sized to the maximum number of in-flight
    /// values that single function holds simultaneously across a GC point.

    fn expand_walk(&mut self, form_bits: u64) -> u64 {
        with_scope(2, |scope| {
            let form = scope.root::<NanBoxTag>(form_bits);
            // Repeatedly expand the head-position macro at this level.
            for _ in 0..self.max_iters {
                let (next, did) = self.expand_one(form.get());
                form.set(next);
                if !did { break; }
            }
            // Walk subforms (may also trigger GC).
            self.walk_subforms(form.get())
        })
    }

    fn expand_one(&mut self, form_bits: u64) -> (u64, bool) {
        if !is_cons(form_bits) { return (form_bits, false); }
        let head = car(form_bits);
        if !is_symbol(head) { return (form_bits, false); }
        let id = as_symbol_id(head);
        let Some(&fref) = self.macro_env.get(&id) else {
            return (form_bits, false);
        };
        // GC POINT: gc.run_jit fires safepoints and may collect. The
        // args_list is what we pass to the macro body; it's a u64 we
        // hold across the JIT call → must be rooted.
        with_scope(1, |scope| {
            let args = scope.root::<NanBoxTag>(cdr(form_bits));
            let result = match self.gc.run_jit(self.jit, fref, &[args.get()], self.jit_gc_policy) {
                JitOutcome::Value(v) => v,
                JitOutcome::Void => NIL,
                other => panic!("macro fn returned non-value outcome: {other:?}"),
            };
            (result, true)
        })
    }

    fn walk_subforms(&mut self, form_bits: u64) -> u64 {
        if !is_cons(form_bits) { return form_bits; }
        // Snapshot head info BEFORE any GC point.
        let head = car(form_bits);
        let head_name = if is_symbol(head) {
            let sym = self.host.sym.borrow();
            Some(sym.name(as_symbol_id(head)).to_string())
        } else {
            None
        };
        match head_name.as_deref() {
            Some("quote") => form_bits, // do not descend
            Some("define") => self.walk_define(form_bits),
            Some("defmacro") => self.walk_defmacro(form_bits),
            Some("let") => self.walk_let(form_bits),
            Some("if") | Some("begin") | Some("set!") | Some("list") => {
                self.walk_each_after_head(form_bits)
            }
            _ => self.walk_each(form_bits),
        }
    }

    fn walk_each(&mut self, form_bits: u64) -> u64 {
        if is_nil(form_bits) { return form_bits; }
        if !is_cons(form_bits) { return form_bits; }
        with_scope(4, |scope| {
            let form = scope.root::<NanBoxTag>(form_bits);
            let head_bits = self.expand_walk(car(form.get()));
            let head = scope.root::<NanBoxTag>(head_bits);
            let tail_bits = self.walk_each(cdr(form.get()));
            let tail = scope.root::<NanBoxTag>(tail_bits);
            alloc_cons(scope, &head, &tail).get()
        })
    }

    fn walk_each_after_head(&mut self, form_bits: u64) -> u64 {
        with_scope(3, |scope| {
            let head = scope.root::<NanBoxTag>(car(form_bits));
            let tail_bits = self.walk_each(cdr(form_bits));
            let tail = scope.root::<NanBoxTag>(tail_bits);
            alloc_cons(scope, &head, &tail).get()
        })
    }

    fn walk_define(&mut self, form_bits: u64) -> u64 {
        // (define <target> body...) — descend into body only.
        with_scope(6, |scope| {
            let define_sym = scope.root::<NanBoxTag>(car(form_bits));
            let rest = cdr(form_bits);
            let target = scope.root::<NanBoxTag>(car(rest));
            let body_bits = self.walk_each(cdr(rest));
            let body = scope.root::<NanBoxTag>(body_bits);
            let inner = alloc_cons(scope, &target, &body);
            alloc_cons(scope, &define_sym, &inner).get()
        })
    }

    fn walk_defmacro(&mut self, form_bits: u64) -> u64 {
        // (defmacro name pattern body...) — descend body only.
        with_scope(8, |scope| {
            let defmacro_sym = scope.root::<NanBoxTag>(car(form_bits));
            let rest = cdr(form_bits);
            let name_sym = scope.root::<NanBoxTag>(car(rest));
            let after_name = cdr(rest);
            let pattern = scope.root::<NanBoxTag>(car(after_name));
            let body_bits = self.walk_each(cdr(after_name));
            let body = scope.root::<NanBoxTag>(body_bits);
            let pair = alloc_cons(scope, &pattern, &body);
            let triple = alloc_cons(scope, &name_sym, &pair);
            alloc_cons(scope, &defmacro_sym, &triple).get()
        })
    }

    fn walk_let(&mut self, form_bits: u64) -> u64 {
        // (let ((x v) ...) body...) — descend into binding values + body.
        // Slot count: let_sym, bindings_root, body, cursor, name_slot,
        //             val_slot, new_bindings_tail, reverse_acc, cursor2,
        //             head_slot, nil_root, plus alloc_cons intermediates.
        with_scope(48, |scope| {
            let let_sym = scope.root::<NanBoxTag>(car(form_bits));
            let rest = cdr(form_bits);
            let bindings_root = scope.root::<NanBoxTag>(car(rest));
            let body_bits = self.walk_each(cdr(rest));
            let body = scope.root::<NanBoxTag>(body_bits);

            // Walk each binding's value. Iterate via a rooted cursor —
            // collecting `Vec<u64>` of binding pointers up front would
            // leave them stale after `expand_walk` (which fires GC) on
            // the next iteration's value form.
            let cursor = scope.root::<NanBoxTag>(bindings_root.get());
            let name_slot = scope.root::<NanBoxTag>(NIL);
            let val_slot = scope.root::<NanBoxTag>(NIL);
            let new_bindings_tail = scope.root::<NanBoxTag>(NIL);

            // We accumulate bindings in reverse to keep the in-progress
            // list rooted in `new_bindings_tail` across allocator calls.
            let mut count: usize = 0;
            loop {
                let cur = cursor.get();
                if !is_cons(cur) { break; }
                let b = car(cur);
                // Capture name + val form into rooted slots BEFORE the
                // GC-firing `expand_walk` so they survive the move.
                name_slot.set(car(b));
                let val_form = car(cdr(b));
                let val_bits = self.expand_walk(val_form);
                val_slot.set(val_bits);
                // Advance cursor (cur may be stale post-GC, so re-fetch).
                let cur2 = cursor.get();
                cursor.set(cdr(cur2));

                // Build the new binding `(name val)` and prepend to tail.
                let nil_root = scope.root::<NanBoxTag>(NIL);
                let pair_inner = alloc_cons(scope, &val_slot, &nil_root);
                let pair = alloc_cons(scope, &name_slot, &pair_inner);
                let new_tail = alloc_cons(scope, &pair, &new_bindings_tail);
                new_bindings_tail.set(new_tail.get());
                count += 1;
            }

            // Reverse the bindings list (we built it in reverse).
            let reverse_acc = scope.root::<NanBoxTag>(NIL);
            let cursor2 = scope.root::<NanBoxTag>(new_bindings_tail.get());
            let head_slot = scope.root::<NanBoxTag>(NIL);
            for _ in 0..count {
                let cur = cursor2.get();
                head_slot.set(car(cur));
                cursor2.set(cdr(cur));
                let new_acc = alloc_cons(scope, &head_slot, &reverse_acc);
                reverse_acc.set(new_acc.get());
            }

            let inner = alloc_cons(scope, &reverse_acc, &body);
            alloc_cons(scope, &let_sym, &inner).get()
        })
    }
}

// ── Quasiquote rewrite ──────────────────────────────────────────────

pub fn quasiquote_rewrite(form: u64, sym: &mut SymbolTable) -> u64 {
    let qq_id = sym.intern("quasiquote");
    let uq_id = sym.intern("unquote");
    let uqs_id = sym.intern("unquote-splicing");
    let quote_id = sym.intern("quote");
    let cons_id = sym.intern("cons");
    let append_id = sym.intern("append");
    rewrite_walk(form, sym, qq_id, uq_id, uqs_id, quote_id, cons_id, append_id)
}

fn rewrite_walk(
    form: u64, sym: &mut SymbolTable,
    qq_id: u32, uq_id: u32, uqs_id: u32,
    quote_id: u32, cons_id: u32, append_id: u32,
) -> u64 {
    if !is_cons(form) {
        return form;
    }
    let head = car(form);
    if is_symbol(head) {
        let h = as_symbol_id(head);
        if h == qq_id {
            // `expr — rewrite expr at quasiquote depth 1.
            let inner = car(cdr(form));
            return rewrite_qq(inner, 1, sym, qq_id, uq_id, uqs_id, quote_id, cons_id, append_id);
        }
    }
    // Walk children
    let mut out: Vec<u64> = Vec::new();
    let mut p = form;
    while is_cons(p) {
        out.push(rewrite_walk(car(p), sym, qq_id, uq_id, uqs_id, quote_id, cons_id, append_id));
        p = cdr(p);
    }
    let tail = if is_nil(p) {
        NIL
    } else {
        rewrite_walk(p, sym, qq_id, uq_id, uqs_id, quote_id, cons_id, append_id)
    };
    let mut result = tail;
    for x in out.into_iter().rev() {
        result = cons_compile_time(x, result);
    }
    result
}

/// Rewrite a quasiquote template at depth `depth` (1 = top-level quasiquote).
fn rewrite_qq(
    form: u64, depth: u32, sym: &mut SymbolTable,
    qq_id: u32, uq_id: u32, uqs_id: u32,
    quote_id: u32, cons_id: u32, append_id: u32,
) -> u64 {
    // Atoms: `(quote atom)
    if !is_cons(form) {
        return cons_compile_time(encode_sym(quote_id), cons_compile_time(form, NIL));
    }
    let head = car(form);
    if is_symbol(head) {
        let h = as_symbol_id(head);
        if h == uq_id && depth == 1 {
            return car(cdr(form));
        }
        if h == uq_id {
            let inner = car(cdr(form));
            let r = rewrite_qq(inner, depth - 1, sym, qq_id, uq_id, uqs_id, quote_id, cons_id, append_id);
            return list_call(sym, &[
                cons_compile_time(encode_sym(quote_id), cons_compile_time(encode_sym(uq_id), NIL)),
                r,
            ]);
        }
        if h == qq_id {
            let inner = car(cdr(form));
            let r = rewrite_qq(inner, depth + 1, sym, qq_id, uq_id, uqs_id, quote_id, cons_id, append_id);
            return list_call(sym, &[
                cons_compile_time(encode_sym(quote_id), cons_compile_time(encode_sym(qq_id), NIL)),
                r,
            ]);
        }
    }
    rewrite_qq_list(form, depth, sym, qq_id, uq_id, uqs_id, quote_id, cons_id, append_id)
}

fn rewrite_qq_list(
    form: u64, depth: u32, sym: &mut SymbolTable,
    qq_id: u32, uq_id: u32, uqs_id: u32,
    quote_id: u32, cons_id: u32, append_id: u32,
) -> u64 {
    let mut items: Vec<u64> = Vec::new();
    let mut p = form;
    while is_cons(p) {
        items.push(car(p));
        p = cdr(p);
    }
    let raw_tail = p;

    let mut result: u64 = if is_nil(raw_tail) {
        cons_compile_time(encode_sym(quote_id), cons_compile_time(NIL, NIL))
    } else {
        rewrite_qq(raw_tail, depth, sym, qq_id, uq_id, uqs_id, quote_id, cons_id, append_id)
    };

    for elem in items.into_iter().rev() {
        if is_cons(elem) {
            let h = car(elem);
            if is_symbol(h) {
                let hid = as_symbol_id(h);
                if hid == uqs_id && depth == 1 {
                    let inner = car(cdr(elem));
                    result = cons_compile_time(
                        encode_sym(append_id),
                        cons_compile_time(inner, cons_compile_time(result, NIL)),
                    );
                    continue;
                }
            }
        }
        let r = rewrite_qq(elem, depth, sym, qq_id, uq_id, uqs_id, quote_id, cons_id, append_id);
        result = cons_compile_time(
            encode_sym(cons_id),
            cons_compile_time(r, cons_compile_time(result, NIL)),
        );
    }
    result
}

fn list_call(sym: &mut SymbolTable, items: &[u64]) -> u64 {
    let list_id = sym.intern("list");
    let mut tail = NIL;
    for &x in items.iter().rev() {
        tail = cons_compile_time(x, tail);
    }
    cons_compile_time(encode_sym(list_id), tail)
}
