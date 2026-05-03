//! Macroexpander.
//!
//! `macroexpand_all(form)` rewrites every macro call (head-position symbol
//! that's a key in `macro_env`) by invoking the JIT-compiled macro body and
//! substituting the result, recursively. Quasiquote is also rewritten here:
//! `\`(...)` becomes explicit `cons` / `list` / `append` calls before the
//! compiler sees it.

use std::collections::HashMap;

use dynir::ir::FuncRef;
use dynlang::gc::DynGcRuntime;
use dynlower::{JitModule, JitOutcome};

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

    fn expand_walk(&mut self, mut form: u64) -> u64 {
        // Repeatedly expand the head-position macro at this level.
        for _ in 0..self.max_iters {
            let (next, did) = self.expand_one(form);
            if !did {
                form = next;
                break;
            }
            form = next;
        }
        // Then walk into subforms — but respect special forms.
        self.walk_subforms(form)
    }

    fn expand_one(&mut self, form: u64) -> (u64, bool) {
        if !is_cons(form) { return (form, false); }
        let head = car(form);
        if !is_symbol(head) { return (form, false); }
        let id = as_symbol_id(head);
        let Some(&fref) = self.macro_env.get(&id) else {
            return (form, false);
        };
        // Pass cdr(form) as the args list. Route through
        // `gc.run_jit_with_threshold(1.0)` — safepoints still fire (so the
        // handler sees a valid session, the no-handler-installed assertion
        // can't trip), but the threshold check skips actual collection.
        //
        // Macroexpansion is a *compile-time* activity: the expander holds
        // unrooted cons-tree handles in Rust locals as it walks the form.
        // Letting GC fire here would relocate those cells out from under
        // the walker. Until microlisp gains full Rust-side rooting (a
        // FrameChain threaded through every cons-handling helper), we mute
        // mid-expansion GC. Top-level user-code execution gets the real
        // `gc.run_jit` (threshold=0) where every allocation collects.
        let args_list = cdr(form);
        let result = match self.gc.run_jit_with_threshold(
            self.jit, fref, &[args_list], 1.0,
        ) {
            JitOutcome::Value(v) => v,
            JitOutcome::Void => NIL,
            other => panic!("macro fn returned non-value outcome: {other:?}"),
        };
        (result, true)
    }

    fn walk_subforms(&mut self, form: u64) -> u64 {
        if !is_cons(form) { return form; }
        let head = car(form);
        if is_symbol(head) {
            let name = {
                let sym = self.host.sym.borrow();
                sym.name(as_symbol_id(head)).to_string()
            };
            match name.as_str() {
                "quote" => return form, // do not descend
                "define" => return self.walk_define(form),
                "defmacro" => return self.walk_defmacro(form),
                "let" => return self.walk_let(form),
                "if" | "begin" | "set!" | "list" => {
                    // descend into all elements after head
                    return self.walk_each_after_head(form);
                }
                _ => {}
            }
        }
        self.walk_each(form)
    }

    fn walk_each(&mut self, form: u64) -> u64 {
        // Walk every element of the (proper) list and rebuild.
        if is_nil(form) { return form; }
        if !is_cons(form) { return form; }
        let head = self.expand_walk(car(form));
        let tail = self.walk_each(cdr(form));
        alloc_cons(head, tail)
    }

    fn walk_each_after_head(&mut self, form: u64) -> u64 {
        let head = car(form);
        let tail = self.walk_each(cdr(form));
        alloc_cons(head, tail)
    }

    fn walk_define(&mut self, form: u64) -> u64 {
        // (define <target> body...)  — descend into body only.
        let define_sym = car(form);
        let rest = cdr(form);
        let target = car(rest);
        let body = cdr(rest);
        let body = self.walk_each(body);
        alloc_cons(define_sym, alloc_cons(target, body))
    }

    fn walk_defmacro(&mut self, form: u64) -> u64 {
        // (defmacro name pattern body...)  — descend body only.
        let defmacro_sym = car(form);
        let rest = cdr(form);
        let name_sym = car(rest);
        let after_name = cdr(rest);
        let pattern = car(after_name);
        let body = cdr(after_name);
        let body = self.walk_each(body);
        alloc_cons(
            defmacro_sym,
            alloc_cons(name_sym, alloc_cons(pattern, body)),
        )
    }

    fn walk_let(&mut self, form: u64) -> u64 {
        // (let ((x v) ...) body...) — descend into binding values + body.
        let let_sym = car(form);
        let rest = cdr(form);
        let bindings = car(rest);
        let body = cdr(rest);

        let mut new_bindings: Vec<u64> = Vec::new();
        for b in list_iter(bindings) {
            let name = car(b);
            let val = car(cdr(b));
            let val = self.expand_walk(val);
            new_bindings.push(alloc_cons(name, alloc_cons(val, NIL)));
        }
        let new_body = self.walk_each(body);
        let new_bindings_list = list_from_slice(&new_bindings);
        alloc_cons(let_sym, alloc_cons(new_bindings_list, new_body))
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
        result = alloc_cons(x, result);
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
        return alloc_cons(encode_sym(quote_id), alloc_cons(form, NIL));
    }
    let head = car(form);
    if is_symbol(head) {
        let h = as_symbol_id(head);
        if h == uq_id && depth == 1 {
            return car(cdr(form));
        }
        if h == uq_id {
            // nested unquote at higher depth: keep as `(list 'unquote (qq inner depth-1))`
            let inner = car(cdr(form));
            let r = rewrite_qq(inner, depth - 1, sym, qq_id, uq_id, uqs_id, quote_id, cons_id, append_id);
            return list_call(sym, &[
                alloc_cons(encode_sym(quote_id), alloc_cons(encode_sym(uq_id), NIL)),
                r,
            ]);
        }
        if h == qq_id {
            // nested quasiquote: descend at depth+1 and wrap with `(list 'quasiquote ...)`
            let inner = car(cdr(form));
            let r = rewrite_qq(inner, depth + 1, sym, qq_id, uq_id, uqs_id, quote_id, cons_id, append_id);
            return list_call(sym, &[
                alloc_cons(encode_sym(quote_id), alloc_cons(encode_sym(qq_id), NIL)),
                r,
            ]);
        }
    }
    // List form: walk elements, building (cons (qq a) (cons (qq b) (append c (qq rest))))
    rewrite_qq_list(form, depth, sym, qq_id, uq_id, uqs_id, quote_id, cons_id, append_id)
}

fn rewrite_qq_list(
    form: u64, depth: u32, sym: &mut SymbolTable,
    qq_id: u32, uq_id: u32, uqs_id: u32,
    quote_id: u32, cons_id: u32, append_id: u32,
) -> u64 {
    // Build right-to-left.
    // First, collect elements + the tail.
    let mut items: Vec<u64> = Vec::new();
    let mut p = form;
    while is_cons(p) {
        items.push(car(p));
        p = cdr(p);
    }
    let raw_tail = p; // either nil or some atom (dotted tail)

    // Tail rewrite.
    let mut result: u64 = if is_nil(raw_tail) {
        alloc_cons(encode_sym(quote_id), alloc_cons(NIL, NIL))
    } else {
        rewrite_qq(raw_tail, depth, sym, qq_id, uq_id, uqs_id, quote_id, cons_id, append_id)
    };

    for elem in items.into_iter().rev() {
        // Check if elem is (unquote-splicing expr) at depth 1
        if is_cons(elem) {
            let h = car(elem);
            if is_symbol(h) {
                let hid = as_symbol_id(h);
                if hid == uqs_id && depth == 1 {
                    let inner = car(cdr(elem));
                    // result = (append <inner> result)
                    result = alloc_cons(
                        encode_sym(append_id),
                        alloc_cons(inner, alloc_cons(result, NIL)),
                    );
                    continue;
                }
            }
        }
        // result = (cons (qq elem) result)
        let r = rewrite_qq(elem, depth, sym, qq_id, uq_id, uqs_id, quote_id, cons_id, append_id);
        result = alloc_cons(
            encode_sym(cons_id),
            alloc_cons(r, alloc_cons(result, NIL)),
        );
    }
    result
}

fn list_call(sym: &mut SymbolTable, items: &[u64]) -> u64 {
    let list_id = sym.intern("list");
    let mut tail = NIL;
    for &x in items.iter().rev() {
        tail = alloc_cons(x, tail);
    }
    alloc_cons(encode_sym(list_id), tail)
}
