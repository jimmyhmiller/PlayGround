//! Quasiquote / unquote / unquote-splicing rewriter.
//!
//! Expands `` `x `` into a tree of `cons`/`__concat`/`quote` calls
//! that, when evaluated, reconstructs the form `x` with each
//! `~y` evaluated in place and each `~@y` spliced in.
//!
//! Nesting: each new `quasiquote` increases the level by 1, each
//! `unquote` / `unquote-splicing` decreases it. Unquotes only fire
//! at level 1.
//!
//! Limitations vs. real Clojure:
//!   - No automatic namespace qualification of bare symbols.
//!   - No auto-gensym (`x#`).
//!
//! Both can be added later when the var registry / gensym story
//! firms up. For bootstrap-quality macros they aren't strictly
//! required.
//!
//! ## Root discipline
//!
//! Every intermediate result is held in a `Rooted<NanBoxTag>` inside
//! a single outer `RootScope`. The rewrite produces a tree of cons
//! cells via repeated heap allocation; if intermediates were
//! returned as raw `u64`, a moving GC fired during a later allocation
//! would invalidate them.

use dynobj::roots::{RootScope, Rooted};

use crate::collections::is_list;
use crate::symbols::SymbolTable;
use crate::value::{self as v, NanBoxTag};

pub struct QqCtx<'a> {
    pub sym: &'a SymbolTable,
}

impl<'a> QqCtx<'a> {
    /// Top-level entry. Opens a fresh `RootScope` of generous size
    /// and threads it through the expansion. The result is read out
    /// of the scope as a raw bit pattern at the very end.
    pub fn expand_top(&mut self, form: u64) -> u64 {
        // 1024 slots is plenty for any realistic macro body (each
        // form contributes a small constant number of intermediates
        // proportional to its depth).
        dynobj::roots::with_scope(1024, |scope| {
            let f = scope.root::<NanBoxTag>(form);
            self.expand(scope, &f, 1).get()
        })
    }

    fn expand<'s>(
        &mut self,
        scope: &'s RootScope<'_>,
        form: &Rooted<'_, NanBoxTag>,
        level: usize,
    ) -> Rooted<'s, NanBoxTag> {
        let bits = form.get();

        // Atoms (and most non-list heap shapes) become `(quote form)`.
        if !v::is_ptr(bits) || !is_list(bits) {
            // Refuse `foo#` auto-gensym tokens inside a quasiquote
            // until they're implemented. Silently passing them through
            // breaks hygiene: every macro using `(let [x# ...] ...)`
            // would expand to a literal `x` and capture at the call
            // site.
            if v::is_sym_id(bits) {
                let id = v::as_sym_id(bits);
                let name = self.sym.name(id).to_string();
                if name.ends_with('#') && name.len() > 1 {
                    unimplemented!(
                        "quasiquote: auto-gensym `{name}` not yet \
                         supported (see TODO.md). Refusing to silently \
                         break macro hygiene."
                    );
                }
            }
            return self.quote(scope, bits);
        }
        // It's a list. First check for unquote / quasiquote heads.
        let head = v::first(bits);
        if v::is_sym_id(head) {
            let name = self.sym.name(v::as_sym_id(head)).to_string();
            match name.as_str() {
                "unquote" => {
                    let inner = v::first(v::rest(bits));
                    if level == 1 {
                        return scope.root::<NanBoxTag>(inner);
                    }
                    let inner_r = scope.root::<NanBoxTag>(inner);
                    let expanded_inner = self.expand(scope, &inner_r, level - 1);
                    let head_sym = self.intern_sym("unquote");
                    return self.list2_bits(scope, head_sym, expanded_inner.get());
                }
                "unquote-splicing" => {
                    if level == 1 {
                        panic!("unquote-splicing not in list context");
                    }
                    let inner = v::first(v::rest(bits));
                    let inner_r = scope.root::<NanBoxTag>(inner);
                    let expanded_inner = self.expand(scope, &inner_r, level - 1);
                    let head_sym = self.intern_sym("unquote-splicing");
                    return self.list2_bits(scope, head_sym, expanded_inner.get());
                }
                "quasiquote" => {
                    let inner = v::first(v::rest(bits));
                    let inner_r = scope.root::<NanBoxTag>(inner);
                    let expanded_inner = self.expand(scope, &inner_r, level + 1);
                    let head_sym = self.intern_sym("quasiquote");
                    return self.list2_bits(scope, head_sym, expanded_inner.get());
                }
                _ => {}
            }
        }
        // Generic list.
        self.expand_list_elements(scope, form, level)
    }

    fn expand_list_elements<'s>(
        &mut self,
        scope: &'s RootScope<'_>,
        form: &Rooted<'_, NanBoxTag>,
        level: usize,
    ) -> Rooted<'s, NanBoxTag> {
        // Snapshot the elements into a Vec — we hold each as a
        // Rooted slot in `scope` so subsequent allocs can't move
        // them out from under us.
        let elem_bits: Vec<u64> = v::list_iter(form.get()).collect();
        let elems: Vec<Rooted<'s, NanBoxTag>> = elem_bits
            .into_iter()
            .map(|b| scope.root::<NanBoxTag>(b))
            .collect();

        // Build right-to-left.
        let mut acc = self.quote(scope, v::NIL);
        for e in elems.iter().rev() {
            let piece: Rooted<'s, NanBoxTag> =
                if let Some(inner_bits) = self.match_splice(e.get(), level) {
                    // ~@inner: contribute inner directly. Must be a list at runtime.
                    scope.root::<NanBoxTag>(inner_bits)
                } else {
                    let expanded = self.expand(scope, e, level);
                    let nil_quote = self.quote(scope, v::NIL);
                    self.cons_call(scope, expanded.get(), nil_quote.get())
                };
            acc = self.concat_call(scope, piece.get(), acc.get());
        }
        acc
    }

    /// If `form` is `(unquote-splicing inner)` at the current level,
    /// returns the inner form's bits. Otherwise None.
    fn match_splice(&mut self, form: u64, level: usize) -> Option<u64> {
        if !v::is_ptr(form) || !is_list(form) {
            return None;
        }
        let head = v::first(form);
        if !v::is_sym_id(head) {
            return None;
        }
        let name = self.sym.name(v::as_sym_id(head)).to_string();
        if name == "unquote-splicing" && level == 1 {
            return Some(v::first(v::rest(form)));
        }
        None
    }

    // ── Constructors ────────────────────────────────────────────────

    fn intern_sym(&mut self, name: &str) -> u64 {
        let id = self.sym.intern(name);
        v::encode_sym_id(id)
    }

    fn quote<'s>(&mut self, scope: &'s RootScope<'_>, datum: u64) -> Rooted<'s, NanBoxTag> {
        let q = self.intern_sym("quote");
        self.list2_bits(scope, q, datum)
    }

    fn concat_call<'s>(
        &mut self,
        scope: &'s RootScope<'_>,
        a: u64,
        b: u64,
    ) -> Rooted<'s, NanBoxTag> {
        let head = self.intern_sym("__concat");
        self.list3_bits(scope, head, a, b)
    }

    fn cons_call<'s>(&mut self, scope: &'s RootScope<'_>, a: u64, b: u64) -> Rooted<'s, NanBoxTag> {
        let head = self.intern_sym("cons");
        self.list3_bits(scope, head, a, b)
    }

    /// `(a b)` rooted in `scope`. Each cell allocation roots its
    /// inputs so the in-flight head/tail survive a GC fired by
    /// the next allocation.
    fn list2_bits<'s>(
        &mut self,
        scope: &'s RootScope<'_>,
        a: u64,
        b: u64,
    ) -> Rooted<'s, NanBoxTag> {
        let a_r = scope.root::<NanBoxTag>(a);
        let b_r = scope.root::<NanBoxTag>(b);
        let tail = v::alloc_list_cell_from_raw(scope, b_r.get(), v::NIL);
        v::alloc_list_cell_from_raw(scope, a_r.get(), tail.get())
    }

    fn list3_bits<'s>(
        &mut self,
        scope: &'s RootScope<'_>,
        a: u64,
        b: u64,
        c: u64,
    ) -> Rooted<'s, NanBoxTag> {
        let a_r = scope.root::<NanBoxTag>(a);
        let b_r = scope.root::<NanBoxTag>(b);
        let c_r = scope.root::<NanBoxTag>(c);
        let tail2 = v::alloc_list_cell_from_raw(scope, c_r.get(), v::NIL);
        let tail1 = v::alloc_list_cell_from_raw(scope, b_r.get(), tail2.get());
        v::alloc_list_cell_from_raw(scope, a_r.get(), tail1.get())
    }
}
