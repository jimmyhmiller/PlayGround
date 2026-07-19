# Collection literals are data — full reader↔runtime unification (SHIPPED)

## Status

**Fixed, via full unification (the old doc's "Option C").** There is no longer a
reader-only "code record" for collections. `[a b]`, `{k v}`, and `#{e}` read to
the *real runtime persistent collections* holding the raw element forms —
exactly Clojure's model. Macros receive actual vectors/maps/sets; `quote` keeps
them literal (`Ir::Const`); only *expression position* lowers a collection to a
constructor call that evaluates its elements.

The proof: the real, unmodified 2156-line `clojure/core.match` (vendored at
`clojure-stub/vendor/core.match/`) now loads **and runs correctly on both
tiers** — vector patterns, map patterns, guards, `:or`, rest patterns, seq
patterns, `fib` via match. Answers verified against real Clojure + core.match
1.1.0. medley's complete suite still passes (287/288, the 1 non-pass is the
deliberate cljs-lenient-arity case).

```clojure
(defmacro vq [v] (list 'quote (list (vector? v) (type-of v))))
(vq [1 2])
;; => (true PersistentVector)     ; used to be (false Vector)
```

## The architecture

`clojure-stub/src/data.rs` is the one collection-representation module:

* **Builders** (`make_vector` / `make_map` / `make_set`): construct the
  phase-appropriate runtime collection **in Rust** — the reader runs before any
  in-language constructor exists. Two layouts coexist by bootstrap phase (the
  same seam `binding_items` always knew about):
  * while `clojure.core` loads: core's `PVec` (record: cnt shift root tail, raw
    arrays down the trie) and the list-backed `Map`/`Set` records;
  * once the cljs types exist (detected by `clojure.core/-EMPTY-PV` being
    defined): `PersistentVector` (VectorNode trie, built leaf-up, >32 elements
    fine), `PersistentArrayMap`, and `PersistentHashSet` (PAM-backed; promotes
    itself to the HAMT on write past the 8-entry threshold).
* **Readers** (`vector_items` / `map_entries` / `set_items` / `is_map_rep`):
  representation-blind structural access for the expander — they accept every
  layout, including walking a `PersistentHashMap` HAMT from macro output.

What changed around it:

* **reader.rs** — `[ ]`/`{ }`/`#{ }` and `#()` param vectors build runtime
  collections; `scan_pct`/`rewrite_bare_pct` walk `Obj::Vector` trie leaves;
  `KEYWORD` remains the one reader tag (keywords were always unified).
* **lib.rs `expand`** — the `quote` branch returns the form untouched
  (`build_quote`/`datum_has_coll` deleted); expression-position lowering covers
  every representation of vectors *and maps and sets*; all
  `record_field0(_, VECTOR/MAP)` sites route through `data::*`; syntax-quote
  gained a set-template branch (`(apply hash-set (-concat …))`).
* **Macro invocation** — untouched. Arguments are data by construction now.
* The legacy `'Vector` record survives ONLY as a print shim (`:arglists`
  display, core's `-realize`).

## What core.match forced us to build (all shipped)

Running real macro code exposed real platform gaps, each fixed on principle:

1. **Lazy expansion spines.** Real macros build code with `map`/`concat` — lazy
   seqs the compiler's cons walk can't traverse. `expand` now realizes a lazy
   spine (or a cons whose *tail* hits a lazy node) through core's
   `-force-spine` before walking. Elements are forced on recursion.
2. **First-class protocol methods.** `(reduce prepend …)` needs methods to be
   vars, as in Clojure. `defprotocol` now also defs each method as a wrapper fn
   over a new `%dispatch` special form (an unconditional dispatch site), one
   clause per declared arity.
3. **Marker interfaces.** `definterface IPseudoPattern` has no methods, so
   satisfaction can't be inferred from impls. `deftype`/`extend-type` emit
   `(-register-marker P 'T)` for each protocol group symbol; `satisfies?`
   consults the registry for method-less protocols.
4. **Symbol metadata.** core.match stores occurrence bind-exprs in symbol meta
   (`:ocr-expr`). Symbols are immediates, so `with-meta` on a symbol records
   into a side registry (`-symbol-meta-reg`, cljs_types). Per-NAME rather than
   per-instance — equivalent whenever meta'd symbols are gensym-unique, which
   is how macros use it.
5. **Class references in value position.** `clojure.lang.IPersistentVector` as
   a dispatch value: the dialect's class values ARE runtime tag symbols
   (`deftype T` binds `T` to `'T`), so `compile::global_ref` resolves a bare
   dotted capitalized symbol to the var `a.b/C` if defined, else to the mapped
   tag symbol. `instance?` also tries the SIMPLE name as a var, so
   `(instance? clojure.lang.ILookup x)` means `satisfies?` of our `ILookup`.
6. **Interop shims**: `Class/forName` (→ symbol interning), `.replace`
   (→ `clojure.string/replace`, literal).
7. **JIT tail-call callable objects.** A record tail-called as a function
   (match's `PatternRow`) panicked on the native tier: the trampoline's general
   tail-call path now routes through the apply-handler hook exactly like
   `invoke` (src/jit_cranelift.rs).

## Phase notes / consciously accepted edges

* A literal map/set of ANY size reads as a PersistentArrayMap (linear lookup,
  still correct); the first write past 8 entries promotes to the HAMT.
  Duplicate literal keys are not detected (Clojure throws at read time).
* A quoted collection read at *core-load* time is a `PVec`/list-backed record;
  if such a value escapes to user time, user-time `vector?` (strict
  `PersistentVector` check) won't claim it. `-realize` handles both for
  printing. In practice syntax-quote templates rebuild fresh collections at
  expansion time, so this is a corner.
* `-pv-seq` naming: core's PVec seq helper and the cljs port's PersistentVector
  seq helper used to COLLIDE (load-order-dependently load-bearing!); the cljs
  one is now `-PV-seq`.

## Test gate (all green)

* `cargo test -p clojure-stub --test run` — 66 tests (incl.
  `collection_literals_are_data`, `real_core_match_end_to_end`)
* `cargo test -p clojure-stub --features jit --test jit` — 13 tests (incl.
  `real_core_match_on_jit`)
* `examples/coresuite` — 620/620 vs the real-Clojure oracle, both tiers
* medley full suite — 287/288 (unchanged; 1 deliberate arity-leniency diff)

Known pre-existing failure (NOT from this work): scheme R7RS conformance
`(null? (list))` → `#f` fails at HEAD.
