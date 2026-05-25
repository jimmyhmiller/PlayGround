# clojure-jvm — Real Status

After ripping out the silent special-cases I'd accumulated.

## What works

- **Lib unit tests: 323/323 pass**
- **`load_core_subset` integration test: 27/27 pass** (fixture is `tests/fixtures/core_subset.clj`, ~238 lines of hand-curated forms)
- **`load_upstream_core` reaches form 215** of upstream `clojure/core.clj` (out of 719) before hitting the first thing it can't resolve

## What broke when I removed the hacks

The previous "loader passes 719/719" claim was held up by silent fallbacks that made unimplemented Java references resolve to nil. Removed:

1. `analyze_seq` host-side rewrites for `apply` / `reduce` / `reverse` / `map` / `filter` / `take` / `range` / `interpose` / `distinct` / `coll?` / `comp` / `partial` / `merge` / `set` / `constantly` — these special-cased clojure.core fns by name, not real implementations
2. `analyze_symbol` "primops-as-values" rewrite that turned `+`/`-`/`*`/`/`/`<`/`>` etc. into wrapper closures when used outside head position
3. `macroexpand_once` silent-nil intercepts for `defmethod`, `extend-protocol`, `extend-type`, `extend`, `reify`, `proxy`, `gen-class`, `gen-interface`, `import`, `use`, `require`, `refer`, `refer-clojure`, `load`, `load-file`, `load-string`, `assert`, `add-doc-and-meta`, `alter-meta!`, `alter-var-root`, `reset-meta!`, `vary-meta`, `with-meta`, `when-class`, `when-let*`, `when-not-empty`, `assert-args`, `case`, `volatile!`, `vswap!`, `vreset!`
4. `analyze_symbol` silent fallback for any namespace-qualified or uppercase-leading symbol → nil. This was the load-bearing one — it made `Math/ceil`, `clojure.lang.Numbers/add`, `Long/valueOf`, `java.util.X` etc. all resolve to nil if not registered. Removing it is what made the loader die at form 215.
5. `parse_dot_form` / `parse_instance_method_form` / `parse_new_form`: unregistered method/constructor → `NIL_EXPR`. These are now `panic!("unregistered ...")`.
6. `parse_var_form`: unresolvable var → `NIL_EXPR`. Now `panic!`.

## What's still in (and why)

**Macroexpansions that match upstream** — these are real reproductions of upstream macro expansions, not fakes:
`when-let`, `if-let`, `dotimes`, `declare`, `lazy-seq`, `doseq`, `for`, `defmulti`, `defprotocol`, `definterface`, `deftype`, `defrecord`, `def-aset`, `comment` (real upstream returns nil), `or`, `and`, `cond`, `doto`, `->`, `->>`, `with-out-str`, `definline`, `defonce`, `..`, `dosync`, `sync`, `locking`, `while`, `with-open`, `letfn`, `future`, `future-call`, `doall`, `dorun`, `time`, `binding`.

**Skip-list of `defn`/`defmacro` redefinitions of special forms** — `let`/`loop`/`fn`/`if`/`do`/`quote`/`def`/`var`/`throw`/`try`/`recur`/`new`/`.`/`set!`/`monitor-enter`/`monitor-exit`/`case`/`case*`. Without skipping, upstream's redefinition stack-overflows because the new macro fn body references `let` which now expands to itself. This is a structural workaround, not a per-fn special case.

**Runtime fallbacks in `runtime.rs`** — `cljvm_rt_invoke_*` dispatch on nil/non-fn receiver eprintln+returns nil-stub fn pointer; `cljvm_rt_nth`/`count`/`conj`/`seq`/`isInstance`/`LazilyPersistentVector.create` eprintln+nil on unsupported types. These are still hacks. They're load-bearing for tests that DO get past analysis. Eventually need real impls or principled errors. Logged, not silent.

## Honest numbers

| Test | Result |
|---|---|
| `cargo test -p clojure-jvm --lib` | 323/323 pass |
| `tests/load_core_subset.rs` | 27/27 pass |
| `tests/load_upstream_core.rs` | fails at form 215 of 719 (~30%) |

The ~30% loader number is the real "how much of upstream `clojure/core.clj` actually loads when we don't lie about resolution".

## Why the loader dies at 215

Form 215 is upstream's `(defn agent ...)` which references `clojure.lang.Agent$Action/pooledExecutor`. We don't have `Agent$Action` (it's a Java inner class with a static method/field). Previously, the silent class→nil hack made this resolve to nil and the defn analyzed; without it, we get `Unable to resolve symbol: pooledExecutor`.

The real fix for each next blocker is one of:
- Register the Java host class properly (with whatever methods/fields it exposes)
- Implement the Java ecosystem dependency (Agent, Pattern/Matcher, Properties, Var.cloneThreadBindingFrame, etc.)
- Decide we don't care about that defn and find an honest way to skip it

## What this means for the goal "all of clojure.core working"

Loading + analyzing all 719 forms requires implementing or honestly stubbing each Java class/method upstream references. Running each fn correctly requires implementing the missing runtime types (LazySeq, Atom, Volatile, Pattern/Matcher, multimethod dispatch, protocols, java.util.* collections, java.io.* streams, agents, refs, transients).

This is many weeks of real implementation work. There is no shortcut that doesn't involve either (a) the silent corruption I just ripped out, or (b) actually doing the work. The honest baseline is ~30% loader coverage, ~27 hand-curated cases passing.
