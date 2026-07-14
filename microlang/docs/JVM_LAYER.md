# The JVM layer, written in the language itself (BUILT)

**Status: shipped.** `clojure-stub/src/clj/host_jvm.clj` is the layer;
`interop_rewrite`/`global_ref` lower syntax only; the shim tables
(`shim_call`/`shim_instance`/`shim_field`), `class_to_tag`'s substring
heuristics, the `class? = false` stub, the `MapEntry.`/`PersistentQueue`
special cases, and `Class/forName`-as-interning are all deleted. Gate: run
70/70 (4 new `jvm_layer_*` tests incl. userland `defclass`), jit 13/13,
coresuite 620/620 both tiers, medley 287/288, core.match end-to-end.

Deviations from the sketch below, from building it:

* **Descriptors are `JvmClass` records, not maps, and the registry/static
  tables are flat `(k v …)` plists** — the layer loads at every prelude boot,
  and in-language hash-map machinery cost ~8ms per class in a debug tree-walk
  (profiled); prim-style eager code brought the whole layer to ~70ms debug /
  ~15ms release. Same lesson as core's own bootstrap macros, now written down.
* **Dot-method dispatch keys are un-namespaced** (`.charAt`, raw): JVM method
  names aren't namespaced, and it frees registrations to come from any ns.
  `%dispatch` / `-proto-method` special-case names starting with `.`.
* **Interfaces may carry `:pred`** (`(defclass clojure.lang.IPersistentMap
  (:kind :interface) (:tag PersistentArrayMap) (:pred map?))`): instance? by
  predicate covers all backing types (PAM + PHM + legacy), while `:tag` remains
  for `extend-type`-on-interface method registration (the compile-time consumer
  reads it from the registry via `jvm_registry_tag` — Rust walking in-language
  data).
* **The throwable ROOTS (Throwable/Exception/Error/Object) stay compile-time
  catch-alls** in the expander: thrown values include strings and plain
  records, so the roots must catch everything — dialect semantics, not
  registry data. Specific classes go through `-jvm-catch-match?` and get real
  superclass/interface walking (`(catch RuntimeException e)` now catches an
  `IllegalArgumentException.` — an upgrade from tag equality).
* Extra registrations discovered by the suites: `js.Error`/`js.RegExp` &
  friends (cljs-targeted `.cljc` branches), `cljs.core/PersistentQueue.EMPTY`
  (the `ns/Class.MEMBER` value shape got general lowering support), and
  `cljs.core.MapEntry`/`cljs.core.PersistentQueue` as default imports.

## The idea

Today "the JVM" lives in Rust as scattered policy: `shim_instance`/`shim_call`
tables in `lib.rs` (`.replace` → clojure.string/replace, `RT/cons` → `%cons`),
the `class_to_tag` substring heuristics, `class?` hardcoded to `false`,
`Class/forName` = symbol interning, `catch_test`'s hardcoded base-class list,
and three-way candidate guessing in `instance_rewrite`. Each was added to make
one library load. They work, but the knowledge is duplicated, compile-time
only, and not extensible from the language.

The principled shape: **the Rust expander lowers interop *syntax* mechanically
and knows zero class names; all JVM semantics live in one in-language
namespace (`host.jvm`) as data + fns.** Exactly the move the collection types
already made — the toolkit stays policy-free, the frontend's Clojure code IS
the platform model. Adding `java.util.UUID` or `StringBuilder` becomes a
library edit, not a Rust edit.

## The two axes (this is the key design point)

JVM interop has a **fast axis** (known method name at a call site:
`(.charAt s 1)`) and a **reflective axis** (names as runtime values:
`Class/forName`, `(class x)`, `instance?`, statics). They want different
mechanisms, and we already have both:

1. **Instance methods = protocol dispatch.** A `.method` call site knows its
   method name at compile time, which is precisely what `Ir::Dispatch`'s
   inline caches key on. So `(.charAt s 1)` lowers to a dispatch on the
   munged method name `.charAt` — same machinery, same inline caches, as fast
   as any protocol call. Registering a class's methods just emits
   `-proto-method` entries.

2. **Reflection = a class registry.** One atom map in `host.jvm`, FQN symbol →
   descriptor. `Class` objects are `(record 'Class fqn)` — finally a real,
   per-instance-honest `class?`. The registry is the single source of truth
   that today's five Rust call sites each approximate.

```clojure
;; host/jvm.clj — descriptor shape (a plain map in the registry)
{:name       'java.lang.String
 :kind       :class                ; :class | :interface | :static  (pure-static holder)
 :tag        'String               ; runtime tag its instances carry (type-of)
 :protocol   nil                   ; for :interface — satisfaction = satisfies?
 :extends    'java.lang.Object
 :implements '[java.lang.CharSequence java.lang.Comparable]
 :statics    {'CASE_INSENSITIVE_ORDER …}         ; name -> value (or atom for mutable)
 :static-fns {'valueOf (fn [x] (str x))}         ; name -> fn
 :ctor       (fn ([] "") ([x] (str x)))}
```

## Defining classes and statics, in the language

```clojure
(ns host.jvm)

(def -registry (%atom-new {}))          ; FQN -> descriptor
(def -tag->class (%atom-new {}))        ; runtime tag -> FQN (for `class`/`type-of` reverse lookup)

(defclass java.lang.Math
  (:kind :static)
  (:static PI 3.141592653589793)
  (:static E  2.718281828459045)
  (:static-fn abs   [x] (if (neg? x) (- x) x))
  (:static-fn floor [x] (%to-long ...))
  (:static-fn max   ([a b] (if (< a b) b a)) ([a b & more] ...)))

(defclass java.lang.String
  (:tag String)
  (:extends java.lang.Object)
  (:implements java.lang.CharSequence)
  (:ctor ([] "") ([x] (str x)))
  ;; each :method emits a -proto-method on tag String under the munged
  ;; name .length/.charAt/… — instance calls are ordinary inline-cached dispatch
  (:method length    [s] (count s))
  (:method charAt    [s i] (nth s i))
  (:method substring ([s b] (subs s b)) ([s b e] (subs s b e)))
  (:method replace   [s t r] (clojure.string/replace s t r))
  (:method indexOf   [s x] ...)
  (:static-fn valueOf [x] (str x)))

;; JVM interfaces map to OUR protocols: instance? = satisfies?.
(definterface- clojure.lang.ILookup           (:protocol ILookup))
(definterface- clojure.lang.IPersistentVector (:protocol IVector))
(definterface- clojure.lang.ISeq              (:protocol ISeq))

;; Pure marker interfaces ride the existing -register-marker registry.
(defclass clojure.lang.RT
  (:kind :static)
  (:static-fn cons  [x s] (%cons x s))
  (:static-fn first [s] (-rt-first s))
  (:static-fn seq   [s] (-rt-seq s)))
```

`defclass` is an ordinary macro in `host.jvm`. It (a) assocs the descriptor
into `-registry` and `-tag->class`, (b) emits `-proto-method` forms for each
`:method`, (c) nothing else. Statics are just data/fns in the descriptor —
a mutable static (rare) is an atom in `:statics`.

## What `defclass` expands to (no new primitives)

`defclass` is an ordinary in-language macro, and everything it expands to
**already exists**. It is not a primitive and it is not `deftype` — it sits
beside `deftype` on the same dispatch substrate:

* `deftype` = *create* a new concrete type (fresh record tag + constructor +
  field registry).
* `defclass` = *describe* a host type by attaching behavior to a type that
  already exists (`String` instances are just our strings; `Math` has no
  instances at all) — plus a registry entry for the reflective axis.

The full expansion of the String example:

```clojure
(defclass java.lang.String
  (:tag String)
  (:extends java.lang.Object)
  (:implements java.lang.CharSequence)
  (:ctor ([] "") ([x] (str x)))
  (:method charAt [s i] (nth s i))
  (:method substring ([s b] (subs s b)) ([s b e] (subs s b e)))
  (:static-fn valueOf [x] (str x))
  (:static CASE_INSENSITIVE_ORDER -ci-comparator))

;; ⇓ macroexpands to ⇓

(do
  ;; 1. The reflective entry: plain data into the registry atoms, via an
  ;;    ordinary defn (-jvm-register! also indexes :tag -> FQN and registers
  ;;    interface markers through the existing -register-marker registry).
  (host.jvm/-jvm-register!
    'java.lang.String
    {:kind       :class
     :tag        'String
     :extends    'java.lang.Object
     :implements '[java.lang.CharSequence]
     :ctor       (fn ([] "") ([x] (str x)))
     :statics    {'CASE_INSENSITIVE_ORDER -ci-comparator}
     :static-fns {'valueOf (fn [x] (str x))}})

  ;; 2. One -proto-method per :method — the SAME internal form that
  ;;    extend-type and deftype emit today (compiles to Ir::DefMethod).
  ;;    The dispatch key is the DOT-MUNGED name, indexed by the :tag.
  (-proto-method .charAt   String (fn [s i] (nth s i)))
  (-proto-method .substring String (fn ([s b] (subs s b)) ([s b e] (subs s b e)))))
```

That's the whole trick: **instance methods are protocol-method entries whose
names happen to start with a dot.** One dispatch key (`.charAt`) with per-type
impls is exactly JVM receiver dispatch; two classes registering `.length` on
their own tags is precisely how our protocols already handle it.

### The call-site pipeline

```clojure
(.charAt s 1)
;; interop_rewrite (Rust, name-blind): head starts with `.` →
(%dispatch host.jvm/.charAt s 1)
;; compile: %dispatch (added for first-class protocol methods) resolves the
;; qualified sym and emits Ir::Dispatch{site, method} — a fresh site id, so
;; the call gets its OWN inline cache, same speed as any protocol call.
```

Details that make this work:

* The method sym is emitted **fully qualified** (`host.jvm/.charAt`) because
  `-proto-method` def-names methods into the namespace that's current when it
  compiles (host.jvm). Qualifying at the lowering site makes user-ns calls
  resolve to the same key with zero refer plumbing.
* `%dispatch` doesn't consult the compiler's methods set, so there is no
  registration-order dependence; an unregistered method is a **dispatch miss
  at runtime — already a catchable error** naming the method and type (the
  JIT-tier catchability was fixed in the core.match work).
* Universal defaults register on `Object` (the dispatch registry's existing
  fallback), e.g. a `.toString` default of `pr-str`. Note the limitation:
  method lookup is exact-tag-then-Object — the `:extends` chain is walked by
  `instance?`/`catch`, **not** by method dispatch, so an intermediate
  superclass method must be registered per concrete tag (a `defclass` option
  can copy parent methods down at registration time if we ever need it).

### Statics and constructors

Reflective path, always available:

```clojure
(Math/abs x)   ⇒ (host.jvm/invoke-static 'java.lang.Math 'abs x)   ; registry lookup + call
Math/PI        ⇒ (host.jvm/static-field  'java.lang.Math 'PI)
(String. x)    ⇒ (host.jvm/construct     'java.lang.String x)      ; :ctor from registry
```

These are ordinary defns over the registry — a map lookup per call, fine for
statics. If a static ever shows up hot, the optimization is local:
`-jvm-register!` can additionally `def` a munged var per static fn
(`java$lang$Math$abs`) and the lowering can target it — but that's a tuning
knob, not part of the design.

### When a host class has no native representation

`String`/`PersistentVector` map onto existing tags. A stateful host class with
no analog (`StringBuilder`, `java.util.Random`) is where `deftype` re-enters:
`defclass` grows a `(:state [fields…])` clause and expands its ctor to an
internal `deftype` — the descriptor's `:tag` is that deftype's tag, and
`:methods` register against it as usual. So the layering is: **defclass
delegates to deftype when it needs storage, and to nothing when it doesn't.**

### Inventory: existing vs new

| piece                              | status |
| ---------------------------------- | ------ |
| `-proto-method` / `Ir::DefMethod` + inline-cached dispatch | exists (extend-type/deftype use it) |
| `%dispatch` special form           | exists (added for first-class protocol methods) |
| `Object` dispatch fallback         | exists |
| `-register-marker` (marker interfaces) | exists (added for core.match) |
| atoms, maps, records for the registry | exist |
| `'Class` wrapper record + `java.lang.Class` methods (`.getName` via its own `defclass`!) | new, in-language |
| `defclass` macro + `host.jvm` defns (`-jvm-register!`, `invoke-static`, `construct`, `class-named`, `for-name`, `instance?`, `assignable-from?`) | new, in-language |
| generic interop lowering + per-ns import table | new, in Rust (mechanical; replaces the shim tables) |

## What the Rust expander shrinks to

All the per-name tables in `interop_rewrite` delete. Lowering becomes purely
syntax-directed:

| surface form            | lowers to                                              |
| ----------------------- | ------------------------------------------------------ |
| `(.m obj a…)`           | dispatch site on method `.m` (miss → registry fallback → catchable "no such method") |
| `(.-f obj)`             | `%field-by-name` (unchanged — deftype fields), fallback `(host.jvm/field obj 'f)` |
| `(C/m a…)`              | `(host.jvm/invoke-static 'FQN 'm a…)`                   |
| `C/FIELD` (value pos)   | `(host.jvm/static-field 'FQN 'FIELD)`                   |
| `(C. a…)` / `(new C a…)`| `(host.jvm/construct 'FQN a…)`                          |
| `(instance? C x)`       | `(host.jvm/instance? (host.jvm/class-named 'FQN) x)`    |
| `C` (value pos, dotted) | `(host.jvm/class-named 'FQN)` → a `'Class` record       |
| `(catch C e …)`         | test `(host.jvm/assignable-from? 'FQN (type-of e))`     |

FQN resolution needs one new piece of compiler state: a per-ns **import
table** (`(:import (java.util UUID))` is currently ignored at `lib.rs:1842`),
seeded with `java.lang.*` like real Clojure, so `Math/abs` and bare `String`
resolve to FQNs at expansion time.

## What falls out for free

* **`Class/forName` becomes honest**: `(or (get @-registry (symbol n))
  (throw (ex-info "ClassNotFoundException" {:name n})))` — a catchable error
  instead of interning a symbol that silently matches nothing.
* **`class?` becomes true** for exactly the `'Class` records; `(class x)` =
  reverse lookup of `(type-of x)` (deftypes auto-register a minimal
  descriptor, so user types reflect too). `.getName`/`.getSimpleName` are just
  `:methods` on `java.lang.Class` itself.
* **Real inheritance for `catch` and `instance?`**: `assignable-from?` walks
  `:extends`/`:implements`, replacing `catch_test`'s hardcoded
  Throwable/Exception/Error list — `(catch Exception e …)` catches a
  `RuntimeException.` because the registry says so.
* **core.match's whole shim chain** (`class?`, `Class/forName`, tag-vs-var
  candidate guessing) collapses into registry lookups with one consistent
  axiom: interfaces → protocols, classes → tags, both behind `'Class` records.

## Bootstrap order

`host.jvm` is dialect code, so it loads after `clojure.core` — but core's own
bootstrap uses a little interop (`(. clojure.lang.RT (cons x s))` in the real
core.clj snippets, `(. (var x) (setMacro))`). Two-stage story:

1. `setMacro` stays intercepted at `eval_form` (it mutates expander state —
   genuinely meta-level).
2. A ~20-line `host.jvm` *seed* (registry atom + `invoke-static` + the RT
   entries) sits early in `clj/core.clj`; the full class library is its own
   namespace loaded after the cljs types (like `clojure.string`). Interop
   lowering emits the same generic calls in both phases — only the registry
   contents grow.

## Consciously out of scope

* Type hints (`^String`) stay advisory/no-op.
* Overload resolution by argument *type* (Java picks `substring(int)` vs
  `substring(int,int)` by static types) — we dispatch by **arity only**, which
  is what the current shims do and what dynamic callers need.
* Actual classloading/bytecode — `:gen-class`, `proxy` are non-goals;
  `reify`/`deftype` already cover the use cases we meet in real libraries.

## Migration checklist (when we build it)

1. Import table in `compile.rs` NsState + `(:import …)` handling.
2. Generic lowering in `interop_rewrite`; delete `shim_instance`/`shim_call`/
   `shim_field`.
3. `host.jvm` seed in clj/core.clj; `host/jvm.clj`-style full library as embedded
   src (Math, String, Object, Exception hierarchy, RT, the ArrayList shims,
   Class itself).
4. Port `class_to_tag` entries into `:tag`/`:protocol` declarations; delete
   the substring heuristics; `instance_rewrite`/`catch_test`/`global_ref`
   consult imports + registry.
5. Gate: full existing suite (66/13/620×2/medley/match) must stay green;
   new tests: forName round-trip + catchable miss, `(class x)`, statics,
   catch-by-superclass.
