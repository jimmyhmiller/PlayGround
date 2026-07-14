# Sketch: a JVM layer written in the language itself

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
   entries) sits early in `core_src`; the full class library is its own
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
3. `host.jvm` seed in core_src; `host/jvm.clj`-style full library as embedded
   src (Math, String, Object, Exception hierarchy, RT, the ArrayList shims,
   Class itself).
4. Port `class_to_tag` entries into `:tag`/`:protocol` declarations; delete
   the substring heuristics; `instance_rewrite`/`catch_test`/`global_ref`
   consult imports + registry.
5. Gate: full existing suite (66/13/620×2/medley/match) must stay green;
   new tests: forName round-trip + catchable miss, `(class x)`, statics,
   catch-by-superclass.
