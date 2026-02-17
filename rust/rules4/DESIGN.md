# Rules4 — A Scoped Term Rewriting Language

## Overview

Rules4 is a programming language where the only primitive operation is the
**rewrite rule**: match a pattern in one scope, produce a result in another.
Functions, data transformations, side effects, and even the debugging/stepping
infrastructure are all expressed as rules. The system is self-observable — rules
about how rewriting happens (meta-rules) use the same syntax and matching
machinery as rules about data.

The language is inspired by:
- **Eve** (Chris Granger) — scopes as named databases, rules as the universal
  construct, `@`-prefixed scope names
- **Term rewriting systems** — pattern matching with logic variables, reduction
  strategies
- **Production rule systems** (RETE/OPS5) — incremental matching across a
  working memory that changes over time

## Core Concepts

### Terms

Everything is a term. Terms are hash-consed (structurally deduplicated) so
equality is O(1) and sharing is free.

```
42                          # number
"hello"                     # string
:keyword                    # keyword (interned symbol)
foo                         # symbol
?x                          # logic variable (binds in patterns, substitutes in templates)
f(x, y)                     # call
[1, 2, 3]                   # array
{name: "Alice", age: 30}    # map
```

### Scopes

A **scope** is a named store of terms. Data lives in scopes. Rules read from
scopes and write to scopes. Scope names start with `@`.

```
@main       # default scope for "normal" computation
@io         # side effects (print, read, etc.)
@meta       # rewrite events (automatically populated by the engine)
@rules      # the rule set itself (rules are data too)
@history    # optional — accumulate rewrite trace
```

Users can define arbitrary scopes. A scope is just a name.

### Rules

A **rule** is a named set of clauses that transforms terms. Each clause has a
left-hand side (pattern to match) and a right-hand side (template to produce).

```
rule fact : @main -> @main {
  fact(0) => 1
  fact(?n) => ?n * fact(?n - 1)
}
```

Anatomy:
- `rule fact` — the rule's name
- `: @main -> @main` — reads from `@main`, writes to `@main`
- `fact(0) => 1` — a clause: pattern `=>` template
- `?n` — logic variable, binds to whatever it matches, substituted on the right

A rule can read from multiple scopes (comma-separated on the left of `->`):

```
rule log-errors : @main, @config -> @errors {
  {status: :error, msg: ?m}, {log_level: :verbose}
    => {error: ?m, timestamp: now()}
}
```

Both input patterns must match simultaneously for the clause to fire.

### Evaluation

The engine repeatedly:
1. Finds a reducible sub-expression in a scope
2. Finds rules whose patterns match it
3. Applies the rewrite (substituting logic variables)
4. Emits the result into the output scope
5. Emits a meta-event describing what happened into `@meta`

A term is **exhausted** when no rules match it or any of its sub-expressions.

## Syntax

### Logic Variables

Prefixed with `?`. Bind on the left of `=>`, substitute on the right.

```
?x          # bind/substitute a single term
?items...   # bind/substitute zero or more terms (spread)
```

Repeated variables require structural equality:

```
rule drop-dup : @main -> @main {
  pair(?x, ?x) => ?x               # only matches when both are equal
}
```

### Guards

`when` adds conditions to a clause:

```
rule validate : @input -> @errors {
  {age: ?a} when ?a < 0
    => {field: :age, msg: "must be non-negative"}

  {email: ?e} when not(contains(?e, "@"))
    => {field: :email, msg: "invalid email"}
}
```

### Keywords

Keywords start with `:` and are interned symbols (like Clojure keywords).
They are atoms — they evaluate to themselves.

```
:ok
:error
:user
```

### Maps

Maps use `{key: value}` syntax. In patterns, maps match structurally — the
pattern only needs to mention the keys it cares about (open matching):

```
# Matches any map that has a "name" key, ignores other keys
{name: ?n} => greet(?n)
```

### Arrays

Arrays use `[a, b, c]` syntax. Spread patterns for matching heads/tails:

```
[?first, ?rest...] => process(?first, [?rest...])
[]                  => :done
```

### Calls

Calls use `f(a, b)` syntax. Infix operators are sugar for calls:

```
?a + ?b        # sugar for add(?a, ?b)
?a * ?b        # sugar for mul(?a, ?b)
?a ++ ?b       # sugar for concat(?a, ?b)
?a == ?b       # sugar for eq(?a, ?b)
```

### Quote

`quote(...)` prevents evaluation of its contents. Useful for treating code as
data (e.g., when adding rules dynamically):

```
quote(fact(?n))     # the term fact(?n) as data, not a call to reduce
```

### Do blocks

`do { ... }` sequences multiple expressions. Each is evaluated in order.
The block's value is the last expression.

```
do {
  let ?x = compute()
  println("got: ", ?x)
  ?x
}
```

## Syntactic Sugar

### Functions

`fn` is sugar for a rule in `@main -> @main`:

```
fn greet(?name) = "Hello, " ++ ?name

# desugars to:
rule greet : @main -> @main {
  greet(?name) => "Hello, " ++ ?name
}
```

Multi-clause:

```
fn fib(0) = 0
fn fib(1) = 1
fn fib(?n) = fib(?n - 1) + fib(?n - 2)

# desugars to:
rule fib : @main -> @main {
  fib(0) => 0
  fib(1) => 1
  fib(?n) => fib(?n - 1) + fib(?n - 2)
}
```

### Let

`let` binds a name in a local scope:

```
let ?x = 5
let ?y = ?x + 1
```

### If

```
if ?cond then ?a else ?b
```

Sugar for rules:

```
rule if : @main -> @main {
  if(true, ?a, ?b) => ?a
  if(false, ?a, ?b) => ?b
}
```

## Data Transformations

The primary use case for the rule syntax. Transform data between scopes:

```
rule normalize-users : @api -> @app {
  {type: "user", name: ?n, age: ?a, email: ?e}
    => {kind: :user, display_name: ?n, contact: {email: ?e}, meta: {age: ?a}}

  {type: "admin", name: ?n, perms: ?p}
    => {kind: :admin, display_name: ?n, permissions: ?p, meta: {}}
}
```

Recursive/nested transformations:

```
rule flatten : @raw -> @flat {
  {wrapper: ?inner}       => ?inner          # strip wrappers
  [?head, ?tail...]       => [normalize(?head), ?tail...]
}
```

AST transformations (e.g., a compiler pass):

```
rule desugar : @surface -> @core {
  when(?test, ?body)
    => if(?test, ?body, nil)

  cond(?test, ?then, ?rest...)
    => if(?test, ?then, cond(?rest...))

  cond()
    => nil

  let(?name, ?val, ?body)
    => apply(lambda(?name, ?body), ?val)
}
```

## Meta-Rules

When a rewrite happens, the engine emits a record into `@meta`:

```
{
  old: <term before rewrite>,
  new: <term after rewrite>,
  sub_old: <sub-expression that matched>,
  sub_new: <replacement sub-expression>,
  rule: <name of the rule that fired>,
  scope: <scope the rewrite happened in>,
}
```

Meta-rules match on these records using ordinary rule syntax:

### Tracing

```
rule trace : @meta -> @io {
  {old: ?x, new: ?y, scope: @main}
    => println(?x, " => ", ?y)
}
```

### Stepping Debugger

```
rule step : @meta -> @io {
  {sub_old: ?s, sub_new: ?t, old: ?x, new: ?y, scope: @main}
    => do {
      println("  ", ?s, " => ", ?t)
      println("  in: ", ?x)
      println("  now: ", ?y)
      readline()
    }
}
```

### Rewrite History

```
rule record : @meta -> @history {
  {old: ?x, new: ?y, rule: ?r, scope: @main}
    => {from: ?x, to: ?y, applied: ?r}
}
```

### Strategy Control

Meta-rules can control the evaluation strategy by emitting into `@strategy`:

```
# Force innermost evaluation: block a rule if its arguments aren't values yet
rule innermost-fact : @meta -> @strategy {
  {rule: :fact, sub_old: ?x} when not(is_value(?x))
    => :block
}

# More general: block any rule with unreduced children
rule innermost : @meta -> @strategy {
  {sub_old: ?call} when has_unreduced_children(?call)
    => :block
}
```

When `@strategy` receives `:block`, the engine skips that rewrite and moves on.

## Dynamic Rules

Rules are data. They can be created at runtime:

```
rule define : @main -> @rules {
  define(?name, ?from, ?to)
    => {name: ?name, scopes: @main -> @main, clauses: [{?from => ?to}]}
}

# Usage — defines a new rewrite rule at runtime:
define(:double, double(?x), ?x * 2)
```

Since `@rules` is just a scope, meta-rules can observe rule changes too:

```
rule log-new-rules : @meta -> @io {
  {scope: @rules, new: ?r}
    => println("New rule added: ", ?r)
}
```

## Pipelines

For multi-stage transformations, a pipeline declares a chain of scopes and
rule sets:

```
pipeline compile {
  @source -> parse   -> @ast
  @ast    -> desugar -> @core
  @core   -> check   -> @typed
  @typed  -> emit    -> @output
}
```

Each arrow names a rule set. The engine runs each stage to exhaustion before
moving to the next.

## Implementation Architecture

### Hash-Consing (Term Store)

All terms are stored in a global deduplicated store. Structurally identical
terms share a single `TermId`. Building a term:

```
intern(Num(5))                  => TermId(0)
intern(Num(1))                  => TermId(1)
intern(Sym("fact"))             => TermId(2)
intern(Call(2, [0]))            => TermId(3)   // fact(5)
intern(Call(2, [1]))            => TermId(4)   // fact(1)
intern(Call(2, [0]))            => TermId(3)   // same as before — deduplicated
```

Benefits:
- **O(1) equality** — compare TermIds, not trees
- **Zero-cost sharing** — history, meta events, scopes all just store TermIds
- **Cheap rewrite events** — `{old: TermId, new: TermId}` is four integers

### Rule Matching (Term Indexing)

The system is designed for **live coding** — rules change constantly as the
programmer edits code. This rules out approaches like RETE networks, which
optimize for the opposite case (stable rules, changing data). RETE's partial
match caches are expensive to rebuild when rules change, and in an interactive
session rules change on every keystroke.

Instead, we use **term indexing by top-level constructor** — a simple index
that maps the outermost symbol of a pattern to the rules that start with it:

```
index = {
    fact  => [rule 0 clause 0, rule 0 clause 1],
    if    => [rule 3 clause 0, rule 3 clause 1],
    _     => [rule 7 clause 0],   // wildcard (pattern starts with ?x)
}
```

When a term like `fact(3)` needs matching, we only check the `fact` bucket
plus the `_` (wildcard) bucket — not every rule in the system. This is O(bucket
size) instead of O(total rules).

Rebuilding the index when rules change is cheap — just re-bucket. No partial
match state to invalidate.

For map patterns like `{type: "user", ...}`, we can index on the set of keys
present, or on the first key-value pair. The index is a fast pre-filter; full
matching still happens per-candidate.

### Incremental Scope Recomputation

For **pipelines** (`@source -> @ast -> @core`), we want incrementality in the
other direction: when a rule changes, which outputs need recomputation?

Each scope tracks **dependencies** — which input terms contributed to which
output terms. When a rule in the `desugar` pass changes:

1. Mark all outputs in `@core` that were produced by `desugar` rules as stale
2. Re-run only the stale terms through the updated rules
3. Propagate staleness downstream if outputs changed

This is similar to build systems (Salsa, Adapton) — fine-grained dependency
tracking with minimal recomputation. The hash-consed term store helps here:
if a re-run produces the same TermId as before, downstream is unaffected.

### Scopes

Each scope is a store of terms (TermIds). The engine maintains:
- A focus (current reducible expression)
- Exhaustion tracking (which sub-expressions have no matching rules)
- Dependency info (which rules produced which terms, for incremental recomputation)

### Evaluation Loop

```
loop {
    for each active scope:
        find next reducible sub-expression (not exhausted, not quoted)
        lookup matching rules via term index
        if @strategy says :block, skip
        apply first matching rule (substitute logic variables)
        emit result into output scope
        emit meta-event into @meta
        update term index if rules changed
    if all scopes exhausted:
        break
}
```

### Compilation (Future)

Hot-path rules (stable, non-dynamic) can be compiled to:
- **Discrimination trees** for deeper pattern dispatch beyond top-level indexing
- **ARM-style bytecode** (Match/Build/Skip/Retract) for individual rule execution
- **Native code** via LLVM for the innermost loops

Dynamic and meta-rules stay interpreted. Two-tier execution.

## Open Questions

- **Negation**: How to express "match if this pattern does NOT appear"? Eve
  used `not(...)` blocks. Needs stratification to avoid paradoxes.
- **Aggregation**: `count`, `sum`, `min` over matched sets. Eve had this.
  Important for data transformation use cases.
- **Ordering/Priority**: When multiple rules match, which fires? First match?
  Priority annotations? Specificity?
- **Recursion limits**: How to prevent infinite rewriting? Fuel? Depth limits?
  Strategy rules?
- **Hygiene**: When meta-rules inject new rules, how to avoid variable capture?
- **Persistence**: Can scopes be backed by disk/database? (Eve explored this.)
- **Incremental pipelines**: In a pipeline, if `@source` changes, can we
  incrementally re-run only the affected stages?

## Example: Complete Program

```
# --- Standard library ---

fn fact(0) = 1
fn fact(?n) = ?n * fact(?n - 1)

fn map(?f, []) = []
fn map(?f, [?head, ?tail...]) = [?f(?head), map(?f, ?tail...)...]

# --- Data transformation ---

rule parse-input : @stdin -> @input {
  line(?text) => parse-json(?text)
}

rule transform : @input -> @output {
  {users: [?users...]}
    => map(normalize-user, [?users...])

  normalize-user({name: ?n, age: ?a})
    => {display: ?n, age: ?a, kind: :user}
}

rule render : @output -> @stdout {
  ?result => to-json(?result)
}

# --- Debugging (toggle by adding/removing) ---

rule trace : @meta -> @io {
  {sub_old: ?s, sub_new: ?t, scope: @main}
    => println(?s, " ~> ", ?t)
}

# --- Entry point ---

fact(10)
```
