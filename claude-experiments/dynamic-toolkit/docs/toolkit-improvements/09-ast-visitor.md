# 09 — AST Visitor Boilerplate

## Implementation status: 📋 Plan only — not implemented

This is the only doc in the series that hasn't been touched. Unlike the
other nine, it's a pure beagle-side refactor (no toolkit primitive to
add) — the recommendation in the doc is **Option C**: write one
generic `walk(ast, &mut closure)` and convert the four hand-rolled
walkers (`collect_mutated_in`, `find_length_of_ident`, `node_count`,
`calls_function`/`calls_in`, `count_in`) to closures. Estimated savings
~400 lines in `crates/beagle/src/lower.rs`, but it doesn't add anything
reusable to the toolkit.

Worth doing as a beagle hygiene task; not part of the toolkit-improvement
push.

## Problem

`crates/beagle/src/lower.rs` contains five hand-rolled `Ast` walkers,
each ~100-200 lines of nearly-identical match arms covering every
node variant:

- `collect_mutated_in` ([`lower.rs:1221-1412`](../../crates/beagle/src/lower.rs#L1221))
  — collects every name that's reassigned or rebound in a subtree.
- `find_length_of_ident` ([`lower.rs:1417-1607`](../../crates/beagle/src/lower.rs#L1417))
  — collects every `length(x)` where `x` is a bare identifier.
- `node_count` / `nc` ([`lower.rs:1613-1817`](../../crates/beagle/src/lower.rs#L1613))
  — sums reachable nodes for the inline-budget heuristic.
- `calls_function` / `calls_in` ([`lower.rs:1822-2006`](../../crates/beagle/src/lower.rs#L1822))
  — true if any call targets a given name.
- `count_property_accesses_with_inlining` / `count_in`
  ([`lower.rs:2038-2221`](../../crates/beagle/src/lower.rs#L2038))
  — same shape, plus "expand inlinable callees".

Each walker reproduces the structural recursion through every Ast
variant: `Program`, `Function`, `If`, `While`, `For`, `Match`,
`MultiArityFunction`, `StructCreation` with `spread`,
`StringInterpolation`, `Try`, `Handle`, `Future`, `Perform`, etc.
Adding a new variant means touching every walker, and it's easy to
forget — silent miscompilation if one of the walkers misses a child.

That's roughly 800 lines of code that exists only because there's no
shared traversal infrastructure.

## Proposed API

This is the only item on the list that's mostly *frontend-side*, not
toolkit-side — the AST belongs to beagle. But the same problem will hit
every dynlang frontend, and `dynlang` is a natural home for a shared
visitor abstraction that languages built on this stack can derive.

### Option A — `Visit` / `VisitMut` trait derived on the AST

```rust
// In dynlang (or a sibling crate) — generic over the AST node type.
pub trait Visit<'ast, A: AstNode> {
    fn visit_node(&mut self, node: &'ast A) {
        // Default: recurse into children.
        node.walk(self);
    }
}

// AstNode is implemented (or derived) by the embedder; provides children().
pub trait AstNode {
    fn walk<V: Visit<Self>>(&self, v: &mut V);
}
```

Each walker becomes a single `Visit` impl that overrides only the
variants it cares about. Beagle's `find_length_of_ident` shrinks to:

```rust
struct LengthIdents(HashSet<String>);
impl Visit<Ast> for LengthIdents {
    fn visit_node(&mut self, node: &Ast) {
        if let Ast::Call { name, args, .. } = node {
            if name == "length" && args.len() == 1 {
                if let Ast::Identifier(x, _) = &args[0] {
                    self.0.insert(x.clone());
                }
            }
        }
        node.walk(self);  // continue recursion
    }
}
```

The exhaustive match goes into a derived `walk` impl on `Ast` —
written *once* — so adding a new variant is a one-place change.

### Option B — `derive(Visit)` proc macro

A proc macro that walks struct/enum fields and emits the `walk` impl.
Same surface as Option A but no manual `walk` body. Higher infra cost
(proc-macro crate, syn dep) but pays back fast.

### Option C — Beagle-only refactor (no toolkit change)

Just write *one* generic walker in beagle and parameterize it on a
closure. Not as clean as a trait but ships immediately. The four
walkers collapse to:

```rust
fn walk(ast: &Ast, f: &mut impl FnMut(&Ast)) {
    f(ast);
    for child in children(ast) {
        walk(child, f);
    }
}

// Then:
let mut idents = HashSet::new();
walk(program, &mut |n| {
    if let Ast::Call { name, args, .. } = n {
        // ...
    }
});
```

`children(ast)` is a single big match — replaces all four giant
matches with one.

## Implementation plan

Recommended sequence:

1. **Land Option C in beagle.** Write `fn children(ast: &Ast) -> impl
   Iterator<Item = &Ast>` once. Convert the four walkers to closures.
   Net delta: ~−400 LOC, no toolkit change. Do this as soon as
   anything else lower.rs-related lands. Pure refactor — no behavior
   change.

2. **If/when dynlang grows a shared AST type** (e.g. for the bootstrap
   compiler in `crates/dynlang` or a future shared parser), promote
   to Option A. Beagle's `Ast` would either gain an `impl Visit` or
   migrate to the shared AST. This is speculative; don't block on it.

3. **Option B is overkill for now.** The proc-macro investment only
   pays back across multiple frontends. Revisit when the second
   frontend (Lua? Lox?) is hand-rolling its own walkers.

## Open questions / risks

- **Beagle's AST has fields, not just structural children.** Some
  variants carry `Option<Box<Ast>>` (e.g. `StructCreation::spread`),
  some carry `Vec<Ast>`, some carry `Vec<(String, Ast)>` (struct fields,
  map pairs), some carry tuple-shaped children with custom enum
  wrappers (`StringInterpolationPart::Expression`). `children()` has
  to handle all of these. The cleanest implementation builds a small
  `Vec<&Ast>` per call; allocs are negligible vs the IR work
  downstream.
- **Mut variants.** None of beagle's current walkers mutate the AST.
  Adding `VisitMut` later would double the code; keep `Visit` only
  until a real need (e.g. an AST rewriter) shows up.
- **Termination of the inlining-aware walker.** `count_in` needs the
  `visited: HashSet<String>` cycle guard for inlinable callees.
  That's a stateful traversal, not pure structural — it lives one
  layer above `walk`. Good news: in Option C, the closure carries
  the state naturally.
