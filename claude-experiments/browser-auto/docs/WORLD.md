# The world system

Flakiness's second home (after timing) is shared, dirty state. bat's answer:

> **A seed is not a procedure that mutates the world. A seed is data that
> describes a world.**

## The algebra

- A **world description** `W` is a set of typed, keyed facts:
  "product `blue-widget` exists with these fields."
- Seeds combine by **merge**: `W₁ ⊕ W₂`.
  - same (type, key), deep-equal value → deduped;
  - same (type, key), different value → **hard error** (never last-writer-wins).
- `⊕` is commutative, associative, idempotent; the empty seed is the identity.
  `given seed A/B/C` in any order means one thing or refuses to compile.
- **References** are first-class: `ref("products", "blue-widget")` inside a
  fact. After merge, a **closure check** verifies every ref resolves. Dangling
  refs are static errors naming the exact fact and field.
- **Patches** are explicit, ordered overrides applied after merge
  (`given patch products "blue-widget" stock 0`). Patching a missing fact is an
  error. Algebraically: seeds form the semilattice; patches are ordinary
  functions applied to the merged result.
- The merged, patched description serializes canonically and hashes to a
  **world fingerprint**. Traces record it; replay verifies it.

## Realization

```
apply(W) = reset the world to empty, then install facts in dependency order
```

Nothing is ever incremental. No seed runs against a dirty world. Isolation is
not a property you test for — it is unrepresentable to violate, because seeds
contain no code.

## The adapter: an open world with a capability ladder

The framework owns the algebra; the app owns what facts mean. Entity types are
open — we never enumerate what can exist. The contract: **every operator you
provide buys a stronger checked guarantee.** Degradation is loud, never
silent — every trace records which guarantees were *proven* vs *asserted*.

```ts
import { defineWorld } from "browser-auto/world";

export default defineWorld({
  reset: async () => { ... },                    // required
  entities: {
    products: {
      install: async (rows, ctx) => ({...ids}),  // required per type
      schema: (row) => null | "error message",   // optional → L1
      read: async (keys) => ({...rows}),         // optional → L2
      needs: ["users"],                          // install-order deps
    },
  },
  session: async (userKey, ctx) => ({ cookies }),// for `given user X signed-in`
  fingerprint: async () => "hash",               // optional → L3
  snapshot: async () => "id",                    // optional → L4
  restore: async (id) => { ... },                // optional → L4
});
```

| level | requires | what bat can then prove |
|---|---|---|
| **L0 trust-me** | `reset` + `install` | full algebra (conflicts, closure, patches) on the description; operators that throw fail with attribution |
| **L1 validated** | `schema` | fact shapes checked at merge time — before any browser or DB is touched |
| **L2 verified** | `read` | post-install read-back diff: "described `stock: 12`, world contains `stock: 0`" |
| **L3 drift-guarded** | `fingerprint` | world checked at flow boundaries — catches leaked state from anywhere |
| **L4 time-travel** | `snapshot`/`restore` | content-addressed world cache; true single-step atomic replay |

`install` may return a `{ key: id }` map; refs to that type resolve through it
(`ctx.id("products", "blue-widget")`). If an installer returns no ids, refs to
that type are an error at closure-check time.

`bat doctor` reports the current level and exactly which ~5-line function to
implement next, and what it buys.

## Transport

- `{ "module": "./e2e/world/world.ts" }` — in-process (the runner imports it).
- `{ "http": "http://localhost:3000/api/__bat" }` — the app mounts a handler
  (`createWorldHandler(world)` from `browser-auto/server`); the runner merges
  and validates locally, then ships the merged facts. The handler refuses to
  exist unless `BAT_TEST=1`.

## Seed files

Pure data. What LLMs write per scenario:

```ts
import { seed, ref } from "browser-auto/world";

export default seed("catalog-basic", {
  users: {
    shopper: { email: "shopper@test.dev", role: "customer" },
  },
  products: {
    "blue-widget": { name: "Blue Widget", price: 1999, stock: 12 },
    "red-widget":  { name: "Red Widget",  price: 2499, stock: 3 },
  },
  discounts: {
    summer: { code: "SUMMER", pct: 10, product: ref("products", "blue-widget") },
  },
});
```
