# Live & Typed

**A language that is fully statically typed *and* fully hot-reloadable at runtime.**

> One-line thesis: static types for everything that is *running*; when a live update
> introduces an inconsistency, running code that reaches it **pauses, is repaired
> interactively, and resumes** — dynamic con-freeness by *trap-and-repair* rather than
> static con-freeness by *proof*. It is "Proteus's problem solved the Lisp way."

This document is the working design record. It captures the problem, the research it
rests on, the design decisions we made (with the alternatives we rejected), the single
primitive that reconciles them, an operational sketch, and the proof-of-concept that
runs the whole thing.

- Python POC: [`livetype_poc.py`](./livetype_poc.py) — the reference semantics, narrated.
- JS engine: [`livetype_engine.mjs`](./livetype_engine.mjs) — verified equal to the POC in Node.
- Explorable lab: [`livetype-lab.html`](./livetype-lab.html) — an editable browser IDE for the design.

---

## 1. The problem: the collision

Static typing says a value's type is fixed and checked *before* it runs. Full hot
reload says *any* definition — including a type — may change *while* values of that
type are live on the heap and mid-flight on the stack. They collide at exactly one
point:

> **A running computation is holding a value whose definition just changed underneath it.**

Every real system is a different answer to that collision. The literature splits into
three concerns that a full solution must all address:

1. **Dependency identity** — how you know what depends on a definition.
2. **Invalidation** — recompute exactly what changed; keep everything else.
3. **Migration + soundness** — keep live values type-sound across a type change.

No existing system is simultaneously (1) fully statically typed, (2) fully live in a
running process, and (3) sound at migration. The gap between them is the contribution.

---

## 2. Research foundations

The pieces we drew from (primary sources):

- **Typed DSU / Proteus** — Stoyle, Hicks, Bierman, Sewell, Neamtiu, *"Mutatis Mutandis:
  Safe and Predictable Dynamic Software Updating"* (POPL 2005 / TOPLAS 2007).
  [popl pdf](http://www.cs.umd.edu/~mwh/papers/proteus-popl.pdf) ·
  [journal pdf](https://www.cs.umd.edu/~mwh/papers/mutatis-journal.pdf).
  Introduces **con-freeness** (after a type changes, no code uses an old-representation
  value *concretely*), **type transformers** (`c : old-rep → new-rep`, checked
  well-typed), **representation consistency** (exactly one live representation of a type
  at a time), and a capability type system that makes safe update points statically
  inferable. Type safety = preservation/progress **plus a controlled failure mode**
  (`UpdEx`): an ill-timed update raises a recoverable exception, never corrupts.
- **DSU systems** — Ginseng (PLDI 2006, in-place, con-freeness + activeness),
  Kitsune/Ekiden (OOPSLA 2012, whole-program state transfer, explicit update points),
  Version Consistency (POPL 2008, a transaction runs entirely in one version). Safe-point
  vocabulary: **quiescence**, **activeness safety**, **con-freeness**, **version
  consistency**.
- **Runtime instance migration (the pragmatic prior art, all untyped)** — Erlang/OTP
  two-version code + `code_change/3`; **CLOS** `update-instance-for-redefined-class` /
  `make-instances-obsolete` (lazy, identity-preserving instance update on `defclass`
  redefinition); Smalltalk `become:` (atomic identity swap); JVM HotSwap / DCEVM / JRebel.
  Common thread: identity-preserving lazy migration is the most ergonomic model, and
  **the default failure mode everywhere is silent data loss** — none checks the
  migration is type-correct.
- **Incremental computation** — self-adjusting computation (Acar), **Adapton** (PLDI
  2014, demand-driven dirty/clean), **salsa/rustc** (query DAG, fingerprint **early
  cutoff**, durability firewall), **Build Systems à la Carte**. Guarantee to lean on:
  **from-scratch consistency** — incrementally-repaired caches are provably identical to
  recomputing everything, so it is safe to keep everything not on the invalidated set.
- **Content-addressed code** — **Unison**: every definition identified by the hash of its
  typed AST; names are metadata; dependencies never break; `update` propagates a patch to
  dependents, auto-deriving where the signature is unchanged.
- **Live typed programming** — **Hazel** (typed holes, evaluation *around* holes,
  fill-and-resume; the one *formally proven* liveness-under-types). React Fast
  Refresh / HMR is the heuristic, unsound engineering baseline. Gradual typing (blame)
  is the sound-boundary story for genuinely dynamic injection.

Synthesis: **content-addressing gives dependency identity, incremental computation gives
provably-coherent invalidation, typed DSU (Proteus) gives sound migration semantics, and
Hazel gives the continuous-liveness discipline — and no one has put all four in one
language.**

---

## 3. The design space and our choices

We walked nine roughly-orthogonal decisions. Our chosen positions:

| # | Dimension | Choice | Lineage |
|---|-----------|--------|---------|
| D1 | Definition identity | **Named + versioned history** (content-hash underneath, names a view) | hybrid / Unison |
| D2 | What a "change" is | **One live version at a time** (representation consistency) | Proteus |
| D3 | When an update applies | **Inferred safe points** (con-free) | Proteus capabilities |
| D4 | Running stack | **In-place, finish-then-switch** | Proteus / Ginseng |
| D5 | Value migration | **Lazy, identity-preserving** (on first touch) | CLOS / `become:` |
| D6 | Who writes migration | **Auto-derive where possible**, else supplied | Unison |
| D7 | Cache invalidation | **Demand-driven dirty/clean** | Adapton |
| D8 | Live-edit type discipline | **Deferred checking + pause-and-repair** (see §4) | CLOS conditions |
| D9 | Failure mode | **Pause & repair, resume** (barrier that yields to a human) | CLOS / Smalltalk |

### The tension resolutions

The nine choices reinforce each other, but five three-way interactions have real
friction. Our resolutions:

- **T1 — deferred fixing vs safe-point inference.** *Superseded by §4.* (Originally about
  "deferred type fixing": a deferred type is an abstract type, abstract types are
  con-free by construction, so deferred types are the *trivial* case and "fixing" one is
  just an ordinary migration — the same machine as T2.) The live-edit story we actually
  want is the condition system in §4, not typed holes.
- **T2 — one live version vs old code still building/mutating values.**
  **Abstract-allocate, migrate on cross.** Old code allocates at old layout, wrapped
  abstractly; the transformer migrates the value the first time *new* code dereferences
  it. (Proteus's exact mechanism; also what makes deferred/abstract types work.)
- **T3 — named+versioned vs auto-derivation needs hashes.**
  **Split substrate: nominal types, content-addressed functions.** Types get nominal
  identity + versions (so an instance has a clear class and a migration chain); functions
  are content-addressed (so call-graph propagation is automatic). *Name a type now, fix
  its representation later.*
- **T4 — two lazy graphs → first-touch latency.**
  **Fully lazy, accept the stutter.** Prototype-grade. The first access after a reload
  pays both a recompile and a migration; fine for now, revisit with background pre-warming
  later.
- **T5 — barrier can wait forever.**
  **Guarantee recurring safe points.** The compiler mandates yield/update points
  (loop-extraction, periodic checks) so a con-free point always recurs within bounded
  time; the barrier always eventually lands.

---

## 4. The reframe that ties it together: type conditions

The pivotal clarification: **D8 is not typed holes (Hazel-style incomplete editor
states).** It is *deferred checking with interactive repair*.

> You apply an update. The program keeps running. When a thread of control **reaches**
> code that is now type-inconsistent — a function that no longer typechecks, or a value
> that can't be auto-migrated — it does **not** crash and the update is **not** rejected.
> It **pauses at that point, surfaces the inconsistency, you fix the code live, and it
> resumes.**

Concretely:

```
type Account = { balance: Int }
fn charge(a: Account, amt: Int) = a.balance := a.balance - amt   // fine

// ---- hot update:  type Account = { balance: Money }  (Money = { cents: Int })

// charge now computes  Money - Int  → inconsistent.
// A request calls charge(acct, 500), control reaches charge → PAUSE.
//   "charge no longer typechecks: balance is Money, amt is Int, Money - Int undefined"
// You edit:  a.balance := a.balance - Money.fromCents(amt)   → typechecks → RESUME.
```

This is the **Common Lisp condition/restart system** (and Smalltalk debugger-driven
development), specialized to type inconsistencies introduced by a hot update, in an
otherwise statically-typed language.

It is the exact **dual of Proteus**:

- **Proteus:** prove *statically* that no old value is ever used wrongly; reject updates
  you can't prove safe.
- **Us:** *let* the inconsistency happen, **detect it dynamically at the point of use,
  and freeze-and-repair** instead of crashing.

### One primitive underneath everything

Every dimension collapses onto a single mechanism — **a type condition raised at the
point of use, with freeze → repair → resume**:

- **D9 (failure)** *is* the condition system.
- **D6 (migration)** unifies with it: a value that can't be auto-migrated raises *the same
  kind of condition*; you supply the transformer live. "Broken function" and
  "un-migratable value" are the *same* event with two `kind`s.
- **D3 (con-freeness)** flips from a static *proof obligation* into a dynamic *invariant*:
  instead of proving ahead of time that no old value is used wrongly, you **trap** the
  moment it would be, and **quarantine** the paused frame.
- **D2 / D5 (one version, lazy migrate)** become: the transactional unit (a "tick")
  commits its new state only at the end; a condition **rolls the tick back**, so a
  half-typed state is never published — quarantine holds *by construction*.

### The soundness invariant we are testing

> **Every committed state is well-typed under the current definitions. A paused frame is
> quarantined — its in-flight, possibly ill-typed state is discarded on rollback — so no
> running code ever observes an ill-typed value.**

If this holds, "well-typed programs don't go wrong" is preserved for everything that is
*actually executing*; the inconsistency is sealed behind the pause.

---

## 5. Operational model

The reference semantics (as implemented in the POC):

**Definitions.** Nominal types carry a list of versioned layouts
(`type_versions[name] = [layout_v1, layout_v2, …]`); the last is *live* (D2). Functions
reference callees by name but the store tracks their versions. A field may have a
**default**, which lets construction omit it and enables auto-migration.

**Safe points.** Updates are applied *between ticks* — the natural quiescence point.
A tick is the transactional unit (D9/T5): `state = tick(state)`, pure, so it can be
re-run.

**Hot update (reconcile).** Parse the new definitions. For each type whose layout
changed, append a new version and attempt to **auto-derive** the migration
(`v_i → v_{i+1}`): each new field is *copied* if present in the old layout at the same
type, or *default-initialized* if it has a default; otherwise the migration is **not
auto-derivable** and a **gap** is recorded. Replace function bodies. **Re-typecheck every
function** against the current types (eager detection) and flag the broken ones — but
only *surface* the breakage when execution reaches it (lazy).

**Migrate-on-cross (D5, T2).** When evaluation reaches `e.f` and `e` is a record tagged
at an older version than current, migrate it up the chain **in place** (identity-preserving,
`become:`-style), applying each registered transformer once. If a step has **no
transformer** (a gap), raise `Pause('migration', …)`.

**Broken-function trap (D8/D9).** When about to *enter* a function flagged broken, raise
`Pause('function', …)` — at the call boundary, *before* the body runs any effects, so
resume is a clean frame restart.

**Pause / repair / resume.** A `Pause` unwinds to the tick boundary; the in-flight state
is discarded (quarantine). The developer supplies a repair — new function source
(re-typechecked before it is accepted) or a migration expression with `old` in scope
(the *trusted transformer boundary*, à la Proteus — checked to produce a valid new-layout
value, not proven semantically correct). The tick **re-runs from the top**.

**Commit + invariant.** On success, the new state is checked structurally well-typed
against the current layouts (`value_ok`) before it is published. This assert is what makes
§4's invariant *tested*, not asserted.

---

## 6. The proof-of-concept

`livetype_poc.py` implements the whole thing (~350 lines: lexer, parser, type checker,
evaluator with lazy migration, versioned types, the condition system) and runs a narrated
scenario. `livetype_engine.mjs` is a JavaScript port **verified byte-for-byte equal** to
the Python (same pauses, same migrations, `cents` 8000/7500/7000). `livetype-lab.html` is
an editable browser IDE built on that engine.

The scenario the POC drives:

1. A bank account is charged 5 every tick. `Account@v1 = { id, balance: Int }`. Runs.
2. **Hot update A** adds `fee: Int = 0`. Auto-derivable → the live value migrates
   `v1 → v2` lazily on next touch, no pause. *(D5 + D6 happy path.)*
3. **Hot update B** changes `balance: Int → Money`. This **breaks `charge`** (`Money - Int`,
   detected eagerly) **and** makes the live value un-auto-migratable.
4. Tick reaches broken `charge` → **⏸ pause (function)** → developer edits `charge` →
   re-typechecks → **↻ resume**.
5. The resumed tick reaches the un-migratable value → **⏸ pause (migration)** → developer
   supplies the `Int → Money` transformer → **↻ resume**.
6. Tick commits; the program keeps running. **Never restarted.** Every committed state
   passes the enforced `value_ok` assert.

Two condition kinds, one mechanism. Run it:

```
python3 livetype_poc.py       # narrated reference run
node livetype_engine.mjs      # the JS engine, verified equal
open livetype-lab.html        # (or the published artifact) the interactive lab
```

---

## 7. What is proven — and what is not

**Proven:** the design is *coherent and runnable*. One primitive reconciles hot updates,
lazy typed migration, and static soundness. The `✓ well-typed` badge is an enforced check,
so nothing ill-typed is ever committed, because a pause rolls the in-flight tick back.

**Deliberately the easy version (open corners):**

1. **Single-threaded.** Quarantine is trivial here because there is one process and
   rollback discards the in-flight state. The real test is **two concurrent processes
   sharing a value mid-migration** — if a running thread and a paused thread both reach
   the same wrong-shaped value, quarantine can leak. *This is the highest-value next
   experiment: it either holds or teaches us the missing rule (world-freeze vs. ownership/
   reachability).*
2. **Pause at call-entry, before effects.** Because a tick is a pure transaction, "restart
   the frame" is free. A pause **mid-expression after a side effect has already happened**
   is the genuinely hard case for resume semantics.
3. **Monotone changes; name+version identity.** Types only move forward (no re-fixing);
   identity is nominal+version, not yet content-addressed (T3's function half is
   unimplemented). Re-fixing a type reopens full migration territory.

---

## 8. Open questions / research agenda

- **Multi-thread quarantine** (corner 1). Does the invariant survive a shared heap? What
  is the minimal extra rule — freeze-the-world at a condition, or a reachability/ownership
  guarantee that the quarantined value is unshared?
- **Effectful resume** (corner 2). What does "resume" mean when a tick can't cleanly roll
  back? Restart with compensation? Checkpoint sub-expressions?
- **Re-fixing / non-monotone changes.** If a type can change more than once, the migration
  chain and con-freeness both need the general treatment.
- **Content-addressed function substrate** (T3). Implement functions-by-hash so
  auto-derivation propagates through dependents for free, and reconcile it with nominal
  versioned types.
- **Con-freeness statically, not just dynamically.** Right now safe points are "between
  ticks." Adopt Proteus's capability annotations to *infer* con-free points and shrink the
  reliance on the pause path — the pause becomes the fallback, not the norm.
- **The two dependency graphs** (T4). Separate the compile-time derived-artifact graph
  (salsa/Adapton, for typechecks/IR/JIT) from the runtime heap-value graph (for
  migration), and add background pre-warming to remove the first-touch stutter.
- **What can a repair be?** Just a function body? Also re-fixing a type? Also a
  hand-supplied migration for one specific value? The richer the repair vocabulary, the
  more this becomes a full live-programming environment.

---

## 9. The one-paragraph summary

A statically-typed language stays sound by keeping every *running* frame well-typed. Make
hot reload first-class by letting updates land at safe points, migrating live values
lazily and identity-preservingly (auto-derived where the change is compatible), and —
when running code reaches something the update broke — raising a **type condition** that
freezes the frame, discards the in-flight (quarantined) state, lets the developer repair
the code or supply a migration live, and resumes by re-running the transaction. This is
Proteus's soundness goal reached the Lisp way: **dynamic con-freeness by trap-and-repair
instead of static con-freeness by proof.** The POC runs it end-to-end; the hard corners
(concurrency, effectful resume, non-monotone change) are the research agenda.
