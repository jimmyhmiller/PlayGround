# Deviations from the Paper

This documents where this implementation deviates from or extends the formalism
in Edwards & Petricek, "Typed Image-based Programming with Structure Editing"
(HATRA'21, arXiv:2110.08993).

Each deviation is classified as:
- **Paper error**: The paper's rule produces an incorrect result.
- **Paper gap**: The paper's rules don't cover this case (no rule matches).
- **Implementation note**: Not a deviation — a clarification or subtlety.

All deviations are verified by property-based tests (5000+ cases each for
commutativity, partial inverses, convergence, etc.).

---

## 1. `retract(x, x)` for non-idempotent edits

**Classification**: Paper error.

### What the paper says

Appendix A, "Equal edits cancel out" shows two diagrams. The left diagram is
`project(x, x) = (Id, Id)`. The right diagram shows `retract(x, x) = (Id, x)`
— the post (right arrow) duplicates the diff (top arrow), so the pre grounds
to Id.

The paper's text (p. 4): "if the left or right edits duplicate the top one then
they make no difference when migrated to the other side, so 'doing the same
thing' demands they ground out to Id."

### What goes wrong

`retract(x, x) = (Id, x)` requires `x ∘ x = x ∘ Id`, i.e., `x² = x`
(idempotence). This holds for Conv (applying the same type conversion twice is
the same as once) but FAILS for Move:

```
Move{0,1} ∘ Move{0,1} on [A, B]:
  First:  [B, Del]
  Second: [Del, Del]     ← NOT equal to Move{0,1}([A,B]) = [B, Del]
```

Furthermore, there is no single edit `adjust` such that `(pre=Id, adjust)`
satisfies commutativity, because `x²` for Move produces effects at two
positions that no single edit can express when starting from the unmodified
document.

The paper's "Conflicting Move" rule (`Move[i,k]` vs `Move[i,j]`, j ≠ k) also
does not cover equal Moves because the condition `j ≠ k` excludes the case
where source and target both match.

### What we do instead

Remove the generic equal-edits-cancel from retract. Handle each edit type:

- **Conv**: `retract(Conv{i,t}, Conv{i,t}) = (Conv{i,t}, Id)` — Conv is
  idempotent, so both the paper's `(Id, Conv)` and our `(Conv, Id)` satisfy
  commutativity. We use `(Conv, Id)` which is consistent with the
  conflict-override pattern used for `Conv{i,t}` vs `Conv{i,u}`.
- **Move**: `retract(Move{i,j}, Move{i,j}) = (Move{i,j}, Conv{i,Del})` —
  algebraically derived. Applying Move{i,j} then Conv{i,Del} gives i=Del,
  j=Del, same as applying Move{i,j} twice.
- **Ins**: Same-id Ins pairs represent the "same insert" per the paper's unique
  identifier mechanism (Section 2.2, Proposition 4). Returns `(Id, Id)`.

### Verification

```
retract(Move{0,1}, Move{0,1}) = (Move{0,1}, Conv{0,Del})
Path 1: Move{0,1} ∘ Move{0,1} on [A,B] → [B,Del] → [Del,Del]
Path 2: Conv{0,Del} ∘ Move{0,1} on [A,B] → [B,Del] → [Del,Del] ✓
```

Tested by `prop_retract_self_satisfies_commutativity` (2000 cases).

---

## 2. Move-Move `pj == di` (pre source = diff target)

**Classification**: Paper gap.

### What the paper says

Appendix A defines three Move-Move cases:

1. **Conflicting** (`pi == di`): `project(Move[i,k], Move[i,j]) = (Move[i,k], Conv[j,del])`, condition `j ≠ k`
2. **Source-to-target** (pre references diff source `j`):
   - `Move[j,k]` → `Move[i,k]` (condition `i ≠ k`)
   - `Move[k,j]` → `Move[k,i]` (condition `i ≠ k`)
   - adjust = `Move[i,j]` (unchanged)
3. **Pass-through**: `x = Move[k,l]` with `i ≠ k ∧ j ≠ k ∧ i ≠ l ∧ j ≠ l`

Case `pj == di` means `l == i` in the pass-through notation, which violates
`i ≠ l`. It also doesn't match source-to-target (which requires referencing
`j = dj`, not `i = di`). And it's not conflicting (`pi ≠ di`). **No rule
applies.**

### What we do instead

Algebraically derived (unique single-edit solution, verified by exhaustive
search over all edit types for concrete documents):

```
project(Move{pi, di}, Move{di, dj}) = (Move{pi, di}, Move{pi, dj})
```

Semantically: the pre reads from `di`, which the diff overwrites with `dj`'s
value. The post still reads from `di` (getting the new value). The adjust
captures the net effect — `pi` ended up with `dj`'s value.

### Verification

```
pre = Move{3,0}, diff = Move{0,2} on [A,B,C,D]:
Path 1: Move{0,2} → [C,B,Del,D]. Move{3,0} → [Del,B,Del,C].
Path 2: Move{3,0} → [Del,B,C,A]. Move{3,2} → [Del,B,Del,C]. ✓
```

Tested by `prop1_project_commutativity` (5000 cases).

---

## 3. Move-Move `pi == dj && pj == di` (swap)

**Classification**: Paper gap.

### What the paper says

This case requires `pi = dj` AND `pj = di`. Checking each rule:

- Source-to-target `Move[j,k]` (where `pi = j = dj`): condition `i ≠ k` means
  `di ≠ pj`. But `pj = di`, so `di ≠ di` fails.
- Source-to-target `Move[k,j]` (where `pj = j = dj`): requires `pj = dj`, but
  `pj = di ≠ dj`.
- Conflicting: requires `pi == di`, but `pi = dj ≠ di`.
- Pass-through: requires `i ≠ l`, but `l = pj = di = i`.

**No rule applies.**

### What we do instead

Algebraically derived (unique single-edit solution):

```
project(Move{dj, di}, Move{di, dj}) = (Move{di, dj}, Move{dj, di})
```

The project swaps pre and diff. This is a degenerate but algebraically correct
solution.

**Proposition 3 violation**: `post = Move{di,dj} = diff`, so
`project(post, diff) = project(diff, diff) = (Id, Id)`, not `(pre, adjust)`.
This means retract cannot recover the original pre for this case. The paper
acknowledges related issues (p. 10): "the asymmetries introduced by overriding
conflicts don't satisfy the definitions of pushout and pullback."

### Verification

```
pre = Move{2,0}, diff = Move{0,2} on [A,B,C]:
Path 1: Move{0,2} → [C,B,Del]. Move{0,2} → [Del,B,Del].
Path 2: Move{2,0} → [Del,B,A]. Move{2,0} → [Del,B,Del]. ✓
```

Tested by `prop1_project_commutativity` (5000 cases).

---

## 4. `edit_a` when `δ ≠ Id`: a_diffs must not be modified

**Classification**: Implementation note (matches the paper; documents a bug we
found and fixed during development).

### What the paper says

Page 6: "If δ ≠ Id or is undefined then the edit is appended to the differences
of A as shown in this diagram:"

```
A' ← ε → A ← aₙ ... ← a₁ → A&B → b₁ ... → bₘ → B
```

The diagram shows ε appended to A's differences with the agreement A&B and
existing diffs **unchanged**.

### The bug we had

Our initial `edit_a` used the translate function's internally-adjusted diffs
in both the `δ = Id` and `δ ≠ Id` cases. The translate function retracts ε
through a_diffs as an intermediate step, modifying them. These modifications
should only be kept when `δ = Id` (edit absorbed into the agreement).

When `δ ≠ Id`, using the adjusted diffs breaks the invariant that
`agreement → a_diffs → A` reconstructs A's state.

### What we do

```rust
if result.delta.is_id() {
    // Absorbed: use adjusted diffs (agreement advanced)
    self.a_diffs = result.a_diffs;
    self.b_diffs = result.b_diffs;
} else {
    // Not absorbed: keep original diffs, just append
    self.a_diffs.push(*epsilon);
}
```

### Verification

Tested by `prop6_convergence` (1000 cases) and
`prop_same_edit_both_sides_no_diff`.

---

## 5. `retract(Conv{mi, t}, Move{mi, mj})` — Conv at Move target

**Classification**: Implementation note (follows from the paper, but the
appendix presentation is easy to misread).

### What the paper says

Two separate rules interact:

1. "Can't project Conv through Move to its target" (crossed-out arrow):
   `project(Conv{mi, _}, Move{mi, mj}) = (Id, Move)` — Conv is overridden.

2. "Move moves source to target":
   `project(Conv{mj, t}, Move{mi, mj}) = (Conv{mi, t}, Move{mi, mj})` — Conv
   at source follows value to target.

### The subtlety

Rule 2 produces `post = Conv{mi, t}` — a Conv at the Move's target. This is
the RESULT of a source Conv being projected, not a new Conv targeting the
destination.

By Proposition 3 (since `post = Conv{mi, t} ≠ Id`):
`retract(Conv{mi, t}, Move{mi, mj}) = (Conv{mj, t}, Move{mi, mj})`.

It's tempting to return None here (since "can't project Conv through Move to
its target"), but that would violate Prop 3. The retract correctly maps the
Conv back from the target to the source.

Meanwhile, `retract(Conv{mj, t}, Move{mi, mj})` genuinely IS None — no project
rule produces a post with `idx == mj` for this diff.

### What we do

```rust
// Conv at diff source (idx == mj): None — no project produces this
(Edit::Conv { idx: i, .. }, Edit::Move { j: mj, .. }) if i == mj => None,
// Conv at diff target (idx == mi): retract to source (Prop 3)
(Edit::Conv { idx: i, ty: t }, Edit::Move { i: mi, j: mj }) if i == mi => {
    Some((Conv{mj, t}, Move{mi, mj}))
}
```

---

## Summary

| # | Case | Classification | Issue | Our solution |
|---|------|---------------|-------|-------------|
| 1 | `retract(x,x)` for Move | Paper error | Requires idempotence (`x²=x`) | Case-by-case: `(Move, Conv{i,Del})` |
| 2 | Move-Move `pj==di` | Paper gap | No rule matches | `(Move{pi,di}, Move{pi,dj})` |
| 3 | Move-Move swap | Paper gap | No rule matches; Prop 3 breaks | `(Move{di,dj}, Move{dj,di})` |
| 4 | `edit_a` when `δ≠Id` | Impl note | Bug fix, now matches paper | Don't modify existing diffs |
| 5 | Conv at Move target | Impl note | Follows from Prop 3 | `(Conv{mj,t}, Move)` not None |
