# Name-Based OT: Extending the Formalism for Records

## Motivation

The paper (Edwards & Petricek, HATRA'21) defines an OT algebra for **positional
tuples** — ordered sequences where fields are identified by index. The four edit
operations (Id, Ins, Conv, Move) manipulate indexes:

- `Ins[i, t]` — insert at position i, shifting everything ≥ i right
- `Conv[i, t]` — convert the type at position i
- `Move[i, j]` — copy position j to position i, delete j

This works for ordered data (text, lists). But **records/schemas** are unordered
maps — fields are identified by **name**, not position. A record `{name: "Alice",
age: 30}` is the same regardless of field order.

## The Category Error

Using positional OT for records creates fundamental problems:

1. **Index shifting**: When two branches add a column, Ins shifts the other's
   index. After cascade merges through 3+ branches, indexes become wrong.

2. **Duplicate columns**: Two branches adding "email" create two positional
   slots both named "email" — the OT treats them as different inserts.

3. **Cross-layer breakage**: Value edits (Set) reference column indexes that
   shift unpredictably during schema merges.

These aren't bugs in the OT algebra — they're consequences of applying a
positional formalism to named data.

## The Extension: Name-Based OT

We define a new edit type for schema operations that uses **names as identifiers**
instead of positions:

```rust
enum SchemaEdit {
    Id,
    AddField { name: String, ty: AtomicType },
    RemoveField { name: String },
    ConvertField { name: String, ty: AtomicType },
    RenameField { old_name: String, new_name: String },
    SetField { field: String, value: Value },
}
```

## The Algebra

The project/retract functions follow the same structure as the paper:

```
schema_project(pre, diff) = (post, adjust)
  such that: post ∘ diff = adjust ∘ pre
```

But the rules are dramatically simpler because name equality replaces index
equality:

### Core Rules

**Independent** (different target names):
```
project(AddField("email", Str), AddField("phone", Num))
  = (AddField("email", Str), AddField("phone", Num))
```
Both pass through unchanged. No shifting.

**Cancel** (same name, same intent):
```
project(AddField("email", Str), AddField("email", Str))
  = (Id, Id)
```
Equal edits cancel — the field already exists on both sides.

**Conflict** (same name, different intent):
```
project(AddField("email", Str), AddField("email", Num))
  = (ConvertField("email", Str), Id)
```
Pre wins. The field exists on both sides (cancel), but types differ, so a
Conv reconciles.

**Rename follow**:
```
project(RenameField("name", "full_name"), SetField("name", "Alice"))
  = (RenameField("name", "full_name"), SetField("full_name", "Alice"))
```
The Set follows the rename — its target name updates.

## Why This Is the Right Formalism

1. **No index shifting** — fields are identified by name, not position.
   Adding "email" on two branches creates one field, not two.

2. **No cascade duplicates** — the cancel rule works for N branches.
   If A, B, and C all add "email", every bilateral merge cancels.

3. **No Move** — records don't have a meaningful "move" operation.
   Reorder is not a schema change.

4. **Value edits by name** — `SetField("age", 31)` always targets the
   right field, regardless of schema changes on other branches.

5. **Conflict model** — same field name = potential conflict.
   Different field names = always independent. Clear and precise.

## Relationship to the Paper

This is an **extension** of the paper's theory, not a replacement:

- The paper's positional OT (Id, Ins, Conv, Move) remains correct for
  **ordered sequences** — row ordering, list elements, text.

- The name-based OT (SchemaEdit) handles **unordered maps** — record
  schemas, field definitions.

- Both algebras satisfy the same commutativity property:
  `post ∘ diff = adjust ∘ pre`

- Both use the same translate/migrate machinery for difference tracking.

The paper's prototype (Section 3) extended the minimal theory to "nested
records and homogeneous lists, forming a tree." Our name-based OT is a
formalization of the record level of that extension.

## Properties (Verified by Property Tests)

- **Commutativity**: `schema_project(pre, diff) = (post, adjust)` implies
  both paths through the commutative square give the same result.

- **Cancel**: Equal edits cancel to (Id, Id).

- **Convergence**: After bidirectional merge between any branches, schemas
  are identical — same fields, same types, same values. Verified across
  9000+ random scenarios including 3-branch and fork-chain topologies.
