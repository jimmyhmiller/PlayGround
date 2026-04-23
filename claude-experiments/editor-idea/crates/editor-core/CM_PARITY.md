# CM6 Parity Audit

This document catalogues every place `editor-core` diverges from CodeMirror 6
while claiming to port its behavior. It covers both behavioral gaps (our code
does less than CM6) and test-suite gaps (our `cm_*.rs` files were adapted or
subset, hiding missing rules).

Scope: `crates/editor-core/` only. Cross-referenced against upstream CM6 test
suites in `@codemirror/commands` and `@codemirror/state`.

High-level shape: we reimplement doc / selection / transaction / history /
commands / comment / indent as plain Rust structs. There is **no port of CM6's
extension system** (Facets, StateFields, Compartments, Effects, Annotations,
ChangeFilters, TransactionFilters, TransactionExtenders) — this is the single
largest omission. Most `cm_state.rs` upstream `describe` blocks depend on that
system and are absent entirely rather than "adapted."

---

## History / Undo-Redo

### Current state

- `src/history.rs` implements `apply_with_history` + `apply_with_history_isolated`,
  undo/redo stacks, and a separate selection-history stack.
- Coalescing: **adjacency + same-op-kind only** (`can_join` at
  `src/history.rs:82-102`). Two single-change inserts merge if
  `prev.from + len == next.from`; two single-change deletes merge if
  `next.to == prev.from`. Multi-change events never join.
- `apply_with_history_isolated` forces a new event — maps to CM6's
  `isolateHistory: "full"`.

### CM6 behavior

- Time-based coalescing driven by `newGroupDelay` (default 500 ms). Adjacent
  edits outside that window start a new group.
- Cursor-move in between edits breaks a group (the `selectionsAfter` trick in
  CM6's `history.ts`).
- `userEvent` inspection: non-typing events (paste, `input.paste`, `delete.cut`)
  are their own group; `addToHistory: false` annotation skips history entirely;
  explicit `isolateHistory` values (`"before" | "after" | "full"`) force boundaries.
- Custom `joinToEvent` predicate can extend the default join rule.
- Effects are preserved through undo/redo and can opt into inversion.
- JSON serialization of history state.

### Gaps

- No time component — relies purely on spatial adjacency, so rapid non-adjacent
  typing never coalesces and slow adjacent typing always coalesces.
- No cursor-move break: typing `a`, move cursor, typing `b` at adjacent position
  would still join in our impl (CM6 would not).
- No `userEvent` concept — paste-is-its-own-group, cut-is-its-own-group, and
  "select" transactions are all handled ad-hoc.
- No `addToHistory: false` annotation — `apply` is the only escape hatch.
- No `isolateHistory: "before" | "after"` — only `"full"` via
  `apply_with_history_isolated`.
- No custom `joinToEvent` predicate.
- No Effects → no inverted-effect storage, no effect mapping.
- No JSON round-trip.
- `can_join` refuses any multi-change event, so multi-cursor typing never
  coalesces (CM6 joins consecutive multi-cursor typing of one char per range).

### Evidence

- `src/history.rs:1-5` — *"Simpler than CM6's: no time-based coalescing, no
  isolateHistory markers, no addToHistory: false annotation. Each call to
  apply_with_history records one event."*
- `src/history.rs:34-35` — *"matches CM6's default newGroupDelay coalescing
  without time tracking"* (literally claims to match a time-based rule without
  tracking time).
- `src/history.rs:82-102` — `can_join` only looks at adjacency + op kind; no
  time, no cursor-move signal, no userEvent.
- `tests/cm_history.rs:1-4` — *"Subset of CodeMirror 6 commands/test/test-history.ts.
  No event coalescing implemented — each apply_with_history is its own undoable
  event. Tests that depend on newGroupDelay-style merging are adapted (one undo
  per type) or skipped."* **This header is stale**: adjacency coalescing was
  added later, and `tests/cm_history.rs:46` now relies on it.
- `tests/cm_history.rs:143` — *"Use type_at(..., 0) to alternate insertion sites
  and avoid coalescing."* Visibly working around the adjacency-only coalescer.
- `tests/cm_history.rs:115-119` — *"For this MVP, pure selection updates aren't
  part of history; we just bypass with apply."*
- `tests/cm_history.rs:237` — *"Use isolated apply so each char is its own event
  regardless of joining."* Bypassing coalescing for the deep chain test.

### Upstream tests not ported

`test-history.ts` has 31 `history` + 11 `undoSelection` + 3 `effects` + 1 `JSON`
= **46 total**. Our `cm_history.rs` has **16 tests**. Unmatched in our port:

- `starts a new event when newGroupDelay elapses` — the canonical time-gap test;
  impossible without a clock.
- `supports a custom join predicate` — no hook exists.
- `allows changes that aren't part of the history` — needs `addToHistory: false`.
- `doesn't merge document changes if there's a selection change in between` —
  the cursor-move break; not implemented.
- `accurately maps changes through each other` / `supports overlapping edits` /
  `supports overlapping unsynced deletes` / `supports non-tracked changes next
  to tracked changes` — depend on concurrent/rebased edits from other sources,
  which requires the annotation/extension system.
- `doesn't get confused by an undo not adding any redo item`.
- `can go back and forth through history when preserving items` — depends on
  `invertedEffects` / effect preservation.
- `rebases selection on undo` — depends on external-change rebasing.
- `truncates history` — we have no `minDepth` config and no history cap.
- `can group events around a non-history transaction` — needs `addToHistory: false`.
- `properly maps selections through non-history changes`.
- Entire `undoSelection` block (11 tests): ~3 ported. Missing: `merges
  selection-only transactions from keyboard`, `doesn't merge selection-only
  transactions from other sources`, `doesn't merge ... if they change the number
  of selections`, `doesn't merge ... if a selection changes empty state`,
  `allows to redo selection-only transactions`, `only changes selection`,
  `can undo a selection through remote changes`, `preserves text inserted inside
  a change`.
- Entire `effects` block (3 tests): no Effects → 0/3.
- Entire `JSON` block (1 test): no serialization → 0/1.

---

## Commands

### Current state

- `src/commands.rs` (2224 lines): `indent_more` / `indent_less` /
  `indent_selection` / `insert_newline_and_indent` / `insert_newline_keep_indent`
  / `delete_trailing_whitespace` / `delete_group_{forward,backward}` /
  `move_line_{up,down}` plus many cursor and selection commands not asserted in
  `cm_commands.rs`.
- `tests/cm_commands.rs`: 41 tests covering the command families above.

### CM6 behavior

- `indentSelection` uses the language's indent service (IndentContext /
  `indentService` facet / tree-sitter grammar) to compute smart indent; bracket
  heuristic is only the fallback.
- `cursorLineUp` / `cursorLineDown` track a **goal column** so moving through a
  short line and back preserves the intended column.
- Bidi-aware `cursor*Left/Right` variants pick the visual side per doc direction.
- `deleteCharBackwardStrict` differs from `deleteCharBackward` by treating
  UTF-16 surrogate pairs differently.
- Multi-unit line comments use the `languageData` facet to pick tokens per-region.

### Gaps

- **No language / tree-sitter hookup**: `indent_selection` is bracket-only.
  Two upstream tests are ported in a **weakened form that asserts the opposite
  of the upstream expectation**:
  - `tests/cm_commands.rs:113-117` — *"CM6 with javascriptLanguage indents this
    to 'if (0)\n  foo()|' because the JS grammar knows if (...) introduces a
    block. Our bracket-only rule sees no open bracket on the previous line, so
    no indent is added — the command is a no-op."* The test asserts `no_apply`.
  - `tests/cm_commands.rs:121-126` — same pattern.
- **No goal-column tracking** for `cursor_line_up/down`.
  `src/commands.rs:1155-1158` — *"Without goal-column tracking yet — column is
  just clamped to the target line's length."* No test in `cm_commands.rs`
  exercises vertical motion, so this gap is silent.
- **No bidi support**. `src/commands.rs:1289-1290` — *"CM6 has bidi-aware
  *Left/*Right variants that pick the visual side. Without bidi support we
  treat them as direct LTR aliases of left/right."*
- **`delete_char_backward_strict` is aliased to `delete_char_backward`**.
  `src/commands.rs:1263-1268` — *"Same as delete_char_backward — CM6's 'strict'
  variant only differs in how it handles UTF-16 surrogate pairs."*
- `indent_less` takes `indent_unit` as a character count, not CM6's column-aware
  dedent using `tabSize` (which we don't have).

### Evidence

- `src/commands.rs:1156-1158` — goal-column gap.
- `src/commands.rs:1263-1268` — strict-backspace alias.
- `src/commands.rs:1289-1290` — bidi aliasing.
- `src/commands.rs:2075-2078` — default-position-mapping is an approximation.
- `tests/cm_commands.rs:113-126` — two tests asserting inverted outcomes.

### Upstream tests not ported

Upstream: indentMore 4, indentLess 5, indentSelection 4, insertNewlineKeepIndent
3, insertNewlineAndIndent 14, deleteTrailingWhitespace 3, deleteGroupForward 9,
deleteGroupBackward 8, moveLineUp 5, moveLineDown 6 = **61 total**.

Ours: ≈63 tests — roughly 1:1 *count*, but:
- 2 tests assert inverted outcomes vs. upstream (counted as ports but prove the
  opposite behavior).
- No vertical `cursorLineUp/Down` tests.

---

## Comment / Toggle Comment

### Current state

- `src/comment.rs` (529 lines) implements `line_comment`, `line_uncomment`,
  `toggle_line_comment`, `block_comment`, `block_uncomment`,
  `toggle_block_comment`, `toggle_block_comment_by_line`. Tokens come from
  `state.comment_tokens: CommentTokens { line, block }`.

### CM6 behavior

- Comment tokens come from a **languageData** facet queried by document
  position, allowing different tokens per region (HTML with embedded JS/CSS).
- `toggleComment` chooses between line and block based on languageData hints.

### Gaps

- No language-data lookup — tokens are a single state-wide pair. Upstream's
  `toggles line comment in multi-language doc` test is **not portable** against
  our model.

### Evidence

- `tests/cm_comment.rs:1-2` — *"Ports of CodeMirror 6
  commands/test/test-comment.ts (line-comment subset). Block-comment +
  multi-language tests deferred (block needs language config)."* (Partially
  obsolete — block-comment IS tested now; only the multi-language test is
  genuinely skipped.)

### Upstream tests not ported

Upstream has 9 line-comment + 8 block-comment + 1 root-level = **18**. Our
`cm_comment.rs` has ~24, structured differently.

- Missing: `toggles line comment in multi-language doc`.
- Extra coverage (no upstream analogue): `one_way_line_comment_tests`,
  `one_way_block_comment_tests` — exercise our standalone functions that CM6
  doesn't expose as separate commands.

---

## EditorState / Transactions

### Current state

- `src/state.rs`: `EditorState { doc, selection, indent_unit, comment_tokens,
  indent_rules, history }`. Fixed-shape, no extension system. `apply` applies
  changes in reverse order and maps selection with CM6-style associativity.
- `src/transaction.rs`: `Transaction { changes, selection }`. `normalize` +
  overlap check, `compose` (via diff of start/end docs — drops intermediate
  granularity, acknowledged at `src/transaction.rs:93-94`), `invert`.
- `src/selection.rs`: `Selection::new` does the correct CM6-style merge
  (overlap + point-range touching), sorts, re-indexes the primary.
- No JSON/serialization. No `tabSize` / `lineSeparator` stored anywhere (only
  `indent_unit`, a string).

### CM6 behavior

- `EditorState` is built from `Extension`s (Facet, StateField, Compartment)
  supporting configuration, per-editor fields, reconfiguration, and compartment
  swaps.
- Transactions carry **annotations** (`Annotation`, e.g. `userEvent`,
  `addToHistory`, `isolateHistory`) and **effects** (`StateEffect<T>`) — both
  missing here.
- `changeFilter` and `transactionFilter` can veto/modify dispatched transactions;
  `transactionExtender` can append.
- `changeByRange` is the canonical way to build per-range changes from a map-fn.
- `tabSize` / `lineSeparator` are facet-controlled config.
- `state.toJSON()` / `EditorState.fromJSON` serialize fields opting in.

### Gaps

- No Facets, StateFields, Compartments, Extensions, Annotations, Effects,
  changeFilter, transactionFilter, transactionExtender, `changeByRange`,
  `tabSize`, `lineSeparator`, JSON.
- `Transaction::compose` is lossy (diffs end states) — CM6's `ChangeSet.compose`
  preserves the full mapping and is used by history for overlapping edits.
  `src/transaction.rs:93-94` — *"This implementation diffs the start and end
  docs and returns a single minimal Change. It loses the granularity of the
  intermediate changes but preserves the application result."*
- `Transaction::normalize` requires non-overlapping sibling changes
  (`src/transaction.rs:83`). CM6 merges overlaps via coordinate rebasing.

### Evidence

- `src/state.rs:9-17` — the `EditorState` struct is 6 fixed fields.
- `src/transaction.rs:52-56` — `Transaction` is `{ changes, selection }` (no
  annotations, no effects).
- `src/lib.rs` — module list contains no `facet`, `effect`, `annotation`,
  `compartment`, `extension`, `filter`.

### Major naming mismatch

`tests/cm_state.rs:1-2` — *"Ports of CodeMirror 6 state/test/test-selection.ts
and state/test/test-charcategory.ts."* Zero of the 27 upstream `test-state.ts`
`it`s are actually present in our file; the file name is misleading.

### Upstream tests not ported (test-state.ts)

**`EditorState` (17 tests)** — only a few meaningfully portable:
- `holds doc and selection properties` — trivially true, not written.
- `can apply changes` — covered transitively by `cm_text.rs`.
- `maps selection through changes` — covered transitively.
- `throws when a change's bounds are invalid` — our `Change::new` asserts
  `from <= to`; out-of-bounds isn't separately validated (needs manual review).
- **Blocked on extension system (most of the block)**: `can store annotations`,
  `stores and updates tab size`, `stores and updates the line separator`,
  `stores and updates fields`, `can be serialized to JSON`, `can preserve fields
  across reconfiguration`, `can replace extension groups`, `preserves
  compartments on reconfigure`, `forgets dropped compartments`, `allows facets
  computed from fields`, `blocks multiple selections when not allowed`.

**`changeByRange` (2 tests)** — no such API → 0/2.
**`changeFilter` (4 tests)** — no filter system → 0/4.
**`transactionFilter` (2 tests)** — no filter system → 0/2.
**`transactionExtender` (2 tests)** — no extender system → 0/2.

---

## Text (Doc) Model

### Current state

- We use `ropey::Rope` directly — no bespoke `Text` class.
- `cm_text.rs` exercises replace / append / delete-all / delete-at-ends /
  boundary insert / random-editing / slice / equality through
  `Transaction::apply`.

### Gaps

- Tree-shape tests (`creates a balanced tree when loading a document`,
  `rebalances on insert`, `collapses on delete`, `can be compared despite
  different tree shape`) are implementation-specific to CM6's tree. Legitimately
  skipped for ropey.
- Iteration tests (9 of them) test CM6's `Text.iter` API. We expose ropey's
  iteration directly — no wrapping tests to confirm consumers rely on the
  expected API surface. Needs manual review.
- **Line-info tests** (`can get line info by line number`, `can get line info
  by position`) — behaviorally load-bearing. We have open-coded logic in
  `src/commands.rs:144-158` (`last_real_line`) to work around ropey's
  trailing-newline-phantom-line. **Recommended port.**
- JSON (`can convert to JSON`) — no JSON.
- `preserves length` — trivially provable, not written.
- `can delete a range at the start of a child node` — unknown if covered by
  random test.

### Evidence

- `tests/cm_text.rs:1-7` — header acknowledges the tree-shape skip.
- `tests/cm_text.rs:59-61` — *"CM6 had a specific test for inserting on internal
  tree boundaries; for ropey we just verify mid-doc inserts compose normally."*

### Upstream tests not ported

Upstream: 29 `Text` tests. Ours: 9.

- Ported: replace, append, delete-all, delete-start, delete-end, boundary insert
  (weakened), repeated append, random edit (deterministic seed), slice.
- Skipped (legitimate — CM6-tree-specific): 4 tree-shape tests.
- Skipped (needs manual review — iteration semantics): 9 iteration tests.
- **Skipped (recommended port — line info)**: 2 line-info tests.
- Skipped (no JSON): 1 serialization test.
- Skipped (trivial/unclear): 4 tests.

---

## Cross-Cutting Gaps

These aren't tied to a single test file but matter for "claims to port CM6":

1. **No Extension / Facet / StateField / Compartment system.** The largest
   architectural gap. Surface symptom: anything upstream-configurable (tabSize,
   lineSeparator, language, allowMultipleSelections, etc.) is absent or
   hard-coded on `EditorState`.
2. **No Annotation or Effect system.** History can't tag events with `userEvent`
   or `addToHistory: false`; commands can't attach effects to transactions.
3. **No ChangeFilter / TransactionFilter / TransactionExtender.** Commands
   produce transactions and those transactions are applied directly; no
   interception layer.
4. **Lossy `Transaction::compose`.** Composes by diffing before/after docs,
   yielding a single minimal `Change`. CM6 composes `ChangeSet`s structurally,
   preserving the coordinate system needed for rebasing and undo against
   concurrent edits.
5. **No bidi / no UTF-16 surrogate semantics.** LTR-only.
6. **No goal-column tracking in vertical motion.**
7. **No language service / tree-sitter integration.** Affects `indent_selection`
   (bracket-only) and multi-language comment tokens.
8. **Stale header comments** in `tests/cm_history.rs` — claims no coalescing
   that we now have.

---

## Recommended Next Steps

In rough priority order — easier / higher-impact first:

1. **Fix stale `tests/cm_history.rs` header.** Trivial. Current claim is a lie.
2. **Rename `cm_state.rs` → `cm_selection.rs`** (or split). Its contents are
   selection + char-category, not state.
3. **Add time-based + cursor-move-break coalescing** to `src/history.rs` (adopt
   `newGroupDelay` with an injectable clock). Unlocks:
   `starts a new event when newGroupDelay elapses`, `doesn't merge document
   changes if there's a selection change in between`.
4. **Add a minimal annotation concept** (even just a
   `HashMap<TypeId, Box<dyn Any>>` on `Transaction`) and a `userEvent`
   annotation. Unlocks: paste-is-own-group, `addToHistory: false` expressivity.
5. **Port the two line-info tests from `test-text.ts`.** They're load-bearing
   for our `commands.rs` line arithmetic — the trailing-newline workaround at
   `src/commands.rs:144-158` deserves a regression net.
6. **Port `test-state.ts` properly** into a new `cm_editor_state.rs` covering
   the portable subset and explicitly labeling the 20+ extension-system tests
   as *blocked on Extension system*.
7. **Decide on `indent_selection` with no language**: either keep the inverted
   asserts (and re-document as *bracket-only behavior*, not *CM6 parity*), or
   feature-gate those tests behind a `language` cargo feature.
8. Deferred (architectural, no quick win): Extension system, ChangeSet
   composition, bidi, tree-sitter integration.
