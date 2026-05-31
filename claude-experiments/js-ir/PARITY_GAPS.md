# test262 parity gaps (converter TODO)

Snapshot from `cargo run --release --example parity -p jsir-swc` on the
regenerated corpus (48,711 upstream-ok entries):

```
            was (start of session)   now
match           47791  (98.1%)    48199  (98.9%)
differ_offset      13  (0.0%)        13  (0.0%)
differ_struct     421  (0.9%)       288  (0.6%)
ours_fail         486  (1.0%)       211  (0.4%)
```

## Fixed this session (+408 files)

- **`super(...)` calls + `super.x` member (read/write/method/computed)** — was
  `CallExpression: missing field callee` (174) plus super-member in the
  `unsupported expression` / `missing field left` buckets. Added `Ref::Super` /
  `Ref::SuperProp` in `from_swc.rs` and `SuperPropExpr` round-trip in
  `to_swc.rs`. **+273 files.**
- **Array-pattern trailing elisions** — swc drops trailing holes in *pattern*
  position (`[a,,]`→`[Some(a)]`, `[,]`→`[]`); Babel keeps them. Reconstructed by
  counting commas after the last parsed element (`array_pat_elements`). **+60.**
- **BigInt `value`/`rawValue`** — upstream stores the raw literal minus the `n`
  suffix and minus numeric separators (`0x33333333`, `1000000` for
  `1_000_000n`), keeping the base prefix — not the decimal. **+75 net.**

Total non-matching: ~907 of 48,711 (1.9%), in clean self-contained clusters.
Each is a concrete, namable converter gap (same shape as the `export function`
bug that dropped its declaration), not a mystery. Listed largest-first within
each bucket.

## How these surface / why they lingered

The harness buckets failures and shows only the top-N signatures. We'd been
triaging the **scope** and **offset** buckets, so the `differ_struct` and
`ours_fail` long tails (127 distinct struct signatures, 13 fail reasons) went
un-inspected. Aggregate parity % + top-N triage hides real bugs. Each signature
below is worth draining to zero.

---

## A. `ours_fail` (211) — files we cannot convert at all

| count | what | cause |
|------:|------|-------|
| **128** | escaped reserved words as member/property names (`obj.break`) | **SWC parser limitation** — rejects with `ExpectedIdent`/`Unexpected`. Confirmed SWC's parser, not our config. Effectively external (would need an SWC patch or a different parser). |
| 28 | optional-chaining calls `a?.b()` | `OptionalCallExpression` not handled — clean self-contained feature to add next |
| 12 | `await` as binding / various | `unsupported expression` long tail |
| ~21 | other SWC parse errors | `Expected` (6), `TS1109` (5), `AsyncConstructor` (4), … — mostly SWC parser gaps |
| 4 | private name in optional chain | `identifier_attr: expected Identifier, got PrivateName` |
| 3 | bigint / computed property keys | `ObjectProperty`/`ClassProperty: missing field key` |

## B. `differ_struct` (288) — converts, but IR differs from upstream

| count | what | cause |
|------:|------|-------|
| ~53 | **module vs script misclassification** | test262 `flags: [module]` files with no `import`/`export` parsed as `script` by us. Needs the test262 flag (not in source) — **harness-level / partially external**; a general converter can only auto-detect (module iff import/export). |
| ~34 | `for (var … of/in …)` declaration shape | `for_in_of_declaration` attr: a sub-case where upstream and ours disagree on the symbol/loc field alignment. Common case (incl. destructuring multi-symbol) already matches; the differing files are a narrower variant — not yet pinned down. |
| ~33 | identifier attr missing **comment uids** | when comments sit inside a name's trivia (`async function /* a */ f /* b */ …`), upstream renders the leading/trailing comment uids in the `#jsir<identifier …>` attr; our `IdentifierAttr` carries no comment-uid trivia. Involved (comment attachment to attribute-position nodes). |
| ~12 | symbol attr `defScopeUid` (`local2`) | scope id not attached in one spot |
| misc | remaining long-tail signatures | ~100 smaller signatures, each a few files |

**Fixed since the original inventory:** array-pattern holes (~61), BigInt
`value`/`rawValue` (~38), and the `export function`/`class` declaration loss
(59) are resolved. The unicode-escaped-identifier cluster largely overlapped the
SWC parse-error bucket (escaped reserved words) and is external.

## C. `differ_offset` (13) — DEFERRED (per user)

All 13 are AnnexB HTML-comment-close (`-->`) / HTML-open (`<!--`) column
offsets (e.g. `single-line-html-close.js`, `comment-multi-line-html-close.js`).
A real but niche converter diff (column off by the length of `-->`). Not a
corpus artifact. Left intentionally.

---

## Suggested order (largest, cleanest first)

1. `super()` calls — 174 (biggest single win)
2. array pattern holes → `jsir.none` — ~61
3. module/script detection from test262 flags — ~53
4. BigInt `rawValue` for non-decimal/large — ~38
5. unicode-escape decoding in identifier names — ~35
6. for-in/of var declaration shape — ~34

## Recently fixed (for reference)

- `export function` / `export class` / `export default function`/`class` dropped
  their declaration (empty `export_named_declaration` region). Fixed in
  `from_swc.rs` via `Ref::Decl` / `Ref::DefaultDecl`. Recovered exactly 59
  corpus files (match 47732 → 47791).
- `\r\n` newline corpus artifact in `gen_oracle.py` (offset bucket 17 → 13).
