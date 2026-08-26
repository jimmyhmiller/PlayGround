# Quarry — A Code Intelligence Protocol for Agents

**Version 0.1 (draft)**

Quarry is a protocol for asking questions about code and requesting mechanical
transformations of code, designed for consumption by autonomous coding agents rather than
interactive editors. It is the answer to the question: *if LSP had been designed in 2026
for a consumer that pays per token, has no cursor, edits the disk directly, runs in
parallel, and cannot debug its own tooling — what would it look like?*

Every design decision below is the inversion of a documented LSP failure mode for agents
(see [CRITIQUE.md](./CRITIQUE.md)).

---

## 1. Design principles

| # | Principle | Inverts (LSP failure) |
|---|-----------|----------------------|
| P1 | **Stateless requests over immutable snapshots.** No sessions, no handshake, no document sync. Every request is self-contained and idempotent. | Stateful sessions, `didOpen`/`didChange` mirroring (A.1, A.2) |
| P2 | **Names over positions.** The primary addressing mode is the symbol name. Positions are an optional convenience, never a requirement, and always anchored by text. | Cursor-position addressing, UTF-16 offsets (B.5–B.7) |
| P3 | **Facts over widgets.** Responses are schema'd, typed facts. Presentation strings (markdown docs) are separate optional fields, never the payload. | Hover-as-markdown, widget-shaped responses (C.10) |
| P4 | **Batch and compose.** One request may carry many questions; one verb may answer a compound question. Round trips are the scarce resource. | One-question-per-request ping-pong (C.8) |
| P5 | **Pull everything.** Nothing is pushed. Diagnostics, index status, progress — all are queries with complete, bounded answers. | Push diagnostics, debounce guessing (C.11) |
| P6 | **Honest epistemics.** Every response declares its completeness, freshness, and coverage. `"none"` and `"unknown"` are different values. It is a protocol violation to return an empty result without saying which one it is. | Silent partial results during warm-up (C.12, E.16) |
| P7 | **Budgets are first-class.** Every request may carry a token/size budget; servers rank, truncate deterministically, and report what was elided with a cursor to get more. | Token-hostile, unranked dumps (E.17) |
| P8 | **Transforms are transactions.** The server *performs* renames/refactors against a snapshot and returns a unified diff plus conflicts — it never hands back positional edit lists for the client to apply. | `WorkspaceEdit` positional patch lists (D.14) |
| P9 | **CLI-first, daemon-optional.** The canonical interface is a single executable speaking JSON on stdout. A daemon may exist purely as a cache; the contract is identical. Agents already know how to run CLIs. | JSON-RPC lifecycle babysitting (A.1, A.3) |
| P10 | **Degrade loudly, uniformly.** Capability is expressed as small, testable conformance *levels* per language, discoverable in one call — not hundreds of independent capability flags. | Capability-negotiation lowest-common-denominator (E.18) |

---

## 2. Data model

### 2.1 Snapshot

All queries execute against a **snapshot**: an immutable view of the workspace.

```json
{ "snapshot": { "vcs": "git", "rev": "8f3ab21", "dirty": ["src/parser.rs"], "id": "snap:8f3ab21+d41c" } }
```

- Default snapshot = current VCS revision + working-tree modifications ("what's on disk
  now"). The server computes `id` as a content hash; two agents on identical trees get
  identical ids and identical answers (cacheable).
- A request may pin `snapshot_id` to guarantee two related queries see the same world.
  Servers keep recent snapshots available on a best-effort basis and return
  `SNAPSHOT_GONE` (never wrong answers) when one has expired.
- There is **no document synchronization**. If the agent edits a file, the next default
  snapshot simply includes it. Nothing to mirror, nothing to desync.

### 2.2 Overlay (hypothetical worlds)

An **overlay** is a snapshot plus a patch that has not been written to disk:

```json
{ "verb": "overlay.create", "base": "snap:8f3ab21+d41c",
  "patch": { "format": "unified-diff", "content": "--- a/src/parser.rs\n+++ b/src/parser.rs\n@@ ..." } }
→ { "overlay_id": "ovl:71c2", "status": "complete" }
```

Every query verb accepts `"overlay": "ovl:71c2"`. This gives agents *speculative
execution*: "if I applied this patch, what would break?" — without touching disk, and
without racing other agents. Overlays are cheap, immutable, and garbage-collected;
parallel subagents each hold their own.

### 2.3 Locator

Where a position is needed at all (e.g. "what is the type of the expression here"),
Quarry uses a **locator** — self-verifying, never encoding-ambiguous:

```json
{ "path": "src/parser.rs", "line": 142, "anchor": "let tok = self.peek_token()", "occurrence": 1 }
```

- `line` is 1-based. `anchor` is a literal substring of that line (or of a small window
  around it, `window: 3`); the server verifies the anchor before answering and returns
  `ANCHOR_MISMATCH` with the actual line text if the file has drifted. **A stale locator
  produces an error, never a wrong answer.**
- Optional `col` is a **byte offset** within the line, UTF-8. UTF-16 does not appear in
  this protocol.
- Responses locate results the same way: `path`, `line`, and the line's text. Agents can
  splice results into their context and verify them against files they've read.

### 2.4 Symbol

Symbols carry stable, global, human-readable IDs (SCIP-style monikers):

```json
{
  "sym": "rust crates.io/serde 1.0.219 de/Deserialize#deserialize().",
  "display": "Deserialize::deserialize",
  "kind": "method",
  "lang": "rust",
  "def": { "path": "~/.cargo/.../de/mod.rs", "line": 531, "text": "fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>" },
  "visibility": "public",
  "external": true
}
```

- `sym` is deterministic across snapshots (rename produces a new `sym`), so agents can
  hold symbol references across turns without staleness.
- Local/anonymous symbols get snapshot-scoped ids (`local:src/parser.rs:142:tok`).

### 2.5 The response envelope (mandatory epistemics)

Every response — every verb, no exceptions — is wrapped:

```json
{
  "status": "complete" | "partial" | "degraded" | "error",
  "confidence": "compiler" | "index" | "syntactic" | "textual",
  "index": { "state": "fresh" | "stale" | "building", "age_s": 4, "built_pct": 100 },
  "coverage": { "files_indexed": 1412, "files_excluded": 12, "files_errored": ["gen/proto.rs"] },
  "elided": { "count": 240, "cursor": "c:9f2" },
  "result": { ... }
}
```

Semantics:

- `status: "complete"` is a **promise**: the result is exhaustive for the question asked,
  at the stated confidence. An agent may act on "zero callers, complete, compiler" by
  deleting the function. An agent must not act that way on `partial` or `degraded` — and
  because the distinction is on every response, harnesses can enforce this mechanically.
- `confidence` names the ladder the answer came from: full semantic analysis
  (`compiler`), a possibly-stale prebuilt index (`index`), syntax-only resolution
  (`syntactic`, e.g. tree-sitter), or text matching (`textual`). A server that can't
  answer semantically **must** either degrade loudly (lower `confidence`, `status:
  "degraded"`) or refuse (`UNSUPPORTED`) — silently thin answers are a conformance
  failure.
- `elided` appears whenever a budget truncated anything, with an exact count and a resume
  cursor. Truncation is deterministic: same request + same snapshot ⇒ same kept set.

### 2.6 Errors

Errors are structured, enumerable, and actionable by a machine:

```json
{ "status": "error", "error": { "code": "INDEX_BUILDING", "retry_after_s": 20,
    "detail": "rust: 62% indexed", "fallback": "syntactic" } }
```

Reserved codes: `UNSUPPORTED` (this language/level doesn't do that — permanent),
`INDEX_BUILDING` (retry), `SNAPSHOT_GONE`, `ANCHOR_MISMATCH`, `AMBIGUOUS` (with the
candidate list), `BUDGET_TOO_SMALL`, `LANG_TOOLCHAIN_MISSING` (with install hint).
An agent — or its harness — can branch on every one of these without a human.

---

## 3. Transport

### 3.1 CLI (canonical)

```
quarry <verb> [flags] [--json request.json] < request.json > response.ndjson
```

- Single static binary. `quarry` multiplexes all languages in the workspace behind one
  interface; language adapters are its problem, not the agent's.
- Requests on stdin or flags; responses as JSON (NDJSON for streams). Exit code 0 =
  envelope delivered (even `status: "error"`); nonzero = transport failure only.
- The CLI transparently uses a warm daemon when one is running and cold-starts otherwise.
  **The daemon is a cache, not a session**: killing it can never change any answer, only
  latency. `quarry warm` starts it explicitly (e.g. from a harness's session-start hook).
- No handshake, no negotiation, no shutdown ritual. The first request may be slow or may
  return `INDEX_BUILDING` with `retry_after_s` — visible, bounded, and honest, instead of
  a hung pipe.

### 3.2 HTTP (optional, identical contract)

`POST /v0/{verb}` with the same JSON bodies, for shared central indexes (monorepo scale:
one index, many agents). Statelessness makes this trivial — there is nothing per-client
to hold.

### 3.3 Batch

Any transport accepts a batch, executed against one snapshot, with one shared budget:

```json
{ "snapshot": "auto", "budget": { "tokens": 4000 },
  "queries": [
    { "id": "q1", "verb": "describe", "name": "Parser.parse_expression" },
    { "id": "q2", "verb": "refs", "name": "Parser.parse_expression", "role": "call", "context_lines": 2 },
    { "id": "q3", "verb": "graph", "name": "Parser.parse_expression", "edge": "callers", "depth": 2 }
  ] }
```

One round trip, one consistent world, one budget shared by declared priority order.

---

## 4. Query verbs

All verbs accept: `snapshot`/`overlay`, `budget`, `lang` filter, and path
include/exclude globs. All responses use the §2.5 envelope. Symbol-taking verbs accept
`name` (qualified, suffix, or fuzzy match — server resolves, returning `AMBIGUOUS` with
candidates when needed), `sym` (exact id), or `at` (locator) — in that order of
preference.

### 4.1 `resolve` — name → symbols

```json
{ "verb": "resolve", "name": "parse_expression", "kinds": ["function", "method"] }
→ { "result": { "symbols": [ {sym...}, {sym...} ] } }
```

Fuzzy by default (`match: "exact" | "suffix" | "fuzzy"`). This is the front door: agents
know names from reading code, conversation, and stack traces — not positions.

### 4.2 `describe` — everything about a symbol

```json
{ "verb": "describe", "name": "Parser.parse_expression" }
→ { "result": {
    "sym": "...", "kind": "method",
    "signature": "fn parse_expression(&mut self, min_bp: u8) -> Result<Expr, ParseError>",
    "type": { "params": [{"name": "min_bp", "type": "u8"}], "returns": "Result<Expr, ParseError>" },
    "visibility": "pub(crate)",
    "attributes": ["#[inline]"],
    "deprecated": null,
    "doc": "Pratt-parses a single expression...",
    "def": { "path": "src/parser.rs", "line": 131, "end_line": 190 },
    "counts": { "refs": 14, "callers": 9, "overrides": 0, "tests_touching": 3 }
  } }
```

Typed fields, resolved types (inference already done — the agent never needs an LSP to
know what an inferred type is), and **counts** — cheap aggregate awareness so the agent
can decide what to query next without paying for the full lists.

### 4.3 `refs` — occurrences, filtered and grouped

```json
{ "verb": "refs", "name": "Config.timeout", "role": ["read", "write"],
  "group_by": "file", "context_lines": 2, "budget": { "items": 20 } }
→ { "result": { "total": 47, "groups": [
      { "path": "src/server.rs", "occurrences": [
        { "line": 88, "role": "write", "text": "config.timeout = Duration::from_secs(30);",
          "context": ["fn apply_defaults(config: &mut Config) {", "..."],
          "enclosing": "apply_defaults" } ] } ] } }
```

Roles: `definition, declaration, call, read, write, import, type-use, test`. Every
occurrence names its **enclosing symbol** — the reference list doubles as a shallow
caller list. `total` is always exact even when items are elided.

### 4.4 `graph` — relationship traversal

```json
{ "verb": "graph", "name": "TokenStream", "edge": "implementors" }
{ "verb": "graph", "name": "Parser.bump", "edge": "callers", "depth": 3, "until": ["main", "test"] }
```

Edges: `callers, callees, implementors, implements, subtypes, supertypes, overrides,
overridden-by, importers, imports, instantiates`. Returns a node/edge list with symbol
ids and locators; `depth` and `until` bound traversal. Workspace-wide by definition —
there is no "open file" scope in this protocol.

### 4.5 `search` — one verb for text, syntax, and symbols

```json
{ "verb": "search",
  "pattern": { "ast": "for $X in $ITER: $BODY", "lang": "python" },
  "where": { "enclosing_kind": "function", "path": "src/**" } }
```

Three pattern types, combinable with the same filters: `regex` (what grep does, with
structured output and the envelope), `ast` (structural, tree-sitter-style
metavariables), `symbol` (kind/name/visibility predicates). This meets agents where they
demonstrably live — grep — and gives the same tool a semantic gear, rather than making
semantic search a different, more awkward tool that the agent will skip.

### 4.6 `check` — pull diagnostics

```json
{ "verb": "check", "scope": { "paths": ["src/parser.rs"], "plus_dependents": true }, "max_wait_s": 30 }
→ { "status": "complete", "result": { "diagnostics": [
      { "severity": "error", "code": "E0308", "path": "src/lexer.rs", "line": 44,
        "message": "mismatched types: expected `u8`, found `usize`",
        "related": [ {"path": "src/parser.rs", "line": 131, "note": "expected due to this signature"} ],
        "fix": { "available": true, "safe": true, "title": "cast with `as u8`" } } ],
      "checked": { "files": 61 }, "clean": false } }
```

- **Request/response, complete, bounded.** `clean: true` under `status: "complete"` is a
  real promise — the edit-check loop ("edit → check → fix → check") becomes one
  deterministic tool call per cycle, which is the single highest-value loop in agent
  coding.
- `plus_dependents` checks the blast radius of the named files, not just the files.
- `max_wait_s` bounds analysis time; timeout returns `partial` with what finished —
  never silence, never a guess.
- Safe fixes can be applied server-side: `{ "verb": "check", "apply_safe_fixes": true }`
  returns a diff (§5 semantics).

### 4.7 `type-of` — arbitrary expression

```json
{ "verb": "type-of", "at": { "path": "src/parser.rs", "line": 142, "anchor": "self.peek_token()" } }
→ { "result": { "expr": "self.peek_token()", "type": "Option<&Token>",
    "expanded": "Option<&'a Token>", "coercions": [] } }
```

The one place a locator is genuinely the right addressing mode — and it's anchored, so
staleness errors instead of lying.

### 4.8 `impact` — what breaks if

```json
{ "verb": "impact", "name": "Parser.parse_expression",
  "change": { "kind": "signature", "to": "fn parse_expression(&mut self, min_bp: u8, ctx: &Ctx) -> Result<Expr, ParseError>" } }
→ { "result": { "breaks": [ { "path": "src/stmt.rs", "line": 77, "reason": "missing argument `ctx`", "enclosing": "parse_statement" } ],
    "count": 9, "tests_affected": ["parser::tests::exprs"] } }
```

Change kinds: `signature`, `rename`, `delete`, `move`, `visibility`, or an arbitrary
`patch` (implemented as overlay + `check` + diff of diagnostics). This is the question
agents ask before every non-trivial edit and currently answer by grepping and hoping.

### 4.9 `brief` — token-budgeted dossier

```json
{ "verb": "brief", "name": "Parser.parse_expression", "intent": "modify", "budget": { "tokens": 3000 } }
```

Returns a single packed briefing: definition with full body, `describe` facts, call
sites with context (ranked: tests and unusual call shapes first), types it depends on
(signatures only), and known invariants from doc comments — trimmed to budget with
elision counts. `intent: "modify" | "call" | "delete" | "understand"` changes the
ranking. This exists because the compound question is the *unit of agent thought*; making
it one call is the protocol's whole point (P4).

### 4.10 `capabilities` / `doctor`

```json
{ "verb": "capabilities" }
→ { "result": { "langs": { "rust": { "level": 3, "confidence": "compiler" },
                            "python": { "level": 2, "confidence": "index" },
                            "sql": { "level": 0, "confidence": "textual" } } } }
```

`doctor` additionally runs self-tests and reports *why* a language is degraded, with a
machine-readable remedy (`LANG_TOOLCHAIN_MISSING` + install command). One call tells the
agent exactly what it may trust, per §7's four levels — not 200 boolean flags.

---

## 5. Transform verbs

Transforms follow one rule: **the server computes and applies; the agent reviews a
diff.** No positional edit lists ever cross the wire toward the agent.

```json
{ "verb": "edit.rename", "sym": "...Parser#parse_expression().", "to": "parse_expr",
  "mode": "diff" }
→ { "status": "complete", "result": {
    "diff": "--- a/src/parser.rs\n+++ b/src/parser.rs\n@@ ...",
    "files_changed": 6, "occurrences_rewritten": 14,
    "conflicts": [ { "path": "docs/api.md", "line": 12, "reason": "textual mention in prose — not rewritten", "text": "call parse_expression with..." } ]
  } }
```

- `mode: "diff"` (default) returns a unified diff for the agent to inspect and apply with
  its own tools, or to hand to `overlay.create` for speculative checking.
  `mode: "apply"` writes to disk atomically (all files or none) and returns the diff as
  a receipt.
- `conflicts` lists everything the server saw but deliberately did not touch (dynamic
  references, string matches, prose) — the judgment calls surface to the agent instead of
  silently going either way.
- Standard transforms: `edit.rename`, `edit.move`, `edit.change-signature` (with
  per-call-site argument mapping), `edit.organize-imports`, `edit.apply-fixes` (from
  `check`), `edit.inline`, `edit.extract`.
- Every transform is validated internally (transform → `check` on overlay → report) so
  the response can include `"post_check": { "clean": true }`. A transform that would
  introduce diagnostics says so *before* anything touches disk.

---

## 6. Budgets, ranking, concurrency

**Budgets.** `budget: { tokens?: n, items?: n, bytes?: n }` on any request; batch-level
budgets are shared by query priority order. Servers estimate tokens with a declared,
versioned tokenizer approximation (`chars/4` at minimum). Over-budget results are
truncated by relevance rank, deterministically, with exact `elided.count` and a cursor.

**Ranking.** Where results are ranked (search, refs, brief), rank is deterministic and
its factors are declared in `capabilities` (e.g. "call-sites in tests first, then by
distinct call shape"). Two identical requests on one snapshot return identical order —
agents and their caches depend on replayability.

**Concurrency.** All query verbs are read-only and safe under arbitrary parallelism —
N subagents share one warm daemon and one index. Overlays isolate speculation per agent.
`edit.*` with `mode: "apply"` takes a workspace write lock and re-validates the base
snapshot first (`SNAPSHOT_GONE` if the tree moved underneath) — optimistic concurrency,
same as agents already handle with git.

---

## 7. Conformance levels

Per language, a server implements a **level**; each level is a small, testable contract,
and `capabilities` reports it. Agents (and harness authors) code against levels, not
feature matrices.

| Level | Name | Guarantees | Typical backing |
|-------|------|-----------|-----------------|
| **L0** | Textual | `search:regex`, file/line locators, envelope semantics | ripgrep |
| **L1** | Syntactic | + `search:ast`, symbol outlines, enclosing-symbol attribution, `resolve` (declared names) | tree-sitter |
| **L2** | Graph | + cross-file `refs`/`graph`/`describe` from a prebuilt index; `confidence: "index"`; freshness reported | SCIP-style index |
| **L3** | Semantic | + `check`, `type-of`, `impact`, `edit.*`, resolved types; `confidence: "compiler"` | compiler frontend |

Every level includes all envelope obligations — an L0 answer is a *trustworthy* textual
answer, which beats an untrustworthy semantic one. The levels also define the fallback
ladder: a server may answer an L3 question at L1 quality only by saying so
(`status: "degraded", confidence: "syntactic"`).

A public conformance suite (golden repos per language, per level, including
warm-up-honesty tests: querying during index build MUST yield `INDEX_BUILDING` or
`partial`, never a bare empty result) is part of the spec. "Implements Quarry L2 for
Python" is a checkable claim, which is the adoption story LSP never had for quality.

---

## 8. What Quarry deliberately omits

- **Completion, signature help, semantic tokens, inlay hints, folding, code lens** —
  presentation for humans typing. Out of scope entirely; this is why the surface is 14
  verbs instead of 40.
- **Document synchronization** — no buffers, no mirrors. The disk (plus overlays) is the
  world.
- **Push notifications of any kind** — nothing arrives unrequested.
- **Editor UI concepts** — no "show document", no progress toasts, no window/* namespace.
- **Long-lived negotiated sessions** — capability is static per snapshot
  (`capabilities`), not negotiated per connection.

---

## 9. Migration and coexistence

Quarry is designed to be *implementable today* as a shim:

- **L0/L1 for every language immediately**: ripgrep + tree-sitter, in the `quarry`
  binary itself. This alone — with honest envelopes and unified search — beats the
  grep-only status quo agents actually use.
- **L2 from existing indexes**: SCIP indexers already exist for the major languages;
  `quarry` mounts a SCIP index and serves `refs`/`graph`/`describe` from it, reporting
  `confidence: "index"` and real freshness.
- **L3 by wrapping compilers first, LSP servers second**: `tsc`, `cargo check`/rust-analyzer's
  library layer, `mypy`/pyright — batch-friendly frontends map naturally onto `check`,
  `type-of`, `impact`. Where only an LSP server exists, a Quarry adapter may drive one
  internally — hiding lifecycle, document sync, and position encoding behind the
  stateless contract, and translating LSP's silences into honest envelope states. The
  babysitting code gets written once, in the adapter, instead of in every harness.
- **For harnesses**: Quarry is one tool definition (`quarry` with a JSON arg), not a
  protocol client. The `brief`, `check`, and `impact` verbs are designed to slot directly
  into existing agent loops (explore → brief; edit → check; before-refactor → impact).

---

## 10. Worked example: one agent turn

Task: *"rename `Config.timeout` to `Config.request_timeout` and fix anything that
breaks."*

```text
1. quarry resolve --name Config.timeout
   → 1 symbol, sym "…Config#timeout." (complete, compiler)

2. quarry edit.rename --sym "…Config#timeout." --to request_timeout --mode diff
   → diff across 6 files, 47 occurrences; 2 conflicts:
     a YAML key "timeout:" in config/default.yml (textual, not rewritten),
     a docstring mention (not rewritten)
   → post_check: clean=false — 1 new error: serde rename attribute now mismatched

3. agent inspects diff + conflicts, decides YAML key must change too;
   quarry overlay.create --patch <diff + yaml fix + serde fix>
   → ovl:2a91

4. quarry check --overlay ovl:2a91 --scope all
   → complete, clean=true

5. agent applies the patch with its own edit tool; commits.
```

Five round trips, every answer carrying its own completeness guarantee, no positions
computed by the model, no server lifecycle visible, nothing pushed, nothing guessed.
Under LSP this is: spawn/init/negotiate, didOpen a file, find an occurrence, compute a
UTF-16 column, request rename, receive a positional WorkspaceEdit, apply it atomically
yourself, replay every edit as didChange, then *wait* to see what diagnostics drift in —
and at no point can anything tell you your reference list was complete.

---

## 11. Amendments from the first implementation (v0.1.1)

A prototype of this spec was built and evaluated by three agents that had not
seen the spec's development — one task-driven, one comprehension/trust-focused,
one adversarial (see `prototype/TESTING.md`). Their findings exposed places
where the v0.1 text was loose enough that a *conformant* implementation could
still commit the protocol's cardinal sin (a wrong answer under a confident
envelope). The following amendments are normative.

**A1. Locator binding (§2.3).** `occurrence` out of range — including `0`,
negative, or non-integer — MUST be an error naming the valid range, never
clamped. When `anchor` is present, the answered symbol's occurrence MUST
lexically overlap the anchor text; an anchor that binds no symbol is
`NOT_FOUND` (listing the occurrences on the line), never a fallback to
"whatever is on the line". Position-resolution that is ambiguous at the
server's confidence level (e.g. a name owned by several types, syntactically
indistinguishable) MUST return `AMBIGUOUS`, never the first candidate.

**A2. Resolution disclosure (§4).** Every symbol-addressed response MUST
disclose how the name resolved: `match: "exact" | "suffix" | "fuzzy"`. A lone
fuzzy hit MUST NOT be silently substituted under a completeness promise: the
server returns `NOT_FOUND` with a `nearest` list unless the request opted into
fuzzy matching. (Found in the wild: `refs evaluate` fuzzy-resolving to
`evaluate_RENAMED` and answering "0 refs, complete" while `evaluate(` call
sites sat on disk.)

**A3. The structural-blind-spot rule (§2.5).** `complete` MUST NOT be claimed
for a question class the stated confidence level structurally cannot see, and
known blind spots MUST be closed or disclosed per verb: constructor calls
count as references to the constructor; a class's instantiations count as
references to `__init__`-equivalents; dynamic access (reflection, `getattr`)
is named in `notes` where the language allows it. "Technically no
name-matches" is not honesty if the question the agent asked has answers the
level cannot represent.

**A4. Scope-narrowing disclosure (§4.6, generally).** When a server answers
less than the request asked — files it cannot check, filters it cannot apply,
parameters it cannot honor (a `depth` it ignored, a `path` filter it dropped)
— it MUST either error (`UNSUPPORTED`) or degrade loudly (`status:
"degraded"` + the skipped scope enumerated in `coverage`). `clean: true` is
forbidden when checked ⊂ requested. Silently ignoring a request parameter
while stamping `complete` is a conformance failure equal to a wrong answer.

**A5. Patch strictness (§2.2).** Any byte of a patch not consumed as a valid
header or hunk ⇒ `PATCH_FAILED` with the unconsumed text. An empty patch is
`PATCH_FAILED`, not an empty overlay. A malformed patch silently yielding a
no-op overlay lets `check --overlay` "verify" the unpatched world — the
worst possible speculative-execution failure.

**A6. Overlay lifecycle (§2.2, §5).** Overlays record their base snapshot id;
any query against an overlay whose base no longer matches the workspace MUST
return `SNAPSHOT_GONE`. `edit.* mode:"apply"` is illegal against an overlay
(the transform was computed in a world that is not the disk) and MUST be
refused.

**A7. Transform apply-gating (§5).** `mode: "apply"` MUST refuse to write
when the transform's own post-check found diagnostics (`TRANSFORM_UNSAFE`,
overridable only by an explicit `force: true`), MUST revalidate the base
snapshot before writing, and MUST reject target names that are invalid in the
language (keywords included). `applied`, `files_changed`, and the diff MUST
describe only writes that actually happened; a transform that rewrote nothing
is `partial`, never `complete`+`applied`.

**A8. Same-named collateral (§5).** At syntactic confidence, occurrences of
same-named symbols the server cannot distinguish from the target MUST be
enumerated as per-symbol `conflicts` entries, not covered by a generic note.

**A9. Errors keep their epistemics (§2.5, §2.6).** Error envelopes carry
`coverage` like any other response: `NOT_FOUND` for a symbol whose defining
file failed to index must say so, or it invites "the symbol doesn't exist".
Reserved codes gain `WORKSPACE_NOT_FOUND`, `BAD_REQUEST` (mandatory for
malformed request shapes — `INTERNAL` is reserved for server defects), and
`TRANSFORM_UNSAFE`.

**A10. Budget and cursor edges (§6).** `items: 0` is a valid budget (count
without content); negative or non-integer budgets are `BAD_REQUEST`; an
unrecognized or stale cursor is `BAD_REQUEST`, never silently offset zero
(silent restarts hand accumulating agents duplicates); `BUDGET_TOO_SMALL`
MUST be raised when a budget cannot fit a single item, with the minimum
stated. The batch wrapper is itself a response and carries the full §2.5
envelope; its status is the worst of its sub-results.

**A11. Transport stdin contract (§3.1).** The CLI MUST NOT block waiting for
stdin when request input is available elsewhere (flags/`--json`) or when the
pipe has no data ready — an agent harness's open-but-silent stdin pipe is the
normal case, and "no handshake, no hung pipe" applies to the CLI's own input
too.

## Appendix A: verb index

| Verb | Question |
|------|----------|
| `resolve` | what symbols match this name? |
| `describe` | tell me everything about this symbol |
| `refs` | where is it used, in what role, by whom? |
| `graph` | callers/callees/implementors/importers, transitively |
| `search` | find text / AST shapes / symbols matching predicates |
| `check` | is this scope free of errors? (and fix the safe ones) |
| `type-of` | what is the type of this expression? |
| `impact` | what breaks if I make this change? |
| `brief` | pack me a budgeted dossier for changing X |
| `overlay.create` | give me a hypothetical world with this patch |
| `edit.rename` / `edit.move` / `edit.change-signature` / … | do this transform, show me the diff and the judgment calls |
| `capabilities` / `doctor` | what can I trust here, and why not more? |
| `warm` | (operational) pre-build the cache |

## Appendix B: envelope JSON Schema (abridged)

```json
{
  "$id": "https://quarry.dev/schema/v0/envelope.json",
  "type": "object",
  "required": ["status", "confidence", "index"],
  "properties": {
    "status": { "enum": ["complete", "partial", "degraded", "error"] },
    "confidence": { "enum": ["compiler", "index", "syntactic", "textual"] },
    "index": {
      "type": "object",
      "required": ["state"],
      "properties": {
        "state": { "enum": ["fresh", "stale", "building"] },
        "age_s": { "type": "number" },
        "built_pct": { "type": "number" } } },
    "coverage": {
      "type": "object",
      "properties": {
        "files_indexed": { "type": "integer" },
        "files_excluded": { "type": "integer" },
        "files_errored": { "type": "array", "items": { "type": "string" } } } },
    "elided": {
      "type": "object",
      "required": ["count"],
      "properties": { "count": { "type": "integer" }, "cursor": { "type": "string" } } },
    "error": {
      "type": "object",
      "required": ["code"],
      "properties": {
        "code": { "type": "string" },
        "retry_after_s": { "type": "number" },
        "detail": { "type": "string" },
        "fallback": { "type": "string" } } },
    "result": {}
  }
}
```
