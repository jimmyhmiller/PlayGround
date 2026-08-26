# quarry — usage (prototype)

One CLI, no server, no setup. Every command prints a JSON **envelope**:

```json
{ "status": "complete|partial|degraded|error",
  "confidence": "compiler|index|syntactic|textual",
  "coverage": {...}, "snapshot": {...},
  "elided": {"count": N, "cursor": "offset:N"},   // present iff truncated
  "notes": [...],                                  // honesty caveats — read them
  "result": {...} }                                // or "error": {"code": ...}
```

Rules of the road:
- `status: complete` means the result is exhaustive *at the stated confidence*.
  You may act on "0 refs + complete" **after checking `textual_mentions`** —
  string/comment/config mentions are outside the symbol graph and can be live
  dynamic access (getattr, registries); `refs` and `impact:delete` disclose
  them. You may NOT act destructively on `partial`.
- `confidence: syntactic` means name-matched: same-named symbols elsewhere may
  be included. Read `notes`.
- Errors are structured: `AMBIGUOUS` includes `candidates`; `ANCHOR_MISMATCH`
  includes `actual_line_text`; `UNSUPPORTED` may include `fallback`.
- If `elided` appears, there is more: re-run with `--cursor <cursor>`.

## Commands

All take `--root <workspace>` (the project root). Extra/complex params go in
`--json '<object>'` which merges over the flags.

```bash
quarry.py resolve  --root R --name parse_expression        # name -> symbols (fuzzy ok)
quarry.py describe --root R --name Config.timeout          # facts + counts
quarry.py refs     --root R --name Config.timeout --group-by file --context-lines 2
quarry.py refs     --root R --name X --role call           # roles: call|read|write|import|definition
quarry.py graph    --root R --name X --edge callers --depth 2   # edges: callers|callees|importers|imports
quarry.py search   --root R --regex "timeout" --path "src/**"   # L0 textual, any file type
quarry.py search   --root R --json '{"pattern":{"symbol":{"kind":"function","name_regex":"^legacy"}}}'
quarry.py check    --root R                                # whole project
quarry.py check    --root R --path src/app/util.py --plus-dependents
quarry.py impact   --root R --name X --json '{"change":{"kind":"signature","to":"def f(a, b): pass"}}'
quarry.py impact   --root R --name X --json '{"change":{"kind":"delete"}}'
quarry.py impact   --root R --name X --json '{"change":{"kind":"rename"}}'   # lists textual mentions a rename will NOT rewrite
quarry.py impact   --root R --json '{"change":{"kind":"patch","patch":{"format":"files","files":{"p.py":"<new content>"}}}}'  # new mypy diagnostics the patch would introduce
quarry.py type-of  --root R --json '{"at":{"path":"src/x.py","line":26,"anchor":"tok = self.peek_token()"},"expr":"self.peek_token()"}'  # mypy-inferred type of expr AFTER the anchored line runs (flow-sensitive)
quarry.py brief    --root R --name X                       # budgeted dossier
quarry.py edit.rename --root R --name Config.timeout --to request_timeout   # returns DIFF + conflicts
quarry.py edit.rename --root R --name X --to Y --mode apply                 # writes to disk
quarry.py overlay.create --root R --json '{"patch":{"format":"files","files":{"path.py":"<full new content>"}}}'
quarry.py overlay.create --root R --json '{"patch":{"format":"unified-diff","content":"--- a/...\n..."}}'
quarry.py <any-verb> --root R --overlay ovl:XXXX           # run any query in the hypothetical world
quarry.py batch    --root R --json '{"queries":[{"id":"q1","verb":"describe","name":"X"}, ...]}'
quarry.py capabilities --root R                            # what to trust here
quarry.py doctor   --root R                                # why, and remedies
```

Budgets: `--budget-items N` / `--budget-tokens N` on any query; truncation is
deterministic and reported in `elided`.

Position-based addressing (rarely needed): pass
`--json '{"at":{"path":"src/x.py","line":30,"anchor":"literal text on that line"}}'`
to symbol verbs. The anchor is verified; if the file drifted you get
`ANCHOR_MISMATCH` with the actual line, never a wrong answer.

Notable limits of this prototype (reported honestly by `capabilities`):
Python only above L0; `search:ast` is `UNSUPPORTED`; reference matching is
syntactic (name-based). With mypy installed, `check` runs at L3 (syntax +
name + attribute + type errors) and `type-of`/`impact:patch` are available;
without it, `check` degrades and says so.

Conformance: `python3 conformance.py --cmd "python3 /path/to/quarry.py"`
runs the protocol's black-box conformance suite (41 checks) against any
implementation.
