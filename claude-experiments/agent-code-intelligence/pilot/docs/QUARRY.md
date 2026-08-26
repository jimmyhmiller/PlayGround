# Your code-intelligence tool: `quarry`

A snapshot-based code-facts CLI. Every response is a JSON envelope with
`status` (complete/partial/degraded/error), `confidence` (compiler/
syntactic/textual), coverage, and `notes` — read the notes: they state
exactly what each answer does and does not cover. Reference matching is
name-based (syntactic): same-named symbols on other classes are included,
and responses disclose string/config mentions via `textual_mentions`.

    Q="python3 /home/user/PlayGround/claude-experiments/agent-code-intelligence/prototype/quarry.py"
    $Q resolve  --root WS --name process               # name -> symbols (AMBIGUOUS lists candidates)
    $Q describe --root WS --name Engine.process        # facts + ref/caller/test counts
    $Q refs     --root WS --name Engine.process --group-by file --context-lines 1
    $Q refs     --root WS --name combine --role call   # roles: call|read|write|import|definition
    $Q graph    --root WS --name Engine.process --edge callers --depth 2   # also: callees|importers|imports
    $Q search   --root WS --regex "combine"            # textual, all file types (yml, md included)
    $Q check    --root WS                              # mypy-backed: syntax/name/attribute/type errors
    $Q check    --root WS --path mill/writer.py --plus-dependents
    $Q type-of  --root WS --json '{"at":{"path":"mill/engine.py","line":30,"anchor":"cooked = apply_rules"},"expr":"cooked"}'
    $Q impact   --root WS --name combine --json '{"change":{"kind":"delete"}}'   # what breaks, incl. textual mentions
    $Q edit.rename --root WS --name Engine.process --to run_one              # dry run: diff + conflicts
    $Q edit.rename --root WS --name Engine.process --to run_one --mode apply # applies (refuses if post-check dirty)
    $Q brief    --root WS --name combine               # packed dossier: definition + call sites

Tips:
- `refs`/`rename` are name-matched: check the `conflicts`/`notes` for
  same-named symbols and textual mentions before acting.
- `check` runs mypy: it catches latent type/attribute errors on paths
  tests never execute.
- A `complete` + 0 result is trustworthy AFTER you check the disclosed
  `textual_mentions` (string/config/doc references are listed there).
- The tool re-reads disk every call: your edits are always visible.
