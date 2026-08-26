# Your code-intelligence tool: `lspcli` (pyright language server)

A warm pyright language server behind a CLI. Semantic and receiver-aware:
references to `Engine.process` will NOT include a same-named method on
another class. All positions are 1-based line and column. First call may
take a few seconds (server warm-up); later calls are fast. Output is JSON.

    L="python3 /home/user/PlayGround/claude-experiments/agent-code-intelligence/pilot/lspcli.py"
    $L symbols     --root WS --query process          # find symbols by name -> file/line/col + kind
    $L definition  --root WS --file mill/engine.py --line 14 --col 20
    $L references  --root WS --file mill/engine.py --line 25 --col 9   # ALL refs incl. declaration
    $L references  --root WS --symbol combine          # name addressing; errors with candidates if ambiguous
    $L hover       --root WS --file mill/engine.py --line 25 --col 9   # signature + type of what's there
    $L diagnostics --root WS                           # full-project type/name/attr errors (pyright)
    $L diagnostics --root WS --file mill/writer.py
    $L rename      --root WS --file mill/engine.py --line 25 --col 9 --new-name run_one          # dry run: edit list
    $L rename      --root WS --symbol combine --new-name merge_records --apply                    # applies to disk

Tips:
- To operate on a symbol you know by name, start with `symbols` (or pass
  `--symbol NAME` to definition/references/hover/rename directly).
- `references` is exact (compiler-grade receiver resolution) but covers
  code only — it cannot see string/config/doc mentions.
- `diagnostics` = pyright at basic strictness: syntax, name, attribute,
  and type errors, including latent paths tests never execute.
- The tool re-reads files from disk on every call: your edits are always
  visible, nothing to sync.
