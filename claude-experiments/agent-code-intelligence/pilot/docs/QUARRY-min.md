# Assigned tool: `quarry`

Snapshot-based code-facts CLI. Every response is a JSON envelope with
status/confidence/notes fields.

    Q="python3 /home/user/PlayGround/claude-experiments/agent-code-intelligence/prototype/quarry.py"
    $Q resolve  --root WS --name NAME
    $Q describe --root WS --name NAME
    $Q refs     --root WS --name NAME [--role call|read|write|import] [--group-by file]
    $Q graph    --root WS --name NAME --edge callers|callees|importers [--depth N]
    $Q search   --root WS --regex PATTERN
    $Q check    --root WS [--path FILE --plus-dependents]
    $Q type-of  --root WS --json '{"at":{"path":P,"line":N,"anchor":TEXT},"expr":E}'
    $Q impact   --root WS --name NAME --json '{"change":{"kind":"delete"}}'
    $Q edit.rename --root WS --name NAME --to NEW [--mode apply]
    $Q brief    --root WS --name NAME
    $Q graph    --root WS --name A --edge path --to B   # is B reachable from A? returns the call chain
Member refs are receiver-partitioned: each occurrence is marked
receiver "proven"|"unknown"; provably-other-class occurrences are
excluded and counted in result.receivers.
