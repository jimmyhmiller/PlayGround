# Assigned tool: `lspcli` (pyright language server)

CLI over a warm pyright language server. Positions are 1-based line/col.
First call may take a few seconds. Output is JSON.

    L="python3 /home/user/PlayGround/claude-experiments/agent-code-intelligence/pilot/lspcli.py"
    $L symbols     --root WS --query NAME
    $L definition  --root WS --file PATH --line N --col M
    $L references  --root WS --file PATH --line N --col M
    $L references  --root WS --symbol NAME
    $L hover       --root WS --file PATH --line N --col M
    $L diagnostics --root WS [--file PATH]
    $L rename      --root WS --file PATH --line N --col M --new-name X [--apply]
