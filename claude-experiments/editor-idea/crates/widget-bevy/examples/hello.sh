#!/bin/sh
# Minimal widget — prints one frame and exits. Demonstrates the basic
# protocol shape and exercises every "leaf" element kind. Each line is
# one NDJSON message.

printf '%s\n' \
'{"type":"title","value":"Hello, widget"}' \
'{"type":"frame","root":{"type":"vstack","gap":6,"pad":12,"children":[{"type":"text","value":"Hello from a subprocess","size":15,"weight":"bold"},{"type":"divider"},{"type":"text","value":"This whole pane is produced by an external process.","color":"#aaa"},{"type":"hstack","gap":8,"children":[{"type":"badge","value":"v0","color":"#3a7a3a"},{"type":"text","value":"shell script, NDJSON over stdout, exits."}]},{"type":"spacer","size":4},{"type":"hstack","gap":6,"children":[{"type":"button","id":"hi","label":"Click me"},{"type":"link","url":"https://github.com","label":"open github.com"}]}]}}'
