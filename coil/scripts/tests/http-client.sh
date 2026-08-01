#!/bin/sh
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
python3 "$repo_dir/tests/http_client_server.py" &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT HUP INT TERM
sleep 1
"$repo_dir/build/bin/coil" build "$repo_dir/tests/http_client_integration.coil" -o /tmp/coil-http-client-test
/tmp/coil-http-client-test
