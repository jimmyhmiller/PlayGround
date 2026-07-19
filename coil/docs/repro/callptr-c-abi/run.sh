#!/bin/sh
# Exits 0 when the C ABI is honoured through call-ptr, 1 while the bug is live.
set -e
cd "$(dirname "$0")"
coil build repro.coil --lib -o librepro.a
cc host.c librepro.a -o repro
./repro
