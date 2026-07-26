#!/usr/bin/env bash
# Real-application acceptance gate: build + serve + curl-smoke the vendored real
# open-source Next.js app-router apps under integration/next-real/ with the REAL diffpack.
# Proves "full support" against genuine third-party app code (vercel/next.js examples,
# MIT), not just the hand-authored fixture.
#
# Deps: each app reuses the fixture's pinned node_modules (identical next/react versions)
# via a symlink, so no network install is needed. Skips cleanly (exit 0) when node is
# absent. Any app that fails to build/serve fails the gate (exit 1) naming it.
set -u
here="$(cd "$(dirname "$0")/../.." && pwd)"
corpus="$here/integration/next-real"
fixture_nm="$here/integration/next-app-router/node_modules"
dp="$here/target/release/diffpack"

if ! command -v node >/dev/null 2>&1; then
  echo "SKIP next-real: node not found"
  exit 0
fi
if [ ! -x "$dp" ]; then
  echo "SKIP next-real: $dp not built (cargo build --release)"
  exit 0
fi
if [ ! -d "$fixture_nm/react" ]; then
  echo "SKIP next-real: fixture node_modules absent (run the fixture install first)"
  exit 0
fi

# app list: "name<TAB>buildEnv"
python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));[print(a["name"]+"\t"+a.get("buildEnv","production")) for a in d["apps"]]' "$corpus/apps.json" > /tmp/next-real-apps.tsv || { echo "next-real: cannot read apps.json"; exit 1; }
# smoke list: "name<TAB>path<TAB>status<TAB>body"
python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));[print("\t".join([a["name"],s["path"],str(s.get("expectStatus",200)),s.get("expectBody","")])) for a in d["apps"] for s in a.get("smoke",[])]' "$corpus/apps.json" > /tmp/next-real-smoke.tsv

fail=0
port=8990

# Build each app once (symlink the fixture's pinned deps).
while IFS=$'\t' read -r name env; do
  app="$corpus/$name"
  echo "== $name: build =="
  rm -rf "$app/node_modules"; ln -s "$fixture_nm" "$app/node_modules"
  rm -rf "$app/.diffpack-output" "$app/.diffpack-next"
  if (cd "$app" && "$dp" build-app . "$env" >/tmp/next-real-$name.log 2>&1); then
    echo "  built."
  else
    echo "FAIL $name: build-app $env failed"; tail -5 /tmp/next-real-$name.log; fail=1
  fi
done < /tmp/next-real-apps.tsv

# Serve + smoke each assertion (grouped by app: start once, curl each path).
prev=""
srv=""
while IFS=$'\t' read -r name path status body; do
  app="$corpus/$name"
  [ -d "$app/.diffpack-output" ] || continue
  if [ "$name" != "$prev" ]; then
    [ -n "$srv" ] && { kill "$srv" 2>/dev/null; pkill -9 -f "server.mjs serve|next-server" 2>/dev/null; }
    port=$((port + 1))
    (cd "$app" && "$dp" start .diffpack-output "$port" >/tmp/next-real-serve-$name.log 2>&1) &
    srv=$!
    for _ in $(seq 1 60); do curl -s -o /dev/null "http://127.0.0.1:$port/" 2>/dev/null && break; sleep 0.2; done
    prev="$name"
  fi
  code=$(curl -s -o /tmp/next-real-body -w "%{http_code}" "http://127.0.0.1:$port$path")
  if [ "$code" != "$status" ]; then
    echo "FAIL $name $path: status $code != $status"; fail=1
  elif [ -n "$body" ] && ! grep -qF "$body" /tmp/next-real-body; then
    echo "FAIL $name $path: body missing '$body'"; fail=1
  else
    echo "PASS $name $path ($code)"
  fi
done < /tmp/next-real-smoke.tsv
[ -n "$srv" ] && { kill "$srv" 2>/dev/null; pkill -9 -f "server.mjs serve|next-server" 2>/dev/null; }

if [ "$fail" = 0 ]; then echo "next-real: PASS (real OSS app-router apps build + serve under diffpack)"; fi
exit $fail
