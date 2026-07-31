#!/usr/bin/env bash
# Two regression gates over the same fixture, both about a dependency that is not there.
#
# A. diffpack must NOT require a dependency no real app has.
#    Its next app-router adapter scaffolds entries that import
#    `react-server-dom-webpack/{client,server}`. No real Next.js app depends on that
#    package — Next never asks for one, it vendors its own copy at
#    `next/dist/compiled/react-server-dom-webpack` — so requiring it made a stock
#    `create-next-app` unbuildable until the user installed something they had no
#    reason to know about. Phase A builds the real fixture with every dependency EXCEPT
#    react-server-dom-webpack and asserts the build SUCCEEDS, off Next's vendored copy.
#
# B. An import that resolves to NOTHING must still fail `build-app` loudly.
#    That was the original subject of this gate (the bundler used to record it as a
#    non-fatal "known gap", exit 0, and still write a `public/client.js` carrying a
#    dangling `require(...)`; the page then died in the browser with the wrong message,
#    "node builtin ... is not available in the browser"). With (A) fixed, the RSC
#    runtime is no longer an example of an unresolvable import, so phase B injects a
#    package that genuinely does not exist and asserts the same invariant: non-zero
#    exit, the package + the importing file + the install command named, and NO output.
#
# Native build (Rust); node is needed only to evaluate the fixture's next.config.
# Strict mode, the ERR net (no abort is ever silent) and fail() — see _gate-prelude.sh.
source "$(dirname "$0")/_gate-prelude.sh"

repo="$(cd "$(dirname "$0")/../.." && pwd)"
fixture="${1:-$repo/integration/next-app-router}"
diffpack="$repo/target/release/diffpack"

[ -x "$diffpack" ] || fail "no release binary at $diffpack (cargo build --release)"
[ -d "$fixture/node_modules" ] || fail "$fixture/node_modules missing (npm install there)"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
app="$work/app"
mkdir -p "$app/node_modules"

# Copy the project sources, then symlink every installed dependency except the
# one under test, so the graph is otherwise exactly the real fixture's.
for entry in "$fixture"/* "$fixture"/.[!.]*; do
  [ -e "$entry" ] || continue
  case "$(basename "$entry")" in
    node_modules|.diffpack-next|.diffpack-next-pages|.diffpack-output|.next|dist) continue ;;
  esac
  cp -R "$entry" "$app/"
done
for dep in "$fixture"/node_modules/* "$fixture"/node_modules/.[!.]*; do
  [ -e "$dep" ] || continue
  name="$(basename "$dep")"
  if [ "$name" = "react-server-dom-webpack" ]; then continue; fi
  ln -s "$dep" "$app/node_modules/$name"
done
[ ! -e "$app/node_modules/react-server-dom-webpack" ] \
  || fail "fixture setup: react-server-dom-webpack should be absent"
[ -d "$app/node_modules/next/dist/compiled/react-server-dom-webpack" ] \
  || fail "fixture setup: the installed next does not vendor react-server-dom-webpack"

# --- A. the app-router build works with NO react-server-dom-webpack installed -------
log="$work/build.log"
set +e
"$diffpack" build-app "$app" client >"$log" 2>&1
status=$?
set -e

echo "--- build-app output (no react-server-dom-webpack installed) ---"
cat "$log"
echo "--- exit status: $status ---"

[ "$status" -eq 0 ] \
  || fail "build-app requires react-server-dom-webpack; it must use the copy next vendors"
[ -e "$app/.diffpack-output/public/client.js" ] \
  || fail "no client.js written by a build that reported success"
# The browser half must be the BROWSER build of the vendored runtime: `client.node.js`
# in a browser bundle would drag node built-ins in and die at runtime.
grep -q "__webpack_chunk_load__" "$app/.diffpack-output/public/client.js" \
  || fail "the client bundle does not carry the flight runtime's webpack seam"

# --- B. an import that resolves to nothing is still fatal ---------------------------
missing="diffpack-no-such-package-9f3c"
# A `"use client"` island, so the bad specifier is in the CLIENT graph this phase
# builds (a Server Component is compiled into the react-server graph instead).
importer="$app/app/Counter.tsx"
[ -f "$importer" ] || fail "fixture setup: expected a \"use client\" island at $importer"
cat >> "$importer" <<TSX

// gate: an import that resolves to NOTHING must fail the build.
import "$missing";
TSX

rm -rf "$app/.diffpack-output"
log="$work/unresolved.log"
set +e
"$diffpack" build-app "$app" client >"$log" 2>&1
status=$?
set -e

echo "--- build-app output (unresolvable import) ---"
cat "$log"
echo "--- exit status: $status ---"

[ "$status" -ne 0 ] || fail "build-app exited 0 with an unresolvable import"
grep -q "$missing" "$log" || fail "the missing package is not named"
grep -q "npm install $missing" "$log" || fail "no install remedy in the message"
grep -q "$importer" "$log" || fail "the importing file is not named"
if grep -q "known gap" "$log"; then fail "an unresolved import is still reported as a 'known gap'"; fi
if grep -q "is not available in the browser" "$log"; then
  fail "still emitting the misleading node-builtin message"
fi
[ ! -e "$app/.diffpack-output/public/client.js" ] \
  || fail "a broken client.js was written despite the unresolved import"

echo "PASS: the RSC runtime comes from next's vendored copy, and an unresolved import is still fatal"
