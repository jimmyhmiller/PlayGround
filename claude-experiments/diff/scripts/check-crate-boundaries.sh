#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

metadata=$(cargo metadata --no-deps --format-version 1)

python3 -c '
import json
import sys

packages = json.load(sys.stdin)["packages"]
workspace = {package["name"] for package in packages}
allowed = {
    "diffpack-core": set(),
    "diffpack-default-loader": {"diffpack-core"},
    "diffpack-vite-compat": {"diffpack-core", "diffpack-default-loader"},
    "diffpack-web": {
        "diffpack-core",
        "diffpack-default-loader",
        "diffpack-vite-compat",
    },
    "diffpack-next": {
        "diffpack-core",
        "diffpack-default-loader",
        "diffpack-vite-compat",
        "diffpack-web",
    },
    "diffpack-tanstack": {
        "diffpack-core",
        "diffpack-default-loader",
        "diffpack-vite-compat",
        "diffpack-web",
    },
}

violations = []
for package in packages:
    name = package["name"]
    if name not in allowed:
        continue
    workspace_dependencies = {
        dependency["name"]
        for dependency in package["dependencies"]
        if dependency["name"] in workspace
    }
    forbidden = workspace_dependencies - allowed[name]
    if forbidden:
        violations.append(
            f"{name} has upward or sibling workspace dependencies: "
            + ", ".join(sorted(forbidden))
        )

if violations:
    print("crate boundary violations:", file=sys.stderr)
    for violation in violations:
        print(f"  - {violation}", file=sys.stderr)
    sys.exit(1)
' <<<"$metadata"

if rg -n '(^|[[:space:]])diffpack[[:space:]]*=' crates/*/Cargo.toml; then
  echo "integration crates must not depend on the root diffpack package" >&2
  exit 1
fi

if rg -n 'pub use diffpack::' crates; then
  echo "workspace crates must not re-export through the root diffpack package" >&2
  exit 1
fi

echo "crate dependency boundaries are valid"
