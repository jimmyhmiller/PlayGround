#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
app="$repo/integration/e2e/.cache/calcom"

need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "missing required command: $1" >&2
    exit 1
  }
}

need cargo
need git
need node
need corepack
need docker
docker compose version >/dev/null 2>&1 || {
  echo "Docker Compose v2 is required (the 'docker compose' command)." >&2
  exit 1
}
docker info >/dev/null 2>&1 || {
  echo "Docker is installed but its daemon is not running." >&2
  exit 1
}

echo "== build diffpack release binary"
(cd "$repo" && cargo build --release)

echo "== fetch pinned cal.com and install its workspace dependencies"
echo "   This is the heavy corpus entry: roughly 3.4 GB and about 20 minutes on a cold machine."
(cd "$repo" && node integration/e2e/fetch.mjs --heavy next-calcom)

echo "== configure the local demo database"
(cd "$repo" && node demo/configure-calcom-env.mjs "$app")

echo "== start Postgres, apply migrations, and seed the demo users/event types"
(cd "$app" && corepack yarn workspace @calcom/prisma db-setup)

echo
echo "Setup complete. Run:"
echo "  cd $repo"
echo "  node demo/server.mjs"
echo
echo "Then open http://localhost:4321"
