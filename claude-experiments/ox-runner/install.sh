#!/usr/bin/env bash
# Install/update ox-runner on this machine. Idempotent: safe to re-run to deploy
# new versions of the scripts. Never touches $OX_HOME/env, config/ or run history.
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OX_HOME="${OX_HOME:-$HOME/ox-runner}"

echo "installing ox-runner into $OX_HOME"

mkdir -p "$OX_HOME"/{bin,inbox,processing,done,failed,runs,state,work,logs,config}
install -m 755 "$SRC/bin/ox" "$SRC/bin/ox-daemon" "$SRC/bin/ox-run" "$OX_HOME/bin/"
install -m 644 "$SRC/bin/ox-lib.sh" "$OX_HOME/bin/"
install -m 755 "$SRC/bin/ox-parse-task.py" "$SRC/bin/ox-render-log.py" "$SRC/bin/ox-format.py" "$SRC/bin/ox-meta.py" "$SRC/bin/ox-snapshot.py" "$OX_HOME/bin/"
[ -f "$SRC/README.md" ] && install -m 644 "$SRC/README.md" "$OX_HOME/README.md"
touch "$OX_HOME/config/repos.tsv" "$OX_HOME/runs.jsonl"

# --- codex provider + profile ------------------------------------------------
CODEX_CFG="$HOME/.codex/config.toml"
mkdir -p "$HOME/.codex"
touch "$CODEX_CFG"
if ! grep -q 'model_providers.openrouter' "$CODEX_CFG"; then
  cp "$CODEX_CFG" "$CODEX_CFG.bak-ox-$(date +%s)"
  cat >> "$CODEX_CFG" <<'EOF'

[model_providers.openrouter]
name = "OpenRouter"
base_url = "https://openrouter.ai/api/v1"
env_key = "OPENROUTER_API_KEY"
wire_api = "chat"

[profiles.ox]
model = "stealth/ox-alpha"
model_provider = "openrouter"
model_reasoning_effort = "high"
EOF
  echo "added openrouter provider + [profiles.ox] to $CODEX_CFG"
else
  echo "codex openrouter provider already configured"
fi

# --- api key -----------------------------------------------------------------
if [ ! -s "$OX_HOME/env" ]; then
  if [ -n "${OPENROUTER_API_KEY:-}" ]; then
    umask 077; printf 'OPENROUTER_API_KEY=%s\n' "$OPENROUTER_API_KEY" > "$OX_HOME/env"
    echo "wrote $OX_HOME/env from \$OPENROUTER_API_KEY"
  else
    echo "!! $OX_HOME/env is missing. Create it with:"
    echo "     umask 077; echo 'OPENROUTER_API_KEY=sk-or-...' > $OX_HOME/env"
  fi
fi
chmod 600 "$OX_HOME/env" 2>/dev/null || true

# --- systemd -----------------------------------------------------------------
mkdir -p "$HOME/.config/systemd/user"
install -m 644 "$SRC/ox-runner.service" "$HOME/.config/systemd/user/ox-runner.service"
systemctl --user daemon-reload

# Without linger the daemon dies when the last SSH session closes.
if ! loginctl show-user "$USER" 2>/dev/null | grep -q 'Linger=yes'; then
  if loginctl enable-linger "$USER" 2>/dev/null; then
    echo "enabled linger (daemon survives logout)"
  else
    echo "!! could not enable linger; run: sudo loginctl enable-linger $USER"
    echo "   otherwise the daemon stops when your last SSH session ends."
  fi
fi

systemctl --user enable ox-runner.service >/dev/null 2>&1 || true

# ox-run is exec'd fresh per task, so updated run logic takes effect on the
# next task with no restart. Only the daemon loop itself needs one - and
# restarting while a task is running would interrupt it.
INFLIGHT="$(ls -1 "$OX_HOME/processing"/*.md 2>/dev/null | wc -l | tr -d ' ')"
if [ "${INFLIGHT:-0}" -gt 0 ] && [ "${1:-}" != "--force-restart" ]; then
  echo
  echo "!! $INFLIGHT run(s) in flight - NOT restarting the daemon."
  echo "   New scripts are installed and apply to the next task automatically."
  echo "   To pick up daemon changes, run 'ox restart' once idle,"
  echo "   or re-run this installer with --force-restart."
else
  systemctl --user restart ox-runner.service
fi

# --- PATH convenience --------------------------------------------------------
mkdir -p "$HOME/.local/bin"
ln -sf "$OX_HOME/bin/ox" "$HOME/.local/bin/ox"

sleep 1
echo
"$OX_HOME/bin/ox" status 5 || true
echo
echo "installed. try:  ox doctor"
