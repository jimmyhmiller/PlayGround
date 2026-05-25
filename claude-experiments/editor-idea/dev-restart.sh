#!/bin/sh
# Build, kill the running GUI, relaunch — leaves daemon children alive
# (they persist across GUI restarts so terminal panes survive).
#
# Usage:
#   ./dev-restart.sh                 # debug build, no extra flags
#   ./dev-restart.sh --release       # release build (much faster runtime)
#   ./dev-restart.sh -- --some-arg   # pass --some-arg to the GUI binary

set -e
cd "$(dirname "$0")"

PROFILE=debug
CARGO_PROFILE_ARGS=""
GUI_ARGS=""

while [ $# -gt 0 ]; do
    case "$1" in
        --release)
            PROFILE=release
            CARGO_PROFILE_ARGS="--release"
            shift
            ;;
        --)
            shift
            GUI_ARGS="$*"
            break
            ;;
        *)
            echo "unknown flag: $1" >&2
            exit 2
            ;;
    esac
done

echo "[dev-restart] building ($PROFILE)..."
cargo build $CARGO_PROFILE_ARGS -p terminal_bevy --bin terminal --bin tbwidget --bin tbopen

# Refresh the .app bundle so it carries the freshly-built binary and
# libghostty-vt dylib (copied in, not symlinked into target/).
# LaunchServices identity stays stable across rebuilds because
# CFBundleIdentifier doesn't change.
./make-bundle.sh ${CARGO_PROFILE_ARGS:+--release}

# Launch via the bundle (not target/$PROFILE/terminal directly) so
# AppKit walks up to Contents/Info.plist and treats us as a bundled
# app: stable Dock tile, pin survival, proper icon.
BIN="TerminalBevy.app/Contents/MacOS/terminal"
if [ ! -x "$BIN" ]; then
    echo "[dev-restart] $BIN not found (bundle build failed?)" >&2
    exit 1
fi

# Kill any existing terminal-bevy GUI. Match BOTH profiles so a
# release-built GUI from a prior run gets cleaned up too, and accept
# both the bundle path (current launch route) and bare target/ paths
# (older runs predating the .app wrapper). Exclude:
#   - `terminal-daemon` binary (separate path; survives by design)
#   - any `terminal --daemon ...` invocation (the daemon-mode subprocess)
ABS_BIN="$(pwd)/$BIN"
KILL=$(ps -ax -o pid,command \
    | awk '($0 ~ /TerminalBevy\.app\/Contents\/MacOS\/terminal($|[[:space:]])/ \
            || $0 ~ /target\/(debug|release)\/terminal($|[[:space:]])/) \
           && $0 !~ /--daemon/ \
           && $0 !~ /terminal-daemon/ { print $1 }')
if [ -n "$KILL" ]; then
    echo "[dev-restart] killing existing GUI(s): $KILL"
    kill $KILL 2>/dev/null || true
    # Give them a beat to release the socket before the new instance binds.
    sleep 0.4
fi

LOG=/tmp/terminal-bevy-${PROFILE}.log
echo "[dev-restart] launching → $LOG"
# The dylib lives inside the bundle (Contents/Frameworks) and the
# binary's rpath was set to @executable_path/../Frameworks by
# make-bundle.sh, so no DYLD_* env vars are needed. `& disown`
# detaches the child cleanly without going through nohup.
"$ABS_BIN" $GUI_ARGS </dev/null >"$LOG" 2>&1 &
NEW_PID=$!
disown $NEW_PID 2>/dev/null || true
echo "[dev-restart] started PID $NEW_PID"
