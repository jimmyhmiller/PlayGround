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

BIN="target/$PROFILE/terminal"
if [ ! -x "$BIN" ]; then
    echo "[dev-restart] $BIN not found after build" >&2
    exit 1
fi

# libghostty-vt-sys writes its dylib under a build-script-hashed dir.
# The hash can change between builds; resolve it freshly each time.
DYLIB_DIR=$(find "target/$PROFILE/build" -type d -path '*libghostty-vt-sys*/out/ghostty-install/lib' 2>/dev/null | head -1)
if [ -z "$DYLIB_DIR" ]; then
    echo "[dev-restart] couldn't find ghostty dylib install dir" >&2
    exit 1
fi

# Kill any existing terminal-bevy GUI. Match BOTH profiles so a
# release-built GUI from a prior run gets cleaned up too, and accept
# both absolute and relative paths (the user might have launched via
# `./target/release/terminal`). Exclude:
#   - `terminal-daemon` binary (separate path; survives by design)
#   - any `terminal --daemon ...` invocation (the daemon-mode subprocess)
ABS_BIN="$(pwd)/$BIN"
KILL=$(ps -ax -o pid,command \
    | awk '$0 ~ /target\/(debug|release)\/terminal($|[[:space:]])/ \
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
# DO NOT pipe through nohup — macOS SIP strips DYLD_* env vars when
# execing SIP-protected binaries (/usr/bin/nohup is one), and the GUI
# needs DYLD_FALLBACK_LIBRARY_PATH to find libghostty-vt.dylib. Use
# `& disown` instead so the child detaches cleanly without that hop.
DYLD_FALLBACK_LIBRARY_PATH="$DYLIB_DIR" \
    "$ABS_BIN" $GUI_ARGS </dev/null >"$LOG" 2>&1 &
NEW_PID=$!
disown $NEW_PID 2>/dev/null || true
echo "[dev-restart] started PID $NEW_PID"
