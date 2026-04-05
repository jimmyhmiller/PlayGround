#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LUA_DIR="$SCRIPT_DIR/lua-5.1.5"

if [ -x "$LUA_DIR/src/luac" ]; then
    echo "lua-5.1.5 already built at $LUA_DIR"
    exit 0
fi

echo "Downloading Lua 5.1.5..."
curl -sL -o /tmp/lua-5.1.5.tar.gz https://www.lua.org/ftp/lua-5.1.5.tar.gz
tar xzf /tmp/lua-5.1.5.tar.gz -C "$SCRIPT_DIR"
rm /tmp/lua-5.1.5.tar.gz

echo "Building Lua 5.1.5..."
case "$(uname)" in
    Darwin) make -C "$LUA_DIR" macosx ;;
    Linux)  make -C "$LUA_DIR" linux  ;;
    *)      make -C "$LUA_DIR" posix  ;;
esac

echo "Done: $LUA_DIR/src/luac"
