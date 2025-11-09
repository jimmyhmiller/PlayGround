#!/bin/sh
# Workaround for Zig issue #24349: Remove duplicate rpaths before running test
set -e

TEST_BIN="$1"
shift

# Fix duplicate rpaths on macOS
if [ "$(uname)" = "Darwin" ]; then
  all_rpaths=$(otool -l "$TEST_BIN" 2>/dev/null | grep -A 2 LC_RPATH | grep path | awk '{print $2}')
  for rpath in $all_rpaths; do
    count=$(echo "$all_rpaths" | grep -c "^$rpath$" || true)
    if [ "$count" -gt 1 ]; then
      while [ "$count" -gt 1 ]; do
        install_name_tool -delete_rpath "$rpath" "$TEST_BIN" 2>/dev/null || true
        count=$((count - 1))
      done
    fi
  done
fi

# Run the test with remaining arguments
exec "$TEST_BIN" "$@"
