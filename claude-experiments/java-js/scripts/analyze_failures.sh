#!/bin/bash

cd test-oracles/test262/test

echo "Analyzing cache files for statement types..."

throw_count=0
try_count=0
switch_count=0
spread_array_count=0
total_cached=0

find . -name "*.js" | grep -v "/module/" | grep -v "/async-" | grep -v "/await/" | grep -v "/class/" | while read f; do
  cache="../../test262-cache/${f}.json"
  if [ -f "$cache" ]; then
    total_cached=$((total_cached + 1))

    if grep -q '"type":"ThrowStatement"' "$cache" 2>/dev/null; then
      throw_count=$((throw_count + 1))
      if [ $throw_count -le 5 ]; then
        echo "THROW: $f"
      fi
    fi

    if grep -q '"type":"TryStatement"' "$cache" 2>/dev/null; then
      try_count=$((try_count + 1))
      if [ $try_count -le 5 ]; then
        echo "TRY: $f"
      fi
    fi

    if grep -q '"type":"SwitchStatement"' "$cache" 2>/dev/null; then
      switch_count=$((switch_count + 1))
      if [ $switch_count -le 5 ]; then
        echo "SWITCH: $f"
      fi
    fi

    if grep -q '"type":"SpreadElement"' "$cache" 2>/dev/null; then
      spread_array_count=$((spread_array_count + 1))
      if [ $spread_array_count -le 5 ]; then
        echo "SPREAD: $f"
      fi
    fi
  fi
done

echo ""
echo "Statement type counts in cached files:"
echo "  ThrowStatement: $throw_count"
echo "  TryStatement: $try_count"
echo "  SwitchStatement: $switch_count"
echo "  SpreadElement (in arrays): $spread_array_count"
echo "  Total cached files: $total_cached"
