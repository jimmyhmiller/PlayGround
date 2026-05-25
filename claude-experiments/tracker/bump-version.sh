#!/bin/bash
#
# bump-version.sh — Update CFBundleShortVersionString and CFBundleVersion
# atomically across both Info.Direct.plist and Info.MAS.plist.
#
# Usage:
#   ./bump-version.sh 1.0.1            # bump short version, auto-increment build
#   ./bump-version.sh 1.0.1 7          # set both explicitly
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <short-version> [build-number]"
    echo "  $0 1.0.1            # auto-increment build"
    echo "  $0 1.0.1 7          # explicit build number"
    exit 1
fi

SHORT_VERSION="$1"
DIRECT="Info.Direct.plist"
MAS="Info.MAS.plist"

for plist in "$DIRECT" "$MAS"; do
    if [ ! -f "$plist" ]; then
        echo "ERROR: $plist not found"
        exit 1
    fi
done

current_direct_build="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleVersion' "$DIRECT")"
current_mas_build="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleVersion' "$MAS")"

if [ "$current_direct_build" != "$current_mas_build" ]; then
    echo "WARNING: Direct ($current_direct_build) and MAS ($current_mas_build) build numbers differ."
    echo "         Using higher value as the starting point."
fi

if [ $# -ge 2 ]; then
    NEW_BUILD="$2"
else
    higher=$(( current_direct_build > current_mas_build ? current_direct_build : current_mas_build ))
    NEW_BUILD=$((higher + 1))
fi

echo "→ Direct: $current_direct_build → $NEW_BUILD"
echo "→ MAS:    $current_mas_build → $NEW_BUILD"
echo "→ Short version: $SHORT_VERSION"

for plist in "$DIRECT" "$MAS"; do
    /usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $SHORT_VERSION" "$plist"
    /usr/libexec/PlistBuddy -c "Set :CFBundleVersion $NEW_BUILD" "$plist"
done

echo
echo "✓ Bumped both plists to $SHORT_VERSION (build $NEW_BUILD)"
echo "  Don't forget to update release notes."
