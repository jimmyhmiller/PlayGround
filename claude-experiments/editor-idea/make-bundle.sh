#!/bin/sh
# Build a self-contained TerminalBevy.app — copies the binary and the
# libghostty-vt dylib INTO the bundle and bakes an rpath so the .app
# launches without DYLD env vars and survives `cargo clean`.
#
# Identity (CFBundleIdentifier) is stable across rebuilds, so the Dock
# tile / pin / icon stay attached.
#
# Usage:
#   ./make-bundle.sh                 # bundle from target/debug
#   ./make-bundle.sh --release       # bundle from target/release
#
# Drop a 1024x1024 icon at TerminalBevy.app/Contents/Resources/icon.icns
# (or generate one from a PNG via `iconutil`) and macOS will pick it up
# next launch.

set -e
cd "$(dirname "$0")"

PROFILE=debug
while [ $# -gt 0 ]; do
    case "$1" in
        --release) PROFILE=release; shift ;;
        --debug)   PROFILE=debug;   shift ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

BUNDLE="TerminalBevy.app"
CONTENTS="$BUNDLE/Contents"
MACOS="$CONTENTS/MacOS"
FRAMEWORKS="$CONTENTS/Frameworks"
RES="$CONTENTS/Resources"
BUNDLE_ID="com.jimmyhmiller.terminal-bevy"
EXEC_NAME="terminal"
SRC_BIN="target/$PROFILE/$EXEC_NAME"

if [ ! -x "$SRC_BIN" ]; then
    echo "[make-bundle] $SRC_BIN not found — run cargo build first" >&2
    exit 1
fi

# libghostty-vt-sys writes the dylib under a build-script-hashed dir.
# The hash can change between builds; resolve it freshly each time.
SRC_DYLIB=$(find "target/$PROFILE/build" -type f -name 'libghostty-vt.*.*.dylib' 2>/dev/null | head -1)
if [ -z "$SRC_DYLIB" ]; then
    echo "[make-bundle] couldn't find libghostty-vt dylib under target/$PROFILE/build" >&2
    exit 1
fi

mkdir -p "$MACOS" "$FRAMEWORKS" "$RES"

# Copy (don't symlink) the binary and dylib. A symlink into target/
# would dangle after `cargo clean`; copies make the .app portable and
# self-contained — you can launch it from Finder, the Dock, or move it
# to /Applications without any env vars.
cp "$SRC_BIN" "$MACOS/$EXEC_NAME"
cp "$SRC_DYLIB" "$FRAMEWORKS/libghostty-vt.dylib"

# The binary links @rpath/libghostty-vt.dylib but ships without any
# LC_RPATH entries (cargo doesn't add one for build-script libs), so
# dyld can't find the dylib at launch. Add an rpath pointing at the
# bundled Frameworks dir. Tolerate "already present" if we re-run.
install_name_tool -add_rpath "@executable_path/../Frameworks" "$MACOS/$EXEC_NAME" 2>/dev/null || true

# Re-sign with an ad-hoc signature. install_name_tool invalidates the
# existing signature; without re-signing, macOS will refuse to launch
# the binary on recent OS versions.
codesign --force --sign - "$FRAMEWORKS/libghostty-vt.dylib" >/dev/null 2>&1 || true
codesign --force --sign - "$MACOS/$EXEC_NAME" >/dev/null 2>&1 || true

# Info.plist — the only thing that matters for Dock identity is
# CFBundleIdentifier. Everything else is cosmetic / required-by-format.
# NSHighResolutionCapable is important: without it AppKit upscales
# everything from 72dpi and the window looks fuzzy on retina.
cat > "$CONTENTS/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>      <string>en</string>
    <key>CFBundleExecutable</key>             <string>$EXEC_NAME</string>
    <key>CFBundleIconFile</key>               <string>icon</string>
    <key>CFBundleIdentifier</key>             <string>$BUNDLE_ID</string>
    <key>CFBundleInfoDictionaryVersion</key>  <string>6.0</string>
    <key>CFBundleName</key>                   <string>TerminalBevy</string>
    <key>CFBundleDisplayName</key>            <string>TerminalBevy</string>
    <key>CFBundlePackageType</key>            <string>APPL</string>
    <key>CFBundleShortVersionString</key>     <string>0.1.0</string>
    <key>CFBundleVersion</key>                <string>1</string>
    <key>LSMinimumSystemVersion</key>         <string>11.0</string>
    <key>NSHighResolutionCapable</key>        <true/>
    <key>NSPrincipalClass</key>               <string>NSApplication</string>
    <key>NSSupportsAutomaticGraphicsSwitching</key> <true/>
</dict>
</plist>
EOF

# Nudge LaunchServices so the new/updated bundle is registered
# immediately — otherwise the first `open` after creation can be slow
# or show the generic icon until Spotlight catches up.
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister \
    -f "$BUNDLE" >/dev/null 2>&1 || true

echo "[make-bundle] $BUNDLE built from target/$PROFILE (self-contained)"
echo "[make-bundle] bundle id: $BUNDLE_ID"
if [ ! -f "$RES/icon.icns" ]; then
    echo "[make-bundle] (no $RES/icon.icns yet — using generic app icon)"
fi
