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

BUNDLE="Jim.app"
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

# Install the app icon from the tracked source (assets/icon/icon.icns).
# Info.plist's CFBundleIconFile is "icon", so it lands as Resources/icon.icns.
# Keeping the source in the repo means the icon survives fresh clones and
# `cargo clean` (the .app bundle itself is gitignored).
if [ -f assets/icon/icon.icns ]; then
    cp assets/icon/icon.icns "$RES/icon.icns"
fi

# The binary links @rpath/libghostty-vt.dylib but ships without any
# LC_RPATH entries (cargo doesn't add one for build-script libs), so
# dyld can't find the dylib at launch. Add an rpath pointing at the
# bundled Frameworks dir. Tolerate "already present" if we re-run.
install_name_tool -add_rpath "@executable_path/../Frameworks" "$MACOS/$EXEC_NAME" 2>/dev/null || true

# Pick a signing identity. A STABLE (self-signed) identity keeps the
# bundle's code identity constant across rebuilds, so macOS TCC grants
# (Microphone, Documents/Desktop/Downloads, Full Disk Access, …) persist
# instead of resetting every build — that reset is why the OS re-prompts
# for file access "all the time" under ad-hoc signing. Falls back to
# ad-hoc if the identity isn't set up; run ./setup-signing.sh once to
# create it. Override the name via TB_SIGN_IDENTITY.
SIGN_ID="${TB_SIGN_IDENTITY:-TerminalBevy Local Signing}"
# NB: no -v — a self-signed identity is untrusted by Gatekeeper, which
# -v filters out, but codesign still signs with it and the resulting
# designated requirement (cert-leaf anchored) is stable across rebuilds.
if security find-identity -p codesigning 2>/dev/null | grep -qF "$SIGN_ID"; then
    SIGN_ARG="$SIGN_ID"
    echo "[make-bundle] signing with stable identity: $SIGN_ID"
else
    SIGN_ARG="-"
    echo "[make-bundle] WARNING: '$SIGN_ID' not found — signing AD-HOC." >&2
    echo "[make-bundle]          TCC grants (mic, Documents, Full Disk) will reset on each rebuild." >&2
    echo "[make-bundle]          Run ./setup-signing.sh once to fix." >&2
fi

# install_name_tool invalidated the existing signature. Re-sign nested
# code (the dylib) first; the whole bundle is signed at the end, after
# Info.plist is in place, so the seal covers the usage-description strings.
codesign --force --sign "$SIGN_ARG" "$FRAMEWORKS/libghostty-vt.dylib" >/dev/null 2>&1 || true

# Info.plist — CFBundleIdentifier drives Dock identity. The NS*UsageDescription
# keys are REQUIRED for the corresponding TCC access: without
# NSMicrophoneUsageDescription, any mic request by a process we're
# responsible for (e.g. Claude Code voice dictation in a child shell) is
# denied outright. The folder keys supply the prompt text for file access.
# NSHighResolutionCapable avoids fuzzy 72dpi upscaling on retina.
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
    <key>CFBundleName</key>                   <string>Jim</string>
    <key>CFBundleDisplayName</key>            <string>Jim</string>
    <key>CFBundlePackageType</key>            <string>APPL</string>
    <key>CFBundleShortVersionString</key>     <string>0.1.0</string>
    <key>CFBundleVersion</key>                <string>1</string>
    <key>LSMinimumSystemVersion</key>         <string>11.0</string>
    <key>NSHighResolutionCapable</key>        <true/>
    <key>NSPrincipalClass</key>               <string>NSApplication</string>
    <key>NSSupportsAutomaticGraphicsSwitching</key> <true/>
    <key>NSMicrophoneUsageDescription</key>   <string>Programs you run in Jim (such as Claude Code voice dictation) use the microphone.</string>
    <key>NSDocumentsFolderUsageDescription</key>  <string>Jim and the programs you run in it work with files in your Documents folder.</string>
    <key>NSDesktopFolderUsageDescription</key>    <string>Jim and the programs you run in it work with files on your Desktop.</string>
    <key>NSDownloadsFolderUsageDescription</key>  <string>Jim and the programs you run in it work with files in your Downloads folder.</string>
    <key>NSRemovableVolumesUsageDescription</key> <string>Jim and the programs you run in it work with files on removable volumes.</string>
    <key>NSNetworkVolumesUsageDescription</key>   <string>Jim and the programs you run in it work with files on network volumes.</string>
</dict>
</plist>
EOF

# Sign the whole bundle LAST — after the binary, dylib, rpath and
# Info.plist are all in place — so the signature seals the bundle
# (including the usage-description strings TCC reads). This also signs
# the main executable. Nested dylib is already signed above.
codesign --force --sign "$SIGN_ARG" "$BUNDLE" 2>/dev/null \
    || echo "[make-bundle] WARNING: bundle codesign failed (app may not launch / TCC unstable)" >&2

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
