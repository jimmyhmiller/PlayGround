#!/bin/bash
#
# build-direct.sh — Build the Direct / Lemon Squeezy distribution.
#
# Produces a Developer ID-signed, notarized, stapled .app and .dmg with
# Sparkle for auto-updates. This is the build for the website download.
#
# Required environment variables:
#   DEVELOPER_ID_APPLICATION  (e.g. "Developer ID Application: Jimmy Miller (7J8U597P7P)")
#   APPLE_ID                  (apple id email for notarytool)
#   APPLE_TEAM_ID             (e.g. "7J8U597P7P")
#   APPLE_APP_PASSWORD        (app-specific password — see appleid.apple.com)
#
# Optional:
#   ICON_STYLE                "new" (default) or "old"
#   SKIP_NOTARIZE             "1" to skip notarization (for local testing)
#
set -euo pipefail

ICON_STYLE="${ICON_STYLE:-new}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ "$ICON_STYLE" != "new" && "$ICON_STYLE" != "old" ]]; then
    echo "Usage: ICON_STYLE=[new|old] ./build-direct.sh"
    exit 1
fi

: "${DEVELOPER_ID_APPLICATION:?Set DEVELOPER_ID_APPLICATION (e.g. 'Developer ID Application: Your Name (TEAMID)')}"

APP="Ease.app"
DMG="Ease.dmg"

echo "→ Cleaning previous bundle"
rm -rf "$APP" "$DMG"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources" "$APP/Contents/Frameworks"

echo "→ Building release binary"
swift build -c release

echo "→ Copying executable"
cp ".build/release/Ease" "$APP/Contents/MacOS/Ease"

echo "→ Writing Info.plist"
cp Info.Direct.plist "$APP/Contents/Info.plist"

# Select icon source based on style
if [[ "$ICON_STYLE" == "new" ]]; then
    ICON_SOURCE="Ease/AppIcon.icon"
    echo "→ Using new icon: Short / Long / Medium"
else
    ICON_SOURCE="Ease/AppIcon-original.icon"
    echo "→ Using old icon: Long / Short / Medium"
fi

# Bar widths to match the chosen icon
if [[ "$ICON_STYLE" == "new" ]]; then
    cat > "$APP/Contents/Resources/MenuBarConfig.plist" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>barWidths</key>
    <array>
        <real>7</real>
        <real>14</real>
        <real>11</real>
    </array>
</dict>
</plist>
EOF
else
    cat > "$APP/Contents/Resources/MenuBarConfig.plist" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>barWidths</key>
    <array>
        <real>14</real>
        <real>8</real>
        <real>11</real>
    </array>
</dict>
</plist>
EOF
fi

echo "→ Compiling icon set (.icon → .icns + AssetCatalog)"
xcrun actool "$ICON_SOURCE" \
    --compile "$APP/Contents/Resources" \
    --app-icon AppIcon \
    --enable-on-demand-resources NO \
    --development-region en \
    --target-device mac \
    --platform macosx \
    --enable-icon-stack-fallback-generation=disabled \
    --include-all-app-icons \
    --minimum-deployment-target 14.0 \
    --output-partial-info-plist /tmp/icon-partial.plist >/dev/null

# Legacy .icns for Dock fallback on older systems
LIGHT_SOURCE="$SCRIPT_DIR/icons/source/icon-1024.png"
if [ ! -f "$LIGHT_SOURCE" ]; then
    echo "ERROR: source icon not found at $LIGHT_SOURCE"
    exit 1
fi
rm -rf /tmp/legacy.iconset
mkdir -p /tmp/legacy.iconset
for sz in 16 32 128 256 512; do
    sips -z $sz $sz "$LIGHT_SOURCE" --out "/tmp/legacy.iconset/icon_${sz}x${sz}.png" >/dev/null
    sips -z $((sz*2)) $((sz*2)) "$LIGHT_SOURCE" --out "/tmp/legacy.iconset/icon_${sz}x${sz}@2x.png" >/dev/null
done
iconutil -c icns /tmp/legacy.iconset -o "$APP/Contents/Resources/AppIcon.icns"

echo "→ Embedding Sparkle.framework"
install_name_tool -add_rpath @executable_path/../Frameworks "$APP/Contents/MacOS/Ease" 2>/dev/null || true
SPARKLE_PATH="$(find .build -name "Sparkle.framework" -type d | head -1)"
if [ -z "$SPARKLE_PATH" ]; then
    echo "ERROR: Sparkle.framework not found in .build — did 'swift build' succeed?"
    exit 1
fi
cp -R "$SPARKLE_PATH" "$APP/Contents/Frameworks/"

echo "→ Signing nested frameworks/XPC services"
SPARKLE="$APP/Contents/Frameworks/Sparkle.framework"
# Sign innermost first so the outer signatures cover sealed contents
codesign --force --options runtime --timestamp \
    --sign "$DEVELOPER_ID_APPLICATION" \
    "$SPARKLE/Versions/B/XPCServices/Downloader.xpc" \
    "$SPARKLE/Versions/B/XPCServices/Installer.xpc" \
    "$SPARKLE/Versions/B/Updater.app" \
    "$SPARKLE/Versions/B/Autoupdate"
codesign --force --options runtime --timestamp \
    --sign "$DEVELOPER_ID_APPLICATION" \
    "$SPARKLE"

echo "→ Signing app (hardened runtime + timestamp)"
codesign --force --options runtime --timestamp \
    --entitlements Ease.Direct.entitlements \
    --sign "$DEVELOPER_ID_APPLICATION" \
    "$APP"

echo "→ Verifying signature"
codesign --verify --deep --strict --verbose=2 "$APP"

if [[ "${SKIP_NOTARIZE:-0}" != "1" ]]; then
    : "${APPLE_ID:?Set APPLE_ID to your Apple ID email}"
    : "${APPLE_TEAM_ID:?Set APPLE_TEAM_ID to your team id}"
    : "${APPLE_APP_PASSWORD:?Set APPLE_APP_PASSWORD to your app-specific password}"

    echo "→ Zipping .app for notarization"
    ditto -c -k --keepParent "$APP" Ease.zip

    echo "→ Submitting to notarytool (this can take several minutes)"
    xcrun notarytool submit Ease.zip \
        --apple-id "$APPLE_ID" \
        --team-id "$APPLE_TEAM_ID" \
        --password "$APPLE_APP_PASSWORD" \
        --wait

    echo "→ Stapling notarization ticket to app"
    xcrun stapler staple "$APP"
    rm -f Ease.zip
else
    echo "→ Skipping notarization (SKIP_NOTARIZE=1)"
fi

echo "→ Building DMG"
if ! command -v create-dmg &> /dev/null; then
    echo "ERROR: create-dmg not found. Install with: brew install create-dmg"
    exit 1
fi
DMG_STAGING="$(mktemp -d)"
cp -R "$APP" "$DMG_STAGING/"
ln -s /Applications "$DMG_STAGING/Applications"
create-dmg \
    --volname "Ease" \
    --window-pos 200 120 \
    --window-size 600 400 \
    --icon-size 100 \
    --icon "Ease.app" 150 185 \
    --hide-extension "Ease.app" \
    --icon "Applications" 450 185 \
    "$DMG" \
    "$DMG_STAGING"
rm -rf "$DMG_STAGING"

if [[ "${SKIP_NOTARIZE:-0}" != "1" ]]; then
    echo "→ Stapling DMG"
    xcrun stapler staple "$DMG"
fi

echo
echo "✓ Done — $DMG ready for distribution."
echo "  Next steps:"
echo "    1. Upload $DMG to Lemon Squeezy (Product → Files)"
echo "    2. Generate Sparkle appcast: bin/generate_appcast (from Sparkle release)"
echo "    3. Tag and publish a GitHub release containing $DMG + appcast.xml"
