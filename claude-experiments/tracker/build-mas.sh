#!/bin/bash
#
# build-mas.sh — Build the Mac App Store distribution.
#
# Produces an "Apple Distribution" or "3rd Party Mac Developer Application"
# signed .app, packaged as a .pkg ready to upload to App Store Connect via
# Transporter. Sparkle is excluded entirely (App Store-prohibited).
#
# Required environment variables:
#   MAS_APP_IDENTITY      Signing identity for the .app
#                         (e.g. "Apple Distribution: Jimmy Miller (7J8U597P7P)"
#                          or  "3rd Party Mac Developer Application: Jimmy Miller (7J8U597P7P)")
#   MAS_INSTALLER_IDENTITY Signing identity for the .pkg installer
#                         (e.g. "3rd Party Mac Developer Installer: Jimmy Miller (7J8U597P7P)")
#   MAS_PROVISION_PROFILE Path to the Mac App Store provisioning profile
#                         (downloaded from developer.apple.com)
#
# Optional:
#   ICON_STYLE            "new" (default) or "old"
#
set -euo pipefail

ICON_STYLE="${ICON_STYLE:-new}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

: "${MAS_APP_IDENTITY:?Set MAS_APP_IDENTITY (e.g. '3rd Party Mac Developer Application: Your Name (TEAMID)')}"
: "${MAS_INSTALLER_IDENTITY:?Set MAS_INSTALLER_IDENTITY (e.g. '3rd Party Mac Developer Installer: Your Name (TEAMID)')}"
: "${MAS_PROVISION_PROFILE:?Set MAS_PROVISION_PROFILE to the path of your MAS provisioning profile}"

if [ ! -f "$MAS_PROVISION_PROFILE" ]; then
    echo "ERROR: provisioning profile not found at $MAS_PROVISION_PROFILE"
    exit 1
fi

APP="Ease-MAS.app"
PKG="Ease.pkg"

echo "→ Cleaning previous bundle"
rm -rf "$APP" "$PKG"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

echo "→ Building release binary (EASE_MAS=1, no Sparkle)"
# Clean .build to ensure no Sparkle artifacts linger from a prior Direct build
rm -rf .build
EASE_MAS=1 swift build -c release

echo "→ Copying executable"
cp ".build/release/Ease" "$APP/Contents/MacOS/Ease"
# Ensure the executable's MacOS name matches CFBundleExecutable
chmod +x "$APP/Contents/MacOS/Ease"

echo "→ Writing Info.plist"
cp Info.MAS.plist "$APP/Contents/Info.plist"

if [[ "$ICON_STYLE" == "new" ]]; then
    ICON_SOURCE="Ease/AppIcon.icon"
else
    ICON_SOURCE="Ease/AppIcon-original.icon"
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

echo "→ Compiling icon set"
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
    --output-partial-info-plist /tmp/icon-partial-mas.plist >/dev/null

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

echo "→ Embedding provisioning profile"
cp "$MAS_PROVISION_PROFILE" "$APP/Contents/embedded.provisionprofile"

echo "→ Signing app (hardened runtime + sandbox entitlements)"
codesign --force --options runtime --timestamp \
    --entitlements Ease.MAS.entitlements \
    --sign "$MAS_APP_IDENTITY" \
    "$APP"

echo "→ Verifying signature"
codesign --verify --deep --strict --verbose=2 "$APP"

echo "→ Building installer package"
productbuild \
    --component "$APP" /Applications \
    --sign "$MAS_INSTALLER_IDENTITY" \
    "$PKG"

echo
echo "✓ Done — $PKG ready for upload."
echo "  Next steps:"
echo "    1. Open Transporter.app (free, App Store)"
echo "    2. Drag $PKG into Transporter and click Deliver"
echo "    3. Wait for processing in App Store Connect (~10–30 min)"
echo "    4. Submit the build for review from App Store Connect"
